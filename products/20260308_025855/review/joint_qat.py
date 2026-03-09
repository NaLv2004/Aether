import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os

from gnn_detector import generate_data_batch, get_tensors_from_batch, compute_ber
from system_model import CellFreeSystem
from policy_network import AdaptiveQuantizer

class QAT_FPGNN(nn.Module):
    def __init__(self, L, K, hidden_dim=64):
        super(QAT_FPGNN, self).__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        
        # 1. 特征提升层: (..., 4) -> (..., hidden_dim) (接收实部、虚部、局部 SNR 和 expected_bits)
        self.feature_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. AP-to-CPU 注意力
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. 多用户干扰消除 (MU-IC)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. 最终残差映射
        self.final_linear = nn.Linear(self.hidden_dim, 2)
        
        # 残差关键零初始化
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 提取量化后的原始软信息
        soft_info = x[..., :2]  # shape (B, L, K, 2)
        
        h = self.feature_mlp(x) # (B, L, K, hidden_dim)
        attn_weights = self.attn_net(h) # (B, L, K, 1)
        attn_weights = torch.softmax(attn_weights, dim=1) 
        
        base_out = torch.sum(soft_info * attn_weights, dim=1)  # shape (B, K, 2)
        
        fused_h = torch.sum(h * attn_weights, dim=1)  # shape (B, K, hidden_dim)
        
        ic_out = self.transformer_ic(fused_h) # shape (B, K, hidden_dim)
        residual_out = self.final_linear(ic_out) # shape (B, K, 2)
        
        return base_out + residual_out

    def load_pretrained(self, model_path):
        """
        继承预训练全精度网络参数，并对输入维度变更进行无损处理
        """
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 处理 feature_mlp.0.weight 维度的改变以继承预训练参数
        if 'feature_mlp.0.weight' in state_dict:
            old_weight = state_dict['feature_mlp.0.weight'] # (hidden_dim, 3)
            new_weight = self.feature_mlp[0].weight.data.clone() # (hidden_dim, 4)
            new_weight[:, :3] = old_weight
            new_weight[:, 3] = 0.0 # 初始化新增的 expected_bits 维度为 0，保持网络预测能力在初期不变
            state_dict['feature_mlp.0.weight'] = new_weight
            
        # 使用 strict=False 忽略由于轻微维度变更引发的问题
        self.load_state_dict(state_dict, strict=False)


class JointQATModel(nn.Module):
    def __init__(self, L, K, hidden_dim=64):
        super(JointQATModel, self).__init__()
        self.quantizer = AdaptiveQuantizer()
        self.gnn = QAT_FPGNN(L, K, hidden_dim)

    def forward(self, x, tau=1.0):
        # x: (B, L, K, 3)
        v = x[..., :2]    # 实部和虚部
        snr = x[..., 2:3] # SNR信息
        
        # 通过自适应量化器得到量化值和期望的比特数
        v_q, expected_bits, w = self.quantizer(v, snr, tau)
        
        # expected_bits 形状为 (B, L, K)，需增加一维以便拼接
        expected_bits = expected_bits.unsqueeze(-1) # (B, L, K, 1)
        
        # 拼接量化值、SNR和预计比特数，共同输入后续的GNN进行特征提升和干涉消除
        x_q = torch.cat([v_q, snr, expected_bits], dim=-1) # (B, L, K, 4)
        out = self.gnn(x_q)
        
        return out, expected_bits


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Quantization-Aware Training for Cell-Free MIMO")
    # 环境超参
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for feature lifting")
    parser.add_argument("--pretrained_model", type=str, default="fp_gnn_model.pth", help="Path to pretrained full-precision GNN")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, Batches={args.batches_per_epoch}, BatchSize={args.batch_size}, LR={args.lr}")
    
    # 1. 实例化环境并锁定大尺度场景
    np.random.seed(42)
    torch.manual_seed(42)
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()
    
    # 2. 预先生成验证集/测试集 (严格分离，固定样本点保证训练中各 epoch 公平对比)
    test_p_tx = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    test_samples = 300
    test_dataset = {}
    print(f"Generating isolated testing set for {len(test_p_tx)} power points...")
    for p in test_p_tx:
        s_hat_np, s_np, local_snr_np = generate_data_batch(sys_model, test_samples, p_tx_dbm=p)
        X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
        test_dataset[p] = (X_test, Y_test, s_hat_np, s_np, local_snr_np)
    print("Testing set generated successfully.\n")
    
    # 3. 初始化端到端联合量化网络模型及优化器与调度器
    model = JointQATModel(L=args.L, K=args.K, hidden_dim=args.hidden_dim).to(device)
    if os.path.exists(args.pretrained_model):
        model.gnn.load_pretrained(args.pretrained_model)
        print(f"Successfully loaded pretrained GNN weights from '{args.pretrained_model}'")
    else:
        print(f"Warning: Pretrained model '{args.pretrained_model}' not found. The GNN part will be trained from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 引入余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    mse_loss_fn = nn.MSELoss()
    
    # 4. 开始在线训练循环
    print("=== Starting Joint QAT Training ===")
    for epoch in range(args.epochs):
        model.train()
        # 退火策略: 更新 Gumbel-Softmax 温度 tau (衰减更平滑，帮助网络找寻更稳固的分配策略)
        tau = max(0.1, 1.0 * (0.95 ** epoch))
        
        epoch_loss = 0.0
        epoch_bits = 0.0
        start_t = time.time()
        
        for _ in range(args.batches_per_epoch):
            # 训练集: 功率在 -20 到 25 dBm 之间随机采样
            p_train = np.random.uniform(-20, 25)
            s_hat_train, s_train, local_snr_train = generate_data_batch(sys_model, args.batch_size, p_train)
            X_train, Y_train = get_tensors_from_batch(s_hat_train, s_train, local_snr_train, device)
            
            optimizer.zero_grad()
            out, expected_bits = model(X_train, tau=tau)
            
            # Loss计算: 最终复预测的 MSE误差 + 确保总体系统平均吞吐在 2.5 Bits左右的约束惩罚
            mse_loss = mse_loss_fn(out, Y_train)
            bit_loss = 0.1 * ((expected_bits.mean() - 2.5) ** 2)
            loss = mse_loss + bit_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bits += expected_bits.mean().item()
            
        # 每一个 Epoch 结束后使用 0 dBm 的固定验证集查探一次收敛性
        model.eval()
        with torch.no_grad():
            X_val, Y_val, _, _, _ = test_dataset[0]
            # 验证集采用接近硬判决的 tau=0.1 评估客观误差
            val_out, _ = model(X_val, tau=0.1) 
            val_loss = mse_loss_fn(val_out, Y_val).item()
            
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_t
        avg_train_loss = epoch_loss / args.batches_per_epoch
        avg_train_bits = epoch_bits / args.batches_per_epoch
        
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] - Time: {elapsed:.2f}s | "
              f"LR: {current_lr:.2e} | Tau: {tau:.3f} | "
              f"Train Loss: {avg_train_loss:.5f} | Avg Bits: {avg_train_bits:.3f} | Val Loss (0dBm): {val_loss:.5f}")
              
        scheduler.step()
        
    print("Joint QAT Training completed!\n")
    
    # 5. 测试与对比 (遍历预先生成并固定的各功率测试点数据)
    print("=== Testing & Performance Comparison ===")
    print(f"{'p_tx (dBm)':<12} | {'Dist-Full BER':<15} | {'QAT-GNN BER':<15} | {'Improvement (%)':<15} | {'Average Bits':<15}")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        for p in test_p_tx:
            X_test, Y_test, s_hat_np, s_np, local_snr_np = test_dataset[p]
            
            # Baseline: 均值合并的 Dist-Full BER
            s_hat_avg = s_hat_np.mean(axis=1) 
            ber_dist_full = compute_ber(s_np, s_hat_avg)
            
            # QAT-GNN 输出及其预测 BER (使用硬判决 tau=0.1 彻底切断软连接)
            out, expected_bits = model(X_test, tau=0.1)
            out_np = out.cpu().numpy()
            s_pred_qat = out_np[..., 0] + 1j * out_np[..., 1]
            ber_qat = compute_ber(s_np, s_pred_qat)
            
            # 增益与统计指标
            improve = ((ber_dist_full - ber_qat) / ber_dist_full * 100) if ber_dist_full > 0 else 0.0
            avg_bits_p = expected_bits.mean().item()
            
            print(f"{p:<12} | {ber_dist_full:<15.4%} | {ber_qat:<15.4%} | {improve:>5.2f} % | {avg_bits_p:<15.2f}")
            
    print("-" * 80)
    
    # 保存模型权重
    torch.save(model.state_dict(), "qat_gnn_model.pth")
    print("\nJoint QAT Model weights have been saved to 'qat_gnn_model.pth'.")

if __name__ == "__main__":
    main()
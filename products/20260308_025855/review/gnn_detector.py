import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os

# 导入基线环境 (需确保系统中有 system_model.py)
from system_model import CellFreeSystem

def parse_args():
    parser = argparse.ArgumentParser(description="Full-Precision GNN Pre-training for Cell-Free MIMO")
    # 环境超参
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # 训练超参
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for feature lifting")
    return parser.parse_args()

def generate_data_batch(sys_model, batch_size, p_tx_dbm):
    """
    向量化生成大量的小尺度信道与发送/接收数据。
    同时计算局部的信噪比作为特征辅助判断权重。
    """
    # 临时计算新功率点的线性功率及对应 beta_w
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # shape: (L, K)
    
    # 1. 批量生成小尺度瑞利衰落信道 H_small: CN(0,1)
    # shape: (batch_size, L, N, K)
    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) + 
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    
    # 将 beta_w 扩展维度到 (1, L, 1, K) 以支持广播乘法
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]
    H = np.sqrt(beta_w_expanded) * h_small
    
    # 2. 批量生成发射符号 s (QPSK)
    bits = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(bits) / np.sqrt(2) # shape: (batch_size, K)
    
    # 3. 计算无噪接收信号并加入 AWGN
    # 使用 einsum 向量化运算: sum_k H[b,l,n,k] * s[b,k]
    y_clean = np.einsum('blnk,bk->bln', H, s)
    z = (np.random.randn(batch_size, sys_model.L, sys_model.N) + 
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z
    
    # 4. CPU端汇集前的局部 AP 端 LMMSE 均衡
    # 计算 R_y = H_l @ H_l^H + sigma^2 * I
    H_conj_trans = H.conj().transpose(0, 1, 3, 2) # (batch_size, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H @ H_conj_trans + noise_cov # shape: (batch_size, L, N, N)
    
    # 逆矩阵计算与 W_l 权重矩阵计算
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_conj_trans @ R_y_inv # shape: (batch_size, L, K, N)
    
    # 获得局部估计 s_hat = W_l @ y
    # y shape 被拓展为 (batch_size, L, N, 1) 方便矩阵乘法
    s_hat = W_l @ y[..., np.newaxis] 
    s_hat = s_hat.squeeze(-1) # 最终 shape: (batch_size, L, K)
    
    # 5. 计算局部信噪比 (dB)
    # H 的 norm 平方求和，在 N (天线数) 维度上
    local_snr = 10 * np.log10(np.sum(np.abs(H)**2, axis=2) / sys_model.noise_w + 1e-12) / 10.0
    
    return s_hat, s, local_snr

def get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device):
    """
    将复数 numpy 数组拆分为实部虚部组合的 Torch Tensors，并拼接入局部 SNR 信息。
    """
    # 拼接出三维特征 (实部, 虚部, 局部信噪比)
    X = np.stack([s_hat_np.real, s_hat_np.imag, local_snr_np], axis=-1)
    Y = np.stack([s_np.real, s_np.imag], axis=-1)
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device)

class FPGNN(nn.Module):
    def __init__(self, L, K, hidden_dim=64):
        super(FPGNN, self).__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        
        # 1. 特征提升层: (..., 3) -> (..., hidden_dim) (接收实部、虚部和局部 SNR)
        self.feature_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. AP-to-CPU 注意力: 评估不同局部信息的可靠度并计算 L 维度融合权重
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 3. 多用户干扰消除 (MU-IC): 基于 Transformer，支持不同用户特征的充分交换
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1,        # 抗过拟合
            norm_first=True     # 稳定残差块的训练
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. 最终残差映射: 将 Transformer 增强后的特征映射为复信号(实部, 虚部)的 2 维空间
        self.final_linear = nn.Linear(self.hidden_dim, 2)
        
        # 残差关键零初始化：使 Transformer 干扰消除在训练初期输出绝对为 0
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # a) 提取原始输入的前两维软信息: s_hat.real, s_hat.imag
        soft_info = x[..., :2]  # shape (B, L, K, 2)
        
        # b) 特征提升与注意力权重计算
        h = self.feature_mlp(x) # (B, L, K, 64)
        attn_weights = self.attn_net(h) # (B, L, K, 1)
        # 在 L (AP维度)进行 softmax 归一化，代表 AP 的可靠性分配
        attn_weights = torch.softmax(attn_weights, dim=1) 
        
        # c) 基准加权输出: 利用学到的注意力直接融合软信息 (物理先验)
        base_out = torch.sum(soft_info * attn_weights, dim=1)  # shape (B, K, 2)
        
        # d) 残差干扰消除: 隐特征融合并交由 Transformer 建模用户间的互相干扰关系
        fused_h = torch.sum(h * attn_weights, dim=1)  # shape (B, K, 64)
        
        ic_out = self.transformer_ic(fused_h) # shape (B, K, 64)
        residual_out = self.final_linear(ic_out) # shape (B, K, 2)
        
        # e) 最终预测 = 基准的自适应分集合并 + 多用户干扰抵消残差
        return base_out + residual_out

def compute_ber(s_true, s_pred_complex):
    """
    由于发送的是 QPSK (-/+ 1 -/+ 1j)/sqrt(2)，只需对实部虚部取符号判断即可。
    """
    s_true_real = np.sign(s_true.real)
    s_true_imag = np.sign(s_true.imag)
    
    s_pred_real = np.sign(s_pred_complex.real)
    s_pred_imag = np.sign(s_pred_complex.imag)
    
    # 防止预测正好在 0 点
    s_pred_real[s_pred_real == 0] = 1
    s_pred_imag[s_pred_imag == 0] = 1
    
    err_real = (s_true_real != s_pred_real).sum()
    err_imag = (s_true_imag != s_pred_imag).sum()
    
    total_bits = s_true.size * 2
    return (err_real + err_imag) / total_bits

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    print(f"Hyperparameters: L={args.L}, K={args.K}, N={args.N}, Epochs={args.epochs}, "
          f"Batches={args.batches_per_epoch}, BatchSize={args.batch_size}, LR={args.lr}")
    
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
    
    # 3. 初始化 FPGNN 及优化器与调度器
    model = FPGNN(L=args.L, K=args.K, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 引入余弦退火学习率以保证训练后期平稳收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    # 4. 开始在线训练循环
    print("=== Starting FPGNN Pre-training ===")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()
        
        for _ in range(args.batches_per_epoch):
            # 训练集: 扩大功率采样范围，包含更多高信噪比情况
            p_train = np.random.uniform(-20, 25)
            s_hat_train, s_train, local_snr_train = generate_data_batch(sys_model, args.batch_size, p_train)
            X_train, Y_train = get_tensors_from_batch(s_hat_train, s_train, local_snr_train, device)
            
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, Y_train)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # 每一个 Epoch 结束后使用 0 dBm 的固定验证集查探一次收敛性
        model.eval()
        with torch.no_grad():
            X_val, Y_val, _, _, _ = test_dataset[0]
            val_out = model(X_val)
            val_loss = criterion(val_out, Y_val).item()
            
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_t
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] - Time: {elapsed:.2f}s | "
              f"LR: {current_lr:.2e} | Train Loss: {epoch_loss/args.batches_per_epoch:.5f} | Val Loss (0dBm): {val_loss:.5f}")
              
        # 每个 Epoch 完成后更新学习率
        scheduler.step()
    
    print("Pre-training completed!\n")
    
    # 5. 测试与对比 (遍历预先生成并固定的各功率测试点数据)
    print("=== Testing & Performance Comparison ===")
    print(f"{'p_tx (dBm)':<12} | {'Dist-Full BER':<15} | {'FPGNN BER':<15} | {'Improvement (%)':<15}")
    print("-" * 65)
    
    model.eval()
    with torch.no_grad():
        for p in test_p_tx:
            X_test, Y_test, s_hat_np, s_np, local_snr_np = test_dataset[p]
            
            # Baseline: 均值合并的 Dist-Full BER
            s_hat_avg = s_hat_np.mean(axis=1) # 对 L 个 AP 的估计直接取均值
            ber_dist_full = compute_ber(s_np, s_hat_avg)
            
            # FPGNN 输出及其预测 BER
            out = model(X_test)
            out_np = out.cpu().numpy()
            s_pred_fpgnn = out_np[..., 0] + 1j * out_np[..., 1]
            ber_fpgnn = compute_ber(s_np, s_pred_fpgnn)
            
            # 增益计算
            improve = ((ber_dist_full - ber_fpgnn) / ber_dist_full * 100) if ber_dist_full > 0 else 0.0
            
            print(f"{p:<12} | {ber_dist_full:<15.4%} | {ber_fpgnn:<15.4%} | {improve:>5.2f} %")
            
    print("-" * 65)
    
    # 保存模型权重
    torch.save(model.state_dict(), "fp_gnn_model.pth")
    print("\nFPGNN Model weights have been saved to 'fp_gnn_model.pth'.")

if __name__ == "__main__":
    main()
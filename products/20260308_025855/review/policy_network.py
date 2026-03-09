import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from lsq_quantizer import LSQQuantizer

class BitwidthPolicyNet(nn.Module):
    """
    自适应位宽策略网络：基于输入信号与信噪比（SNR）输出选择特定位宽的倾向。
    """
    def __init__(self):
        super(BitwidthPolicyNet, self).__init__()
        # 包含一个简单的 MLP
        self.fc1 = nn.Linear(3, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        # 输入 x 形状为 (..., 3)
        x = self.fc1(x)
        x = self.relu(x)
        # 输出 logits 代表选择 0-bit, 2-bit, 4-bit 的倾向
        logits = self.fc2(x)
        return logits

class AdaptiveQuantizer(nn.Module):
    """
    自适应量化模块：根据策略网络的输出使用 Gumbel-Softmax 分配并计算特征量化。
    """
    def __init__(self):
        super(AdaptiveQuantizer, self).__init__()
        self.policy_net = BitwidthPolicyNet()
        # 针对 2-bit 和 4-bit 分别实例化的量化器
        self.q2 = LSQQuantizer(num_bits=2)
        self.q4 = LSQQuantizer(num_bits=4)

    def forward(self, v, snr, tau=1.0):
        # 1. 拼接信号与信噪比，得到特征维度大小为 3
        # v: (B, K, 2)
        # snr: (B, K, 1)
        x = torch.cat([v, snr], dim=-1) # (B, K, 3)
        
        # 2. 传入策略网络获取各比特选择倾向的 logits
        logits = self.policy_net(x)
        
        # 3. 使用 Gumbel-Softmax 选择位宽
        # hard=True 保证前向输出严格为 0 或 1 (one-hot)，反向传播时直通估计梯度
        w = F.gumbel_softmax(logits, tau=tau, hard=True)
        
        # 4. 计算不同位宽下的量化结果
        v_q2 = self.q2(v)
        v_q4 = self.q4(v)
        
        # 5. 按 one-hot 掩码矩阵 w 合成最终输出
        v_q = w[..., 0:1] * 0.0 + w[..., 1:2] * v_q2 + w[..., 2:3] * v_q4
        
        # 6. 计算预计比特消耗
        expected_bits = w[..., 0] * 0 + w[..., 1] * 2 + w[..., 2] * 4
        
        return v_q, expected_bits, w

def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Quantizer Policy Network")
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--lambda_val', type=float, default=0.5, help='Weight penalty coefficient for bit allocation')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for Gumbel Softmax')
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(42)

    # 构造合成数据
    B = 2000
    K = 8
    
    # 随机生成原始信号和对应信噪比
    v_full = torch.randn(B, K, 2) * 2.0
    # SNR 范围限制在 -10 到 20 dB 之间
    snr_full = torch.rand(B, K, 1) * 30.0 - 10.0
    
    # 分割训练集和测试集 (80% / 20%)
    train_size = int(B * 0.8)
    v_train, v_test = v_full[:train_size], v_full[train_size:]
    snr_train, snr_test = snr_full[:train_size], snr_full[train_size:]

    model = AdaptiveQuantizer()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 基于 SNR 构建 MSE 权重以模拟物理先验：高 SNR 区域误差惩罚大
    snr_weight_train = torch.clamp((snr_train + 10) / 30, 0.1, 1.0)
    snr_weight_test = torch.clamp((snr_test + 10) / 30, 0.1, 1.0)

    print("--- Starting Training of Adaptive Quantizer ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # 1. 前向传播
        v_q, expected_bits, w = model(v_train, snr_train, tau=args.tau)
        
        # 2. 计算误差与比特数损失
        mse_per_element = torch.sum((v_q - v_train)**2, dim=-1)
        weighted_mse = (mse_per_element * snr_weight_train.squeeze(-1)).mean()
        bit_penalty = expected_bits.mean()
        
        # 3. 总体损失函数设计
        loss = weighted_mse + args.lambda_val * bit_penalty
        
        # 4. 反向传播更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 每 50 个 epoch 打印一次记录
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                # 固定的测试集评估
                v_q_t, expected_bits_t, w_t = model(v_test, snr_test, tau=args.tau)
                mse_per_element_t = torch.sum((v_q_t - v_test)**2, dim=-1)
                weighted_mse_t = (mse_per_element_t * snr_weight_test.squeeze(-1)).mean()
                bit_penalty_t = expected_bits_t.mean()
                loss_t = weighted_mse_t + args.lambda_val * bit_penalty_t
                
                # 计算各种位宽分配比例
                w_mean = w_t.mean(dim=(0, 1))
                
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {loss.item():.4f}, Train WMSE: {weighted_mse.item():.4f}, Train Bits: {bit_penalty.item():.4f} | "
                  f"Test Loss: {loss_t.item():.4f}, Test WMSE: {weighted_mse_t.item():.4f}, Test Bits: {bit_penalty_t.item():.4f} | "
                  f"Test W Dist: [0b:{w_mean[0].item():.3f}, 2b:{w_mean[1].item():.3f}, 4b:{w_mean[2].item():.3f}]")

    print("\n--- Final Test Evaluation ---")
    model.eval()
    with torch.no_grad():
        _, expected_bits_t, _ = model(v_test, snr_test, tau=args.tau)
        
        # 将 SNR 分离成三个范围段并分类评估
        low_mask = snr_test < 0.0
        mid_mask = (snr_test >= 0.0) & (snr_test <= 10.0)
        high_mask = snr_test > 10.0
        
        # 获取掩码下的信道分配位数情况
        avg_bits_low = expected_bits_t[low_mask.squeeze(-1)].mean().item() if low_mask.any() else 0.0
        avg_bits_mid = expected_bits_t[mid_mask.squeeze(-1)].mean().item() if mid_mask.any() else 0.0
        avg_bits_high = expected_bits_t[high_mask.squeeze(-1)].mean().item() if high_mask.any() else 0.0
        
        print(f"Low SNR (<0dB) Avg Bits  : {avg_bits_low:.4f}")
        print(f"Mid SNR (0~10dB) Avg Bits: {avg_bits_mid:.4f}")
        print(f"High SNR (>10dB) Avg Bits: {avg_bits_high:.4f}")

if __name__ == "__main__":
    main()
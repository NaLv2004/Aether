import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from lsq_quantizer import LSQQuantizer

class BitwidthPolicyNet(nn.Module):
    """
    自适应位宽策略网络：基于输入信号与信噪比（SNR）输出选择特定位宽的倾向。
    现在支持两个独立的策略输出：一个用于解调结果，一个用于信道系数。
    """
    def __init__(self, input_dim=7):
        super(BitwidthPolicyNet, self).__init__()
        # input_dim: 解调结果实部、虚部、局部SNR、信道范数、信道功率、b_demod、b_channel = 7维
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        
        # 两个独立的策略头：一个用于解调结果，一个用于信道系数
        self.fc2_demod = nn.Linear(32, 3)  # 解调结果的位宽策略 (0-bit, 2-bit, 4-bit)
        self.fc2_channel = nn.Linear(32, 3)  # 信道系数的位宽策略 (0-bit, 2-bit, 4-bit)

    def forward(self, x):
        # 输入 x 形状为 (..., input_dim)
        h = self.fc1(x)
        h = self.relu(h)
        
        # 输出两个独立的logits
        logits_demod = self.fc2_demod(h)
        logits_channel = self.fc2_channel(h)
        
        return logits_demod, logits_channel

class AdaptiveQuantizer(nn.Module):
    """
    自适应量化模块：根据策略网络的输出使用 Gumbel-Softmax 分配并计算特征量化。
    现在同时处理解调结果和信道系数的量化。
    """
    def __init__(self):
        super(AdaptiveQuantizer, self).__init__()
        self.policy_net = BitwidthPolicyNet(input_dim=7)
        
        # 解调结果的量化器
        self.q2_demod = LSQQuantizer(num_bits=2)
        self.q4_demod = LSQQuantizer(num_bits=4)
        
        # 信道系数的量化器
        self.q2_channel = LSQQuantizer(num_bits=2)
        self.q4_channel = LSQQuantizer(num_bits=4)

    def forward(self, v_demod, snr, channel_norm, channel_power, H=None, tau=1.0):
        """
        v_demod: 解调结果 (B, K, 2) - 实部虚部
        snr: 局部信噪比 (B, K, 1)
        channel_norm: 信道范数 (B, K, 1)
        channel_power: 信道功率 (B, K, 1)
        H: 原始信道系数 (B, L, N, K) - 可选，用于信道系数量化
        tau: Gumbel-Softmax温度
        """
        # 1. 拼接特征：解调结果实部虚部、SNR、信道范数、信道功率、b_demod、b_channel
        # 初始化b_demod和b_channel为0（后续会更新）
        b_demod = torch.zeros_like(snr)
        b_channel = torch.zeros_like(snr)
        x = torch.cat([v_demod, snr, channel_norm, channel_power, b_demod, b_channel], dim=-1)  # (B, K, 7)
        
        # 2. 传入策略网络获取两个独立的logits
        logits_demod, logits_channel = self.policy_net(x)
        
        # 3. 使用 Gumbel-Softmax 选择位宽
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B, K, 3)
        w_channel = F.gumbel_softmax(logits_channel, tau=tau, hard=True)  # (B, K, 3)
        
        # 4. 计算解调结果的不同位宽量化结果
        v_q2_demod = self.q2_demod(v_demod)
        v_q4_demod = self.q4_demod(v_demod)
        
        # 5. 按 one-hot 掩码矩阵合成解调结果的最终输出
        v_q_demod = w_demod[..., 0:1] * 0.0 + w_demod[..., 1:2] * v_q2_demod + w_demod[..., 2:3] * v_q4_demod
        
        # 6. 计算解调结果的预计比特消耗
        expected_bits_demod = w_demod[..., 0] * 0 + w_demod[..., 1] * 2 + w_demod[..., 2] * 4
        
        # 7. 信道系数的量化处理（如果提供了H）
        channel_q = None
        expected_bits_channel = torch.zeros_like(expected_bits_demod)
        
        if H is not None:
            # H shape: (B, L, N, K)
            # 将H转换为复数张量并分离实部虚部
            B, L, N, K = H.shape
            
            # 对信道系数进行2-bit和4-bit量化
            # 分别量化实部和虚部
            H_real = H.real
            H_imag = H.imag
            
            # 量化实部
            H_real_q2 = self.q2_channel(H_real)
            H_real_q4 = self.q4_channel(H_real)
            
            # 量化虚部
            H_imag_q2 = self.q2_channel(H_imag)
            H_imag_q4 = self.q4_channel(H_imag)
            
            # 重新组合为复数
            H_q2 = H_real_q2 + 1j * H_imag_q2
            H_q4 = H_real_q4 + 1j * H_imag_q4
            
            # 按w_channel掩码选择最终的量化信道系数
            # w_channel shape: (B, K, 3)，需要扩展到(B, L, N, K)
            w_channel_expanded = w_channel.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, K, 3)
            
            # 合成最终的量化信道系数
            channel_q = (w_channel_expanded[..., 0:1] * 0.0 + 
                        w_channel_expanded[..., 1:2] * H_q2.unsqueeze(-1) + 
                        w_channel_expanded[..., 2:3] * H_q4.unsqueeze(-1)).squeeze(-1)
            
            # 计算信道系数的预计比特消耗（每个用户k）
            # 信道系数包含L个AP，每个AP有N个天线，共L*N*2个实数（实部虚部各一份）
            bits_per_user = L * N * 2  # 实部虚部各占一份
            expected_bits_channel = (w_channel[..., 0] * 0 + 
                                    w_channel[..., 1] * 2 * bits_per_user + 
                                    w_channel[..., 2] * 4 * bits_per_user)
        
        print(f"[AdaptiveQuantizer] Demod bits distribution - 0b: {w_demod[..., 0].mean():.3f}, "
              f"2b: {w_demod[..., 1].mean():.3f}, 4b: {w_demod[..., 2].mean():.3f}")
        print(f"[AdaptiveQuantizer] Channel bits distribution - 0b: {w_channel[..., 0].mean():.3f}, "
              f"2b: {w_channel[..., 1].mean():.3f}, 4b: {w_channel[..., 2].mean():.3f}")
        print(f"[AdaptiveQuantizer] Avg demod bits: {expected_bits_demod.mean():.3f}, "
              f"Avg channel bits: {expected_bits_channel.mean():.3f}")
        
        return v_q_demod, expected_bits_demod, w_demod, channel_q, expected_bits_channel, w_channel

def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Quantizer Policy Network with Dual Strategies")
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
    L = 16
    N = 4
    
    # 随机生成原始信号和对应信噪比
    v_full = torch.randn(B, K, 2) * 2.0
    snr_full = torch.rand(B, K, 1) * 30.0 - 10.0
    
    # 信道统计特征
    channel_norm_full = torch.rand(B, K, 1) * 2.0 + 0.5
    channel_power_full = torch.rand(B, K, 1) * 3.0 + 0.1
    
    # 生成信道系数（复数）
    H_full = (torch.randn(B, L, N, K) + 1j * torch.randn(B, L, N, K)) / torch.sqrt(torch.tensor(2.0))
    
    # 分割训练集和测试集 (80% / 20%)
    train_size = int(B * 0.8)
    v_train, v_test = v_full[:train_size], v_full[train_size:]
    snr_train, snr_test = snr_full[:train_size], snr_full[train_size:]
    channel_norm_train, channel_norm_test = channel_norm_full[:train_size], channel_norm_full[train_size:]
    channel_power_train, channel_power_test = channel_power_full[:train_size], channel_power_full[train_size:]
    H_train, H_test = H_full[:train_size], H_full[train_size:]

    model = AdaptiveQuantizer()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 基于 SNR 构建 MSE 权重
    snr_weight_train = torch.clamp((snr_train + 10) / 30, 0.1, 1.0)
    snr_weight_test = torch.clamp((snr_test + 10) / 30, 0.1, 1.0)

    print("--- Starting Training of Adaptive Quantizer with Dual Strategies ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # 1. 前向传播
        v_q, expected_bits_demod, w_demod, channel_q, expected_bits_channel, w_channel = model(
            v_train, snr_train, channel_norm_train, channel_power_train, H=H_train, tau=args.tau
        )
        
        # 2. 计算误差与比特数损失
        mse_per_element = torch.sum((v_q - v_train)**2, dim=-1)
        weighted_mse = (mse_per_element * snr_weight_train.squeeze(-1)).mean()
        bit_penalty_demod = expected_bits_demod.mean()
        bit_penalty_channel = expected_bits_channel.mean()
        bit_penalty = bit_penalty_demod + bit_penalty_channel
        
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
                v_q_t, expected_bits_demod_t, w_demod_t, channel_q_t, expected_bits_channel_t, w_channel_t = model(
                    v_test, snr_test, channel_norm_test, channel_power_test, H=H_test, tau=args.tau
                )
                mse_per_element_t = torch.sum((v_q_t - v_test)**2, dim=-1)
                weighted_mse_t = (mse_per_element_t * snr_weight_test.squeeze(-1)).mean()
                bit_penalty_demod_t = expected_bits_demod_t.mean()
                bit_penalty_channel_t = expected_bits_channel_t.mean()
                bit_penalty_t = bit_penalty_demod_t + bit_penalty_channel_t
                loss_t = weighted_mse_t + args.lambda_val * bit_penalty_t
                
                # 计算各种位宽分配比例
                w_demod_mean = w_demod_t.mean(dim=(0, 1))
                w_channel_mean = w_channel_t.mean(dim=(0, 1))
                
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {loss.item():.4f}, Train WMSE: {weighted_mse.item():.4f}, "
                  f"Train Bits(Demod/Channel): {bit_penalty_demod.item():.4f}/{bit_penalty_channel.item():.4f} | "
                  f"Test Loss: {loss_t.item():.4f}, Test WMSE: {weighted_mse_t.item():.4f}, "
                  f"Test Bits(Demod/Channel): {bit_penalty_demod_t.item():.4f}/{bit_penalty_channel_t.item():.4f} | "
                  f"Demod W: [0b:{w_demod_mean[0].item():.3f}, 2b:{w_demod_mean[1].item():.3f}, 4b:{w_demod_mean[2].item():.3f}] | "
                  f"Channel W: [0b:{w_channel_mean[0].item():.3f}, 2b:{w_channel_mean[1].item():.3f}, 4b:{w_channel_mean[2].item():.3f}]")

    print("\n--- Final Test Evaluation ---")
    model.eval()
    with torch.no_grad():
        _, expected_bits_demod_t, _, _, expected_bits_channel_t, _ = model(
            v_test, snr_test, channel_norm_test, channel_power_test, H=H_test, tau=args.tau
        )
        
        # 将 SNR 分离成三个范围段并分类评估
        low_mask = snr_test < 0.0
        mid_mask = (snr_test >= 0.0) & (snr_test <= 10.0)
        high_mask = snr_test > 10.0
        
        # 获取掩码下的信道分配位数情况
        avg_bits_demod_low = expected_bits_demod_t[low_mask.squeeze(-1)].mean().item() if low_mask.any() else 0.0
        avg_bits_demod_mid = expected_bits_demod_t[mid_mask.squeeze(-1)].mean().item() if mid_mask.any() else 0.0
        avg_bits_demod_high = expected_bits_demod_t[high_mask.squeeze(-1)].mean().item() if high_mask.any() else 0.0
        
        avg_bits_channel_low = expected_bits_channel_t[low_mask.squeeze(-1)].mean().item() if low_mask.any() else 0.0
        avg_bits_channel_mid = expected_bits_channel_t[mid_mask.squeeze(-1)].mean().item() if mid_mask.any() else 0.0
        avg_bits_channel_high = expected_bits_channel_t[high_mask.squeeze(-1)].mean().item() if high_mask.any() else 0.0
        
        print(f"Low SNR (<0dB) Avg Demod Bits  : {avg_bits_demod_low:.4f}, Avg Channel Bits: {avg_bits_channel_low:.4f}")
        print(f"Mid SNR (0~10dB) Avg Demod Bits: {avg_bits_demod_mid:.4f}, Avg Channel Bits: {avg_bits_channel_mid:.4f}")
        print(f"High SNR (>10dB) Avg Demod Bits: {avg_bits_demod_high:.4f}, Avg Channel Bits: {avg_bits_channel_high:.4f}")

if __name__ == "__main__":
    main()
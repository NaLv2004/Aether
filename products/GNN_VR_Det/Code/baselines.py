import torch
from torch.utils.data import DataLoader
import argparse
from dataset import NearFieldDataset

def evaluate_baselines(args):
    snr_list = args.snr_list
    N = args.N
    K = args.K
    num_samples = args.num_samples
    batch_size = args.batch_size
    
    print(f"==================================================")
    print(f"Evaluating Baselines: Genie-Aided VR-ZF and VR-MMSE")
    print(f"Settings: N={N}, K={K}, num_samples={num_samples}, batch_size={batch_size}")
    print(f"SNR List (dB): {snr_list}")
    print(f"==================================================\n")
    
    for snr in snr_list:
        print(f"Generating dataset for SNR = {snr} dB...")
        # 实例化测试数据集，固定当前 SNR
        dataset = NearFieldDataset(num_samples=num_samples, N=N, K=K, snr_range=(snr, snr))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 总比特数 = 2 (实部与虚部) * K (用户数) * num_samples (样本数)
        total_bits = 2 * K * num_samples
        errors_zf = 0
        errors_mmse = 0
        
        for batch_idx, (H, x, y) in enumerate(dataloader):
            # H: (B, N, K), x: (B, K, 1), y: (B, N, 1)
            B = H.shape[0]
            
            # 计算噪声方差 sigma^2
            signal_power = torch.sum(torch.abs(H)**2, dim=(1,2), keepdim=True) / N
            sigma2 = signal_power / (10 ** (snr / 10))
            
            # 计算 H 的共轭转置
            H_H = H.mH # (B, K, N)
            
            # 计算 H^H H 和 H^H y
            HtH = torch.bmm(H_H, H) # (B, K, K)
            Hty = torch.bmm(H_H, y) # (B, K, 1)
            
            # ================= VR-ZF Detection =================
            # \hat{x}_{ZF} = (H^H H)^{-1} H^H y
            try:
                x_zf = torch.linalg.solve(HtH, Hty)
            except RuntimeError:
                # 异常处理：万一出现奇异矩阵，回退使用伪逆
                x_zf = torch.bmm(torch.linalg.pinv(HtH), Hty)
            
            # ================= VR-MMSE Detection =================
            # \hat{x}_{MMSE} = (H^H H + \sigma^2 I_K)^{-1} H^H y
            I_K = torch.eye(K, dtype=H.dtype, device=H.device).unsqueeze(0) # (1, K, K)
            HtH_mmse = HtH + sigma2 * I_K
            try:
                x_mmse = torch.linalg.solve(HtH_mmse, Hty)
            except RuntimeError:
                x_mmse = torch.bmm(torch.linalg.pinv(HtH_mmse), Hty)
            
            # ================= 解调与计算 BER =================
            # 真实符号的比特位（取符号：正数 -> 1，负数 -> -1）
            true_real_bits = torch.sign(x.real)
            true_imag_bits = torch.sign(x.imag)
            
            # ZF 解调比特位
            zf_real_bits = torch.sign(x_zf.real)
            zf_imag_bits = torch.sign(x_zf.imag)
            
            # MMSE 解调比特位
            mmse_real_bits = torch.sign(x_mmse.real)
            mmse_imag_bits = torch.sign(x_mmse.imag)
            
            # 统计错误比特数
            errors_zf += torch.sum(true_real_bits != zf_real_bits).item()
            errors_zf += torch.sum(true_imag_bits != zf_imag_bits).item()
            
            errors_mmse += torch.sum(true_real_bits != mmse_real_bits).item()
            errors_mmse += torch.sum(true_imag_bits != mmse_imag_bits).item()
        
        # 计算当前 SNR 下的误码率
        ber_zf = errors_zf / total_bits
        ber_mmse = errors_mmse / total_bits
        
        # 打印控制台验收结果
        print(f"[Results] SNR: {snr:5.1f} dB | VR-ZF BER: {ber_zf:.6f} | VR-MMSE BER: {ber_mmse:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VR-ZF and VR-MMSE Baselines")
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of test samples per SNR (default: 2000)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for dataloader (default: 100)')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    args = parser.parse_args()
    
    evaluate_baselines(args)
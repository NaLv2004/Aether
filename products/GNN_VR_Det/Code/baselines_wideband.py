import torch
from torch.utils.data import DataLoader
import argparse
from dataset_wideband import NearFieldWidebandDataset

def evaluate_baselines_wideband(args):
    snr_list = args.snr_list
    N = args.N
    K = args.K
    F = args.F
    num_samples = args.num_samples
    batch_size = args.batch_size
    
    print(f"==================================================")
    print(f"Evaluating Wideband Baselines: Genie-Aided VR-ZF and VR-MMSE")
    print(f"Settings: N={N}, K={K}, F={F}, num_samples={num_samples}, batch_size={batch_size}")
    print(f"SNR List (dB): {snr_list}")
    print(f"==================================================\n")
    
    for snr in snr_list:
        print(f"Generating wideband dataset for SNR = {snr} dB...")
        # 实例化宽带测试数据集，固定当前 SNR
        dataset = NearFieldWidebandDataset(
            num_samples=num_samples, 
            N=N, K=K, F=F, 
            f_c=args.f_c, B_bw=args.B, 
            snr_range=(snr, snr)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 总比特数 = 2 (实部与虚部) * K (用户数) * F (子载波数) * num_samples (样本数)
        total_bits = 2 * K * F * num_samples
        errors_zf = 0
        errors_mmse = 0
        
        for batch_idx, (H, x, y) in enumerate(dataloader):
            # H: (B, F, N, K), x: (B, F, K, 1), y: (B, F, N, 1)
            B_size = H.shape[0]
            
            # 将多载波维度融合到 Batch 维度，方便并行使用纯矩阵运算
            H_reshaped = H.view(B_size * F, N, K)
            x_reshaped = x.view(B_size * F, K, 1)
            y_reshaped = y.view(B_size * F, N, 1)
            
            if batch_idx == 0:
                print(f"  [Debug] Batch 0 original H shape: {H.shape}, x shape: {x.shape}, y shape: {y.shape}")
                print(f"  [Debug] Batch 0 reshaped H shape: {H_reshaped.shape}, x shape: {x_reshaped.shape}, y shape: {y_reshaped.shape}")
            
            # 计算噪声方差 sigma^2
            signal_power = torch.sum(torch.abs(H_reshaped)**2, dim=(1,2), keepdim=True) / N
            sigma2 = signal_power / (10 ** (snr / 10))
            
            # 计算 H 的共轭转置
            H_H = H_reshaped.mH # (B*F, K, N)
            
            # 计算 H^H H 和 H^H y
            HtH = torch.bmm(H_H, H_reshaped) # (B*F, K, K)
            Hty = torch.bmm(H_H, y_reshaped) # (B*F, K, 1)
            
            # ================= VR-ZF Detection =================
            # \hat{x}_{ZF} = (H^H H)^{-1} H^H y
            try:
                x_zf = torch.linalg.solve(HtH, Hty)
            except RuntimeError:
                # 异常处理：万一出现奇异矩阵，回退使用伪逆
                x_zf = torch.bmm(torch.linalg.pinv(HtH), Hty)
            
            # ================= VR-MMSE Detection =================
            # \hat{x}_{MMSE} = (H^H H + \sigma^2 I_K)^{-1} H^H y
            I_K = torch.eye(K, dtype=H_reshaped.dtype, device=H_reshaped.device).unsqueeze(0) # (1, K, K)
            HtH_mmse = HtH + sigma2 * I_K
            try:
                x_mmse = torch.linalg.solve(HtH_mmse, Hty)
            except RuntimeError:
                x_mmse = torch.bmm(torch.linalg.pinv(HtH_mmse), Hty)
            
            # ================= 解调与计算 BER =================
            # 真实符号的比特位（取符号：正数 -> 1，负数 -> -1）
            true_real_bits = torch.sign(x_reshaped.real)
            true_imag_bits = torch.sign(x_reshaped.imag)
            
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
        print(f"[Results] SNR: {snr:5.1f} dB | Wideband VR-ZF BER: {ber_zf:.6f} | Wideband VR-MMSE BER: {ber_mmse:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wideband VR-ZF and VR-MMSE Baselines")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of test samples per SNR (default: 1000)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--F', type=int, default=16, help='Number of subcarriers (default: 16)')
    parser.add_argument('--f_c', type=float, default=0.1e12, help='Carrier frequency in Hz (default: 0.1e12)')
    parser.add_argument('--B', type=float, default=10e9, help='Total Bandwidth in Hz (default: 10e9)')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for dataloader (default: 50)')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    args = parser.parse_args()
    
    evaluate_baselines_wideband(args)
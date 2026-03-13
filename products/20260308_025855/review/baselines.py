import numpy as np
import argparse
from system_model import CellFreeSystem, quantize_signal

def calculate_ber(s_true, s_hat):
    """
    计算基于QPSK解调的误码率（BER）。
    通过比较实际信号和估计信号实部虚部的正负号来统计误码数。
    由于每个符号具有实部和虚部，因此每个符号包含2个bit。
    """
    err_re = np.sign(np.real(s_true)) != np.sign(np.real(s_hat))
    err_im = np.sign(np.imag(s_true)) != np.sign(np.imag(s_hat))
    return np.sum(err_re) + np.sum(err_im)

def main():
    parser = argparse.ArgumentParser(description="6G Cell-Free MIMO Baselines (C-MMSE, Dist-LSFD vs Distributed Quantization)")
    parser.add_argument("--L", type=int, default=16, help="Access Points (APs) 的数量")
    parser.add_argument("--N", type=int, default=4, help="每个 AP 的天线数量")
    parser.add_argument("--K", type=int, default=8, help="单天线用户数量")
    parser.add_argument("--area_size", type=float, default=1.0, help="区域大小 (km)")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="系统带宽 (Hz)")
    parser.add_argument("--noise_psd", type=float, default=-174, help="噪声功率谱密度 (dBm/Hz)")
    parser.add_argument("--epochs", type=int, default=300, help="每个功率点进行蒙特卡洛仿真的 epoch 次数")
    parser.add_argument("--p_tx_list", type=str, default="0,5,10,15,20,25,30", help="要扫描的用户发射功率列表 (dBm), 以逗号分隔")
    parser.add_argument("--heuristic_bits", type=float, default=2.5, help="启发式算法下每个链路平均量化比特预算")
    
    args = parser.parse_args()
    
    p_tx_levels = [float(p) for p in args.p_tx_list.split(',')]
    
    print(f"Starting Baselines Simulation with {args.epochs} epochs per power level.")
    print(f"System Configuration: {args.L} APs, {args.N} antennas/AP, {args.K} users.")
    
    results = []

    # 设置随机数种子保证可复现性
    np.random.seed(42)
    
    # 1. 在循环之前实例化系统并生成大尺度衰落场景
    sys = CellFreeSystem(
        L=args.L, N=args.N, K=args.K, 
        area_size=args.area_size, bandwidth=args.bandwidth, 
        noise_psd_dbm=args.noise_psd, p_tx_dbm=0
    )
    # 生成用户与AP的环境并计算大尺度衰落，固定系统场景
    sys.generate_scenario()
    
    # =========================================================================
    # Heuristic Bit Allocation (M5 Revision Update)
    # 根据各链路的大尺度衰落进行预算约束的非均匀比特分配 (4-bit, 2-bit, 0-bit)
    # =========================================================================
    bit_alloc = np.zeros((sys.L, sys.K), dtype=int)
    for k in range(sys.K):
        # 提取第k个用户到各个AP的大尺度衰落系数 (越大代表信道越好)
        beta_k = sys.beta[:, k] 
        sorted_indices = np.argsort(beta_k)[::-1] # 降序排序
        
        # 针对该用户可使用的总分配比特预算
        budget = int(args.heuristic_bits * sys.L) 
        
        for idx in sorted_indices:
            if budget >= 4:
                bit_alloc[idx, k] = 4
                budget -= 4
            elif budget >= 2:
                bit_alloc[idx, k] = 2
                budget -= 2
            else:
                bit_alloc[idx, k] = 0
                
    avg_bits_allocated = np.mean(bit_alloc)
    print(f"\n[Heuristic Allocation Strategy Setup]")
    print(f"Target Average Bits/Link: {args.heuristic_bits}")
    print(f"Actual Average Bits/Link: {avg_bits_allocated:.2f}")
    print(f"Sample Allocation for User 0 (Bits per AP):\n{bit_alloc[:, 0]}\n")
    # =========================================================================
    
    for p_tx in p_tx_levels:
        print(f"\n[Evaluating Transmit Power: {p_tx} dBm]")
        
        # 2. 手动更新该系统在此轮蒙特卡洛测试下对应的发射功率和大尺度参数功率分量
        sys.p_tx_w = 10 ** ((p_tx - 30) / 10)
        sys.beta_w = sys.p_tx_w * sys.beta
        
        # 计算该功率下的平均 Receive SNR
        snr_linear = sys.beta_w / sys.noise_w
        avg_snr_db = 10 * np.log10(np.mean(snr_linear))
        print(f"Average Receive SNR: {avg_snr_db:.2f} dB")
        
        # 初始化误码统计变量
        err_c_mmse = 0
        err_dist_full = 0
        err_dist_lsfd = 0
        err_dist_q4 = 0
        err_dist_q2 = 0
        err_heuristic = 0
        
        total_bits = args.epochs * args.K * 2  # 总bit数为: epoch次数 * 用户数 * 2 bit/符号
        
        for epoch in range(args.epochs):
            # 在每个 epoch 中，仅生成小尺度衰落信道、信号与接收信号
            H = sys.generate_channel()           # shape: L x N x K
            s = sys.generate_transmit_signals()  # shape: K
            y = sys.simulate_reception(H, s)     # shape: L x N
            
            # --- a) 集中式 C-MMSE ---
            H_all = H.reshape(sys.L * sys.N, sys.K)
            y_all = y.reshape(sys.L * sys.N)
            R_y_all = H_all.conj().T @ H_all + sys.noise_w * np.eye(sys.K)
            s_hat_c_mmse = np.linalg.inv(R_y_all) @ H_all.conj().T @ y_all
            
            err_c_mmse += calculate_ber(s, s_hat_c_mmse)
            
            # --- b) 分布式接收与各策略量化处理 ---
            s_hat_l_list = []
            s_hat_q4_list = []
            s_hat_q2_list = []
            s_hat_heuristic_list = []
            
            for l in range(sys.L):
                # 每个 AP 计算 Local LMMSE 检测结果
                R_y_l = H[l] @ H[l].conj().T + sys.noise_w * np.eye(sys.N)
                W_l = H[l].conj().T @ np.linalg.inv(R_y_l)
                s_hat_l = W_l @ y[l]
                
                s_hat_l_list.append(s_hat_l)
                # 均等量化
                s_hat_q4_list.append(quantize_signal(s_hat_l, [4] * sys.K))
                s_hat_q2_list.append(quantize_signal(s_hat_l, [2] * sys.K))
                # 启发式比特分配量化
                s_hat_heuristic_list.append(quantize_signal(s_hat_l, bit_alloc[l, :]))
                
            s_hat_l_mat = np.array(s_hat_l_list)                   # L x K
            s_hat_q4_mat = np.array(s_hat_q4_list)                 # L x K
            s_hat_q2_mat = np.array(s_hat_q2_list)                 # L x K
            s_hat_heuristic_mat = np.array(s_hat_heuristic_list)   # L x K
            
            # --- c) Mean-Pooling 聚合 ---
            s_hat_dist_full = np.mean(s_hat_l_mat, axis=0)
            s_hat_dist_q4 = np.mean(s_hat_q4_mat, axis=0)
            s_hat_dist_q2 = np.mean(s_hat_q2_mat, axis=0)
            
            # --- d) LSFD 加权聚合 (Large-Scale Fading Decoding) ---
            weight_sum = np.sum(sys.beta, axis=0) # 对L维度求和，shape: K
            
            # Dist-LSFD-Full
            s_hat_dist_lsfd = np.sum(s_hat_l_mat * sys.beta, axis=0) / weight_sum
            # Heuristic-Bit LSFD Aggregation
            s_hat_dist_heuristic = np.sum(s_hat_heuristic_mat * sys.beta, axis=0) / weight_sum
            
            # 统计误码率
            err_dist_full += calculate_ber(s, s_hat_dist_full)
            err_dist_lsfd += calculate_ber(s, s_hat_dist_lsfd)
            err_dist_q4 += calculate_ber(s, s_hat_dist_q4)
            err_dist_q2 += calculate_ber(s, s_hat_dist_q2)
            err_heuristic += calculate_ber(s, s_hat_dist_heuristic)
            
        # 计算各种方案下的平均 BER
        ber_c_mmse = err_c_mmse / total_bits
        ber_dist_full = err_dist_full / total_bits
        ber_dist_lsfd = err_dist_lsfd / total_bits
        ber_dist_q4 = err_dist_q4 / total_bits
        ber_dist_q2 = err_dist_q2 / total_bits
        ber_heuristic = err_heuristic / total_bits
        
        print(f"Results for P_tx = {p_tx} dBm:")
        print(f"  C-MMSE BER:         {ber_c_mmse:.5f}")
        print(f"  Dist-Full BER:      {ber_dist_full:.5f}")
        print(f"  Dist-LSFD BER:      {ber_dist_lsfd:.5f}")
        print(f"  Dist-Q4 BER:        {ber_dist_q4:.5f}")
        print(f"  Dist-Q2 BER:        {ber_dist_q2:.5f}")
        print(f"  Heuristic-Bit BER:  {ber_heuristic:.5f}")
        
        results.append((p_tx, avg_snr_db, ber_c_mmse, ber_dist_full, ber_dist_lsfd, ber_dist_q4, ber_dist_q2, ber_heuristic))
        
    # 输出最终的格式化对比表格
    print("\n" + "="*110)
    print("                                      FINAL BER RESULTS TABLE")
    print("="*110)
    print(f"{'P_tx (dBm)':<12} | {'Avg SNR (dB)':<14} | {'C-MMSE':<10} | {'Dist-Full':<10} | {'Dist-LSFD':<10} | {'Dist-Q4':<10} | {'Dist-Q2':<10} | {'Heuristic':<10}")
    print("-" * 110)
    for res in results:
        print(f"{res[0]:<12.1f} | {res[1]:<14.2f} | {res[2]:<10.5f} | {res[3]:<10.5f} | {res[4]:<10.5f} | {res[5]:<10.5f} | {res[6]:<10.5f} | {res[7]:<10.5f}")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()
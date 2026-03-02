import torch
import argparse
from dataset import NearFieldDataset

def get_sparse_edge_index(H, snr_db, S):
    """
    基于导频信号能量，提取二分图的稀疏边索引 (Edge Index)。
    
    Args:
        H (torch.Tensor): Batched 信道矩阵，维度 [B, N, K]
        snr_db (float): 信噪比 (dB)
        S (int): 每个用户保留的 Top-S 天线数量
        
    Returns:
        edge_index (torch.Tensor): PyG 格式的全局边索引，维度 [2, B * K * S]
        top_idx (torch.Tensor): 提取的局部天线索引，维度 [B, S, K]
    """
    B, N, K = H.shape
    
    # === 1. 导频模拟 ===
    # 计算信号的平均功率
    signal_power = torch.mean(torch.abs(H)**2)
    # 根据 SNR 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # 生成与 H 同维度的复高斯白噪声 N_p
    noise_real = torch.randn_like(H.real) * torch.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(H.imag) * torch.sqrt(noise_power / 2)
    N_p = noise_real + 1j * noise_imag
    
    # 接收到的导频信号 Y_p = H + N_p
    Y_p = H + N_p
    
    # === 2. 能量计算与 Top-S 提取 ===
    # 计算能量矩阵 E
    E = torch.abs(Y_p)**2  # [B, N, K]
    
    # 对每个用户（在天线维度 dim=1 上），提取能量最大的前 S 个天线索引
    _, top_idx = torch.topk(E, S, dim=1)  # [B, S, K]
    
    # === 3. 构建 edge_index (支持 PyG Batched Graph) ===
    # 为了支持 Batched Graph，需要加上 Batch 偏移。
    # 第 b 个 Batch 中的天线 n 的全局索引: b * N + n
    # 第 b 个 Batch 中的用户 k 的全局索引: b * K + k
    
    # 构造 b 和 k 的索引网格
    b_idx = torch.arange(B, device=H.device).view(B, 1, 1).expand(B, S, K)
    k_idx = torch.arange(K, device=H.device).view(1, 1, K).expand(B, S, K)
    
    # 计算全局索引
    antenna_global_idx = b_idx * N + top_idx  # [B, S, K]
    user_global_idx = b_idx * K + k_idx       # [B, S, K]
    
    # 拼接并展平，构建 edge_index [2, B * K * S]
    edge_index = torch.stack([
        antenna_global_idx.flatten(), 
        user_global_idx.flatten()
    ], dim=0)
    
    return edge_index, top_idx

if __name__ == '__main__':
    # 参数暴露
    parser = argparse.ArgumentParser(description="VR Discovery and Edge Index Construction")
    parser.add_argument('--num_samples', type=int, default=64, help='Number of test samples (Batch Size)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas')
    parser.add_argument('--K', type=int, default=8, help='Number of users')
    parser.add_argument('--S', type=int, default=77, help='Number of top antennas to retain per user')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    args = parser.parse_args()
    
    print(f"==================================================")
    print(f"VR Discovery: Top-S Antenna Selection based on Pilot Energy")
    print(f"Settings: N={args.N}, K={args.K}, S={args.S}, num_samples={args.num_samples}")
    print(f"SNR List (dB): {args.snr_list}")
    print(f"==================================================\n")
    
    # 实例化一个小型测试集
    dataset = NearFieldDataset(num_samples=args.num_samples, N=args.N, K=args.K, snr_range=(0, 0))
    # 取出全部数据作为 1 个 Batch
    H, x, y = dataset[:]
    B = H.shape[0]
    
    expected_edges = B * args.K * args.S
    
    # 遍历不同的 SNR 进行验证
    for snr in args.snr_list:
        edge_index, top_idx = get_sparse_edge_index(H, snr, args.S)
        
        # 验证维度是否严格等于 [2, B * K * S]
        assert edge_index.shape == (2, expected_edges), \
            f"Shape mismatch! Expected (2, {expected_edges}), got {edge_index.shape}"
            
        # 物理意义验证 (VR Hit Rate)
        # 真实的 VR 区域对应的信道元素是非零的
        true_mask = (torch.abs(H) > 0)
        
        # 检查提取的这些天线是否真的落在真实的 VR 掩码内
        selected_true = torch.gather(true_mask, 1, top_idx)
        
        # 计算当前的 Hit Rate
        hit_rate = selected_true.float().mean().item() * 100
        
        # 打印验证结果
        print(f"[SNR: {snr:5.1f} dB] Edge Index Shape: {edge_index.shape} | Expected Edges: {expected_edges} | VR Hit Rate: {hit_rate:.2f}%")
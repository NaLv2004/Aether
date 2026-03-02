import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sparse_gnn_wideband import SparseMPNNWideband, get_sparse_edge_index_wideband
from dataset_wideband import NearFieldWidebandDataset

def train_model(args, device):
    print("Generating Training Set in memory...")
    train_dataset = NearFieldWidebandDataset(
        num_samples=args.num_train, N=args.N_train, K=args.K, F=args.F,
        f_c=args.f_c, B_bw=args.B_bw, snr_range=(args.snr_min, args.snr_max)
    )
    print("Generating Validation Set in memory...")
    val_dataset = NearFieldWidebandDataset(
        num_samples=args.num_val, N=args.N_train, K=args.K, F=args.F,
        f_c=args.f_c, B_bw=args.B_bw, snr_range=(args.snr_min, args.snr_max)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SparseMPNNWideband(num_layers=args.num_layers, hidden_dim=args.hidden_dim, scale_factor=args.scale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting Online Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for H, x, y in train_loader:
            H, x, y = H.to(device), x.to(device), y.to(device)
            B, F, N, K = H.shape
            
            # 使用训练集的 S_train 进行构图
            edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=args.train_snr, S=args.S_train)
            edge_index = edge_index.to(device)
            
            # 融合多载波到 Batch 维度
            H_reshaped = H.view(B * F, N, K)
            x_reshaped = x.view(B * F, K, 1)
            y_reshaped = y.view(B * F, N, 1)
            
            optimizer.zero_grad()
            out = model(y_reshaped, H_reshaped, edge_index, args.S_train, F, args.f_c, args.B_bw)
            
            x_true = torch.cat([x_reshaped.real, x_reshaped.imag], dim=-1)
            loss = criterion(out, x_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * B
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for H, x, y in val_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                B, F, N, K = H.shape
                
                edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=args.train_snr, S=args.S_train)
                edge_index = edge_index.to(device)
                
                H_reshaped = H.view(B * F, N, K)
                x_reshaped = x.view(B * F, K, 1)
                y_reshaped = y.view(B * F, N, 1)
                
                out = model(y_reshaped, H_reshaped, edge_index, args.S_train, F, args.f_c, args.B_bw)
                x_true = torch.cat([x_reshaped.real, x_reshaped.imag], dim=-1)
                loss = criterion(out, x_true)
                val_loss += loss.item() * B
                
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    return model

def test_model(args, model, device):
    print("\n==================================================")
    print("Zero-shot Scale-up Testing")
    print("==================================================\n")
    
    results = {
        'N_list': args.N_list,
        'snr_list': args.snr_list,
        'gnn_ber': {N: [] for N in args.N_list},
        'mmse_ber': {N: [] for N in args.N_list}
    }
    
    model.eval()
    with torch.no_grad():
        for N in args.N_list:
            if N == 256:
                S = 77
            elif N == 1024:
                S = 308
            else:
                S = int(N * 0.3)
                
            print(f"\n--- Testing for N={N}, S={S} ---")
            
            for snr in args.snr_list:
                print(f"Generating test dataset for SNR = {snr} dB...")
                test_dataset = NearFieldWidebandDataset(
                    num_samples=args.num_test, N=N, K=args.K, F=args.F,
                    f_c=args.f_c, B_bw=args.B_bw, snr_range=(snr, snr)
                )
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                total_bits = 2 * args.K * args.F * args.num_test
                errors_gnn = 0
                errors_mmse = 0
                
                for H, x, y in test_loader:
                    H, x, y = H.to(device), x.to(device), y.to(device)
                    B_size, F, N_dim, K = H.shape
                    
                    # --- Sparse-GNN Zero-shot Inference ---
                    edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=snr, S=S)
                    edge_index = edge_index.to(device)
                    
                    H_reshaped = H.view(B_size * F, N_dim, K)
                    x_reshaped = x.view(B_size * F, K, 1)
                    y_reshaped = y.view(B_size * F, N_dim, 1)
                    
                    out = model(y_reshaped, H_reshaped, edge_index, S, F, args.f_c, args.B_bw)
                    
                    pred_bits = torch.sign(out)
                    true_bits = torch.cat([torch.sign(x_reshaped.real), torch.sign(x_reshaped.imag)], dim=-1)
                    errors_gnn += torch.sum(pred_bits != true_bits).item()
                    
                    # --- VR-MMSE Baseline Inference ---
                    signal_power = torch.sum(torch.abs(H_reshaped)**2, dim=(1,2), keepdim=True) / N_dim
                    sigma2 = signal_power / (10 ** (snr / 10))
                    
                    H_H = H_reshaped.mH
                    HtH = torch.bmm(H_H, H_reshaped)
                    Hty = torch.bmm(H_H, y_reshaped)
                    
                    I_K = torch.eye(K, dtype=H_reshaped.dtype, device=device).unsqueeze(0)
                    HtH_mmse = HtH + sigma2 * I_K
                    try:
                        x_mmse = torch.linalg.solve(HtH_mmse, Hty)
                    except RuntimeError:
                        x_mmse = torch.bmm(torch.linalg.pinv(HtH_mmse), Hty)
                        
                    mmse_real_bits = torch.sign(x_mmse.real)
                    mmse_imag_bits = torch.sign(x_mmse.imag)
                    true_real_bits = torch.sign(x_reshaped.real)
                    true_imag_bits = torch.sign(x_reshaped.imag)
                    
                    errors_mmse += torch.sum(true_real_bits != mmse_real_bits).item()
                    errors_mmse += torch.sum(true_imag_bits != mmse_imag_bits).item()
                    
                ber_gnn = errors_gnn / total_bits
                ber_mmse = errors_mmse / total_bits
                
                results['gnn_ber'][N].append(ber_gnn)
                results['mmse_ber'][N].append(ber_mmse)
                
                print(f"[Results] N: {N:4d} | SNR: {snr:5.1f} dB | Sparse-GNN BER: {ber_gnn:.6f} | VR-MMSE BER: {ber_mmse:.6f}")
                
    return results

def benchmark(args, model, device):
    print("\n==================================================")
    print("Inference Time Benchmarking (Batch Size = 1)")
    print("==================================================\n")
    
    times = {
        'gnn': {N: 0.0 for N in args.N_list},
        'mmse': {N: 0.0 for N in args.N_list}
    }
    
    model.eval()
    with torch.no_grad():
        for N in args.N_list:
            if N == 256:
                S = 77
            elif N == 1024:
                S = 308
            else:
                S = int(N * 0.3)
            
            # 生成 Batch Size = 1 的虚拟数据
            dataset = NearFieldWidebandDataset(
                num_samples=1, N=N, K=args.K, F=args.F,
                f_c=args.f_c, B_bw=args.B_bw, snr_range=(10, 10)
            )
            H, x, y = dataset[0]
            H = H.unsqueeze(0).to(device)  # (1, F, N, K)
            x = x.unsqueeze(0).to(device)  # (1, F, K, 1)
            y = y.unsqueeze(0).to(device)  # (1, F, N, 1)
            
            B_size, F, N_dim, K = H.shape
            H_reshaped = H.view(B_size * F, N_dim, K)
            y_reshaped = y.view(B_size * F, N_dim, 1)
            I_K = torch.eye(K, dtype=H_reshaped.dtype, device=device).unsqueeze(0)
            
            # --- Warmup Sparse-GNN ---
            edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=10.0, S=S)
            model(y_reshaped, H_reshaped, edge_index.to(device), S, F, args.f_c, args.B_bw)
            
            # --- Benchmark Sparse-GNN ---
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(100):
                # 必须包含构图时间
                edge_idx, _ = get_sparse_edge_index_wideband(H, snr_db=10.0, S=S)
                model(y_reshaped, H_reshaped, edge_idx.to(device), S, F, args.f_c, args.B_bw)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()
            times['gnn'][N] = (t1 - t0) / 100 * 1000  # ms
            
            # --- Warmup VR-MMSE ---
            signal_power = torch.sum(torch.abs(H_reshaped)**2, dim=(1,2), keepdim=True) / N_dim
            sigma2 = signal_power / (10 ** (10 / 10))
            H_H = H_reshaped.mH
            HtH = torch.bmm(H_H, H_reshaped)
            Hty = torch.bmm(H_H, y_reshaped)
            HtH_mmse = HtH + sigma2 * I_K
            torch.linalg.solve(HtH_mmse, Hty)
            
            # --- Benchmark VR-MMSE ---
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(100):
                signal_power = torch.sum(torch.abs(H_reshaped)**2, dim=(1,2), keepdim=True) / N_dim
                sigma2 = signal_power / (10 ** (10 / 10))
                H_H = H_reshaped.mH
                HtH = torch.bmm(H_H, H_reshaped)
                Hty = torch.bmm(H_H, y_reshaped)
                HtH_mmse = HtH + sigma2 * I_K
                torch.linalg.solve(HtH_mmse, Hty)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()
            times['mmse'][N] = (t1 - t0) / 100 * 1000  # ms
            
            print(f"N={N:4d} | Avg Sparse-GNN Time (inc. graph gen): {times['gnn'][N]:.3f} ms | Avg VR-MMSE Time: {times['mmse'][N]:.3f} ms")
            
    return times

def plot_results(results, times):
    # ================= Plot BER =================
    plt.figure(figsize=(10, 7))
    snr_list = results['snr_list']
    markers = ['o', 's', '^', 'v']
    colors = ['blue', 'red', 'green', 'orange']
    
    idx = 0
    for N in results['N_list']:
        plt.semilogy(snr_list, results['gnn_ber'][N], marker=markers[idx % 4], color=colors[idx % 4], linestyle='-', label=f'Sparse-GNN (N={N})')
        idx += 1
        plt.semilogy(snr_list, results['mmse_ber'][N], marker=markers[idx % 4], color=colors[idx % 4], linestyle='--', label=f'VR-MMSE (N={N})')
        idx += 1
        
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (Log Scale)')
    plt.title('BER vs SNR for Scale-up Evaluation')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('fig_ber.png')
    plt.close()
    
    # ================= Plot Complexity =================
    plt.figure(figsize=(8, 6))
    labels = [f'N={N}' for N in results['N_list']]
    gnn_times = [times['gnn'][N] for N in results['N_list']]
    mmse_times = [times['mmse'][N] for N in results['N_list']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, gnn_times, width, label='Sparse-GNN')
    rects2 = ax.bar(x + width/2, mmse_times, width, label='VR-MMSE')
    
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time Comparison (Batch Size = 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('fig_complexity.png')
    plt.close()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==================================================")
    print("Scale-up Evaluation for Freq-Aware Sparse-GNN")
    print(f"Device: {device}")
    print("==================================================\n")
    
    # 1. 在线训练
    model = train_model(args, device)
    
    # 2. Zero-shot 规模扩展测试
    results = test_model(args, model, device)
    
    # 3. 推断耗时对比
    times = benchmark(args, model, device)
    
    # 4. 绘制对比图表
    plot_results(results, times)
    print("\nEvaluation Complete. Plots saved to 'fig_ber.png' and 'fig_complexity.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale-up Evaluation for Sparse-GNN and VR-MMSE")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--num_train', type=int, default=2000, help='Number of training samples (default: 2000)')
    parser.add_argument('--num_val', type=int, default=200, help='Number of validation samples (default: 200)')
    parser.add_argument('--num_test', type=int, default=500, help='Number of test samples per SNR (default: 500)')
    parser.add_argument('--N_train', type=int, default=256, help='Number of antennas for training (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--F', type=int, default=16, help='Number of subcarriers (default: 16)')
    parser.add_argument('--f_c', type=float, default=0.1e12, help='Carrier frequency in Hz (default: 0.1e12)')
    parser.add_argument('--B_bw', type=float, default=10e9, help='Total Bandwidth in Hz (default: 10e9)')
    parser.add_argument('--S_train', type=int, default=77, help='Number of retained antennas per user for training (default: 77)')
    parser.add_argument('--train_snr', type=float, default=10.0, help='SNR for VR discovery during training (default: 10.0)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB for training (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB for training (default: 15)')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    parser.add_argument('--N_list', type=int, nargs='+', default=[256, 1024], help='List of Antenna sizes to evaluate')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of message passing layers (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size (default: 64)')
    parser.add_argument('--scale_factor', type=float, default=1e5, help='Scaling factor for H and y (default: 1e5)')
    
    args = parser.parse_args()
    main(args)
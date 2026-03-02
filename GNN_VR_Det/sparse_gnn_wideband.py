import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from dataset_wideband import NearFieldWidebandDataset

def get_sparse_edge_index_wideband(H, snr_db, S):
    """
    基于导频信号能量，提取宽带二分图的共享稀疏边索引 (Edge Index)。
    """
    B, F, N, K = H.shape
    
    # === 1. 导频模拟 ===
    signal_power = torch.mean(torch.abs(H)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    noise_real = torch.randn_like(H.real) * torch.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(H.imag) * torch.sqrt(noise_power / 2)
    N_p = noise_real + 1j * noise_imag
    
    Y_p = H + N_p
    
    # === 2. 在子载波维度上求平均能量 ===
    E = torch.mean(torch.abs(Y_p)**2, dim=1)  # [B, N, K]
    
    # === 3. Top-S 提取 ===
    _, top_idx = torch.topk(E, S, dim=1)  # [B, S, K]
    
    # === 4. 扩展到所有子载波 ===
    top_idx_expand = top_idx.unsqueeze(1).expand(B, F, S, K)  # [B, F, S, K]
    top_idx_flat = top_idx_expand.reshape(B * F, S, K)  # [B*F, S, K]
    
    # === 5. 构建 edge_index ===
    B_eff = B * F
    b_idx = torch.arange(B_eff, device=H.device).view(B_eff, 1, 1).expand(B_eff, S, K)
    k_idx = torch.arange(K, device=H.device).view(1, 1, K).expand(B_eff, S, K)
    
    antenna_global_idx = b_idx * N + top_idx_flat
    user_global_idx = b_idx * K + k_idx
    
    edge_index = torch.stack([
        antenna_global_idx.flatten(), 
        user_global_idx.flatten()
    ], dim=0)
    
    return edge_index, top_idx_flat

class SparseMPNNLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SparseMPNNLayer, self).__init__()
        # A2U MLPs
        self.mlp_a2u = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.mlp_u = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # U2A MLPs
        self.mlp_u2a = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h_v, h_u, e_feat, edge_index, S):
        # ================= A2U (Antenna to User) =================
        src_v = h_v[edge_index[0]] # (E, hidden_dim)
        dst_u = h_u[edge_index[1]] # (E, hidden_dim)
        
        msg_a2u_in = torch.cat([src_v, dst_u, e_feat], dim=-1) # (E, 192)
        msg_a2u_out = self.mlp_a2u(msg_a2u_in) # (E, 64)
        
        # Aggregate to users using pure PyTorch scatter_add_
        m_u = torch.zeros_like(h_u) # (B_eff*K, 64)
        idx_u = edge_index[1].unsqueeze(1).expand(-1, msg_a2u_out.size(1)).long()
        m_u.scatter_add_(0, idx_u, msg_a2u_out)
        m_u = m_u / S # Divide by fixed degree S
        
        h_u_new = torch.cat([h_u, m_u], dim=-1) # (B_eff*K, 128)
        h_u_out = self.mlp_u(h_u_new) # (B_eff*K, 64)

        # ================= U2A (User to Antenna) =================
        src_u = h_u_out[edge_index[1]] # (E, hidden_dim)
        dst_v = h_v[edge_index[0]] # (E, hidden_dim)
        
        msg_u2a_in = torch.cat([src_u, dst_v, e_feat], dim=-1) # (E, 192)
        msg_u2a_out = self.mlp_u2a(msg_u2a_in) # (E, 64)
        
        # Aggregate to antennas using pure PyTorch scatter_add_
        m_v = torch.zeros_like(h_v) # (B_eff*N, 64)
        idx_v = edge_index[0].unsqueeze(1).expand(-1, msg_u2a_out.size(1)).long()
        m_v.scatter_add_(0, idx_v, msg_u2a_out)
        
        # Dynamic degree calculation for antennas
        degree_v = torch.zeros(h_v.size(0), 1, dtype=h_v.dtype, device=h_v.device)
        ones = torch.ones(edge_index.size(1), 1, dtype=h_v.dtype, device=h_v.device)
        idx_deg = edge_index[0].unsqueeze(1).long()
        degree_v.scatter_add_(0, idx_deg, ones)
        degree_v = torch.clamp(degree_v, min=1.0) # Prevent division by zero
        
        m_v = m_v / degree_v
        
        h_v_new = torch.cat([h_v, m_v], dim=-1) # (B_eff*N, 128)
        h_v_out = self.mlp_a(h_v_new) # (B_eff*N, 64)

        return h_v_out, h_u_out

class SparseMPNNWideband(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, scale_factor=1e5):
        super(SparseMPNNWideband, self).__init__()
        self.scale_factor = scale_factor
        self.emb_v = nn.Linear(2, hidden_dim)
        self.emb_u = nn.Linear(2, hidden_dim)
        self.emb_e = nn.Linear(3, hidden_dim) # 3 features: real, imag, delta_f
        
        self.layers = nn.ModuleList([SparseMPNNLayer(hidden_dim) for _ in range(num_layers)])
        
        self.readout = nn.Linear(hidden_dim, 2)

    def forward(self, y, H, edge_index, S, F, f_c, B_bw):
        # y: (B_eff, N, 1), H: (B_eff, N, K)
        B_eff, N, K = H.shape
        
        # Scale inputs
        y_scaled = y * self.scale_factor
        
        # Node flattening
        v = torch.cat([y_scaled.real, y_scaled.imag], dim=-1).view(B_eff * N, 2)
        u = torch.zeros((B_eff * K, 2), dtype=torch.float32, device=y.device)
        
        # Sparse edge features extraction
        b_idx = (edge_index[0] // N).long()
        n_idx = (edge_index[0] % N).long()
        k_idx = (edge_index[1] % K).long()
        
        H_edges = H[b_idx, n_idx, k_idx] * self.scale_factor
        
        # Frequency offset calculation
        m_idx = b_idx % F
        f_m = f_c + (B_bw / F) * (m_idx - F / 2)
        delta_f = (f_m - f_c) / f_c
        
        e = torch.stack([H_edges.real, H_edges.imag, delta_f.float()], dim=-1) # (E, 3)
        
        # Embeddings
        h_v = self.emb_v(v) # (B_eff*N, 64)
        h_u = self.emb_u(u) # (B_eff*K, 64)
        e_feat = self.emb_e(e) # (E, 64)
        
        # Message Passing Layers
        for layer in self.layers:
            h_v, h_u = layer(h_v, h_u, e_feat, edge_index, S)
            
        # Readout
        out = self.readout(h_u) # (B_eff*K, 2)
        out = out.view(B_eff, K, 2)
        return out

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Training Wideband Freq-Aware Sparse-GNN")
    print(f"Settings: N={args.N}, K={args.K}, F={args.F}, S={args.S}")
    print(f"Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
    print(f"Train samples={args.num_train}, Val samples={args.num_val}, Test samples={args.num_test}")
    print(f"f_c={args.f_c} Hz, B_bw={args.B_bw} Hz")
    print(f"Using Device: {device}")
    print(f"==================================================\n")
    
    # Generate static train/val sets in memory
    print("Generating training dataset in memory...")
    train_dataset = NearFieldWidebandDataset(
        num_samples=args.num_train, N=args.N, K=args.K, F=args.F,
        f_c=args.f_c, B_bw=args.B_bw, snr_range=(args.snr_min, args.snr_max)
    )
    print("Generating validation dataset in memory...")
    val_dataset = NearFieldWidebandDataset(
        num_samples=args.num_val, N=args.N, K=args.K, F=args.F,
        f_c=args.f_c, B_bw=args.B_bw, snr_range=(args.snr_min, args.snr_max)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SparseMPNNWideband(num_layers=args.num_layers, hidden_dim=args.hidden_dim, scale_factor=args.scale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for H, x, y in train_loader:
            H, x, y = H.to(device), x.to(device), y.to(device)
            B, F, N, K = H.shape
            
            # Dynamic wideband sparse graph generation
            edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=args.train_snr, S=args.S)
            edge_index = edge_index.to(device)
            
            # Reshape to merge batch and subcarrier dimensions
            H_reshaped = H.view(B * F, N, K)
            x_reshaped = x.view(B * F, K, 1)
            y_reshaped = y.view(B * F, N, 1)
            
            optimizer.zero_grad()
            out = model(y_reshaped, H_reshaped, edge_index, args.S, args.F, args.f_c, args.B_bw)
            
            # Ground truth: real and imag parts of x
            x_true = torch.cat([x_reshaped.real, x_reshaped.imag], dim=-1)
            loss = criterion(out, x_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * B
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for H, x, y in val_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                B, F, N, K = H.shape
                
                edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=args.train_snr, S=args.S)
                edge_index = edge_index.to(device)
                
                H_reshaped = H.view(B * F, N, K)
                x_reshaped = x.view(B * F, K, 1)
                y_reshaped = y.view(B * F, N, 1)
                
                out = model(y_reshaped, H_reshaped, edge_index, args.S, args.F, args.f_c, args.B_bw)
                x_true = torch.cat([x_reshaped.real, x_reshaped.imag], dim=-1)
                loss = criterion(out, x_true)
                val_loss += loss.item() * B
                
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Average Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
    print("\n==================================================")
    print("Training finished. Starting Evaluation on Test Sets (Different SNRs)...")
    print("==================================================\n")
    
    snr_list = args.snr_list
    model.eval()
    with torch.no_grad():
        for snr in snr_list:
            print(f"Generating test dataset for SNR = {snr} dB...")
            test_dataset = NearFieldWidebandDataset(
                num_samples=args.num_test, N=args.N, K=args.K, F=args.F,
                f_c=args.f_c, B_bw=args.B_bw, snr_range=(snr, snr)
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            errors = 0
            total_bits = 2 * args.K * args.F * args.num_test
            
            for H, x, y in test_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                B, F, N, K = H.shape
                
                # 使用真实的测试 snr 进行构图
                edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=snr, S=args.S)
                edge_index = edge_index.to(device)
                
                H_reshaped = H.view(B * F, N, K)
                x_reshaped = x.view(B * F, K, 1)
                y_reshaped = y.view(B * F, N, 1)
                
                out = model(y_reshaped, H_reshaped, edge_index, args.S, args.F, args.f_c, args.B_bw)
                
                # 解调预测比特和真实比特
                pred_bits = torch.sign(out)
                true_bits = torch.cat([torch.sign(x_reshaped.real), torch.sign(x_reshaped.imag)], dim=-1)
                
                # 统计错误数
                errors += torch.sum(pred_bits != true_bits).item()
                
            ber = errors / total_bits
            print(f"[Test Results] SNR: {snr:5.1f} dB | Wideband Sparse-GNN BER: {ber:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Wideband Freq-Aware Sparse-GNN")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--num_train', type=int, default=2000, help='Number of training samples (default: 2000)')
    parser.add_argument('--num_val', type=int, default=200, help='Number of validation samples (default: 200)')
    parser.add_argument('--num_test', type=int, default=500, help='Number of test samples per SNR (default: 500)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--F', type=int, default=16, help='Number of subcarriers (default: 16)')
    parser.add_argument('--f_c', type=float, default=0.1e12, help='Carrier frequency in Hz (default: 0.1e12)')
    parser.add_argument('--B_bw', type=float, default=10e9, help='Total Bandwidth in Hz (default: 10e9)')
    parser.add_argument('--S', type=int, default=77, help='Number of retained antennas per user (default: 77)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB for training (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB for training (default: 15)')
    parser.add_argument('--train_snr', type=float, default=10.0, help='SNR for VR discovery during training (default: 10.0)')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of message passing layers (default: 6)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size (default: 128)')
    parser.add_argument('--scale_factor', type=float, default=1e5, help='Scaling factor for H and y (default: 1e5)')
    
    args = parser.parse_args()
    main(args)
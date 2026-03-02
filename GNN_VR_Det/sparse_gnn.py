import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from dataset import NearFieldDataset, get_dataloaders
from vr_discovery import get_sparse_edge_index
from mpnn import MPNN

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
        m_u = torch.zeros_like(h_u) # (B*K, 64)
        idx_u = edge_index[1].unsqueeze(1).expand(-1, msg_a2u_out.size(1)).long()
        m_u.scatter_add_(0, idx_u, msg_a2u_out)
        m_u = m_u / S # Divide by fixed degree S
        
        h_u_new = torch.cat([h_u, m_u], dim=-1) # (B*K, 128)
        h_u_out = self.mlp_u(h_u_new) # (B*K, 64)

        # ================= U2A (User to Antenna) =================
        src_u = h_u_out[edge_index[1]] # (E, hidden_dim)
        dst_v = h_v[edge_index[0]] # (E, hidden_dim)
        
        msg_u2a_in = torch.cat([src_u, dst_v, e_feat], dim=-1) # (E, 192)
        msg_u2a_out = self.mlp_u2a(msg_u2a_in) # (E, 64)
        
        # Aggregate to antennas using pure PyTorch scatter_add_
        m_v = torch.zeros_like(h_v) # (B*N, 64)
        idx_v = edge_index[0].unsqueeze(1).expand(-1, msg_u2a_out.size(1)).long()
        m_v.scatter_add_(0, idx_v, msg_u2a_out)
        
        # Dynamic degree calculation for antennas
        degree_v = torch.zeros(h_v.size(0), 1, dtype=h_v.dtype, device=h_v.device)
        ones = torch.ones(edge_index.size(1), 1, dtype=h_v.dtype, device=h_v.device)
        idx_deg = edge_index[0].unsqueeze(1).long()
        degree_v.scatter_add_(0, idx_deg, ones)
        degree_v = torch.clamp(degree_v, min=1.0) # Prevent division by zero
        
        m_v = m_v / degree_v
        
        h_v_new = torch.cat([h_v, m_v], dim=-1) # (B*N, 128)
        h_v_out = self.mlp_a(h_v_new) # (B*N, 64)

        return h_v_out, h_u_out

class SparseMPNN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, scale_factor=1e5):
        super(SparseMPNN, self).__init__()
        self.scale_factor = scale_factor
        self.emb_v = nn.Linear(2, hidden_dim)
        self.emb_u = nn.Linear(2, hidden_dim)
        self.emb_e = nn.Linear(2, hidden_dim)
        
        self.layers = nn.ModuleList([SparseMPNNLayer(hidden_dim) for _ in range(num_layers)])
        
        self.readout = nn.Linear(hidden_dim, 2)

    def forward(self, y, H, edge_index, S):
        B, N, K = H.shape
        
        # Scale inputs
        y_scaled = y * self.scale_factor
        
        # Node flattening
        v = torch.cat([y_scaled.real, y_scaled.imag], dim=-1).view(B * N, 2)
        u = torch.zeros((B * K, 2), dtype=torch.float32, device=y.device)
        
        # Sparse edge features extraction
        b_idx = (edge_index[0] // N).long()
        n_idx = (edge_index[0] % N).long()
        k_idx = (edge_index[1] % K).long()
        
        H_edges = H[b_idx, n_idx, k_idx] * self.scale_factor
        e = torch.stack([H_edges.real, H_edges.imag], dim=-1) # (E, 2)
        
        # Embeddings
        h_v = self.emb_v(v) # (B*N, 64)
        h_u = self.emb_u(u) # (B*K, 64)
        e_feat = self.emb_e(e) # (E, 64)
        
        # Message Passing Layers
        for layer in self.layers:
            h_v, h_u = layer(h_v, h_u, e_feat, edge_index, S)
            
        # Readout
        out = self.readout(h_u) # (B*K, 2)
        out = out.view(B, K, 2)
        return out

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Training Sparse-GNN for MIMO Detection")
    print(f"Settings: N={args.N}, K={args.K}, S={args.S}, Epochs={args.epochs}, Batch Size={args.batch_size}")
    print(f"Learning Rate={args.lr}, Hidden Dim={args.hidden_dim}, Layers={args.num_layers}")
    print(f"Scale Factor={args.scale_factor}, Train VR SNR={args.train_snr} dB")
    print(f"Using Device: {device}")
    print(f"==================================================\n")
    
    # Generate static train/val sets in memory
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, 
        num_train=args.num_train, 
        num_val=args.num_val, 
        num_test=10, # minimal dummy test set
        N=args.N, 
        K=args.K, 
        snr_range=(args.snr_min, args.snr_max)
    )
    
    model = SparseMPNN(num_layers=args.num_layers, hidden_dim=args.hidden_dim, scale_factor=args.scale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for H, x, y in train_loader:
            H, x, y = H.to(device), x.to(device), y.to(device)
            
            # Dynamic sparse graph generation
            edge_index, _ = get_sparse_edge_index(H, snr_db=args.train_snr, S=args.S)
            edge_index = edge_index.to(device)
            
            optimizer.zero_grad()
            out = model(y, H, edge_index, args.S)
            
            # Ground truth: real and imag parts of x
            x_true = torch.cat([x.real, x.imag], dim=-1)
            loss = criterion(out, x_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * H.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for H, x, y in val_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                edge_index, _ = get_sparse_edge_index(H, snr_db=args.train_snr, S=args.S)
                edge_index = edge_index.to(device)
                
                out = model(y, H, edge_index, args.S)
                x_true = torch.cat([x.real, x.imag], dim=-1)
                loss = criterion(out, x_true)
                val_loss += loss.item() * H.size(0)
                
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Average Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
    print("\n==================================================")
    print("Training finished. Starting Evaluation on Test Sets (Different SNRs)...")
    print("==================================================\n")
    
    snr_list = args.snr_list
    model.eval()
    with torch.no_grad():
        for snr in snr_list:
            print(f"Generating test dataset for SNR = {snr} dB...")
            # 动态生成特定 SNR 的测试集
            test_dataset = NearFieldDataset(num_samples=args.num_test, N=args.N, K=args.K, snr_range=(snr, snr))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            errors = 0
            total_bits = 2 * args.K * args.num_test
            
            for H, x, y in test_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                
                # 使用真实的测试 snr 进行构图
                edge_index, _ = get_sparse_edge_index(H, snr_db=snr, S=args.S)
                edge_index = edge_index.to(device)
                
                out = model(y, H, edge_index, args.S) # (B, K, 2)
                
                # 解调预测比特和真实比特
                pred_bits = torch.sign(out)
                true_bits = torch.cat([torch.sign(x.real), torch.sign(x.imag)], dim=-1)
                
                # 统计错误数
                errors += torch.sum(pred_bits != true_bits).item()
                
            ber = errors / total_bits
            print(f"[Test Results] SNR: {snr:5.1f} dB | Sparse-GNN BER: {ber:.6f}\n")

    print("==================================================")
    print("Computation Efficiency Comparison")
    print("==================================================\n")
    # 随机生成一个 Batch 的数据进行测速
    B_eff, N_eff, K_eff = 64, args.N, args.K
    H_eff = torch.randn(B_eff, N_eff, K_eff, dtype=torch.complex64).to(device)
    y_eff = torch.randn(B_eff, N_eff, 1, dtype=torch.complex64).to(device)
    
    full_model = MPNN(num_layers=args.num_layers, hidden_dim=args.hidden_dim, scale_factor=args.scale_factor).to(device)
    full_model.eval()
    sparse_model = model
    sparse_model.eval()
    
    with torch.no_grad():
        # Warmup
        full_model(y_eff, H_eff)
        edge_index_eff, _ = get_sparse_edge_index(H_eff, snr_db=10.0, S=args.S)
        sparse_model(y_eff, H_eff, edge_index_eff.to(device), args.S)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            full_model(y_eff, H_eff)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2 = time.time()
        for _ in range(100):
            edge_index_eff, _ = get_sparse_edge_index(H_eff, snr_db=10.0, S=args.S)
            sparse_model(y_eff, H_eff, edge_index_eff.to(device), args.S)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t3 = time.time()
        
    print(f"Full MPNN Time (100 passes): {t1 - t0:.4f} seconds")
    print(f"Sparse MPNN Time (100 passes, including graph gen): {t3 - t2:.4f} seconds")
    print(f"Speedup: {(t1 - t0) / (t3 - t2):.2f}x\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Sparse-GNN for MIMO Detection")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_train', type=int, default=10000, help='Number of training samples (default: 10000)')
    parser.add_argument('--num_val', type=int, default=1000, help='Number of validation samples (default: 1000)')
    parser.add_argument('--num_test', type=int, default=2000, help='Number of test samples per SNR (default: 2000)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--S', type=int, default=77, help='Number of retained antennas per user (default: 77)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB for training (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB for training (default: 15)')
    parser.add_argument('--train_snr', type=float, default=10.0, help='SNR for VR discovery during training (default: 10.0)')
    parser.add_argument('--snr_list', type=float, nargs='+', default=[-5, 0, 5, 10, 15], help='List of SNRs in dB to evaluate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of message passing layers (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size (default: 64)')
    parser.add_argument('--scale_factor', type=float, default=1e5, help='Scaling factor for H and y (default: 1e5)')
    
    args = parser.parse_args()
    main(args)
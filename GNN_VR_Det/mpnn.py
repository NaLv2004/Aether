import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from dataset import NearFieldDataset, get_dataloaders

class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super(MPNNLayer, self).__init__()
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

    def forward(self, h_v, h_u, h_e):
        B, N, _ = h_v.shape
        _, K, _ = h_u.shape

        # ================= A2U (Antenna to User) =================
        # Expand h_v and h_u to (B, N, K, hidden_dim)
        h_v_exp = h_v.unsqueeze(2).expand(B, N, K, -1)
        h_u_exp = h_u.unsqueeze(1).expand(B, N, K, -1)
        
        # Concat [h_v, h_u, h_e] -> (B, N, K, 192)
        msg_a2u_in = torch.cat([h_v_exp, h_u_exp, h_e], dim=-1)
        msg_a2u_out = self.mlp_a2u(msg_a2u_in) # (B, N, K, 64)
        
        # Aggregate over Antennas (dim=1)
        m_u = torch.mean(msg_a2u_out, dim=1) # (B, K, 64)
        
        # Update User representation
        h_u_new = torch.cat([h_u, m_u], dim=-1) # (B, K, 128)
        h_u = self.mlp_u(h_u_new) # (B, K, 64)

        # ================= U2A (User to Antenna) =================
        # Use updated h_u, expand to (B, N, K, hidden_dim)
        h_u_exp2 = h_u.unsqueeze(1).expand(B, N, K, -1)
        
        # Concat [h_u, h_v, h_e] -> (B, N, K, 192)
        msg_u2a_in = torch.cat([h_u_exp2, h_v_exp, h_e], dim=-1)
        msg_u2a_out = self.mlp_u2a(msg_u2a_in) # (B, N, K, 64)
        
        # Aggregate over Users (dim=2)
        m_v = torch.mean(msg_u2a_out, dim=2) # (B, N, 64)
        
        # Update Antenna representation
        h_v_new = torch.cat([h_v, m_v], dim=-1) # (B, N, 128)
        h_v = self.mlp_a(h_v_new) # (B, N, 64)

        return h_v, h_u

class MPNN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, scale_factor=1e5):
        super(MPNN, self).__init__()
        self.scale_factor = scale_factor
        self.emb_v = nn.Linear(2, hidden_dim)
        self.emb_u = nn.Linear(2, hidden_dim)
        self.emb_e = nn.Linear(2, hidden_dim)
        
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim) for _ in range(num_layers)])
        
        self.readout = nn.Linear(hidden_dim, 2)

    def forward(self, y, H):
        B, N, K = H.shape
        
        # Scale inputs to avoid vanishing gradients
        y_scaled = y * self.scale_factor
        H_scaled = H * self.scale_factor
        
        # Node features
        v = torch.cat([y_scaled.real, y_scaled.imag], dim=-1) # (B, N, 2)
        u = torch.zeros((B, K, 2), dtype=torch.float32, device=y.device) # (B, K, 2)
        
        # Edge features
        e = torch.stack([H_scaled.real, H_scaled.imag], dim=-1) # (B, N, K, 2)
        
        # Embeddings
        h_v = self.emb_v(v) # (B, N, 64)
        h_u = self.emb_u(u) # (B, K, 64)
        h_e = self.emb_e(e) # (B, N, K, 64)
        
        # Message Passing Layers
        for layer in self.layers:
            h_v, h_u = layer(h_v, h_u, h_e)
            
        # Readout
        out = self.readout(h_u) # (B, K, 2)
        return out

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Training MPNN for MIMO Detection")
    print(f"Settings: N={args.N}, K={args.K}, Epochs={args.epochs}, Batch Size={args.batch_size}")
    print(f"Learning Rate={args.lr}, Hidden Dim={args.hidden_dim}, Layers={args.num_layers}")
    print(f"Scale Factor={args.scale_factor}")
    print(f"Using Device: {device}")
    print(f"==================================================\n")
    
    # Generate static train/val sets in memory
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, 
        num_train=args.num_train, 
        num_val=args.num_val, 
        num_test=args.num_test, 
        N=args.N, 
        K=args.K, 
        snr_range=(args.snr_min, args.snr_max)
    )
    
    model = MPNN(num_layers=args.num_layers, hidden_dim=args.hidden_dim, scale_factor=args.scale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for H, x, y in train_loader:
            H, x, y = H.to(device), x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(y, H)
            
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
                out = model(y, H)
                x_true = torch.cat([x.real, x.imag], dim=-1)
                loss = criterion(out, x_true)
                val_loss += loss.item() * H.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Average Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
    print("\n==================================================")
    print("Training finished. Starting Evaluation on Test Sets (Different SNRs)...")
    print("==================================================\n")
    
    snr_list = [-5, 0, 5, 10, 15]
    model.eval()
    with torch.no_grad():
        for snr in snr_list:
            print(f"Generating test dataset for SNR = {snr} dB...")
            # 动态生成特定 SNR 的测试集
            test_dataset = NearFieldDataset(num_samples=2000, N=args.N, K=args.K, snr_range=(snr, snr))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            errors = 0
            total_bits = 2 * args.K * 2000
            
            for H, x, y in test_loader:
                H, x, y = H.to(device), x.to(device), y.to(device)
                out = model(y, H) # (B, K, 2)
                
                # 解调预测比特和真实比特
                pred_bits = torch.sign(out)
                true_bits = torch.cat([torch.sign(x.real), torch.sign(x.imag)], dim=-1)
                
                # 统计错误数
                errors += torch.sum(pred_bits != true_bits).item()
                
            ber = errors / total_bits
            print(f"[Test Results] SNR: {snr:5.1f} dB | MPNN BER: {ber:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate MPNN for MIMO Detection")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_train', type=int, default=10000, help='Number of training samples (default: 10000)')
    parser.add_argument('--num_val', type=int, default=1000, help='Number of validation samples (default: 1000)')
    parser.add_argument('--num_test', type=int, default=10, help='Number of dummy test samples for init (default: 10)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB for training (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB for training (default: 15)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of message passing layers (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size (default: 64)')
    parser.add_argument('--scale_factor', type=float, default=1e5, help='Scaling factor for H and y (default: 1e5)')
    
    args = parser.parse_args()
    main(args)
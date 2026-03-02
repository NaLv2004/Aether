import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

class NearFieldDataset(Dataset):
    def __init__(self, num_samples, N=256, K=8, snr_range=(-5, 15)):
        """
        Near-Field Non-Stationary Channel Dataset.
        Args:
            num_samples (int): Number of samples to generate.
            N (int): Number of antennas.
            K (int): Number of users.
            snr_range (tuple): Range of SNR in dB.
        """
        self.num_samples = num_samples
        self.N = N
        self.K = K
        self.snr_range = snr_range
        
        # Physics constants
        self.f = 0.1e12  # 0.1 THz (100 GHz)
        self.c = 3e8     # Speed of light (m/s)
        self.lam = self.c / self.f
        self.d = self.lam / 2
        
        # Generate all data in memory at initialization
        self.generate_data()
        
    def generate_data(self):
        # 1. Antenna coordinates q_n: shape (N, 3)
        n_idx = torch.arange(self.N)
        q_n = torch.zeros((self.N, 3))
        # Uniform linear array along y-axis
        q_n[:, 1] = (n_idx - self.N / 2) * self.d
        
        # 2. User distance and angle
        # Distances: D in [5, 30] meters
        D = 5 + 25 * torch.rand(self.num_samples, self.K)
        self.D_all = D
        # Theta in [-pi/2, pi/2]
        theta = -np.pi / 2 + np.pi * torch.rand(self.num_samples, self.K)
        
        # User coordinates p_k: shape (num_samples, K, 3)
        p_k = torch.zeros((self.num_samples, self.K, 3))
        p_k[:, :, 0] = D * torch.cos(theta)
        p_k[:, :, 1] = D * torch.sin(theta)
        
        # 3. Distance between antennas and users r_{n,k} = || p_k - q_n ||_2
        # p_k expanded: (num_samples, K, 1, 3)
        # q_n expanded: (1, 1, N, 3)
        diff = p_k.unsqueeze(2) - q_n.unsqueeze(0).unsqueeze(0)  # (num_samples, K, N, 3)
        r_nk = torch.norm(diff, dim=-1)                          # (num_samples, K, N)
        r_nk = r_nk.transpose(1, 2)                              # (num_samples, N, K)
        
        # 4. True VR (Visible Region) Mask M_{n,k}
        c_k = torch.randint(0, self.N, (self.num_samples, self.K))  # (num_samples, K)
        R_v = 0.3 * self.N
        n_idx_expand = torch.arange(self.N).unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        c_k_expand = c_k.unsqueeze(1)                                  # (num_samples, 1, K)
        # Condition: |n - c_k| <= R_v
        M_nk = (torch.abs(n_idx_expand - c_k_expand) <= R_v).float()   # (num_samples, N, K)
        
        # 5. Channel Matrix H
        # H_{n,k} = M_{n,k} * (lambda / (4 * pi * r_{n,k})) * exp(-j * 2 * pi * r_{n,k} / lambda)
        amplitude = M_nk * (self.lam / (4 * np.pi * r_nk))
        phase = -2 * np.pi * r_nk / self.lam
        self.H_all = amplitude * torch.exp(1j * phase)  # (num_samples, N, K)
        
        # 6. QPSK symbols x
        # Values uniformly chosen from {1+j, 1-j, -1+j, -1-j} / sqrt(2)
        real_part = (torch.randint(0, 2, (self.num_samples, self.K, 1)) * 2 - 1).float() / np.sqrt(2)
        imag_part = (torch.randint(0, 2, (self.num_samples, self.K, 1)) * 2 - 1).float() / np.sqrt(2)
        self.x_all = real_part + 1j * imag_part  # (num_samples, K, 1)
        
        # 7. Received signal y = Hx + n
        # Noise-free signal Hx: (num_samples, N, 1)
        Hx = torch.bmm(self.H_all, self.x_all)
        
        # Sample SNR uniformly from snr_range (in dB)
        snr_db = self.snr_range[0] + (self.snr_range[1] - self.snr_range[0]) * torch.rand(self.num_samples, 1, 1)
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate signal power and required noise power
        signal_power = torch.sum(torch.abs(Hx)**2, dim=1, keepdim=True) / self.N  # (num_samples, 1, 1)
        noise_power = signal_power / snr_linear
        
        # Complex Gaussian white noise
        noise_real = torch.randn((self.num_samples, self.N, 1)) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn((self.num_samples, self.N, 1)) * torch.sqrt(noise_power / 2)
        self.noise_all = noise_real + 1j * noise_imag
        
        # Final noisy received signal
        self.y_all = Hx + self.noise_all

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.H_all[idx], self.x_all[idx], self.y_all[idx]


def get_dataloaders(batch_size=64, num_train=10000, num_val=1000, num_test=1000, N=256, K=8, snr_range=(-5, 15)):
    """
    Returns train, validation, and test dataloaders containing the near-field non-stationary channel data.
    """
    print(f"Generating Training Set ({num_train} samples)...")
    train_dataset = NearFieldDataset(num_train, N=N, K=K, snr_range=snr_range)
    
    print(f"Generating Validation Set ({num_val} samples)...")
    val_dataset = NearFieldDataset(num_val, N=N, K=K, snr_range=snr_range)
    
    print(f"Generating Test Set ({num_test} samples)...")
    test_dataset = NearFieldDataset(num_test, N=N, K=K, snr_range=snr_range)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Near Field Dataset Generator")
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_train', type=int, default=10000, help='Number of training samples (default: 10000)')
    parser.add_argument('--num_val', type=int, default=1000, help='Number of validation samples (default: 1000)')
    parser.add_argument('--num_test', type=int, default=1000, help='Number of testing samples (default: 1000)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB (default: 15)')
    args = parser.parse_args()

    # Create a small dataset for testing and printing as requested
    print(f"=== Testing NearFieldDataset with 10 samples (N={args.N}, K={args.K}) ===")
    dataset = NearFieldDataset(num_samples=10, N=args.N, K=args.K, snr_range=(args.snr_min, args.snr_max))
    
    H, x, y = dataset[0]
    print(f"\n[1] Tensor dimensions:")
    print(f"    H shape: {H.shape}")
    print(f"    x shape: {x.shape}")
    print(f"    y shape: {y.shape}")
    
    # Distance D of users in the first sample
    D = dataset.D_all[0]
    print(f"\n[2] Distances D of users in the first sample (meters):")
    print(f"    {D.numpy()}")
    
    # Sparsity of H in the first sample
    non_zero_elements = torch.sum(H != 0).item()
    total_elements = H.numel()
    sparsity = non_zero_elements / total_elements
    print(f"\n[3] Sparsity of H (non-zero ratio) in the first sample:")
    print(f"    {sparsity*100:.2f}% (Expected ~60%)")
    
    # Actual received SNR
    Hx = torch.matmul(H, x)
    noise = dataset.noise_all[0]
    signal_power = torch.sum(torch.abs(Hx)**2)
    noise_power = torch.sum(torch.abs(noise)**2)
    actual_snr = 10 * torch.log10(signal_power / noise_power)
    print(f"\n[4] Actual received SNR for the first sample:")
    print(f"    {actual_snr.item():.2f} dB (Expected in [{args.snr_min}, {args.snr_max}] dB)")
    
    # Average energy magnitude of H
    avg_energy = torch.mean(torch.abs(dataset.H_all)**2)
    print(f"\n[5] Average energy magnitude of H over all 10 samples:")
    print(f"    {avg_energy.item():.2e}")
    print("\n==================================================================")
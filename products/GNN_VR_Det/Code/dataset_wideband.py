import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

class NearFieldWidebandDataset(Dataset):
    def __init__(self, num_samples, N=256, K=8, F=16, f_c=0.1e12, B_bw=10e9, snr_range=(-5, 15)):
        """
        Near-Field Wideband Non-Stationary Channel Dataset.
        Args:
            num_samples (int): Number of samples to generate.
            N (int): Number of antennas.
            K (int): Number of users.
            F (int): Number of subcarriers.
            f_c (float): Carrier frequency in Hz.
            B_bw (float): Total Bandwidth in Hz.
            snr_range (tuple): Range of SNR in dB.
        """
        self.num_samples = num_samples
        self.N = N
        self.K = K
        self.F = F
        self.f_c = f_c
        self.B_bw = B_bw
        self.snr_range = snr_range
        
        # Physics constants
        self.c = 3e8     # Speed of light (m/s)
        self.d = self.c / (2 * self.f_c) # Antenna spacing based on carrier frequency
        
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
        
        # 5. Wideband Channel Matrix H
        # Subcarrier frequencies
        m_idx = torch.arange(self.F)
        f_m = self.f_c + (self.B_bw / self.F) * (m_idx - self.F / 2) # (F,)
        
        # Reshape for broadcasting
        f_m_expand = f_m.view(1, self.F, 1, 1)    # (1, F, 1, 1)
        r_nk_expand = r_nk.unsqueeze(1)           # (num_samples, 1, N, K)
        M_nk_expand = M_nk.unsqueeze(1)           # (num_samples, 1, N, K)
        
        # H_{m,n,k} = M_{n,k} * (c / (4 * pi * f_m * r_{n,k})) * exp(-j * 2 * pi * f_m * r_{n,k} / c)
        amplitude = M_nk_expand * (self.c / (4 * np.pi * f_m_expand * r_nk_expand))
        phase = -2 * np.pi * f_m_expand * r_nk_expand / self.c
        self.H_all = amplitude * torch.exp(1j * phase)  # (num_samples, F, N, K)
        
        # 6. QPSK symbols x (independent for each subcarrier)
        real_part = (torch.randint(0, 2, (self.num_samples, self.F, self.K, 1)) * 2 - 1).float() / np.sqrt(2)
        imag_part = (torch.randint(0, 2, (self.num_samples, self.F, self.K, 1)) * 2 - 1).float() / np.sqrt(2)
        self.x_all = real_part + 1j * imag_part  # (num_samples, F, K, 1)
        
        # 7. Received signal y = Hx + n
        # Hx: (num_samples, F, N, 1)
        Hx = torch.matmul(self.H_all, self.x_all)
        
        # Sample SNR uniformly from snr_range (in dB) independently for each sample and subcarrier
        snr_db = self.snr_range[0] + (self.snr_range[1] - self.snr_range[0]) * torch.rand(self.num_samples, self.F, 1, 1)
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate signal power and required noise power
        signal_power = torch.sum(torch.abs(Hx)**2, dim=2, keepdim=True) / self.N  # (num_samples, F, 1, 1)
        noise_power = signal_power / snr_linear
        
        # Complex Gaussian white noise
        noise_real = torch.randn((self.num_samples, self.F, self.N, 1)) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn((self.num_samples, self.F, self.N, 1)) * torch.sqrt(noise_power / 2)
        self.noise_all = noise_real + 1j * noise_imag
        
        # Final noisy received signal
        self.y_all = Hx + self.noise_all

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.H_all[idx], self.x_all[idx], self.y_all[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Near Field Wideband Dataset Generator")
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate (default: 10)')
    parser.add_argument('--N', type=int, default=256, help='Number of antennas (default: 256)')
    parser.add_argument('--K', type=int, default=8, help='Number of users (default: 8)')
    parser.add_argument('--F', type=int, default=16, help='Number of subcarriers (default: 16)')
    parser.add_argument('--f_c', type=float, default=0.1e12, help='Carrier frequency in Hz (default: 0.1e12)')
    parser.add_argument('--B', type=float, default=10e9, help='Total Bandwidth in Hz (default: 10e9)')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB (default: -5)')
    parser.add_argument('--snr_max', type=float, default=15, help='Maximum SNR in dB (default: 15)')
    args = parser.parse_args()

    print(f"=== Testing NearFieldWidebandDataset with {args.num_samples} samples (N={args.N}, K={args.K}, F={args.F}) ===")
    dataset = NearFieldWidebandDataset(
        num_samples=args.num_samples, 
        N=args.N, K=args.K, F=args.F, 
        f_c=args.f_c, B_bw=args.B, 
        snr_range=(args.snr_min, args.snr_max)
    )
    
    H, x, y = dataset[0]
    print(f"\n[1] Tensor dimensions (single sample):")
    print(f"    H shape: {H.shape}")
    print(f"    x shape: {x.shape}")
    print(f"    y shape: {y.shape}")
    
    print(f"\n[2] Tensor dimensions (all samples in dataset):")
    print(f"    H_all shape: {dataset.H_all.shape}")
    print(f"    x_all shape: {dataset.x_all.shape}")
    print(f"    y_all shape: {dataset.y_all.shape}")
    
    # Sparsity of H
    non_zero_elements = torch.sum(dataset.H_all != 0).item()
    total_elements = dataset.H_all.numel()
    sparsity = non_zero_elements / total_elements
    print(f"\n[3] Sparsity of H (non-zero ratio) over all samples:")
    print(f"    {sparsity*100:.2f}% (Expected ~60%)")
    
    # Average energy magnitude of H
    avg_energy = torch.mean(torch.abs(dataset.H_all)**2)
    print(f"\n[4] Average energy magnitude of H over all samples:")
    print(f"    {avg_energy.item():.2e}")
    print("\n==================================================================")
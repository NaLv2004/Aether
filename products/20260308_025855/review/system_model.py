import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="6G Cell-Free MIMO System Modeler and Dataset Generator")
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km (default 1km x 1km)")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    parser.add_argument("--p_tx", type=float, default=23.0, help="Transmit power of users in dBm")
    parser.add_argument("--epochs", type=int, default=100, help="Number of channel realizations (data samples) to generate")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of dataset used for training")
    parser.add_argument("--quant_bits", type=str, default="-1,0,2,4,8", help="Comma-separated list of bits for quantization tests (-1 means float32)")
    return parser.parse_args()

class CellFreeSystem:
    def __init__(self, L, N, K, area_size, bandwidth, noise_psd_dbm, p_tx_dbm):
        self.L = L
        self.N = N
        self.K = K
        self.area_size = area_size
        self.bandwidth = bandwidth
        
        # Calculate Noise Power in linear scale (Watts)
        # Noise_dBm = PSD_dBm/Hz + 10*log10(BW)
        self.noise_power_dbm = noise_psd_dbm + 10 * np.log10(bandwidth)
        self.noise_w = 10 ** ((self.noise_power_dbm - 30) / 10)
        
        # Transmit Power in linear scale (Watts)
        self.p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
        
        # Locations and Large-scale Fading placeholders
        self.ap_pos = None
        self.user_pos = None
        self.beta = None # Large scale fading coefficients (linear)
        self.beta_w = None # beta scaled by P_tx
        
    def generate_scenario(self):
        """
        Generate random positions for APs and Users, and calculate large-scale fading.
        Using 3GPP Dense Urban model: PL(dB) = 128.1 + 37.6 * log10(d[km])
        Shadowing standard deviation = 8 dB
        """
        # 1. Generate Locations
        self.ap_pos = np.random.uniform(0, self.area_size, size=(self.L, 2))
        self.user_pos = np.random.uniform(0, self.area_size, size=(self.K, 2))
        
        # 2. Compute Distances and Large-Scale Fading
        dist = np.zeros((self.L, self.K))
        pl_db = np.zeros((self.L, self.K))
        shadowing = np.random.normal(0, 8, size=(self.L, self.K))
        
        for l in range(self.L):
            for k in range(self.K):
                # Calculate distance in km, minimum distance set to 10m (0.01km) to avoid infinity
                d = np.linalg.norm(self.ap_pos[l] - self.user_pos[k])
                d = max(d, 0.01)
                dist[l, k] = d
                pl_db[l, k] = 128.1 + 37.6 * np.log10(d)
                
        # Total Path Loss in dB
        total_loss_db = pl_db + shadowing
        self.beta = 10 ** (-total_loss_db / 10)
        self.beta_w = self.p_tx_w * self.beta
        
        # Print Physical Insights
        snr_db = 10 * np.log10(self.beta_w / self.noise_w)
        print("=== Scenario Generated ===")
        print(f"Total APs: {self.L}, Total Users: {self.K}")
        print(f"Noise Power: {self.noise_power_dbm:.2f} dBm ({self.noise_w:.2e} W)")
        print(f"Average Receive SNR range: {np.min(snr_db):.2f} dB to {np.max(snr_db):.2f} dB")
        print("==========================\n")

    def generate_channel(self):
        """
        Generate channel realization including large-scale and small-scale fading.
        H_l shape: N x K
        """
        H = np.zeros((self.L, self.N, self.K), dtype=complex)
        for l in range(self.L):
            for k in range(self.K):
                # Rayleigh small-scale fading (CN(0, 1))
                h_small = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)
                # Combine with large scale fading and Tx power
                H[l, :, k] = np.sqrt(self.beta_w[l, k]) * h_small
        return H

    def generate_transmit_signals(self):
        """
        Generate QPSK modulated symbols for K users.
        """
        bits = np.random.randint(0, 4, size=self.K)
        # QPSK mapping: (+/- 1 +/- 1j) / sqrt(2)
        mapping = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
        s = np.array([mapping[b] for b in bits]) / np.sqrt(2)
        return s

    def simulate_reception(self, H, s):
        """
        AP side received signals: y_l = H_l * s + z_l
        """
        y = np.zeros((self.L, self.N), dtype=complex)
        for l in range(self.L):
            # AWGN noise
            z_l = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2) * np.sqrt(self.noise_w)
            y[l] = H[l] @ s + z_l
        return y

    def local_lmmse_detection(self, H_l, y_l):
        """
        Local LMMSE detection: s_hat_l = H_l^H * (H_l * H_l^H + sigma^2 * I)^(-1) * y_l
        """
        # H_l: N x K matrix
        # R_y: N x N matrix
        R_y = H_l @ H_l.conj().T + self.noise_w * np.eye(self.N)
        W_l = H_l.conj().T @ np.linalg.inv(R_y)
        s_hat_l = W_l @ y_l
        return s_hat_l

def quantize_signal(s_hat_l, b_array):
    """
    Uniform quantizer. Dynamically calculates step based on signal dynamic range.
    Quantizes real and imaginary parts separately.
    b_array: list or array of bit widths for each element in s_hat_l
    """
    q_out = np.zeros_like(s_hat_l)
    if len(s_hat_l) == 0:
        return q_out
    
    # Estimate dynamic range from the whole block 
    max_r = max(np.max(np.abs(np.real(s_hat_l))), 1e-12)
    max_i = max(np.max(np.abs(np.imag(s_hat_l))), 1e-12)
    
    for k in range(len(s_hat_l)):
        b = b_array[k]
        if b <= 0:
            q_out[k] = 0j
            continue
        
        levels = 2**b
        
        # Quantize Real Part
        step_r = 2 * max_r / (levels - 1)
        idx_r = np.round((np.real(s_hat_l[k]) + max_r) / step_r)
        idx_r = np.clip(idx_r, 0, levels - 1)
        q_r = idx_r * step_r - max_r
        
        # Quantize Imaginary Part
        step_i = 2 * max_i / (levels - 1)
        idx_i = np.round((np.imag(s_hat_l[k]) + max_i) / step_i)
        idx_i = np.clip(idx_i, 0, levels - 1)
        q_i = idx_i * step_i - max_i
        
        q_out[k] = q_r + 1j * q_i
        
    return q_out

def main():
    args = parse_args()
    np.random.seed(42) # For reproducibility
    
    # Initialize system
    sys_model = CellFreeSystem(
        L=args.L, N=args.N, K=args.K, 
        area_size=args.area_size, bandwidth=args.bandwidth, 
        noise_psd_dbm=args.noise_psd, p_tx_dbm=args.p_tx
    )
    
    # 1. Generate large scale scenario
    sys_model.generate_scenario()
    
    quant_bits_list = [int(x) for x in args.quant_bits.split(',')]
    mse_results = {b: 0.0 for b in quant_bits_list}
    
    # Data collecting structures for AI Dataset Generation
    dataset_H = []
    dataset_y = []
    dataset_s = []

    print("Generating data samples and evaluating quantization impacts...")
    # 2. Loop over specified number of epochs (channel realizations)
    for epoch in range(args.epochs):
        H = sys_model.generate_channel()
        s = sys_model.generate_transmit_signals()
        y = sys_model.simulate_reception(H, s)
        
        # Append for dataset
        dataset_H.append(H)
        dataset_y.append(y)
        dataset_s.append(s)
        
        # Test detection and quantization for this epoch
        s_hat = np.zeros((sys_model.L, sys_model.K), dtype=complex)
        for l in range(sys_model.L):
            s_hat[l] = sys_model.local_lmmse_detection(H[l], y[l])
        
        for b in quant_bits_list:
            epoch_mse = 0.0
            for l in range(sys_model.L):
                if b == -1: # No quantization (float32)
                    s_q = s_hat[l]
                else:
                    s_q = quantize_signal(s_hat[l], [b] * sys_model.K)
                
                # Compute MSE: E[||s - s_hat_l||^2]
                mse = np.linalg.norm(s - s_q)**2
                epoch_mse += mse
            
            # Average over APs
            mse_results[b] += (epoch_mse / sys_model.L)
            
        if (epoch + 1) % max(1, (args.epochs // 5)) == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] processed.")
    
    # 3. Print Results
    print("\n=== Evaluation Results ===")
    for b in quant_bits_list:
        avg_mse = mse_results[b] / args.epochs
        if b == -1:
            print(f"Quantization: Float32 (No Quantization) | Average MSE: {avg_mse:.4f} | AP Avg Payload: {sys_model.K * 32} bits")
        else:
            payload = sys_model.K * b
            print(f"Quantization: {b}-bit                | Average MSE: {avg_mse:.4f} | AP Avg Payload: {payload} bits")
    print("==========================\n")
    
    # 4. Dataset splitting and saving for AI modeling
    H_data = np.array(dataset_H)
    y_data = np.array(dataset_y)
    s_data = np.array(dataset_s)
    
    num_train = int(args.epochs * args.train_ratio)
    np.savez("cell_free_mimo_dataset.npz", 
             train_H=H_data[:num_train], train_y=y_data[:num_train], train_s=s_data[:num_train],
             test_H=H_data[num_train:], test_y=y_data[num_train:], test_s=s_data[num_train:])
    
    print(f"Dataset successfully saved to 'cell_free_mimo_dataset.npz'.")
    print(f"Explicit Train/Test Split => Train Size: {num_train}, Test Size: {args.epochs - num_train}")

if __name__ == "__main__":
    main()
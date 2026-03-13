import numpy as np
import torch
import torch.nn as nn
import argparse
import time

from gnn_detector import generate_data_batch, get_tensors_from_batch, compute_ber
from system_model import CellFreeSystem
from joint_qat import JointQATModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of Generalization and Ablation for QAT-GNN")
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K_train", type=int, default=8, help="Number of users in training")
    parser.add_argument("--K_test", type=int, default=12, help="Number of users for generalization test")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--model_path", type=str, default="qat_gnn_model.pth", help="Path to trained QAT-GNN model")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of samples for testing")
    return parser.parse_args()

class ZeroAttn(nn.Module):
    """
    Returns zeros with the same shape as the input hidden states, 
    effectively making the softmax weights equal (1/L).
    """
    def __init__(self):
        super(ZeroAttn, self).__init__()
    def forward(self, h):
        # h shape: (B, L, K, hidden_dim)
        # return shape: (B, L, K, 1)
        return torch.zeros(h.size(0), h.size(1), h.size(2), 1).to(h.device)

class NoBitAwareGNNWrapper(nn.Module):
    """
    Wraps the GNN forward method to ensure the 4th channel (bit info) is zeroed out.
    """
    def __init__(self, original_gnn):
        super(NoBitAwareGNNWrapper, self).__init__()
        self.original_gnn = original_gnn
        
    def forward(self, x):
        # x shape: (B, L, K, 4)
        x_mod = x.clone()
        x_mod[..., 3] = 0.0 # Force bit information to 0
        return self.original_gnn(x_mod)

def evaluate_model(model, dataset, test_p_tx, ablation_type=None):
    model.eval()
    results = {}
    
    # Save original state
    original_gnn_forward = model.gnn.forward
    original_attn_net = model.gnn.attn_net
    
    if ablation_type == 'no_bit_aware':
        # Replace GNN forward with a version that zeros out bit info
        model.gnn = NoBitAwareGNNWrapper(model.gnn)
    elif ablation_type == 'no_attention':
        # Replace attn_net with Identity/Zero Attn
        model.gnn.attn_net = ZeroAttn()

    with torch.no_grad():
        for p in test_p_tx:
            X_test, Y_test, s_hat_np, s_np, local_snr_np = dataset[p]
            
            if ablation_type == 'fixed_2bit':
                v = X_test[..., :2]
                snr = X_test[..., 2:3]
                # Force 2-bit quantization using the trained q2 quantizer
                v_q = model.quantizer.q2(v)
                expected_bits = torch.ones(v_q.shape[:-1] + (1,), device=v_q.device) * 2.0
                x_q = torch.cat([v_q, snr, expected_bits], dim=-1)
                out = model.gnn(x_q)
            elif ablation_type == 'fixed_4bit':
                v = X_test[..., :2]
                snr = X_test[..., 2:3]
                # Force 4-bit quantization using the trained q4 quantizer
                v_q = model.quantizer.q4(v)
                expected_bits = torch.ones(v_q.shape[:-1] + (1,), device=v_q.device) * 4.0
                x_q = torch.cat([v_q, snr, expected_bits], dim=-1)
                out = model.gnn(x_q)
            else:
                out, expected_bits = model(X_test, tau=0.1)
            
            out_np = out.cpu().numpy()
            s_pred_qat = out_np[..., 0] + 1j * out_np[..., 1]
            ber_qat = compute_ber(s_np, s_pred_qat)
            
            s_hat_avg = s_hat_np.mean(axis=1) 
            ber_dist_full = compute_ber(s_np, s_hat_avg)
            
            avg_bits = expected_bits.mean().item()
            results[p] = {'ber': ber_qat, 'avg_bits': avg_bits, 'ber_baseline': ber_dist_full}
            
    # Restore original state
    if ablation_type == 'no_bit_aware':
        model.gnn = model.gnn.original_gnn
    elif ablation_type == 'no_attention':
        model.gnn.attn_net = original_attn_net
        
    return results

def calculate_scenario_stats(sys_model, p_tx_dbm):
    """
    Calculate average path loss (beta) and average receive power for the scenario.
    """
    # Average large-scale fading coefficient beta
    avg_beta = np.mean(sys_model.beta)
    avg_path_loss_db = -10 * np.log10(avg_beta + 1e-15)
    
    # Average receive power (linear)
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    avg_rx_power_w = p_tx_w * avg_beta
    avg_rx_power_dbm = 10 * np.log10(avg_rx_power_w + 1e-15) + 30
    
    return avg_path_loss_db, avg_rx_power_dbm

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load Model
    model_k8 = JointQATModel(L=args.L, K=args.K_train, hidden_dim=args.hidden_dim).to(device)
    try:
        model_k8.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded trained model from {args.model_path}")
    except Exception as e:
        print(f"Warning: Could not load {args.model_path}, using untrained weights. Error: {e}")

    # =====================================================================
    # 1. Generalization Test (K=8 vs K=12) with Stats
    # =====================================================================
    print("\n" + "="*80)
    print(f"1. Generalization Test (Train K={args.K_train} -> Test K={args.K_test})")
    print("="*80)
    
    test_p_tx_gen = [-10, 0, 10, 20]
    
    # Generate K=8 dataset
    np.random.seed(42)
    torch.manual_seed(42)
    sys_model_k8 = CellFreeSystem(args.L, args.N, args.K_train, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model_k8.generate_scenario()
    dataset_k8 = {}
    print(f"Generating data for K={args.K_train} scenario...")
    for p in test_p_tx_gen:
        s_hat_np, s_np, local_snr_np = generate_data_batch(sys_model_k8, args.test_samples, p_tx_dbm=p)
        X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
        dataset_k8[p] = (X_test, Y_test, s_hat_np, s_np, local_snr_np)
        
    # Generate K=12 dataset
    np.random.seed(42)
    torch.manual_seed(42)
    sys_model_k12 = CellFreeSystem(args.L, args.N, args.K_test, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model_k12.generate_scenario()
    dataset_k12 = {}
    print(f"Generating data for K={args.K_test} scenario...")
    for p in test_p_tx_gen:
        s_hat_np, s_np, local_snr_np = generate_data_batch(sys_model_k12, args.test_samples, p_tx_dbm=p)
        X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
        dataset_k12[p] = (X_test, Y_test, s_hat_np, s_np, local_snr_np)

    res_k8 = evaluate_model(model_k8, dataset_k8, test_p_tx_gen)
    
    model_k12 = JointQATModel(L=args.L, K=args.K_test, hidden_dim=args.hidden_dim).to(device)
    model_k12.load_state_dict(model_k8.state_dict())
    res_k12 = evaluate_model(model_k12, dataset_k12, test_p_tx_gen)

    # Detailed Output with Scenario Stats
    print(f"\n{'P_tx':<6} | {'K8 BER':<8} | {'K12 BER':<9} | {'Drop%':<7} | {'K8 Bits':<8} | {'K12 Bits':<8} | {'K8 PL(dB)':<10} | {'K12 PL(dB)':<10}")
    print("-" * 105)
    for p in test_p_tx_gen:
        ber8 = res_k8[p]['ber']
        ber12 = res_k12[p]['ber']
        drop = ((ber12 - ber8) / ber8 * 100) if ber8 > 0 else 0
        pl8, rx8 = calculate_scenario_stats(sys_model_k8, p)
        pl12, rx12 = calculate_scenario_stats(sys_model_k12, p)
        print(f"{p:<6} | {ber8:<8.4%} | {ber12:<9.4%} | {drop:<7.2f} | {res_k8[p]['avg_bits']:<8.2f} | {res_k12[p]['avg_bits']:<8.2f} | {pl8:<10.1f} | {pl12:<10.1f}")

    # =====================================================================
    # 2. Bit Allocation Distribution Analysis (K=8)
    # =====================================================================
    print("\n" + "="*80)
    print(f"2. Bit Allocation Distribution Analysis (K={args.K_train})")
    print("="*80)
    
    model_k8.eval()
    print(f"{'P_tx (dBm)':<12} | {'0-bit (%)':<12} | {'2-bit (%)':<12} | {'4-bit (%)':<12} | {'Avg Bits':<10}")
    print("-" * 70)
    with torch.no_grad():
        for p in test_p_tx_gen:
            X_test, _, _, _, _ = dataset_k8[p]
            v = X_test[..., :2]
            snr = X_test[..., 2:3]
            _, expected_bits, w = model_k8.quantizer(v, snr, tau=0.1)
            
            w_mean = w.mean(dim=(0, 1, 2)).cpu().numpy() * 100
            avg_bits = expected_bits.mean().item()
            
            print(f"{p:<12} | {w_mean[0]:<12.2f} | {w_mean[1]:<12.2f} | {w_mean[2]:<12.2f} | {avg_bits:<10.2f}")

    # =====================================================================
    # 3. Ablation Study & Bitwidth Comparison (K=8)
    # =====================================================================
    print("\n" + "="*80)
    print(f"3. Ablation Study & Bitwidth Comparison (K={args.K_train})")
    print("="*80)
    
    res_no_bit = evaluate_model(model_k8, dataset_k8, test_p_tx_gen, ablation_type='no_bit_aware')
    res_no_attn = evaluate_model(model_k8, dataset_k8, test_p_tx_gen, ablation_type='no_attention')
    res_fixed_2 = evaluate_model(model_k8, dataset_k8, test_p_tx_gen, ablation_type='fixed_2bit')
    res_fixed_4 = evaluate_model(model_k8, dataset_k8, test_p_tx_gen, ablation_type='fixed_4bit')
    
    print(f"{'P_tx (dBm)':<10} | {'Full Model':<12} | {'Fixed 2-bit':<12} | {'Fixed 4-bit':<12} | {'No Bit-aware':<14} | {'No Attention':<14}")
    print("-" * 90)
    for p in test_p_tx_gen:
        ber_full = res_k8[p]['ber']
        ber_fixed_2 = res_fixed_2[p]['ber']
        ber_fixed_4 = res_fixed_4[p]['ber']
        ber_no_bit = res_no_bit[p]['ber']
        ber_no_attn = res_no_attn[p]['ber']
        print(f"{p:<10} | {ber_full:<12.4%} | {ber_fixed_2:<12.4%} | {ber_fixed_4:<12.4%} | {ber_no_bit:<14.4%} | {ber_no_attn:<14.4%}")

if __name__ == "__main__":
    main()
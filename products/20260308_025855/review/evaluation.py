import numpy as np
import torch
import torch.nn as nn
import argparse
import time

from gnn_detector import generate_data_batch, get_tensors_from_batch, compute_ber
from system_model import CellFreeSystem
from joint_qat import JointQATModel
from lsq_quantizer import LSQQuantizer

def complexity_analysis(N, K):
    print("\n" + "="*50)
    print("=== Complexity Analysis ===")
    print("="*50)
    # LMMSE complexity roughly: H is N x K.
    # H H^H: N * N * K MACs -> 2 N^2 K FLOPs
    # Inverse of N x N: O(N^3) -> ~ 2/3 N^3 FLOPs
    # H^H ( ... ) y: ...
    # Rough estimate for LMMSE per AP
    flops_lmmse_real = 2 * (N**2) * K + (2/3) * (N**3) + 2 * N * K + 2 * N
    # Multiply by 4 because complex operations take ~4x real FLOPs
    flops_lmmse = flops_lmmse_real * 4
    
    # Policy Net complexity: MLP 3 -> 32 -> 3
    # Layer 1: 3 * 32 MACs = 96 MACs -> 192 FLOPs
    # Layer 2: 32 * 3 MACs = 96 MACs -> 192 FLOPs
    # Total roughly 384 FLOPs
    flops_policy = (3 * 32 + 32 * 3) * 2
    
    print(f"Assumption: N (Antennas per AP) = {N}, K (Users) = {K}")
    print(f"1. Local LMMSE FLOPs per AP (Approx): {flops_lmmse:.0f} FLOPs")
    print(f"2. BitwidthPolicyNet FLOPs per AP   : {flops_policy:.0f} FLOPs")
    print(f"-> The policy network requires only about {flops_policy/flops_lmmse:.2%} of the computation of Local LMMSE.")
    print("="*50 + "\n")

def simulate_inference(model, x, snr, drop_rate=0.0):
    """
    Custom inference simulating actual deployment:
    1. Hard selection using argmax.
    2. Optional AP dropout simulation.
    """
    B, L, K, _ = x.shape
    v = x[..., :2]
    
    # 1. Policy Network Inference
    policy_input = torch.cat([v, snr], dim=-1)
    logits = model.quantizer.policy_net(policy_input) # (B, L, K, 3)
    
    # 2. Hard Selection
    choices = torch.argmax(logits, dim=-1) # (B, L, K)
    
    # 3. Simulate AP Dropout
    if drop_rate > 0.0:
        # Create a dropout mask for APs (B, L)
        # 1 means active, 0 means dropped
        drop_mask = (torch.rand(B, L, device=x.device) > drop_rate).float()
        drop_mask = drop_mask.unsqueeze(-1) # (B, L, 1)
        # Force choice to 0 (0-bit) for dropped APs
        # Removed .squeeze(-1) to correctly broadcast from (B, L, 1) -> (B, L, K)
        choices = choices * drop_mask.long()
        
    # Create one-hot vectors
    w = torch.nn.functional.one_hot(choices, num_classes=3).float()
    
    # 4. Quantization
    v_q2 = model.quantizer.q2(v)
    v_q4 = model.quantizer.q4(v)
    
    v_q = w[..., 0:1] * 0.0 + w[..., 1:2] * v_q2 + w[..., 2:3] * v_q4
    expected_bits = w[..., 0] * 0 + w[..., 1] * 2 + w[..., 2] * 4
    expected_bits = expected_bits.unsqueeze(-1).float()
    
    # 5. GNN Forward
    x_q = torch.cat([v_q, snr, expected_bits], dim=-1)
    out = model.gnn(x_q)
    
    return out, expected_bits, w

def simulate_dist_q2(v, snr, gnn_model):
    """
    Baseline: Force all to 2-bit
    """
    q2 = LSQQuantizer(num_bits=2).to(v.device)
    v_q2 = q2(v)
    expected_bits = torch.ones_like(snr) * 2.0
    x_q = torch.cat([v_q2, snr, expected_bits], dim=-1)
    out = gnn_model(x_q)
    return out

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Robustness of QAT GNN Model")
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    parser.add_argument("--model_path", type=str, default="qat_gnn_model.pth", help="Path to trained QAT GNN")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of samples per power point")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    complexity_analysis(args.N, args.K)
    
    np.random.seed(42)
    torch.manual_seed(42)
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()
    
    model = JointQATModel(L=args.L, K=args.K, hidden_dim=64).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded '{args.model_path}'")
    except Exception as e:
        print(f"Failed to load model: {e}. Using random weights for evaluation.")
    
    model.eval()
    
    test_p_tx = [-10, -5, 0, 5, 10, 15, 20]
    drop_rates = [0.0, 0.25, 0.50]
    
    print("\n" + "="*100)
    print(f"{'p_tx(dBm)':<10} | {'Dist-Full':<12} | {'Dist-Q2':<12} | {'QAT (0% Drop)':<15} | {'QAT (25% Drop)':<15} | {'QAT (50% Drop)':<15}")
    print("-" * 100)
    
    with torch.no_grad():
        for p in test_p_tx:
            s_hat_np, s_np, local_snr_np = generate_data_batch(sys_model, args.test_samples, p_tx_dbm=p)
            X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
            
            snr = X_test[..., 2:3]
            
            # Baseline 1: Dist-Full
            s_hat_avg = s_hat_np.mean(axis=1) 
            ber_dist_full = compute_ber(s_np, s_hat_avg)
            
            # Baseline 2: Dist-Q2
            out_q2 = simulate_dist_q2(X_test[..., :2], snr, model.gnn)
            s_pred_q2 = out_q2.cpu().numpy()[..., 0] + 1j * out_q2.cpu().numpy()[..., 1]
            ber_dist_q2 = compute_ber(s_np, s_pred_q2)
            
            # QAT Models with different drop rates
            bers_qat = []
            for dr in drop_rates:
                out_qat, _, _ = simulate_inference(model, X_test, snr, drop_rate=dr)
                s_pred_qat = out_qat.cpu().numpy()[..., 0] + 1j * out_qat.cpu().numpy()[..., 1]
                bers_qat.append(compute_ber(s_np, s_pred_qat))
                
            print(f"{p:<10} | {ber_dist_full:<12.4%} | {ber_dist_q2:<12.4%} | {bers_qat[0]:<15.4%} | {bers_qat[1]:<15.4%} | {bers_qat[2]:<15.4%}")
            
    print("="*100 + "\n")
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
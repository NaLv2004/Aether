import numpy as np
import torch
import argparse
import time

from system_model import CellFreeSystem
from gnn_detector import FPGNN, generate_data_batch, get_tensors_from_batch, compute_ber


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Full-Precision GNN (FPGNN) Baseline for Cell-Free MIMO")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Evaluation parameters
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples per power point")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for FPGNN feature lifting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_path", type=str, default="fp_gnn_model.pth", help="Path to pre-trained FPGNN weights")
    parser.add_argument("--power_points", type=str, default="-10,-5,0,5,10,15,20",
                        help="Comma-separated list of transmit power points in dBm")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse power points
    power_points = [int(x.strip()) for x in args.power_points.split(",")]

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FPGNN Baseline Evaluation for Cell-Free MIMO")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"Random Seed:     {args.seed}")
    print(f"System Config:   L={args.L}, N={args.N}, K={args.K}")
    print(f"Area Size:       {args.area_size} km")
    print(f"Bandwidth:       {args.bandwidth / 1e6:.1f} MHz")
    print(f"Noise PSD:       {args.noise_psd} dBm/Hz")
    print(f"Test Samples:    {args.test_samples} per power point")
    print(f"Hidden Dim:      {args.hidden_dim}")
    print(f"Model Path:      {args.model_path}")
    print(f"Power Points:    {power_points} dBm")
    print("=" * 70)

    # 1. Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 2. Initialize system model and generate large-scale scenario
    print("\n[Step 1] Initializing Cell-Free MIMO system model...")
    sys_model = CellFreeSystem(
        args.L, args.N, args.K,
        args.area_size, args.bandwidth, args.noise_psd,
        0.0  # p_tx_dbm placeholder (overridden per power point in generate_data_batch)
    )
    sys_model.generate_scenario()
    print(f"  AP positions shape:   {sys_model.ap_pos.shape}")
    print(f"  User positions shape: {sys_model.user_pos.shape}")
    print(f"  Beta shape:           {sys_model.beta.shape}")
    print(f"  Noise power (W):      {sys_model.noise_w:.4e}")

    # 3. Load pre-trained FPGNN model
    print(f"\n[Step 2] Loading pre-trained FPGNN model from '{args.model_path}'...")
    model = FPGNN(L=args.L, K=args.K, hidden_dim=args.hidden_dim).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  Model loaded successfully!")

    # 4. Generate test data and evaluate at each power point
    print(f"\n[Step 3] Generating test data ({args.test_samples} samples per power point)...")
    results = []

    for p_tx_dbm in power_points:
        print(f"\n  --- Evaluating at P_tx = {p_tx_dbm} dBm ---")
        start_t = time.time()

        # Generate test batch
        s_hat_np, s_np, local_snr_np = generate_data_batch(sys_model, args.test_samples, p_tx_dbm)
        print(f"    s_hat shape: {s_hat_np.shape}  (batch, L, K)")
        print(f"    s_true shape: {s_np.shape}  (batch, K)")
        print(f"    local_snr shape: {local_snr_np.shape}  (batch, L, K)")

        # --- Baseline: Dist-Full (mean pooling across APs) ---
        s_hat_avg = s_hat_np.mean(axis=1)  # shape: (batch, K)
        ber_dist_full = compute_ber(s_np, s_hat_avg)
        print(f"    Dist-Full (mean pooling) BER: {ber_dist_full:.6f}")

        # --- FPGNN prediction ---
        X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
        print(f"    X_test tensor shape: {X_test.shape}")
        print(f"    Y_test tensor shape: {Y_test.shape}")

        with torch.no_grad():
            out = model(X_test)  # shape: (batch, K, 2)

        out_np = out.cpu().numpy()
        s_pred_fpgnn = out_np[..., 0] + 1j * out_np[..., 1]
        ber_fpgnn = compute_ber(s_np, s_pred_fpgnn)
        print(f"    FPGNN BER:                    {ber_fpgnn:.6f}")

        # Compute improvement
        if ber_dist_full > 0:
            improvement = (ber_dist_full - ber_fpgnn) / ber_dist_full * 100.0
        else:
            improvement = 0.0
        print(f"    Improvement:                  {improvement:.2f}%")

        elapsed = time.time() - start_t
        print(f"    Evaluation time:              {elapsed:.3f}s")

        results.append({
            "p_tx_dbm": p_tx_dbm,
            "ber_dist_full": ber_dist_full,
            "ber_fpgnn": ber_fpgnn,
            "improvement": improvement
        })

    # 5. Print formatted summary table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'P_tx(dBm)':>10} | {'Dist-Full BER':>15} | {'FPGNN BER':>15} | {'Improvement(%)':>15}"
    print(header)
    print("-" * 70)

    for r in results:
        p_str = f"{r['p_tx_dbm']:>10d}"
        df_str = f"{r['ber_dist_full']:>15.6f}"
        fp_str = f"{r['ber_fpgnn']:>15.6f}"
        im_str = f"{r['improvement']:>14.2f}%"
        print(f"{p_str} | {df_str} | {fp_str} | {im_str}")

    print("-" * 70)

    # 6. Print additional statistics
    avg_dist_full = np.mean([r["ber_dist_full"] for r in results])
    avg_fpgnn = np.mean([r["ber_fpgnn"] for r in results])
    avg_improvement = np.mean([r["improvement"] for r in results])

    print(f"\n{'Average':>10} | {avg_dist_full:>15.6f} | {avg_fpgnn:>15.6f} | {avg_improvement:>14.2f}%")
    print("=" * 70)

    # 7. Print raw data arrays for easy copy-paste into papers/plots
    print("\n[Raw Data for Plotting]")
    p_list = [str(r["p_tx_dbm"]) for r in results]
    df_list = [f"{r['ber_dist_full']:.6f}" for r in results]
    fp_list = [f"{r['ber_fpgnn']:.6f}" for r in results]
    im_list = [f"{r['improvement']:.2f}" for r in results]

    print("Power points (dBm): [" + ", ".join(p_list) + "]")
    print("Dist-Full BER:      [" + ", ".join(df_list) + "]")
    print("FPGNN BER:          [" + ", ".join(fp_list) + "]")
    print("Improvement (%):    [" + ", ".join(im_list) + "]")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
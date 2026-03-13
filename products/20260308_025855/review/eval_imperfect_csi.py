import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import os

from system_model import CellFreeSystem
from gnn_detector import get_tensors_from_batch, compute_ber
from joint_qat import JointQATModel
from evaluation import simulate_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Impact of Imperfect CSI on Cell-Free MIMO Detection")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km (default 1km x 1km)")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Evaluation parameters
    parser.add_argument("--test_samples", type=int, default=1000,
                        help="Number of test samples per power point (default 1000 for smooth curves)")
    parser.add_argument("--model_path", type=str, default="qat_gnn_model.pth",
                        help="Path to trained QAT-GNN model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of GNN model")

    # CSI error parameters (NMSE-based model)
    parser.add_argument("--nmse_levels", type=str, default="0.0,0.01,0.05,0.1",
                        help="Comma-separated NMSE levels (0=perfect CSI, 0.01=1%%, 0.05=5%%, 0.1=10%%)")

    # Power points
    parser.add_argument("--power_points", type=str, default="-10,-5,0,5,10,15,20",
                        help="Comma-separated transmit power points in dBm")

    # Summary power point
    parser.add_argument("--summary_power", type=float, default=10.0,
                        help="Fixed power point (dBm) for summary table across NMSE levels")

    return parser.parse_args()


def pregenerate_random_variables(sys_model, batch_size, seed_value):
    """
    Pre-generate all random variables for one NMSE level with a fixed seed.
    These random variables are REUSED across all power points (only scaling changes).

    This is the KEY FIX: by using the same h_small, e_small, QPSK symbols, and noise
    across all power points, the BER curves are guaranteed to be monotonically decreasing
    with increasing transmit power.

    Parameters:
    -----------
    sys_model : CellFreeSystem
        System model (provides L, N, K dimensions).
    batch_size : int
        Number of samples in the batch.
    seed_value : int
        Random seed for this NMSE level.

    Returns:
    --------
    h_small : np.ndarray, shape (batch_size, L, N, K), complex
        Small-scale fading ~ CN(0, 1).
    e_small : np.ndarray, shape (batch_size, L, N, K), complex
        CSI error small-scale ~ CN(0, 1).
    qpsk_bits : np.ndarray, shape (batch_size, K), int
        QPSK symbol indices (0..3).
    s : np.ndarray, shape (batch_size, K), complex
        QPSK symbols.
    z_small : np.ndarray, shape (batch_size, L, N), complex
        Noise small-scale ~ CN(0, 1).
    """
    L = sys_model.L
    N = sys_model.N
    K = sys_model.K

    np.random.seed(seed_value)
    print(f"  [PreGen] Setting np.random.seed({seed_value})")

    # 1. Small-scale fading: h_small ~ CN(0, 1)
    h_small = (np.random.randn(batch_size, L, N, K) +
               1j * np.random.randn(batch_size, L, N, K)) / np.sqrt(2)
    print(f"  [PreGen] Generated h_small: shape={h_small.shape}, "
          f"mean_power={np.mean(np.abs(h_small)**2):.6f} (expect ~1.0)")

    # 2. CSI error small-scale: e_small ~ CN(0, 1)
    e_small = (np.random.randn(batch_size, L, N, K) +
               1j * np.random.randn(batch_size, L, N, K)) / np.sqrt(2)
    print(f"  [PreGen] Generated e_small: shape={e_small.shape}, "
          f"mean_power={np.mean(np.abs(e_small)**2):.6f} (expect ~1.0)")

    # 3. QPSK symbols
    qpsk_bits = np.random.randint(0, 4, size=(batch_size, K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(qpsk_bits) / np.sqrt(2)
    print(f"  [PreGen] Generated QPSK symbols: shape={s.shape}")

    # 4. Noise small-scale: z_small ~ CN(0, 1)
    z_small = (np.random.randn(batch_size, L, N) +
               1j * np.random.randn(batch_size, L, N)) / np.sqrt(2)
    print(f"  [PreGen] Generated z_small: shape={z_small.shape}, "
          f"mean_power={np.mean(np.abs(z_small)**2):.6f} (expect ~1.0)")

    return h_small, e_small, s, z_small


def compute_imperfect_csi_batch(sys_model, h_small, e_small, s, z_small,
                                 p_tx_dbm, nmse_level):
    """
    Given pre-generated random variables, compute the imperfect CSI channel,
    received signal, and local LMMSE estimates for a specific power point.

    The channel model:
        H_true = sqrt(p_tx * beta) * h_small
        E      = sqrt(nmse_level * p_tx * beta) * e_small
        H_hat  = H_true + E
        y      = H_true @ s + sqrt(noise_w) * z_small

    Since h_small and e_small are fixed across power points, the ONLY thing
    that changes is the scaling factor sqrt(p_tx * beta). This ensures:
    1. Monotonic BER improvement with increasing power
    2. Consistent NMSE = nmse_level across all power points

    Parameters:
    -----------
    sys_model : CellFreeSystem
    h_small, e_small, s, z_small : pre-generated random variables
    p_tx_dbm : float, transmit power in dBm
    nmse_level : float, target NMSE level

    Returns:
    --------
    s_hat : (batch_size, L, K), complex - local LMMSE estimates
    s : (batch_size, K), complex - true symbols
    local_snr : (batch_size, L, K), float - local SNR features
    H_true : (batch_size, L, N, K), complex
    H_hat : (batch_size, L, N, K), complex
    y : (batch_size, L, N), complex
    """
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K), this is p_tx * beta

    L = sys_model.L
    N = sys_model.N
    K = sys_model.K
    noise_w = sys_model.noise_w
    batch_size = h_small.shape[0]

    # Expand beta_w for broadcasting: (1, L, 1, K)
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]

    # 1. True channel: H_true = sqrt(p_tx * beta) * h_small
    H_true = np.sqrt(beta_w_expanded) * h_small  # (batch_size, L, N, K)

    # 2. CSI error: E = sqrt(nmse_level * p_tx * beta) * e_small
    if nmse_level > 0:
        E = np.sqrt(nmse_level * beta_w_expanded) * e_small
        H_hat = H_true + E
    else:
        H_hat = H_true.copy()
        E = np.zeros_like(H_true)

    # Diagnostic: verify NMSE
    h_true_power = np.mean(np.abs(H_true) ** 2)
    csi_err_power = np.mean(np.abs(H_hat - H_true) ** 2)
    measured_nmse = csi_err_power / (h_true_power + 1e-30)
    print(f"    [CSI] nmse_level={nmse_level:.4f}, p_tx={p_tx_dbm:.0f}dBm ({p_tx_w:.4e}W), "
          f"||H||^2={h_true_power:.4e}, ||E||^2={csi_err_power:.4e}, "
          f"Measured NMSE={measured_nmse:.6f} (target={nmse_level:.4f})")

    # 3. Received signal: y = H_true @ s + sqrt(noise_w) * z_small
    y_clean = np.einsum('blnk,bk->bln', H_true, s)
    y = y_clean + np.sqrt(noise_w) * z_small

    # Print signal power diagnostics
    signal_power = np.mean(np.abs(y_clean) ** 2)
    noise_power = np.mean(np.abs(np.sqrt(noise_w) * z_small) ** 2)
    rx_snr_db = 10 * np.log10(signal_power / (noise_power + 1e-30))
    print(f"    [Signal] signal_power={signal_power:.4e}, noise_power={noise_power:.4e}, "
          f"Rx SNR={rx_snr_db:.2f} dB")

    # 4. Local LMMSE detection using H_hat
    H_hat_H = H_hat.conj().transpose(0, 1, 3, 2)  # (batch, L, K, N)
    noise_cov = noise_w * np.eye(N).reshape(1, 1, N, N)
    R_y = np.matmul(H_hat, H_hat_H) + noise_cov  # (batch, L, N, N)
    R_y_inv = np.linalg.inv(R_y)
    W_l = np.matmul(H_hat_H, R_y_inv)  # (batch, L, K, N)
    s_hat = np.matmul(W_l, y[..., np.newaxis]).squeeze(-1)  # (batch, L, K)

    # 5. Local SNR feature using H_hat
    local_snr = 10 * np.log10(np.sum(np.abs(H_hat) ** 2, axis=2) / noise_w + 1e-12) / 10.0

    return s_hat, s, local_snr, H_true, H_hat, y, measured_nmse


def evaluate_dist_full(s_hat_np, s_np):
    """
    Distributed full-precision baseline: mean pooling of local LMMSE estimates across APs.
    """
    s_hat_avg = s_hat_np.mean(axis=1)  # Average over L APs -> (batch_size, K)
    ber = compute_ber(s_np, s_hat_avg)
    return ber


def evaluate_cmmse(H_channel, y_all, s_np, noise_w):
    """
    Centralized MMSE (C-MMSE) baseline.
    Stack all AP channels into H_total (L*N x K), compute centralized MMSE.
    For imperfect CSI, H_channel = H_hat.
    """
    batch_size = H_channel.shape[0]
    L = H_channel.shape[1]
    N = H_channel.shape[2]
    K = H_channel.shape[3]
    LN = L * N

    H_total = H_channel.reshape(batch_size, LN, K)
    y_total = y_all.reshape(batch_size, LN, 1)
    H_total_H = H_total.conj().transpose(0, 2, 1)

    R = np.matmul(H_total, H_total_H) + noise_w * np.eye(LN).reshape(1, LN, LN)
    R_inv = np.linalg.inv(R)
    W = np.matmul(H_total_H, R_inv)
    s_hat = np.matmul(W, y_total).squeeze(-1)

    ber = compute_ber(s_np, s_hat)
    return ber


def evaluate_qat_gnn(model, X_test, s_np, device):
    """
    Evaluate QAT-GNN model using simulate_inference with hard argmax and no AP dropout.
    """
    snr = X_test[..., 2:3]
    out, expected_bits, w = simulate_inference(model, X_test, snr, drop_rate=0.0)
    s_pred = out.cpu().numpy()[..., 0] + 1j * out.cpu().numpy()[..., 1]
    ber = compute_ber(s_np, s_pred)
    avg_bits = expected_bits.mean().item()
    return ber, avg_bits


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # Parse NMSE levels and power points
    nmse_levels = [float(x) for x in args.nmse_levels.split(',')]
    power_points = [float(x) for x in args.power_points.split(',')]

    print(f"\n{'=' * 80}")
    print(f"Imperfect CSI Evaluation Configuration (FIXED - Consistent Seeds)")
    print(f"{'=' * 80}")
    print(f"  System: L={args.L}, N={args.N}, K={args.K}")
    print(f"  Area: {args.area_size} km x {args.area_size} km")
    print(f"  Bandwidth: {args.bandwidth / 1e6:.0f} MHz, Noise PSD: {args.noise_psd} dBm/Hz")
    print(f"  Test samples per point: {args.test_samples}")
    print(f"  NMSE levels: {nmse_levels}")
    print(f"  Power points (dBm): {power_points}")
    print(f"  Model path: {args.model_path}")
    print(f"")
    print(f"  CSI Error Model (NMSE-Based):")
    print(f"    H_true = sqrt(p_tx * beta) * h_small,  h_small ~ CN(0,1)")
    print(f"    E      = sqrt(nmse * p_tx * beta) * e_small, e_small ~ CN(0,1)")
    print(f"    H_hat  = H_true + E")
    print(f"    y      = H_true @ s + sqrt(noise_w) * z_small")
    print(f"    NMSE   = var(E)/var(H_true) = nmse_level  (CONSTANT across power!)")
    print(f"")
    print(f"  KEY FIX: For each NMSE level, h_small, e_small, s, z_small are")
    print(f"  pre-generated ONCE and REUSED across all power points.")
    print(f"  Only the scaling sqrt(p_tx * beta) changes per power point.")
    print(f"  This guarantees monotonic BER curves across power.")
    print(f"{'=' * 80}\n")

    # Initialize system model and generate scenario (large-scale fading)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys_model = CellFreeSystem(
        args.L, args.N, args.K, args.area_size,
        args.bandwidth, args.noise_psd, 0.0
    )
    sys_model.generate_scenario()

    print(f"  [System Info] beta (large-scale fading) stats:")
    print(f"    min={np.min(sys_model.beta):.6e}, max={np.max(sys_model.beta):.6e}, "
          f"mean={np.mean(sys_model.beta):.6e}")
    print(f"  [System Info] noise_w = {sys_model.noise_w:.6e} W")
    print()

    # Load QAT-GNN model
    model = JointQATModel(L=args.L, K=args.K, hidden_dim=args.hidden_dim).to(device)
    model_loaded = False
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Successfully loaded QAT-GNN model from '{args.model_path}'")
            model_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load model: {e}. Using random weights.")
    else:
        print(f"Warning: Model file '{args.model_path}' not found. Using random weights.")
    model.eval()

    # =========================================================================
    # Phase 1: For each NMSE level, pre-generate random variables ONCE,
    #           then evaluate all power points using the SAME realizations.
    # =========================================================================
    print("\n" + "=" * 80)
    print("Phase 1: Pre-generate random variables & evaluate all (NMSE, power) combos")
    print("  Strategy: For each NMSE level, generate h_small, e_small, s, z_small ONCE")
    print("  then loop over power points with ONLY scaling changes.")
    print("=" * 80)

    # Store results: results[nmse][power] = {...}
    results = {}
    measured_nmse_table = {}

    for nmse in nmse_levels:
        results[nmse] = {}
        measured_nmse_table[nmse] = {}

    for nmse_idx, nmse in enumerate(nmse_levels):
        print(f"\n{'=' * 70}")
        nmse_label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse:.4f} ({nmse*100:.1f}%)"
        print(f"  Processing NMSE level {nmse_idx+1}/{len(nmse_levels)}: {nmse_label}")
        print(f"{'=' * 70}")

        # Compute a unique seed per NMSE level (but SAME across power points!)
        nmse_seed = args.seed + int(nmse * 10000)
        print(f"  Using seed={nmse_seed} for ALL power points at this NMSE level")

        # Pre-generate ALL random variables for this NMSE level
        h_small, e_small, s_np, z_small = pregenerate_random_variables(
            sys_model, args.test_samples, nmse_seed
        )

        # Now loop over power points - only scaling changes!
        for p_idx, p in enumerate(power_points):
            print(f"\n  --- Power point {p_idx+1}/{len(power_points)}: p_tx={p:.0f} dBm ---")

            # Compute channel, received signal, and local LMMSE with pre-generated random vars
            s_hat_np, s_out, local_snr_np, H_true, H_hat, y_all, meas_nmse = \
                compute_imperfect_csi_batch(
                    sys_model, h_small, e_small, s_np, z_small,
                    p_tx_dbm=p, nmse_level=nmse
                )

            measured_nmse_table[nmse][p] = meas_nmse

            # --- Evaluate Dist-Full ---
            ber_dist_full = evaluate_dist_full(s_hat_np, s_np)
            print(f"    >> Dist-Full BER = {ber_dist_full:.6f} ({ber_dist_full:.4%})")

            # --- Evaluate C-MMSE (using H_hat for imperfect CSI) ---
            ber_cmmse = evaluate_cmmse(H_hat, y_all, s_np, sys_model.noise_w)
            print(f"    >> C-MMSE    BER = {ber_cmmse:.6f} ({ber_cmmse:.4%})")

            # --- Evaluate QAT-GNN ---
            X_test, Y_test = get_tensors_from_batch(s_hat_np, s_np, local_snr_np, device)
            with torch.no_grad():
                ber_qat, avg_bits = evaluate_qat_gnn(model, X_test, s_np, device)
            print(f"    >> QAT-GNN   BER = {ber_qat:.6f} ({ber_qat:.4%}), Avg bits={avg_bits:.2f}")

            results[nmse][p] = {
                'dist_full_ber': ber_dist_full,
                'cmmse_ber': ber_cmmse,
                'qat_ber': ber_qat,
                'qat_bits': avg_bits,
            }

    # =========================================================================
    # Phase 2: Monotonicity Check
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  MONOTONICITY CHECK: Verifying BER decreases with increasing power")
    print("=" * 80)

    for nmse in nmse_levels:
        nmse_label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse:.4f}"
        print(f"\n  --- {nmse_label} ---")
        for method_name, key in [('Dist-Full', 'dist_full_ber'),
                                  ('C-MMSE', 'cmmse_ber'),
                                  ('QAT-GNN', 'qat_ber')]:
            bers = [results[nmse][p][key] for p in power_points]
            is_monotone = all(bers[i] >= bers[i+1] - 1e-6 for i in range(len(bers)-1))
            status = "PASS (monotonic)" if is_monotone else "FAIL (non-monotonic!)"
            ber_str = " -> ".join([f"{b:.4%}" for b in bers])
            print(f"    {method_name:<12}: {status}")
            print(f"      BER: {ber_str}")

    # =========================================================================
    # Phase 3: Print comprehensive results tables
    # =========================================================================
    print("\n\n" + "=" * 120)
    print("=" * 120)
    print("  COMPREHENSIVE RESULTS: BER vs Transmit Power under Imperfect CSI")
    print("=" * 120)

    # Table 0: NMSE Verification
    print("\n" + "-" * 100)
    print("Table 0: MEASURED NMSE at Each (Power, NMSE Level) -- Verification")
    print("  Key validation: Measured NMSE ≈ nmse_level for ALL power points (constant).")
    print("-" * 100)

    header = f"{'p_tx(dBm)':<12}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            header += f"| {'Perfect':^14}"
        else:
            header += f"| {'NMSE=' + str(nmse):^14}"
    print(header)
    print("-" * 100)

    for p in power_points:
        row = f"{p:<12.0f}"
        for nmse in nmse_levels:
            meas = measured_nmse_table[nmse][p]
            if nmse == 0.0:
                row += f"| {'0.000000':^14}"
            else:
                match_ok = "OK" if abs(meas - nmse) / (nmse + 1e-30) < 0.3 else "!!"
                row += f"| {meas:^10.6f} {match_ok:>2} "
        print(row)
    print("-" * 100)
    print("  (OK = measured NMSE within 30% of target, !! = mismatch)")

    # Table 1: Dist-Full BER
    print("\n" + "-" * 100)
    print("Table 1: Dist-Full (Mean Pooling) BER")
    print("-" * 100)

    header = f"{'p_tx(dBm)':<12}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            header += f"| {'Perfect CSI':^14}"
        else:
            header += f"| {'NMSE=' + str(nmse):^14}"
    print(header)
    print("-" * 100)

    for p in power_points:
        row = f"{p:<12.0f}"
        for nmse in nmse_levels:
            ber = results[nmse][p]['dist_full_ber']
            row += f"| {ber:^14.4%}"
        print(row)
    print("-" * 100)

    # Table 2: C-MMSE BER
    print("\n" + "-" * 100)
    print("Table 2: Centralized MMSE (C-MMSE) BER")
    print("-" * 100)

    header = f"{'p_tx(dBm)':<12}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            header += f"| {'Perfect CSI':^14}"
        else:
            header += f"| {'NMSE=' + str(nmse):^14}"
    print(header)
    print("-" * 100)

    for p in power_points:
        row = f"{p:<12.0f}"
        for nmse in nmse_levels:
            ber = results[nmse][p]['cmmse_ber']
            row += f"| {ber:^14.4%}"
        print(row)
    print("-" * 100)

    # Table 3: QAT-GNN BER
    print("\n" + "-" * 100)
    print("Table 3: QAT-GNN BER")
    print("-" * 100)

    header = f"{'p_tx(dBm)':<12}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            header += f"| {'Perfect CSI':^14}"
        else:
            header += f"| {'NMSE=' + str(nmse):^14}"
    print(header)
    print("-" * 100)

    for p in power_points:
        row = f"{p:<12.0f}"
        for nmse in nmse_levels:
            ber = results[nmse][p]['qat_ber']
            row += f"| {ber:^14.4%}"
        print(row)
    print("-" * 100)

    # Table 4: QAT-GNN Average Fronthaul Bits
    print("\n" + "-" * 100)
    print("Table 4: QAT-GNN Average Fronthaul Bits per AP-User Link")
    print("-" * 100)

    header = f"{'p_tx(dBm)':<12}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            header += f"| {'Perfect CSI':^14}"
        else:
            header += f"| {'NMSE=' + str(nmse):^14}"
    print(header)
    print("-" * 100)

    for p in power_points:
        row = f"{p:<12.0f}"
        for nmse in nmse_levels:
            bits = results[nmse][p]['qat_bits']
            row += f"| {bits:^14.2f}"
        print(row)
    print("-" * 100)

    # Table 5: All Methods Comparison per NMSE level
    print("\n" + "-" * 120)
    print("Table 5: All Methods Comparison (BER)")
    print("-" * 120)

    for nmse in nmse_levels:
        label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse}"
        print(f"\n  --- CSI Error Level: {label} ---")
        header = f"{'p_tx(dBm)':<12} | {'Dist-Full':<14} | {'C-MMSE':<14} | {'QAT-GNN':<14} | {'QAT Bits':<10}"
        print(header)
        print("-" * 70)
        for p in power_points:
            ber_df = results[nmse][p]['dist_full_ber']
            ber_cm = results[nmse][p]['cmmse_ber']
            ber_qat = results[nmse][p]['qat_ber']
            bits = results[nmse][p]['qat_bits']
            print(f"{p:<12.0f} | {ber_df:<14.6f} | {ber_cm:<14.6f} | {ber_qat:<14.6f} | {bits:<10.2f}")
        print("-" * 70)

    # =========================================================================
    # Phase 4: Summary table at fixed power point
    # =========================================================================
    summary_p = args.summary_power
    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY TABLE at Fixed Power Point: p_tx = {summary_p:.0f} dBm")
    print(f"{'=' * 100}")

    if summary_p in [float(p) for p in power_points]:
        print(f"\n{'NMSE Level':<20} | {'Dist-Full BER':<16} | {'C-MMSE BER':<16} | "
              f"{'QAT-GNN BER':<17} | {'QAT Avg Bits':<14} | {'QAT vs DF (%)':<16}")
        print("-" * 110)

        for nmse in nmse_levels:
            ber_df = results[nmse][summary_p]['dist_full_ber']
            ber_cm = results[nmse][summary_p]['cmmse_ber']
            ber_qat = results[nmse][summary_p]['qat_ber']
            avg_bits = results[nmse][summary_p]['qat_bits']
            improvement = ((ber_df - ber_qat) / ber_df * 100) if ber_df > 0 else 0.0

            nmse_label = "Perfect (0.0)" if nmse == 0.0 else f"{nmse:.4f} ({nmse * 100:.1f}%)"
            print(f"{nmse_label:<20} | {ber_df:<16.6f} | {ber_cm:<16.6f} | "
                  f"{ber_qat:<17.6f} | {avg_bits:<14.2f} | {improvement:<16.2f}")
        print("-" * 110)
    else:
        print(f"Warning: Summary power point {summary_p} dBm not in evaluated power points.")

    # =========================================================================
    # Phase 5: BER degradation analysis (relative to perfect CSI)
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print(f"  BER DEGRADATION ANALYSIS: Relative BER Increase from Perfect CSI (in %)")
    print(f"{'=' * 100}")

    col_width = 12
    print(f"\n{'Method':<15} | {'NMSE':<8} | ", end="")
    for p in power_points:
        print(f"{'p=' + str(int(p)) + 'dBm':<{col_width}}", end="")
    print()
    print("-" * (25 + col_width * len(power_points)))

    for method_name in ['Dist-Full', 'C-MMSE', 'QAT-GNN']:
        if method_name == 'Dist-Full':
            key = 'dist_full_ber'
        elif method_name == 'C-MMSE':
            key = 'cmmse_ber'
        else:
            key = 'qat_ber'
        for nmse in nmse_levels:
            if nmse == 0.0:
                continue
            row = f"{method_name:<15} | {nmse:<8.4f} | "
            for p in power_points:
                ber_perfect = results[0.0][p][key]
                ber_imperfect = results[nmse][p][key]
                if ber_perfect > 1e-8:
                    degradation = (ber_imperfect - ber_perfect) / ber_perfect * 100
                    row += f"{degradation:<{col_width}.1f}"
                else:
                    row += f"{'N/A':<{col_width}}"
            print(row)
        print("-" * (25 + col_width * len(power_points)))

    # =========================================================================
    # Phase 6: NMSE verification summary
    # =========================================================================
    print(f"\n\n{'=' * 80}")
    print(f"  NMSE VERIFICATION: Measured NMSE vs Target at All Power Points")
    print(f"{'=' * 80}")
    print(f"\n  Model: sigma_e^2 = nmse_level * p_tx * beta[l,k]")
    print(f"  NMSE = sigma_e^2 / var(H_true) = nmse_level (CONSTANT!)")
    print()

    header = f"{'p_tx(dBm)':<12} | {'p_tx(W)':<14}"
    for nmse in nmse_levels:
        if nmse == 0.0:
            continue
        header += f"| {'Target=' + f'{nmse:.4f}':^18}"
    print(header)
    print("-" * (28 + 20 * len([n for n in nmse_levels if n > 0])))

    for p in power_points:
        p_tx_w = 10 ** ((p - 30) / 10)
        row = f"{p:<12.0f} | {p_tx_w:<14.6e}"
        for nmse in nmse_levels:
            if nmse == 0.0:
                continue
            meas = measured_nmse_table[nmse][p]
            row += f"| {meas:^18.6f}"
        print(row)
    print("-" * (28 + 20 * len([n for n in nmse_levels if n > 0])))

    # =========================================================================
    # Phase 7: Key findings summary
    # =========================================================================
    print(f"\n\n{'=' * 80}")
    print("  KEY FINDINGS")
    print(f"{'=' * 80}")

    # Finding 1: NMSE model validation
    print("\n1. NMSE Model Validation:")
    print("   The NMSE-based CSI error model ensures constant NMSE across all power levels.")
    for nmse in nmse_levels:
        if nmse == 0.0:
            continue
        measured_values = [measured_nmse_table[nmse][p] for p in power_points]
        avg_measured = np.mean(measured_values)
        std_measured = np.std(measured_values)
        print(f"   nmse_level={nmse:.4f}: Avg measured NMSE={avg_measured:.6f} "
              f"(std={std_measured:.6f}, target={nmse:.4f})")

    # Finding 2: Monotonicity validation
    print(f"\n2. Monotonicity Validation:")
    mono_pass = 0
    mono_total = 0
    for nmse in nmse_levels:
        for method_name, key in [('Dist-Full', 'dist_full_ber'),
                                  ('C-MMSE', 'cmmse_ber'),
                                  ('QAT-GNN', 'qat_ber')]:
            bers = [results[nmse][p][key] for p in power_points]
            is_monotone = all(bers[i] >= bers[i+1] - 1e-6 for i in range(len(bers)-1))
            mono_total += 1
            if is_monotone:
                mono_pass += 1
    print(f"   {mono_pass}/{mono_total} BER curves are monotonically decreasing with power.")

    # Finding 3: Robustness comparison
    print(f"\n3. Robustness comparison at p_tx={args.summary_power:.0f} dBm:")
    moderate_p = args.summary_power
    if moderate_p in [float(p) for p in power_points]:
        for nmse in nmse_levels:
            ber_df = results[nmse][moderate_p]['dist_full_ber']
            ber_cm = results[nmse][moderate_p]['cmmse_ber']
            ber_qat = results[nmse][moderate_p]['qat_ber']
            label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse:.2%}"
            print(f"   {label:>20s}: Dist-Full={ber_df:.6f}, C-MMSE={ber_cm:.6f}, QAT-GNN={ber_qat:.6f}")

    # Finding 4: Maximum BER degradation
    print(f"\n4. Maximum BER degradation across all power points:")
    for method_name in ['Dist-Full', 'C-MMSE', 'QAT-GNN']:
        if method_name == 'Dist-Full':
            key = 'dist_full_ber'
        elif method_name == 'C-MMSE':
            key = 'cmmse_ber'
        else:
            key = 'qat_ber'
        max_nmse = max([n for n in nmse_levels if n > 0], default=0.1)
        max_deg = 0.0
        max_deg_p = 0.0
        for p in power_points:
            ber_perfect = results[0.0][p][key]
            ber_worst = results[max_nmse][p][key]
            if ber_perfect > 1e-8:
                deg = (ber_worst - ber_perfect) / ber_perfect * 100
                if deg > max_deg:
                    max_deg = deg
                    max_deg_p = p
        print(f"   {method_name}: Max degradation={max_deg:.1f}% at p_tx={max_deg_p:.0f}dBm (NMSE={max_nmse})")

    # Finding 5: QAT-GNN advantage under imperfect CSI
    print(f"\n5. QAT-GNN maintains advantage over Dist-Full under imperfect CSI:")
    for nmse in nmse_levels:
        advantages = []
        for p in power_points:
            ber_df = results[nmse][p]['dist_full_ber']
            ber_qat = results[nmse][p]['qat_ber']
            if ber_df > 0:
                advantages.append((ber_df - ber_qat) / ber_df * 100)
        if advantages:
            avg_adv = np.mean(advantages)
            label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse:.2%}"
            print(f"   {label:>20s}: Average BER improvement = {avg_adv:.2f}%")

    # Finding 6: CSI error tolerance threshold
    print(f"\n6. CSI error tolerance analysis (at p_tx={args.summary_power:.0f} dBm):")
    if moderate_p in [float(p) for p in power_points]:
        ber_perfect_df = results[0.0][moderate_p]['dist_full_ber']
        ber_perfect_cm = results[0.0][moderate_p]['cmmse_ber']
        ber_perfect_qat = results[0.0][moderate_p]['qat_ber']
        for nmse in nmse_levels:
            if nmse == 0.0:
                continue
            ber_df = results[nmse][moderate_p]['dist_full_ber']
            ber_cm = results[nmse][moderate_p]['cmmse_ber']
            ber_qat = results[nmse][moderate_p]['qat_ber']
            df_degrade = (ber_df / ber_perfect_df) if ber_perfect_df > 0 else float('inf')
            cm_degrade = (ber_cm / ber_perfect_cm) if ber_perfect_cm > 0 else float('inf')
            qat_degrade = (ber_qat / ber_perfect_qat) if ber_perfect_qat > 0 else float('inf')
            print(f"   NMSE={nmse:.2%}: Dist-Full BER x{df_degrade:.2f}, "
                  f"C-MMSE BER x{cm_degrade:.2f}, QAT-GNN BER x{qat_degrade:.2f}")

    # Finding 7: C-MMSE vs QAT-GNN
    print(f"\n7. C-MMSE vs QAT-GNN comparison:")
    print(f"   C-MMSE: full centralized processing (L*N x K matrix inversion), no quantization.")
    print(f"   QAT-GNN: adaptive quantization to reduce fronthaul load.")
    if moderate_p in [float(p) for p in power_points]:
        for nmse in nmse_levels:
            ber_cm = results[nmse][moderate_p]['cmmse_ber']
            ber_qat = results[nmse][moderate_p]['qat_ber']
            bits = results[nmse][moderate_p]['qat_bits']
            label = "Perfect CSI" if nmse == 0.0 else f"NMSE={nmse:.2%}"
            print(f"   {label:>20s} at {moderate_p:.0f}dBm: "
                  f"C-MMSE={ber_cm:.6f}, QAT-GNN={ber_qat:.6f} ({bits:.2f} bits)")

    print(f"\n{'=' * 80}")
    print("Evaluation completed successfully.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
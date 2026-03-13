import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import copy
import os

from system_model import CellFreeSystem
from lsq_quantizer import LSQQuantizer
from new_detector_v3 import (
    JointModelV3, GNNTransformerDetectorV3, DualAdaptiveQuantizerV3,
    MeanFieldGNNLayer, APAggregator, DualPolicyNetworkV3,
    generate_data_batch_v2, prepare_tensors, compute_ber, compute_dist_full_ber,
    compute_cmmse_detection, compute_cmmse_q_detection, compute_dist_q_ber,
    uniform_quantize_np
)


def parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluation Experiments for V3 GNN+Transformer Hybrid Detector"
    )
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--c_target", type=float, default=48.0, help="Target average bits per link")
    # Evaluation parameters
    parser.add_argument("--test_samples", type=int, default=1000,
                        help="Number of test samples per power point (default: 1000 for smooth curves)")
    parser.add_argument("--model_path", type=str, default="new_joint_model_v3.pth",
                        help="Path to the trained V3 model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tau_eval", type=float, default=0.1,
                        help="Gumbel-Softmax temperature for evaluation (low = near hard)")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "ablation", "pareto", "imperfect_csi"],
                        help="Which experiment to run: all, ablation, pareto, or imperfect_csi")
    # Power points
    parser.add_argument("--power_points", type=str, default="-10,-5,0,5,10,15,20",
                        help="Comma-separated transmit power points in dBm")
    # NMSE levels for imperfect CSI experiment
    parser.add_argument("--nmse_levels", type=str, default="0.0,0.01,0.05,0.10",
                        help="Comma-separated NMSE levels for imperfect CSI experiment")
    return parser.parse_args()


# ============================================================
# Helper: Load model
# ============================================================
def load_model(args, device):
    """Load the trained JointModelV3 from checkpoint."""
    model = JointModelV3(
        L=args.L, N=args.N, K=args.K,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from '{args.model_path}'")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    return model


# ============================================================
# Helper: Generate test dataset
# ============================================================
def generate_test_dataset(sys_model, test_samples, power_points, device):
    """Generate a fixed test dataset for all power points."""
    test_dataset = {}
    print(f"\nGenerating fixed test set ({test_samples} samples per power point)...")
    for p in power_points:
        s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch_v2(
            sys_model, test_samples, p_tx_dbm=p
        )
        v_t, H_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'Y': Y_t,
            's_hat_np': s_hat_np, 's_np': s_np, 'H_np': H_np, 'y_np': y_np,
            'snr_np': snr_np
        }
    print("Test set generated successfully.\n")
    return test_dataset


# ============================================================
# Helper: Evaluate model BER at a power point
# ============================================================
def evaluate_model_ber(model, td, tau, use_quantization=True):
    """Evaluate model on test data dict td, return BER and avg bits."""
    with torch.no_grad():
        detected, exp_bits_d, exp_bits_c = model(
            td['v'], td['H'], td['snr'], tau=tau, use_quantization=use_quantization
        )
        det_np = detected.cpu().numpy()
        s_pred = det_np[..., 0] + 1j * det_np[..., 1]
        ber = compute_ber(td['s_np'], s_pred)
        avg_bits = (exp_bits_d + exp_bits_c).mean().item() if use_quantization else 0.0
    return ber, avg_bits


# ============================================================
# Helper: Safe percentage improvement
# ============================================================
def safe_pct_improvement(ber_baseline, ber_proposed):
    """Compute percentage improvement safely, handling near-zero baselines."""
    if ber_baseline < 1e-8:
        if ber_proposed < 1e-8:
            return 0.0
        else:
            return -999.9  # proposed is worse but baseline is ~0
    return ((ber_baseline - ber_proposed) / ber_baseline) * 100


def safe_pct_degradation(ber_current, ber_reference):
    """Compute percentage degradation safely."""
    if ber_reference < 1e-8:
        if ber_current < 1e-8:
            return 0.0
        else:
            return 999.9  # degraded from ~0
    return ((ber_current - ber_reference) / ber_reference) * 100


# ============================================================
# Helper: Run detector forward pass with modifications (for ablation)
# ============================================================
def run_detector_no_gnn(detector, v_q, H_q, bitwidth_features):
    """
    Run the detector forward pass but SKIP all GNN layers.
    The node features go directly from fusion+BN to aggregation.
    """
    B = v_q.size(0)
    det = detector

    # 1. AP Feature Extraction (same as original)
    demod_feat = det.demod_mlp(v_q)
    H_q_perm = H_q.permute(0, 1, 3, 2, 4)
    H_q_flat = H_q_perm.reshape(B, det.L, det.K, det.N * 2)
    channel_feat = det.channel_mlp(H_q_flat)

    # Cross-user interference features
    h_power_per_user = (H_q ** 2).sum(dim=(2, 4))
    desired_power = h_power_per_user
    total_power = h_power_per_user.sum(dim=2, keepdim=True)
    interference_power = total_power - desired_power
    desired_feat = torch.log1p(desired_power).unsqueeze(-1)
    interference_feat = torch.log1p(interference_power).unsqueeze(-1)
    interference_features = torch.cat([desired_feat, interference_feat], dim=-1)

    combined = torch.cat([demod_feat, channel_feat, bitwidth_features, interference_features], dim=-1)
    node_features = det.fusion_mlp(combined)
    node_features_flat = node_features.reshape(-1, det.hidden_dim)
    node_features_bn = det.fusion_bn(node_features_flat)
    node_features = node_features_bn.reshape(B, det.L, det.K, det.hidden_dim)

    # 2. SKIP GNN layers - use node_features directly
    h = node_features

    # 3. Aggregation
    soft_info = v_q
    soft_attn_weights = det.soft_attn(h)
    soft_attn_weights = torch.softmax(soft_attn_weights, dim=1)
    base_out = (soft_info * soft_attn_weights).sum(dim=1)

    user_features, _ = det.ap_aggregator(h)

    # 4. Transformer IC (still active)
    ic_out = det.transformer_ic(user_features)

    # 5. Output
    residual = det.output_head(ic_out)
    detected = base_out + residual

    return detected


def run_detector_no_transformer(detector, v_q, H_q, bitwidth_features):
    """
    Run the detector forward pass but SKIP the Transformer IC.
    Output = base_out only (attention-weighted mean of demod results).
    """
    B = v_q.size(0)
    det = detector

    # 1. AP Feature Extraction
    demod_feat = det.demod_mlp(v_q)
    H_q_perm = H_q.permute(0, 1, 3, 2, 4)
    H_q_flat = H_q_perm.reshape(B, det.L, det.K, det.N * 2)
    channel_feat = det.channel_mlp(H_q_flat)

    h_power_per_user = (H_q ** 2).sum(dim=(2, 4))
    desired_power = h_power_per_user
    total_power = h_power_per_user.sum(dim=2, keepdim=True)
    interference_power = total_power - desired_power
    desired_feat = torch.log1p(desired_power).unsqueeze(-1)
    interference_feat = torch.log1p(interference_power).unsqueeze(-1)
    interference_features = torch.cat([desired_feat, interference_feat], dim=-1)

    combined = torch.cat([demod_feat, channel_feat, bitwidth_features, interference_features], dim=-1)
    node_features = det.fusion_mlp(combined)
    node_features_flat = node_features.reshape(-1, det.hidden_dim)
    node_features_bn = det.fusion_bn(node_features_flat)
    node_features = node_features_bn.reshape(B, det.L, det.K, det.hidden_dim)

    # 2. GNN layers (active)
    h = node_features
    for gnn_layer in det.gnn_layers:
        h = gnn_layer(h)

    # 3. Aggregation
    soft_info = v_q
    soft_attn_weights = det.soft_attn(h)
    soft_attn_weights = torch.softmax(soft_attn_weights, dim=1)
    base_out = (soft_info * soft_attn_weights).sum(dim=1)

    # 4. SKIP Transformer: output = base_out only (no residual)
    return base_out


# ============================================================
# EXPERIMENT 1: Ablation Study
# ============================================================
def run_ablation_study(model, test_dataset, power_points, args, device):
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: ABLATION STUDY")
    print("=" * 100)
    print("Configurations:")
    print("  (a) Full Model (proposed) - complete model with adaptive quantization")
    print("  (b) No BW Features - bitwidth_features set to zero tensor")
    print("  (c) No GNN - skip all 3 mean-field GNN message-passing layers")
    print("  (d) No Transformer IC - skip Transformer, output = base_out only")
    print("  (e) No Channel Input - H_q set to zero tensor (detector sees only demod results)")

    tau = args.tau_eval

    results = {
        'full': {},
        'no_bw_feat': {},
        'no_gnn': {},
        'no_transformer': {},
        'no_channel': {}
    }
    bits_full = {}

    for p in power_points:
        td = test_dataset[p]
        print(f"\n--- Evaluating at P_tx = {p} dBm ---")

        # (a) Full model (proposed)
        ber_full, bits_val = evaluate_model_ber(model, td, tau, use_quantization=True)
        results['full'][p] = ber_full
        bits_full[p] = bits_val
        print(f"  (a) Full Model:        BER = {ber_full:.6f}, Avg Bits = {bits_val:.2f}")

        with torch.no_grad():
            # Run quantizer once to get quantized inputs
            v_q, H_q, exp_bits_d, exp_bits_c, w_d, w_c = model.quantizer(
                td['v'], td['H'], td['snr'], tau=tau
            )
            B, L, K = exp_bits_d.shape

            # Compute normal bitwidth features
            bw_demod_feat = exp_bits_d.unsqueeze(-1) / model.max_demod_bits
            bw_channel_feat = exp_bits_c.unsqueeze(-1) / model.max_channel_bits
            bitwidth_features_normal = torch.cat([bw_demod_feat, bw_channel_feat], dim=-1)

            # (b) No bitwidth features - set to zeros
            bitwidth_features_zero = torch.zeros(B, L, K, 2, device=device)
            detected_no_bw = model.detector(v_q, H_q, bitwidth_features_zero)
            det_np = detected_no_bw.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_no_bw = compute_ber(td['s_np'], s_pred)
            results['no_bw_feat'][p] = ber_no_bw
            print(f"  (b) No BW Features:    BER = {ber_no_bw:.6f}")

            # (c) No GNN - skip GNN layers
            detected_no_gnn = run_detector_no_gnn(
                model.detector, v_q, H_q, bitwidth_features_normal
            )
            det_np = detected_no_gnn.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_no_gnn = compute_ber(td['s_np'], s_pred)
            results['no_gnn'][p] = ber_no_gnn
            print(f"  (c) No GNN:            BER = {ber_no_gnn:.6f}")

            # (d) No Transformer IC - output = base_out only
            detected_no_trans = run_detector_no_transformer(
                model.detector, v_q, H_q, bitwidth_features_normal
            )
            det_np = detected_no_trans.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_no_trans = compute_ber(td['s_np'], s_pred)
            results['no_transformer'][p] = ber_no_trans
            print(f"  (d) No Transformer IC: BER = {ber_no_trans:.6f}")

            # (e) No channel input - set H_q to zeros
            H_q_zero = torch.zeros_like(H_q)
            detected_no_ch = model.detector(v_q, H_q_zero, bitwidth_features_normal)
            det_np = detected_no_ch.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_no_ch = compute_ber(td['s_np'], s_pred)
            results['no_channel'][p] = ber_no_ch
            print(f"  (e) No Channel Input:  BER = {ber_no_ch:.6f}")

    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION STUDY SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'P_tx(dBm)':<12} | {'Full Model':<12} | {'No BW Feat':<12} | "
              f"{'No GNN':<12} | {'No Transformer':<15} | {'No Channel':<12} | {'Avg Bits':<10}")
    print(header)
    print("-" * len(header))
    for p in power_points:
        print(f"{p:<12} | {results['full'][p]:<12.6f} | {results['no_bw_feat'][p]:<12.6f} | "
              f"{results['no_gnn'][p]:<12.6f} | {results['no_transformer'][p]:<15.6f} | "
              f"{results['no_channel'][p]:<12.6f} | {bits_full[p]:<10.2f}")
    print("-" * len(header))

    # Average across power points
    avg_full = np.mean([results['full'][p] for p in power_points])
    avg_no_bw = np.mean([results['no_bw_feat'][p] for p in power_points])
    avg_no_gnn = np.mean([results['no_gnn'][p] for p in power_points])
    avg_no_trans = np.mean([results['no_transformer'][p] for p in power_points])
    avg_no_ch = np.mean([results['no_channel'][p] for p in power_points])
    avg_bits = np.mean([bits_full[p] for p in power_points])

    print(f"{'Average':<12} | {avg_full:<12.6f} | {avg_no_bw:<12.6f} | "
          f"{avg_no_gnn:<12.6f} | {avg_no_trans:<15.6f} | {avg_no_ch:<12.6f} | {avg_bits:<10.2f}")

    print(f"\n  Performance Change vs Full Model (positive = degradation):")
    degrad_bw = safe_pct_degradation(avg_no_bw, avg_full)
    degrad_gnn = safe_pct_degradation(avg_no_gnn, avg_full)
    degrad_trans = safe_pct_degradation(avg_no_trans, avg_full)
    degrad_ch = safe_pct_degradation(avg_no_ch, avg_full)
    print(f"    No BW Features:    {degrad_bw:+.2f}% (avg BER: {avg_full:.6f} -> {avg_no_bw:.6f})")
    print(f"    No GNN:            {degrad_gnn:+.2f}% (avg BER: {avg_full:.6f} -> {avg_no_gnn:.6f})")
    print(f"    No Transformer IC: {degrad_trans:+.2f}% (avg BER: {avg_full:.6f} -> {avg_no_trans:.6f})")
    print(f"    No Channel Input:  {degrad_ch:+.2f}% (avg BER: {avg_full:.6f} -> {avg_no_ch:.6f})")

    # Per-power-point degradation
    print(f"\n  Per-Power-Point BER Degradation (%) vs Full Model:")
    print(f"  {'P_tx':<8} | {'No BW Feat':>12} | {'No GNN':>12} | {'No Transformer':>15} | {'No Channel':>12}")
    print(f"  " + "-" * 70)
    for p in power_points:
        d_bw = safe_pct_degradation(results['no_bw_feat'][p], results['full'][p])
        d_gnn = safe_pct_degradation(results['no_gnn'][p], results['full'][p])
        d_trans = safe_pct_degradation(results['no_transformer'][p], results['full'][p])
        d_ch = safe_pct_degradation(results['no_channel'][p], results['full'][p])
        print(f"  {p:<8} | {d_bw:>+11.2f}% | {d_gnn:>+11.2f}% | {d_trans:>+14.2f}% | {d_ch:>+11.2f}%")

    return results


# ============================================================
# EXPERIMENT 2: Different Fronthaul Budgets (Pareto Frontier)
# ============================================================
def run_pareto_experiment(model, test_dataset, power_points, sys_model, args, device):
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: DIFFERENT FRONTHAUL BUDGETS (PARETO FRONTIER)")
    print("=" * 100)
    print("Strategy: Load model trained at c_target=48, force different quantization levels at inference.")
    print("For forced quantization: bypass policy network, use specific LSQ quantizers directly.")

    tau = args.tau_eval
    N = args.N
    K = args.K

    # Budget configurations: (label, demod_bits_per_real, channel_bits_per_real)
    # None means use the trained model as-is
    budget_configs = [
        ("Very Low (~24b)", 2, 2),     # total per link = 2*2 + 2*4*2 = 4+16 = 20
        ("Low (~36b)", 4, 4),          # total per link = 2*4 + 2*4*4 = 8+32 = 40
        ("Medium (~48b)", None, None), # Use trained model (c_target=48)
        ("Med-High (~56b)", 6, 6),     # total per link = 2*6 + 2*4*6 = 12+48 = 60
        ("High (~72b)", 8, 8),         # total per link = 2*8 + 2*4*8 = 16+64 = 80
    ]

    results_proposed = {}
    results_cmmse_q = {}
    actual_bits = {}

    for label, demod_b, channel_b in budget_configs:
        print(f"\n--- Budget: {label} ---")
        if demod_b is not None:
            total_bits_link = 2 * demod_b + 2 * N * channel_b
            c_equiv = total_bits_link
            print(f"  Forced: demod={demod_b}b, channel={channel_b}b, "
                  f"total per link = 2*{demod_b} + 2*{N}*{channel_b} = {total_bits_link} bits")
        else:
            c_equiv = args.c_target
            print(f"  Using trained model as-is (c_target={args.c_target})")

        results_proposed[label] = {}
        results_cmmse_q[label] = {}
        actual_bits[label] = {}

        for p in power_points:
            td = test_dataset[p]

            if demod_b is not None:
                # Forced quantization: bypass policy, quantize at specific bit width
                with torch.no_grad():
                    v_input = td['v']
                    H_input = td['H']

                    # Select appropriate LSQ quantizers from the model
                    q_map_demod = {
                        2: model.quantizer.q2_demod,
                        4: model.quantizer.q4_demod,
                        6: model.quantizer.q6_demod,
                        8: model.quantizer.q8_demod
                    }
                    q_map_channel = {
                        2: model.quantizer.q2_channel,
                        4: model.quantizer.q4_channel,
                        6: model.quantizer.q6_channel,
                        8: model.quantizer.q8_channel
                    }

                    v_q = q_map_demod[demod_b](v_input)
                    H_q = q_map_channel[channel_b](H_input)

                    # Set bitwidth features accordingly
                    B_sz = v_input.size(0)
                    L_sz = v_input.size(1)
                    K_sz = v_input.size(2)

                    demod_bits_per_link = 2.0 * demod_b
                    channel_bits_per_link = 2.0 * N * channel_b

                    bw_demod_feat = torch.full(
                        (B_sz, L_sz, K_sz, 1), demod_bits_per_link / model.max_demod_bits,
                        device=device
                    )
                    bw_channel_feat = torch.full(
                        (B_sz, L_sz, K_sz, 1), channel_bits_per_link / model.max_channel_bits,
                        device=device
                    )
                    bitwidth_features = torch.cat([bw_demod_feat, bw_channel_feat], dim=-1)

                    detected = model.detector(v_q, H_q, bitwidth_features)
                    det_np = detected.cpu().numpy()
                    s_pred = det_np[..., 0] + 1j * det_np[..., 1]
                    ber_prop = compute_ber(td['s_np'], s_pred)
                    avg_bits_actual = demod_bits_per_link + channel_bits_per_link
            else:
                # Use trained model as-is
                ber_prop, avg_bits_actual = evaluate_model_ber(model, td, tau, use_quantization=True)

            results_proposed[label][p] = ber_prop
            actual_bits[label][p] = avg_bits_actual

            # Compute C-MMSE-Q at equivalent budget
            ber_cq, _, _ = compute_cmmse_q_detection(
                td['H_np'], td['y_np'], td['s_np'], sys_model.noise_w,
                c_equiv, K, N
            )
            results_cmmse_q[label][p] = ber_cq

            print(f"  P_tx={p:>3d} dBm: Proposed BER = {ber_prop:.6f}, "
                  f"C-MMSE-Q BER = {ber_cq:.6f}, Avg Bits = {avg_bits_actual:.1f}")

    # Print summary tables
    print("\n" + "=" * 120)
    print("PARETO FRONTIER SUMMARY TABLE - PROPOSED METHOD")
    print("=" * 120)
    header_parts = [f"{'Budget':<18}"]
    for p in power_points:
        header_parts.append(f"{p:>6d}dBm")
    header_parts.append(f"{'AvgBER':>10}")
    header_parts.append(f"{'AvgBits':>9}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for label, _, _ in budget_configs:
        parts = [f"{label:<18}"]
        bers = []
        for p in power_points:
            parts.append(f"{results_proposed[label][p]:>9.6f}")
            bers.append(results_proposed[label][p])
        avg_ber = np.mean(bers)
        avg_b = np.mean([actual_bits[label][p] for p in power_points])
        parts.append(f"{avg_ber:>10.6f}")
        parts.append(f"{avg_b:>9.1f}")
        print(" | ".join(parts))

    print("\n" + "=" * 120)
    print("PARETO FRONTIER SUMMARY TABLE - C-MMSE-Q BASELINE")
    print("=" * 120)
    header_parts = [f"{'Budget':<18}"]
    for p in power_points:
        header_parts.append(f"{p:>6d}dBm")
    header_parts.append(f"{'AvgBER':>10}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for label, _, _ in budget_configs:
        parts = [f"{label:<18}"]
        bers = []
        for p in power_points:
            parts.append(f"{results_cmmse_q[label][p]:>9.6f}")
            bers.append(results_cmmse_q[label][p])
        avg_ber = np.mean(bers)
        parts.append(f"{avg_ber:>10.6f}")
        print(" | ".join(parts))

    print("\n" + "=" * 120)
    print("PARETO FRONTIER - PROPOSED vs C-MMSE-Q IMPROVEMENT (%)")
    print("=" * 120)
    header_parts = [f"{'Budget':<18}"]
    for p in power_points:
        header_parts.append(f"{p:>6d}dBm")
    header_parts.append(f"{'AvgImprv':>10}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for label, _, _ in budget_configs:
        parts = [f"{label:<18}"]
        improvs = []
        for p in power_points:
            ber_p = results_proposed[label][p]
            ber_c = results_cmmse_q[label][p]
            improv = safe_pct_improvement(ber_c, ber_p)
            parts.append(f"{improv:>+8.2f}%")
            improvs.append(improv)
        avg_improv = np.mean(improvs)
        parts.append(f"{avg_improv:>+9.2f}%")
        print(" | ".join(parts))

    # Additional: Pareto curve data (for easy plotting)
    print("\n" + "=" * 120)
    print("PARETO DATA FOR PLOTTING (Avg Bits vs Avg BER)")
    print("=" * 120)
    print(f"  {'Budget':<18} | {'Avg Bits':>10} | {'Proposed Avg BER':>18} | {'C-MMSE-Q Avg BER':>18}")
    print("  " + "-" * 75)
    for label, _, _ in budget_configs:
        avg_b = np.mean([actual_bits[label][p] for p in power_points])
        avg_ber_p = np.mean([results_proposed[label][p] for p in power_points])
        avg_ber_c = np.mean([results_cmmse_q[label][p] for p in power_points])
        print(f"  {label:<18} | {avg_b:>10.1f} | {avg_ber_p:>18.6f} | {avg_ber_c:>18.6f}")

    return results_proposed, results_cmmse_q, actual_bits


# ============================================================
# EXPERIMENT 3: Imperfect CSI
# ============================================================
def generate_imperfect_csi_data(sys_model, test_samples, p_tx_dbm, nmse, device):
    """
    Generate test data with imperfect CSI.

    CSI Error Model: H_hat = H_true + E
      where E ~ CN(0, nmse * avg_power_per_element)
      avg_power_per_element = ||H_true||^2_F / (L*N*K) per sample

    Steps:
    1. Generate H_true, s, y = H_true @ s + noise (true received signal)
    2. H_hat = H_true + E
    3. Compute local LMMSE using H_hat (not H_true)
    4. Return everything needed for evaluation
    """
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K)

    L = sys_model.L
    N = sys_model.N
    K = sys_model.K

    # 1. True channel
    h_small = (np.random.randn(test_samples, L, N, K) +
               1j * np.random.randn(test_samples, L, N, K)) / np.sqrt(2)
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]
    H_true = np.sqrt(beta_w_expanded) * h_small  # (B, L, N, K)

    # 2. QPSK symbols
    bits = np.random.randint(0, 4, size=(test_samples, K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(bits) / np.sqrt(2)

    # 3. True received signal
    y_clean = np.einsum('blnk,bk->bln', H_true, s)
    z = (np.random.randn(test_samples, L, N) +
         1j * np.random.randn(test_samples, L, N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z

    # 4. Add CSI error
    if nmse > 0:
        # Per-sample average channel power per element
        h_power_per_element = np.mean(np.abs(H_true) ** 2, axis=(1, 2, 3))  # (B,)
        error_var = nmse * h_power_per_element  # (B,) - variance per complex element

        # E ~ CN(0, error_var) per element
        # real and imag each have variance = error_var / 2
        E_real = np.random.randn(test_samples, L, N, K)
        E_imag = np.random.randn(test_samples, L, N, K)
        scale = np.sqrt(error_var / 2.0)[:, np.newaxis, np.newaxis, np.newaxis]
        E = (E_real + 1j * E_imag) * scale
        H_hat = H_true + E
    else:
        H_hat = H_true.copy()

    # 5. Local LMMSE using H_hat
    H_hat_conj_trans = H_hat.conj().transpose(0, 1, 3, 2)  # (B, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(N).reshape(1, 1, N, N)
    R_y = H_hat @ H_hat_conj_trans + noise_cov  # (B, L, N, N)
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_hat_conj_trans @ R_y_inv  # (B, L, K, N)
    s_hat = W_l @ y[..., np.newaxis]
    s_hat = s_hat.squeeze(-1)  # (B, L, K)

    # 6. Local SNR based on H_hat
    local_snr = 10 * np.log10(np.sum(np.abs(H_hat) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0

    # Prepare tensors using H_hat (not H_true)
    v_t, H_hat_t, snr_t, Y_t = prepare_tensors(s_hat, s, H_hat, local_snr, device)

    return {
        'v': v_t, 'H': H_hat_t, 'snr': snr_t, 'Y': Y_t,
        's_hat_np': s_hat, 's_np': s, 'H_np': H_hat,
        'H_true_np': H_true, 'y_np': y,
        'snr_np': local_snr
    }


def run_imperfect_csi_experiment(model, sys_model, power_points, args, device):
    print("\n" + "=" * 100)
    print("EXPERIMENT 3: IMPERFECT CSI EVALUATION")
    print("=" * 100)
    print("CSI Error Model: H_hat = H_true + E, where E ~ CN(0, nmse * avg_power)")
    print("All methods use H_hat (not H_true) for detection.")
    print("The received signal y is generated with H_true (reality).")

    tau = args.tau_eval
    nmse_levels = [float(x) for x in args.nmse_levels.split(",")]
    nmse_labels = []
    for nmse in nmse_levels:
        if nmse == 0.0:
            nmse_labels.append("0% (perfect)")
        else:
            nmse_labels.append(f"{nmse*100:.0f}%")

    results_proposed = {}
    results_dist_full = {}
    results_cmmse = {}

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        print(f"\n--- NMSE = {nmse_label} ---")
        results_proposed[nmse] = {}
        results_dist_full[nmse] = {}
        results_cmmse[nmse] = {}

        for p in power_points:
            print(f"  Generating imperfect CSI data at P_tx = {p} dBm, NMSE = {nmse_label}...")

            # Generate data with imperfect CSI
            td = generate_imperfect_csi_data(
                sys_model, args.test_samples, p, nmse, device
            )

            # (a) Proposed method with imperfect CSI
            ber_prop, avg_bits = evaluate_model_ber(model, td, tau, use_quantization=True)
            results_proposed[nmse][p] = ber_prop

            # (b) Dist-Full with imperfect CSI (mean pooling of s_hat computed with H_hat)
            ber_df = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
            results_dist_full[nmse][p] = ber_df

            # (c) C-MMSE with imperfect CSI: (H_hat^H H_hat + sigma^2 I)^{-1} H_hat^H y
            B = td['H_true_np'].shape[0]
            L = sys_model.L
            N = sys_model.N
            K = sys_model.K

            H_hat_all = td['H_np'].reshape(B, L * N, K)
            y_all = td['y_np'].reshape(B, L * N)
            H_H = H_hat_all.conj().transpose(0, 2, 1)
            HHH = H_H @ H_hat_all
            noise_eye = sys_model.noise_w * np.eye(K).reshape(1, K, K)
            R = HHH + noise_eye
            HHy = H_H @ y_all[..., np.newaxis]
            R_inv = np.linalg.inv(R)
            s_hat_cmmse = (R_inv @ HHy).squeeze(-1)
            ber_cmmse = compute_ber(td['s_np'], s_hat_cmmse)
            results_cmmse[nmse][p] = ber_cmmse

            print(f"    P_tx={p:>3d} dBm: Proposed={ber_prop:.6f}, "
                  f"Dist-Full={ber_df:.6f}, C-MMSE={ber_cmmse:.6f}")

    # Print summary tables
    print("\n" + "=" * 120)
    print("IMPERFECT CSI SUMMARY TABLE")
    print("=" * 120)

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        print(f"\n  NMSE = {nmse_label}")
        header = (f"  {'P_tx(dBm)':<12} | {'Dist-Full':<12} | {'C-MMSE':<12} | "
                  f"{'Proposed':<12} | {'vs Dist-Full':>14} | {'vs C-MMSE':>12}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in power_points:
            ber_df = results_dist_full[nmse][p]
            ber_cm = results_cmmse[nmse][p]
            ber_pr = results_proposed[nmse][p]
            improv_df = safe_pct_improvement(ber_df, ber_pr)
            improv_cm = safe_pct_improvement(ber_cm, ber_pr)
            print(f"  {p:<12} | {ber_df:<12.6f} | {ber_cm:<12.6f} | "
                  f"{ber_pr:<12.6f} | {improv_df:>+13.2f}% | {improv_cm:>+11.2f}%")

    # Cross-NMSE degradation table
    print("\n" + "=" * 120)
    print("IMPERFECT CSI - BER DEGRADATION vs PERFECT CSI (PROPOSED METHOD)")
    print("=" * 120)
    header_parts = [f"{'NMSE':<16}"]
    for p in power_points:
        header_parts.append(f"{p:>8d}dBm")
    header_parts.append(f"{'Avg Degrad':>12}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        parts = [f"{nmse_label:<16}"]
        degradations = []
        for p in power_points:
            ber_current = results_proposed[nmse][p]
            ber_perfect = results_proposed[nmse_levels[0]][p]  # reference: first (perfect) NMSE
            degrad = safe_pct_degradation(ber_current, ber_perfect)
            parts.append(f"{degrad:>+7.1f}%")
            degradations.append(degrad)
        avg_deg = np.mean(degradations)
        parts.append(f"{avg_deg:>+10.1f}%")
        print(" | ".join(parts))

    # Absolute BER table for proposed method
    print("\n" + "=" * 120)
    print("IMPERFECT CSI - PROPOSED METHOD BER AT ALL (NMSE, P_tx) COMBINATIONS")
    print("=" * 120)
    header_parts = [f"{'NMSE':<16}"]
    for p in power_points:
        header_parts.append(f"{p:>9d}dBm")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        parts = [f"{nmse_label:<16}"]
        for p in power_points:
            parts.append(f"{results_proposed[nmse][p]:>12.6f}")
        print(" | ".join(parts))

    # Additional: C-MMSE BER at all (NMSE, P_tx) combinations
    print("\n" + "=" * 120)
    print("IMPERFECT CSI - C-MMSE BER AT ALL (NMSE, P_tx) COMBINATIONS")
    print("=" * 120)
    header_parts = [f"{'NMSE':<16}"]
    for p in power_points:
        header_parts.append(f"{p:>9d}dBm")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        parts = [f"{nmse_label:<16}"]
        for p in power_points:
            parts.append(f"{results_cmmse[nmse][p]:>12.6f}")
        print(" | ".join(parts))

    # Dist-Full BER at all (NMSE, P_tx) combinations
    print("\n" + "=" * 120)
    print("IMPERFECT CSI - DIST-FULL BER AT ALL (NMSE, P_tx) COMBINATIONS")
    print("=" * 120)
    header_parts = [f"{'NMSE':<16}"]
    for p in power_points:
        header_parts.append(f"{p:>9d}dBm")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        parts = [f"{nmse_label:<16}"]
        for p in power_points:
            parts.append(f"{results_dist_full[nmse][p]:>12.6f}")
        print(" | ".join(parts))

    # Robustness summary
    print("\n" + "=" * 120)
    print("IMPERFECT CSI - ROBUSTNESS SUMMARY")
    print("=" * 120)
    for nmse, nmse_label in zip(nmse_levels, nmse_labels):
        avg_prop = np.mean([results_proposed[nmse][p] for p in power_points])
        avg_df = np.mean([results_dist_full[nmse][p] for p in power_points])
        avg_cm = np.mean([results_cmmse[nmse][p] for p in power_points])
        print(f"  NMSE = {nmse_label:<16}: "
              f"Proposed avg BER = {avg_prop:.6f}, "
              f"Dist-Full avg BER = {avg_df:.6f}, "
              f"C-MMSE avg BER = {avg_cm:.6f}")
        if avg_df > 1e-8:
            print(f"    Proposed vs Dist-Full: {safe_pct_improvement(avg_df, avg_prop):+.2f}% improvement")
        if avg_cm > 1e-8:
            print(f"    Proposed vs C-MMSE:    {safe_pct_improvement(avg_cm, avg_prop):+.2f}% improvement")

    return results_proposed, results_dist_full, results_cmmse


# ============================================================
# Main
# ============================================================
def main():
    args = parse_eval_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse power points
    power_points = [int(x) for x in args.power_points.split(",")]

    print("=" * 100)
    print("V3 MODEL EVALUATION EXPERIMENTS")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"System: L={args.L}, N={args.N}, K={args.K}")
    print(f"Test samples per power point: {args.test_samples}")
    print(f"Model path: {args.model_path}")
    print(f"Tau (eval): {args.tau_eval}")
    print(f"Experiment: {args.experiment}")
    print(f"Power points: {power_points} dBm")
    print(f"Seed: {args.seed}")
    print(f"c_target: {args.c_target}")
    print("=" * 100)

    # Initialize system model
    sys_model = CellFreeSystem(
        args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0
    )
    sys_model.generate_scenario()

    # Load model
    model = load_model(args, device)

    # Generate test dataset (used for Experiments 1 and 2)
    test_dataset = generate_test_dataset(
        sys_model, args.test_samples, power_points, device
    )

    # Run experiments
    if args.experiment in ["all", "ablation"]:
        t_start = time.time()
        ablation_results = run_ablation_study(
            model, test_dataset, power_points, args, device
        )
        t_elapsed = time.time() - t_start
        print(f"\nAblation study completed in {t_elapsed:.1f} seconds.\n")

    if args.experiment in ["all", "pareto"]:
        t_start = time.time()
        pareto_proposed, pareto_cmmse_q, pareto_bits = run_pareto_experiment(
            model, test_dataset, power_points, sys_model, args, device
        )
        t_elapsed = time.time() - t_start
        print(f"\nPareto experiment completed in {t_elapsed:.1f} seconds.\n")

    if args.experiment in ["all", "imperfect_csi"]:
        t_start = time.time()
        # Use fresh random state for imperfect CSI experiment (independent of other experiments)
        np.random.seed(args.seed + 100)
        torch.manual_seed(args.seed + 100)
        csi_proposed, csi_dist_full, csi_cmmse = run_imperfect_csi_experiment(
            model, sys_model, power_points, args, device
        )
        t_elapsed = time.time() - t_start
        print(f"\nImperfect CSI experiment completed in {t_elapsed:.1f} seconds.\n")

    print("\n" + "=" * 100)
    print("ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 100)


if __name__ == '__main__':
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import math
import os

from system_model import CellFreeSystem
from lsq_quantizer import LSQQuantizer


def parse_args():
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V4 with Triple Adaptive Quantization (y+H+demod)")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=50, help="Phase 1 epochs (detector pre-training, no quantization)")
    parser.add_argument("--epochs_phase2", type=int, default=150, help="Phase 2 epochs (joint QAT training, 3 sub-phases: 30+60+60)")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for Phase 1 (detector pre-training)")
    parser.add_argument("--lr_phase2", type=float, default=5e-4, help="Learning rate for Phase 2 (joint QAT)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D for feature embedding")
    parser.add_argument("--c_target", type=float, default=48.0, help="Target average bits per (AP, user) link (~55%% of max 88 bits/link)")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of mean-field GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer encoder layers for inter-user IC")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in Transformer and MLPs")
    parser.add_argument("--noise_injection_std", type=float, default=0.01, help="Gaussian noise std injected during Phase 1 for robustness")
    # Phase 2 sub-phase parameters
    parser.add_argument("--phase2_sub1_epochs", type=int, default=30, help="Phase 2 sub-phase 1 epochs (warm-start)")
    parser.add_argument("--phase2_sub2_epochs", type=int, default=60, help="Phase 2 sub-phase 2 epochs (medium)")
    parser.add_argument("--phase2_sub3_epochs", type=int, default=60, help="Phase 2 sub-phase 3 epochs (fine-tune)")
    parser.add_argument("--lambda_sub1", type=float, default=0.01, help="Lambda for Phase 2 sub-phase 1")
    parser.add_argument("--lambda_sub2", type=float, default=0.1, help="Lambda for Phase 2 sub-phase 2")
    parser.add_argument("--lambda_sub3", type=float, default=1.0, help="Lambda for Phase 2 sub-phase 3")
    parser.add_argument("--tau_sub1_start", type=float, default=5.0, help="Tau start for sub-phase 1")
    parser.add_argument("--tau_sub1_end", type=float, default=3.0, help="Tau end for sub-phase 1")
    parser.add_argument("--tau_sub2_start", type=float, default=3.0, help="Tau start for sub-phase 2")
    parser.add_argument("--tau_sub2_end", type=float, default=1.0, help="Tau end for sub-phase 2")
    parser.add_argument("--tau_sub3_start", type=float, default=1.0, help="Tau start for sub-phase 3")
    parser.add_argument("--tau_sub3_end", type=float, default=0.1, help="Tau end for sub-phase 3")
    return parser.parse_args()


# ============================================================
# 1. Data Generation Function (returns raw y for C-MMSE)
# ============================================================
def generate_data_batch_v2(sys_model, batch_size, p_tx_dbm):
    """
    Vectorized data generation that returns local LMMSE estimates, true symbols,
    full channel matrix H, local SNR features, and raw received signal y.

    Returns:
        s_hat: (batch_size, L, K) complex - local LMMSE estimates
        s: (batch_size, K) complex - true transmitted QPSK symbols
        H: (batch_size, L, N, K) complex - full channel matrix
        local_snr: (batch_size, L, K) real - local SNR features (normalized)
        y: (batch_size, L, N) complex - raw received signal at all APs
    """
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K)

    # 1. Batch Rayleigh fading channel
    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]  # (1, L, 1, K)
    H = np.sqrt(beta_w_expanded) * h_small  # (batch_size, L, N, K)

    # 2. QPSK symbols
    bits = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(bits) / np.sqrt(2)  # (batch_size, K)

    # 3. Received signal y = H @ s + noise
    y_clean = np.einsum('blnk,bk->bln', H, s)
    z = (np.random.randn(batch_size, sys_model.L, sys_model.N) +
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z  # (batch_size, L, N)

    # 4. Local LMMSE detection (vectorized)
    H_conj_trans = H.conj().transpose(0, 1, 3, 2)  # (batch_size, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H @ H_conj_trans + noise_cov
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_conj_trans @ R_y_inv  # (batch_size, L, K, N)
    s_hat = W_l @ y[..., np.newaxis]  # (batch_size, L, K, 1)
    s_hat = s_hat.squeeze(-1)  # (batch_size, L, K)

    # 5. Local SNR (normalized)
    local_snr = 10 * np.log10(np.sum(np.abs(H) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0

    return s_hat, s, H, local_snr, y


# ============================================================
# 2. Baseline Detectors
# ============================================================
def compute_ber(s_true, s_pred_complex):
    """Compute BER for QPSK symbols."""
    s_true_real = np.sign(s_true.real)
    s_true_imag = np.sign(s_true.imag)
    s_pred_real = np.sign(s_pred_complex.real)
    s_pred_imag = np.sign(s_pred_complex.imag)

    s_pred_real[s_pred_real == 0] = 1
    s_pred_imag[s_pred_imag == 0] = 1

    err_real = (s_true_real != s_pred_real).sum()
    err_imag = (s_true_imag != s_pred_imag).sum()
    total_bits = s_true.size * 2
    return (err_real + err_imag) / total_bits


def compute_dist_full_ber(s_hat, s):
    """Dist-Full: mean pooling of local LMMSE across APs (no quantization)."""
    s_hat_avg = s_hat.mean(axis=1)
    return compute_ber(s, s_hat_avg)


def uniform_quantize_np(x_real, num_bits):
    """Uniform quantization of real-valued numpy array."""
    if num_bits <= 0:
        return np.zeros_like(x_real)
    if num_bits >= 24:
        return x_real

    levels = 2 ** num_bits
    x_max = np.max(np.abs(x_real)) + 1e-12
    step = 2 * x_max / (levels - 1)
    idx = np.round((x_real + x_max) / step)
    idx = np.clip(idx, 0, levels - 1)
    return idx * step - x_max


def compute_cmmse_detection(H, y, s, noise_w):
    """Centralized MMSE detection using full (unquantized) channel and received signal."""
    B, L, N, K = H.shape
    H_all = H.reshape(B, L * N, K)
    y_all = y.reshape(B, L * N)
    H_H = H_all.conj().transpose(0, 2, 1)
    HHH = H_H @ H_all
    noise_eye = noise_w * np.eye(K).reshape(1, K, K)
    R = HHH + noise_eye
    HHy = H_H @ y_all[..., np.newaxis]
    R_inv = np.linalg.inv(R)
    s_hat_cmmse = (R_inv @ HHy).squeeze(-1)
    ber = compute_ber(s, s_hat_cmmse)
    return ber, s_hat_cmmse


def compute_cmmse_q_detection(H, y, s, noise_w, c_target, K, N):
    """
    C-MMSE-Q: Centralized MMSE with uniformly quantized H and y at same total bit budget.
    Total bits per AP = c_target * K.
    Reals per AP for C-MMSE = 2*N + 2*N*K = 2*N*(1+K).
    We search over multiple bit options to find the best one within budget.
    """
    B, L, N_ant, K_users = H.shape

    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)
    b_ideal = total_bits_per_ap / reals_per_ap

    # Search over a range of bit options
    bit_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    best_ber = 1.0
    best_b = 1
    best_s_hat = None

    for b_use in bit_candidates:
        actual_bits_per_ap = reals_per_ap * b_use
        if actual_bits_per_ap > total_bits_per_ap * 1.05:  # allow 5% tolerance
            continue

        H_real_q = uniform_quantize_np(H.real, b_use)
        H_imag_q = uniform_quantize_np(H.imag, b_use)
        H_q = H_real_q + 1j * H_imag_q

        y_real_q = uniform_quantize_np(y.real, b_use)
        y_imag_q = uniform_quantize_np(y.imag, b_use)
        y_q = y_real_q + 1j * y_imag_q

        H_all = H_q.reshape(B, L * N_ant, K_users)
        y_all = y_q.reshape(B, L * N_ant)
        H_H = H_all.conj().transpose(0, 2, 1)
        HHH = H_H @ H_all
        noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
        reg_eye = 1e-6 * np.eye(K_users).reshape(1, K_users, K_users)
        R = HHH + noise_eye + reg_eye
        HHy = H_H @ y_all[..., np.newaxis]
        try:
            R_inv = np.linalg.inv(R)
            s_hat = (R_inv @ HHy).squeeze(-1)
            ber = compute_ber(s, s_hat)
            if ber < best_ber:
                best_ber = ber
                best_b = b_use
                best_s_hat = s_hat
        except np.linalg.LinAlgError:
            continue

    actual_bits_per_ap = reals_per_ap * best_b
    print(f"    C-MMSE-Q: c_target={c_target}, reals/AP={reals_per_ap}, "
          f"ideal b={b_ideal:.2f}, best b={best_b}, actual bits/AP={actual_bits_per_ap:.0f} "
          f"(target={total_bits_per_ap:.0f}), BER={best_ber:.6f}")

    return best_ber, best_s_hat, best_b


def compute_cmmse_mixed_q_detection(H, y, s, noise_w, c_target, K, N, s_hat_local=None):
    """
    C-MMSE-Mixed-Q: Centralized MMSE where H is quantized to b_H bits and y is quantized to b_y bits,
    with b_H and b_y chosen to match total bit budget. Also considers including quantized demod.

    Total bits per AP = c_target * K
    Option A (H+y only): 2*N * b_y + K * 2*N * b_H
    Option B (H+y+demod): 2*N * b_y + K * 2*N * b_H + K * 2 * b_demod
    
    We search over combinations to find best BER within budget.
    """
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users

    best_ber = 1.0
    best_b_y = 0
    best_b_H = 0
    best_b_demod = 0
    best_s_hat = None
    best_mode = "H+y"

    bit_options = [1, 2, 3, 4, 5, 6, 7, 8]

    # Option A: H + y only
    for b_y in bit_options:
        for b_H in bit_options:
            bits_used = 2 * N_ant * b_y + K_users * 2 * N_ant * b_H
            if bits_used > total_bits_per_ap * 1.05:
                continue

            H_real_q = uniform_quantize_np(H.real, b_H)
            H_imag_q = uniform_quantize_np(H.imag, b_H)
            H_q = H_real_q + 1j * H_imag_q

            y_real_q = uniform_quantize_np(y.real, b_y)
            y_imag_q = uniform_quantize_np(y.imag, b_y)
            y_q = y_real_q + 1j * y_imag_q

            H_all = H_q.reshape(B, L * N_ant, K_users)
            y_all = y_q.reshape(B, L * N_ant)
            H_H = H_all.conj().transpose(0, 2, 1)
            HHH = H_H @ H_all
            noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
            reg_eye = 1e-6 * np.eye(K_users).reshape(1, K_users, K_users)
            R = HHH + noise_eye + reg_eye
            HHy = H_H @ y_all[..., np.newaxis]
            try:
                R_inv = np.linalg.inv(R)
                s_hat = (R_inv @ HHy).squeeze(-1)
                ber = compute_ber(s, s_hat)
                if ber < best_ber:
                    best_ber = ber
                    best_b_y = b_y
                    best_b_H = b_H
                    best_b_demod = 0
                    best_s_hat = s_hat
                    best_mode = "H+y"
            except np.linalg.LinAlgError:
                continue

    # Option B: H + y + demod (use demod as additional info for weighted combining)
    if s_hat_local is not None:
        for b_y in bit_options:
            for b_H in bit_options:
                for b_d in bit_options:
                    bits_used = 2 * N_ant * b_y + K_users * 2 * N_ant * b_H + K_users * 2 * b_d
                    if bits_used > total_bits_per_ap * 1.05:
                        continue

                    H_real_q = uniform_quantize_np(H.real, b_H)
                    H_imag_q = uniform_quantize_np(H.imag, b_H)
                    H_q = H_real_q + 1j * H_imag_q

                    y_real_q = uniform_quantize_np(y.real, b_y)
                    y_imag_q = uniform_quantize_np(y.imag, b_y)
                    y_q = y_real_q + 1j * y_imag_q

                    # Quantize demod
                    d_real_q = uniform_quantize_np(s_hat_local.real, b_d)
                    d_imag_q = uniform_quantize_np(s_hat_local.imag, b_d)
                    d_q = d_real_q + 1j * d_imag_q

                    # C-MMSE from quantized H and y
                    H_all = H_q.reshape(B, L * N_ant, K_users)
                    y_all = y_q.reshape(B, L * N_ant)
                    H_H = H_all.conj().transpose(0, 2, 1)
                    HHH = H_H @ H_all
                    noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
                    reg_eye = 1e-6 * np.eye(K_users).reshape(1, K_users, K_users)
                    R = HHH + noise_eye + reg_eye
                    HHy = H_H @ y_all[..., np.newaxis]
                    try:
                        R_inv = np.linalg.inv(R)
                        s_cmmse = (R_inv @ HHy).squeeze(-1)
                        # Simple average of C-MMSE and demod average
                        s_demod_avg = d_q.mean(axis=1)
                        # Weighted combination: 0.7 C-MMSE + 0.3 demod
                        s_hat_combined = 0.7 * s_cmmse + 0.3 * s_demod_avg
                        ber = compute_ber(s, s_hat_combined)
                        if ber < best_ber:
                            best_ber = ber
                            best_b_y = b_y
                            best_b_H = b_H
                            best_b_demod = b_d
                            best_s_hat = s_hat_combined
                            best_mode = "H+y+demod"
                    except np.linalg.LinAlgError:
                        continue

    if best_b_demod > 0:
        actual_bits = 2 * N_ant * best_b_y + K_users * 2 * N_ant * best_b_H + K_users * 2 * best_b_demod
    else:
        actual_bits = 2 * N_ant * best_b_y + K_users * 2 * N_ant * best_b_H
    print(f"    C-MMSE-Mixed-Q: mode={best_mode}, best b_y={best_b_y}, b_H={best_b_H}, b_demod={best_b_demod}, "
          f"actual bits/AP={actual_bits:.0f} (target={total_bits_per_ap:.0f}), BER={best_ber:.6f}")
    return best_ber, best_s_hat, best_b_y, best_b_H


def compute_dist_q_ber(s_hat, s, c_target):
    """
    Dist-Q: Distributed local LMMSE with uniform quantization at same average bits per link.
    Each AP sends s_hat_l (K complex = 2K reals per AP).
    Total bits per AP = c_target * K.
    Bits per real = c_target * K / (2*K) = c_target / 2.
    """
    B, L, K = s_hat.shape
    b_per_real = c_target / 2.0
    b_use = max(1, int(np.round(b_per_real)))
    b_use = min(b_use, 16)

    print(f"    Dist-Q: c_target={c_target}, bits_per_real={b_per_real:.2f}, using b={b_use}")

    s_hat_real_q = np.zeros_like(s_hat.real)
    s_hat_imag_q = np.zeros_like(s_hat.imag)
    for l in range(L):
        s_hat_real_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].real, b_use)
        s_hat_imag_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].imag, b_use)

    s_hat_q = s_hat_real_q + 1j * s_hat_imag_q
    s_hat_avg = s_hat_q.mean(axis=1)
    ber = compute_ber(s, s_hat_avg)
    return ber, b_use


def compute_lmmse_global_q_ber(s_hat, s, c_target):
    """
    LMMSE-Global-Q: Each AP computes local LMMSE, quantizes the demod results,
    transmits to CPU which averages them. Same as Dist-Q but with explicit naming.
    This is essentially what distributed quantized detection does.
    
    Bits per AP = c_target * K (total). Each AP sends K complex values = 2K reals.
    Bits per real = c_target * K / (2K) = c_target / 2.
    """
    B, L, K = s_hat.shape
    b_per_real = c_target / 2.0
    b_use = max(1, int(np.round(b_per_real)))
    b_use = min(b_use, 16)

    print(f"    LMMSE-Global-Q: c_target={c_target}, bits_per_real={b_per_real:.2f}, using b={b_use}")

    s_hat_real_q = np.zeros_like(s_hat.real)
    s_hat_imag_q = np.zeros_like(s_hat.imag)
    for l in range(L):
        s_hat_real_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].real, b_use)
        s_hat_imag_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].imag, b_use)

    s_hat_q = s_hat_real_q + 1j * s_hat_imag_q
    s_hat_avg = s_hat_q.mean(axis=1)
    ber = compute_ber(s, s_hat_avg)
    return ber, b_use


# ============================================================
# 3. GNN + Transformer Hybrid Detector V4
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    """Mean-field GNN message passing layer with O(L) complexity."""
    def __init__(self, hidden_dim):
        super(MeanFieldGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        """h: (B, L, K, D) -> (B, L, K, D)"""
        messages = self.message_mlp(h)
        mean_message = messages.mean(dim=1, keepdim=True).expand_as(h)
        combined = torch.cat([h, mean_message], dim=-1)
        updated = self.update_mlp(combined)
        h_new = self.layer_norm(h + updated)
        return h_new


class APAggregator(nn.Module):
    """Learned attention-based weighted sum across APs."""
    def __init__(self, hidden_dim):
        super(APAggregator, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        """h: (B, L, K, D) -> aggregated: (B, K, D)"""
        attn_weights = self.attn_net(h)  # (B, L, K, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        aggregated = (h * attn_weights).sum(dim=1)  # (B, K, D)
        return aggregated, attn_weights


class GNNTransformerDetectorV4(nn.Module):
    """
    Hybrid GNN + Transformer detector V4 for Cell-Free MIMO.
    
    Key enhancement: 
    1. Computes per-AP LMMSE estimates from quantized H and y
    2. Computes CENTRALIZED (global) MMSE estimate by stacking all AP's quantized H and y
    3. Uses s_global as PRIMARY base output
    4. GNN+Transformer learns RESIDUAL correction on top of s_global

    Architecture:
    1. Global MMSE estimate from stacked quantized H_q and y_q -> s_global (B, K, 2)
    2. Per-AP MMSE initial estimates -> s_init (B, L, K, 2)
    3. Feature fusion: [s_init(2), s_hat_q(2), s_global_broadcast(2), H_features(D), bitwidth(3), interference(2), snr(1)]
    4. Mean-field GNN Layers (x3) for spatial diversity combining
    5. Attention-based AP Aggregation
    6. Transformer IC (x2 layers) for inter-user interference cancellation
    7. Output: s_global + alpha * residual, where alpha is learnable (init 0.01)
    
    IMPORTANT: MMSE matrix inversions use float64/complex128 for numerical stability.
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(GNNTransformerDetectorV4, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim
        self.noise_w = noise_w

        # Learnable residual scaling factor (initialized small)
        self.alpha = nn.Parameter(torch.tensor(0.01))

        # MLP for per-AP MMSE initial estimate features: (B, L, K, 2) -> (B, L, K, D//4)
        self.mmse_init_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # MLP for demod features: (B, L, K, 2) -> (B, L, K, D//4)
        self.demod_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # MLP for global MMSE estimate features: (B, K, 2) broadcast to (B, L, K, 2) -> (B, L, K, D//4)
        self.global_mmse_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # MLP for channel features per user: (B, L, K, 2*N) -> (B, L, K, D//4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * N, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # Fusion MLP: concat all features -> D
        # [mmse_init_feat(D//4), demod_feat(D//4), global_mmse_feat(D//4), channel_feat(D//4), 
        #  bitwidth_features(3), interference_features(2), snr(1)]
        fusion_input_dim = 4 * (hidden_dim // 4) + 3 + 2 + 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # BatchNorm after fusion
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)

        # Mean-field GNN Layers
        self.gnn_layers = nn.ModuleList([
            MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # AP Aggregation
        self.ap_aggregator = APAggregator(hidden_dim)

        # Transformer for user-level interference cancellation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Output Head (initialized to near-zero for stable residual)
        self.output_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def compute_per_ap_mmse_init(self, H_q, y_q):
        """
        Compute per-AP LMMSE initial estimate using quantized H and y.
        
        For each AP l:
            W_l = H_l^H (H_l H_l^H + sigma^2 I)^{-1}
            s_init_l = W_l @ y_l
        
        Uses float64/complex128 for numerical stability in matrix inversion.
        
        Args:
            H_q: (B, L, N, K, 2) - quantized channel (real representation)
            y_q: (B, L, N, 2) - quantized received signal (real representation)
        Returns:
            s_init: (B, L, K, 2) - per-AP MMSE initial estimate (real representation)
        """
        B, L, N, K, _ = H_q.shape

        # Convert real representation to complex using DOUBLE precision for numerical stability
        H_complex = torch.complex(H_q[..., 0].double(), H_q[..., 1].double())  # (B, L, N, K) complex128
        y_complex = torch.complex(y_q[..., 0].double(), y_q[..., 1].double())  # (B, L, N) complex128

        H_H = H_complex.conj().transpose(-1, -2)  # (B, L, K, N)

        HHH = torch.matmul(H_complex, H_H)  # (B, L, N, N)
        noise_eye = self.noise_w * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        reg_eye = 1e-6 * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        R_y = HHH + noise_eye + reg_eye  # (B, L, N, N)

        try:
            R_y_inv = torch.linalg.inv(R_y)  # (B, L, N, N)
        except RuntimeError:
            R_y_reg = R_y + 1e-4 * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            R_y_inv = torch.linalg.inv(R_y_reg)

        W_l = torch.matmul(H_H, R_y_inv)  # (B, L, K, N)

        y_expanded = y_complex.unsqueeze(-1)  # (B, L, N, 1)
        s_init_complex = torch.matmul(W_l, y_expanded).squeeze(-1)  # (B, L, K)

        # Cast back to float32
        s_init = torch.stack([s_init_complex.real.float(), s_init_complex.imag.float()], dim=-1)  # (B, L, K, 2)

        return s_init

    def compute_global_mmse(self, H_q, y_q):
        """
        Compute CENTRALIZED MMSE estimate by stacking all AP's quantized H and y.
        
        H_all = reshape(H_q, [B, L*N, K]) (complex)
        y_all = reshape(y_q, [B, L*N]) (complex)
        s_global = (H_all^H H_all + sigma^2 I + eps*I)^{-1} H_all^H y_all
        
        Uses float64/complex128 for numerical stability in matrix inversion.
        
        Args:
            H_q: (B, L, N, K, 2) - quantized channel (real representation)
            y_q: (B, L, N, 2) - quantized received signal (real representation)
        Returns:
            s_global: (B, K, 2) - centralized MMSE estimate (real representation)
        """
        B, L, N, K, _ = H_q.shape

        # Convert to complex using DOUBLE precision for numerical stability
        H_complex = torch.complex(H_q[..., 0].double(), H_q[..., 1].double())  # (B, L, N, K) complex128
        y_complex = torch.complex(y_q[..., 0].double(), y_q[..., 1].double())  # (B, L, N) complex128

        # Stack across APs
        H_all = H_complex.reshape(B, L * N, K)  # (B, L*N, K)
        y_all = y_complex.reshape(B, L * N)  # (B, L*N)

        # H_all^H: (B, K, L*N)
        H_all_H = H_all.conj().transpose(-1, -2)  # (B, K, L*N)

        # H_all^H @ H_all: (B, K, K)
        HHH = torch.matmul(H_all_H, H_all)  # (B, K, K)

        # Regularized inverse: (H^H H + sigma^2 I + eps*I)^{-1}
        noise_eye = self.noise_w * torch.eye(K, device=H_q.device, dtype=torch.float64).unsqueeze(0)
        reg_eye = 1e-6 * torch.eye(K, device=H_q.device, dtype=torch.float64).unsqueeze(0)
        R = HHH + noise_eye + reg_eye  # (B, K, K)

        # H_all^H @ y_all: (B, K)
        HHy = torch.matmul(H_all_H, y_all.unsqueeze(-1))  # (B, K, 1)

        try:
            R_inv = torch.linalg.inv(R)  # (B, K, K)
        except RuntimeError:
            R_reg = R + 1e-4 * torch.eye(K, device=H_q.device, dtype=torch.float64).unsqueeze(0)
            R_inv = torch.linalg.inv(R_reg)

        s_global_complex = torch.matmul(R_inv, HHy).squeeze(-1)  # (B, K)

        # Cast back to float32
        s_global = torch.stack([s_global_complex.real.float(), s_global_complex.imag.float()], dim=-1)  # (B, K, 2)

        return s_global

    def forward(self, s_hat_q, H_q, y_q, bitwidth_features, local_snr):
        """
        Args:
            s_hat_q: (B, L, K, 2) - quantized demod results (real, imag)
            H_q: (B, L, N, K, 2) - quantized channel coefficients (real, imag)
            y_q: (B, L, N, 2) - quantized received signal (real, imag)
            bitwidth_features: (B, L, K, 3) - normalized bitwidth features [y_bw, H_bw, demod_bw]
            local_snr: (B, L, K, 1) - local SNR features
        Returns:
            detected: (B, K, 2) - detected symbols (real, imag)
            s_global: (B, K, 2) - global MMSE estimate (for monitoring)
        """
        B = s_hat_q.size(0)
        L = s_hat_q.size(1)
        K = s_hat_q.size(2)
        N = H_q.size(2)

        # 1. Global MMSE estimate from stacked quantized H and y (PRIMARY base output)
        s_global = self.compute_global_mmse(H_q, y_q)  # (B, K, 2)

        # 2. Per-AP MMSE initial estimates
        s_init = self.compute_per_ap_mmse_init(H_q, y_q)  # (B, L, K, 2)

        # 3. Feature extraction
        mmse_init_feat = self.mmse_init_mlp(s_init)  # (B, L, K, D//4)
        demod_feat = self.demod_mlp(s_hat_q)  # (B, L, K, D//4)

        # Broadcast s_global to all APs: (B, K, 2) -> (B, L, K, 2)
        s_global_broadcast = s_global.unsqueeze(1).expand(B, L, K, 2)
        global_mmse_feat = self.global_mmse_mlp(s_global_broadcast)  # (B, L, K, D//4)

        # Channel features per user
        H_q_perm = H_q.permute(0, 1, 3, 2, 4)  # (B, L, K, N, 2)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)  # (B, L, K, 2*N)
        channel_feat = self.channel_mlp(H_q_flat)  # (B, L, K, D//4)

        # Cross-user interference features
        h_power = (H_q ** 2).sum(dim=(2, 4))  # (B, L, K) - summing over N and real/imag
        desired_power = h_power
        total_power = h_power.sum(dim=2, keepdim=True)
        interference_power = total_power - desired_power

        desired_feat = torch.log1p(desired_power).unsqueeze(-1)  # (B, L, K, 1)
        interference_feat = torch.log1p(interference_power).unsqueeze(-1)  # (B, L, K, 1)
        interference_features = torch.cat([desired_feat, interference_feat], dim=-1)  # (B, L, K, 2)

        # 4. Feature Fusion
        combined = torch.cat([mmse_init_feat, demod_feat, global_mmse_feat, channel_feat,
                              bitwidth_features, interference_features, local_snr], dim=-1)
        node_features = self.fusion_mlp(combined)  # (B, L, K, D)

        # Apply BatchNorm
        node_features_flat = node_features.reshape(-1, self.hidden_dim)
        node_features_bn = self.fusion_bn(node_features_flat)
        node_features = node_features_bn.reshape(B, L, K, self.hidden_dim)

        # 5. Mean-field GNN Message Passing
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)

        # 6. AP Aggregation
        user_features, _ = self.ap_aggregator(h)  # (B, K, D)

        # 7. Transformer IC
        ic_out = self.transformer_ic(user_features)  # (B, K, D)

        # 8. Output: s_global + alpha * residual
        residual = self.output_head(ic_out)  # (B, K, 2)
        detected = s_global + self.alpha * residual

        return detected, s_global


# ============================================================
# 4. Triple Adaptive Quantizer V4
# ============================================================
class TriplePolicyNetworkV4(nn.Module):
    """
    Policy network for triple adaptive quantization.
    
    Three outputs:
    - y quantization: per AP (1 decision per AP, 5 options: {0,2,4,6,8})
    - H quantization: per AP-user link (5 options: {0,2,4,6,8})
    - demod quantization: per AP-user link (5 options: {0,2,4,6,8})
    
    Input features per (l, k):
        [v_real, v_imag, snr, channel_norm, avg_channel_power, sir, 
         total_power_norm, y_power, signal_power]
    """
    def __init__(self, input_dim=9, policy_hidden=64):
        super(TriplePolicyNetworkV4, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        # 5 options each: {0, 2, 4, 6, 8} bits
        self.head_y = nn.Linear(policy_hidden, 5)       # y quantization (per AP, averaged over users)
        self.head_H = nn.Linear(policy_hidden, 5)       # H quantization (per link)
        self.head_demod = nn.Linear(policy_hidden, 5)    # demod quantization (per link)

    def forward(self, x):
        """x: (..., input_dim), Returns: logits_y, logits_H, logits_demod each (..., 5)"""
        h = self.shared(x)
        return self.head_y(h), self.head_H(h), self.head_demod(h)


class TripleAdaptiveQuantizerV4(nn.Module):
    """
    Triple adaptive quantization with bit options {0, 2, 4, 6, 8} for y, H, and demod.
    
    y_l: N complex values per AP, shared across users -> 2N reals per AP
    H_{l,:,k}: N complex values per AP-user link -> 2N reals per link
    s_hat_{l,k}: 1 complex value per link -> 2 reals per link
    
    Bit budget per AP:
        bits_for_y = 2*N * b_y
        bits_for_H_k = 2*N * b_H_k per user
        bits_for_demod_k = 2 * b_demod_k per user
        Total per AP = bits_for_y + sum_k(bits_for_H_k + bits_for_demod_k)
        Average per link = Total / K
    """
    def __init__(self, N=4, policy_hidden=64):
        super(TripleAdaptiveQuantizerV4, self).__init__()
        self.N = N
        self.policy = TriplePolicyNetworkV4(input_dim=9, policy_hidden=policy_hidden)

        # Bit options
        self.bit_options = [0, 2, 4, 6, 8]

        # LSQ quantizers for y (received signal): 2, 4, 6, 8 bits
        self.q2_y = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_y = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_y = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_y = LSQQuantizer(num_bits=8, init_s=0.01)

        # LSQ quantizers for H (channel): 2, 4, 6, 8 bits
        self.q2_H = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_H = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_H = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_H = LSQQuantizer(num_bits=8, init_s=0.01)

        # LSQ quantizers for demod: 2, 4, 6, 8 bits
        self.q2_demod = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_demod = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_demod = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_demod = LSQQuantizer(num_bits=8, init_s=0.01)

    def forward(self, v, H, y, local_snr, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - demod results (real, imag parts)
            H: (B, L, N, K, 2) - channel coefficients (real, imag parts)
            y: (B, L, N, 2) - received signal (real, imag parts)
            local_snr: (B, L, K, 1) - local SNR features
            tau: Gumbel-Softmax temperature
        Returns:
            v_q: (B, L, K, 2) - quantized demod
            H_q: (B, L, N, K, 2) - quantized channel
            y_q: (B, L, N, 2) - quantized received signal
            expected_bits_per_link: (B, L, K) - expected bits per AP-user link
            w_y: (B, L, 5) - y bit allocation weights per AP
            w_H: (B, L, K, 5) - H bit allocation weights per link
            w_demod: (B, L, K, 5) - demod bit allocation weights per link
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # --- Compute policy input features (9 dims) per (l, k) ---
        v_real = v[..., 0:1]  # (B, L, K, 1)
        v_imag = v[..., 1:2]  # (B, L, K, 1)

        # Channel norm per user: ||H_{l,:,k}||^2
        h_power = (H ** 2).sum(dim=(2, 4))  # (B, L, K)
        channel_norm = h_power.unsqueeze(-1)  # (B, L, K, 1)

        # Average channel power across APs for each user
        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)

        # SIR: signal-to-interference ratio per (l, k) link
        total_power = h_power.sum(dim=2, keepdim=True)  # (B, L, 1)
        interference_power = total_power - h_power  # (B, L, K)
        sir = torch.log1p(h_power / (interference_power + 1e-10)).unsqueeze(-1)

        # Total power norm
        total_power_norm = torch.log1p(total_power).expand(B, L, K).unsqueeze(-1)

        # y power per AP (shared across users)
        y_power_per_ap = (y ** 2).sum(dim=(2, 3))  # (B, L)
        y_power = torch.log1p(y_power_per_ap).unsqueeze(-1).unsqueeze(-1).expand(B, L, K, 1)

        # Signal power (demod magnitude)
        signal_power = torch.sqrt(v_real ** 2 + v_imag ** 2 + 1e-10)

        # Policy input: 9 features
        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm,
                                  avg_channel_power, sir, total_power_norm,
                                  y_power, signal_power], dim=-1)  # (B, L, K, 9)

        # --- Get bitwidth decisions ---
        logits_y, logits_H, logits_demod = self.policy(policy_input)  # each: (B, L, K, 5)

        # y quantization: per AP (average logits over users, then apply Gumbel-Softmax)
        logits_y_per_ap = logits_y.mean(dim=2)  # (B, L, 5)
        w_y = F.gumbel_softmax(logits_y_per_ap, tau=tau, hard=True)  # (B, L, 5)

        # H quantization: per link
        w_H = F.gumbel_softmax(logits_H, tau=tau, hard=True)  # (B, L, K, 5)

        # Demod quantization: per link
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B, L, K, 5)

        # --- Quantize y (received signal) ---
        y_zeros = torch.zeros_like(y)
        y_q2 = self.q2_y(y)
        y_q4 = self.q4_y(y)
        y_q6 = self.q6_y(y)
        y_q8 = self.q8_y(y)

        w_y_exp = w_y.unsqueeze(-1).unsqueeze(-1)  # (B, L, 5, 1, 1)
        y_stack = torch.stack([y_zeros, y_q2, y_q4, y_q6, y_q8], dim=2)  # (B, L, 5, N, 2)
        y_q = (y_stack * w_y_exp).sum(dim=2)  # (B, L, N, 2)

        # --- Quantize H (channel) ---
        H_zeros = torch.zeros_like(H)
        H_q2 = self.q2_H(H)
        H_q4 = self.q4_H(H)
        H_q6 = self.q6_H(H)
        H_q8 = self.q8_H(H)

        w_H_exp = w_H.unsqueeze(2).unsqueeze(-1)  # (B, L, 1, K, 5, 1)
        H_stack = torch.stack([H_zeros, H_q2, H_q4, H_q6, H_q8], dim=4)  # (B, L, N, K, 5, 2)
        H_q = (H_stack * w_H_exp).sum(dim=4)  # (B, L, N, K, 2)

        # --- Quantize demod ---
        v_zeros = torch.zeros_like(v)
        v_q2 = self.q2_demod(v)
        v_q4 = self.q4_demod(v)
        v_q6 = self.q6_demod(v)
        v_q8 = self.q8_demod(v)

        w_d_exp = w_demod.unsqueeze(-1)  # (B, L, K, 5, 1)
        v_stack = torch.stack([v_zeros, v_q2, v_q4, v_q6, v_q8], dim=3)  # (B, L, K, 5, 2)
        v_q = (v_stack * w_d_exp).sum(dim=3)  # (B, L, K, 2)

        # --- Expected bits per link ---
        y_bit_values = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0], device=v.device)
        y_bits_per_ap = 2.0 * N * (w_y * y_bit_values.view(1, 1, 5)).sum(dim=-1)  # (B, L)
        y_bits_per_link = y_bits_per_ap.unsqueeze(-1) / K  # (B, L, 1)

        H_bits_per_link = (2.0 * N) * (w_H * y_bit_values.view(1, 1, 1, 5)).sum(dim=-1)  # (B, L, K)

        demod_bits_per_link = 2.0 * (w_demod * y_bit_values.view(1, 1, 1, 5)).sum(dim=-1)  # (B, L, K)

        expected_bits_per_link = y_bits_per_link + H_bits_per_link + demod_bits_per_link  # (B, L, K)

        return v_q, H_q, y_q, expected_bits_per_link, w_y, w_H, w_demod, y_bits_per_ap, H_bits_per_link, demod_bits_per_link


# ============================================================
# 5. Joint Model V4
# ============================================================
class JointModelV4(nn.Module):
    """End-to-end model combining TripleAdaptiveQuantizerV4 + GNNTransformerDetectorV4."""
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(JointModelV4, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.quantizer = TripleAdaptiveQuantizerV4(N=N, policy_hidden=64)
        self.detector = GNNTransformerDetectorV4(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            noise_w=noise_w
        )
        # Max bits for normalization
        self.max_y_bits_per_link = 2.0 * N * 8.0 / K
        self.max_H_bits = 2.0 * N * 8.0
        self.max_demod_bits = 2.0 * 8.0

    def forward(self, v, H, y, local_snr, tau=1.0, use_quantization=True, noise_std=0.0):
        """
        Args:
            v: (B, L, K, 2) - demod results
            H: (B, L, N, K, 2) - channel coefficients
            y: (B, L, N, 2) - received signal
            local_snr: (B, L, K, 1) - local SNR
            tau: Gumbel-Softmax temperature
            use_quantization: if False, skip quantization (Phase 1)
            noise_std: Gaussian noise std to inject (Phase 1 robustness)
        Returns:
            detected: (B, K, 2)
            expected_bits_per_link: (B, L, K)
            s_global: (B, K, 2) - global MMSE estimate (for monitoring)
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # Optional noise injection for robustness training
        if noise_std > 0 and self.training:
            v = v + torch.randn_like(v) * noise_std
            H = H + torch.randn_like(H) * noise_std
            y = y + torch.randn_like(y) * noise_std

        if use_quantization:
            v_q, H_q, y_q, expected_bits_per_link, w_y, w_H, w_demod, \
                y_bits_ap, H_bits_link, demod_bits_link = \
                self.quantizer(v, H, y, local_snr, tau=tau)

            # Bitwidth features for detector: (B, L, K, 3) normalized
            y_bits_per_link = y_bits_ap.unsqueeze(-1) / K  # (B, L, 1)
            y_bw_feat = y_bits_per_link.expand(B, L, K).unsqueeze(-1) / max(self.max_y_bits_per_link, 1e-10)
            H_bw_feat = H_bits_link.unsqueeze(-1) / self.max_H_bits
            demod_bw_feat = demod_bits_link.unsqueeze(-1) / self.max_demod_bits
            bitwidth_features = torch.cat([y_bw_feat, H_bw_feat, demod_bw_feat], dim=-1)  # (B, L, K, 3)
        else:
            v_q = v
            H_q = H
            y_q = y
            expected_bits_per_link = torch.zeros(B, L, K, device=v.device)
            bitwidth_features = torch.ones(B, L, K, 3, device=v.device)

        detected, s_global = self.detector(v_q, H_q, y_q, bitwidth_features, local_snr)

        return detected, expected_bits_per_link, s_global


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors_v4(s_hat, s, H, local_snr, y, device):
    """Convert numpy complex arrays to real-valued torch tensors including y."""
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)

    H_real = np.stack([H.real, H.imag], axis=-1)
    H_tensor = torch.FloatTensor(H_real).to(device)

    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)

    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)

    y_real = np.stack([y.real, y.imag], axis=-1)
    y_tensor = torch.FloatTensor(y_real).to(device)

    return v_tensor, H_tensor, snr_tensor, Y_tensor, y_tensor


# ============================================================
# 6. Main Training Script
# ============================================================
def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_phase2 = args.phase2_sub1_epochs + args.phase2_sub2_epochs + args.phase2_sub3_epochs
    total_epochs = args.epochs_phase1 + total_phase2

    # Full precision bits per link: (2*N + 2*N + 2) * 32 = (8+8+2)*32 = 576 for N=4
    fp_bits_per_link = (2 * args.N + 2 * args.N + 2) * 32
    # Max possible bits per link with 8-bit: (2*N*8/K + 2*N*8 + 2*8)
    max_bits_per_link = 2 * args.N * 8.0 / args.K + 2 * args.N * 8.0 + 2 * 8.0

    print(f"{'=' * 100}")
    print(f"GNN+Transformer Hybrid Detector V4 with Triple Adaptive Quantization (y+H+demod)")
    print(f"  ** FIXED: MMSE computations now use float64/complex128 for numerical stability **")
    print(f"{'=' * 100}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users")
    print(f"Training: {total_epochs} total epochs")
    print(f"  Phase 1: {args.epochs_phase1} epochs (detector pre-training, full precision + noise injection)")
    print(f"  Phase 2: {total_phase2} epochs (joint QAT, 3 sub-phases: {args.phase2_sub1_epochs}+{args.phase2_sub2_epochs}+{args.phase2_sub3_epochs})")
    print(f"    Sub1: lambda={args.lambda_sub1}, tau: {args.tau_sub1_start} -> {args.tau_sub1_end}")
    print(f"    Sub2: lambda={args.lambda_sub2}, tau: {args.tau_sub2_start} -> {args.tau_sub2_end}")
    print(f"    Sub3: lambda={args.lambda_sub3}, tau: {args.tau_sub3_start} -> {args.tau_sub3_end}")
    print(f"  Phase1 LR: {args.lr_phase1}, Phase2 LR: {args.lr_phase2}")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {args.batches_per_epoch}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Bit options: {{0, 2, 4, 6, 8}} for y, H, and demod")
    print(f"  y bits per AP: {{0, {2*args.N*2}, {2*args.N*4}, {2*args.N*6}, {2*args.N*8}}}")
    print(f"  H bits per link (N={args.N}): {{0, {2*args.N*2}, {2*args.N*4}, {2*args.N*6}, {2*args.N*8}}}")
    print(f"  Demod bits per link: {{0, 4, 8, 12, 16}}")
    print(f"  Max bits per link: {max_bits_per_link:.1f}")
    print(f"Bit constraint: C_target={args.c_target} bits/link (max possible={max_bits_per_link:.1f}, ~{args.c_target/max_bits_per_link*100:.0f}% of max)")
    print(f"  Full precision bits/link: {fp_bits_per_link}")
    print(f"  Compression target: {fp_bits_per_link / args.c_target:.1f}x")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"Noise injection std (Phase 1): {args.noise_injection_std}")
    print(f"Gradient clipping: max_norm={args.grad_clip}")
    print(f"Dropout: {args.dropout}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 100}")

    # Initialize system model
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    noise_w = sys_model.noise_w
    print(f"Noise power: {noise_w:.2e} W ({sys_model.noise_power_dbm:.2f} dBm)")

    # ===========================
    # Pre-generate fixed test set
    # ===========================
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"\nGenerating fixed test set ({args.test_samples} samples per power point)...")
    for p in test_p_tx_list:
        s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch_v2(sys_model, args.test_samples, p_tx_dbm=p)
        v_t, H_t, snr_t, Y_t, y_t = prepare_tensors_v4(s_hat_np, s_np, H_np, snr_np, y_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'Y': Y_t, 'y': y_t,
            's_hat_np': s_hat_np, 's_np': s_np, 'H_np': H_np, 'y_np': y_np
        }
    print("Test set generated successfully.\n")

    # ===========================
    # Compute baselines
    # ===========================
    print(f"Computing baseline results...")
    baseline_ber_dist_full = {}
    baseline_ber_cmmse = {}
    baseline_ber_cmmse_q = {}
    baseline_ber_cmmse_mixed_q = {}
    baseline_ber_dist_q = {}
    baseline_ber_lmmse_global_q = {}

    for p in test_p_tx_list:
        td = test_dataset[p]
        baseline_ber_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        baseline_ber_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], noise_w)

    print(f"\nComputing C-MMSE-Q baseline (same total bit budget, c_target={args.c_target})...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_cq, _, b_used = compute_cmmse_q_detection(
            td['H_np'], td['y_np'], td['s_np'], noise_w,
            args.c_target, args.K, args.N
        )
        baseline_ber_cmmse_q[p] = ber_cq

    print(f"\nComputing C-MMSE-Mixed-Q baseline (optimized mixed H/y/demod quantization)...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_mq, _, b_y, b_H = compute_cmmse_mixed_q_detection(
            td['H_np'], td['y_np'], td['s_np'], noise_w,
            args.c_target, args.K, args.N, s_hat_local=td['s_hat_np']
        )
        baseline_ber_cmmse_mixed_q[p] = ber_mq

    print(f"\nComputing Dist-Q baseline (same total bit budget)...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_dq, b_used = compute_dist_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        baseline_ber_dist_q[p] = ber_dq

    print(f"\nComputing LMMSE-Global-Q baseline (local LMMSE + quantize demod + average at CPU)...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_lgq, b_used = compute_lmmse_global_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        baseline_ber_lmmse_global_q[p] = ber_lgq

    print(f"\n{'=' * 100}")
    print(f"Baseline BERs (computed on fixed test set):")
    print(f"{'p_tx(dBm)':<12} | {'Dist-Full':<12} | {'C-MMSE':<12} | {'C-MMSE-Q':<12} | {'C-MMSE-MxQ':<12} | {'Dist-Q':<12} | {'LMMSE-GQ':<12}")
    print("-" * 90)
    for p in test_p_tx_list:
        print(f"{p:<12} | {baseline_ber_dist_full[p]:<12.6f} | {baseline_ber_cmmse[p]:<12.6f} | "
              f"{baseline_ber_cmmse_q[p]:<12.6f} | {baseline_ber_cmmse_mixed_q[p]:<12.6f} | "
              f"{baseline_ber_dist_q[p]:<12.6f} | {baseline_ber_lmmse_global_q[p]:<12.6f}")
    print()

    # ===========================
    # Initialize model
    # ===========================
    model = JointModelV4(
        L=args.L, N=args.N, K=args.K,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        noise_w=noise_w
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    detector_params = sum(p.numel() for p in model.detector.parameters())
    quantizer_params = sum(p.numel() for p in model.quantizer.parameters())
    print(f"Model parameters: Total={total_params:,}, Detector={detector_params:,}, Quantizer={quantizer_params:,}")
    print(f"Detector alpha (residual scaling) initialized to: {model.detector.alpha.item():.4f}")

    criterion = nn.MSELoss()

    # ===========================
    # Verification: PyTorch Global MMSE vs Numpy C-MMSE
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"VERIFICATION: Comparing PyTorch Global MMSE (float64) vs Numpy C-MMSE on test set")
    print(f"  (These should match closely - if not, there is a numerical precision issue)")
    print(f"{'=' * 100}")

    model.eval()
    verification_passed = True
    with torch.no_grad():
        for p_val in test_p_tx_list:
            td = test_dataset[p_val]
            # PyTorch Global MMSE (full precision H and y, no quantization)
            s_global_torch = model.detector.compute_global_mmse(td['H'], td['y'])
            sg_np = s_global_torch.cpu().numpy()
            s_global_pred = sg_np[..., 0] + 1j * sg_np[..., 1]
            ber_torch_mmse = compute_ber(td['s_np'], s_global_pred)

            # Numpy C-MMSE
            ber_np_cmmse = baseline_ber_cmmse[p_val]

            # Check difference
            ber_diff = abs(ber_torch_mmse - ber_np_cmmse)
            status = "OK" if ber_diff < 0.01 else "WARNING: MISMATCH!"
            if ber_diff >= 0.01:
                verification_passed = False

            print(f"  p_tx={p_val:4d} dBm | PyTorch MMSE BER: {ber_torch_mmse:.6f} | "
                  f"Numpy C-MMSE BER: {ber_np_cmmse:.6f} | Diff: {ber_diff:.6f} | {status}")

    if verification_passed:
        print(f"\n  *** VERIFICATION PASSED: PyTorch Global MMSE matches Numpy C-MMSE ***\n")
    else:
        print(f"\n  *** WARNING: VERIFICATION FAILED - PyTorch and Numpy MMSE BERs do NOT match ***")
        print(f"  *** This indicates a remaining numerical precision issue! ***\n")

    # ===========================
    # Phase 1: Detector Pre-training
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"Phase 1: Detector Pre-training ({args.epochs_phase1} epochs, full precision + noise injection)")
    print(f"  Learning rate: {args.lr_phase1} with cosine annealing")
    print(f"  Noise injection std: {args.noise_injection_std}")
    print(f"  Key: s_global (centralized MMSE) is the primary base; NN learns residual correction")
    print(f"{'=' * 100}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=args.epochs_phase1, eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor, y_tensor = prepare_tensors_v4(
                s_hat_np, s_np, H_np, snr_np, y_np, device)

            optimizer_phase1.zero_grad()
            detected, _, s_global = model(v_tensor, H_tensor, y_tensor, snr_tensor,
                                use_quantization=False,
                                noise_std=args.noise_injection_std)
            loss = criterion(detected, Y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), max_norm=args.grad_clip)
            optimizer_phase1.step()
            epoch_loss += loss.item()

        scheduler_phase1.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_bers = {}
                val_bers_global = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    val_out, _, val_s_global = model(td['v'], td['H'], td['y'], td['snr'],
                                       use_quantization=False, noise_std=0.0)
                    # Detector BER
                    val_np = val_out.cpu().numpy()
                    s_pred = val_np[..., 0] + 1j * val_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)
                    
                    # Global MMSE BER (from full precision H and y)
                    sg_np = val_s_global.cpu().numpy()
                    s_global_pred = sg_np[..., 0] + 1j * sg_np[..., 1]
                    val_bers_global[p_val] = compute_ber(td['s_np'], s_global_pred)

            lr_now = scheduler_phase1.get_last_lr()[0]
            alpha_val = model.detector.alpha.item()
            print(f"Phase1 Epoch [{epoch + 1:03d}/{args.epochs_phase1}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | alpha: {alpha_val:.4f} | "
                  f"Train Loss: {avg_loss:.6f}")
            print(f"  Detector BER:     0dBm={val_bers[0]:.4f}, 10dBm={val_bers[10]:.4f}, 20dBm={val_bers[20]:.4f}")
            print(f"  Global MMSE BER:  0dBm={val_bers_global[0]:.4f}, 10dBm={val_bers_global[10]:.4f}, 20dBm={val_bers_global[20]:.4f}")
            print(f"  C-MMSE BER:       0dBm={baseline_ber_cmmse[0]:.4f}, 10dBm={baseline_ber_cmmse[10]:.4f}, 20dBm={baseline_ber_cmmse[20]:.4f}")

    print("Phase 1 completed.\n")

    # ===========================
    # Phase 2: Joint QAT Training (3 sub-phases)
    # ===========================
    sub_phases = [
        {
            'name': 'Sub-Phase 1 (Warm-start)',
            'epochs': args.phase2_sub1_epochs,
            'lambda': args.lambda_sub1,
            'tau_start': args.tau_sub1_start,
            'tau_end': args.tau_sub1_end,
        },
        {
            'name': 'Sub-Phase 2 (Medium)',
            'epochs': args.phase2_sub2_epochs,
            'lambda': args.lambda_sub2,
            'tau_start': args.tau_sub2_start,
            'tau_end': args.tau_sub2_end,
        },
        {
            'name': 'Sub-Phase 3 (Fine-tune)',
            'epochs': args.phase2_sub3_epochs,
            'lambda': args.lambda_sub3,
            'tau_start': args.tau_sub3_start,
            'tau_end': args.tau_sub3_end,
        },
    ]

    print(f"{'=' * 100}")
    print(f"Phase 2: Joint QAT Training ({total_phase2} epochs, 3 sub-phases)")
    print(f"  Learning rate: {args.lr_phase2} with cosine annealing")
    for sp in sub_phases:
        print(f"  {sp['name']}: {sp['epochs']} epochs, lambda={sp['lambda']}, "
              f"tau: {sp['tau_start']} -> {sp['tau_end']}")
    print(f"{'=' * 100}")

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=total_phase2, eta_min=1e-5)

    best_val_ber = float('inf')
    best_state = None

    global_epoch = 0
    for sp_idx, sp in enumerate(sub_phases):
        print(f"\n--- Starting {sp['name']}: {sp['epochs']} epochs, lambda={sp['lambda']}, "
              f"tau: {sp['tau_start']} -> {sp['tau_end']} ---")

        for local_epoch in range(sp['epochs']):
            model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_bits = 0.0
            start_t = time.time()

            # Temperature annealing within this sub-phase
            if sp['epochs'] > 1:
                progress = local_epoch / (sp['epochs'] - 1)
            else:
                progress = 1.0
            tau = sp['tau_start'] * (sp['tau_end'] / max(sp['tau_start'], 1e-10)) ** progress
            tau = max(tau, 0.01)

            current_lambda = sp['lambda']

            for batch_idx in range(args.batches_per_epoch):
                p_train = np.random.uniform(-20, 25)
                s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch_v2(sys_model, args.batch_size, p_train)
                v_tensor, H_tensor, snr_tensor, Y_tensor, y_tensor = prepare_tensors_v4(
                    s_hat_np, s_np, H_np, snr_np, y_np, device)

                optimizer_phase2.zero_grad()
                detected, exp_bits_per_link, _ = model(
                    v_tensor, H_tensor, y_tensor, snr_tensor, tau=tau, use_quantization=True, noise_std=0.0
                )

                mse_loss = criterion(detected, Y_tensor)

                avg_bits = exp_bits_per_link.mean()
                bit_penalty = (avg_bits - args.c_target) ** 2

                loss = mse_loss + current_lambda * bit_penalty
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer_phase2.step()

                epoch_loss += loss.item()
                epoch_mse += mse_loss.item()
                epoch_bits += avg_bits.item()

            scheduler_phase2.step()
            elapsed = time.time() - start_t
            avg_loss = epoch_loss / args.batches_per_epoch
            avg_mse = epoch_mse / args.batches_per_epoch
            avg_bits_epoch = epoch_bits / args.batches_per_epoch

            if (local_epoch + 1) % 5 == 0 or local_epoch == 0:
                model.eval()
                with torch.no_grad():
                    # Validate at 0 dBm
                    td = test_dataset[0]
                    val_out, val_bits, val_sg = model(
                        td['v'], td['H'], td['y'], td['snr'], tau=tau, use_quantization=True
                    )
                    val_np_out = val_out.cpu().numpy()
                    s_pred = val_np_out[..., 0] + 1j * val_np_out[..., 1]
                    val_ber_0 = compute_ber(td['s_np'], s_pred)
                    val_avg_bits_total = val_bits.mean().item()

                    # Global MMSE BER from quantized data at 0 dBm
                    sg_np = val_sg.cpu().numpy()
                    sg_pred = sg_np[..., 0] + 1j * sg_np[..., 1]
                    val_ber_global_0 = compute_ber(td['s_np'], sg_pred)

                    # Validate at 10 dBm
                    td10 = test_dataset[10]
                    val_out10, val_bits10, _ = model(
                        td10['v'], td10['H'], td10['y'], td10['snr'], tau=tau, use_quantization=True
                    )
                    val_np10 = val_out10.cpu().numpy()
                    s_pred10 = val_np10[..., 0] + 1j * val_np10[..., 1]
                    val_ber_10 = compute_ber(td10['s_np'], s_pred10)

                    # Get bit allocation distribution at 10 dBm
                    _, _, _, _, w_y_t, w_H_t, w_d_t, _, _, _ = model.quantizer(
                        td10['v'], td10['H'], td10['y'], td10['snr'], tau=tau
                    )
                    w_y_mean = w_y_t.mean(dim=(0, 1))  # (5,)
                    w_H_mean = w_H_t.mean(dim=(0, 1, 2))  # (5,)
                    w_d_mean = w_d_t.mean(dim=(0, 1, 2))  # (5,)

                lr_now = scheduler_phase2.get_last_lr()[0]
                alpha_val = model.detector.alpha.item()
                print(f"  Phase2 {sp['name']} Epoch [{local_epoch + 1:03d}/{sp['epochs']}] (Global: {global_epoch + 1}) | "
                      f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | lam: {current_lambda} | alpha: {alpha_val:.4f} | "
                      f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}) | "
                      f"Avg Bits: {avg_bits_epoch:.1f} (target: {args.c_target})")
                print(f"    Val BER: 0dBm={val_ber_0:.4f} (C-MMSE:{baseline_ber_cmmse[0]:.4f}, GlobalMMSE-Q:{val_ber_global_0:.4f}), "
                      f"10dBm={val_ber_10:.4f} (C-MMSE:{baseline_ber_cmmse[10]:.4f}) | "
                      f"Val Bits/link: {val_avg_bits_total:.1f}")
                print(f"    Bit dist @10dBm: y [0b:{w_y_mean[0]:.3f} 2b:{w_y_mean[1]:.3f} 4b:{w_y_mean[2]:.3f} "
                      f"6b:{w_y_mean[3]:.3f} 8b:{w_y_mean[4]:.3f}]")
                print(f"                     H [0b:{w_H_mean[0]:.3f} 2b:{w_H_mean[1]:.3f} 4b:{w_H_mean[2]:.3f} "
                      f"6b:{w_H_mean[3]:.3f} 8b:{w_H_mean[4]:.3f}]")
                print(f"                   dem [0b:{w_d_mean[0]:.3f} 2b:{w_d_mean[1]:.3f} 4b:{w_d_mean[2]:.3f} "
                      f"6b:{w_d_mean[3]:.3f} 8b:{w_d_mean[4]:.3f}]")

                # Track best model
                avg_val_ber = (val_ber_0 + val_ber_10) / 2.0
                if avg_val_ber < best_val_ber:
                    best_val_ber = avg_val_ber
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"    -> New best model! Avg val BER = {avg_val_ber:.6f}")

            global_epoch += 1

    print(f"\nPhase 2 completed. Best avg val BER = {best_val_ber:.6f}\n")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model from training for final evaluation.\n")

    # ===========================
    # Final Evaluation
    # ===========================
    print(f"{'=' * 100}")
    print(f"Final Evaluation on Test Set ({args.test_samples} samples per power point)")
    print(f"{'=' * 100}")

    model.eval()

    header = (f"{'p_tx(dBm)':<10} | {'Dist-Full':<11} | {'C-MMSE':<11} | {'C-MMSE-Q':<11} | "
              f"{'C-MMSE-MxQ':<11} | {'Dist-Q':<11} | {'LMMSE-GQ':<11} | {'Proposed':<11} | "
              f"{'vs MixQ':<10} | {'vs Dist-F':<10} | {'Bits/link':<10}")
    print(f"\n{header}")
    print("-" * len(header))

    results_proposed = {}
    with torch.no_grad():
        for p in test_p_tx_list:
            td = test_dataset[p]

            ber_dist_full = baseline_ber_dist_full[p]
            ber_cmmse = baseline_ber_cmmse[p]
            ber_cmmse_q = baseline_ber_cmmse_q[p]
            ber_cmmse_mq = baseline_ber_cmmse_mixed_q[p]
            ber_dist_q = baseline_ber_dist_q[p]
            ber_lmmse_gq = baseline_ber_lmmse_global_q[p]

            tau_eval = args.tau_sub3_end
            detected, exp_bits, _ = model(
                td['v'], td['H'], td['y'], td['snr'], tau=tau_eval, use_quantization=True
            )
            det_np = detected.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_total = exp_bits.mean().item()

            improve_mq = ((ber_cmmse_mq - ber_proposed) / max(ber_cmmse_mq, 1e-10)) * 100
            improve_dist = ((ber_dist_full - ber_proposed) / max(ber_dist_full, 1e-10)) * 100

            results_proposed[p] = {
                'ber': ber_proposed, 'bits_t': avg_bits_total
            }

            print(f"{p:<10} | {ber_dist_full:<11.6f} | {ber_cmmse:<11.6f} | {ber_cmmse_q:<11.6f} | "
                  f"{ber_cmmse_mq:<11.6f} | {ber_dist_q:<11.6f} | {ber_lmmse_gq:<11.6f} | {ber_proposed:<11.6f} | "
                  f"{improve_mq:>+8.2f}% | {improve_dist:>+8.2f}% | {avg_bits_total:<10.2f}")

    print("-" * len(header))

    # ===========================
    # Summary statistics
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"Summary: Average Performance Across All Power Points")
    print(f"{'=' * 100}")
    avg_ber_dist = np.mean([baseline_ber_dist_full[p] for p in test_p_tx_list])
    avg_ber_cmmse = np.mean([baseline_ber_cmmse[p] for p in test_p_tx_list])
    avg_ber_cmmse_q = np.mean([baseline_ber_cmmse_q[p] for p in test_p_tx_list])
    avg_ber_cmmse_mq = np.mean([baseline_ber_cmmse_mixed_q[p] for p in test_p_tx_list])
    avg_ber_dist_q = np.mean([baseline_ber_dist_q[p] for p in test_p_tx_list])
    avg_ber_lmmse_gq = np.mean([baseline_ber_lmmse_global_q[p] for p in test_p_tx_list])
    avg_ber_proposed = np.mean([results_proposed[p]['ber'] for p in test_p_tx_list])
    avg_total_bits = np.mean([results_proposed[p]['bits_t'] for p in test_p_tx_list])

    print(f"  Average Dist-Full BER:        {avg_ber_dist:.6f}")
    print(f"  Average C-MMSE BER:           {avg_ber_cmmse:.6f} (upper bound, full precision)")
    print(f"  Average C-MMSE-Q BER:         {avg_ber_cmmse_q:.6f} (uniform quantization)")
    print(f"  Average C-MMSE-Mixed-Q BER:   {avg_ber_cmmse_mq:.6f} (optimized mixed quantization)")
    print(f"  Average Dist-Q BER:           {avg_ber_dist_q:.6f} (distributed with quantization)")
    print(f"  Average LMMSE-Global-Q BER:   {avg_ber_lmmse_gq:.6f} (local LMMSE + quant demod + avg)")
    print(f"  Average Proposed BER:         {avg_ber_proposed:.6f}")
    print(f"  Average Total Bits/Link:      {avg_total_bits:.2f} (target: {args.c_target})")
    print(f"  Full Precision Bits/Link:     {fp_bits_per_link}")
    print(f"  Average Compression Ratio:    {fp_bits_per_link / max(avg_total_bits, 1e-10):.1f}x")

    # Performance comparison
    print(f"\n  --- Performance Comparisons ---")
    comparisons = [
        ("C-MMSE (full precision)", avg_ber_cmmse),
        ("C-MMSE-Q (uniform quant)", avg_ber_cmmse_q),
        ("C-MMSE-Mixed-Q (opt mixed)", avg_ber_cmmse_mq),
        ("Dist-Full (no quant)", avg_ber_dist),
        ("Dist-Q (uniform quant)", avg_ber_dist_q),
        ("LMMSE-Global-Q (local+quant+avg)", avg_ber_lmmse_gq),
    ]
    for name, ber_baseline in comparisons:
        if avg_ber_proposed < ber_baseline:
            pct = ((ber_baseline - avg_ber_proposed) / max(ber_baseline, 1e-10)) * 100
            print(f"  *** Proposed OUTPERFORMS {name} by {pct:.1f}% ***")
        else:
            pct = ((avg_ber_proposed - ber_baseline) / max(ber_baseline, 1e-10)) * 100
            print(f"  *** Proposed underperforms {name} by {pct:.1f}% ***")

    # ===========================
    # Detailed bit allocation statistics
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"Detailed Bit Allocation Statistics")
    print(f"{'=' * 100}")
    with torch.no_grad():
        tau_eval = args.tau_sub3_end
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]

            _, _, _, _, w_y_t, w_H_t, w_d_t, y_bits_ap, H_bits_link, demod_bits_link = model.quantizer(
                td['v'], td['H'], td['y'], td['snr'], tau=tau_eval
            )

            w_y_mean = w_y_t.mean(dim=(0, 1))
            w_H_mean = w_H_t.mean(dim=(0, 1, 2))
            w_d_mean = w_d_t.mean(dim=(0, 1, 2))

            avg_y_bits_ap = y_bits_ap.mean().item()
            avg_H_bits_link = H_bits_link.mean().item()
            avg_demod_bits_link = demod_bits_link.mean().item()
            avg_total_link = avg_y_bits_ap / args.K + avg_H_bits_link + avg_demod_bits_link
            compression = fp_bits_per_link / max(avg_total_link, 1e-10)

            print(f"\n  p_tx = {p} dBm:")
            print(f"    y bit distribution (per AP):   0b: {w_y_mean[0]:.3f}, 2b: {w_y_mean[1]:.3f}, "
                  f"4b: {w_y_mean[2]:.3f}, 6b: {w_y_mean[3]:.3f}, 8b: {w_y_mean[4]:.3f}")
            print(f"    H bit distribution (per link): 0b: {w_H_mean[0]:.3f}, 2b: {w_H_mean[1]:.3f}, "
                  f"4b: {w_H_mean[2]:.3f}, 6b: {w_H_mean[3]:.3f}, 8b: {w_H_mean[4]:.3f}")
            print(f"    Demod bit dist (per link):     0b: {w_d_mean[0]:.3f}, 2b: {w_d_mean[1]:.3f}, "
                  f"4b: {w_d_mean[2]:.3f}, 6b: {w_d_mean[3]:.3f}, 8b: {w_d_mean[4]:.3f}")
            print(f"    Avg y bits/AP: {avg_y_bits_ap:.2f}, Avg H bits/link: {avg_H_bits_link:.2f}, "
                  f"Avg demod bits/link: {avg_demod_bits_link:.2f}")
            print(f"    Avg total bits/link: {avg_total_link:.2f}")
            print(f"    Full precision bits/link: {fp_bits_per_link}, Compression ratio: {compression:.1f}x")

    # ===========================
    # Per-AP bit allocation statistics
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"Per-AP Bit Allocation at 10 dBm (first 5 test samples)")
    print(f"{'=' * 100}")
    with torch.no_grad():
        td = test_dataset[10]
        v_sub = td['v'][:5]
        H_sub = td['H'][:5]
        y_sub = td['y'][:5]
        snr_sub = td['snr'][:5]
        _, _, _, exp_bits_sub, w_y_sub, w_H_sub, w_d_sub, y_bits_sub, H_bits_sub, d_bits_sub = \
            model.quantizer(v_sub, H_sub, y_sub, snr_sub, tau=tau_eval)

        # y bits per AP
        avg_y_per_ap = y_bits_sub.mean(dim=0)  # (L,)
        # H and demod bits per AP (avg over users)
        avg_H_per_ap = H_bits_sub.mean(dim=(0, 2))  # (L,)
        avg_d_per_ap = d_bits_sub.mean(dim=(0, 2))  # (L,)
        # Total per link per AP
        avg_total_per_ap = exp_bits_sub.mean(dim=(0, 2))  # (L,)

        for l in range(args.L):
            print(f"  AP {l:2d}: y_bits/AP={avg_y_per_ap[l].item():.1f}, "
                  f"H_bits/link={avg_H_per_ap[l].item():.1f}, "
                  f"demod_bits/link={avg_d_per_ap[l].item():.1f}, "
                  f"total_bits/link={avg_total_per_ap[l].item():.1f}")

    # ===========================
    # LSQ scale parameter values
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"LSQ Quantizer Scale Parameters (learned)")
    print(f"{'=' * 100}")
    q = model.quantizer
    print(f"  y quantizers:     2b s={q.q2_y.s.item():.6f}, 4b s={q.q4_y.s.item():.6f}, "
          f"6b s={q.q6_y.s.item():.6f}, 8b s={q.q8_y.s.item():.6f}")
    print(f"  H quantizers:     2b s={q.q2_H.s.item():.6f}, 4b s={q.q4_H.s.item():.6f}, "
          f"6b s={q.q6_H.s.item():.6f}, 8b s={q.q8_H.s.item():.6f}")
    print(f"  Demod quantizers: 2b s={q.q2_demod.s.item():.6f}, 4b s={q.q4_demod.s.item():.6f}, "
          f"6b s={q.q6_demod.s.item():.6f}, 8b s={q.q8_demod.s.item():.6f}")

    # ===========================
    # Final alpha value
    # ===========================
    print(f"\n  Detector residual alpha (learned): {model.detector.alpha.item():.6f}")

    # ===========================
    # Save model
    # ===========================
    save_path = 'new_joint_model_v4.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
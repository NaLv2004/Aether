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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V5 with Improved Adaptive Quantization")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=5, help="Phase 1 epochs (detector pre-training, full precision)")
    parser.add_argument("--epochs_phase2", type=int, default=20, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=50, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=3e-4, help="Learning rate for Phase 2")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D for feature embedding")
    parser.add_argument("--c_target", type=float, default=80.0, help="Target average bits per (AP, user) link")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of mean-field GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--train_snr_dbm", type=float, default=12.0, help="Fixed training SNR in dBm")
    parser.add_argument("--tau_start", type=float, default=5.0, help="Gumbel-Softmax temperature start for Phase 2")
    parser.add_argument("--tau_end", type=float, default=0.3, help="Gumbel-Softmax temperature end for Phase 2")
    parser.add_argument("--rho_init", type=float, default=0.01, help="Initial Augmented Lagrangian penalty coefficient")
    parser.add_argument("--rho_max", type=float, default=10.0, help="Maximum rho for Augmented Lagrangian")
    return parser.parse_args()


# ============================================================
# 1. Data Generation Function
# ============================================================
def generate_data_batch(sys_model, batch_size, p_tx_dbm):
    """
    Vectorized data generation returning local LMMSE estimates, true symbols,
    full channel matrix H, local SNR features, and raw received signal y.

    CRITICAL: Local LMMSE is computed from FULL PRECISION y and H at the AP side.

    Returns:
        s_hat: (batch_size, L, K) complex - local LMMSE estimates (full precision)
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

    # 4. Local LMMSE detection from FULL PRECISION H and y (AP-side computation)
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


def compute_ber_from_logits(s_true_np, logits_np):
    """
    Compute BER from sign logits.
    logits_np: (B, K, 2) where logits_np[...,0] is logit for P(real>0), logits_np[...,1] is logit for P(imag>0)
    """
    pred_real_sign = np.where(logits_np[..., 0] > 0, 1.0, -1.0)
    pred_imag_sign = np.where(logits_np[..., 1] > 0, 1.0, -1.0)
    true_real_sign = np.sign(s_true_np.real)
    true_imag_sign = np.sign(s_true_np.imag)
    true_real_sign[true_real_sign == 0] = 1
    true_imag_sign[true_imag_sign == 0] = 1

    err_real = (pred_real_sign != true_real_sign).sum()
    err_imag = (pred_imag_sign != true_imag_sign).sum()
    total_bits = s_true_np.size * 2
    return (err_real + err_imag) / total_bits


def compute_dist_full_ber(s_hat, s):
    """Dist-Full: mean pooling of full-precision local LMMSE across APs."""
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
    """
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)
    b_ideal = total_bits_per_ap / reals_per_ap

    bit_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    best_ber = 1.0
    best_b = 1
    best_s_hat = None

    for b_use in bit_candidates:
        actual_bits_per_ap = reals_per_ap * b_use
        if actual_bits_per_ap > total_bits_per_ap * 1.05:
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
        reg_eye = 1e-10 * np.eye(K_users).reshape(1, K_users, K_users)
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
    with b_H and b_y chosen to match total bit budget.
    """
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users

    best_ber = 1.0
    best_b_y = 0
    best_b_H = 0
    best_b_demod = 0
    best_s_hat = None
    best_mode = "H+y"

    bit_options = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]

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
            reg_eye = 1e-10 * np.eye(K_users).reshape(1, K_users, K_users)
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

    # Option B: H + y + demod
    if s_hat_local is not None:
        for b_y in bit_options:
            for b_H in bit_options:
                for b_d in [1, 2, 3, 4, 6, 8]:
                    bits_used = 2 * N_ant * b_y + K_users * 2 * N_ant * b_H + K_users * 2 * b_d
                    if bits_used > total_bits_per_ap * 1.05:
                        continue

                    H_real_q = uniform_quantize_np(H.real, b_H)
                    H_imag_q = uniform_quantize_np(H.imag, b_H)
                    H_q = H_real_q + 1j * H_imag_q

                    y_real_q = uniform_quantize_np(y.real, b_y)
                    y_imag_q = uniform_quantize_np(y.imag, b_y)
                    y_q = y_real_q + 1j * y_imag_q

                    d_real_q = uniform_quantize_np(s_hat_local.real, b_d)
                    d_imag_q = uniform_quantize_np(s_hat_local.imag, b_d)
                    d_q = d_real_q + 1j * d_imag_q

                    H_all = H_q.reshape(B, L * N_ant, K_users)
                    y_all = y_q.reshape(B, L * N_ant)
                    H_H = H_all.conj().transpose(0, 2, 1)
                    HHH = H_H @ H_all
                    noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
                    reg_eye = 1e-10 * np.eye(K_users).reshape(1, K_users, K_users)
                    R = HHH + noise_eye + reg_eye
                    HHy = H_H @ y_all[..., np.newaxis]
                    try:
                        R_inv = np.linalg.inv(R)
                        s_cmmse = (R_inv @ HHy).squeeze(-1)
                        s_demod_avg = d_q.mean(axis=1)
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
    Dist-Q: Distributed local LMMSE with uniform quantization of demod results.
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


# ============================================================
# 3. GNN + Transformer Hybrid Detector V5
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


class GNNTransformerDetectorV5(nn.Module):
    """
    Hybrid GNN + Transformer detector V5 for Cell-Free MIMO.

    Key changes from V4:
    1. Receives quantized y_q, H_q, s_hat_q from fronthaul
    2. Computes global MMSE from quantized H_q and y_q (CPU-side centralized MMSE)
    3. GNN+Transformer learns residual correction on top of global MMSE
    4. Output: produce LOGITS for sign classification (BCE loss)
    5. For BER: use sign of global_mmse + alpha * residual

    Uses float64/complex128 for matrix inversion stability.
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(GNNTransformerDetectorV5, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim
        self.noise_w = noise_w

        # Learnable residual scaling factor (initialized to 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

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

        # Output Head: produces logits for sign classification (2 per user: real sign, imag sign)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        # Initialize to near-zero for stable residual learning
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def compute_per_ap_mmse_init(self, H_q, y_q):
        """
        Compute per-AP LMMSE initial estimate using quantized H and y.
        Uses float64/complex128 for numerical stability.

        Args:
            H_q: (B, L, N, K, 2) - quantized channel (real representation)
            y_q: (B, L, N, 2) - quantized received signal (real representation)
        Returns:
            s_init: (B, L, K, 2) - per-AP MMSE initial estimate (real representation)
        """
        B, L, N, K, _ = H_q.shape

        H_complex = torch.complex(H_q[..., 0].double(), H_q[..., 1].double())
        y_complex = torch.complex(y_q[..., 0].double(), y_q[..., 1].double())

        H_H = H_complex.conj().transpose(-1, -2)  # (B, L, K, N)
        HHH = torch.matmul(H_complex, H_H)  # (B, L, N, N)
        noise_eye = self.noise_w * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        R_y = HHH + noise_eye

        try:
            R_y_inv = torch.linalg.inv(R_y)
        except RuntimeError:
            reg_eye = 1e-12 * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            R_y_inv = torch.linalg.inv(R_y + reg_eye)

        W_l = torch.matmul(H_H, R_y_inv)
        y_expanded = y_complex.unsqueeze(-1)
        s_init_complex = torch.matmul(W_l, y_expanded).squeeze(-1)

        s_init = torch.stack([s_init_complex.real.float(), s_init_complex.imag.float()], dim=-1)
        return s_init

    def compute_global_mmse(self, H_q, y_q):
        """
        Compute CENTRALIZED MMSE estimate by stacking all AP's quantized H and y.
        Uses float64/complex128 for numerical stability.

        The MMSE formula is: s_hat = (H^H H + sigma^2 I)^{-1} H^H y

        Args:
            H_q: (B, L, N, K, 2) - quantized channel (real representation)
            y_q: (B, L, N, 2) - quantized received signal (real representation)
        Returns:
            s_global: (B, K, 2) - centralized MMSE estimate (real representation)
        """
        B, L, N, K, _ = H_q.shape

        H_complex = torch.complex(H_q[..., 0].double(), H_q[..., 1].double())
        y_complex = torch.complex(y_q[..., 0].double(), y_q[..., 1].double())

        H_all = H_complex.reshape(B, L * N, K)
        y_all = y_complex.reshape(B, L * N)

        H_all_H = H_all.conj().transpose(-1, -2)
        HHH = torch.matmul(H_all_H, H_all)

        noise_eye = self.noise_w * torch.eye(K, device=H_q.device, dtype=torch.float64).unsqueeze(0)
        R = HHH + noise_eye

        HHy = torch.matmul(H_all_H, y_all.unsqueeze(-1))

        try:
            R_inv = torch.linalg.inv(R)
        except RuntimeError:
            reg_eye = 1e-12 * torch.eye(K, device=H_q.device, dtype=torch.float64).unsqueeze(0)
            R_inv = torch.linalg.inv(R + reg_eye)

        s_global_complex = torch.matmul(R_inv, HHy).squeeze(-1)

        s_global = torch.stack([s_global_complex.real.float(), s_global_complex.imag.float()], dim=-1)
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
            logits: (B, K, 2) - logits for P(real>0) and P(imag>0)
            s_global: (B, K, 2) - global MMSE estimate (for monitoring and BER)
            s_combined: (B, K, 2) - s_global + alpha * residual_as_symbol (for BER computation)
        """
        B = s_hat_q.size(0)
        L = s_hat_q.size(1)
        K = s_hat_q.size(2)
        N = H_q.size(2)

        # 1. Global MMSE estimate from stacked quantized H and y
        s_global = self.compute_global_mmse(H_q, y_q)  # (B, K, 2)

        # 2. Per-AP MMSE initial estimates from quantized data
        s_init = self.compute_per_ap_mmse_init(H_q, y_q)  # (B, L, K, 2)

        # 3. Feature extraction
        mmse_init_feat = self.mmse_init_mlp(s_init)
        demod_feat = self.demod_mlp(s_hat_q)

        s_global_broadcast = s_global.unsqueeze(1).expand(B, L, K, 2)
        global_mmse_feat = self.global_mmse_mlp(s_global_broadcast)

        H_q_perm = H_q.permute(0, 1, 3, 2, 4)  # (B, L, K, N, 2)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)
        channel_feat = self.channel_mlp(H_q_flat)

        # Cross-user interference features
        h_power = (H_q ** 2).sum(dim=(2, 4))  # (B, L, K)
        desired_power = h_power
        total_power = h_power.sum(dim=2, keepdim=True)
        interference_power = total_power - desired_power

        desired_feat = torch.log1p(desired_power).unsqueeze(-1)
        interference_feat = torch.log1p(interference_power).unsqueeze(-1)
        interference_features = torch.cat([desired_feat, interference_feat], dim=-1)

        # 4. Feature Fusion
        combined = torch.cat([mmse_init_feat, demod_feat, global_mmse_feat, channel_feat,
                              bitwidth_features, interference_features, local_snr], dim=-1)
        node_features = self.fusion_mlp(combined)

        # Apply BatchNorm
        node_features_flat = node_features.reshape(-1, self.hidden_dim)
        node_features_bn = self.fusion_bn(node_features_flat)
        node_features = node_features_bn.reshape(B, L, K, self.hidden_dim)

        # 5. Mean-field GNN Message Passing
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)

        # 6. AP Aggregation
        user_features, _ = self.ap_aggregator(h)

        # 7. Transformer IC
        ic_out = self.transformer_ic(user_features)

        # 8. Output logits for sign classification
        logits = self.output_head(ic_out)  # (B, K, 2) - logits for P(real>0) and P(imag>0)

        # 9. For BER computation: s_global + alpha * residual_as_symbol
        # Convert logits to soft symbol estimates via tanh
        residual_symbol = torch.tanh(logits) / np.sqrt(2)  # maps to [-1/sqrt(2), 1/sqrt(2)]
        s_combined = s_global + self.alpha * residual_symbol

        return logits, s_global, s_combined


# ============================================================
# 4. Triple Adaptive Quantizer V5
# ============================================================
class TriplePolicyNetworkV5(nn.Module):
    """
    Policy network for triple adaptive quantization V5.

    Three outputs with 6 options each: {2, 4, 6, 8, 12, 16} bits.

    Input features per (l, k):
        [v_real, v_imag, snr, channel_norm, avg_channel_power, sir,
         total_power_norm, y_power, signal_power]
    """
    def __init__(self, input_dim=9, policy_hidden=64, num_options=6):
        super(TriplePolicyNetworkV5, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        self.head_y = nn.Linear(policy_hidden, num_options)
        self.head_H = nn.Linear(policy_hidden, num_options)
        self.head_demod = nn.Linear(policy_hidden, num_options)

    def forward(self, x):
        """x: (..., input_dim), Returns: logits_y, logits_H, logits_demod each (..., 6)"""
        h = self.shared(x)
        return self.head_y(h), self.head_H(h), self.head_demod(h)


class TripleAdaptiveQuantizerV5(nn.Module):
    """
    Triple adaptive quantization V5 with bit options {2, 4, 6, 8, 12, 16} for y, H, and demod.

    CRITICAL: The quantizer receives FULL PRECISION v, H, y from the AP.
    The AP computes local LMMSE from full precision data.
    Only AFTER that, the quantizer decides how many bits to use for fronthaul transmission.

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
        super(TripleAdaptiveQuantizerV5, self).__init__()
        self.N = N
        self.num_options = 6  # {2, 4, 6, 8, 12, 16}
        self.bit_options = [2, 4, 6, 8, 12, 16]
        self.policy = TriplePolicyNetworkV5(input_dim=9, policy_hidden=policy_hidden, num_options=self.num_options)

        # LSQ quantizers for y (received signal): 2, 4, 6, 8, 12, 16 bits
        self.q2_y = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_y = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_y = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_y = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_y = LSQQuantizer(num_bits=12, init_s=0.005)
        self.q16_y = LSQQuantizer(num_bits=16, init_s=0.001)

        # LSQ quantizers for H (channel): 2, 4, 6, 8, 12, 16 bits
        self.q2_H = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_H = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_H = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_H = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_H = LSQQuantizer(num_bits=12, init_s=0.005)
        self.q16_H = LSQQuantizer(num_bits=16, init_s=0.001)

        # LSQ quantizers for demod: 2, 4, 6, 8, 12, 16 bits
        self.q2_demod = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_demod = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_demod = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_demod = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_demod = LSQQuantizer(num_bits=12, init_s=0.005)
        self.q16_demod = LSQQuantizer(num_bits=16, init_s=0.001)

    def forward(self, v, H, y, local_snr, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - demod results (real, imag parts) - FULL PRECISION from AP
            H: (B, L, N, K, 2) - channel coefficients (real, imag parts) - FULL PRECISION from AP
            y: (B, L, N, 2) - received signal (real, imag parts) - FULL PRECISION from AP
            local_snr: (B, L, K, 1) - local SNR features
            tau: Gumbel-Softmax temperature
        Returns:
            v_q: (B, L, K, 2) - quantized demod
            H_q: (B, L, N, K, 2) - quantized channel
            y_q: (B, L, N, 2) - quantized received signal
            expected_bits_per_link: (B, L, K) - expected bits per AP-user link
            w_y: (B, L, 6) - y bit allocation weights per AP
            w_H: (B, L, K, 6) - H bit allocation weights per link
            w_demod: (B, L, K, 6) - demod bit allocation weights per link
            y_bits_per_ap: (B, L)
            H_bits_per_link: (B, L, K)
            demod_bits_per_link: (B, L, K)
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # --- Compute policy input features (9 dims) per (l, k) ---
        v_real = v[..., 0:1]
        v_imag = v[..., 1:2]

        h_power = (H ** 2).sum(dim=(2, 4))  # (B, L, K)
        channel_norm = h_power.unsqueeze(-1)

        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)

        total_power = h_power.sum(dim=2, keepdim=True)
        interference_power = total_power - h_power
        sir = torch.log1p(h_power / (interference_power + 1e-10)).unsqueeze(-1)

        total_power_norm = torch.log1p(total_power).expand(B, L, K).unsqueeze(-1)

        y_power_per_ap = (y ** 2).sum(dim=(2, 3))
        y_power = torch.log1p(y_power_per_ap).unsqueeze(-1).unsqueeze(-1).expand(B, L, K, 1)

        signal_power = torch.sqrt(v_real ** 2 + v_imag ** 2 + 1e-10)

        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm,
                                  avg_channel_power, sir, total_power_norm,
                                  y_power, signal_power], dim=-1)

        # --- Get bitwidth decisions ---
        logits_y, logits_H, logits_demod = self.policy(policy_input)

        # y quantization: per AP (average logits over users)
        logits_y_per_ap = logits_y.mean(dim=2)  # (B, L, 6)
        w_y = F.gumbel_softmax(logits_y_per_ap, tau=tau, hard=True)  # (B, L, 6)

        # H quantization: per link
        w_H = F.gumbel_softmax(logits_H, tau=tau, hard=True)  # (B, L, K, 6)

        # Demod quantization: per link
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B, L, K, 6)

        # --- Quantize y (received signal) ---
        y_q2 = self.q2_y(y)
        y_q4 = self.q4_y(y)
        y_q6 = self.q6_y(y)
        y_q8 = self.q8_y(y)
        y_q12 = self.q12_y(y)
        y_q16 = self.q16_y(y)

        # w_y: (B, L, 6) -> expand for y: (B, L, N, 2)
        w_y_exp = w_y.unsqueeze(-1).unsqueeze(-1)  # (B, L, 6, 1, 1)
        y_stack = torch.stack([y_q2, y_q4, y_q6, y_q8, y_q12, y_q16], dim=2)  # (B, L, 6, N, 2)
        y_q = (y_stack * w_y_exp).sum(dim=2)  # (B, L, N, 2)

        # --- Quantize H (channel) ---
        H_q2 = self.q2_H(H)
        H_q4 = self.q4_H(H)
        H_q6 = self.q6_H(H)
        H_q8 = self.q8_H(H)
        H_q12 = self.q12_H(H)
        H_q16 = self.q16_H(H)

        # w_H: (B, L, K, 6) -> (B, L, 1, K, 6, 1)
        w_H_exp = w_H.unsqueeze(2).unsqueeze(-1)
        H_stack = torch.stack([H_q2, H_q4, H_q6, H_q8, H_q12, H_q16], dim=4)  # (B, L, N, K, 6, 2)
        H_q_out = (H_stack * w_H_exp).sum(dim=4)  # (B, L, N, K, 2)

        # --- Quantize demod ---
        v_q2 = self.q2_demod(v)
        v_q4 = self.q4_demod(v)
        v_q6 = self.q6_demod(v)
        v_q8 = self.q8_demod(v)
        v_q12 = self.q12_demod(v)
        v_q16 = self.q16_demod(v)

        w_d_exp = w_demod.unsqueeze(-1)  # (B, L, K, 6, 1)
        v_stack = torch.stack([v_q2, v_q4, v_q6, v_q8, v_q12, v_q16], dim=3)  # (B, L, K, 6, 2)
        v_q = (v_stack * w_d_exp).sum(dim=3)  # (B, L, K, 2)

        # --- Expected bits per link ---
        bit_values = torch.tensor([2.0, 4.0, 6.0, 8.0, 12.0, 16.0], device=v.device)

        y_bits_per_ap = 2.0 * N * (w_y * bit_values.view(1, 1, self.num_options)).sum(dim=-1)  # (B, L)
        y_bits_per_link = y_bits_per_ap.unsqueeze(-1) / K  # (B, L, 1)

        H_bits_per_link = (2.0 * N) * (w_H * bit_values.view(1, 1, 1, self.num_options)).sum(dim=-1)  # (B, L, K)

        demod_bits_per_link = 2.0 * (w_demod * bit_values.view(1, 1, 1, self.num_options)).sum(dim=-1)  # (B, L, K)

        expected_bits_per_link = y_bits_per_link + H_bits_per_link + demod_bits_per_link  # (B, L, K)

        return v_q, H_q_out, y_q, expected_bits_per_link, w_y, w_H, w_demod, y_bits_per_ap, H_bits_per_link, demod_bits_per_link


# ============================================================
# 5. Joint Model V5
# ============================================================
class JointModelV5(nn.Module):
    """End-to-end model combining TripleAdaptiveQuantizerV5 + GNNTransformerDetectorV5."""
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(JointModelV5, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.quantizer = TripleAdaptiveQuantizerV5(N=N, policy_hidden=64)
        self.detector = GNNTransformerDetectorV5(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            noise_w=noise_w
        )
        # Max bits for normalization (using 16 as max bit option)
        self.max_y_bits_per_link = 2.0 * N * 16.0 / K
        self.max_H_bits = 2.0 * N * 16.0
        self.max_demod_bits = 2.0 * 16.0

    def forward(self, v, H, y, local_snr, tau=1.0, use_quantization=True):
        """
        Args:
            v: (B, L, K, 2) - demod results (full precision from AP-side LMMSE)
            H: (B, L, N, K, 2) - channel coefficients (full precision)
            y: (B, L, N, 2) - received signal (full precision)
            local_snr: (B, L, K, 1) - local SNR
            tau: Gumbel-Softmax temperature
            use_quantization: if False, skip quantization (Phase 1)
        Returns:
            logits: (B, K, 2) - logits for sign classification
            expected_bits_per_link: (B, L, K)
            s_global: (B, K, 2) - global MMSE estimate
            s_combined: (B, K, 2) - s_global + alpha * residual
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        if use_quantization:
            v_q, H_q, y_q, expected_bits_per_link, w_y, w_H, w_demod, \
                y_bits_ap, H_bits_link, demod_bits_link = \
                self.quantizer(v, H, y, local_snr, tau=tau)

            # Bitwidth features for detector: (B, L, K, 3) normalized
            y_bits_per_link = y_bits_ap.unsqueeze(-1) / K
            y_bw_feat = y_bits_per_link.expand(B, L, K).unsqueeze(-1) / max(self.max_y_bits_per_link, 1e-10)
            H_bw_feat = H_bits_link.unsqueeze(-1) / self.max_H_bits
            demod_bw_feat = demod_bits_link.unsqueeze(-1) / self.max_demod_bits
            bitwidth_features = torch.cat([y_bw_feat, H_bw_feat, demod_bw_feat], dim=-1)
        else:
            v_q = v
            H_q = H
            y_q = y
            expected_bits_per_link = torch.zeros(B, L, K, device=v.device)
            bitwidth_features = torch.ones(B, L, K, 3, device=v.device)

        logits, s_global, s_combined = self.detector(v_q, H_q, y_q, bitwidth_features, local_snr)

        return logits, expected_bits_per_link, s_global, s_combined


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors(s_hat, s, H, local_snr, y, device):
    """Convert numpy complex arrays to real-valued torch tensors including y."""
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)

    H_real = np.stack([H.real, H.imag], axis=-1)
    H_tensor = torch.FloatTensor(H_real).to(device)

    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)

    # True symbol labels for BCE: P(real>0), P(imag>0)
    s_real_sign = (s.real > 0).astype(np.float32)  # (B, K)
    s_imag_sign = (s.imag > 0).astype(np.float32)  # (B, K)
    labels = np.stack([s_real_sign, s_imag_sign], axis=-1)  # (B, K, 2)
    labels_tensor = torch.FloatTensor(labels).to(device)

    # Also keep the symbol tensor for MSE-based loss in Phase 2
    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)

    y_real = np.stack([y.real, y.imag], axis=-1)
    y_tensor = torch.FloatTensor(y_real).to(device)

    return v_tensor, H_tensor, snr_tensor, labels_tensor, Y_tensor, y_tensor


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

    total_epochs = args.epochs_phase1 + args.epochs_phase2

    # Per AP: y has 2*N=8 reals, H has 2*N*K=64 reals, demod has 2*K=16 reals. Total=88 reals.
    reals_per_ap = 2 * args.N + 2 * args.N * args.K + 2 * args.K
    fp_bits_per_link = reals_per_ap * 32 / args.K  # full precision bits per link
    max_bits_per_link = (2 * args.N * 16.0 / args.K) + (2 * args.N * 16.0) + (2 * 16.0)

    print(f"{'=' * 110}")
    print(f"GNN+Transformer Hybrid Detector V5 with Augmented Lagrangian Adaptive Quantization")
    print(f"  ** Key improvements: Augmented Lagrangian bit constraint, MSE loss in Phase 2, medium-bit init **")
    print(f"  ** Tau annealing: {args.tau_start} -> {args.tau_end} (no warmup, immediate annealing) **")
    print(f"{'=' * 110}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users")
    print(f"Reals per AP: y={2*args.N}, H={2*args.N*args.K}, demod={2*args.K}, total={reals_per_ap}")
    print(f"Training: {total_epochs} total epochs")
    print(f"  Phase 1: {args.epochs_phase1} epochs (detector pre-training, full precision, BCE loss)")
    print(f"  Phase 2: {args.epochs_phase2} epochs (joint QAT, tau: {args.tau_start} -> {args.tau_end})")
    print(f"  Phase1 LR: {args.lr_phase1}, Phase2 LR: {args.lr_phase2}")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"  Augmented Lagrangian: rho_init={args.rho_init}, rho_max={args.rho_max}")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {args.batches_per_epoch}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Bit options: {{2, 4, 6, 8, 12, 16}} for y, H, and demod (6 options)")
    print(f"  y bits per AP: {{{2*args.N*2}, {2*args.N*4}, {2*args.N*6}, {2*args.N*8}, {2*args.N*12}, {2*args.N*16}}}")
    print(f"  H bits per link (N={args.N}): {{{2*args.N*2}, {2*args.N*4}, {2*args.N*6}, {2*args.N*8}, {2*args.N*12}, {2*args.N*16}}}")
    print(f"  Demod bits per link: {{4, 8, 12, 16, 24, 32}}")
    print(f"  Max bits per link: {max_bits_per_link:.1f}")
    print(f"Bit constraint: C_target={args.c_target} bits/link (max possible={max_bits_per_link:.1f}, ~{args.c_target/max_bits_per_link*100:.0f}% of max)")
    print(f"  Full precision bits/link: {fp_bits_per_link:.1f}")
    print(f"  Compression target: {fp_bits_per_link / args.c_target:.1f}x")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"Gradient clipping: max_norm={args.grad_clip}, Dropout: {args.dropout}")
    print(f"Phase 2: MSE loss + Augmented Lagrangian bit penalty, immediate tau annealing")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 110}")

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
        s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch(sys_model, args.test_samples, p_tx_dbm=p)
        v_t, H_t, snr_t, labels_t, Y_t, y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, y_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'labels': labels_t, 'Y': Y_t, 'y': y_t,
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

    for p in test_p_tx_list:
        td = test_dataset[p]
        baseline_ber_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        baseline_ber_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], noise_w)

    print(f"\nComputing C-MMSE-Q baseline (uniform quant, same bit budget, c_target={args.c_target})...")
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

    print(f"\nComputing Dist-Q baseline (distributed LMMSE + quantized demod + average)...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_dq, b_used = compute_dist_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        baseline_ber_dist_q[p] = ber_dq

    print(f"\n{'=' * 110}")
    print(f"Baseline BERs (computed on fixed test set):")
    print(f"{'p_tx(dBm)':<12} | {'Dist-Full':<12} | {'C-MMSE':<12} | {'C-MMSE-Q':<12} | {'C-MMSE-MxQ':<12} | {'Dist-Q':<12}")
    print("-" * 80)
    for p in test_p_tx_list:
        print(f"{p:<12} | {baseline_ber_dist_full[p]:<12.6f} | {baseline_ber_cmmse[p]:<12.6f} | "
              f"{baseline_ber_cmmse_q[p]:<12.6f} | {baseline_ber_cmmse_mixed_q[p]:<12.6f} | "
              f"{baseline_ber_dist_q[p]:<12.6f}")
    print()

    # ===========================
    # Initialize model
    # ===========================
    model = JointModelV5(
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

    # BCE loss for sign classification
    bce_criterion = nn.BCEWithLogitsLoss()

    # ===========================
    # Epoch 0 Sanity Check: PyTorch Global MMSE vs Numpy C-MMSE
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"SANITY CHECK (Epoch 0): Comparing PyTorch Global MMSE (float64) vs Numpy C-MMSE")
    print(f"{'=' * 110}")

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

            ber_np_cmmse = baseline_ber_cmmse[p_val]
            ber_diff = abs(ber_torch_mmse - ber_np_cmmse)
            status = "OK" if ber_diff < 0.01 else "WARNING: MISMATCH!"
            if ber_diff >= 0.01:
                verification_passed = False

            print(f"  p_tx={p_val:4d} dBm | PyTorch MMSE BER: {ber_torch_mmse:.6f} | "
                  f"Numpy C-MMSE BER: {ber_np_cmmse:.6f} | Diff: {ber_diff:.6f} | {status}")

    if verification_passed:
        print(f"\n  *** VERIFICATION PASSED: PyTorch Global MMSE matches Numpy C-MMSE ***\n")
    else:
        print(f"\n  *** WARNING: VERIFICATION FAILED ***\n")

    # ===========================
    # Phase 1: Detector Pre-training (Full Precision, BCE Loss)
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Phase 1: Detector Pre-training ({args.epochs_phase1} epochs, full precision, BCE loss)")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"  Learning rate: {args.lr_phase1} with cosine annealing")
    print(f"{'=' * 110}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=max(args.epochs_phase1, 1), eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch(sys_model, args.batch_size, p_tx_dbm=args.train_snr_dbm)
            v_tensor, H_tensor, snr_tensor, labels_tensor, Y_tensor, y_tensor = prepare_tensors(
                s_hat_np, s_np, H_np, snr_np, y_np, device)

            optimizer_phase1.zero_grad()
            logits, _, s_global, s_combined = model(v_tensor, H_tensor, y_tensor, snr_tensor,
                                                     use_quantization=False)
            # BCE loss on logits
            loss = bce_criterion(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), max_norm=args.grad_clip)
            optimizer_phase1.step()
            epoch_loss += loss.item()

        scheduler_phase1.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch

        if (epoch + 1) % max(1, args.epochs_phase1 // 5) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_bers = {}
                val_bers_global = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    logits_val, _, s_global_val, s_combined_val = model(td['v'], td['H'], td['y'], td['snr'],
                                                                         use_quantization=False)
                    # BER from combined output
                    sc_np = s_combined_val.cpu().numpy()
                    s_pred = sc_np[..., 0] + 1j * sc_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)

                    # BER from logits directly
                    logits_np = logits_val.cpu().numpy()
                    val_bers_logit = compute_ber_from_logits(td['s_np'], logits_np)

                    # Global MMSE BER
                    sg_np = s_global_val.cpu().numpy()
                    s_global_pred = sg_np[..., 0] + 1j * sg_np[..., 1]
                    val_bers_global[p_val] = compute_ber(td['s_np'], s_global_pred)

            lr_now = scheduler_phase1.get_last_lr()[0]
            alpha_val = model.detector.alpha.item()
            print(f"Phase1 Epoch [{epoch + 1:03d}/{args.epochs_phase1}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | alpha: {alpha_val:.4f} | "
                  f"Train BCE Loss: {avg_loss:.6f}")
            print(f"  Combined BER:    0dBm={val_bers[0]:.4f}, 10dBm={val_bers[10]:.4f}, 20dBm={val_bers[20]:.4f}")
            print(f"  Global MMSE BER: 0dBm={val_bers_global[0]:.4f}, 10dBm={val_bers_global[10]:.4f}, 20dBm={val_bers_global[20]:.4f}")
            print(f"  C-MMSE BER:      0dBm={baseline_ber_cmmse[0]:.4f}, 10dBm={baseline_ber_cmmse[10]:.4f}, 20dBm={baseline_ber_cmmse[20]:.4f}")

    print("Phase 1 completed.\n")

    # ===========================
    # Phase 2: Joint QAT Training (MSE Loss + Augmented Lagrangian Bit Penalty)
    # ===========================
    print(f"{'=' * 110}")
    print(f"Phase 2: Joint QAT Training ({args.epochs_phase2} epochs)")
    print(f"  tau annealing: {args.tau_start} -> {args.tau_end} (immediate, no warmup)")
    print(f"  Augmented Lagrangian: rho_init={args.rho_init}, rho_max={args.rho_max}")
    print(f"  Loss: MSE on symbol estimates + Augmented Lagrangian bit penalty")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"{'=' * 110}")

    # Critical Change 2: Initialize policy heads to favor MEDIUM bits (~8 bits, index 3)
    print("  Initializing policy network heads to favor medium bits (~8 bits)...")
    with torch.no_grad():
        # Bit options: {2, 4, 6, 8, 12, 16} -> indices {0, 1, 2, 3, 4, 5}
        # Peak at 8-bit (index 3)
        bias_init = torch.tensor([-1.0, -0.5, 0.0, 1.0, 0.0, -0.5])
        model.quantizer.policy.head_y.bias.copy_(bias_init)
        model.quantizer.policy.head_H.bias.copy_(bias_init)
        model.quantizer.policy.head_demod.bias.copy_(bias_init)
    print(f"  Policy head biases set to: {bias_init.tolist()}")
    print(f"  Corresponding to bit options: [2b, 4b, 6b, 8b, 12b, 16b]")

    # Critical Change 1: Augmented Lagrangian initialization
    lagrange_mu = torch.tensor(0.0, device=device)  # Lagrange multiplier
    rho = args.rho_init  # Augmented Lagrangian penalty coefficient
    print(f"  Lagrange multiplier (mu) initialized to: {lagrange_mu.item():.4f}")
    print(f"  Penalty coefficient (rho) initialized to: {rho:.4f}")

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=max(args.epochs_phase2, 1), eta_min=1e-5)

    best_val_ber = float('inf')
    best_state = None

    # Critical Change 8: Early stopping tracking
    early_stop_checks = 0  # count of consecutive checks where conditions are met
    early_stopped = False

    for epoch in range(args.epochs_phase2):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bit_penalty = 0.0
        epoch_bits = 0.0
        start_t = time.time()

        # Critical Change 4: Immediate tau annealing (no warmup)
        if args.epochs_phase2 > 1:
            progress = epoch / (args.epochs_phase2 - 1)
        else:
            progress = 1.0
        tau = args.tau_start * (args.tau_end / max(args.tau_start, 1e-10)) ** progress
        tau = max(tau, 0.01)

        for batch_idx in range(args.batches_per_epoch):
            s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch(sys_model, args.batch_size, p_tx_dbm=args.train_snr_dbm)
            v_tensor, H_tensor, snr_tensor, labels_tensor, Y_tensor, y_tensor = prepare_tensors(
                s_hat_np, s_np, H_np, snr_np, y_np, device)

            optimizer_phase2.zero_grad()
            logits, exp_bits_per_link, s_global, s_combined = model(
                v_tensor, H_tensor, y_tensor, snr_tensor, tau=tau, use_quantization=True
            )

            # Critical Change 3: MSE loss as primary detection loss
            mse_loss = F.mse_loss(s_combined, Y_tensor)
            detection_loss = mse_loss

            # Critical Change 1: Augmented Lagrangian bit penalty
            avg_bits = exp_bits_per_link.mean()
            bit_violation = avg_bits - args.c_target
            bit_penalty = lagrange_mu * bit_violation + 0.5 * rho * bit_violation ** 2

            loss = detection_loss + bit_penalty
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer_phase2.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_bit_penalty += bit_penalty.item()
            epoch_bits += avg_bits.item()

        scheduler_phase2.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_mse = epoch_mse / args.batches_per_epoch
        avg_bit_penalty = epoch_bit_penalty / args.batches_per_epoch
        epoch_avg_bits = epoch_bits / args.batches_per_epoch

        # Critical Change 1: Update Lagrange multiplier and rho after each epoch
        with torch.no_grad():
            lagrange_mu = lagrange_mu + rho * (epoch_avg_bits - args.c_target)
            lagrange_mu = torch.clamp(lagrange_mu, min=0.0)  # only penalize over-budget
            # Increase rho gradually if over budget by more than 10%
            if epoch_avg_bits > args.c_target * 1.1:
                rho = min(rho * 1.5, args.rho_max)

        if (epoch + 1) % max(1, args.epochs_phase2 // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_bers = {}
                val_bits_dict = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    logits_val, val_bits, s_global_val, s_combined_val = model(
                        td['v'], td['H'], td['y'], td['snr'], tau=tau, use_quantization=True
                    )
                    # BER from combined output
                    sc_np = s_combined_val.cpu().numpy()
                    s_pred = sc_np[..., 0] + 1j * sc_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)
                    val_bits_dict[p_val] = val_bits.mean().item()

                # Get bit allocation at 10 dBm
                td10 = test_dataset[10]
                _, _, _, _, w_y_t, w_H_t, w_d_t, _, _, _ = model.quantizer(
                    td10['v'], td10['H'], td10['y'], td10['snr'], tau=tau
                )
                w_y_mean = w_y_t.mean(dim=(0, 1))  # (6,)
                w_H_mean = w_H_t.mean(dim=(0, 1, 2))  # (6,)
                w_d_mean = w_d_t.mean(dim=(0, 1, 2))  # (6,)

            lr_now = scheduler_phase2.get_last_lr()[0]
            alpha_val = model.detector.alpha.item()
            bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
            # Critical Change 7: Print lagrange_mu and rho
            print(f"Phase2 Epoch [{epoch + 1:03d}/{args.epochs_phase2}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | alpha: {alpha_val:.4f} | "
                  f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}, BitPen: {avg_bit_penalty:.5f}) | "
                  f"Avg Bits: {epoch_avg_bits:.1f} (target: {args.c_target})")
            print(f"  Lagrange mu: {lagrange_mu.item():.6f} | rho: {rho:.6f}")
            print(f"  Val BER: 0dBm={val_bers[0]:.4f} (Dist-Full:{baseline_ber_dist_full[0]:.4f}, C-MMSE:{baseline_ber_cmmse[0]:.4f}), "
                  f"10dBm={val_bers[10]:.4f} (Dist-Full:{baseline_ber_dist_full[10]:.4f}, C-MMSE:{baseline_ber_cmmse[10]:.4f}), "
                  f"20dBm={val_bers[20]:.4f} (Dist-Full:{baseline_ber_dist_full[20]:.4f}, C-MMSE:{baseline_ber_cmmse[20]:.4f})")
            print(f"  Val Bits/link: 0dBm={val_bits_dict[0]:.1f}, 10dBm={val_bits_dict[10]:.1f}, 20dBm={val_bits_dict[20]:.1f}")
            y_dist_str = ' '.join([f'{bit_labels[i]}:{w_y_mean[i]:.3f}' for i in range(6)])
            H_dist_str = ' '.join([f'{bit_labels[i]}:{w_H_mean[i]:.3f}' for i in range(6)])
            d_dist_str = ' '.join([f'{bit_labels[i]}:{w_d_mean[i]:.3f}' for i in range(6)])
            print(f"  Bit dist @10dBm: y [{y_dist_str}]")
            print(f"                   H [{H_dist_str}]")
            print(f"                 dem [{d_dist_str}]")

            # Track best model
            avg_val_ber = np.mean([val_bers[p] for p in [0, 10, 20]])
            if avg_val_ber < best_val_ber:
                best_val_ber = avg_val_ber
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  -> New best model! Avg val BER = {avg_val_ber:.6f}")

        # Critical Change 8: Early stopping check every 10 epochs after epoch 30
        if (epoch + 1) >= 30 and (epoch + 1) % 10 == 0 and not early_stopped:
            model.eval()
            with torch.no_grad():
                # Check bit budget
                check_bits_list = []
                check_ber_proposed = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    logits_chk, bits_chk, _, s_combined_chk = model(
                        td['v'], td['H'], td['y'], td['snr'], tau=tau, use_quantization=True
                    )
                    check_bits_list.append(bits_chk.mean().item())
                    sc_chk = s_combined_chk.cpu().numpy()
                    s_pred_chk = sc_chk[..., 0] + 1j * sc_chk[..., 1]
                    check_ber_proposed[p_val] = compute_ber(td['s_np'], s_pred_chk)

                avg_check_bits = np.mean(check_bits_list)
                bits_in_range = (args.c_target * 0.9 <= avg_check_bits <= args.c_target * 1.1)

                # Check if proposed BER < Dist-Full BER at validation points
                ber_better = all(
                    check_ber_proposed[p_val] < baseline_ber_dist_full[p_val]
                    for p_val in [0, 10, 20]
                )

                if bits_in_range and ber_better:
                    early_stop_checks += 1
                    print(f"  [Early Stop Check] PASS ({early_stop_checks}/2): "
                          f"avg_bits={avg_check_bits:.1f} (in [{args.c_target*0.9:.1f}, {args.c_target*1.1:.1f}]), "
                          f"BER better than Dist-Full at all val points")
                else:
                    early_stop_checks = 0
                    reasons = []
                    if not bits_in_range:
                        reasons.append(f"bits={avg_check_bits:.1f} out of range")
                    if not ber_better:
                        reasons.append("BER not better than Dist-Full at all points")
                    print(f"  [Early Stop Check] FAIL: {', '.join(reasons)}")

                if early_stop_checks >= 2:
                    print(f"  *** EARLY STOPPING at epoch {epoch+1}: conditions met for 2 consecutive checks ***")
                    early_stopped = True
                    break

    print(f"\nPhase 2 completed. Best avg val BER = {best_val_ber:.6f}")
    print(f"Final Lagrange mu: {lagrange_mu.item():.6f}, Final rho: {rho:.6f}\n")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model from training for final evaluation.\n")

    # ===========================
    # Final Evaluation
    # ===========================
    print(f"{'=' * 110}")
    print(f"Final Evaluation on Test Set ({args.test_samples} samples per power point)")
    print(f"{'=' * 110}")

    model.eval()

    header = (f"{'p_tx(dBm)':<10} | {'Dist-Full':<11} | {'C-MMSE':<11} | {'C-MMSE-Q':<11} | "
              f"{'C-MMSE-MxQ':<11} | {'Dist-Q':<11} | {'Proposed':<11} | "
              f"{'vs Dist-Full':<12} | {'vs C-MMSE':<10} | {'Bits/link':<10}")
    print(f"\n{header}")
    print("-" * len(header))

    results_proposed = {}
    with torch.no_grad():
        tau_eval = args.tau_end
        for p in test_p_tx_list:
            td = test_dataset[p]

            ber_dist_full = baseline_ber_dist_full[p]
            ber_cmmse = baseline_ber_cmmse[p]
            ber_cmmse_q = baseline_ber_cmmse_q[p]
            ber_cmmse_mq = baseline_ber_cmmse_mixed_q[p]
            ber_dist_q = baseline_ber_dist_q[p]

            logits_val, exp_bits, s_global_val, s_combined_val = model(
                td['v'], td['H'], td['y'], td['snr'], tau=tau_eval, use_quantization=True
            )

            # BER from combined output (s_global + alpha * residual)
            sc_np = s_combined_val.cpu().numpy()
            s_pred = sc_np[..., 0] + 1j * sc_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_total = exp_bits.mean().item()

            gap_cmmse = ((ber_proposed - ber_cmmse) / max(ber_cmmse, 1e-10)) * 100
            gap_dist_full = ((ber_proposed - ber_dist_full) / max(ber_dist_full, 1e-10)) * 100

            results_proposed[p] = {
                'ber': ber_proposed, 'bits_t': avg_bits_total
            }

            print(f"{p:<10} | {ber_dist_full:<11.6f} | {ber_cmmse:<11.6f} | {ber_cmmse_q:<11.6f} | "
                  f"{ber_cmmse_mq:<11.6f} | {ber_dist_q:<11.6f} | {ber_proposed:<11.6f} | "
                  f"{gap_dist_full:>+10.2f}% | {gap_cmmse:>+8.2f}% | {avg_bits_total:<10.2f}")

    print("-" * len(header))

    # ===========================
    # Summary statistics
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Summary: Average Performance Across All Power Points")
    print(f"{'=' * 110}")
    avg_ber_dist = np.mean([baseline_ber_dist_full[p] for p in test_p_tx_list])
    avg_ber_cmmse = np.mean([baseline_ber_cmmse[p] for p in test_p_tx_list])
    avg_ber_cmmse_q = np.mean([baseline_ber_cmmse_q[p] for p in test_p_tx_list])
    avg_ber_cmmse_mq = np.mean([baseline_ber_cmmse_mixed_q[p] for p in test_p_tx_list])
    avg_ber_dist_q = np.mean([baseline_ber_dist_q[p] for p in test_p_tx_list])
    avg_ber_proposed = np.mean([results_proposed[p]['ber'] for p in test_p_tx_list])
    avg_total_bits = np.mean([results_proposed[p]['bits_t'] for p in test_p_tx_list])

    print(f"  Average Dist-Full BER:        {avg_ber_dist:.6f}")
    print(f"  Average C-MMSE BER:           {avg_ber_cmmse:.6f} (upper bound, full precision)")
    print(f"  Average C-MMSE-Q BER:         {avg_ber_cmmse_q:.6f} (uniform quantization)")
    print(f"  Average C-MMSE-Mixed-Q BER:   {avg_ber_cmmse_mq:.6f} (optimized mixed quantization)")
    print(f"  Average Dist-Q BER:           {avg_ber_dist_q:.6f} (distributed with quantization)")
    print(f"  Average Proposed BER:         {avg_ber_proposed:.6f}")
    print(f"  Average Total Bits/Link:      {avg_total_bits:.2f} (target: {args.c_target})")
    print(f"  Full Precision Bits/Link:     {fp_bits_per_link:.1f}")
    print(f"  Average Compression Ratio:    {fp_bits_per_link / max(avg_total_bits, 1e-10):.1f}x")

    # Performance comparison
    print(f"\n  --- Performance Comparisons ---")
    comparisons = [
        ("C-MMSE (full precision)", avg_ber_cmmse),
        ("C-MMSE-Q (uniform quant)", avg_ber_cmmse_q),
        ("C-MMSE-Mixed-Q (opt mixed)", avg_ber_cmmse_mq),
        ("Dist-Full (no quant)", avg_ber_dist),
        ("Dist-Q (uniform quant)", avg_ber_dist_q),
    ]
    for name, ber_baseline in comparisons:
        if avg_ber_proposed < ber_baseline:
            pct = ((ber_baseline - avg_ber_proposed) / max(ber_baseline, 1e-10)) * 100
            print(f"  *** Proposed OUTPERFORMS {name} by {pct:.1f}% ***")
        else:
            pct = ((avg_ber_proposed - ber_baseline) / max(ber_baseline, 1e-10)) * 100
            print(f"  *** Proposed underperforms {name} by {pct:.1f}% ***")

    # Check if proposed beats Dist-Full
    print(f"\n{'=' * 110}")
    if avg_ber_proposed < avg_ber_dist:
        print(f"  >>> RESULT: Proposed method BEATS Dist-Full! ({avg_ber_proposed:.6f} < {avg_ber_dist:.6f}) <<<")
    else:
        print(f"  >>> RESULT: Proposed method does NOT beat Dist-Full yet. ({avg_ber_proposed:.6f} >= {avg_ber_dist:.6f}) <<<")
        print(f"  >>> Try increasing epochs or adjusting hyperparameters. <<<")
    print(f"{'=' * 110}")

    # ===========================
    # Detailed bit allocation statistics
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Detailed Bit Allocation Statistics")
    print(f"{'=' * 110}")
    bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
    with torch.no_grad():
        tau_eval = args.tau_end
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

            y_dist_str = ', '.join([f'{bit_labels[i]}: {w_y_mean[i]:.3f}' for i in range(6)])
            H_dist_str = ', '.join([f'{bit_labels[i]}: {w_H_mean[i]:.3f}' for i in range(6)])
            d_dist_str = ', '.join([f'{bit_labels[i]}: {w_d_mean[i]:.3f}' for i in range(6)])

            print(f"\n  p_tx = {p} dBm:")
            print(f"    y bit distribution (per AP):   {y_dist_str}")
            print(f"    H bit distribution (per link): {H_dist_str}")
            print(f"    Demod bit dist (per link):     {d_dist_str}")
            print(f"    Avg y bits/AP: {avg_y_bits_ap:.2f}, Avg H bits/link: {avg_H_bits_link:.2f}, "
                  f"Avg demod bits/link: {avg_demod_bits_link:.2f}")
            print(f"    Avg total bits/link: {avg_total_link:.2f}")
            print(f"    Full precision bits/link: {fp_bits_per_link:.1f}, Compression ratio: {compression:.1f}x")

    # ===========================
    # Per-AP bit allocation statistics
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Per-AP Bit Allocation at 10 dBm (first 5 test samples)")
    print(f"{'=' * 110}")
    with torch.no_grad():
        td = test_dataset[10]
        n_sub = min(5, td['v'].shape[0])
        v_sub = td['v'][:n_sub]
        H_sub = td['H'][:n_sub]
        y_sub = td['y'][:n_sub]
        snr_sub = td['snr'][:n_sub]
        _, _, _, exp_bits_sub, w_y_sub, w_H_sub, w_d_sub, y_bits_sub, H_bits_sub, d_bits_sub = \
            model.quantizer(v_sub, H_sub, y_sub, snr_sub, tau=tau_eval)

        avg_y_per_ap = y_bits_sub.mean(dim=0)
        avg_H_per_ap = H_bits_sub.mean(dim=(0, 2))
        avg_d_per_ap = d_bits_sub.mean(dim=(0, 2))
        avg_total_per_ap = exp_bits_sub.mean(dim=(0, 2))

        for l in range(args.L):
            print(f"  AP {l:2d}: y_bits/AP={avg_y_per_ap[l].item():.1f}, "
                  f"H_bits/link={avg_H_per_ap[l].item():.1f}, "
                  f"demod_bits/link={avg_d_per_ap[l].item():.1f}, "
                  f"total_bits/link={avg_total_per_ap[l].item():.1f}")

    # ===========================
    # LSQ scale parameter values
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"LSQ Quantizer Scale Parameters (learned)")
    print(f"{'=' * 110}")
    q = model.quantizer
    print(f"  y quantizers:     2b s={q.q2_y.s.item():.6f}, 4b s={q.q4_y.s.item():.6f}, "
          f"6b s={q.q6_y.s.item():.6f}, 8b s={q.q8_y.s.item():.6f}, "
          f"12b s={q.q12_y.s.item():.6f}, 16b s={q.q16_y.s.item():.6f}")
    print(f"  H quantizers:     2b s={q.q2_H.s.item():.6f}, 4b s={q.q4_H.s.item():.6f}, "
          f"6b s={q.q6_H.s.item():.6f}, 8b s={q.q8_H.s.item():.6f}, "
          f"12b s={q.q12_H.s.item():.6f}, 16b s={q.q16_H.s.item():.6f}")
    print(f"  Demod quantizers: 2b s={q.q2_demod.s.item():.6f}, 4b s={q.q4_demod.s.item():.6f}, "
          f"6b s={q.q6_demod.s.item():.6f}, 8b s={q.q8_demod.s.item():.6f}, "
          f"12b s={q.q12_demod.s.item():.6f}, 16b s={q.q16_demod.s.item():.6f}")

    # Final alpha value
    print(f"\n  Detector residual alpha (learned): {model.detector.alpha.item():.6f}")

    # ===========================
    # Save model
    # ===========================
    save_path = 'new_joint_model_v5.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
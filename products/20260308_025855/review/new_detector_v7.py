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
    parser = argparse.ArgumentParser(description="Cell-Free MIMO Detector V7: Demod + Side-Info Adaptive Quantization")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=8, help="Phase 1 epochs (detector pre-training, full precision)")
    parser.add_argument("--epochs_phase2", type=int, default=30, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=50, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=5e-4, help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=3e-4, help="Learning rate for Phase 2")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for GNN")
    parser.add_argument("--c_target", type=float, default=80.0, help="Target average bits per (AP, user) link")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--train_snr_dbm", type=float, default=12.0, help="Fixed training SNR in dBm")
    parser.add_argument("--tau_start", type=float, default=3.0, help="Gumbel-Softmax temperature start")
    parser.add_argument("--tau_end", type=float, default=0.1, help="Gumbel-Softmax temperature end")
    parser.add_argument("--lambda_bit_start", type=float, default=0.001, help="Initial bit penalty weight")
    parser.add_argument("--lambda_bit_max", type=float, default=0.5, help="Max bit penalty weight")
    return parser.parse_args()


# ============================================================
# 1. Data Generation
# ============================================================
def generate_data_batch(sys_model, batch_size, p_tx_dbm):
    """
    Generate batch of Cell-Free MIMO data with side information.

    Each AP computes locally (full precision):
    - s_hat_{l,k}: local LMMSE estimate (complex, 2 reals)
    - g_{l,k} = ||h_{l,:,k}||^2: channel gain (1 real)
    - interference_{l,k} = sum_{j!=k} ||h_{l,:,j}||^2: interference power (1 real)
    - local_mse_{l,k}: estimated MSE of local estimate (1 real)

    Returns:
        s_hat: (B, L, K) complex - local LMMSE estimates
        s: (B, K) complex - true QPSK symbols
        H: (B, L, N, K) complex - full channel (for baselines only)
        local_snr: (B, L, K) real - local SNR (log-normalized)
        y: (B, L, N) complex - received signal (for baselines only)
        side_info: (B, L, K, 3) real - [channel_gain, interference, local_mse]
    """
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta

    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    beta_expanded = beta_w[np.newaxis, :, np.newaxis, :]
    H = np.sqrt(beta_expanded) * h_small

    bits = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(bits) / np.sqrt(2)

    y_clean = np.einsum('blnk,bk->bln', H, s)
    z = (np.random.randn(batch_size, sys_model.L, sys_model.N) +
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z

    # Local LMMSE
    H_H = H.conj().transpose(0, 1, 3, 2)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H @ H_H + noise_cov
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_H @ R_y_inv
    s_hat = (W_l @ y[..., np.newaxis]).squeeze(-1)

    # Side information computed at the AP
    h_power = np.sum(np.abs(H) ** 2, axis=2)  # (B, L, K) = ||h_{l,:,k}||^2
    total_power = h_power.sum(axis=2, keepdims=True)  # (B, L, 1)
    interference = total_power - h_power  # (B, L, K)

    # Local SNR (for feature encoding)
    local_snr = 10 * np.log10(h_power / sys_model.noise_w + 1e-12) / 10.0

    # Local MSE estimate: MSE_k approx 1/(1 + SINR_k)
    sinr = h_power / (interference + sys_model.noise_w + 1e-12)
    local_mse = 1.0 / (1.0 + sinr)

    # Normalize side info for quantization stability
    # channel_gain: log scale
    channel_gain_log = np.log10(h_power + 1e-15)
    # interference: log scale
    interference_log = np.log10(interference + 1e-15)

    side_info = np.stack([channel_gain_log, interference_log, local_mse], axis=-1)  # (B, L, K, 3)

    return s_hat, s, H, local_snr, y, side_info


def compute_ber(s_true, s_pred_complex):
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


# ============================================================
# 2. Baselines
# ============================================================
def compute_dist_full_ber(s_hat, s):
    s_avg = s_hat.mean(axis=1)
    return compute_ber(s, s_avg)


def uniform_quantize_np(x_real, num_bits):
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
    B, L, N, K = H.shape
    H_all = H.reshape(B, L * N, K)
    y_all = y.reshape(B, L * N)
    H_H = H_all.conj().transpose(0, 2, 1)
    HHH = H_H @ H_all
    noise_eye = noise_w * np.eye(K).reshape(1, K, K)
    R = HHH + noise_eye
    HHy = H_H @ y_all[..., np.newaxis]
    R_inv = np.linalg.inv(R)
    s_hat = (R_inv @ HHy).squeeze(-1)
    return compute_ber(s, s_hat), s_hat


def compute_dist_q_ber(s_hat, s, c_target):
    """Dist-Q: uniform quantization of demod, same bit budget."""
    B, L, K = s_hat.shape
    # Budget: c_target bits per link, demod has 2 reals
    # bits_per_real = c_target / 2
    b_per_real = c_target / 2.0
    b_use = max(1, int(np.round(b_per_real)))
    b_use = min(b_use, 16)

    s_hat_q = np.zeros_like(s_hat)
    for l in range(L):
        s_hat_q[:, l, :] = (uniform_quantize_np(s_hat[:, l, :].real, b_use) +
                             1j * uniform_quantize_np(s_hat[:, l, :].imag, b_use))
    s_avg = s_hat_q.mean(axis=1)
    return compute_ber(s, s_avg), b_use


def compute_cmmse_q_detection(H, y, s, noise_w, c_target, K, N):
    """C-MMSE-Q: Centralized MMSE with uniformly quantized H and y."""
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)
    best_ber = 1.0
    best_b = 1

    for b_use in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
        actual_bits = reals_per_ap * b_use
        if actual_bits > total_bits_per_ap * 1.05:
            continue
        H_q = uniform_quantize_np(H.real, b_use) + 1j * uniform_quantize_np(H.imag, b_use)
        y_q = uniform_quantize_np(y.real, b_use) + 1j * uniform_quantize_np(y.imag, b_use)
        H_all = H_q.reshape(B, L * N_ant, K_users)
        y_all = y_q.reshape(B, L * N_ant)
        H_H = H_all.conj().transpose(0, 2, 1)
        HHH = H_H @ H_all
        noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
        R = HHH + noise_eye + 1e-10 * np.eye(K_users).reshape(1, K_users, K_users)
        HHy = H_H @ y_all[..., np.newaxis]
        try:
            R_inv = np.linalg.inv(R)
            s_hat = (R_inv @ HHy).squeeze(-1)
            ber = compute_ber(s, s_hat)
            if ber < best_ber:
                best_ber = ber
                best_b = b_use
        except:
            continue

    actual = reals_per_ap * best_b
    print(f"    C-MMSE-Q: best b={best_b}, actual bits/AP={actual} (target={total_bits_per_ap:.0f}), BER={best_ber:.6f}")
    return best_ber, best_b


def compute_mmse_weighted_ber(s_hat, s, side_info):
    """
    Oracle MMSE-weighted combining using full-precision side info.
    Weight each AP's estimate by its channel gain (SINR-based weighting).

    s_hat: (B, L, K) complex
    side_info: (B, L, K, 3) - [channel_gain_log, interference_log, local_mse]
    """
    B, L, K = s_hat.shape
    # Channel gain (linear) from log10 representation
    channel_gain = 10 ** side_info[..., 0]  # (B, L, K)

    # MMSE optimal weight proportional to SINR = channel_gain / (interference + noise)
    # For simplicity use channel_gain as weight (proportional to quality)
    weights = channel_gain / (channel_gain.sum(axis=1, keepdims=True) + 1e-12)  # (B, L, K)
    s_weighted = (s_hat * weights).sum(axis=1)  # (B, K)
    return compute_ber(s, s_weighted)


# ============================================================
# 3. Demod + Side-Info Quantizer V7
# ============================================================

class DualAdaptiveQuantizerV7(nn.Module):
    """
    Dual adaptive quantization for demod results AND side information.

    Per link (l, k):
    - Demod: s_hat_{l,k} has 2 reals -> quantized with b_demod bits per real
    - Side info: [channel_gain, interference, mse] has 3 reals -> quantized with b_side bits per real

    Total bits per link = 2 * b_demod + 3 * b_side
    Target: c_target bits per link on average

    Bit options for both: {2, 4, 6, 8, 12, 16} bits per real
    """
    def __init__(self, num_options=6, policy_hidden=48):
        super().__init__()
        self.num_options = num_options
        self.bit_options = [2, 4, 6, 8, 12, 16]

        # Policy network: decides both demod and side-info bit allocations
        # Input: [demod_real, demod_imag, snr, mse, magnitude, ch_gain, interf] = 7
        self.policy_shared = nn.Sequential(
            nn.Linear(7, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
        )
        self.head_demod = nn.Linear(policy_hidden, num_options)
        self.head_side = nn.Linear(policy_hidden, num_options)

        # Initialize to favor ~8 bits (index 3)
        with torch.no_grad():
            bias_init = torch.tensor([-1.0, -0.3, 0.2, 1.0, 0.2, -0.3])
            self.head_demod.bias.copy_(bias_init)
            self.head_side.bias.copy_(bias_init)

        # LSQ quantizers for demod (6 levels)
        self.q_demod = nn.ModuleList([
            LSQQuantizer(num_bits=b, init_s=max(0.5 / (2**(b-2)), 0.001))
            for b in [2, 4, 6, 8, 12, 16]
        ])

        # LSQ quantizers for side info (6 levels)
        self.q_side = nn.ModuleList([
            LSQQuantizer(num_bits=b, init_s=max(0.5 / (2**(b-2)), 0.001))
            for b in [2, 4, 6, 8, 12, 16]
        ])

    def forward(self, v, side_info, local_snr, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - full precision demod (real, imag)
            side_info: (B, L, K, 3) - [channel_gain_log, interference_log, mse]
            local_snr: (B, L, K, 1)
            tau: Gumbel-Softmax temperature
        Returns:
            v_q: (B, L, K, 2) - quantized demod
            side_q: (B, L, K, 3) - quantized side info
            bits_per_link: (B, L, K)
            w_demod: (B, L, K, 6) - demod bit weights
            w_side: (B, L, K, 6) - side info bit weights
        """
        B, L, K, _ = v.shape

        magnitude = torch.sqrt(v[..., 0:1]**2 + v[..., 1:2]**2 + 1e-10)
        policy_input = torch.cat([v, local_snr, side_info[..., 2:3], magnitude,
                                   side_info[..., 0:1], side_info[..., 1:2]], dim=-1)  # 7 dims

        h = self.policy_shared(policy_input)
        logits_demod = self.head_demod(h)  # (B, L, K, 6)
        logits_side = self.head_side(h)    # (B, L, K, 6)

        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)
        w_side = F.gumbel_softmax(logits_side, tau=tau, hard=True)

        # Quantize demod at each level
        v_levels = torch.stack([q(v) for q in self.q_demod], dim=3)  # (B, L, K, 6, 2)
        w_d_exp = w_demod.unsqueeze(-1)  # (B, L, K, 6, 1)
        v_q = (v_levels * w_d_exp).sum(dim=3)

        # Quantize side info at each level
        side_levels = torch.stack([q(side_info) for q in self.q_side], dim=3)  # (B, L, K, 6, 3)
        w_s_exp = w_side.unsqueeze(-1)  # (B, L, K, 6, 1)
        side_q = (side_levels * w_s_exp).sum(dim=3)

        # Bits per link: 2*b_demod + 3*b_side
        bit_values = torch.tensor([2.0, 4.0, 6.0, 8.0, 12.0, 16.0], device=v.device)
        demod_bits = 2.0 * (w_demod * bit_values.view(1, 1, 1, self.num_options)).sum(dim=-1)
        side_bits = 3.0 * (w_side * bit_values.view(1, 1, 1, self.num_options)).sum(dim=-1)
        bits_per_link = demod_bits + side_bits

        return v_q, side_q, bits_per_link, w_demod, w_side, demod_bits, side_bits


# ============================================================
# 4. GNN Detector V7
# ============================================================

class FeatureEncoderV7(nn.Module):
    """Encode demod + side info into node features."""
    def __init__(self, hidden_dim):
        super().__init__()
        # Input: demod(2) + side_info(3) + snr(1) + magnitude(1) + phase(1) = 8
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, demod_q, side_q, local_snr):
        magnitude = torch.sqrt(demod_q[..., 0:1]**2 + demod_q[..., 1:2]**2 + 1e-10)
        phase = torch.atan2(demod_q[..., 1:2], demod_q[..., 0:1] + 1e-10) / math.pi
        x = torch.cat([demod_q, side_q, local_snr, magnitude, phase], dim=-1)
        return self.mlp(x)


class APAttentionLayerV7(nn.Module):
    """Attention over APs for each user."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        h2, _ = self.attn(h, h, h)
        h = self.norm1(h + h2)
        h2 = self.ffn(h)
        h = self.norm2(h + h2)
        return h


class UserICLayerV7(nn.Module):
    """Inter-user interference cancellation via attention."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        h2, _ = self.attn(h, h, h)
        h = self.norm1(h + h2)
        h2 = self.ffn(h)
        h = self.norm2(h + h2)
        return h


class GNNDetectorV7(nn.Module):
    """
    GNN Detector V7 with side-information-aware combining.

    Architecture:
    1. Encode quantized demod + quantized side info -> node features
    2. Quality-aware AP weighting: use side info to compute MMSE-like combining weights
    3. AP attention layers for learned combining
    4. User IC for interference cancellation
    5. Residual: output = quality_weighted_avg + learned_correction

    Key insight: The side information (channel gain, interference) enables
    the network to approximate MMSE combining weights, which is the key
    to outperforming equal-weight averaging (Dist-Full).
    """
    def __init__(self, L, K, hidden_dim=64, num_gnn_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim

        # Feature encoder
        self.encoder = FeatureEncoderV7(hidden_dim)

        # Quality-aware weight predictor
        # Uses side info to predict per-AP combining weights
        self.quality_weight_net = nn.Sequential(
            nn.Linear(3, 32),  # side_info: [ch_gain, interf, mse]
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # AP attention layers
        self.ap_layers = nn.ModuleList([
            APAttentionLayerV7(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # Learned AP aggregation
        self.ap_agg_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # User IC
        self.user_ic = UserICLayerV7(hidden_dim, num_heads, dropout)

        # Output head: correction to quality-weighted average
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        # Initialize correction to near-zero
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        # Learnable blending parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, demod_q, side_q, local_snr):
        """
        Args:
            demod_q: (B, L, K, 2) - quantized demod
            side_q: (B, L, K, 3) - quantized side info
            local_snr: (B, L, K, 1)
        Returns:
            s_est: (B, K, 2) - final symbol estimate
            s_quality_avg: (B, K, 2) - quality-weighted average (analytical baseline)
            quality_weights: (B, L, K) - learned quality weights
        """
        B, L, K, _ = demod_q.shape

        # 1. Quality-weighted average (analytical, using side info)
        # Weight based on channel gain / (interference + noise_proxy)
        # side_q: [ch_gain_log, interf_log, mse]
        quality_logits = self.quality_weight_net(side_q).squeeze(-1)  # (B, L, K)
        quality_weights = torch.softmax(quality_logits, dim=1)  # (B, L, K)
        s_quality_avg = (demod_q * quality_weights.unsqueeze(-1)).sum(dim=1)  # (B, K, 2)

        # 2. Encode features
        h = self.encoder(demod_q, side_q, local_snr)  # (B, L, K, D)

        # 3. AP attention for each user
        h_per_user = h.permute(0, 2, 1, 3).reshape(B * K, L, self.hidden_dim)
        for layer in self.ap_layers:
            h_per_user = layer(h_per_user)

        # 4. Learned AP aggregation
        agg_weights = self.ap_agg_net(h_per_user)  # (B*K, L, 1)
        agg_weights = torch.softmax(agg_weights, dim=1)
        user_features = (h_per_user * agg_weights).sum(dim=1)  # (B*K, D)
        user_features = user_features.reshape(B, K, self.hidden_dim)

        # 5. User IC
        user_features = self.user_ic(user_features)

        # 6. Correction
        correction = self.correction_head(user_features)  # (B, K, 2)

        # 7. Final: quality_avg + alpha * correction
        s_est = s_quality_avg + self.alpha * correction

        return s_est, s_quality_avg, quality_weights


# ============================================================
# 5. Joint Model V7
# ============================================================

class JointModelV7(nn.Module):
    def __init__(self, L, K, hidden_dim=64, num_gnn_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        self.quantizer = DualAdaptiveQuantizerV7(policy_hidden=48)
        self.detector = GNNDetectorV7(L, K, hidden_dim, num_gnn_layers, num_heads, dropout)

    def forward(self, v, side_info, local_snr, tau=1.0, use_quantization=True):
        """
        Args:
            v: (B, L, K, 2) - full precision demod
            side_info: (B, L, K, 3) - full precision side info
            local_snr: (B, L, K, 1)
        Returns:
            s_est, bits_per_link, s_quality_avg, w_demod, w_side, demod_bits, side_bits, quality_weights
        """
        if use_quantization:
            v_q, side_q, bits_per_link, w_demod, w_side, demod_bits, side_bits = \
                self.quantizer(v, side_info, local_snr, tau)
        else:
            v_q = v
            side_q = side_info
            B = v.shape[0]
            bits_per_link = torch.zeros(B, self.L, self.K, device=v.device)
            w_demod = w_side = None
            demod_bits = side_bits = torch.zeros(B, self.L, self.K, device=v.device)

        s_est, s_quality_avg, quality_weights = self.detector(v_q, side_q, local_snr)
        return s_est, bits_per_link, s_quality_avg, w_demod, w_side, demod_bits, side_bits, quality_weights


# ============================================================
# 6. Helpers
# ============================================================

def prepare_tensors(s_hat, s, local_snr, side_info, device):
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)
    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)
    side_tensor = torch.FloatTensor(side_info).to(device)
    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)
    return v_tensor, snr_tensor, side_tensor, Y_tensor


# ============================================================
# 7. Main
# ============================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Bits per link: 2*b_demod + 3*b_side
    # With b_demod=b_side=16: max = 2*16 + 3*16 = 80 bits
    # With b_demod=b_side=2: min = 2*2 + 3*2 = 10 bits
    # FP: 2*32 + 3*32 = 160 bits per link
    max_bits_per_link = 2 * 16 + 3 * 16  # 80
    min_bits_per_link = 2 * 2 + 3 * 2    # 10
    fp_bits_per_link = 2 * 32 + 3 * 32   # 160

    print(f"{'='*100}")
    print(f"Cell-Free MIMO Detector V7: Demod + Side-Info Adaptive Quantization")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} ant/AP, K={args.K} users")
    print(f"Architecture:")
    print(f"  - APs compute local LMMSE at full precision")
    print(f"  - Each AP sends per-user: quantized demod (2 reals) + side info (3 reals)")
    print(f"  - Side info: channel_gain_log, interference_log, local_mse")
    print(f"  - Adaptive bit allocation for demod and side info independently")
    print(f"  - Bits per link: 2*b_demod + 3*b_side, range [{min_bits_per_link}, {max_bits_per_link}]")
    print(f"  - GNN uses quality-weighted combining + learned correction + user IC")
    print(f"Bit budget: c_target={args.c_target} bits/link (FP={fp_bits_per_link}, max={max_bits_per_link})")
    if args.c_target > max_bits_per_link:
        print(f"  NOTE: c_target > max possible = {max_bits_per_link}, will be effectively capped")
    print(f"Training: Phase1={args.epochs_phase1} eps (FP), Phase2={args.epochs_phase2} eps (QAT)")
    print(f"  LR: Phase1={args.lr_phase1}, Phase2={args.lr_phase2}")
    print(f"  tau: {args.tau_start} -> {args.tau_end}")
    print(f"  lambda_bit: {args.lambda_bit_start} -> {args.lambda_bit_max}")
    print(f"Hidden dim: {args.hidden_dim}, GNN layers: {args.num_gnn_layers}")
    print(f"Batch: {args.batch_size} x {args.batches_per_epoch}/epoch, Seed: {args.seed}")
    print(f"{'='*100}")

    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()
    noise_w = sys_model.noise_w

    # Test set
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"\nGenerating fixed test set ({args.test_samples} samples/point)...")
    for p in test_p_tx_list:
        s_hat, s, H, snr, y, side = generate_data_batch(sys_model, args.test_samples, p)
        v_t, snr_t, side_t, Y_t = prepare_tensors(s_hat, s, snr, side, device)
        test_dataset[p] = {
            'v': v_t, 'snr': snr_t, 'side': side_t, 'Y': Y_t,
            's_hat_np': s_hat, 's_np': s, 'H_np': H, 'y_np': y,
            'side_np': side
        }
    print("Test set ready.\n")

    # Baselines
    print("Computing baselines...")
    bl_dist_full = {}
    bl_cmmse = {}
    bl_dist_q = {}
    bl_cmmse_q = {}
    bl_mmse_wt = {}

    for p in test_p_tx_list:
        td = test_dataset[p]
        bl_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        bl_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], noise_w)
        bl_dist_q[p], _ = compute_dist_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        bl_mmse_wt[p] = compute_mmse_weighted_ber(td['s_hat_np'], td['s_np'], td['side_np'])

    print(f"\nC-MMSE-Q baselines:")
    for p in test_p_tx_list:
        td = test_dataset[p]
        bl_cmmse_q[p], _ = compute_cmmse_q_detection(td['H_np'], td['y_np'], td['s_np'], noise_w,
                                                       args.c_target, args.K, args.N)

    print(f"\n{'='*100}")
    print(f"Baselines:")
    print(f"{'p_tx':<8} | {'Dist-Full':<12} | {'MMSE-Wt':<12} | {'C-MMSE':<12} | {'C-MMSE-Q':<12} | {'Dist-Q':<12}")
    print("-" * 80)
    for p in test_p_tx_list:
        print(f"{p:<8} | {bl_dist_full[p]:<12.6f} | {bl_mmse_wt[p]:<12.6f} | "
              f"{bl_cmmse[p]:<12.6f} | {bl_cmmse_q[p]:<12.6f} | {bl_dist_q[p]:<12.6f}")
    print()

    # Model
    model = JointModelV7(
        L=args.L, K=args.K, hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers, num_heads=args.num_heads, dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    det_params = sum(p.numel() for p in model.detector.parameters())
    q_params = sum(p.numel() for p in model.quantizer.parameters())
    print(f"Parameters: Total={total_params:,}, Detector={det_params:,}, Quantizer={q_params:,}")

    # ======== Phase 1: Full precision pre-training ========
    print(f"\n{'='*100}")
    print(f"Phase 1: Detector pre-training ({args.epochs_phase1} epochs, full precision)")
    print(f"{'='*100}")

    optimizer1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.epochs_phase1, 1), eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(args.batches_per_epoch):
            s_hat, s, H, snr, y, side = generate_data_batch(sys_model, args.batch_size, args.train_snr_dbm)
            v_t, snr_t, side_t, Y_t = prepare_tensors(s_hat, s, snr, side, device)

            optimizer1.zero_grad()
            s_est, _, s_qa, _, _, _, _, qw = model(v_t, side_t, snr_t, use_quantization=False)
            loss = F.mse_loss(s_est, Y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), args.grad_clip)
            optimizer1.step()
            epoch_loss += loss.item()

        scheduler1.step()
        avg_loss = epoch_loss / args.batches_per_epoch
        elapsed = time.time() - t0

        if (epoch + 1) % max(1, args.epochs_phase1 // 4) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                bers = {}
                bers_qa = {}
                for p_v in [0, 10, 20]:
                    td = test_dataset[p_v]
                    s_est_val, _, s_qa_val, _, _, _, _, _ = model(td['v'], td['side'], td['snr'],
                                                                    use_quantization=False)
                    pred = s_est_val.cpu().numpy()
                    bers[p_v] = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
                    pred_qa = s_qa_val.cpu().numpy()
                    bers_qa[p_v] = compute_ber(td['s_np'], pred_qa[..., 0] + 1j * pred_qa[..., 1])

            alpha_val = model.detector.alpha.item()
            print(f"Phase1 [{epoch+1:03d}/{args.epochs_phase1}] | {elapsed:.1f}s | Loss: {avg_loss:.6f} | "
                  f"alpha: {alpha_val:.4f}")
            print(f"  Full est BER: 0dB={bers[0]:.4f}({bl_dist_full[0]:.4f}) "
                  f"10dB={bers[10]:.4f}({bl_dist_full[10]:.4f}) "
                  f"20dB={bers[20]:.4f}({bl_dist_full[20]:.4f})")
            print(f"  QualAvg BER:  0dB={bers_qa[0]:.4f}({bl_mmse_wt[0]:.4f}) "
                  f"10dB={bers_qa[10]:.4f}({bl_mmse_wt[10]:.4f}) "
                  f"20dB={bers_qa[20]:.4f}({bl_mmse_wt[20]:.4f})")

    print("Phase 1 done.\n")

    # ======== Phase 2: Joint QAT ========
    print(f"{'='*100}")
    print(f"Phase 2: Joint QAT ({args.epochs_phase2} epochs)")
    print(f"{'='*100}")

    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.epochs_phase2, 1), eta_min=1e-5)

    best_val_ber = float('inf')
    best_state = None

    for epoch in range(args.epochs_phase2):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bits = 0.0
        epoch_demod_bits = 0.0
        epoch_side_bits = 0.0
        t0 = time.time()

        # Annealing
        if args.epochs_phase2 > 1:
            progress = epoch / (args.epochs_phase2 - 1)
        else:
            progress = 1.0
        tau = args.tau_start * (args.tau_end / max(args.tau_start, 1e-10)) ** progress
        tau = max(tau, 0.01)
        lambda_bit = args.lambda_bit_start + (args.lambda_bit_max - args.lambda_bit_start) * progress

        for _ in range(args.batches_per_epoch):
            s_hat, s, H, snr, y, side = generate_data_batch(sys_model, args.batch_size, args.train_snr_dbm)
            v_t, snr_t, side_t, Y_t = prepare_tensors(s_hat, s, snr, side, device)

            optimizer2.zero_grad()
            s_est, bits_per_link, s_qa, w_d, w_s, d_bits, s_bits, qw = \
                model(v_t, side_t, snr_t, tau=tau, use_quantization=True)

            mse_loss = F.mse_loss(s_est, Y_t)

            # Bit constraint
            avg_bits = bits_per_link.mean()
            bit_violation = avg_bits - args.c_target
            # Two-sided soft penalty: quadratic for over-budget, small linear for under
            if args.c_target < max_bits_per_link:
                bit_loss = lambda_bit * (F.relu(bit_violation) ** 2 + 0.01 * F.relu(-bit_violation))
            else:
                bit_loss = torch.tensor(0.0, device=device)

            loss = mse_loss + bit_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer2.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_bits += avg_bits.item()
            epoch_demod_bits += d_bits.mean().item()
            epoch_side_bits += s_bits.mean().item()

        scheduler2.step()
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_mse = epoch_mse / args.batches_per_epoch
        avg_bits_ep = epoch_bits / args.batches_per_epoch
        avg_db = epoch_demod_bits / args.batches_per_epoch
        avg_sb = epoch_side_bits / args.batches_per_epoch
        elapsed = time.time() - t0

        if (epoch + 1) % max(1, args.epochs_phase2 // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                bers = {}
                bits_dict = {}
                for p_v in [0, 10, 20]:
                    td = test_dataset[p_v]
                    s_est_val, bits_val, _, _, _, _, _, _ = \
                        model(td['v'], td['side'], td['snr'], tau=tau, use_quantization=True)
                    pred = s_est_val.cpu().numpy()
                    bers[p_v] = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
                    bits_dict[p_v] = bits_val.mean().item()

                # Bit distribution at 10dBm
                td10 = test_dataset[10]
                _, _, _, w_d10, w_s10, _, _, _ = \
                    model(td10['v'], td10['side'], td10['snr'], tau=tau, use_quantization=True)
                if w_d10 is not None:
                    wd_mean = w_d10.mean(dim=(0, 1, 2))
                    ws_mean = w_s10.mean(dim=(0, 1, 2))
                    bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
                    wd_str = ' '.join([f'{bit_labels[i]}:{wd_mean[i]:.3f}' for i in range(6)])
                    ws_str = ' '.join([f'{bit_labels[i]}:{ws_mean[i]:.3f}' for i in range(6)])
                else:
                    wd_str = ws_str = "N/A"

            alpha_val = model.detector.alpha.item()
            avg_val_ber = np.mean([bers[p_v] for p_v in [0, 10, 20]])

            print(f"Phase2 [{epoch+1:03d}/{args.epochs_phase2}] | {elapsed:.1f}s | tau={tau:.3f} | "
                  f"λ_bit={lambda_bit:.4f} | Loss={avg_loss:.5f} (MSE={avg_mse:.5f}) | "
                  f"alpha={alpha_val:.4f}")
            print(f"  Bits: total={avg_bits_ep:.1f}/{args.c_target} (demod={avg_db:.1f}, side={avg_sb:.1f})")
            print(f"  BER: 0dB={bers[0]:.4f}({bl_dist_full[0]:.4f}/{bl_mmse_wt[0]:.4f}) "
                  f"10dB={bers[10]:.4f}({bl_dist_full[10]:.4f}/{bl_mmse_wt[10]:.4f}) "
                  f"20dB={bers[20]:.4f}({bl_dist_full[20]:.4f}/{bl_mmse_wt[20]:.4f})")
            print(f"  Val bits: {bits_dict[0]:.1f}/{bits_dict[10]:.1f}/{bits_dict[20]:.1f}")
            print(f"  Demod dist @10dB: [{wd_str}]")
            print(f"  Side  dist @10dB: [{ws_str}]")

            if avg_val_ber < best_val_ber:
                best_val_ber = avg_val_ber
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  -> Best model! avg BER={avg_val_ber:.6f}")

    print(f"\nPhase 2 done. Best avg val BER = {best_val_ber:.6f}\n")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model.\n")

    # ======== Final Evaluation ========
    print(f"{'='*100}")
    print(f"Final Evaluation ({args.test_samples} samples/point)")
    print(f"{'='*100}")

    model.eval()
    header = (f"{'p_tx':<8} | {'Dist-Full':<11} | {'MMSE-Wt':<11} | {'C-MMSE':<11} | "
              f"{'C-MMSE-Q':<11} | {'Dist-Q':<11} | {'Proposed':<11} | "
              f"{'vs DF':<10} | {'vs MW':<10} | {'Bits':<8} | {'D-bits':<8} | {'S-bits':<8}")
    print(f"\n{header}")
    print("-" * len(header))

    results = {}
    with torch.no_grad():
        tau_eval = args.tau_end
        for p in test_p_tx_list:
            td = test_dataset[p]
            s_est_val, bits_val, _, _, _, d_bits_val, s_bits_val, _ = \
                model(td['v'], td['side'], td['snr'], tau=tau_eval, use_quantization=True)
            pred = s_est_val.cpu().numpy()
            ber_prop = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
            avg_bits = bits_val.mean().item()
            avg_db = d_bits_val.mean().item()
            avg_sb = s_bits_val.mean().item()

            gap_df = ((ber_prop - bl_dist_full[p]) / max(bl_dist_full[p], 1e-10)) * 100
            gap_mw = ((ber_prop - bl_mmse_wt[p]) / max(bl_mmse_wt[p], 1e-10)) * 100

            results[p] = {'ber': ber_prop, 'bits': avg_bits, 'dbits': avg_db, 'sbits': avg_sb}

            print(f"{p:<8} | {bl_dist_full[p]:<11.6f} | {bl_mmse_wt[p]:<11.6f} | "
                  f"{bl_cmmse[p]:<11.6f} | {bl_cmmse_q[p]:<11.6f} | {bl_dist_q[p]:<11.6f} | "
                  f"{ber_prop:<11.6f} | {gap_df:>+8.1f}% | {gap_mw:>+8.1f}% | "
                  f"{avg_bits:<8.1f} | {avg_db:<8.1f} | {avg_sb:<8.1f}")

    print("-" * len(header))

    # Summary
    print(f"\n{'='*100}")
    print(f"Summary")
    print(f"{'='*100}")
    avg_df = np.mean([bl_dist_full[p] for p in test_p_tx_list])
    avg_mw = np.mean([bl_mmse_wt[p] for p in test_p_tx_list])
    avg_cm = np.mean([bl_cmmse[p] for p in test_p_tx_list])
    avg_cq = np.mean([bl_cmmse_q[p] for p in test_p_tx_list])
    avg_dq = np.mean([bl_dist_q[p] for p in test_p_tx_list])
    avg_prop = np.mean([results[p]['ber'] for p in test_p_tx_list])
    avg_bits = np.mean([results[p]['bits'] for p in test_p_tx_list])
    avg_dbits = np.mean([results[p]['dbits'] for p in test_p_tx_list])
    avg_sbits = np.mean([results[p]['sbits'] for p in test_p_tx_list])

    print(f"  Dist-Full (no quant):       {avg_df:.6f}")
    print(f"  MMSE-Weighted (oracle wt):  {avg_mw:.6f}")
    print(f"  C-MMSE (full precision):    {avg_cm:.6f}")
    print(f"  C-MMSE-Q (uniform quant):   {avg_cq:.6f}")
    print(f"  Dist-Q (uniform quant):     {avg_dq:.6f}")
    print(f"  Proposed V7:                {avg_prop:.6f}")
    print(f"  Avg bits/link:              {avg_bits:.1f} (target: {args.c_target})")
    print(f"    Demod bits/link:          {avg_dbits:.1f}")
    print(f"    Side info bits/link:      {avg_sbits:.1f}")
    print(f"  FP bits/link:               {fp_bits_per_link}")
    print(f"  Compression:                {fp_bits_per_link/max(avg_bits, 1e-10):.1f}x")

    comparisons = [
        ("Dist-Full", avg_df), ("MMSE-Weighted", avg_mw),
        ("C-MMSE", avg_cm), ("C-MMSE-Q", avg_cq), ("Dist-Q", avg_dq)
    ]
    for name, bl in comparisons:
        if avg_prop < bl:
            pct = ((bl - avg_prop) / max(bl, 1e-10)) * 100
            print(f"  >> OUTPERFORMS {name} by {pct:.1f}%")
        else:
            pct = ((avg_prop - bl) / max(bl, 1e-10)) * 100
            print(f"  >> Underperforms {name} by {pct:.1f}%")

    # Check key result
    print(f"\n{'='*100}")
    beats_df = avg_prop < avg_df
    beats_mw = avg_prop < avg_mw
    if beats_df:
        print(f"  >>> RESULT: Proposed V7 BEATS Dist-Full! ({avg_prop:.6f} < {avg_df:.6f}) <<<")
    else:
        print(f"  >>> RESULT: Proposed V7 does NOT beat Dist-Full. ({avg_prop:.6f} >= {avg_df:.6f}) <<<")
    if beats_mw:
        print(f"  >>> Proposed V7 BEATS MMSE-Weighted oracle! ({avg_prop:.6f} < {avg_mw:.6f}) <<<")
    print(f"{'='*100}")

    # Bit allocation details
    print(f"\n{'='*100}")
    print(f"Bit Allocation Details")
    print(f"{'='*100}")
    bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
    with torch.no_grad():
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]
            _, bits_val, _, w_d, w_s, d_b, s_b, _ = \
                model(td['v'], td['side'], td['snr'], tau=args.tau_end, use_quantization=True)
            avg_b = bits_val.mean().item()
            avg_db_val = d_b.mean().item()
            avg_sb_val = s_b.mean().item()
            if w_d is not None:
                wd_m = w_d.mean(dim=(0, 1, 2))
                ws_m = w_s.mean(dim=(0, 1, 2))
                wd_str = ', '.join([f'{bit_labels[i]}: {wd_m[i]:.3f}' for i in range(6)])
                ws_str = ', '.join([f'{bit_labels[i]}: {ws_m[i]:.3f}' for i in range(6)])
            else:
                wd_str = ws_str = "N/A"
            print(f"  p_tx={p:4d} dBm: total={avg_b:.1f} (demod={avg_db_val:.1f}, side={avg_sb_val:.1f})")
            print(f"    Demod dist: [{wd_str}]")
            print(f"    Side  dist: [{ws_str}]")

    # Per-AP stats
    print(f"\nPer-AP bit allocation at 10 dBm (first 5 samples):")
    with torch.no_grad():
        td = test_dataset[10]
        n_sub = min(5, td['v'].shape[0])
        _, bits_sub, _, _, _, _, _, _ = model(td['v'][:n_sub], td['side'][:n_sub], td['snr'][:n_sub],
                                               tau=args.tau_end, use_quantization=True)
        avg_per_ap = bits_sub.mean(dim=(0, 2))
        for l in range(args.L):
            print(f"  AP {l:2d}: avg bits/link = {avg_per_ap[l].item():.1f}")

    # LSQ scales
    print(f"\nLSQ Quantizer Scales:")
    for i, b in enumerate([2, 4, 6, 8, 12, 16]):
        sd = model.quantizer.q_demod[i].s.item()
        ss = model.quantizer.q_side[i].s.item()
        print(f"  {b}b: demod_s={sd:.6f}, side_s={ss:.6f}")

    print(f"\nDetector alpha (correction blend): {model.detector.alpha.item():.6f}")

    # Quality weight statistics
    print(f"\nQuality weight statistics at 10 dBm:")
    with torch.no_grad():
        td = test_dataset[10]
        _, _, _, _, _, _, _, qw = model(td['v'], td['side'], td['snr'], tau=args.tau_end, use_quantization=True)
        qw_np = qw.cpu().numpy()
        # Show weight statistics per AP
        qw_per_ap = qw_np.mean(axis=(0, 2))  # (L,)
        for l in range(args.L):
            print(f"  AP {l:2d}: avg quality weight = {qw_per_ap[l]:.4f}")

    # Save
    save_path = 'new_joint_model_v7.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total params: {total_params:,}")
    print("Done!")


if __name__ == '__main__':
    main()
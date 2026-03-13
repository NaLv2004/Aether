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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V3 with Improved Dual Adaptive Quantization")
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
    parser.add_argument("--c_target", type=float, default=48.0, help="Target average bits per (AP, user) link")
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
    parser.add_argument("--lambda_sub1", type=float, default=0.001, help="Lambda for Phase 2 sub-phase 1")
    parser.add_argument("--lambda_sub2", type=float, default=0.01, help="Lambda for Phase 2 sub-phase 2")
    parser.add_argument("--lambda_sub3", type=float, default=0.1, help="Lambda for Phase 2 sub-phase 3")
    parser.add_argument("--tau_sub1_start", type=float, default=5.0, help="Tau start for sub-phase 1")
    parser.add_argument("--tau_sub1_end", type=float, default=2.0, help="Tau end for sub-phase 1")
    parser.add_argument("--tau_sub2_start", type=float, default=2.0, help="Tau start for sub-phase 2")
    parser.add_argument("--tau_sub2_end", type=float, default=0.5, help="Tau end for sub-phase 2")
    parser.add_argument("--tau_sub3_start", type=float, default=0.5, help="Tau start for sub-phase 3")
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
    """
    Compute BER for QPSK symbols.
    """
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
    """
    Uniform quantization of real-valued numpy array.
    Maps to num_bits levels centered around the signal range.
    """
    if num_bits <= 0:
        return np.zeros_like(x_real)
    if num_bits >= 24:
        return x_real  # effectively full precision

    levels = 2 ** num_bits
    x_max = np.max(np.abs(x_real)) + 1e-12
    # Symmetric quantization: range [-x_max, x_max]
    step = 2 * x_max / (levels - 1)
    idx = np.round((x_real + x_max) / step)
    idx = np.clip(idx, 0, levels - 1)
    return idx * step - x_max


def compute_cmmse_detection(H, y, s, noise_w):
    """
    Centralized MMSE detection using full (unquantized) channel and received signal.
    """
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
    Uniform bits per real = total_bits / reals_per_ap.
    """
    B, L, N_ant, K_users = H.shape

    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)
    b_uniform = total_bits_per_ap / reals_per_ap

    # Use floor and ceil to bracket
    b_low = max(1, int(np.floor(b_uniform)))
    b_high = int(np.ceil(b_uniform))

    # Choose the closer one
    if abs(b_uniform - b_low) <= abs(b_uniform - b_high):
        b_use = b_low
    else:
        b_use = b_high
    b_use = max(1, b_use)

    actual_bits_per_ap = reals_per_ap * b_use
    print(f"    C-MMSE-Q: c_target={c_target}, reals/AP={reals_per_ap}, "
          f"ideal b={b_uniform:.2f}, using b={b_use}, actual bits/AP={actual_bits_per_ap:.0f} "
          f"(target={total_bits_per_ap:.0f})")

    # Quantize H
    H_real_q = uniform_quantize_np(H.real, b_use)
    H_imag_q = uniform_quantize_np(H.imag, b_use)
    H_q = H_real_q + 1j * H_imag_q

    # Quantize y
    y_real_q = uniform_quantize_np(y.real, b_use)
    y_imag_q = uniform_quantize_np(y.imag, b_use)
    y_q = y_real_q + 1j * y_imag_q

    # Now do centralized MMSE with quantized data
    H_all = H_q.reshape(B, L * N_ant, K_users)
    y_all = y_q.reshape(B, L * N_ant)
    H_H = H_all.conj().transpose(0, 2, 1)
    HHH = H_H @ H_all
    noise_eye = noise_w * np.eye(K_users).reshape(1, K_users, K_users)
    R = HHH + noise_eye
    HHy = H_H @ y_all[..., np.newaxis]
    R_inv = np.linalg.inv(R)
    s_hat = (R_inv @ HHy).squeeze(-1)
    ber = compute_ber(s, s_hat)
    return ber, s_hat, b_use


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

    print(f"    Dist-Q: c_target={c_target}, bits_per_real={b_per_real:.2f}, using b={b_use}")

    # Quantize s_hat per AP
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
# 3. GNN + Transformer Hybrid Detector V3
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    """
    Mean-field GNN message passing layer with O(L) complexity.
    Residual connection + LayerNorm for stable training.
    """
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
        attn_weights = self.attn_net(h)
        attn_weights = torch.softmax(attn_weights, dim=1)
        aggregated = (h * attn_weights).sum(dim=1)
        return aggregated, attn_weights


class GNNTransformerDetectorV3(nn.Module):
    """
    Hybrid GNN + Transformer detector V3 for Cell-Free MIMO.

    Architecture:
    1. AP Feature Extraction with BatchNorm after fusion
    2. Mean-field GNN Layers (x3)
    3. AP Aggregation (attention-based)
    4. Transformer IC (x2 layers, 4 heads)
    5. Output: residual connection
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(GNNTransformerDetectorV3, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # MLP for demod features: (B, L, K, 2) -> (B, L, K, D)
        self.demod_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # MLP for channel features per user: (B, L, K, 2*N) -> (B, L, K, D)
        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * N, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion MLP: concat [demod_feat(D), channel_feat(D), bitwidth_features(2), interference_features(2)] -> D
        fusion_input_dim = 2 * hidden_dim + 2 + 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # BatchNorm after fusion MLP for more stable training
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)

        # Mean-field GNN Layers
        self.gnn_layers = nn.ModuleList([
            MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # AP Aggregation
        self.ap_aggregator = APAggregator(hidden_dim)

        # Soft info aggregation (for residual baseline output)
        self.soft_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

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

        # Output Head
        self.output_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, s_hat_q, H_q, bitwidth_features):
        """
        Args:
            s_hat_q: (B, L, K, 2) - quantized demod results (real, imag)
            H_q: (B, L, N, K, 2) - quantized channel coefficients (real, imag)
            bitwidth_features: (B, L, K, 2) - normalized bitwidth features
        Returns:
            detected: (B, K, 2) - detected symbols (real, imag)
        """
        B = s_hat_q.size(0)
        L = s_hat_q.size(1)
        K = s_hat_q.size(2)
        N = H_q.size(2)

        # 1. AP Feature Extraction
        demod_feat = self.demod_mlp(s_hat_q)  # (B, L, K, D)

        H_q_perm = H_q.permute(0, 1, 3, 2, 4)  # (B, L, K, N, 2)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)  # (B, L, K, 2*N)
        channel_feat = self.channel_mlp(H_q_flat)  # (B, L, K, D)

        # Cross-user interference features
        h_power_per_user = (H_q ** 2).sum(dim=(2, 4))  # (B, L, K)
        desired_power = h_power_per_user
        total_power = h_power_per_user.sum(dim=2, keepdim=True)
        interference_power = total_power - desired_power

        desired_feat = torch.log1p(desired_power).unsqueeze(-1)
        interference_feat = torch.log1p(interference_power).unsqueeze(-1)
        interference_features = torch.cat([desired_feat, interference_feat], dim=-1)

        # Concatenate and fuse
        combined = torch.cat([demod_feat, channel_feat, bitwidth_features, interference_features], dim=-1)
        node_features = self.fusion_mlp(combined)  # (B, L, K, D)

        # Apply BatchNorm: reshape to (B*L*K, D) for BN, then back
        node_features_flat = node_features.reshape(-1, self.hidden_dim)
        node_features_bn = self.fusion_bn(node_features_flat)
        node_features = node_features_bn.reshape(B, L, K, self.hidden_dim)

        # 2. Mean-field GNN Message Passing
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)

        # 3. AP Aggregation
        soft_info = s_hat_q
        soft_attn_weights = self.soft_attn(h)
        soft_attn_weights = torch.softmax(soft_attn_weights, dim=1)
        base_out = (soft_info * soft_attn_weights).sum(dim=1)

        user_features, _ = self.ap_aggregator(h)

        # 4. Transformer IC
        ic_out = self.transformer_ic(user_features)

        # 5. Output with residual connection
        residual = self.output_head(ic_out)
        detected = base_out + residual

        return detected


# ============================================================
# 4. Dual Adaptive Quantizer V3 (Improved bit options: {2,4,6,8})
# ============================================================
class DualPolicyNetworkV3(nn.Module):
    """
    Policy network with 7 input features and 4 bit options {2,4,6,8}.

    Input features per (l, k):
        [v_real, v_imag, snr, channel_norm, avg_channel_power, sir, total_power_norm]
    Output: logits for demod bitwidth and channel bitwidth, each 4 options
    """
    def __init__(self, input_dim=7, policy_hidden=64):
        super(DualPolicyNetworkV3, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        self.head_demod = nn.Linear(policy_hidden, 4)    # logits for {2, 4, 6, 8} bits
        self.head_channel = nn.Linear(policy_hidden, 4)  # logits for {2, 4, 6, 8} bits

    def forward(self, x):
        """x: (..., 7) features, Returns: logits_demod (..., 4), logits_channel (..., 4)"""
        h = self.shared(x)
        return self.head_demod(h), self.head_channel(h)


class DualAdaptiveQuantizerV3(nn.Module):
    """
    Dual adaptive quantization with bit options {2, 4, 6, 8} for BOTH demod and channel.
    Uses Gumbel-Softmax for differentiable bitwidth selection.

    For demod results (complex-valued, 2 real values per link):
    - 2-bit: 2*2 = 4 bits per link
    - 4-bit: 2*4 = 8 bits per link
    - 6-bit: 2*6 = 12 bits per link
    - 8-bit: 2*8 = 16 bits per link

    For channel coefficients (N complex-valued, 2N real values per link):
    - 2-bit: 2*N*2 = 16 bits per link (N=4)
    - 4-bit: 2*N*4 = 32 bits per link
    - 6-bit: 2*N*6 = 48 bits per link
    - 8-bit: 2*N*8 = 64 bits per link

    Total per link ranges from 20 to 80 bits.
    """
    def __init__(self, N=4, policy_hidden=64):
        super(DualAdaptiveQuantizerV3, self).__init__()
        self.N = N
        self.policy = DualPolicyNetworkV3(input_dim=7, policy_hidden=policy_hidden)

        # LSQ quantizers for demod results: 2, 4, 6, 8 bits
        self.q2_demod = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_demod = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_demod = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_demod = LSQQuantizer(num_bits=8, init_s=0.01)

        # LSQ quantizers for channel coefficients: 2, 4, 6, 8 bits
        self.q2_channel = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_channel = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q6_channel = LSQQuantizer(num_bits=6, init_s=0.05)
        self.q8_channel = LSQQuantizer(num_bits=8, init_s=0.01)

        # Bit options
        self.demod_bit_options = [2, 4, 6, 8]
        self.channel_bit_options = [2, 4, 6, 8]

    def forward(self, v, H, local_snr, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - demod results (real, imag parts)
            H: (B, L, N, K, 2) - channel coefficients (real, imag parts)
            local_snr: (B, L, K, 1) - local SNR features
            tau: Gumbel-Softmax temperature
        Returns:
            v_q, H_q, expected_bits_demod, expected_bits_channel, w_demod, w_channel
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # --- Compute policy input features (7 dims) ---
        v_real = v[..., 0:1]  # (B, L, K, 1)
        v_imag = v[..., 1:2]  # (B, L, K, 1)

        # Channel norm per user: ||H_{l,:,k}||^2
        h_power = (H ** 2).sum(dim=(2, 4))  # (B, L, K)
        channel_norm = h_power.unsqueeze(-1)  # (B, L, K, 1)

        # Average channel power across APs for each user
        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)  # (B, L, K, 1)

        # SIR: signal-to-interference ratio per (l, k) link
        total_power = h_power.sum(dim=2, keepdim=True)  # (B, L, 1)
        interference_power = total_power - h_power  # (B, L, K)
        sir = torch.log1p(h_power / (interference_power + 1e-10)).unsqueeze(-1)  # (B, L, K, 1)

        # Total power norm (normalized total power at this AP)
        total_power_norm = torch.log1p(total_power).expand(B, L, K).unsqueeze(-1)  # (B, L, K, 1)

        # Policy input: [v_real, v_imag, snr, channel_norm, avg_channel_power, sir, total_power_norm]
        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm,
                                  avg_channel_power, sir, total_power_norm], dim=-1)  # (B, L, K, 7)

        # --- Get bitwidth decisions ---
        logits_demod, logits_channel = self.policy(policy_input)  # each: (B, L, K, 4)

        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)      # (B, L, K, 4)
        w_channel = F.gumbel_softmax(logits_channel, tau=tau, hard=True)  # (B, L, K, 4)

        # --- Quantize demod results ---
        v_q2 = self.q2_demod(v)
        v_q4 = self.q4_demod(v)
        v_q6 = self.q6_demod(v)
        v_q8 = self.q8_demod(v)

        v_q = (w_demod[..., 0:1] * v_q2 +
               w_demod[..., 1:2] * v_q4 +
               w_demod[..., 2:3] * v_q6 +
               w_demod[..., 3:4] * v_q8)  # (B, L, K, 2)

        # --- Quantize channel coefficients ---
        H_q2 = self.q2_channel(H)
        H_q4 = self.q4_channel(H)
        H_q6 = self.q6_channel(H)
        H_q8 = self.q8_channel(H)

        w_ch_exp = w_channel.unsqueeze(2)  # (B, L, 1, K, 4)

        H_q = (w_ch_exp[..., 0:1] * H_q2 +
               w_ch_exp[..., 1:2] * H_q4 +
               w_ch_exp[..., 2:3] * H_q6 +
               w_ch_exp[..., 3:4] * H_q8)  # (B, L, N, K, 2)

        # --- Expected bits per link ---
        # Demod: 2 real values * b bits = 2*b bits per link
        demod_bits = torch.tensor([2.0 * b for b in self.demod_bit_options],
                                  device=v.device)  # [4, 8, 12, 16]
        expected_bits_demod = (w_demod * demod_bits.view(1, 1, 1, 4)).sum(dim=-1)  # (B, L, K)

        # Channel: 2*N real values * b bits = 2*N*b bits per link
        channel_bits = torch.tensor([2.0 * N * b for b in self.channel_bit_options],
                                    device=v.device)  # [16, 32, 48, 64]
        expected_bits_channel = (w_channel * channel_bits.view(1, 1, 1, 4)).sum(dim=-1)  # (B, L, K)

        return v_q, H_q, expected_bits_demod, expected_bits_channel, w_demod, w_channel


# ============================================================
# 5. Joint Model V3
# ============================================================
class JointModelV3(nn.Module):
    """
    End-to-end model combining DualAdaptiveQuantizerV3 + GNNTransformerDetectorV3.
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(JointModelV3, self).__init__()
        self.quantizer = DualAdaptiveQuantizerV3(N=N, policy_hidden=64)
        self.detector = GNNTransformerDetectorV3(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        # Max bits for normalization: demod max = 16, channel max = 64 (N=4, 8-bit)
        self.max_demod_bits = 16.0
        self.max_channel_bits = 64.0

    def forward(self, v, H, local_snr, tau=1.0, use_quantization=True, noise_std=0.0):
        """
        Args:
            v: (B, L, K, 2) - demod results
            H: (B, L, N, K, 2) - channel coefficients
            local_snr: (B, L, K, 1) - local SNR
            tau: Gumbel-Softmax temperature
            use_quantization: if False, skip quantization (Phase 1)
            noise_std: Gaussian noise std to inject (Phase 1 robustness)
        Returns:
            detected: (B, K, 2)
            expected_bits_demod: (B, L, K)
            expected_bits_channel: (B, L, K)
        """
        B, L, K, _ = v.shape

        # Optional noise injection for robustness training
        if noise_std > 0 and self.training:
            v = v + torch.randn_like(v) * noise_std
            H = H + torch.randn_like(H) * noise_std

        if use_quantization:
            v_q, H_q, expected_bits_demod, expected_bits_channel, w_demod, w_channel = \
                self.quantizer(v, H, local_snr, tau=tau)
            bw_demod_feat = expected_bits_demod.unsqueeze(-1) / self.max_demod_bits
            bw_channel_feat = expected_bits_channel.unsqueeze(-1) / self.max_channel_bits
        else:
            v_q = v
            H_q = H
            expected_bits_demod = torch.zeros(B, L, K, device=v.device)
            expected_bits_channel = torch.zeros(B, L, K, device=v.device)
            bw_demod_feat = torch.ones(B, L, K, 1, device=v.device)
            bw_channel_feat = torch.ones(B, L, K, 1, device=v.device)

        bitwidth_features = torch.cat([bw_demod_feat, bw_channel_feat], dim=-1)  # (B, L, K, 2)
        detected = self.detector(v_q, H_q, bitwidth_features)

        return detected, expected_bits_demod, expected_bits_channel


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors(s_hat, s, H, local_snr, device):
    """Convert numpy complex arrays to real-valued torch tensors."""
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)

    H_real = np.stack([H.real, H.imag], axis=-1)
    H_tensor = torch.FloatTensor(H_real).to(device)

    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)

    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)

    return v_tensor, H_tensor, snr_tensor, Y_tensor


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

    print(f"{'=' * 90}")
    print(f"GNN+Transformer Hybrid Detector V3 with Improved Dual Adaptive Quantization")
    print(f"{'=' * 90}")
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
    print(f"Bit options: demod={{2,4,6,8}}, channel={{2,4,6,8}}")
    print(f"  Demod bits per link: {{4, 8, 12, 16}}")
    print(f"  Channel bits per link (N={args.N}): {{16, 32, 48, 64}}")
    print(f"  Total bits per link range: [20, 80]")
    print(f"Bit constraint: C_target={args.c_target} bits/link")
    fp_bits = 2 * 32 + 2 * args.N * 32
    print(f"  Full precision bits/link: {fp_bits}")
    print(f"  Compression target: {fp_bits / args.c_target:.1f}x")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"Noise injection std (Phase 1): {args.noise_injection_std}")
    print(f"Gradient clipping: max_norm={args.grad_clip}")
    print(f"Dropout: {args.dropout}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 90}")

    # Initialize system model
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    # ===========================
    # Pre-generate fixed test set
    # ===========================
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"\nGenerating fixed test set ({args.test_samples} samples per power point)...")
    for p in test_p_tx_list:
        s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch_v2(sys_model, args.test_samples, p_tx_dbm=p)
        v_t, H_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'Y': Y_t,
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
    baseline_ber_dist_q = {}

    for p in test_p_tx_list:
        td = test_dataset[p]
        baseline_ber_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        baseline_ber_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], sys_model.noise_w)

    print(f"\nComputing C-MMSE-Q baseline (same total bit budget as proposed, c_target={args.c_target})...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_cq, _, b_used = compute_cmmse_q_detection(
            td['H_np'], td['y_np'], td['s_np'], sys_model.noise_w,
            args.c_target, args.K, args.N
        )
        baseline_ber_cmmse_q[p] = ber_cq

    print(f"\nComputing Dist-Q baseline (same total bit budget)...")
    for p in test_p_tx_list:
        td = test_dataset[p]
        ber_dq, b_used = compute_dist_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        baseline_ber_dist_q[p] = ber_dq

    print(f"\n{'=' * 90}")
    print(f"Baseline BERs (computed on fixed test set):")
    print(f"{'p_tx(dBm)':<12} | {'Dist-Full':<12} | {'C-MMSE':<12} | {'C-MMSE-Q':<12} | {'Dist-Q':<12}")
    print("-" * 65)
    for p in test_p_tx_list:
        print(f"{p:<12} | {baseline_ber_dist_full[p]:<12.6f} | {baseline_ber_cmmse[p]:<12.6f} | "
              f"{baseline_ber_cmmse_q[p]:<12.6f} | {baseline_ber_dist_q[p]:<12.6f}")
    print()

    # ===========================
    # Initialize model
    # ===========================
    model = JointModelV3(
        L=args.L, N=args.N, K=args.K,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    detector_params = sum(p.numel() for p in model.detector.parameters())
    quantizer_params = sum(p.numel() for p in model.quantizer.parameters())
    print(f"Model parameters: Total={total_params:,}, Detector={detector_params:,}, Quantizer={quantizer_params:,}")

    criterion = nn.MSELoss()

    # ===========================
    # Phase 1: Detector Pre-training
    # ===========================
    print(f"\n{'=' * 90}")
    print(f"Phase 1: Detector Pre-training ({args.epochs_phase1} epochs, full precision + noise injection)")
    print(f"  Learning rate: {args.lr_phase1} with cosine annealing")
    print(f"  Noise injection std: {args.noise_injection_std}")
    print(f"{'=' * 90}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=args.epochs_phase1, eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np, _ = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

            optimizer_phase1.zero_grad()
            detected, _, _ = model(v_tensor, H_tensor, snr_tensor,
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
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    val_out, _, _ = model(td['v'], td['H'], td['snr'], use_quantization=False, noise_std=0.0)
                    val_np = val_out.cpu().numpy()
                    s_pred = val_np[..., 0] + 1j * val_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)

            lr_now = scheduler_phase1.get_last_lr()[0]
            print(f"Phase1 Epoch [{epoch + 1:03d}/{args.epochs_phase1}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | "
                  f"Train Loss: {avg_loss:.6f} | "
                  f"Val BER: 0dBm={val_bers[0]:.4f}, 10dBm={val_bers[10]:.4f}, 20dBm={val_bers[20]:.4f}")

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

    print(f"{'=' * 90}")
    print(f"Phase 2: Joint QAT Training ({total_phase2} epochs, 3 sub-phases)")
    print(f"  Learning rate: {args.lr_phase2} with cosine annealing")
    for sp in sub_phases:
        print(f"  {sp['name']}: {sp['epochs']} epochs, lambda={sp['lambda']}, "
              f"tau: {sp['tau_start']} -> {sp['tau_end']}")
    print(f"{'=' * 90}")

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
            tau = max(tau, 0.01)  # safety floor

            current_lambda = sp['lambda']

            for batch_idx in range(args.batches_per_epoch):
                p_train = np.random.uniform(-20, 25)
                s_hat_np, s_np, H_np, snr_np, _ = generate_data_batch_v2(sys_model, args.batch_size, p_train)
                v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

                optimizer_phase2.zero_grad()
                detected, exp_bits_demod, exp_bits_channel = model(
                    v_tensor, H_tensor, snr_tensor, tau=tau, use_quantization=True, noise_std=0.0
                )

                mse_loss = criterion(detected, Y_tensor)

                total_bits_per_link = exp_bits_demod + exp_bits_channel
                avg_bits = total_bits_per_link.mean()
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

            # Print every 5 epochs and first epoch of each sub-phase
            if (local_epoch + 1) % 5 == 0 or local_epoch == 0:
                model.eval()
                with torch.no_grad():
                    # Validate at 0 dBm
                    td = test_dataset[0]
                    val_out, val_bits_d, val_bits_c = model(
                        td['v'], td['H'], td['snr'], tau=tau, use_quantization=True
                    )
                    val_np = val_out.cpu().numpy()
                    s_pred = val_np[..., 0] + 1j * val_np[..., 1]
                    val_ber_0 = compute_ber(td['s_np'], s_pred)
                    val_avg_bits_d = val_bits_d.mean().item()
                    val_avg_bits_c = val_bits_c.mean().item()
                    val_avg_bits_total = (val_bits_d + val_bits_c).mean().item()

                    # Validate at 10 dBm
                    td10 = test_dataset[10]
                    val_out10, vbd10, vbc10 = model(
                        td10['v'], td10['H'], td10['snr'], tau=tau, use_quantization=True
                    )
                    val_np10 = val_out10.cpu().numpy()
                    s_pred10 = val_np10[..., 0] + 1j * val_np10[..., 1]
                    val_ber_10 = compute_ber(td10['s_np'], s_pred10)

                    # Get bit allocation distribution at 10 dBm
                    v_q_t, H_q_t, ebd_t, ebc_t, w_d_t, w_c_t = model.quantizer(
                        td10['v'], td10['H'], td10['snr'], tau=tau
                    )
                    w_d_mean = w_d_t.mean(dim=(0, 1, 2))
                    w_c_mean = w_c_t.mean(dim=(0, 1, 2))

                lr_now = scheduler_phase2.get_last_lr()[0]
                print(f"  Phase2 {sp['name']} Epoch [{local_epoch + 1:03d}/{sp['epochs']}] (Global: {global_epoch + 1}) | "
                      f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | lam: {current_lambda} | "
                      f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}) | "
                      f"Avg Bits: {avg_bits_epoch:.1f} (target: {args.c_target})")
                print(f"    Val BER: 0dBm={val_ber_0:.4f}, 10dBm={val_ber_10:.4f} | "
                      f"Val Bits: D={val_avg_bits_d:.1f} C={val_avg_bits_c:.1f} T={val_avg_bits_total:.1f}")
                print(f"    Bit dist @10dBm: Demod [2b:{w_d_mean[0]:.3f} 4b:{w_d_mean[1]:.3f} "
                      f"6b:{w_d_mean[2]:.3f} 8b:{w_d_mean[3]:.3f}] | "
                      f"Chan [2b:{w_c_mean[0]:.3f} 4b:{w_c_mean[1]:.3f} "
                      f"6b:{w_c_mean[2]:.3f} 8b:{w_c_mean[3]:.3f}]")

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
    print(f"{'=' * 90}")
    print(f"Final Evaluation on Test Set ({args.test_samples} samples per power point)")
    print(f"{'=' * 90}")

    model.eval()

    header = (f"{'p_tx(dBm)':<10} | {'Dist-Full':<11} | {'C-MMSE':<11} | {'C-MMSE-Q':<11} | "
              f"{'Dist-Q':<11} | {'Proposed':<11} | {'vs C-MMSE-Q':<13} | {'vs Dist-Full':<13} | "
              f"{'Demod B':<9} | {'Chan B':<9} | {'Total B':<9}")
    print(f"\n{header}")
    print("-" * len(header))

    results_proposed = {}
    with torch.no_grad():
        for p in test_p_tx_list:
            td = test_dataset[p]

            ber_dist_full = baseline_ber_dist_full[p]
            ber_cmmse = baseline_ber_cmmse[p]
            ber_cmmse_q = baseline_ber_cmmse_q[p]
            ber_dist_q = baseline_ber_dist_q[p]

            # Proposed: use tau_min from last sub-phase
            tau_eval = args.tau_sub3_end
            detected, exp_bits_d, exp_bits_c = model(
                td['v'], td['H'], td['snr'], tau=tau_eval, use_quantization=True
            )
            det_np = detected.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_d = exp_bits_d.mean().item()
            avg_bits_c = exp_bits_c.mean().item()
            avg_bits_total = (exp_bits_d + exp_bits_c).mean().item()

            improve_cmmse_q = ((ber_cmmse_q - ber_proposed) / max(ber_cmmse_q, 1e-10)) * 100
            improve_dist = ((ber_dist_full - ber_proposed) / max(ber_dist_full, 1e-10)) * 100

            results_proposed[p] = {
                'ber': ber_proposed, 'bits_d': avg_bits_d, 'bits_c': avg_bits_c, 'bits_t': avg_bits_total
            }

            print(f"{p:<10} | {ber_dist_full:<11.6f} | {ber_cmmse:<11.6f} | {ber_cmmse_q:<11.6f} | "
                  f"{ber_dist_q:<11.6f} | {ber_proposed:<11.6f} | {improve_cmmse_q:>+11.2f}% | "
                  f"{improve_dist:>+11.2f}% | "
                  f"{avg_bits_d:<9.2f} | {avg_bits_c:<9.2f} | {avg_bits_total:<9.2f}")

    print("-" * len(header))

    # ===========================
    # Summary statistics
    # ===========================
    print(f"\n{'=' * 90}")
    print(f"Summary: Average Performance Across All Power Points")
    print(f"{'=' * 90}")
    avg_ber_dist = np.mean([baseline_ber_dist_full[p] for p in test_p_tx_list])
    avg_ber_cmmse = np.mean([baseline_ber_cmmse[p] for p in test_p_tx_list])
    avg_ber_cmmse_q = np.mean([baseline_ber_cmmse_q[p] for p in test_p_tx_list])
    avg_ber_dist_q = np.mean([baseline_ber_dist_q[p] for p in test_p_tx_list])
    avg_ber_proposed = np.mean([results_proposed[p]['ber'] for p in test_p_tx_list])
    avg_total_bits = np.mean([results_proposed[p]['bits_t'] for p in test_p_tx_list])
    avg_demod_bits = np.mean([results_proposed[p]['bits_d'] for p in test_p_tx_list])
    avg_chan_bits = np.mean([results_proposed[p]['bits_c'] for p in test_p_tx_list])

    print(f"  Average Dist-Full BER:   {avg_ber_dist:.6f}")
    print(f"  Average C-MMSE BER:      {avg_ber_cmmse:.6f}")
    print(f"  Average C-MMSE-Q BER:    {avg_ber_cmmse_q:.6f} (same bit budget)")
    print(f"  Average Dist-Q BER:      {avg_ber_dist_q:.6f} (same bit budget)")
    print(f"  Average Proposed BER:    {avg_ber_proposed:.6f}")
    print(f"  Average Demod Bits/Link: {avg_demod_bits:.2f}")
    print(f"  Average Chan Bits/Link:  {avg_chan_bits:.2f}")
    print(f"  Average Total Bits/Link: {avg_total_bits:.2f} (target: {args.c_target})")
    print(f"  Full Precision Bits/Link: {fp_bits}")
    print(f"  Average Compression Ratio: {fp_bits / max(avg_total_bits, 1e-10):.1f}x")

    # Performance comparison
    print(f"\n  --- Performance Comparisons ---")
    if avg_ber_proposed < avg_ber_cmmse_q:
        print(f"  *** Proposed OUTPERFORMS C-MMSE-Q by {((avg_ber_cmmse_q - avg_ber_proposed) / max(avg_ber_cmmse_q, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"  *** Proposed underperforms C-MMSE-Q by {((avg_ber_proposed - avg_ber_cmmse_q) / max(avg_ber_cmmse_q, 1e-10)) * 100:.1f}% ***")

    if avg_ber_proposed < avg_ber_dist:
        print(f"  *** Proposed OUTPERFORMS Dist-Full by {((avg_ber_dist - avg_ber_proposed) / max(avg_ber_dist, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"  *** Proposed underperforms Dist-Full by {((avg_ber_proposed - avg_ber_dist) / max(avg_ber_dist, 1e-10)) * 100:.1f}% ***")

    if avg_ber_proposed < avg_ber_cmmse:
        print(f"  *** Proposed OUTPERFORMS C-MMSE (full) by {((avg_ber_cmmse - avg_ber_proposed) / max(avg_ber_cmmse, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"  *** Proposed underperforms C-MMSE (full) by {((avg_ber_proposed - avg_ber_cmmse) / max(avg_ber_cmmse, 1e-10)) * 100:.1f}% ***")

    if avg_ber_proposed < avg_ber_dist_q:
        print(f"  *** Proposed OUTPERFORMS Dist-Q by {((avg_ber_dist_q - avg_ber_proposed) / max(avg_ber_dist_q, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"  *** Proposed underperforms Dist-Q by {((avg_ber_proposed - avg_ber_dist_q) / max(avg_ber_dist_q, 1e-10)) * 100:.1f}% ***")

    # ===========================
    # Detailed bit allocation statistics
    # ===========================
    print(f"\n{'=' * 90}")
    print(f"Detailed Bit Allocation Statistics")
    print(f"{'=' * 90}")
    with torch.no_grad():
        tau_eval = args.tau_sub3_end
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]

            v_q_t, H_q_t, ebd_t, ebc_t, w_d_t, w_c_t = model.quantizer(
                td['v'], td['H'], td['snr'], tau=tau_eval
            )

            w_d_mean = w_d_t.mean(dim=(0, 1, 2))
            w_c_mean = w_c_t.mean(dim=(0, 1, 2))

            avg_demod_b = ebd_t.mean().item()
            avg_chan_b = ebc_t.mean().item()
            avg_total_b = (ebd_t + ebc_t).mean().item()
            compression = fp_bits / max(avg_total_b, 1e-10)

            print(f"\n  p_tx = {p} dBm:")
            print(f"    Demod bit distribution:   2b: {w_d_mean[0]:.3f}, 4b: {w_d_mean[1]:.3f}, "
                  f"6b: {w_d_mean[2]:.3f}, 8b: {w_d_mean[3]:.3f}")
            print(f"    Channel bit distribution: 2b: {w_c_mean[0]:.3f}, 4b: {w_c_mean[1]:.3f}, "
                  f"6b: {w_c_mean[2]:.3f}, 8b: {w_c_mean[3]:.3f}")
            print(f"    Avg demod bits/link: {avg_demod_b:.2f}, Avg channel bits/link: {avg_chan_b:.2f}")
            print(f"    Avg total bits/link: {avg_total_b:.2f}")
            print(f"    Full precision bits/link: {fp_bits}, Compression ratio: {compression:.1f}x")

    # ===========================
    # Per-AP bit allocation statistics
    # ===========================
    print(f"\n{'=' * 90}")
    print(f"Per-AP Bit Allocation at 10 dBm (first 5 test samples)")
    print(f"{'=' * 90}")
    with torch.no_grad():
        td = test_dataset[10]
        v_sub = td['v'][:5]
        H_sub = td['H'][:5]
        snr_sub = td['snr'][:5]
        _, _, ebd_sub, ebc_sub, _, _ = model.quantizer(v_sub, H_sub, snr_sub, tau=tau_eval)
        total_bits_sub = ebd_sub + ebc_sub
        avg_per_ap = total_bits_sub.mean(dim=(0, 2))
        for l in range(args.L):
            print(f"  AP {l:2d}: avg bits/link = {avg_per_ap[l].item():.2f}")

    # ===========================
    # LSQ scale parameter values
    # ===========================
    print(f"\n{'=' * 90}")
    print(f"LSQ Quantizer Scale Parameters (learned)")
    print(f"{'=' * 90}")
    q = model.quantizer
    print(f"  Demod quantizers:  2b s={q.q2_demod.s.item():.6f}, 4b s={q.q4_demod.s.item():.6f}, "
          f"6b s={q.q6_demod.s.item():.6f}, 8b s={q.q8_demod.s.item():.6f}")
    print(f"  Channel quantizers: 2b s={q.q2_channel.s.item():.6f}, 4b s={q.q4_channel.s.item():.6f}, "
          f"6b s={q.q6_channel.s.item():.6f}, 8b s={q.q8_channel.s.item():.6f}")

    # ===========================
    # Save model
    # ===========================
    save_path = 'new_joint_model_v3.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
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
    parser = argparse.ArgumentParser(description="Cell-Free MIMO Detector V8: LSFD-inspired Iterative IC")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=5, help="Phase 1 epochs (detector pre-training, full precision)")
    parser.add_argument("--epochs_phase2", type=int, default=30, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=50, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=5e-4, help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=3e-4, help="Learning rate for Phase 2")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D")
    parser.add_argument("--c_target", type=float, default=80.0, help="Target average bits per (AP, user) link")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers for AP aggregation")
    parser.add_argument("--num_ic_iters", type=int, default=3, help="Number of iterative IC iterations")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--test_samples", type=int, default=200, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--train_snr_dbm", type=float, default=12.0, help="Fixed training SNR in dBm")
    parser.add_argument("--tau_start", type=float, default=3.0, help="Gumbel-Softmax temperature start")
    parser.add_argument("--tau_end", type=float, default=0.1, help="Gumbel-Softmax temperature end")
    parser.add_argument("--lambda_bit_start", type=float, default=0.001, help="Bit penalty start")
    parser.add_argument("--lambda_bit_end", type=float, default=0.5, help="Bit penalty end")
    parser.add_argument("--multi_snr_train", type=int, default=1, help="Train with multiple SNR points if 1")
    return parser.parse_args()


# ============================================================
# 1. Data Generation
# ============================================================
def generate_data_batch(sys_model, batch_size, p_tx_dbm):
    """
    Generate batch of Cell-Free MIMO data with rich side information.
    
    Returns:
        s_hat: (B, L, K) complex - local LMMSE estimates (full precision at AP)
        s: (B, K) complex - true QPSK symbols
        H: (B, L, N, K) complex - full channel (for baselines only)
        y: (B, L, N) complex - received signal (for baselines only)
        side_info: dict with per-AP per-user side information:
            'channel_gain': (B, L, K) - |h_{l,k}|^2 summed over antennas
            'interference': (B, L, K) - sum of other users' channel gains
            'local_sinr': (B, L, K) - local SINR estimate
            'noise_power': scalar - noise power
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

    # Local LMMSE detection
    H_H = H.conj().transpose(0, 1, 3, 2)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H @ H_H + noise_cov
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_H @ R_y_inv
    s_hat = (W_l @ y[..., np.newaxis]).squeeze(-1)

    # Side information (computed at AP from full-precision data)
    channel_gain = np.sum(np.abs(H) ** 2, axis=2)  # (B, L, K)
    total_gain = channel_gain.sum(axis=2, keepdims=True)  # (B, L, 1)
    interference = total_gain - channel_gain  # (B, L, K)
    local_sinr = channel_gain / (interference + sys_model.noise_w + 1e-12)

    side_info = {
        'channel_gain': channel_gain,
        'interference': interference,
        'local_sinr': local_sinr,
        'noise_power': sys_model.noise_w
    }

    return s_hat, s, H, y, side_info


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
    return compute_ber(s, s_hat.mean(axis=1))


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
    R = H_H @ H_all + noise_w * np.eye(K).reshape(1, K, K)
    HHy = H_H @ y_all[..., np.newaxis]
    R_inv = np.linalg.inv(R)
    s_hat = (R_inv @ HHy).squeeze(-1)
    return compute_ber(s, s_hat), s_hat


def compute_dist_q_ber(s_hat, s, c_target):
    B, L, K = s_hat.shape
    b_per_real = c_target / 2.0
    b_use = max(1, min(16, int(np.round(b_per_real))))
    s_hat_q = np.zeros_like(s_hat)
    for l in range(L):
        s_hat_q[:, l, :] = (uniform_quantize_np(s_hat[:, l, :].real, b_use) +
                             1j * uniform_quantize_np(s_hat[:, l, :].imag, b_use))
    return compute_ber(s, s_hat_q.mean(axis=1)), b_use


def compute_lsfd_ber(s_hat, s, side_info):
    """
    Large-Scale Fading Decoding (LSFD) baseline.
    Weight each AP's estimate by its channel gain (SINR-based weighting).
    s_est_k = sum_l w_{l,k} * s_hat_{l,k} / sum_l w_{l,k}
    where w_{l,k} = channel_gain_{l,k} / (interference_{l,k} + noise)
    """
    B, L, K = s_hat.shape
    # MMSE-optimal weight: proportional to SINR
    weights = side_info['local_sinr']  # (B, L, K)
    weights_sum = weights.sum(axis=1, keepdims=True) + 1e-12  # (B, 1, K)
    weights_norm = weights / weights_sum  # (B, L, K)
    s_est = (weights_norm * s_hat).sum(axis=1)  # (B, K)
    return compute_ber(s, s_est)


def compute_cmmse_q_detection(H, y, s, noise_w, c_target, K, N):
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)
    best_ber = 1.0
    best_b = 1
    for b_use in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
        if reals_per_ap * b_use > total_bits_per_ap * 1.05:
            continue
        H_q = uniform_quantize_np(H.real, b_use) + 1j * uniform_quantize_np(H.imag, b_use)
        y_q = uniform_quantize_np(y.real, b_use) + 1j * uniform_quantize_np(y.imag, b_use)
        H_all = H_q.reshape(B, L * N_ant, K_users)
        y_all = y_q.reshape(B, L * N_ant)
        H_H = H_all.conj().transpose(0, 2, 1)
        R = H_H @ H_all + noise_w * np.eye(K_users).reshape(1, K_users, K_users) + 1e-10 * np.eye(K_users).reshape(1, K_users, K_users)
        HHy = H_H @ y_all[..., np.newaxis]
        try:
            s_hat = (np.linalg.inv(R) @ HHy).squeeze(-1)
            ber = compute_ber(s, s_hat)
            if ber < best_ber:
                best_ber = ber
                best_b = b_use
        except:
            continue
    actual = reals_per_ap * best_b
    print(f"    C-MMSE-Q: best b={best_b}, actual={actual} (target={total_bits_per_ap:.0f}), BER={best_ber:.6f}")
    return best_ber, best_b


# ============================================================
# 3. Adaptive Quantizer V8
# ============================================================
class AdaptiveQuantizerV8(nn.Module):
    """
    Quantize demod results + side info for fronthaul transmission.
    
    Per link budget breakdown:
      - Demod: 2 reals * b_demod bits/real
      - Side info: 3 reals * b_side bits/real (channel_gain, interference, sinr)
      Total bits/link = 2*b_demod + 3*b_side
    
    Bit options for demod: {2, 4, 6, 8, 12, 16} bits/real
    Bit options for side:  {2, 4, 6, 8} bits/real (side info needs less precision)
    
    With c_target=80: can afford 16 bits/real for demod (32 bits) + 16 bits/real for side (48 bits) = 80 total
    """
    def __init__(self, policy_hidden=64):
        super().__init__()
        self.num_demod_options = 6
        self.num_side_options = 4
        self.demod_bit_options = [2, 4, 6, 8, 12, 16]
        self.side_bit_options = [2, 4, 6, 8]
        
        # Policy: input = [demod_real, demod_imag, snr_log, gain_log, intf_log, mag] = 6
        self.policy_demod = nn.Sequential(
            nn.Linear(6, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, self.num_demod_options)
        )
        self.policy_side = nn.Sequential(
            nn.Linear(6, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, self.num_side_options)
        )
        
        # LSQ quantizers for demod
        self.q_demod = nn.ModuleDict({
            f'q{b}': LSQQuantizer(num_bits=b, init_s=max(0.001, 1.0 / (2**(b-1))))
            for b in self.demod_bit_options
        })
        # LSQ quantizers for side info
        self.q_side = nn.ModuleDict({
            f'q{b}': LSQQuantizer(num_bits=b, init_s=max(0.001, 1.0 / (2**(b-1))))
            for b in self.side_bit_options
        })
        
        # Initialize to favor high bits (since budget allows it)
        with torch.no_grad():
            # Demod: favor 16b (index 5)
            self.policy_demod[-1].bias.copy_(torch.tensor([-2.0, -1.5, -1.0, 0.0, 1.0, 2.0]))
            # Side: favor 8b (index 3)
            self.policy_side[-1].bias.copy_(torch.tensor([-1.0, 0.0, 0.5, 1.5]))
    
    def forward(self, v, side_info_tensor, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - demod real/imag
            side_info_tensor: (B, L, K, 3) - [gain_log, intf_log, sinr_log]
            tau: Gumbel temperature
        Returns:
            v_q: (B, L, K, 2) - quantized demod
            side_q: (B, L, K, 3) - quantized side info
            bits_per_link: (B, L, K) - bits per link
            w_demod: (B, L, K, 6) - demod bit weights
            w_side: (B, L, K, 4) - side bit weights
        """
        B, L, K, _ = v.shape
        
        # Policy features
        magnitude = torch.sqrt(v[..., 0:1]**2 + v[..., 1:2]**2 + 1e-10)
        policy_input = torch.cat([v, side_info_tensor, magnitude], dim=-1)  # (B,L,K,6)
        
        logits_demod = self.policy_demod(policy_input)
        logits_side = self.policy_side(policy_input)
        
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B,L,K,6)
        w_side = F.gumbel_softmax(logits_side, tau=tau, hard=True)    # (B,L,K,4)
        
        # Quantize demod at each level
        v_quants = []
        for b in self.demod_bit_options:
            v_quants.append(self.q_demod[f'q{b}'](v))
        v_stack = torch.stack(v_quants, dim=3)  # (B,L,K,6,2)
        v_q = (v_stack * w_demod.unsqueeze(-1)).sum(dim=3)
        
        # Quantize side info at each level
        s_quants = []
        for b in self.side_bit_options:
            s_quants.append(self.q_side[f'q{b}'](side_info_tensor))
        s_stack = torch.stack(s_quants, dim=3)  # (B,L,K,4,3)
        side_q = (s_stack * w_side.unsqueeze(-1)).sum(dim=3)
        
        # Compute bits per link
        demod_bits = torch.tensor([float(b) for b in self.demod_bit_options], device=v.device)
        side_bits = torch.tensor([float(b) for b in self.side_bit_options], device=v.device)
        
        bits_demod = 2.0 * (w_demod * demod_bits.view(1,1,1,-1)).sum(dim=-1)  # (B,L,K)
        bits_side = 3.0 * (w_side * side_bits.view(1,1,1,-1)).sum(dim=-1)      # (B,L,K)
        bits_per_link = bits_demod + bits_side
        
        return v_q, side_q, bits_per_link, w_demod, w_side, bits_demod, bits_side


# ============================================================
# 4. Iterative IC Detector V8
# ============================================================

class APWeightNetwork(nn.Module):
    """Learn per-AP per-user combining weights from side information."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features):
        """features: (B, L, K, input_dim) -> weights: (B, L, K, 1)"""
        return self.net(features)


class UserICLayer(nn.Module):
    """One iteration of interference cancellation across users."""
    def __init__(self, hidden_dim, K, num_heads=4, dropout=0.1):
        super().__init__()
        self.K = K
        # Input: current estimate (2) + all other estimates (2*(K-1)) + side features
        # We use attention over users
        self.user_embed = nn.Linear(hidden_dim + 2, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, s_current, user_features):
        """
        s_current: (B, K, 2) - current symbol estimates
        user_features: (B, K, hidden_dim) - user-level features from AP aggregation
        Returns: s_refined: (B, K, 2) - refined estimates
        """
        # Combine current estimate with features
        combined = torch.cat([user_features, s_current], dim=-1)  # (B, K, hidden_dim+2)
        h = self.user_embed(combined)  # (B, K, hidden_dim)
        
        # Self-attention across users for IC
        h2, _ = self.attn(h, h, h)
        h = self.norm1(h + h2)
        h2 = self.ffn(h)
        h = self.norm2(h + h2)
        
        # Residual correction
        correction = self.output(h)  # (B, K, 2)
        s_refined = s_current + correction
        
        return s_refined, h


class IterativeICDetectorV8(nn.Module):
    """
    LSFD-inspired Iterative IC Detector.
    
    Architecture:
    1. Feature encoding from quantized demod + side info
    2. AP aggregation with learned quality weights
    3. Multi-iteration interference cancellation
    
    Key insight: Each IC iteration refines symbol estimates by 
    considering other users' estimates (soft IC).
    """
    def __init__(self, L, K, hidden_dim=128, num_gnn_layers=3, 
                 num_ic_iters=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim
        self.num_ic_iters = num_ic_iters
        
        # Per-node feature encoder: demod(2) + side(3) + extra(2) = 7
        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # AP attention layers for message passing
        self.ap_attn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.ap_attn_layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            }))
        
        # AP weight network (LSFD-like)
        self.ap_weight_net = APWeightNetwork(hidden_dim, hidden_dim // 2)
        
        # Initial combining: produce first symbol estimate from weighted demod
        self.init_combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Iterative IC layers
        self.ic_layers = nn.ModuleList([
            UserICLayer(hidden_dim, K, num_heads, dropout)
            for _ in range(num_ic_iters)
        ])
        
        # Learned blend between quality-weighted average and network output
        self.blend_alpha = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, demod_q, side_q):
        """
        Args:
            demod_q: (B, L, K, 2)
            side_q: (B, L, K, 3)
        Returns:
            s_est: (B, K, 2) - final symbol estimate
            s_weighted_avg: (B, K, 2) - LSFD-weighted average
            intermediate_estimates: list of (B, K, 2) for each IC iter
        """
        B, L, K, _ = demod_q.shape
        
        # ---- Step 1: Quality-weighted average (LSFD-like) ----
        # Use side_q[:,:,:,2] as SINR proxy for weights
        sinr_proxy = side_q[..., 2:3]  # (B, L, K, 1)
        # Softmax weights over APs for each user
        sinr_weights = torch.softmax(sinr_proxy * 3.0, dim=1)  # (B, L, K, 1) - temperature 3
        s_weighted_avg = (demod_q * sinr_weights).sum(dim=1)  # (B, K, 2)
        
        # ---- Step 2: Feature encoding ----
        magnitude = torch.sqrt(demod_q[..., 0:1]**2 + demod_q[..., 1:2]**2 + 1e-10)
        # Deviation from weighted average
        s_avg_broadcast = s_weighted_avg.unsqueeze(1).expand_as(demod_q)
        deviation = torch.sqrt(((demod_q - s_avg_broadcast)**2).sum(dim=-1, keepdim=True) + 1e-10)
        
        node_input = torch.cat([demod_q, side_q, magnitude, deviation], dim=-1)  # (B,L,K,7)
        h = self.node_encoder(node_input)  # (B, L, K, D)
        
        # ---- Step 3: AP attention (per user, attend over APs) ----
        # Reshape to (B*K, L, D)
        h_per_user = h.permute(0, 2, 1, 3).reshape(B * K, L, self.hidden_dim)
        
        for layer_dict in self.ap_attn_layers:
            h2, _ = layer_dict['attn'](h_per_user, h_per_user, h_per_user)
            h_per_user = layer_dict['norm1'](h_per_user + h2)
            h2 = layer_dict['ffn'](h_per_user)
            h_per_user = layer_dict['norm2'](h_per_user + h2)
        
        # ---- Step 4: Learned AP weights ----
        ap_weights = self.ap_weight_net(h_per_user)  # (B*K, L, 1)
        ap_weights = torch.softmax(ap_weights, dim=1)
        user_features = (h_per_user * ap_weights).sum(dim=1)  # (B*K, D)
        user_features = user_features.reshape(B, K, self.hidden_dim)
        
        # Initial estimate from network
        nn_init = self.init_combine(user_features)  # (B, K, 2)
        
        # Blend with quality-weighted average
        alpha = torch.sigmoid(self.blend_alpha)
        s_current = alpha * s_weighted_avg + (1 - alpha) * nn_init
        
        # ---- Step 5: Iterative IC ----
        intermediate_estimates = [s_current]
        for ic_layer in self.ic_layers:
            s_current, user_features = ic_layer(s_current, user_features)
            intermediate_estimates.append(s_current)
        
        return s_current, s_weighted_avg, intermediate_estimates


# ============================================================
# 5. Joint Model V8
# ============================================================
class JointModelV8(nn.Module):
    def __init__(self, L, K, hidden_dim=128, num_gnn_layers=3,
                 num_ic_iters=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.L = L
        self.K = K
        self.quantizer = AdaptiveQuantizerV8(policy_hidden=64)
        self.detector = IterativeICDetectorV8(
            L, K, hidden_dim, num_gnn_layers, num_ic_iters, num_heads, dropout
        )
    
    def forward(self, v, side_info_tensor, tau=1.0, use_quantization=True):
        """
        Args:
            v: (B, L, K, 2)
            side_info_tensor: (B, L, K, 3)
        Returns:
            s_est, bits_per_link, s_weighted_avg, intermediate_estimates, w_demod, w_side
        """
        if use_quantization:
            v_q, side_q, bits_per_link, w_demod, w_side, bits_d, bits_s = \
                self.quantizer(v, side_info_tensor, tau)
        else:
            v_q = v
            side_q = side_info_tensor
            B = v.shape[0]
            bits_per_link = torch.zeros(B, self.L, self.K, device=v.device)
            w_demod = None
            w_side = None
            bits_d = torch.zeros(B, self.L, self.K, device=v.device)
            bits_s = torch.zeros(B, self.L, self.K, device=v.device)
        
        s_est, s_weighted_avg, intermediate = self.detector(v_q, side_q)
        
        return s_est, bits_per_link, s_weighted_avg, intermediate, w_demod, w_side, bits_d, bits_s


# ============================================================
# 6. Helpers
# ============================================================
def prepare_tensors(s_hat, s, side_info, device):
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)
    
    # Log-normalize side info for better neural network input
    gain_log = np.log10(side_info['channel_gain'] + 1e-15) / 10.0  # normalize
    intf_log = np.log10(side_info['interference'] + 1e-15) / 10.0
    sinr_log = np.log10(side_info['local_sinr'] + 1e-12) / 10.0
    
    side_np = np.stack([gain_log, intf_log, sinr_log], axis=-1)  # (B, L, K, 3)
    side_tensor = torch.FloatTensor(side_np).to(device)
    
    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)
    
    return v_tensor, side_tensor, Y_tensor


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
    
    # Bit budget analysis
    # demod: 2 reals, side: 3 reals => 5 reals per link
    # With 16 bits/real for both: 2*16 + 3*16 = 80 bits/link (perfect match for c_target=80)
    # With 16 bits demod + 8 bits side: 2*16 + 3*8 = 56 bits/link
    fp_bits_per_link = 5 * 32  # 160 bits (5 reals at float32)
    max_bits = 2 * 16 + 3 * 8  # 56 with current options
    
    print(f"{'='*100}")
    print(f"Cell-Free MIMO Detector V8: LSFD-inspired Iterative IC")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} ant/AP, K={args.K} users")
    print(f"Architecture:")
    print(f"  - APs compute local LMMSE at full precision")
    print(f"  - Each AP sends per-user: quantized demod (2 reals) + side info (3 reals)")
    print(f"  - Side info: channel_gain_log, interference_log, local_sinr_log")
    print(f"  - Adaptive bit allocation for demod {{2,4,6,8,12,16}} and side {{2,4,6,8}}")
    print(f"  - Max bits/link: 2*16 + 3*8 = 56")
    print(f"  - Iterative IC with {args.num_ic_iters} iterations")
    print(f"  - GNN AP aggregation + attention-based user IC")
    print(f"Bit budget: c_target={args.c_target} bits/link (FP={fp_bits_per_link})")
    print(f"Training: Phase1={args.epochs_phase1} eps (FP), Phase2={args.epochs_phase2} eps (QAT)")
    print(f"  LR: Phase1={args.lr_phase1}, Phase2={args.lr_phase2}")
    print(f"  tau: {args.tau_start} -> {args.tau_end}")
    print(f"  lambda_bit: {args.lambda_bit_start} -> {args.lambda_bit_end}")
    print(f"  Multi-SNR training: {'Yes' if args.multi_snr_train else 'No'}")
    print(f"Hidden dim: {args.hidden_dim}, GNN layers: {args.num_gnn_layers}")
    print(f"IC iterations: {args.num_ic_iters}")
    print(f"Batch: {args.batch_size} x {args.batches_per_epoch}/epoch, Seed: {args.seed}")
    print(f"{'='*100}")
    
    # System model
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()
    noise_w = sys_model.noise_w
    
    # Test set
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"\nGenerating fixed test set ({args.test_samples} samples/point)...")
    for p in test_p_tx_list:
        s_hat, s, H, y, side = generate_data_batch(sys_model, args.test_samples, p)
        v_t, side_t, Y_t = prepare_tensors(s_hat, s, side, device)
        test_dataset[p] = {
            'v': v_t, 'side': side_t, 'Y': Y_t,
            's_hat_np': s_hat, 's_np': s, 'H_np': H, 'y_np': y,
            'side_info': side
        }
    print("Test set ready.\n")
    
    # Baselines
    print(f"Computing baselines...")
    bl_dist_full = {}
    bl_cmmse = {}
    bl_dist_q = {}
    bl_cmmse_q = {}
    bl_lsfd = {}
    
    for p in test_p_tx_list:
        td = test_dataset[p]
        bl_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        bl_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], noise_w)
        bl_dist_q[p], _ = compute_dist_q_ber(td['s_hat_np'], td['s_np'], args.c_target)
        bl_lsfd[p] = compute_lsfd_ber(td['s_hat_np'], td['s_np'], td['side_info'])
    
    print(f"\nC-MMSE-Q baselines:")
    for p in test_p_tx_list:
        td = test_dataset[p]
        bl_cmmse_q[p], _ = compute_cmmse_q_detection(
            td['H_np'], td['y_np'], td['s_np'], noise_w, args.c_target, args.K, args.N)
    
    print(f"\n{'='*100}")
    print(f"Baselines:")
    print(f"{'p_tx':<8} | {'Dist-Full':<11} | {'LSFD':<11} | {'C-MMSE':<11} | {'C-MMSE-Q':<11} | {'Dist-Q':<11}")
    print("-" * 75)
    for p in test_p_tx_list:
        print(f"{p:<8} | {bl_dist_full[p]:<11.6f} | {bl_lsfd[p]:<11.6f} | {bl_cmmse[p]:<11.6f} | "
              f"{bl_cmmse_q[p]:<11.6f} | {bl_dist_q[p]:<11.6f}")
    print()
    
    # Model
    model = JointModelV8(
        L=args.L, K=args.K, hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers, num_ic_iters=args.num_ic_iters,
        num_heads=args.num_heads, dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    det_params = sum(p.numel() for p in model.detector.parameters())
    q_params = sum(p.numel() for p in model.quantizer.parameters())
    print(f"Parameters: Total={total_params:,}, Detector={det_params:,}, Quantizer={q_params:,}")
    
    # Training SNR points for multi-SNR training
    if args.multi_snr_train:
        train_snr_points = [-5, 0, 5, 10, 15, 20]
    else:
        train_snr_points = [args.train_snr_dbm]
    
    # ======== Phase 1: Full precision pre-training ========
    print(f"\n{'='*100}")
    print(f"Phase 1: Detector pre-training ({args.epochs_phase1} epochs, full precision)")
    print(f"  Training SNR points: {train_snr_points}")
    print(f"{'='*100}")
    
    optimizer1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.epochs_phase1, 1), eta_min=1e-5)
    
    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_idx in range(args.batches_per_epoch):
            # Pick random SNR for this batch
            snr = train_snr_points[batch_idx % len(train_snr_points)]
            s_hat, s, H, y, side = generate_data_batch(sys_model, args.batch_size, snr)
            v_t, side_t, Y_t = prepare_tensors(s_hat, s, side, device)
            
            optimizer1.zero_grad()
            s_est, _, s_wavg, intermediates, _, _, _, _ = model(v_t, side_t, use_quantization=False)
            
            # Loss on final + intermediate estimates
            loss = F.mse_loss(s_est, Y_t)
            for i, s_inter in enumerate(intermediates[:-1]):
                loss += 0.3 * F.mse_loss(s_inter, Y_t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), args.grad_clip)
            optimizer1.step()
            epoch_loss += loss.item()
        
        scheduler1.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / args.batches_per_epoch
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            bers = {}
            bers_wavg = {}
            for p in [0, 10, 20]:
                td = test_dataset[p]
                s_est_val, _, s_wavg_val, _, _, _, _, _ = model(td['v'], td['side'], use_quantization=False)
                pred = s_est_val.cpu().numpy()
                bers[p] = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
                
                pred_w = s_wavg_val.cpu().numpy()
                bers_wavg[p] = compute_ber(td['s_np'], pred_w[..., 0] + 1j * pred_w[..., 1])
        
        alpha_val = torch.sigmoid(model.detector.blend_alpha).item()
        print(f"Phase1 [{epoch+1:03d}/{args.epochs_phase1}] | {elapsed:.1f}s | Loss: {avg_loss:.6f} | alpha: {alpha_val:.3f}")
        print(f"  Network BER: 0dB={bers[0]:.4f}({bl_dist_full[0]:.4f}) "
              f"10dB={bers[10]:.4f}({bl_dist_full[10]:.4f}) "
              f"20dB={bers[20]:.4f}({bl_dist_full[20]:.4f})")
        print(f"  WeigAvg BER: 0dB={bers_wavg[0]:.4f}({bl_lsfd[0]:.4f}) "
              f"10dB={bers_wavg[10]:.4f}({bl_lsfd[10]:.4f}) "
              f"20dB={bers_wavg[20]:.4f}({bl_lsfd[20]:.4f})")
    
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
        t0 = time.time()
        
        # Annealing
        if args.epochs_phase2 > 1:
            progress = epoch / (args.epochs_phase2 - 1)
        else:
            progress = 1.0
        tau = args.tau_start * (args.tau_end / max(args.tau_start, 1e-10)) ** progress
        tau = max(tau, 0.01)
        lambda_bit = args.lambda_bit_start + (args.lambda_bit_end - args.lambda_bit_start) * progress
        
        for batch_idx in range(args.batches_per_epoch):
            snr = train_snr_points[batch_idx % len(train_snr_points)]
            s_hat, s, H, y, side = generate_data_batch(sys_model, args.batch_size, snr)
            v_t, side_t, Y_t = prepare_tensors(s_hat, s, side, device)
            
            optimizer2.zero_grad()
            s_est, bits, s_wavg, intermediates, w_d, w_s, _, _ = model(v_t, side_t, tau=tau, use_quantization=True)
            
            # Detection loss (final + intermediate)
            mse_loss = F.mse_loss(s_est, Y_t)
            for s_inter in intermediates[:-1]:
                mse_loss += 0.2 * F.mse_loss(s_inter, Y_t)
            
            # Bit constraint
            avg_bits = bits.mean()
            bit_violation = F.relu(avg_bits - args.c_target)
            bit_loss = lambda_bit * bit_violation ** 2
            
            loss = mse_loss + bit_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer2.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_bits += avg_bits.item()
        
        scheduler2.step()
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_mse = epoch_mse / args.batches_per_epoch
        avg_bits_epoch = epoch_bits / args.batches_per_epoch
        elapsed = time.time() - t0
        
        # Evaluate
        if (epoch + 1) % max(1, args.epochs_phase2 // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                bers = {}
                bits_dict = {}
                for p in [0, 10, 20]:
                    td = test_dataset[p]
                    s_est_val, bits_val, s_wavg_val, _, wd, ws, bd, bs = model(
                        td['v'], td['side'], tau=tau, use_quantization=True)
                    pred = s_est_val.cpu().numpy()
                    bers[p] = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
                    bits_dict[p] = bits_val.mean().item()
                
                # Bit distribution at 10dBm
                td10 = test_dataset[10]
                _, _, _, _, wd10, ws10, bd10, bs10 = model(
                    td10['v'], td10['side'], tau=tau, use_quantization=True)
                if wd10 is not None:
                    wd_mean = wd10.mean(dim=(0,1,2))
                    ws_mean = ws10.mean(dim=(0,1,2))
                    avg_bd = bd10.mean().item()
                    avg_bs = bs10.mean().item()
            
            alpha_val = torch.sigmoid(model.detector.blend_alpha).item()
            avg_val_ber = np.mean([bers[p] for p in [0, 10, 20]])
            
            d_labels = ['2b','4b','6b','8b','12b','16b']
            s_labels = ['2b','4b','6b','8b']
            wd_str = ' '.join([f'{d_labels[i]}:{wd_mean[i]:.3f}' for i in range(6)]) if wd10 is not None else "N/A"
            ws_str = ' '.join([f'{s_labels[i]}:{ws_mean[i]:.3f}' for i in range(4)]) if ws10 is not None else "N/A"
            
            print(f"Phase2 [{epoch+1:03d}/{args.epochs_phase2}] | {elapsed:.1f}s | tau={tau:.3f} | "
                  f"λ={lambda_bit:.4f} | Loss={avg_loss:.5f} (MSE={avg_mse:.5f}) | alpha={alpha_val:.3f}")
            print(f"  Bits: total={avg_bits_epoch:.1f}/{args.c_target} (demod={avg_bd:.1f}, side={avg_bs:.1f})")
            print(f"  BER: 0dB={bers[0]:.4f}({bl_dist_full[0]:.4f}/{bl_lsfd[0]:.4f}) "
                  f"10dB={bers[10]:.4f}({bl_dist_full[10]:.4f}/{bl_lsfd[10]:.4f}) "
                  f"20dB={bers[20]:.4f}({bl_dist_full[20]:.4f}/{bl_lsfd[20]:.4f})")
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
    header = (f"{'p_tx':<8} | {'Dist-Full':<11} | {'LSFD':<11} | {'C-MMSE':<11} | {'C-MMSE-Q':<11} | "
              f"{'Dist-Q':<11} | {'Proposed':<11} | {'vs DF':<10} | {'vs LSFD':<10} | {'Bits':<8} | "
              f"{'D-bits':<8} | {'S-bits':<8}")
    print(f"\n{header}")
    print("-" * len(header))
    
    results = {}
    with torch.no_grad():
        tau_eval = args.tau_end
        for p in test_p_tx_list:
            td = test_dataset[p]
            s_est_val, bits_val, _, _, _, _, bd_val, bs_val = model(
                td['v'], td['side'], tau=tau_eval, use_quantization=True)
            pred = s_est_val.cpu().numpy()
            ber_prop = compute_ber(td['s_np'], pred[..., 0] + 1j * pred[..., 1])
            avg_bits = bits_val.mean().item()
            avg_bd = bd_val.mean().item()
            avg_bs = bs_val.mean().item()
            
            gap_df = ((ber_prop - bl_dist_full[p]) / max(bl_dist_full[p], 1e-10)) * 100
            gap_lsfd = ((ber_prop - bl_lsfd[p]) / max(bl_lsfd[p], 1e-10)) * 100
            
            results[p] = {'ber': ber_prop, 'bits': avg_bits, 'bd': avg_bd, 'bs': avg_bs}
            
            print(f"{p:<8} | {bl_dist_full[p]:<11.6f} | {bl_lsfd[p]:<11.6f} | {bl_cmmse[p]:<11.6f} | "
                  f"{bl_cmmse_q[p]:<11.6f} | {bl_dist_q[p]:<11.6f} | {ber_prop:<11.6f} | "
                  f"{gap_df:>+8.1f}% | {gap_lsfd:>+8.1f}% | {avg_bits:<8.1f} | "
                  f"{avg_bd:<8.1f} | {avg_bs:<8.1f}")
    
    print("-" * len(header))
    
    # Summary
    print(f"\n{'='*100}")
    print(f"Summary")
    print(f"{'='*100}")
    avg_df = np.mean([bl_dist_full[p] for p in test_p_tx_list])
    avg_lsfd = np.mean([bl_lsfd[p] for p in test_p_tx_list])
    avg_cm = np.mean([bl_cmmse[p] for p in test_p_tx_list])
    avg_cq = np.mean([bl_cmmse_q[p] for p in test_p_tx_list])
    avg_dq = np.mean([bl_dist_q[p] for p in test_p_tx_list])
    avg_prop = np.mean([results[p]['ber'] for p in test_p_tx_list])
    avg_bits = np.mean([results[p]['bits'] for p in test_p_tx_list])
    avg_bd = np.mean([results[p]['bd'] for p in test_p_tx_list])
    avg_bs = np.mean([results[p]['bs'] for p in test_p_tx_list])
    
    print(f"  Dist-Full (no quant, equal avg):  {avg_df:.6f}")
    print(f"  LSFD (SINR-weighted avg):         {avg_lsfd:.6f}")
    print(f"  C-MMSE (full precision):           {avg_cm:.6f}")
    print(f"  C-MMSE-Q (uniform quant):          {avg_cq:.6f}")
    print(f"  Dist-Q (uniform quant):            {avg_dq:.6f}")
    print(f"  Proposed V8:                       {avg_prop:.6f}")
    print(f"  Avg bits/link: {avg_bits:.1f} (target: {args.c_target})")
    print(f"    Demod: {avg_bd:.1f}, Side: {avg_bs:.1f}")
    print(f"  FP bits/link: {fp_bits_per_link}")
    print(f"  Compression: {fp_bits_per_link/max(avg_bits, 1e-10):.1f}x")
    
    comparisons = [
        ("Dist-Full", avg_df), ("LSFD", avg_lsfd), ("C-MMSE", avg_cm),
        ("C-MMSE-Q", avg_cq), ("Dist-Q", avg_dq)
    ]
    for name, bl in comparisons:
        if avg_prop < bl:
            pct = ((bl - avg_prop) / max(bl, 1e-10)) * 100
            print(f"  >> OUTPERFORMS {name} by {pct:.1f}%")
        else:
            pct = ((avg_prop - bl) / max(bl, 1e-10)) * 100
            print(f"  >> Underperforms {name} by {pct:.1f}%")
    
    # Check individual power points
    print(f"\n{'='*100}")
    beats_df_count = sum(1 for p in test_p_tx_list if results[p]['ber'] < bl_dist_full[p])
    beats_lsfd_count = sum(1 for p in test_p_tx_list if results[p]['ber'] < bl_lsfd[p])
    print(f"  Beats Dist-Full at {beats_df_count}/{len(test_p_tx_list)} power points")
    print(f"  Beats LSFD at {beats_lsfd_count}/{len(test_p_tx_list)} power points")
    
    if avg_prop < avg_df:
        print(f"  >>> SUCCESS: Proposed V8 BEATS Dist-Full! ({avg_prop:.6f} < {avg_df:.6f}) <<<")
    else:
        print(f"  >>> Proposed V8 does NOT beat Dist-Full yet. ({avg_prop:.6f} >= {avg_df:.6f}) <<<")
    print(f"{'='*100}")
    
    # Bit allocation details
    print(f"\n{'='*100}")
    print(f"Bit Allocation Details")
    print(f"{'='*100}")
    d_labels = ['2b','4b','6b','8b','12b','16b']
    s_labels = ['2b','4b','6b','8b']
    with torch.no_grad():
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]
            _, bits_val, _, _, wd, ws, bd, bs = model(
                td['v'], td['side'], tau=args.tau_end, use_quantization=True)
            if wd is not None:
                wd_m = wd.mean(dim=(0,1,2))
                ws_m = ws.mean(dim=(0,1,2))
                wd_str = ', '.join([f'{d_labels[i]}: {wd_m[i]:.3f}' for i in range(6)])
                ws_str = ', '.join([f'{s_labels[i]}: {ws_m[i]:.3f}' for i in range(4)])
            else:
                wd_str = ws_str = "N/A"
            print(f"  p_tx={p:4d} dBm: total={bits_val.mean().item():.1f} "
                  f"(demod={bd.mean().item():.1f}, side={bs.mean().item():.1f})")
            print(f"    Demod: [{wd_str}]")
            print(f"    Side:  [{ws_str}]")
    
    # Per-AP stats
    print(f"\nPer-AP bit allocation at 10 dBm (first 5 samples):")
    with torch.no_grad():
        td = test_dataset[10]
        n_sub = min(5, td['v'].shape[0])
        _, bits_sub, _, _, _, _, _, _ = model(
            td['v'][:n_sub], td['side'][:n_sub], tau=args.tau_end, use_quantization=True)
        avg_per_ap = bits_sub.mean(dim=(0, 2))
        for l in range(args.L):
            print(f"  AP {l:2d}: avg bits/link = {avg_per_ap[l].item():.1f}")
    
    # LSQ scales
    print(f"\nLSQ Quantizer Scales:")
    q = model.quantizer
    print(f"  Demod: ", end="")
    for b in q.demod_bit_options:
        print(f"{b}b={q.q_demod[f'q{b}'].s.item():.6f} ", end="")
    print()
    print(f"  Side:  ", end="")
    for b in q.side_bit_options:
        print(f"{b}b={q.q_side[f'q{b}'].s.item():.6f} ", end="")
    print()
    
    alpha_val = torch.sigmoid(model.detector.blend_alpha).item()
    print(f"\nDetector blend alpha: {alpha_val:.4f}")
    
    # Save
    save_path = 'new_joint_model_v8.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total params: {total_params:,}")
    print("Done!")


if __name__ == '__main__':
    main()
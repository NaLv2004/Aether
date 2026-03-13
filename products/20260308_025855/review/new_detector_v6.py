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
    parser = argparse.ArgumentParser(description="Cell-Free MIMO Detector V6: Quantized Local LMMSE + GNN Combiner")
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
    """Centralized MMSE detection using full (unquantized) channel and received signal.
    No reg_eye - only noise_eye for regularization."""
    B, L, N, K = H.shape
    H_all = H.reshape(B, L * N, K)
    y_all = y.reshape(B, L * N)
    H_H = H_all.conj().transpose(0, 2, 1)
    HHH = H_H @ H_all
    noise_eye = noise_w * np.eye(K).reshape(1, K, K)
    R = HHH + noise_eye
    HHy = H_H @ y_all[..., np.newaxis]
    try:
        R_inv = np.linalg.inv(R)
        s_hat_cmmse = (R_inv @ HHy).squeeze(-1)
        ber = compute_ber(s, s_hat_cmmse)
        return ber, s_hat_cmmse
    except np.linalg.LinAlgError:
        print("    C-MMSE: matrix inversion failed, returning BER=0.5")
        return 0.5, np.zeros_like(s)


def compute_cmmse_q_detection(H, y, s, noise_w, c_target, K, N):
    """
    C-MMSE-Q: Centralized MMSE with uniformly quantized H and y at same total bit budget.
    Total bits per AP = c_target * K.
    Reals per AP for C-MMSE = 2*N + 2*N*K = 2*N*(1+K).
    No reg_eye - only noise_eye.
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
        R = HHH + noise_eye
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
    No reg_eye - only noise_eye.
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
            R = HHH + noise_eye
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
                    R = HHH + noise_eye
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
# 3. Adaptive Demod Quantizer
# ============================================================
class AdaptiveDemodQuantizer(nn.Module):
    """
    Adaptive quantizer for local LMMSE estimates (s_hat) and channel quality indicators (CQI).
    
    Each AP sends per user:
    - Quantized s_hat_{l,k}: 1 complex value = 2 reals, quantized at b_demod bits/real
    - Quantized channel quality: ||h_{l,:,k}||^2 (1 real), quantized at b_cqi bits/real
    
    Total bits per link = 2 * b_demod + 1 * b_cqi
    
    Bit options for b_demod: {2, 4, 6, 8, 12, 16} (6 options)
    Bit options for b_cqi: {2, 4, 8} (3 options)
    """
    def __init__(self, policy_hidden=64):
        super(AdaptiveDemodQuantizer, self).__init__()
        self.demod_bit_options = [2, 4, 6, 8, 12, 16]
        self.cqi_bit_options = [2, 4, 8]
        self.num_demod_opts = 6
        self.num_cqi_opts = 3

        # Policy network: shared MLP(6 -> 64 -> 64) then two heads
        self.shared = nn.Sequential(
            nn.Linear(6, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        self.head_demod = nn.Linear(policy_hidden, self.num_demod_opts)
        self.head_cqi = nn.Linear(policy_hidden, self.num_cqi_opts)

        # LSQ quantizers for demod (6 options)
        self.q_demod = nn.ModuleList([
            LSQQuantizer(num_bits=b, init_s=max(0.5 / 2 ** (b - 2), 0.001))
            for b in self.demod_bit_options
        ])
        # LSQ quantizers for CQI (3 options)
        self.q_cqi = nn.ModuleList([
            LSQQuantizer(num_bits=b, init_s=max(0.5 / 2 ** (b - 2), 0.001))
            for b in self.cqi_bit_options
        ])

    def forward(self, s_hat_real, channel_power, local_snr, tau=1.0):
        """
        Args:
            s_hat_real: (B, L, K, 2) - local LMMSE estimates [real, imag]
            channel_power: (B, L, K) - ||h_{l,:,k}||^2 per link
            local_snr: (B, L, K, 1) - normalized local SNR
            tau: Gumbel-Softmax temperature
        Returns:
            s_hat_q: (B, L, K, 2) - quantized demod
            cqi_q: (B, L, K, 1) - quantized channel quality
            bits_per_link: (B, L, K) - expected bits per link
            w_demod: (B, L, K, 6) - demod bit weights
            w_cqi: (B, L, K, 3) - CQI bit weights
        """
        B, L, K, _ = s_hat_real.shape

        # Compute policy features
        s_r = s_hat_real[..., 0:1]  # (B,L,K,1)
        s_i = s_hat_real[..., 1:2]  # (B,L,K,1)
        cp = channel_power.unsqueeze(-1)  # (B,L,K,1)
        cp_norm = torch.log1p(cp)  # normalized
        avg_cp = channel_power.mean(dim=1, keepdim=True).unsqueeze(-1).expand_as(cp)  # (B,L,K,1)
        avg_cp_norm = torch.log1p(avg_cp)
        total_cp = channel_power.sum(dim=2, keepdim=True)  # (B,L,1)
        interf = total_cp - channel_power  # (B,L,K)
        sir = torch.log1p(channel_power / (interf + 1e-10)).unsqueeze(-1)  # (B,L,K,1)

        policy_input = torch.cat([s_r, s_i, local_snr, cp_norm, avg_cp_norm, sir], dim=-1)  # (B,L,K,6)

        h = self.shared(policy_input)  # (B,L,K,64)
        logits_demod = self.head_demod(h)  # (B,L,K,6)
        logits_cqi = self.head_cqi(h)  # (B,L,K,3)

        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B,L,K,6)
        w_cqi = F.gumbel_softmax(logits_cqi, tau=tau, hard=True)  # (B,L,K,3)

        # Quantize demod
        q_versions = [q(s_hat_real) for q in self.q_demod]  # list of (B,L,K,2)
        q_stack = torch.stack(q_versions, dim=3)  # (B,L,K,6,2)
        s_hat_q = (q_stack * w_demod.unsqueeze(-1)).sum(dim=3)  # (B,L,K,2)

        # Quantize CQI
        cp_expanded = cp  # (B,L,K,1)
        cqi_versions = [q(cp_expanded) for q in self.q_cqi]  # list of (B,L,K,1)
        cqi_stack = torch.stack(cqi_versions, dim=3)  # (B,L,K,3,1)
        cqi_q = (cqi_stack * w_cqi.unsqueeze(-1)).sum(dim=3)  # (B,L,K,1)

        # Expected bits per link
        demod_bit_vals = torch.tensor([float(b) for b in self.demod_bit_options], device=s_hat_real.device)
        cqi_bit_vals = torch.tensor([float(b) for b in self.cqi_bit_options], device=s_hat_real.device)

        exp_demod_bits = 2.0 * (w_demod * demod_bit_vals).sum(dim=-1)  # (B,L,K)
        exp_cqi_bits = 1.0 * (w_cqi * cqi_bit_vals).sum(dim=-1)  # (B,L,K)
        bits_per_link = exp_demod_bits + exp_cqi_bits  # (B,L,K)

        return s_hat_q, cqi_q, bits_per_link, w_demod, w_cqi


# ============================================================
# 4. GNN Combiner (Detector)
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    """Mean-field GNN message passing layer with O(L) complexity."""
    def __init__(self, hidden_dim):
        super(MeanFieldGNNLayer, self).__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        """h: (B, L, K, D) -> (B, L, K, D)"""
        msg = self.msg_mlp(h)
        mean_msg = msg.mean(dim=1, keepdim=True).expand_as(h)  # mean over APs
        h_new = self.upd_mlp(torch.cat([h, mean_msg], dim=-1))
        return self.norm(h + h_new)


class GNNCombiner(nn.Module):
    """
    GNN-based combiner that takes quantized local LMMSE estimates and channel quality
    from all APs and produces detected symbols. No MMSE recomputation at CPU.
    
    Input per (l,k): [s_hat_q_real(1), s_hat_q_imag(1), cqi_q(1), bitwidth_norm(1), snr(1)] = 5 features
    """
    def __init__(self, L, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(GNNCombiner, self).__init__()
        self.L = L
        self.K = K
        self.hidden_dim = hidden_dim

        # Input: [s_hat_q_real(1), s_hat_q_imag(1), cqi_q(1), bitwidth_norm(1), snr(1)] = 5 features
        self.input_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Attention-based AP aggregation
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Transformer for inter-user IC
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # Output: predict symbol (real, imag)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, s_hat_q, cqi_q, bits_norm, local_snr):
        """
        Args:
            s_hat_q: (B, L, K, 2) - quantized demod results
            cqi_q: (B, L, K, 1) - quantized channel quality
            bits_norm: (B, L, K, 1) - normalized bitwidth feature
            local_snr: (B, L, K, 1) - local SNR
        Returns:
            detected: (B, K, 2) - detected symbols [real, imag]
        """
        B, L, K, _ = s_hat_q.shape

        # Input features
        x = torch.cat([s_hat_q, cqi_q, bits_norm, local_snr], dim=-1)  # (B,L,K,5)
        h = self.input_mlp(x)  # (B,L,K,D)

        # BatchNorm
        h_flat = h.reshape(-1, self.hidden_dim)
        h_flat = self.input_bn(h_flat)
        h = h_flat.reshape(B, L, K, self.hidden_dim)

        # GNN
        for gnn in self.gnn_layers:
            h = gnn(h)

        # AP aggregation with attention
        attn = self.attn_net(h)  # (B,L,K,1)
        attn = torch.softmax(attn, dim=1)  # softmax over APs
        user_feat = (h * attn).sum(dim=1)  # (B,K,D)

        # Transformer IC
        user_feat = self.transformer(user_feat)  # (B,K,D)

        # Output
        detected = self.output_head(user_feat)  # (B,K,2)
        return detected


# ============================================================
# 5. Joint Model V6
# ============================================================
class JointModelV6(nn.Module):
    """
    End-to-end model: AdaptiveDemodQuantizer + GNNCombiner.
    
    Key idea: Each AP computes local LMMSE from full-precision data, then quantizes
    the local estimates before sending to CPU. The CPU uses a learned GNN to combine
    the quantized estimates (better than simple averaging).
    """
    def __init__(self, L, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(JointModelV6, self).__init__()
        self.quantizer = AdaptiveDemodQuantizer(policy_hidden=64)
        self.detector = GNNCombiner(L, K, hidden_dim, num_gnn_layers,
                                     num_transformer_layers, num_heads, dropout)
        self.max_bits = 2.0 * 16.0 + 1.0 * 8.0  # max possible bits per link = 40

    def forward(self, s_hat_real, channel_power, local_snr, tau=1.0, use_quantization=True):
        """
        Args:
            s_hat_real: (B, L, K, 2) - local LMMSE estimates [real, imag]
            channel_power: (B, L, K) - ||h_{l,:,k}||^2
            local_snr: (B, L, K, 1) - local SNR
            tau: Gumbel-Softmax temperature
            use_quantization: if False, skip quantization (Phase 1)
        Returns:
            detected: (B, K, 2) - detected symbols
            bits_per_link: (B, L, K) - expected bits per link
        """
        B, L, K, _ = s_hat_real.shape
        if use_quantization:
            s_hat_q, cqi_q, bits_per_link, w_d, w_c = self.quantizer(
                s_hat_real, channel_power, local_snr, tau
            )
            bits_norm = bits_per_link.unsqueeze(-1) / self.max_bits
        else:
            s_hat_q = s_hat_real
            cqi_q = channel_power.unsqueeze(-1)
            bits_per_link = torch.zeros(B, L, K, device=s_hat_real.device)
            bits_norm = torch.ones(B, L, K, 1, device=s_hat_real.device)

        detected = self.detector(s_hat_q, cqi_q, bits_norm, local_snr)
        return detected, bits_per_link


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors(s_hat, s, H, local_snr, y, device):
    """Convert numpy arrays to torch tensors for V6 model."""
    # s_hat: (B,L,K) complex -> s_hat_real: (B,L,K,2) float
    s_hat_real = torch.FloatTensor(np.stack([s_hat.real, s_hat.imag], axis=-1)).to(device)
    # channel_power: ||h_{l,:,k}||^2 -> (B,L,K) float
    channel_power = torch.FloatTensor(np.sum(np.abs(H) ** 2, axis=2)).to(device)  # sum over N antennas
    # local_snr: (B,L,K,1)
    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)
    # true symbols: (B,K,2)
    Y = torch.FloatTensor(np.stack([s.real, s.imag], axis=-1)).to(device)
    return s_hat_real, channel_power, snr_tensor, Y


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

    # In V6: each link only sends s_hat (2 reals) + CQI (1 real) = 3 reals per link
    # Max bits per link = 2*16 + 1*8 = 40 bits
    # Min bits per link = 2*2 + 1*2 = 6 bits
    max_bits_per_link = 2.0 * 16.0 + 1.0 * 8.0
    min_bits_per_link = 2.0 * 2.0 + 1.0 * 2.0
    # Full precision bits per link (V5 style for comparison)
    reals_per_ap_v5 = 2 * args.N + 2 * args.N * args.K + 2 * args.K
    fp_bits_per_link = reals_per_ap_v5 * 32 / args.K

    print(f"{'=' * 110}")
    print(f"Cell-Free MIMO Detector V6: Quantized Local LMMSE + Learned GNN Combiner")
    print(f"  ** Key idea: Each AP computes local LMMSE, quantizes estimates + CQI, sends to CPU **")
    print(f"  ** CPU uses GNN to combine quantized estimates (NO MMSE recomputation at CPU) **")
    print(f"{'=' * 110}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users")
    print(f"Per-link payload: s_hat (2 reals) + CQI (1 real) = 3 reals per link")
    print(f"  Demod bit options: {{2, 4, 6, 8, 12, 16}} bits/real")
    print(f"  CQI bit options: {{2, 4, 8}} bits/real")
    print(f"  Bits per link range: [{min_bits_per_link:.0f}, {max_bits_per_link:.0f}]")
    print(f"Training: {total_epochs} total epochs")
    print(f"  Phase 1: {args.epochs_phase1} epochs (detector pre-training, full precision, MSE loss)")
    print(f"  Phase 2: {args.epochs_phase2} epochs (joint QAT, tau: {args.tau_start} -> {args.tau_end})")
    print(f"  Phase1 LR: {args.lr_phase1}, Phase2 LR: {args.lr_phase2}")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {args.batches_per_epoch}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Bit constraint: C_target={args.c_target} bits/link (max possible={max_bits_per_link:.1f})")
    print(f"  Full precision bits/link (V5 style): {fp_bits_per_link:.1f}")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"Gradient clipping: max_norm={args.grad_clip}, Dropout: {args.dropout}")
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
        s_hat_real_t, cp_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, y_np, device)
        test_dataset[p] = {
            's_hat_real': s_hat_real_t, 'channel_power': cp_t, 'snr': snr_t, 'Y': Y_t,
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
    model = JointModelV6(
        L=args.L, K=args.K,
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

    # MSE loss
    mse_criterion = nn.MSELoss()

    # ===========================
    # Phase 1: Detector Pre-training (Full Precision, MSE Loss)
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Phase 1: Detector Pre-training ({args.epochs_phase1} epochs, full precision, MSE loss)")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"  Learning rate: {args.lr_phase1} with cosine annealing")
    print(f"{'=' * 110}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1, T_max=max(args.epochs_phase1, 1), eta_min=1e-5
    )

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch(
                sys_model, args.batch_size, p_tx_dbm=args.train_snr_dbm
            )
            s_hat_real_t, cp_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, y_np, device)

            optimizer_phase1.zero_grad()
            detected, _ = model(s_hat_real_t, cp_t, snr_t, use_quantization=False)
            loss = mse_criterion(detected, Y_t)
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
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    detected_val, _ = model(
                        td['s_hat_real'], td['channel_power'], td['snr'], use_quantization=False
                    )
                    det_np = detected_val.cpu().numpy()
                    s_pred = det_np[..., 0] + 1j * det_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)

            lr_now = scheduler_phase1.get_last_lr()[0]
            print(f"Phase1 Epoch [{epoch + 1:03d}/{args.epochs_phase1}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | "
                  f"Train MSE Loss: {avg_loss:.6f}")
            print(f"  Val BER: 0dBm={val_bers[0]:.4f} (Dist-Full:{baseline_ber_dist_full[0]:.4f}), "
                  f"10dBm={val_bers[10]:.4f} (Dist-Full:{baseline_ber_dist_full[10]:.4f}), "
                  f"20dBm={val_bers[20]:.4f} (Dist-Full:{baseline_ber_dist_full[20]:.4f})")

    print("Phase 1 completed.\n")

    # ===========================
    # Phase 2: Joint QAT Training (MSE Loss + Augmented Lagrangian Bit Penalty)
    # ===========================
    print(f"{'=' * 110}")
    print(f"Phase 2: Joint QAT Training ({args.epochs_phase2} epochs)")
    print(f"  tau annealing: {args.tau_start} -> {args.tau_end} (immediate, no warmup)")
    print(f"  Loss: MSE on symbol estimates + Augmented Lagrangian bit penalty")
    print(f"  Fixed training SNR: {args.train_snr_dbm} dBm")
    print(f"{'=' * 110}")

    # Initialize policy heads to favor medium bits
    print("  Initializing policy network heads...")
    with torch.no_grad():
        # Demod: Bit options {2, 4, 6, 8, 12, 16} -> Peak at 8-bit (index 3)
        bias_demod = torch.tensor([-1.0, -0.5, 0.0, 1.0, 0.0, -0.5])
        model.quantizer.head_demod.bias.copy_(bias_demod)
        print(f"  head_demod bias: {bias_demod.tolist()} (favor 8-bit)")
        
        # CQI: Bit options {2, 4, 8} -> Peak at 4-bit (index 1)
        bias_cqi = torch.tensor([-0.5, 1.0, 0.0])
        model.quantizer.head_cqi.bias.copy_(bias_cqi)
        print(f"  head_cqi bias: {bias_cqi.tolist()} (favor 4-bit)")

    # Augmented Lagrangian initialization
    lagrange_mu = torch.tensor(0.0, device=device)
    rho = 0.01
    rho_max = 10.0
    print(f"  Lagrange multiplier (mu) initialized to: {lagrange_mu.item():.4f}")
    print(f"  Penalty coefficient (rho) initialized to: {rho:.4f}")

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=max(args.epochs_phase2, 1), eta_min=1e-5
    )

    best_val_ber = float('inf')
    best_state = None

    for epoch in range(args.epochs_phase2):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bit_penalty = 0.0
        epoch_bits = 0.0
        start_t = time.time()

        # Tau annealing: exponential decay
        if args.epochs_phase2 > 1:
            progress = epoch / (args.epochs_phase2 - 1)
        else:
            progress = 1.0
        tau = args.tau_start * (args.tau_end / max(args.tau_start, 1e-10)) ** progress
        tau = max(tau, 0.01)

        for batch_idx in range(args.batches_per_epoch):
            s_hat_np, s_np, H_np, snr_np, y_np = generate_data_batch(
                sys_model, args.batch_size, p_tx_dbm=args.train_snr_dbm
            )
            s_hat_real_t, cp_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, y_np, device)

            optimizer_phase2.zero_grad()
            detected, exp_bits_per_link = model(
                s_hat_real_t, cp_t, snr_t, tau=tau, use_quantization=True
            )

            # MSE loss
            mse_loss = mse_criterion(detected, Y_t)

            # Augmented Lagrangian bit penalty
            avg_bits = exp_bits_per_link.mean()
            bit_violation = avg_bits - args.c_target
            bit_penalty = lagrange_mu * bit_violation + 0.5 * rho * bit_violation ** 2

            loss = mse_loss + bit_penalty
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

        # Update Lagrange multiplier and rho
        with torch.no_grad():
            lagrange_mu = lagrange_mu + rho * (epoch_avg_bits - args.c_target)
            lagrange_mu = torch.clamp(lagrange_mu, min=0.0)
            if epoch_avg_bits > args.c_target * 1.1:
                rho = min(rho * 1.5, rho_max)

        # Validation every 5 epochs or first/last
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs_phase2 - 1:
            model.eval()
            with torch.no_grad():
                val_bers = {}
                val_bits_dict = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    detected_val, val_bits = model(
                        td['s_hat_real'], td['channel_power'], td['snr'],
                        tau=tau, use_quantization=True
                    )
                    det_np = detected_val.cpu().numpy()
                    s_pred = det_np[..., 0] + 1j * det_np[..., 1]
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred)
                    val_bits_dict[p_val] = val_bits.mean().item()

                # Get bit allocation at 10 dBm
                td10 = test_dataset[10]
                s_hat_q_10, cqi_q_10, bits_10, w_d_10, w_c_10 = model.quantizer(
                    td10['s_hat_real'], td10['channel_power'], td10['snr'], tau=tau
                )
                w_d_mean = w_d_10.mean(dim=(0, 1, 2))  # (6,)
                w_c_mean = w_c_10.mean(dim=(0, 1, 2))  # (3,)

            lr_now = scheduler_phase2.get_last_lr()[0]
            demod_bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
            cqi_bit_labels = ['2b', '4b', '8b']

            print(f"Phase2 Epoch [{epoch + 1:03d}/{args.epochs_phase2}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | "
                  f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}, BitPen: {avg_bit_penalty:.5f}) | "
                  f"Avg Bits: {epoch_avg_bits:.1f} (target: {args.c_target})")
            print(f"  mu: {lagrange_mu.item():.6f} | rho: {rho:.6f}")
            print(f"  Val BER: 0dBm={val_bers[0]:.4f} (Dist-Full:{baseline_ber_dist_full[0]:.4f}, C-MMSE:{baseline_ber_cmmse[0]:.4f}), "
                  f"10dBm={val_bers[10]:.4f} (Dist-Full:{baseline_ber_dist_full[10]:.4f}, C-MMSE:{baseline_ber_cmmse[10]:.4f}), "
                  f"20dBm={val_bers[20]:.4f} (Dist-Full:{baseline_ber_dist_full[20]:.4f}, C-MMSE:{baseline_ber_cmmse[20]:.4f})")
            print(f"  Val Bits/link: 0dBm={val_bits_dict[0]:.1f}, 10dBm={val_bits_dict[10]:.1f}, 20dBm={val_bits_dict[20]:.1f}")
            d_dist_str = ' '.join([f'{demod_bit_labels[i]}:{w_d_mean[i]:.3f}' for i in range(6)])
            c_dist_str = ' '.join([f'{cqi_bit_labels[i]}:{w_c_mean[i]:.3f}' for i in range(3)])
            print(f"  Bit dist @10dBm: demod [{d_dist_str}] cqi [{c_dist_str}]")

            # Track best model based on average BER at 0, 10, 20 dBm
            avg_val_ber = np.mean([val_bers[p] for p in [0, 10, 20]])
            if avg_val_ber < best_val_ber:
                best_val_ber = avg_val_ber
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  -> New best model! Avg val BER = {avg_val_ber:.6f}")

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
              f"{'vs Dist-Full':<12} | {'Bits/link':<10}")
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

            detected_val, exp_bits = model(
                td['s_hat_real'], td['channel_power'], td['snr'],
                tau=tau_eval, use_quantization=True
            )

            det_np = detected_val.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_total = exp_bits.mean().item()

            gap_dist_full = ((ber_proposed - ber_dist_full) / max(ber_dist_full, 1e-10)) * 100

            results_proposed[p] = {
                'ber': ber_proposed, 'bits': avg_bits_total
            }

            print(f"{p:<10} | {ber_dist_full:<11.6f} | {ber_cmmse:<11.6f} | {ber_cmmse_q:<11.6f} | "
                  f"{ber_cmmse_mq:<11.6f} | {ber_dist_q:<11.6f} | {ber_proposed:<11.6f} | "
                  f"{gap_dist_full:>+10.2f}% | {avg_bits_total:<10.2f}")

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
    avg_total_bits = np.mean([results_proposed[p]['bits'] for p in test_p_tx_list])

    print(f"  Average Dist-Full BER:        {avg_ber_dist:.6f}")
    print(f"  Average C-MMSE BER:           {avg_ber_cmmse:.6f} (upper bound, full precision)")
    print(f"  Average C-MMSE-Q BER:         {avg_ber_cmmse_q:.6f} (uniform quantization)")
    print(f"  Average C-MMSE-Mixed-Q BER:   {avg_ber_cmmse_mq:.6f} (optimized mixed quantization)")
    print(f"  Average Dist-Q BER:           {avg_ber_dist_q:.6f} (distributed with quantization)")
    print(f"  Average Proposed BER:         {avg_ber_proposed:.6f}")
    print(f"  Average Total Bits/Link:      {avg_total_bits:.2f} (target: {args.c_target})")
    print(f"  Full Precision Bits/Link (V5): {fp_bits_per_link:.1f}")
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
    demod_bit_labels = ['2b', '4b', '6b', '8b', '12b', '16b']
    cqi_bit_labels = ['2b', '4b', '8b']
    with torch.no_grad():
        tau_eval = args.tau_end
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]

            s_hat_q_val, cqi_q_val, bits_val, w_d_val, w_c_val = model.quantizer(
                td['s_hat_real'], td['channel_power'], td['snr'], tau=tau_eval
            )

            w_d_mean = w_d_val.mean(dim=(0, 1, 2))  # (6,)
            w_c_mean = w_c_val.mean(dim=(0, 1, 2))  # (3,)

            avg_bits_link = bits_val.mean().item()

            d_dist_str = ', '.join([f'{demod_bit_labels[i]}: {w_d_mean[i]:.3f}' for i in range(6)])
            c_dist_str = ', '.join([f'{cqi_bit_labels[i]}: {w_c_mean[i]:.3f}' for i in range(3)])

            # Compute expected demod and CQI bits
            demod_bit_vals = torch.tensor([2.0, 4.0, 6.0, 8.0, 12.0, 16.0], device=device)
            cqi_bit_vals = torch.tensor([2.0, 4.0, 8.0], device=device)
            avg_demod_bits = 2.0 * (w_d_mean * demod_bit_vals).sum().item()
            avg_cqi_bits = 1.0 * (w_c_mean * cqi_bit_vals).sum().item()

            print(f"\n  p_tx = {p} dBm:")
            print(f"    Demod bit distribution: {d_dist_str}")
            print(f"    CQI bit distribution:   {c_dist_str}")
            print(f"    Avg demod bits/link: {avg_demod_bits:.2f}, Avg CQI bits/link: {avg_cqi_bits:.2f}")
            print(f"    Avg total bits/link: {avg_bits_link:.2f}")

    # ===========================
    # Per-AP bit allocation at 10 dBm
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"Per-AP Bit Allocation at 10 dBm (first 5 test samples)")
    print(f"{'=' * 110}")
    with torch.no_grad():
        td = test_dataset[10]
        n_sub = min(5, td['s_hat_real'].shape[0])
        s_hat_sub = td['s_hat_real'][:n_sub]
        cp_sub = td['channel_power'][:n_sub]
        snr_sub = td['snr'][:n_sub]

        _, _, bits_sub, w_d_sub, w_c_sub = model.quantizer(s_hat_sub, cp_sub, snr_sub, tau=tau_eval)

        avg_bits_per_ap = bits_sub.mean(dim=(0, 2))  # (L,)
        for l in range(args.L):
            print(f"  AP {l:2d}: avg bits/link = {avg_bits_per_ap[l].item():.1f}")

    # ===========================
    # LSQ scale parameter values
    # ===========================
    print(f"\n{'=' * 110}")
    print(f"LSQ Quantizer Scale Parameters (learned)")
    print(f"{'=' * 110}")
    q = model.quantizer
    demod_scales = [q.q_demod[i].s.item() for i in range(6)]
    cqi_scales = [q.q_cqi[i].s.item() for i in range(3)]
    print(f"  Demod quantizers: " + ', '.join([f'{demod_bit_labels[i]} s={demod_scales[i]:.6f}' for i in range(6)]))
    print(f"  CQI quantizers:   " + ', '.join([f'{cqi_bit_labels[i]} s={cqi_scales[i]:.6f}' for i in range(3)]))

    # ===========================
    # Save model
    # ===========================
    save_path = 'new_joint_model_v6.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
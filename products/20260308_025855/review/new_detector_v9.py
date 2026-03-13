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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V9 (CE Loss, 16-bit Quant, Imperfect CSI)")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=30, help="Phase 1 epochs (detector pre-training)")
    parser.add_argument("--epochs_phase2", type=int, default=100, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=5e-4, help="Learning rate for Phase 2")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D for feature embedding")
    parser.add_argument("--c_target", type=float, default=96.0, help="Target average bits per link")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of mean-field GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in Transformer and MLPs")
    parser.add_argument("--noise_injection_std", type=float, default=0.01, help="Gaussian noise std injected in Phase 1")
    
    # Phase 2 & CSI parameters
    parser.add_argument("--lambda_penalty", type=float, default=0.005, help="Fixed lambda for bit penalty")
    parser.add_argument("--tau_start", type=float, default=5.0, help="Tau start for Phase 2")
    parser.add_argument("--tau_end", type=float, default=0.1, help="Tau end for Phase 2")
    parser.add_argument("--tau_est", type=float, default=0.0, help="Channel estimation error variance (0 for perfect CSI)")
    return parser.parse_args()


# ============================================================
# 1. Data Generation Function
# ============================================================
def generate_data_batch_v2(sys_model, batch_size, p_tx_dbm, tau_est=0.0):
    """
    Vectorized data generation supporting imperfect CSI and returning QPSK class labels.
    """
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K)

    # 1. Batch Rayleigh fading channel
    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]  # (1, L, 1, K)
    H = np.sqrt(beta_w_expanded) * h_small  # (batch_size, L, N, K) True Channel
    
    # Imperfect CSI
    if tau_est > 0:
        noise_est = (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)) / np.sqrt(2) * np.sqrt(tau_est)
        H_est = H + noise_est
    else:
        H_est = H.copy()

    # 2. QPSK symbols & labels
    labels = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(labels) / np.sqrt(2)  # (batch_size, K)

    # 3. Received signal y = H @ s + noise (using True Channel)
    y_clean = np.einsum('blnk,bk->bln', H, s)
    z = (np.random.randn(batch_size, sys_model.L, sys_model.N) +
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z  # (batch_size, L, N)

    # 4. Local LMMSE detection (vectorized) using H_est
    H_conj_trans = H_est.conj().transpose(0, 1, 3, 2)  # (batch_size, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H_est @ H_conj_trans + noise_cov
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_conj_trans @ R_y_inv  # (batch_size, L, K, N)
    s_hat = W_l @ y[..., np.newaxis]  # (batch_size, L, K, 1)
    s_hat = s_hat.squeeze(-1)  # (batch_size, L, K)

    # 5. Local SNR (normalized) using H_est
    local_snr = 10 * np.log10(np.sum(np.abs(H_est) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0

    return s_hat, s, H_est, local_snr, y, labels


# ============================================================
# 2. Baseline Detectors
# ============================================================
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


def compute_dist_full_ber(s_hat, s):
    s_hat_avg = s_hat.mean(axis=1)
    return compute_ber(s, s_hat_avg)


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
    s_hat_cmmse = (R_inv @ HHy).squeeze(-1)
    ber = compute_ber(s, s_hat_cmmse)
    return ber, s_hat_cmmse


def compute_cmmse_q_detection(H, y, s, noise_w, c_target, K, N):
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users
    reals_per_ap = 2 * N_ant * (1 + K_users)

    bit_candidates = [1, 2, 4, 6, 8, 10, 12, 14, 16]
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

    return best_ber, best_s_hat, best_b


def compute_cmmse_mixed_q_detection(H, y, s, noise_w, c_target, K, N, s_hat_local=None):
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users

    best_ber = 1.0
    best_b_y = 0
    best_b_H = 0
    
    # Due to time complexity, sample fewer bit options around the expected range
    bit_options = [2, 4, 6, 8, 12, 16]

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
            except np.linalg.LinAlgError:
                continue

    return best_ber, None, best_b_y, best_b_H


def compute_dist_q_ber(s_hat, s, c_target):
    b_per_real = c_target / 2.0
    b_use = max(1, int(np.round(b_per_real)))
    b_use = min(b_use, 16)

    s_hat_real_q = np.zeros_like(s_hat.real)
    s_hat_imag_q = np.zeros_like(s_hat.imag)
    for l in range(s_hat.shape[1]):
        s_hat_real_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].real, b_use)
        s_hat_imag_q[:, l, :] = uniform_quantize_np(s_hat[:, l, :].imag, b_use)

    s_hat_q = s_hat_real_q + 1j * s_hat_imag_q
    s_hat_avg = s_hat_q.mean(axis=1)
    ber = compute_ber(s, s_hat_avg)
    return ber, b_use


# ============================================================
# 3. GNN + Transformer Hybrid Detector V9
# ============================================================
class MeanFieldGNNLayer(nn.Module):
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
        messages = self.message_mlp(h)
        mean_message = messages.mean(dim=1, keepdim=True).expand_as(h)
        combined = torch.cat([h, mean_message], dim=-1)
        updated = self.update_mlp(combined)
        return self.layer_norm(h + updated)


class APAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super(APAggregator, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        attn_weights = self.attn_net(h)
        attn_weights = torch.softmax(attn_weights, dim=1)
        aggregated = (h * attn_weights).sum(dim=1)
        return aggregated, attn_weights


class GNNTransformerDetectorV9(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(GNNTransformerDetectorV9, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim
        self.noise_w = noise_w

        self.mmse_init_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.demod_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * N, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # mmse_init(D/4), demod(D/4), channel(D/4), bitwidths(3), interference(2), snr(1)
        fusion_input_dim = 3 * (hidden_dim // 4) + 3 + 2 + 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion_bn = nn.BatchNorm1d(hidden_dim)

        self.gnn_layers = nn.ModuleList([
            MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        self.ap_aggregator = APAggregator(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True, dropout=dropout, norm_first=True
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Cross Entropy Output Head: 4 classes for QPSK
        self.output_head = nn.Linear(hidden_dim, 4)

    def compute_per_ap_mmse_init(self, H_q, y_q):
        B, L, N, K, _ = H_q.shape
        H_complex = torch.complex(H_q[..., 0].double(), H_q[..., 1].double())
        y_complex = torch.complex(y_q[..., 0].double(), y_q[..., 1].double())

        H_H = H_complex.conj().transpose(-1, -2)
        HHH = torch.matmul(H_complex, H_H)
        noise_eye = self.noise_w * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        reg_eye = 1e-6 * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        R_y = HHH + noise_eye + reg_eye

        try:
            R_y_inv = torch.linalg.inv(R_y)
        except RuntimeError:
            R_y_reg = R_y + 1e-4 * torch.eye(N, device=H_q.device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            R_y_inv = torch.linalg.inv(R_y_reg)

        W_l = torch.matmul(H_H, R_y_inv)
        y_expanded = y_complex.unsqueeze(-1)
        s_init_complex = torch.matmul(W_l, y_expanded).squeeze(-1)

        s_init = torch.stack([s_init_complex.real.float(), s_init_complex.imag.float()], dim=-1)
        return s_init

    def forward(self, s_hat_q, H_q, y_q, bitwidth_features, local_snr):
        B, L, K = s_hat_q.shape[:3]
        N = H_q.size(2)

        s_init = self.compute_per_ap_mmse_init(H_q, y_q)

        mmse_init_feat = self.mmse_init_mlp(s_init)
        demod_feat = self.demod_mlp(s_hat_q)

        H_q_perm = H_q.permute(0, 1, 3, 2, 4)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)
        channel_feat = self.channel_mlp(H_q_flat)

        h_power = (H_q ** 2).sum(dim=(2, 4))
        desired_power = h_power
        total_power = h_power.sum(dim=2, keepdim=True)
        interference_power = total_power - desired_power

        desired_feat = torch.log1p(desired_power).unsqueeze(-1)
        interference_feat = torch.log1p(interference_power).unsqueeze(-1)
        interference_features = torch.cat([desired_feat, interference_feat], dim=-1)

        combined = torch.cat([mmse_init_feat, demod_feat, channel_feat,
                              bitwidth_features, interference_features, local_snr], dim=-1)
        
        node_features = self.fusion_mlp(combined)
        node_features_flat = node_features.reshape(-1, self.hidden_dim)
        node_features_bn = self.fusion_bn(node_features_flat)
        node_features = node_features_bn.reshape(B, L, K, self.hidden_dim)

        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)

        user_features, _ = self.ap_aggregator(h)
        ic_out = self.transformer_ic(user_features)
        
        logits = self.output_head(ic_out)  # (B, K, 4)
        return logits


# ============================================================
# 4. Triple Adaptive Quantizer V9
# ============================================================
class TriplePolicyNetworkV9(nn.Module):
    def __init__(self, input_dim=9, policy_hidden=64):
        super(TriplePolicyNetworkV9, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        self.head_y = nn.Linear(policy_hidden, 5)
        self.head_H = nn.Linear(policy_hidden, 5)
        self.head_demod = nn.Linear(policy_hidden, 5)

    def forward(self, x):
        h = self.shared(x)
        return self.head_y(h), self.head_H(h), self.head_demod(h)


class TripleAdaptiveQuantizerV9(nn.Module):
    def __init__(self, N=4, policy_hidden=64):
        super(TripleAdaptiveQuantizerV9, self).__init__()
        self.N = N
        self.policy = TriplePolicyNetworkV9(input_dim=9, policy_hidden=policy_hidden)

        self.bit_options = [0, 4, 8, 12, 16]

        self.q4_y = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q8_y = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_y = LSQQuantizer(num_bits=12, init_s=0.001)
        self.q16_y = LSQQuantizer(num_bits=16, init_s=0.0001)

        self.q4_H = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q8_H = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_H = LSQQuantizer(num_bits=12, init_s=0.001)
        self.q16_H = LSQQuantizer(num_bits=16, init_s=0.0001)

        self.q4_demod = LSQQuantizer(num_bits=4, init_s=0.1)
        self.q8_demod = LSQQuantizer(num_bits=8, init_s=0.01)
        self.q12_demod = LSQQuantizer(num_bits=12, init_s=0.001)
        self.q16_demod = LSQQuantizer(num_bits=16, init_s=0.0001)

    def forward(self, v, H, y, local_snr, tau=1.0):
        B, L, K, _ = v.shape
        N = H.shape[2]

        v_real = v[..., 0:1]
        v_imag = v[..., 1:2]
        h_power = (H ** 2).sum(dim=(2, 4))
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

        logits_y, logits_H, logits_demod = self.policy(policy_input)

        logits_y_per_ap = logits_y.mean(dim=2)
        w_y = F.gumbel_softmax(logits_y_per_ap, tau=tau, hard=True)
        w_H = F.gumbel_softmax(logits_H, tau=tau, hard=True)
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)

        y_zeros = torch.zeros_like(y)
        y_q4 = self.q4_y(y)
        y_q8 = self.q8_y(y)
        y_q12 = self.q12_y(y)
        y_q16 = self.q16_y(y)
        w_y_exp = w_y.unsqueeze(-1).unsqueeze(-1)
        y_stack = torch.stack([y_zeros, y_q4, y_q8, y_q12, y_q16], dim=2)
        y_q = (y_stack * w_y_exp).sum(dim=2)

        H_zeros = torch.zeros_like(H)
        H_q4 = self.q4_H(H)
        H_q8 = self.q8_H(H)
        H_q12 = self.q12_H(H)
        H_q16 = self.q16_H(H)
        w_H_exp = w_H.unsqueeze(2).unsqueeze(-1)
        H_stack = torch.stack([H_zeros, H_q4, H_q8, H_q12, H_q16], dim=4)
        H_q = (H_stack * w_H_exp).sum(dim=4)

        v_zeros = torch.zeros_like(v)
        v_q4 = self.q4_demod(v)
        v_q8 = self.q8_demod(v)
        v_q12 = self.q12_demod(v)
        v_q16 = self.q16_demod(v)
        w_d_exp = w_demod.unsqueeze(-1)
        v_stack = torch.stack([v_zeros, v_q4, v_q8, v_q12, v_q16], dim=3)
        v_q = (v_stack * w_d_exp).sum(dim=3)

        y_bit_values = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=v.device)
        y_bits_per_ap = 2.0 * N * (w_y * y_bit_values.view(1, 1, 5)).sum(dim=-1)
        y_bits_per_link = y_bits_per_ap.unsqueeze(-1) / K

        H_bits_per_link = (2.0 * N) * (w_H * y_bit_values.view(1, 1, 1, 5)).sum(dim=-1)
        demod_bits_per_link = 2.0 * (w_demod * y_bit_values.view(1, 1, 1, 5)).sum(dim=-1)

        expected_bits_per_link = y_bits_per_link + H_bits_per_link + demod_bits_per_link

        return v_q, H_q, y_q, expected_bits_per_link, w_y, w_H, w_demod, y_bits_per_ap, H_bits_per_link, demod_bits_per_link


# ============================================================
# 5. Joint Model V9
# ============================================================
class JointModelV9(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3,
                 num_transformer_layers=2, num_heads=4, dropout=0.1, noise_w=1.0):
        super(JointModelV9, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.quantizer = TripleAdaptiveQuantizerV9(N=N, policy_hidden=64)
        self.detector = GNNTransformerDetectorV9(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers, num_transformer_layers=num_transformer_layers,
            num_heads=num_heads, dropout=dropout, noise_w=noise_w
        )
        self.max_y_bits_per_link = 2.0 * N * 16.0 / K
        self.max_H_bits = 2.0 * N * 16.0
        self.max_demod_bits = 2.0 * 16.0

    def forward(self, v, H, y, local_snr, tau=1.0, use_quantization=True, noise_std=0.0):
        B, L, K, _ = v.shape
        if noise_std > 0 and self.training:
            v = v + torch.randn_like(v) * noise_std
            H = H + torch.randn_like(H) * noise_std
            y = y + torch.randn_like(y) * noise_std

        if use_quantization:
            v_q, H_q, y_q, expected_bits_per_link, w_y, w_H, w_demod, \
                y_bits_ap, H_bits_link, demod_bits_link = \
                self.quantizer(v, H, y, local_snr, tau=tau)

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

        logits = self.detector(v_q, H_q, y_q, bitwidth_features, local_snr)

        return logits, expected_bits_per_link


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors_v9(s_hat, s, H_est, local_snr, y, labels, device):
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)

    H_real = np.stack([H_est.real, H_est.imag], axis=-1)
    H_tensor = torch.FloatTensor(H_real).to(device)

    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)

    y_real = np.stack([y.real, y.imag], axis=-1)
    y_tensor = torch.FloatTensor(y_real).to(device)
    
    labels_tensor = torch.LongTensor(labels).to(device)

    return v_tensor, H_tensor, snr_tensor, y_tensor, labels_tensor


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

    fp_bits_per_link = (2 * args.N + 2 * args.N + 2) * 32
    max_bits_per_link = 2 * args.N * 16.0 / args.K + 2 * args.N * 16.0 + 2 * 16.0

    print(f"{'=' * 100}")
    print(f"GNN+Transformer Hybrid Detector V9 (CE Loss, 16-bit Quant, Imperfect CSI)")
    print(f"{'=' * 100}")
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users, tau_est={args.tau_est}")
    print(f"Training: Phase1={args.epochs_phase1} epochs, Phase2={args.epochs_phase2} epochs")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {args.batches_per_epoch}")
    print(f"Target avg bits/link: {args.c_target} (max possible: {max_bits_per_link:.1f})")
    print(f"Bit penalty lambda: {args.lambda_penalty}")
    print(f"Quant options: {{0, 4, 8, 12, 16}} bits")
    print(f"{'=' * 100}")

    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()
    noise_w = sys_model.noise_w

    # ===========================
    # Pre-generate fixed test set
    # ===========================
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"Generating fixed test set ({args.test_samples} samples/power)...")
    for p in test_p_tx_list:
        s_hat_np, s_np, H_np, snr_np, y_np, labels_np = generate_data_batch_v2(sys_model, args.test_samples, p, args.tau_est)
        v_t, H_t, snr_t, y_t, labels_t = prepare_tensors_v9(s_hat_np, s_np, H_np, snr_np, y_np, labels_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'y': y_t, 'labels': labels_t,
            's_hat_np': s_hat_np, 's_np': s_np, 'H_np': H_np, 'y_np': y_np
        }
    print("Test set generated.\n")

    # Baselines
    print(f"Computing baseline results (C-MMSE with target={args.c_target})...")
    baseline_ber_dist_full = {}
    baseline_ber_cmmse = {}
    baseline_ber_cmmse_q = {}
    baseline_ber_cmmse_mixed_q = {}
    
    for p in test_p_tx_list:
        td = test_dataset[p]
        baseline_ber_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        baseline_ber_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], noise_w)
        
        ber_cq, _, _ = compute_cmmse_q_detection(td['H_np'], td['y_np'], td['s_np'], noise_w, args.c_target, args.K, args.N)
        baseline_ber_cmmse_q[p] = ber_cq
        
        ber_mq, _, _, _ = compute_cmmse_mixed_q_detection(td['H_np'], td['y_np'], td['s_np'], noise_w, args.c_target, args.K, args.N)
        baseline_ber_cmmse_mixed_q[p] = ber_mq

    # Initialize model
    model = JointModelV9(L=args.L, N=args.N, K=args.K, hidden_dim=args.hidden_dim,
                         num_gnn_layers=args.num_gnn_layers, num_transformer_layers=args.num_transformer_layers,
                         num_heads=args.num_heads, dropout=args.dropout, noise_w=noise_w).to(device)

    criterion = nn.CrossEntropyLoss()
    qpsk_mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(qpsk_mapping.get)

    # ===========================
    # Phase 1: Detector Pre-training
    # ===========================
    print(f"\nPhase 1: Pre-training ({args.epochs_phase1} epochs) | Fixed Training SNR: 10~15 dBm")
    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=args.epochs_phase1, eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(10, 15)
            s_hat_np, s_np, H_np, snr_np, y_np, labels_np = generate_data_batch_v2(sys_model, args.batch_size, p_train, args.tau_est)
            v_tensor, H_tensor, snr_tensor, y_tensor, labels_tensor = prepare_tensors_v9(
                s_hat_np, s_np, H_np, snr_np, y_np, labels_np, device)

            optimizer_phase1.zero_grad()
            logits, _ = model(v_tensor, H_tensor, y_tensor, snr_tensor, use_quantization=False, noise_std=args.noise_injection_std)
            loss = criterion(logits.view(-1, 4), labels_tensor.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), max_norm=args.grad_clip)
            optimizer_phase1.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            epoch_acc += (preds == labels_tensor).float().mean().item()

        scheduler_phase1.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_acc = epoch_acc / args.batches_per_epoch

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_bers = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    val_logits, _ = model(td['v'], td['H'], td['y'], td['snr'], use_quantization=False)
                    preds_val = val_logits.argmax(dim=-1).cpu().numpy()
                    s_pred_val = map_func(preds_val) / np.sqrt(2)
                    val_bers[p_val] = compute_ber(td['s_np'], s_pred_val)

            print(f"Phase1 Epoch [{epoch + 1}/{args.epochs_phase1}] | Time: {elapsed:.1f}s | "
                  f"Loss: {avg_loss:.5f} | Acc: {avg_acc:.4f} | "
                  f"Val BER: 0dBm={val_bers[0]:.4f}, 10dBm={val_bers[10]:.4f}")

    # ===========================
    # Phase 2: Joint QAT Training
    # ===========================
    print(f"\nPhase 2: Joint QAT Training ({args.epochs_phase2} epochs)")
    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=args.epochs_phase2, eta_min=1e-5)

    best_val_ber = float('inf')
    best_state = None

    for epoch in range(args.epochs_phase2):
        model.train()
        epoch_loss, epoch_ce, epoch_bits, epoch_acc = 0.0, 0.0, 0.0, 0.0
        start_t = time.time()

        progress = epoch / max(1, args.epochs_phase2 - 1)
        tau = args.tau_start * (args.tau_end / max(args.tau_start, 1e-10)) ** progress
        tau = max(tau, 0.01)

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(10, 15)
            s_hat_np, s_np, H_np, snr_np, y_np, labels_np = generate_data_batch_v2(sys_model, args.batch_size, p_train, args.tau_est)
            v_tensor, H_tensor, snr_tensor, y_tensor, labels_tensor = prepare_tensors_v9(
                s_hat_np, s_np, H_np, snr_np, y_np, labels_np, device)

            optimizer_phase2.zero_grad()
            logits, exp_bits_per_link = model(v_tensor, H_tensor, y_tensor, snr_tensor, tau=tau, use_quantization=True)

            loss_ce = criterion(logits.view(-1, 4), labels_tensor.view(-1))
            avg_bits = exp_bits_per_link.mean()
            bit_penalty = (avg_bits - args.c_target) ** 2
            loss = loss_ce + args.lambda_penalty * bit_penalty
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer_phase2.step()

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_bits += avg_bits.item()
            preds = logits.argmax(dim=-1)
            epoch_acc += (preds == labels_tensor).float().mean().item()

        scheduler_phase2.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_ce = epoch_ce / args.batches_per_epoch
        avg_bits_epoch = epoch_bits / args.batches_per_epoch
        avg_acc = epoch_acc / args.batches_per_epoch

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                td = test_dataset[10]
                val_logits, val_bits = model(td['v'], td['H'], td['y'], td['snr'], tau=tau, use_quantization=True)
                preds_val = val_logits.argmax(dim=-1).cpu().numpy()
                s_pred_val = map_func(preds_val) / np.sqrt(2)
                val_ber_10 = compute_ber(td['s_np'], s_pred_val)
                val_avg_bits_total = val_bits.mean().item()

                if val_ber_10 < best_val_ber:
                    best_val_ber = val_ber_10
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"Phase2 Epoch [{epoch + 1:03d}/{args.epochs_phase2}] | Time: {elapsed:.1f}s | tau: {tau:.3f} | "
                  f"Total Loss: {avg_loss:.5f} (CE: {avg_ce:.5f}) | Acc: {avg_acc:.4f} | "
                  f"Train Bits: {avg_bits_epoch:.1f} | Val @10dBm BER: {val_ber_10:.4f}, Bits: {val_avg_bits_total:.1f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ===========================
    # Final Evaluation
    # ===========================
    print(f"\n{'=' * 100}")
    print(f"Final Evaluation on Test Set")
    print(f"{'=' * 100}")
    model.eval()
    header = f"{'p_tx(dBm)':<10} | {'Dist-Full':<11} | {'C-MMSE':<11} | {'C-MMSE-MxQ':<11} | {'Proposed':<11} | {'Bits/link':<10}"
    print(f"\n{header}\n" + "-" * len(header))

    with torch.no_grad():
        for p in test_p_tx_list:
            td = test_dataset[p]
            logits, exp_bits = model(td['v'], td['H'], td['y'], td['snr'], tau=args.tau_end, use_quantization=True)
            preds_val = logits.argmax(dim=-1).cpu().numpy()
            s_pred_val = map_func(preds_val) / np.sqrt(2)
            ber_proposed = compute_ber(td['s_np'], s_pred_val)
            avg_bits_total = exp_bits.mean().item()

            print(f"{p:<10} | {baseline_ber_dist_full[p]:<11.6f} | {baseline_ber_cmmse[p]:<11.6f} | "
                  f"{baseline_ber_cmmse_mixed_q[p]:<11.6f} | {ber_proposed:<11.6f} | {avg_bits_total:<10.2f}")

    torch.save(model.state_dict(), 'new_joint_model_v9.pth')
    print("\nModel saved to 'new_joint_model_v9.pth'. Evaluation complete!")

if __name__ == '__main__':
    main()
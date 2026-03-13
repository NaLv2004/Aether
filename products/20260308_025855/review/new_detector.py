import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import os

from system_model import CellFreeSystem
from lsq_quantizer import LSQQuantizer


def parse_args():
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector with Dual Adaptive Quantization")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (Phase1 + Phase2)")
    parser.add_argument("--phase1_epochs", type=int, default=20, help="Phase 1 epochs (detector pre-training)")
    parser.add_argument("--batches_per_epoch", type=int, default=80, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension D for feature embedding")
    parser.add_argument("--lambda_val", type=float, default=0.1, help="Lagrangian multiplier for bit constraint")
    parser.add_argument("--c_target", type=float, default=20.0, help="Target average bits per (AP, user) link")
    parser.add_argument("--tau_init", type=float, default=2.0, help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--tau_min", type=float, default=0.3, help="Minimum Gumbel-Softmax temperature")
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


# ============================================================
# 1. Data Generation Function
# ============================================================
def generate_data_batch_v2(sys_model, batch_size, p_tx_dbm):
    """
    Vectorized data generation that returns local LMMSE estimates, true symbols,
    full channel matrix H, and local SNR features.
    
    Returns:
        s_hat: (batch_size, L, K) complex - local LMMSE estimates
        s: (batch_size, K) complex - true transmitted QPSK symbols
        H: (batch_size, L, N, K) complex - full channel matrix
        local_snr: (batch_size, L, K) real - local SNR features (normalized)
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

    # 4. Local LMMSE detection (vectorized over batch and APs)
    H_conj_trans = H.conj().transpose(0, 1, 3, 2)  # (batch_size, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H @ H_conj_trans + noise_cov  # (batch_size, L, N, N)
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_conj_trans @ R_y_inv  # (batch_size, L, K, N)
    s_hat = W_l @ y[..., np.newaxis]  # (batch_size, L, K, 1)
    s_hat = s_hat.squeeze(-1)  # (batch_size, L, K)

    # 5. Local SNR (normalized to roughly [0, 1] range)
    local_snr = 10 * np.log10(np.sum(np.abs(H) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0
    # shape: (batch_size, L, K)

    return s_hat, s, H, local_snr


# ============================================================
# 2. GNN + Transformer Hybrid Detector
# ============================================================
class GNNLayer(nn.Module):
    """
    GNN message passing layer operating on AP dimension for each user independently.
    Uses attention-based aggregation across APs.
    """
    def __init__(self, hidden_dim):
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Attention MLP: takes concatenation of two node features
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        """
        h: (B, L, K, D) - node features for each AP and user
        Returns: (B, L, K, D) - updated node features
        """
        B, L, K, D = h.shape

        # Compute messages from all APs: (B, L, K, D)
        messages = self.message_mlp(h)  # (B, L, K, D)

        # Compute pairwise attention scores
        # Expand for pairwise: h_i (B, L, 1, K, D) vs h_j (B, 1, L, K, D)
        h_i = h.unsqueeze(2).expand(B, L, L, K, D)  # (B, L, L, K, D) - receiving node
        h_j = h.unsqueeze(1).expand(B, L, L, K, D)  # (B, L, L, K, D) - sending node
        
        # Concatenate pairs
        h_pair = torch.cat([h_i, h_j], dim=-1)  # (B, L, L, K, 2D)
        
        # Compute attention logits
        attn_logits = self.attn_mlp(h_pair).squeeze(-1)  # (B, L, L, K)
        
        # Softmax over source dimension (dim=2 is the sender dimension)
        attn_weights = torch.softmax(attn_logits, dim=2)  # (B, L, L, K)

        # Weighted aggregation of messages
        # messages: (B, L, K, D) -> expand to (B, 1, L, K, D) for broadcasting
        messages_expanded = messages.unsqueeze(1).expand(B, L, L, K, D)  # (B, L_recv, L_send, K, D)
        attn_expanded = attn_weights.unsqueeze(-1)  # (B, L, L, K, 1)
        
        aggregated = (attn_expanded * messages_expanded).sum(dim=2)  # (B, L, K, D)

        # Residual connection + layer norm
        h_new = self.layer_norm(h + aggregated)
        return h_new


class APAggregator(nn.Module):
    """
    Learned attention-based aggregation across APs to produce per-user features.
    """
    def __init__(self, hidden_dim):
        super(APAggregator, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        """
        h: (B, L, K, D) -> aggregated: (B, K, D)
        """
        attn_weights = self.attn_net(h)  # (B, L, K, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # softmax over L
        aggregated = (h * attn_weights).sum(dim=1)  # (B, K, D)
        return aggregated, attn_weights


class GNNTransformerDetector(nn.Module):
    """
    Hybrid GNN + Transformer detector for Cell-Free MIMO.
    
    Architecture:
    1. AP Feature Extraction: MLPs to process quantized demod results and channel coefficients
    2. GNN Layers: Message passing across APs for spatial aggregation
    3. AP Aggregation: Learned attention-based fusion across APs
    4. Transformer: User-level interference cancellation
    5. Output Head: Predict detected symbols with residual connection
    """
    def __init__(self, L, N, K, hidden_dim=64, num_gnn_layers=2, num_transformer_layers=2, num_heads=4):
        super(GNNTransformerDetector, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # --- AP Feature Extraction ---
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

        # Projection after concatenation: (B, L, K, 2*D + 2) -> (B, L, K, D)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- GNN Layers ---
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # --- AP Aggregation ---
        self.ap_aggregator = APAggregator(hidden_dim)

        # --- Soft info aggregation (for residual) ---
        self.soft_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # --- Transformer for user-level IC ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Output Head ---
        self.output_head = nn.Linear(hidden_dim, 2)
        # Zero-init for stable residual learning
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, s_hat_q, H_q, bitwidth_features):
        """
        Args:
            s_hat_q: (B, L, K, 2) - quantized demod results (real, imag)
            H_q: (B, L, N, K, 2) - quantized channel coefficients (real, imag)
            bitwidth_features: (B, L, K, 2) - bitwidth features (demod_bits, channel_bits)
        
        Returns:
            detected: (B, K, 2) - detected symbols (real, imag)
        """
        B = s_hat_q.size(0)
        L = s_hat_q.size(1)
        K = s_hat_q.size(2)
        N = H_q.size(2)

        # --- 1. AP Feature Extraction ---
        # Demod features
        demod_feat = self.demod_mlp(s_hat_q)  # (B, L, K, D)

        # Channel features: reshape H_q from (B, L, N, K, 2) to per-user features
        # For each user k, concatenate N antenna real+imag: (B, L, K, 2*N)
        H_q_perm = H_q.permute(0, 1, 3, 2, 4)  # (B, L, K, N, 2)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)  # (B, L, K, 2*N)
        channel_feat = self.channel_mlp(H_q_flat)  # (B, L, K, D)

        # Concatenate all features
        combined = torch.cat([demod_feat, channel_feat, bitwidth_features], dim=-1)  # (B, L, K, 2D+2)
        node_features = self.fusion_mlp(combined)  # (B, L, K, D)

        # --- 2. GNN Message Passing ---
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)  # (B, L, K, D)

        # --- 3. AP Aggregation ---
        # Soft info weighted mean (for residual baseline)
        soft_info = s_hat_q  # (B, L, K, 2)
        soft_attn_weights = self.soft_attn(h)  # (B, L, K, 1)
        soft_attn_weights = torch.softmax(soft_attn_weights, dim=1)
        base_out = (soft_info * soft_attn_weights).sum(dim=1)  # (B, K, 2)

        # Feature aggregation for Transformer input
        user_features, _ = self.ap_aggregator(h)  # (B, K, D)

        # --- 4. Transformer IC ---
        ic_out = self.transformer_ic(user_features)  # (B, K, D)

        # --- 5. Output ---
        residual = self.output_head(ic_out)  # (B, K, 2)
        detected = base_out + residual

        return detected


# ============================================================
# 3. Dual Adaptive Quantizer
# ============================================================
class DualPolicyNetwork(nn.Module):
    """
    Policy network that outputs bitwidth decisions for both demod results and channel coefficients.
    Input features per (l, k): [v_real, v_imag, snr, channel_norm_per_user, avg_channel_power] -> 5 dims
    Output: logits for demod bitwidth and channel bitwidth, each choosing from {0, 2, 4} bits
    """
    def __init__(self, input_dim=5):
        super(DualPolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.head_demod = nn.Linear(32, 3)   # {0, 2, 4} bits
        self.head_channel = nn.Linear(32, 3)  # {0, 2, 4} bits

    def forward(self, x):
        """
        x: (..., 5) features
        Returns: logits_demod (..., 3), logits_channel (..., 3)
        """
        h = self.shared(x)
        return self.head_demod(h), self.head_channel(h)


class DualAdaptiveQuantizer(nn.Module):
    """
    Dual adaptive quantization module that quantizes both demod results and channel coefficients.
    Uses Gumbel-Softmax for differentiable bitwidth selection.
    """
    def __init__(self, N=4):
        super(DualAdaptiveQuantizer, self).__init__()
        self.N = N
        self.policy = DualPolicyNetwork(input_dim=5)

        # LSQ quantizers for demod results
        self.q2_demod = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_demod = LSQQuantizer(num_bits=4, init_s=0.1)

        # LSQ quantizers for channel coefficients
        self.q2_channel = LSQQuantizer(num_bits=2, init_s=0.5)
        self.q4_channel = LSQQuantizer(num_bits=4, init_s=0.1)

    def forward(self, v, H, local_snr, tau=1.0):
        """
        Args:
            v: (B, L, K, 2) - demod results (real, imag parts)
            H: (B, L, N, K, 2) - channel coefficients (real, imag parts)
            local_snr: (B, L, K, 1) - local SNR features
            tau: Gumbel-Softmax temperature
        
        Returns:
            v_q: (B, L, K, 2) - quantized demod results
            H_q: (B, L, N, K, 2) - quantized channel coefficients
            expected_bits_demod: (B, L, K)
            expected_bits_channel: (B, L, K)
            w_demod: (B, L, K, 3) - Gumbel-Softmax weights for demod
            w_channel: (B, L, K, 3) - Gumbel-Softmax weights for channel
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # --- Compute policy input features ---
        # v_real, v_imag: (B, L, K)
        v_real = v[..., 0:1]  # (B, L, K, 1)
        v_imag = v[..., 1:2]  # (B, L, K, 1)
        
        # Channel norm per user: ||H_{l,:,k}||^2, sum over N antennas and 2 (real+imag)
        # H shape: (B, L, N, K, 2)
        h_power = (H ** 2).sum(dim=(2, 4))  # (B, L, K) - sum over N and real/imag
        channel_norm = h_power.unsqueeze(-1)  # (B, L, K, 1)
        
        # Average channel power across APs for each user
        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)  # (B, L, K, 1)

        # Policy input: [v_real, v_imag, snr, channel_norm, avg_channel_power]
        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm, avg_channel_power], dim=-1)  # (B, L, K, 5)

        # --- Get bitwidth decisions ---
        logits_demod, logits_channel = self.policy(policy_input)  # each: (B, L, K, 3)

        # Gumbel-Softmax
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B, L, K, 3)
        w_channel = F.gumbel_softmax(logits_channel, tau=tau, hard=True)  # (B, L, K, 3)

        # --- Quantize demod results ---
        v_q2 = self.q2_demod(v)  # (B, L, K, 2)
        v_q4 = self.q4_demod(v)  # (B, L, K, 2)
        
        # Weighted combination based on Gumbel-Softmax selection
        v_q = (w_demod[..., 0:1] * 0.0 +
               w_demod[..., 1:2] * v_q2 +
               w_demod[..., 2:3] * v_q4)  # (B, L, K, 2)

        # --- Quantize channel coefficients ---
        H_q2 = self.q2_channel(H)  # (B, L, N, K, 2)
        H_q4 = self.q4_channel(H)  # (B, L, N, K, 2)
        
        # Expand w_channel for channel dimensions: (B, L, K, 3) -> (B, L, 1, K, 3)
        w_ch_exp = w_channel.unsqueeze(2)  # (B, L, 1, K, 3)
        
        H_q = (w_ch_exp[..., 0:1] * 0.0 +
               w_ch_exp[..., 1:2] * H_q2 +
               w_ch_exp[..., 2:3] * H_q4)  # (B, L, N, K, 2)

        # --- Expected bits ---
        # Demod: each complex value = 2 real values, each quantized at b bits -> 2*b bits per (l,k)
        expected_bits_demod = (w_demod[..., 0] * 0.0 +
                               w_demod[..., 1] * (2.0 * 2) +
                               w_demod[..., 2] * (2.0 * 4))  # (B, L, K)

        # Channel: N complex values = 2*N real values, each at b bits -> 2*N*b bits per (l,k)
        expected_bits_channel = (w_channel[..., 0] * 0.0 +
                                 w_channel[..., 1] * (2.0 * N * 2) +
                                 w_channel[..., 2] * (2.0 * N * 4))  # (B, L, K)

        return v_q, H_q, expected_bits_demod, expected_bits_channel, w_demod, w_channel


# ============================================================
# 4. Joint Model
# ============================================================
class JointModel(nn.Module):
    """
    End-to-end model combining DualAdaptiveQuantizer + GNNTransformerDetector.
    """
    def __init__(self, L, N, K, hidden_dim=64, num_gnn_layers=2, num_transformer_layers=2, num_heads=4):
        super(JointModel, self).__init__()
        self.quantizer = DualAdaptiveQuantizer(N=N)
        self.detector = GNNTransformerDetector(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads
        )

    def forward(self, v, H, local_snr, tau=1.0, use_quantization=True):
        """
        Args:
            v: (B, L, K, 2) - demod results (real+imag)
            H: (B, L, N, K, 2) - channel coefficients (real+imag)
            local_snr: (B, L, K, 1) - local SNR
            tau: Gumbel-Softmax temperature
            use_quantization: if False, skip quantization (Phase 1)
        
        Returns:
            detected: (B, K, 2) - detected symbols
            expected_bits_demod: (B, L, K)
            expected_bits_channel: (B, L, K)
        """
        B, L, K, _ = v.shape

        if use_quantization:
            v_q, H_q, expected_bits_demod, expected_bits_channel, w_demod, w_channel = \
                self.quantizer(v, H, local_snr, tau=tau)
            # Bitwidth features: normalized bit values for demod and channel
            bw_demod_feat = expected_bits_demod.unsqueeze(-1) / 8.0  # normalize by max (2*4=8)
            bw_channel_feat = expected_bits_channel.unsqueeze(-1) / 32.0  # normalize by max (2*4*4=32)
        else:
            v_q = v
            H_q = H
            expected_bits_demod = torch.zeros(B, L, K, device=v.device)
            expected_bits_channel = torch.zeros(B, L, K, device=v.device)
            # Full precision bitwidth features (indicate max)
            bw_demod_feat = torch.ones(B, L, K, 1, device=v.device)
            bw_channel_feat = torch.ones(B, L, K, 1, device=v.device)

        bitwidth_features = torch.cat([bw_demod_feat, bw_channel_feat], dim=-1)  # (B, L, K, 2)

        detected = self.detector(v_q, H_q, bitwidth_features)

        return detected, expected_bits_demod, expected_bits_channel


# ============================================================
# Helper functions
# ============================================================
def prepare_tensors(s_hat, s, H, local_snr, device):
    """
    Convert numpy complex arrays to real-valued torch tensors.
    """
    # s_hat: (B, L, K) complex -> (B, L, K, 2) real
    v = np.stack([s_hat.real, s_hat.imag], axis=-1)
    v_tensor = torch.FloatTensor(v).to(device)

    # H: (B, L, N, K) complex -> (B, L, N, K, 2) real
    H_real = np.stack([H.real, H.imag], axis=-1)
    H_tensor = torch.FloatTensor(H_real).to(device)

    # local_snr: (B, L, K) -> (B, L, K, 1)
    snr_tensor = torch.FloatTensor(local_snr).unsqueeze(-1).to(device)

    # s: (B, K) complex -> (B, K, 2) real
    Y = np.stack([s.real, s.imag], axis=-1)
    Y_tensor = torch.FloatTensor(Y).to(device)

    return v_tensor, H_tensor, snr_tensor, Y_tensor


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
    """
    Baseline: mean pooling across APs.
    """
    s_hat_avg = s_hat.mean(axis=1)  # (B, K)
    return compute_ber(s, s_hat_avg)


# ============================================================
# 5. Main Training Script
# ============================================================
def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=" * 70)
    print(f"GNN+Transformer Hybrid Detector with Dual Adaptive Quantization")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users")
    print(f"Training: {args.epochs} epochs (Phase1: {args.phase1_epochs}, Phase2: {args.epochs - args.phase1_epochs})")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}, Hidden dim: {args.hidden_dim}")
    print(f"Bit constraint: lambda={args.lambda_val}, C_target={args.c_target} bits/link")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"=" * 70)

    # Initialize system model
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    # Pre-generate fixed test set
    test_p_tx_list = [-10, -5, 0, 5, 10, 15, 20]
    test_dataset = {}
    print(f"\nGenerating fixed test set ({args.test_samples} samples per power point)...")
    for p in test_p_tx_list:
        s_hat_np, s_np, H_np, snr_np = generate_data_batch_v2(sys_model, args.test_samples, p_tx_dbm=p)
        v_t, H_t, snr_t, Y_t = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)
        test_dataset[p] = {
            'v': v_t, 'H': H_t, 'snr': snr_t, 'Y': Y_t,
            's_hat_np': s_hat_np, 's_np': s_np
        }
    print("Test set generated successfully.\n")

    # Initialize model
    model = JointModel(
        L=args.L, N=args.N, K=args.K,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    detector_params = sum(p.numel() for p in model.detector.parameters())
    quantizer_params = sum(p.numel() for p in model.quantizer.parameters())
    print(f"Model parameters: Total={total_params:,}, Detector={detector_params:,}, Quantizer={quantizer_params:,}")

    criterion = nn.MSELoss()

    # ===========================
    # Phase 1: Detector Pre-training (no quantization)
    # ===========================
    print(f"\n{'=' * 70}")
    print(f"Phase 1: Detector Pre-training ({args.phase1_epochs} epochs, full precision)")
    print(f"{'=' * 70}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=args.phase1_epochs, eta_min=1e-5)

    for epoch in range(args.phase1_epochs):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

            optimizer_phase1.zero_grad()
            detected, _, _ = model(v_tensor, H_tensor, snr_tensor, use_quantization=False)
            loss = criterion(detected, Y_tensor)
            loss.backward()
            optimizer_phase1.step()
            epoch_loss += loss.item()

        scheduler_phase1.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                td = test_dataset[0]
                val_out, _, _ = model(td['v'], td['H'], td['snr'], use_quantization=False)
                val_loss = criterion(val_out, td['Y']).item()
                val_np = val_out.cpu().numpy()
                s_pred = val_np[..., 0] + 1j * val_np[..., 1]
                val_ber = compute_ber(td['s_np'], s_pred)

            lr_now = scheduler_phase1.get_last_lr()[0]
            print(f"Phase1 Epoch [{epoch + 1:02d}/{args.phase1_epochs}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | "
                  f"Train Loss: {avg_loss:.5f} | Val Loss(0dBm): {val_loss:.5f} | Val BER(0dBm): {val_ber:.4f}")

    print("Phase 1 completed.\n")

    # ===========================
    # Phase 2: Joint QAT Training
    # ===========================
    phase2_epochs = args.epochs - args.phase1_epochs
    print(f"{'=' * 70}")
    print(f"Phase 2: Joint QAT Training ({phase2_epochs} epochs)")
    print(f"{'=' * 70}")

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=phase2_epochs, eta_min=1e-5)

    for epoch in range(phase2_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bits = 0.0
        start_t = time.time()

        # Temperature annealing
        progress = epoch / max(1, phase2_epochs - 1)
        tau = args.tau_init * (args.tau_min / args.tau_init) ** progress

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

            optimizer_phase2.zero_grad()
            detected, exp_bits_demod, exp_bits_channel = model(
                v_tensor, H_tensor, snr_tensor, tau=tau, use_quantization=True
            )

            # Detection loss
            mse_loss = criterion(detected, Y_tensor)

            # Bit constraint loss
            # Total bits per (l, k) link = demod_bits + channel_bits
            total_bits_per_link = exp_bits_demod + exp_bits_channel  # (B, L, K)
            avg_bits = total_bits_per_link.mean()
            bit_penalty = (avg_bits - args.c_target) ** 2

            loss = mse_loss + args.lambda_val * bit_penalty
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer_phase2.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_bits += avg_bits.item()

        scheduler_phase2.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch
        avg_mse = epoch_mse / args.batches_per_epoch
        avg_bits_epoch = epoch_bits / args.batches_per_epoch

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                td = test_dataset[0]
                val_out, val_bits_d, val_bits_c = model(
                    td['v'], td['H'], td['snr'], tau=tau, use_quantization=True
                )
                val_loss = criterion(val_out, td['Y']).item()
                val_np = val_out.cpu().numpy()
                s_pred = val_np[..., 0] + 1j * val_np[..., 1]
                val_ber = compute_ber(td['s_np'], s_pred)
                val_avg_bits_d = val_bits_d.mean().item()
                val_avg_bits_c = val_bits_c.mean().item()
                val_avg_bits_total = (val_bits_d + val_bits_c).mean().item()

            lr_now = scheduler_phase2.get_last_lr()[0]
            print(f"Phase2 Epoch [{epoch + 1:02d}/{phase2_epochs}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | "
                  f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}) | "
                  f"Avg Bits: {avg_bits_epoch:.1f} (target: {args.c_target}) | "
                  f"Val BER(0dBm): {val_ber:.4f} | "
                  f"Val Bits: D={val_avg_bits_d:.1f} C={val_avg_bits_c:.1f} T={val_avg_bits_total:.1f}")

    print("\nPhase 2 completed.\n")

    # ===========================
    # Final Evaluation
    # ===========================
    print(f"{'=' * 70}")
    print(f"Final Evaluation on Test Set ({args.test_samples} samples per power point)")
    print(f"{'=' * 70}")

    model.eval()
    print(f"\n{'p_tx(dBm)':<12} | {'Dist-Full BER':<15} | {'GNN-Trans BER':<15} | "
          f"{'Improve(%)':<12} | {'Avg Demod Bits':<15} | {'Avg Chan Bits':<15} | {'Avg Total Bits':<15}")
    print("-" * 110)

    with torch.no_grad():
        for p in test_p_tx_list:
            td = test_dataset[p]

            # Baseline: Dist-Full (mean pooling)
            ber_baseline = compute_dist_full_ber(td['s_hat_np'], td['s_np'])

            # Proposed: GNN+Transformer with quantization
            detected, exp_bits_d, exp_bits_c = model(
                td['v'], td['H'], td['snr'], tau=args.tau_min, use_quantization=True
            )
            det_np = detected.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_d = exp_bits_d.mean().item()
            avg_bits_c = exp_bits_c.mean().item()
            avg_bits_total = (exp_bits_d + exp_bits_c).mean().item()

            improve = ((ber_baseline - ber_proposed) / max(ber_baseline, 1e-10)) * 100

            print(f"{p:<12} | {ber_baseline:<15.4%} | {ber_proposed:<15.4%} | "
                  f"{improve:>10.2f}% | {avg_bits_d:<15.2f} | {avg_bits_c:<15.2f} | {avg_bits_total:<15.2f}")

    print("-" * 110)

    # Print detailed bit allocation statistics
    print(f"\n{'=' * 70}")
    print(f"Detailed Bit Allocation Statistics")
    print(f"{'=' * 70}")
    with torch.no_grad():
        for p in [0, 10, 20]:
            td = test_dataset[p]
            _, exp_bits_d, exp_bits_c = model(
                td['v'], td['H'], td['snr'], tau=args.tau_min, use_quantization=True
            )
            
            # Get quantizer weights
            v_q, H_q, ebd, ebc, w_d, w_c = model.quantizer(
                td['v'], td['H'], td['snr'], tau=args.tau_min
            )
            
            w_d_mean = w_d.mean(dim=(0, 1, 2))
            w_c_mean = w_c.mean(dim=(0, 1, 2))
            
            print(f"\np_tx = {p} dBm:")
            print(f"  Demod bitwidth distribution:   0-bit: {w_d_mean[0]:.3f}, 2-bit: {w_d_mean[1]:.3f}, 4-bit: {w_d_mean[2]:.3f}")
            print(f"  Channel bitwidth distribution: 0-bit: {w_c_mean[0]:.3f}, 2-bit: {w_c_mean[1]:.3f}, 4-bit: {w_c_mean[2]:.3f}")
            print(f"  Avg demod bits/link: {exp_bits_d.mean().item():.2f}, Avg channel bits/link: {exp_bits_c.mean().item():.2f}")
            print(f"  Avg total bits/link: {(exp_bits_d + exp_bits_c).mean().item():.2f}")
            
            # Full precision reference
            fp_bits = 2 * 32 + 2 * args.N * 32  # demod (2 reals * 32 bits) + channel (2N reals * 32 bits)
            compression = fp_bits / max((exp_bits_d + exp_bits_c).mean().item(), 1e-10)
            print(f"  Full precision bits/link: {fp_bits}, Compression ratio: {compression:.1f}x")

    # Save model
    save_path = 'new_joint_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"\nTotal model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
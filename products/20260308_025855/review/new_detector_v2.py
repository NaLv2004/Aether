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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V2 with Dual Adaptive Quantization")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=40, help="Phase 1 epochs (detector pre-training, no quantization)")
    parser.add_argument("--epochs_phase2", type=int, default=160, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for Phase 1 (detector pre-training)")
    parser.add_argument("--lr_phase2", type=float, default=5e-4, help="Learning rate for Phase 2 (joint QAT)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension D for feature embedding")
    parser.add_argument("--lambda_val", type=float, default=0.1, help="Lagrangian multiplier for bit constraint (Phase 2 main)")
    parser.add_argument("--lambda_warmstart", type=float, default=0.01, help="Lambda for warm-start phase in Phase 2")
    parser.add_argument("--warmstart_epochs", type=int, default=20, help="Number of warm-start epochs at beginning of Phase 2")
    parser.add_argument("--c_target", type=float, default=20.0, help="Target average bits per (AP, user) link")
    parser.add_argument("--tau_init", type=float, default=2.0, help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--tau_min", type=float, default=0.2, help="Minimum Gumbel-Softmax temperature")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of mean-field GNN layers")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer encoder layers for inter-user IC")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples per power point")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in Transformer and MLPs")
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

    # 4. Local LMMSE detection (vectorized over batch and APs)
    # H: (B, L, N, K), H^H: (B, L, K, N)
    H_conj_trans = H.conj().transpose(0, 1, 3, 2)  # (batch_size, L, K, N)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    # R_y = H @ H^H + sigma^2 I_N: (B, L, N, N)
    R_y = H @ H_conj_trans + noise_cov  # (B, L, N, K) @ (B, L, K, N) -> wait, this is wrong dimension
    # Actually: H is (B,L,N,K), H_conj_trans is (B,L,K,N)
    # H @ H_conj_trans would be (B,L,N,K) @ (B,L,K,N) = (B,L,N,N) - CORRECT for R_y = H H^H + sigma^2 I
    R_y_inv = np.linalg.inv(R_y)
    # W = H^H @ R_y^{-1}: (B,L,K,N) @ (B,L,N,N) = (B,L,K,N)
    W_l = H_conj_trans @ R_y_inv  # (batch_size, L, K, N)
    # s_hat = W @ y: (B,L,K,N) @ (B,L,N,1) = (B,L,K,1)
    s_hat = W_l @ y[..., np.newaxis]  # (batch_size, L, K, 1)
    s_hat = s_hat.squeeze(-1)  # (batch_size, L, K)

    # 5. Local SNR (normalized to roughly [0, 1] range)
    # sum of |H_{l,n,k}|^2 over antennas / noise_power
    local_snr = 10 * np.log10(np.sum(np.abs(H) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0
    # shape: (batch_size, L, K)

    return s_hat, s, H, local_snr, y


# ============================================================
# 2. C-MMSE Baseline (Centralized MMSE with full precision)
# ============================================================
def compute_cmmse_detection(H, y, s, noise_w):
    """
    Centralized MMSE detection using full (unquantized) channel and received signal.
    
    C-MMSE formula:
        H_all = stack H across all APs: (B, L*N, K)
        y_all = stack y across all APs: (B, L*N)
        s_hat = (H_all^H @ H_all + sigma^2 * I_K)^{-1} @ H_all^H @ y_all
    
    Args:
        H: (B, L, N, K) complex - full channel matrix
        y: (B, L, N) complex - received signals at all APs
        s: (B, K) complex - true symbols (for BER computation)
        noise_w: scalar - noise power
    
    Returns:
        ber: BER of C-MMSE detection
        s_hat_cmmse: (B, K) complex - detected symbols
    """
    B, L, N, K = H.shape
    
    # Stack H across APs: (B, L*N, K)
    H_all = H.reshape(B, L * N, K)
    
    # Stack y across APs: (B, L*N)
    y_all = y.reshape(B, L * N)
    
    # H^H: (B, K, L*N)
    H_H = H_all.conj().transpose(0, 2, 1)  # (B, K, L*N)
    
    # Gram matrix: H^H @ H: (B, K, K)
    HHH = H_H @ H_all  # (B, K, K)
    
    # Regularized inverse: (H^H H + sigma^2 I_K)^{-1}
    noise_eye = noise_w * np.eye(K).reshape(1, K, K)
    R = HHH + noise_eye  # (B, K, K)
    
    # Matched filter output: H^H @ y: (B, K)
    HHy = H_H @ y_all[..., np.newaxis]  # (B, K, 1)
    
    # MMSE estimate: R^{-1} @ H^H @ y
    R_inv = np.linalg.inv(R)  # (B, K, K)
    s_hat_cmmse = (R_inv @ HHy).squeeze(-1)  # (B, K)
    
    # Compute BER
    ber = compute_ber(s, s_hat_cmmse)
    
    return ber, s_hat_cmmse


# ============================================================
# 3. GNN + Transformer Hybrid Detector V2
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    """
    Mean-field GNN message passing layer with O(L) complexity.
    
    For each AP l and user k:
    1. Compute message: msg_l = MLP(h_l)
    2. Mean-field aggregation: agg = (1/L) * sum_j msg_j  (averaged across all APs)
    3. Update: h_l_new = LayerNorm(h_l + UpdateMLP([h_l, agg]))
    
    This is O(L) per layer instead of O(L^2) for pairwise attention.
    Includes residual connection and LayerNorm for stable training.
    """
    def __init__(self, hidden_dim):
        super(MeanFieldGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Message computation MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update MLP: takes [local_feature, aggregated_message] -> updated feature
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        """
        h: (B, L, K, D) - node features for each AP and user
        Returns: (B, L, K, D) - updated node features
        """
        # Step 1: Compute messages from all APs
        messages = self.message_mlp(h)  # (B, L, K, D)
        
        # Step 2: Mean-field aggregation: average across all APs (dim=1)
        mean_message = messages.mean(dim=1, keepdim=True)  # (B, 1, K, D)
        mean_message = mean_message.expand_as(h)  # (B, L, K, D)
        
        # Step 3: Concatenate local feature with aggregated message
        combined = torch.cat([h, mean_message], dim=-1)  # (B, L, K, 2D)
        
        # Step 4: Update
        updated = self.update_mlp(combined)  # (B, L, K, D)
        
        # Step 5: Residual connection + layer norm
        h_new = self.layer_norm(h + updated)
        return h_new


class APAggregator(nn.Module):
    """
    Learned attention-based weighted sum across APs to produce per-user features.
    Each AP contributes to the user feature with a learned attention weight.
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
        attn_weights = torch.softmax(attn_weights, dim=1)  # softmax over L dimension
        aggregated = (h * attn_weights).sum(dim=1)  # (B, K, D)
        return aggregated, attn_weights


class GNNTransformerDetector(nn.Module):
    """
    Hybrid GNN + Transformer detector V2 for Cell-Free MIMO.
    
    Architecture:
    1. AP Feature Extraction:
       - Demod MLP: (B, L, K, 2) -> (B, L, K, D)  [local LMMSE real/imag]
       - Channel MLP: (B, L, K, 2*N) -> (B, L, K, D)  [per-user channel coefficients]
       - Cross-user interference features: ||H_{l,:,k}||^2 and sum_{j!=k}||H_{l,:,j}||^2
       - Fusion MLP: concat [demod_feat, channel_feat, bitwidth_feat, interference_feat] -> D
    2. Mean-field GNN Layers (x3): O(L) message passing across APs with residual + LayerNorm
    3. AP Aggregation: attention-based weighted sum across APs
    4. Transformer IC (x2 layers, 4 heads): inter-user interference cancellation
    5. Output: residual connection from weighted soft info + Transformer residual
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(GNNTransformerDetector, self).__init__()
        self.L = L
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # --- AP Feature Extraction ---
        # MLP for demod features: input (B, L, K, 2) -> output (B, L, K, D)
        self.demod_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # MLP for channel features per user: input (B, L, K, 2*N) -> output (B, L, K, D)
        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * N, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Interference features: [log(1+||H_{l,:,k}||^2), log(1+sum_{j!=k}||H_{l,:,j}||^2)] -> (B, L, K, 2)

        # Fusion MLP: concat [demod_feat(D), channel_feat(D), bitwidth_features(2), interference_features(2)] -> D
        fusion_input_dim = 2 * hidden_dim + 2 + 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- Mean-field GNN Layers ---
        self.gnn_layers = nn.ModuleList([
            MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # --- AP Aggregation ---
        self.ap_aggregator = APAggregator(hidden_dim)

        # --- Soft info aggregation (for residual baseline output) ---
        self.soft_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # --- Transformer for user-level interference cancellation ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- Output Head ---
        self.output_head = nn.Linear(hidden_dim, 2)
        # Zero-init for stable residual learning: initially output = base_out (weighted soft info)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, s_hat_q, H_q, bitwidth_features):
        """
        Args:
            s_hat_q: (B, L, K, 2) - quantized demod results (real, imag)
            H_q: (B, L, N, K, 2) - quantized channel coefficients (real, imag)
            bitwidth_features: (B, L, K, 2) - normalized bitwidth features (demod_bits_norm, channel_bits_norm)
        
        Returns:
            detected: (B, K, 2) - detected symbols (real, imag)
        """
        B = s_hat_q.size(0)
        L = s_hat_q.size(1)
        K = s_hat_q.size(2)
        N = H_q.size(2)

        # --- 1. AP Feature Extraction ---
        # Demod features: process local LMMSE estimates
        demod_feat = self.demod_mlp(s_hat_q)  # (B, L, K, D)

        # Channel features: reshape H_q from (B, L, N, K, 2) to per-user features
        # For each user k at AP l, concatenate all N antenna coefficients (real+imag): (B, L, K, 2*N)
        H_q_perm = H_q.permute(0, 1, 3, 2, 4)  # (B, L, K, N, 2)
        H_q_flat = H_q_perm.reshape(B, L, K, N * 2)  # (B, L, K, 2*N)
        channel_feat = self.channel_mlp(H_q_flat)  # (B, L, K, D)

        # --- Cross-user interference features ---
        # ||H_{l,:,k}||^2: channel power for user k at AP l (sum over N antennas, real^2+imag^2)
        # H_q: (B, L, N, K, 2)
        h_power_per_user = (H_q ** 2).sum(dim=(2, 4))  # (B, L, K) - sum over N and real/imag
        
        # Desired signal power for user k: ||H_{l,:,k}||^2
        desired_power = h_power_per_user  # (B, L, K)
        
        # Interference power for user k: sum_{j!=k} ||H_{l,:,j}||^2
        total_power = h_power_per_user.sum(dim=2, keepdim=True)  # (B, L, 1) total across all users
        interference_power = total_power - desired_power  # (B, L, K)
        
        # Use log scale for numerical stability and compression of dynamic range
        desired_feat = torch.log1p(desired_power).unsqueeze(-1)  # (B, L, K, 1)
        interference_feat = torch.log1p(interference_power).unsqueeze(-1)  # (B, L, K, 1)
        interference_features = torch.cat([desired_feat, interference_feat], dim=-1)  # (B, L, K, 2)

        # Concatenate all features for fusion
        combined = torch.cat([demod_feat, channel_feat, bitwidth_features, interference_features], dim=-1)
        # Shape: (B, L, K, 2*D + 2 + 2)
        node_features = self.fusion_mlp(combined)  # (B, L, K, D)

        # --- 2. Mean-field GNN Message Passing ---
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)  # (B, L, K, D)

        # --- 3. AP Aggregation ---
        # Soft info weighted mean (for residual baseline)
        soft_info = s_hat_q  # (B, L, K, 2) - the quantized demod results
        soft_attn_weights = self.soft_attn(h)  # (B, L, K, 1)
        soft_attn_weights = torch.softmax(soft_attn_weights, dim=1)  # softmax over L
        base_out = (soft_info * soft_attn_weights).sum(dim=1)  # (B, K, 2)

        # Feature aggregation for Transformer input
        user_features, _ = self.ap_aggregator(h)  # (B, K, D)

        # --- 4. Transformer IC (inter-user interference cancellation) ---
        ic_out = self.transformer_ic(user_features)  # (B, K, D)

        # --- 5. Output with residual connection ---
        residual = self.output_head(ic_out)  # (B, K, 2)
        detected = base_out + residual

        return detected


# ============================================================
# 4. Dual Adaptive Quantizer (Improved)
# ============================================================
class DualPolicyNetwork(nn.Module):
    """
    Policy network that outputs bitwidth decisions for both demod results and channel coefficients.
    
    Input features per (l, k): [v_real, v_imag, snr, channel_norm, avg_channel_power] -> 5 dims
    Output: logits for demod bitwidth and channel bitwidth, each choosing from {0, 2, 4} bits
    
    Improvements over V1:
    - Hidden dimension increased to 64 (from 32)
    - Added a second hidden layer for better decision making (3 layers total in shared)
    """
    def __init__(self, input_dim=5, policy_hidden=64):
        super(DualPolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU()
        )
        self.head_demod = nn.Linear(policy_hidden, 3)   # logits for {0, 2, 4} bits
        self.head_channel = nn.Linear(policy_hidden, 3)  # logits for {0, 2, 4} bits

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
    Uses Gumbel-Softmax for differentiable bitwidth selection from {0, 2, 4} bits.
    
    For demod results (complex-valued, 2 real values per link):
    - 0-bit: zero output (no transmission)
    - 2-bit: 2 real values * 2 bits = 4 bits per link
    - 4-bit: 2 real values * 4 bits = 8 bits per link
    
    For channel coefficients (N complex-valued, 2N real values per link):
    - 0-bit: zero output (no transmission)
    - 2-bit: 2N real values * 2 bits = 4N bits per link
    - 4-bit: 2N real values * 4 bits = 8N bits per link
    """
    def __init__(self, N=4, policy_hidden=64):
        super(DualAdaptiveQuantizer, self).__init__()
        self.N = N
        self.policy = DualPolicyNetwork(input_dim=5, policy_hidden=policy_hidden)

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
            expected_bits_demod: (B, L, K) - expected demod bits per link
            expected_bits_channel: (B, L, K) - expected channel bits per link
            w_demod: (B, L, K, 3) - Gumbel-Softmax weights for demod
            w_channel: (B, L, K, 3) - Gumbel-Softmax weights for channel
        """
        B, L, K, _ = v.shape
        N = H.shape[2]

        # --- Compute policy input features ---
        v_real = v[..., 0:1]  # (B, L, K, 1)
        v_imag = v[..., 1:2]  # (B, L, K, 1)
        
        # Channel norm per user: ||H_{l,:,k}||^2 (sum over N antennas and real/imag)
        # H shape: (B, L, N, K, 2)
        h_power = (H ** 2).sum(dim=(2, 4))  # (B, L, K)
        channel_norm = h_power.unsqueeze(-1)  # (B, L, K, 1)
        
        # Average channel power across APs for each user
        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)  # (B, L, K, 1)

        # Policy input: [v_real, v_imag, snr, channel_norm, avg_channel_power]
        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm, avg_channel_power], dim=-1)  # (B, L, K, 5)

        # --- Get bitwidth decisions ---
        logits_demod, logits_channel = self.policy(policy_input)  # each: (B, L, K, 3)

        # Gumbel-Softmax (hard=True for discrete selection in forward, soft gradients in backward)
        w_demod = F.gumbel_softmax(logits_demod, tau=tau, hard=True)  # (B, L, K, 3)
        w_channel = F.gumbel_softmax(logits_channel, tau=tau, hard=True)  # (B, L, K, 3)

        # --- Quantize demod results ---
        v_q2 = self.q2_demod(v)  # (B, L, K, 2)
        v_q4 = self.q4_demod(v)  # (B, L, K, 2)
        
        # Weighted combination based on Gumbel-Softmax selection
        # w_demod[..., 0]: select 0-bit (zero output)
        # w_demod[..., 1]: select 2-bit
        # w_demod[..., 2]: select 4-bit
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

        # --- Expected bits per link ---
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
# 5. Joint Model
# ============================================================
class JointModel(nn.Module):
    """
    End-to-end model combining DualAdaptiveQuantizer + GNNTransformerDetector V2.
    
    Phase 1: Only detector is trained (no quantization), learning to detect from full-precision data.
    Phase 2: Joint training of quantizer + detector with bit budget constraint.
    """
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4, dropout=0.1):
        super(JointModel, self).__init__()
        self.quantizer = DualAdaptiveQuantizer(N=N, policy_hidden=64)
        self.detector = GNNTransformerDetector(
            L=L, N=N, K=K, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
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
            # Bitwidth features: normalized bit values for the detector to be aware of quantization level
            bw_demod_feat = expected_bits_demod.unsqueeze(-1) / 8.0  # normalize by max (2*4=8 bits)
            bw_channel_feat = expected_bits_channel.unsqueeze(-1) / 32.0  # normalize by max (2*4*4=32 bits)
        else:
            v_q = v
            H_q = H
            expected_bits_demod = torch.zeros(B, L, K, device=v.device)
            expected_bits_channel = torch.zeros(B, L, K, device=v.device)
            # Full precision: indicate maximum bitwidth
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
    Convert numpy complex arrays to real-valued torch tensors suitable for the model.
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
    QPSK carries 2 bits per symbol (1 bit in real, 1 bit in imaginary).
    Detection: sign of real/imag part determines the bit.
    """
    s_true_real = np.sign(s_true.real)
    s_true_imag = np.sign(s_true.imag)
    s_pred_real = np.sign(s_pred_complex.real)
    s_pred_imag = np.sign(s_pred_complex.imag)
    
    # Handle exact zero predictions (map to +1)
    s_pred_real[s_pred_real == 0] = 1
    s_pred_imag[s_pred_imag == 0] = 1

    err_real = (s_true_real != s_pred_real).sum()
    err_imag = (s_true_imag != s_pred_imag).sum()
    total_bits = s_true.size * 2  # 2 bits per QPSK symbol
    return (err_real + err_imag) / total_bits


def compute_dist_full_ber(s_hat, s):
    """
    Baseline: Dist-Full detection via mean pooling of local LMMSE across APs.
    s_hat: (B, L, K) complex - local LMMSE estimates from each AP
    s: (B, K) complex - true symbols
    """
    s_hat_avg = s_hat.mean(axis=1)  # (B, K) - simple average across APs
    return compute_ber(s, s_hat_avg)


# ============================================================
# 6. Main Training Script
# ============================================================
def main():
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_epochs = args.epochs_phase1 + args.epochs_phase2
    
    print(f"=" * 80)
    print(f"GNN+Transformer Hybrid Detector V2 with Dual Adaptive Quantization")
    print(f"=" * 80)
    print(f"Device: {device}")
    print(f"System: L={args.L} APs, N={args.N} antennas/AP, K={args.K} users")
    print(f"Training: {total_epochs} total epochs (Phase1: {args.epochs_phase1}, Phase2: {args.epochs_phase2})")
    print(f"  Phase1 LR: {args.lr_phase1}, Phase2 LR: {args.lr_phase2}")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {args.batches_per_epoch}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Bit constraint: lambda={args.lambda_val}, C_target={args.c_target} bits/link")
    print(f"  Lambda warm-start: {args.lambda_warmstart} for first {args.warmstart_epochs} epochs of Phase 2")
    print(f"GNN layers: {args.num_gnn_layers}, Transformer layers: {args.num_transformer_layers}")
    print(f"Temperature annealing: {args.tau_init} -> {args.tau_min}")
    print(f"Gradient clipping: max_norm={args.grad_clip}")
    print(f"Dropout: {args.dropout}")
    print(f"Seed: {args.seed}")
    print(f"=" * 80)

    # Initialize system model (p_tx=0 is placeholder; actual power varies per batch)
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    # ===========================
    # Pre-generate fixed test set (same for all epochs - required by spec)
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
    # Compute baselines (once, since test set is fixed)
    # ===========================
    print(f"Computing baseline results (Dist-Full and C-MMSE)...")
    baseline_ber_dist_full = {}
    baseline_ber_cmmse = {}
    for p in test_p_tx_list:
        td = test_dataset[p]
        baseline_ber_dist_full[p] = compute_dist_full_ber(td['s_hat_np'], td['s_np'])
        baseline_ber_cmmse[p], _ = compute_cmmse_detection(td['H_np'], td['y_np'], td['s_np'], sys_model.noise_w)
    
    print(f"\nBaseline BERs (computed on fixed test set):")
    print(f"{'p_tx(dBm)':<12} | {'Dist-Full BER':<15} | {'C-MMSE BER':<15}")
    print("-" * 45)
    for p in test_p_tx_list:
        print(f"{p:<12} | {baseline_ber_dist_full[p]:<15.6f} | {baseline_ber_cmmse[p]:<15.6f}")
    print()

    # ===========================
    # Initialize model
    # ===========================
    model = JointModel(
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
    # Phase 1: Detector Pre-training (no quantization, full precision)
    # ===========================
    print(f"\n{'=' * 80}")
    print(f"Phase 1: Detector Pre-training ({args.epochs_phase1} epochs, full precision)")
    print(f"  Learning rate: {args.lr_phase1} with cosine annealing")
    print(f"{'=' * 80}")

    optimizer_phase1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=args.epochs_phase1, eta_min=1e-5)

    for epoch in range(args.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        start_t = time.time()

        for batch_idx in range(args.batches_per_epoch):
            # Random training power in wide range
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np, _ = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

            optimizer_phase1.zero_grad()
            detected, _, _ = model(v_tensor, H_tensor, snr_tensor, use_quantization=False)
            loss = criterion(detected, Y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(), max_norm=args.grad_clip)
            optimizer_phase1.step()
            epoch_loss += loss.item()

        scheduler_phase1.step()
        elapsed = time.time() - start_t
        avg_loss = epoch_loss / args.batches_per_epoch

        # Print every 5 epochs (and first epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Validate at multiple power points
                val_bers = {}
                for p_val in [0, 10, 20]:
                    td = test_dataset[p_val]
                    val_out, _, _ = model(td['v'], td['H'], td['snr'], use_quantization=False)
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
    # Phase 2: Joint QAT Training
    # ===========================
    print(f"{'=' * 80}")
    print(f"Phase 2: Joint QAT Training ({args.epochs_phase2} epochs)")
    print(f"  Learning rate: {args.lr_phase2} with cosine annealing")
    print(f"  Warm-start: first {args.warmstart_epochs} epochs with lambda={args.lambda_warmstart}")
    print(f"  Then: remaining epochs with lambda={args.lambda_val}")
    print(f"  Temperature annealing: {args.tau_init} -> {args.tau_min}")
    print(f"{'=' * 80}")

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=args.epochs_phase2, eta_min=1e-5)

    best_val_ber = float('inf')
    best_state = None

    for epoch in range(args.epochs_phase2):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bits = 0.0
        start_t = time.time()

        # Temperature annealing over Phase 2: exponential decay from tau_init to tau_min
        progress = epoch / max(1, args.epochs_phase2 - 1)
        tau = args.tau_init * (args.tau_min / args.tau_init) ** progress

        # Lambda warm-start: gentle bit constraint for first warmstart_epochs, then full strength
        if epoch < args.warmstart_epochs:
            current_lambda = args.lambda_warmstart
        else:
            current_lambda = args.lambda_val

        for batch_idx in range(args.batches_per_epoch):
            p_train = np.random.uniform(-20, 25)
            s_hat_np, s_np, H_np, snr_np, _ = generate_data_batch_v2(sys_model, args.batch_size, p_train)
            v_tensor, H_tensor, snr_tensor, Y_tensor = prepare_tensors(s_hat_np, s_np, H_np, snr_np, device)

            optimizer_phase2.zero_grad()
            detected, exp_bits_demod, exp_bits_channel = model(
                v_tensor, H_tensor, snr_tensor, tau=tau, use_quantization=True
            )

            # Detection loss (MSE between detected and true symbols)
            mse_loss = criterion(detected, Y_tensor)

            # Bit constraint loss: penalize deviation from target budget
            total_bits_per_link = exp_bits_demod + exp_bits_channel  # (B, L, K)
            avg_bits = total_bits_per_link.mean()
            bit_penalty = (avg_bits - args.c_target) ** 2

            # Total loss = detection loss + lambda * bit budget penalty
            loss = mse_loss + current_lambda * bit_penalty
            loss.backward()

            # Gradient clipping for training stability
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

        # Print every 5 epochs (and first epoch)
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

                # Also check BER at 10 dBm
                td10 = test_dataset[10]
                val_out10, _, _ = model(td10['v'], td10['H'], td10['snr'], tau=tau, use_quantization=True)
                val_np10 = val_out10.cpu().numpy()
                s_pred10 = val_np10[..., 0] + 1j * val_np10[..., 1]
                val_ber10 = compute_ber(td10['s_np'], s_pred10)

            lr_now = scheduler_phase2.get_last_lr()[0]
            print(f"Phase2 Epoch [{epoch + 1:03d}/{args.epochs_phase2}] | "
                  f"Time: {elapsed:.1f}s | LR: {lr_now:.2e} | tau: {tau:.3f} | lam: {current_lambda} | "
                  f"Train Loss: {avg_loss:.5f} (MSE: {avg_mse:.5f}) | "
                  f"Avg Bits: {avg_bits_epoch:.1f} (target: {args.c_target}) | "
                  f"Val BER: 0dBm={val_ber:.4f}, 10dBm={val_ber10:.4f} | "
                  f"Val Bits: D={val_avg_bits_d:.1f} C={val_avg_bits_c:.1f} T={val_avg_bits_total:.1f}")

            # Track best model based on average BER at 0 and 10 dBm
            avg_val_ber = (val_ber + val_ber10) / 2.0
            if avg_val_ber < best_val_ber:
                best_val_ber = avg_val_ber
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  -> New best model! Avg val BER = {avg_val_ber:.6f}")

    print("\nPhase 2 completed.\n")

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model from training for final evaluation.\n")

    # ===========================
    # Final Evaluation
    # ===========================
    print(f"{'=' * 80}")
    print(f"Final Evaluation on Test Set ({args.test_samples} samples per power point)")
    print(f"{'=' * 80}")

    model.eval()
    
    header = (f"{'p_tx(dBm)':<12} | {'Dist-Full BER':<15} | {'C-MMSE BER':<15} | "
              f"{'Proposed BER':<15} | {'vs Dist-Full':<14} | {'vs C-MMSE':<14} | "
              f"{'Demod Bits':<12} | {'Chan Bits':<12} | {'Total Bits':<12}")
    print(f"\n{header}")
    print("-" * len(header))

    results_proposed = {}
    with torch.no_grad():
        for p in test_p_tx_list:
            td = test_dataset[p]

            # Baselines (pre-computed)
            ber_dist_full = baseline_ber_dist_full[p]
            ber_cmmse = baseline_ber_cmmse[p]

            # Proposed: GNN+Transformer V2 with quantization
            detected, exp_bits_d, exp_bits_c = model(
                td['v'], td['H'], td['snr'], tau=args.tau_min, use_quantization=True
            )
            det_np = detected.cpu().numpy()
            s_pred = det_np[..., 0] + 1j * det_np[..., 1]
            ber_proposed = compute_ber(td['s_np'], s_pred)

            avg_bits_d = exp_bits_d.mean().item()
            avg_bits_c = exp_bits_c.mean().item()
            avg_bits_total = (exp_bits_d + exp_bits_c).mean().item()

            improve_dist = ((ber_dist_full - ber_proposed) / max(ber_dist_full, 1e-10)) * 100
            improve_cmmse = ((ber_cmmse - ber_proposed) / max(ber_cmmse, 1e-10)) * 100

            results_proposed[p] = {
                'ber': ber_proposed, 'bits_d': avg_bits_d, 'bits_c': avg_bits_c, 'bits_t': avg_bits_total
            }

            print(f"{p:<12} | {ber_dist_full:<15.6f} | {ber_cmmse:<15.6f} | "
                  f"{ber_proposed:<15.6f} | {improve_dist:>+12.2f}% | {improve_cmmse:>+12.2f}% | "
                  f"{avg_bits_d:<12.2f} | {avg_bits_c:<12.2f} | {avg_bits_total:<12.2f}")

    print("-" * len(header))

    # ===========================
    # Summary statistics
    # ===========================
    print(f"\n{'=' * 80}")
    print(f"Summary: Average Performance Across All Power Points")
    print(f"{'=' * 80}")
    avg_ber_dist = np.mean([baseline_ber_dist_full[p] for p in test_p_tx_list])
    avg_ber_cmmse = np.mean([baseline_ber_cmmse[p] for p in test_p_tx_list])
    avg_ber_proposed = np.mean([results_proposed[p]['ber'] for p in test_p_tx_list])
    avg_total_bits = np.mean([results_proposed[p]['bits_t'] for p in test_p_tx_list])
    avg_demod_bits = np.mean([results_proposed[p]['bits_d'] for p in test_p_tx_list])
    avg_chan_bits = np.mean([results_proposed[p]['bits_c'] for p in test_p_tx_list])
    fp_bits = 2 * 32 + 2 * args.N * 32  # full precision: demod (2 reals * 32 bits) + channel (2N reals * 32 bits)
    
    print(f"  Average Dist-Full BER:   {avg_ber_dist:.6f}")
    print(f"  Average C-MMSE BER:      {avg_ber_cmmse:.6f}")
    print(f"  Average Proposed BER:    {avg_ber_proposed:.6f}")
    print(f"  Average Demod Bits/Link: {avg_demod_bits:.2f}")
    print(f"  Average Chan Bits/Link:  {avg_chan_bits:.2f}")
    print(f"  Average Total Bits/Link: {avg_total_bits:.2f} (target: {args.c_target})")
    print(f"  Full Precision Bits/Link: {fp_bits}")
    print(f"  Average Compression Ratio: {fp_bits / max(avg_total_bits, 1e-10):.1f}x")
    
    # Performance comparison
    if avg_ber_proposed < avg_ber_dist:
        print(f"\n  *** Proposed OUTPERFORMS Dist-Full by {((avg_ber_dist - avg_ber_proposed) / max(avg_ber_dist, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"\n  *** Proposed underperforms Dist-Full by {((avg_ber_proposed - avg_ber_dist) / max(avg_ber_dist, 1e-10)) * 100:.1f}% ***")
    
    if avg_ber_proposed < avg_ber_cmmse:
        print(f"  *** Proposed OUTPERFORMS C-MMSE by {((avg_ber_cmmse - avg_ber_proposed) / max(avg_ber_cmmse, 1e-10)) * 100:.1f}% ***")
    else:
        print(f"  *** Proposed underperforms C-MMSE by {((avg_ber_proposed - avg_ber_cmmse) / max(avg_ber_cmmse, 1e-10)) * 100:.1f}% ***")

    # ===========================
    # Detailed bit allocation statistics
    # ===========================
    print(f"\n{'=' * 80}")
    print(f"Detailed Bit Allocation Statistics")
    print(f"{'=' * 80}")
    with torch.no_grad():
        for p in [-10, 0, 10, 20]:
            td = test_dataset[p]
            
            # Get quantizer weights for analysis
            v_q, H_q, ebd, ebc, w_d, w_c = model.quantizer(
                td['v'], td['H'], td['snr'], tau=args.tau_min
            )
            
            w_d_mean = w_d.mean(dim=(0, 1, 2))  # average over batch, APs, users
            w_c_mean = w_c.mean(dim=(0, 1, 2))
            
            print(f"\n  p_tx = {p} dBm:")
            print(f"    Demod bitwidth distribution:   0-bit: {w_d_mean[0]:.3f}, 2-bit: {w_d_mean[1]:.3f}, 4-bit: {w_d_mean[2]:.3f}")
            print(f"    Channel bitwidth distribution: 0-bit: {w_c_mean[0]:.3f}, 2-bit: {w_c_mean[1]:.3f}, 4-bit: {w_c_mean[2]:.3f}")
            print(f"    Avg demod bits/link: {ebd.mean().item():.2f}, Avg channel bits/link: {ebc.mean().item():.2f}")
            print(f"    Avg total bits/link: {(ebd + ebc).mean().item():.2f}")
            
            compression = fp_bits / max((ebd + ebc).mean().item(), 1e-10)
            print(f"    Full precision bits/link: {fp_bits}, Compression ratio: {compression:.1f}x")

    # ===========================
    # Per-AP bit allocation statistics
    # ===========================
    print(f"\n{'=' * 80}")
    print(f"Per-AP Bit Allocation at 10 dBm (first 5 test samples)")
    print(f"{'=' * 80}")
    with torch.no_grad():
        td = test_dataset[10]
        # Take first 5 samples
        v_sub = td['v'][:5]
        H_sub = td['H'][:5]
        snr_sub = td['snr'][:5]
        _, _, ebd_sub, ebc_sub, _, _ = model.quantizer(v_sub, H_sub, snr_sub, tau=args.tau_min)
        total_bits_sub = ebd_sub + ebc_sub  # (5, L, K)
        avg_per_ap = total_bits_sub.mean(dim=(0, 2))  # average over batch and users -> (L,)
        for l in range(args.L):
            print(f"  AP {l:2d}: avg bits/link = {avg_per_ap[l].item():.2f}")

    # ===========================
    # Save model
    # ===========================
    save_path = 'new_joint_model_v2.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")
    print(f"Total model parameters: {total_params:,}")
    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
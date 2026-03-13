import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import os
from system_model import CellFreeSystem
from lsq_quantizer import LSQQuantizer

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Script for V13 Model")
    parser.add_argument("--model_path", type=str, default="new_joint_model_v13.pth")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--area_size", type=float, default=1.0)
    parser.add_argument("--bandwidth", type=float, default=20e6)
    parser.add_argument("--noise_psd", type=float, default=-174)
    parser.add_argument("--test_samples", type=int, default=1000, help="Samples for evaluation smoothing")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ============================================================
# Baselines
# ============================================================
def compute_ber(s_true, s_pred_complex):
    """
    Compute Bit Error Rate for QPSK.
    s_true: Complex numpy array or labels.
    s_pred_complex: Complex numpy array.
    """
    if np.iscomplexobj(s_true):
        s_true_real, s_true_imag = np.sign(s_true.real), np.sign(s_true.imag)
    else:
        # Assume labels 0-3
        mapping = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
        s_true_c = np.vectorize(mapping.get)(s_true)
        s_true_real, s_true_imag = np.sign(s_true_c.real), np.sign(s_true_c.imag)
        
    s_pred_real, s_pred_imag = np.sign(s_pred_complex.real), np.sign(s_pred_complex.imag)
    s_pred_real[s_pred_real == 0] = 1; s_pred_imag[s_pred_imag == 0] = 1
    
    err = (s_true_real != s_pred_real).sum() + (s_true_imag != s_pred_imag).sum()
    return err / (s_pred_complex.size * 2)

def compute_dist_full_ber(s_hat):
    """
    Maps soft estimates to QPSK and handles AP dimension.
    Input s_hat: (B, L, K) or (B, K)
    Output: (B, K) QPSK symbols
    """
    if s_hat.ndim == 3:
        # Perform distributed averaging (Mean Pooling) across APs
        s_hat = s_hat.mean(axis=1) # Shape (B, K)
        
    qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    # Distance to each QPSK constellation point
    dist = np.abs(s_hat[..., np.newaxis] - qpsk)
    idx = np.argmin(dist, axis=-1)
    return qpsk[idx]

def compute_cmmse_ber(y, H, sys_model):
    """
    Centralized MMSE Detection (Perfect CSI & No Quantization)
    """
    B, L, N, K = H.shape
    H_flat = H.reshape(B, L*N, K)
    y_flat = y.reshape(B, L*N, 1)
    H_H = H_flat.conj().transpose(0, 2, 1)
    # Using matrix inversion lemma for efficiency (K x K instead of LN x LN)
    W = np.linalg.inv(H_H @ H_flat + sys_model.noise_w * np.eye(K))
    s_hat = (W @ H_H @ y_flat).squeeze(-1)
    return compute_dist_full_ber(s_hat)

def compute_cmmse_mixed_q_detection(y, H, sys_model, bits=8):
    """
    Centralized MMSE with Quantized Input (Baseline for high-resolution link)
    """
    B, L, N, K = H.shape
    def q_array(x, b):
        if b >= 16: return x
        scale = np.max(np.abs(x)) + 1e-10
        q_levels = 2**b
        xr = np.round(((x.real/scale)+1)/2 * (q_levels-1)) / (q_levels-1) * 2 - 1
        xi = np.round(((x.imag/scale)+1)/2 * (q_levels-1)) / (q_levels-1) * 2 - 1
        return (xr + 1j * xi) * scale
        
    y_q = q_array(y, bits)
    H_q = q_array(H, bits)
    H_flat = H_q.reshape(B, L*N, K)
    y_flat = y_q.reshape(B, L*N, 1)
    H_H = H_flat.conj().transpose(0, 2, 1)
    # K x K inversion
    s_hat = (np.linalg.inv(H_H @ H_flat + sys_model.noise_w * np.eye(K)) @ H_H @ y_flat).squeeze(-1)
    return compute_dist_full_ber(s_hat)

def compute_dist_q_ber(s_hat, bits):
    """
    Distributed Detection (Local LMMSE) with Quantized Transmission
    """
    if bits <= 0: return np.zeros((s_hat.shape[0], s_hat.shape[-1]), dtype=complex)
    q_levels = 2**bits
    scale = np.max(np.abs(s_hat)) + 1e-10
    def q(x):
        return np.round(((x/scale)+1)/2 * (q_levels-1)) / (q_levels-1) * 2 - 1
    s_q = (q(s_hat.real) + 1j * q(s_hat.imag)) * scale
    return compute_dist_full_ber(s_q)

# ============================================================
# Model Definitions (Eval Version)
# ============================================================
class TriplePolicyNetworkV13_Eval(nn.Module):
    def __init__(self, input_dim=9, policy_hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden), nn.ReLU()
        )
        self.head_y = nn.Linear(policy_hidden, 5)
        self.head_H = nn.Linear(policy_hidden, 5)
        self.head_demod = nn.Linear(policy_hidden, 5)

    def forward(self, x):
        h = self.shared(x)
        return self.head_y(h), self.head_H(h), self.head_demod(h)

class TripleAdaptiveQuantizerV13_Eval(nn.Module):
    def __init__(self, N, policy_hidden=64):
        super().__init__()
        self.N = N
        self.policy = TriplePolicyNetworkV13_Eval(input_dim=9, policy_hidden=policy_hidden)
        self.bit_options = [0, 4, 8, 12, 16]
        self.qs_y = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])
        self.qs_H = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])
        self.qs_d = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])

    def forward(self, v, H, y, local_snr, tau=1.0, bit_penalty_bias=0.0):
        B, L, K, _ = v.shape
        v_real, v_imag = v[..., 0:1], v[..., 1:2]
        h_power = (H ** 2).sum(dim=(2, 4))
        channel_norm = h_power.unsqueeze(-1)
        avg_channel_power = h_power.mean(dim=1, keepdim=True).expand(B, L, K).unsqueeze(-1)
        total_power = h_power.sum(dim=2, keepdim=True)
        sir = torch.log1p(h_power / (total_power - h_power + 1e-10)).unsqueeze(-1)
        total_power_norm = torch.log1p(total_power).expand(B, L, K).unsqueeze(-1)
        y_power = torch.log1p((y**2).sum(dim=(2, 3))).view(B, L, 1, 1).expand(B, L, K, 1)
        signal_power = torch.sqrt(v_real**2 + v_imag**2 + 1e-10)

        policy_input = torch.cat([v_real, v_imag, local_snr, channel_norm,
                                  avg_channel_power, sir, total_power_norm,
                                  y_power, signal_power], dim=-1)

        ly, lh, ld = self.policy(policy_input)
        
        penalty = bit_penalty_bias * torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=v.device)
        ly = ly - penalty; lh = lh - penalty; ld = ld - penalty

        w_y = F.gumbel_softmax(ly.mean(dim=2), tau=tau, hard=True)
        w_H = F.gumbel_softmax(lh, tau=tau, hard=True)
        w_d = F.gumbel_softmax(ld, tau=tau, hard=True)

        def apply_q(x, w, qs, dims_for_w):
            stacked = torch.stack([(q(x) if q else torch.zeros_like(x)) for q in qs], dim=-1)
            return (stacked * w.view(*dims_for_w, 5)).sum(dim=-1)

        y_q = apply_q(y, w_y, self.qs_y, [B, L, 1, 1])
        H_q = apply_q(H, w_H, self.qs_H, [B, L, 1, K, 1])
        v_q = apply_q(v, w_d, self.qs_d, [B, L, K, 1])

        bit_vals = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=v.device)
        b_y_link = (2 * self.N * (w_y * bit_vals).sum(-1, keepdim=True) / K).expand(B, L, K)
        b_H_link = 2 * self.N * (w_H * bit_vals).sum(-1)
        b_d_link = 2 * (w_d * bit_vals).sum(-1)
        
        return v_q, H_q, y_q, b_y_link + b_H_link + b_d_link, b_y_link, b_H_link, b_d_link

class MeanFieldGNNLayer_Eval(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, h):
        mean_msg = self.message_mlp(h).mean(dim=1, keepdim=True).expand_as(h)
        return self.layer_norm(h + self.update_mlp(torch.cat([h, mean_msg], dim=-1)))

class GNNTransformerDetectorV13_Eval(nn.Module):
    def __init__(self, L, N, K, hidden_dim=256, num_gnn_layers=3, num_transformer_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.demod_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.channel_mlp = nn.Sequential(nn.Linear(2 * N, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.mrc_norm = nn.LayerNorm(2)
        self.mrc_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim))
        fusion_dim = 2 * (hidden_dim // 4) + 3 + 2 + 1 
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        self.gnn_layers = nn.ModuleList([MeanFieldGNNLayer_Eval(hidden_dim) for _ in range(num_gnn_layers)])
        self.ap_agg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
                                               batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        self.output_head = nn.Linear(hidden_dim, 4)
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('constellation', torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32) / math.sqrt(2))

    def forward(self, s_hat_q, H_q, y_q, bw_feat, local_snr, ablation_no_mrc=False, ablation_no_bit_aware=False, ablation_no_attention=False):
        B, L, K = s_hat_q.shape[:3]; N = H_q.size(2)
        H_c = torch.complex(H_q[..., 0], H_q[..., 1]); y_c = torch.complex(y_q[..., 0], y_q[..., 1])
        s_mrc_c = torch.matmul(H_c.reshape(B, L*N, K).conj().transpose(-1, -2), y_c.reshape(B, L*N, 1)).squeeze(-1)
        s_mrc_feat = torch.stack([s_mrc_c.real, s_mrc_c.imag], dim=-1)
        mrc_embed = self.mrc_mlp(self.mrc_norm(s_mrc_feat))
        if ablation_no_mrc: mrc_embed = torch.zeros_like(mrc_embed)
        
        h_pow = (H_q ** 2).sum(dim=(2, 4))
        interf = torch.log1p(torch.stack([h_pow, h_pow.sum(dim=2, keepdim=True) - h_pow], dim=-1))
        if ablation_no_bit_aware: bw_feat = torch.zeros_like(bw_feat)
        
        node_feat = self.fusion_mlp(torch.cat([self.demod_mlp(s_hat_q), self.channel_mlp(H_q.permute(0, 1, 3, 2, 4).reshape(B, L, K, N*2)),
                                              bw_feat, interf, local_snr], dim=-1))
        h = self.fusion_bn(node_feat.reshape(-1, self.hidden_dim)).reshape(B, L, K, self.hidden_dim)
        for gnn in self.gnn_layers: h = gnn(h)
        
        if ablation_no_attention:
            user_feat = h.mean(dim=1) + mrc_embed
        else:
            attn = torch.softmax(self.ap_agg(h), dim=1)
            user_feat = (h * attn).sum(dim=1) + mrc_embed
            
        ic_out = self.transformer(user_feat)
        dist_sq = ((s_hat_q.mean(dim=1).unsqueeze(2) - self.constellation.unsqueeze(0).unsqueeze(0))**2).sum(dim=-1)
        return -dist_sq * self.base_scale + self.alpha * self.output_head(ic_out)

class JointModelV13_Eval(nn.Module):
    def __init__(self, L, N, K, hidden_dim=256):
        super().__init__()
        self.quantizer = TripleAdaptiveQuantizerV13_Eval(N)
        self.detector = GNNTransformerDetectorV13_Eval(L, N, K, hidden_dim)

    def forward(self, v, H, y, snr, tau=0.1, bit_penalty_bias=0.0, 
                ablation_no_mrc=False, ablation_no_bit_aware=False, ablation_no_attention=False):
        v_q, H_q, y_q, exp_bits, b_y, b_H, b_d = self.quantizer(v, H, y, snr, tau, bit_penalty_bias)
        bw_feat = torch.stack([b_y/(32*self.quantizer.N/8), b_H/(32*self.quantizer.N), b_d/32.0], dim=-1)
        logits = self.detector(v_q, H_q, y_q, bw_feat, snr, ablation_no_mrc, ablation_no_bit_aware, ablation_no_attention)
        return logits, exp_bits

# ============================================================
# Utilities
# ============================================================
def generate_eval_batch(sys_model, batch_size, p_tx_dbm, tau_est=0.0, K_override=None):
    K = K_override if K_override else sys_model.K
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta[:, :K] if K_override else p_tx_w * sys_model.beta
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]
    
    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, K)) / np.sqrt(2)
    H = np.sqrt(beta_w_expanded) * h_small
    
    if tau_est > 0:
        e_small = (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)) / np.sqrt(2)
        noise_est = np.sqrt(tau_est * beta_w_expanded) * e_small
        H_est = H + noise_est
    else:
        H_est = H.copy()
        
    labels = np.random.randint(0, 4, size=(batch_size, K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    s = np.vectorize(mapping.get)(labels) / np.sqrt(2)
    y = np.einsum('blnk,bk->bln', H, s) + (np.random.randn(batch_size, sys_model.L, sys_model.N) + 
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    
    # Local LMMSE for initial input to model
    H_H = H_est.conj().transpose(0, 1, 3, 2)
    R_y_inv = np.linalg.inv(H_est @ H_H + sys_model.noise_w * np.eye(sys_model.N))
    s_hat = (H_H @ R_y_inv @ y[..., np.newaxis]).squeeze(-1)
    
    local_snr = 10 * np.log10(np.sum(np.abs(H_est)**2, axis=2) / sys_model.noise_w + 1e-12) / 10.0
    return s_hat, s, H_est, local_snr, y, labels

def run_model_eval(model, s_hat, H_est, y, local_snr, labels, device, **kwargs):
    v_t = torch.tensor(np.stack([s_hat.real, s_hat.imag], -1), dtype=torch.float32).to(device)
    H_t = torch.tensor(np.stack([H_est.real, H_est.imag], -1), dtype=torch.float32).to(device)
    y_t = torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device)
    snr_t = torch.tensor(local_snr, dtype=torch.float32).unsqueeze(-1).to(device)
    with torch.no_grad():
        logits, eb = model(v_t, H_t, y_t, snr_t, **kwargs)
        pred = logits.argmax(-1).cpu().numpy()
        qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / math.sqrt(2)
        ber = compute_ber(labels, qpsk[pred])
    return ber, eb.mean().item()

# ============================================================
# Main Execution
# ============================================================
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    model = JointModelV13_Eval(args.L, args.N, args.K, args.hidden_dim).to(device)
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    p_tx_list = [-10, -5, 0, 5, 10, 15, 20]

    # --- Task 1 & 2 ---
    for task_name, tau_val in [("Task 1: Perfect CSI", 0.0), ("Task 2: Imperfect CSI", 0.05)]:
        print(f"\n{'='*20} {task_name} (tau={tau_val}) {'='*20}")
        print(f"{'Ptx':>5} | {'Dist-Full':>10} | {'C-MMSE':>10} | {'C-MMSE-MxQ':>10} | {'Proposed BER':>12} | {'Proposed Bits':>10}")
        for p in p_tx_list:
            sh, s_true, H, snr, y, lbl = generate_eval_batch(sys_model, args.test_samples, p, tau_val)
            
            ber_dist = compute_ber(s_true, compute_dist_full_ber(sh))
            ber_cmmse = compute_ber(s_true, compute_cmmse_ber(y, H, sys_model))
            ber_mxq = compute_ber(s_true, compute_cmmse_mixed_q_detection(y, H, sys_model, 8))
            
            ber_p, bits_p = run_model_eval(model, sh, H, y, snr, lbl, device)
            print(f"{p:5} | {ber_dist:10.5f} | {ber_cmmse:10.5f} | {ber_mxq:10.5f} | {ber_p:12.5f} | {bits_p:10.1f}")

    # --- Task 3: Generalization ---
    print(f"\n{'='*20} Task 3: Generalization to K=12 (tau=0.05) {'='*20}")
    sys_k12 = CellFreeSystem(args.L, args.N, 12, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_k12.generate_scenario()
    print(f"{'Ptx':>5} | {'Dist-Full (K12)':>15} | {'Proposed (K12)':>15}")
    for p in p_tx_list:
        sh, s_true, H, snr, y, lbl = generate_eval_batch(sys_k12, args.test_samples, p, 0.05, K_override=12)
        ber_dist = compute_ber(s_true, compute_dist_full_ber(sh))
        ber_p, _ = run_model_eval(model, sh, H, y, snr, lbl, device)
        print(f"{p:5} | {ber_dist:15.5f} | {ber_p:15.5f}")

    # --- Task 4: Pareto Front ---
    print(f"\n{'='*20} Task 4: Pareto Front (Ptx=10dBm, tau=0.05) {'='*20}")
    p_pareto = 10
    sh, s_true, H, snr, y, lbl = generate_eval_batch(sys_model, args.test_samples, p_pareto, 0.05)
    penalties = [0.0, 0.02, 0.05, 0.1, 0.2]
    print(f"{'Penalty':>8} | {'Avg Bits':>10} | {'Prop BER':>10} | {'Baseline (Q-Dist) BER'}")
    for bp in penalties:
        ber_p, bits_p = run_model_eval(model, sh, H, y, snr, lbl, device, bit_penalty_bias=bp)
        # Heuristic comparison: use bits_p as a guide for quantization resolution
        ber_base = compute_ber(s_true, compute_dist_q_ber(sh, int(np.clip(bits_p/16, 1, 8))))
        print(f"{bp:8.2f} | {bits_p:10.1f} | {ber_p:10.5f} | {ber_base:10.5f}")

    # --- Task 5: Ablation ---
    print(f"\n{'='*20} Task 5: Ablation Study (tau=0.05) {'='*20}")
    ab_p_tx = [-10, 0, 10, 20]
    modes = [
        ("Full Proposed", {}),
        ("No-MRC", {"ablation_no_mrc": True}),
        ("No-Bit-Aware", {"ablation_no_bit_aware": True}),
        ("No-Attention", {"ablation_no_attention": True})
    ]
    header = "Ptx   " + "".join([f"| {m[0]:>15} " for m in modes])
    print(header)
    for p in ab_p_tx:
        sh, s_true, H, snr, y, lbl = generate_eval_batch(sys_model, args.test_samples, p, 0.05)
        row = f"{p:<6}"
        for name, m_kwargs in modes:
            ber_p, _ = run_model_eval(model, sh, H, y, snr, lbl, device, **m_kwargs)
            row += f"| {ber_p:15.5f} "
        print(row)

if __name__ == "__main__":
    main()
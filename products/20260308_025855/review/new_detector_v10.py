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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V10 (MRC-Enhanced, Residual Logits, CE Loss)")
    # System parameters
    parser.add_argument("--L", type=int, default=16, help="Number of Access Points (APs)")
    parser.add_argument("--N", type=int, default=4, help="Number of antennas per AP")
    parser.add_argument("--K", type=int, default=8, help="Number of single-antenna users")
    parser.add_argument("--area_size", type=float, default=1.0, help="Area size in km")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="System bandwidth in Hz")
    parser.add_argument("--noise_psd", type=float, default=-174, help="Noise PSD in dBm/Hz")
    
    # Training hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=20, help="Phase 1 epochs (detector pre-training)")
    parser.add_argument("--epochs_phase2", type=int, default=40, help="Phase 2 epochs (joint QAT training)")
    parser.add_argument("--batches_per_epoch", type=int, default=100, help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
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
def generate_data_batch_v10(sys_model, batch_size, p_tx_dbm, tau_est=0.0):
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K)

    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    beta_w_expanded = beta_w[np.newaxis, :, np.newaxis, :]
    H = np.sqrt(beta_w_expanded) * h_small  # (batch_size, L, N, K)
    
    if tau_est > 0:
        noise_est = (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)) / np.sqrt(2) * np.sqrt(tau_est)
        H_est = H + noise_est
    else:
        H_est = H.copy()

    labels = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    map_func = np.vectorize(mapping.get)
    s = map_func(labels) / np.sqrt(2)

    y_clean = np.einsum('blnk,bk->bln', H, s)
    z = (np.random.randn(batch_size, sys_model.L, sys_model.N) +
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    y = y_clean + z

    H_conj_trans = H_est.conj().transpose(0, 1, 3, 2)
    noise_cov = sys_model.noise_w * np.eye(sys_model.N).reshape(1, 1, sys_model.N, sys_model.N)
    R_y = H_est @ H_conj_trans + noise_cov
    R_y_inv = np.linalg.inv(R_y)
    W_l = H_conj_trans @ R_y_inv
    s_hat = (W_l @ y[..., np.newaxis]).squeeze(-1)

    local_snr = 10 * np.log10(np.sum(np.abs(H_est) ** 2, axis=2) / sys_model.noise_w + 1e-12) / 10.0

    return s_hat, s, H_est, local_snr, y, labels


# ============================================================
# 2. Baseline Tools
# ============================================================
def compute_ber(s_true, s_pred_complex):
    s_true_real = np.sign(s_true.real)
    s_true_imag = np.sign(s_true.imag)
    s_pred_real = np.sign(s_pred_complex.real)
    s_pred_imag = np.sign(s_pred_complex.imag)
    s_pred_real[s_pred_real == 0] = 1
    s_pred_imag[s_pred_imag == 0] = 1
    err = (s_true_real != s_pred_real).sum() + (s_true_imag != s_pred_imag).sum()
    return err / (s_true.size * 2)

def compute_dist_full_ber(s_hat, s):
    return compute_ber(s, s_hat.mean(axis=1))

def uniform_quantize_np(x_real, num_bits):
    if num_bits <= 0: return np.zeros_like(x_real)
    if num_bits >= 24: return x_real
    levels = 2 ** num_bits
    x_max = np.max(np.abs(x_real)) + 1e-12
    step = 2 * x_max / (levels - 1)
    idx = np.clip(np.round((x_real + x_max) / step), 0, levels - 1)
    return idx * step - x_max

def compute_cmmse_mixed_q_detection(H, y, s, noise_w, c_target, K, N):
    B, L, N_ant, K_users = H.shape
    total_bits_per_ap = c_target * K_users
    bit_options = [2, 4, 6, 8, 12, 16]
    best_ber = 1.0
    for b_y in bit_options:
        for b_H in bit_options:
            if (2 * N_ant * b_y + K_users * 2 * N_ant * b_H) > total_bits_per_ap * 1.05: continue
            H_q = uniform_quantize_np(H.real, b_H) + 1j * uniform_quantize_np(H.imag, b_H)
            y_q = uniform_quantize_np(y.real, b_y) + 1j * uniform_quantize_np(y.imag, b_y)
            H_all = H_q.reshape(B, L * N_ant, K_users)
            y_all = y_q.reshape(B, L * N_ant, 1)
            H_H = H_all.conj().transpose(0, 2, 1)
            R = H_H @ H_all + (noise_w + 1e-6) * np.eye(K_users)
            try:
                s_hat = (np.linalg.inv(R) @ (H_H @ y_all)).squeeze(-1)
                ber = compute_ber(s, s_hat)
                if ber < best_ber: best_ber = ber
            except: continue
    return best_ber

# ============================================================
# 3. Improved GNN + Transformer Hybrid Detector V10
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        messages = self.message_mlp(h)
        mean_message = messages.mean(dim=1, keepdim=True).expand_as(h)
        updated = self.update_mlp(torch.cat([h, mean_message], dim=-1))
        return self.layer_norm(h + updated)

class APAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, h):
        attn_weights = torch.softmax(self.attn_net(h), dim=1)
        return (h * attn_weights).sum(dim=1), attn_weights

class GNNTransformerDetectorV10(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.demod_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.channel_mlp = nn.Sequential(nn.Linear(2 * N, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.mrc_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim))
        
        # Fusion
        fusion_input_dim = 2 * (hidden_dim // 4) + 3 + 2 + 1 # demod, channel, bitwidths(3), interference(2), snr(1)
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)

        self.gnn_layers = nn.ModuleList([MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)])
        self.ap_aggregator = APAggregator(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
                                                   batch_first=True, dropout=dropout, norm_first=True)
        self.transformer_ic = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Residual Logits Parameters
        self.output_head = nn.Linear(hidden_dim, 4)
        self.base_scale = nn.Parameter(torch.tensor(10.0))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # QPSK Constellation Tensor
        constellation = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32) / math.sqrt(2)
        self.register_buffer('constellation', constellation)

    def forward(self, s_hat_q, H_q, y_q, bitwidth_features, local_snr):
        B, L, K = s_hat_q.shape[:3]
        N = H_q.size(2)

        # 1. Global MRC Feature
        H_c = torch.complex(H_q[..., 0], H_q[..., 1]) # (B, L, N, K)
        y_c = torch.complex(y_q[..., 0], y_q[..., 1]) # (B, L, N)
        H_all = H_c.reshape(B, L * N, K)
        y_all = y_c.reshape(B, L * N, 1)
        s_mrc_c = torch.matmul(H_all.conj().transpose(-1, -2), y_all).squeeze(-1) # (B, K)
        s_mrc_feat = torch.stack([s_mrc_c.real, s_mrc_c.imag], dim=-1) # (B, K, 2)
        mrc_embed = self.mrc_mlp(s_mrc_feat)

        # 2. Local Node Feature Encoding
        demod_feat = self.demod_mlp(s_hat_q)
        H_flat = H_q.permute(0, 1, 3, 2, 4).reshape(B, L, K, N * 2)
        channel_feat = self.channel_mlp(H_flat)

        h_pow = (H_q ** 2).sum(dim=(2, 4))
        interf_feat = torch.log1p(torch.stack([h_pow, h_pow.sum(dim=2, keepdim=True) - h_pow], dim=-1))
        
        combined = torch.cat([demod_feat, channel_feat, bitwidth_features, interf_feat, local_snr], dim=-1)
        node_features = self.fusion_mlp(combined)
        node_features = self.fusion_bn(node_features.reshape(-1, self.hidden_dim)).reshape(B, L, K, self.hidden_dim)

        # 3. GNN Propagation
        h = node_features
        for gnn in self.gnn_layers: h = gnn(h)

        # 4. Global Aggregation & Transformer IC
        user_features, _ = self.ap_aggregator(h)
        user_features = user_features + mrc_embed
        ic_out = self.transformer_ic(user_features)

        # 5. Base Logits Residual
        s_hat_avg = s_hat_q.mean(dim=1) # (B, K, 2)
        dist_sq = ((s_hat_avg.unsqueeze(2) - self.constellation.unsqueeze(0).unsqueeze(0))**2).sum(dim=-1)
        base_logits = -dist_sq * self.base_scale
        
        logits = base_logits + self.alpha * self.output_head(ic_out)
        return logits

# ============================================================
# 4. Adaptive Quantizer & Joint Model
# ============================================================
class TripleAdaptiveQuantizerV10(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.policy = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 15)) # 5*3
        self.qs = nn.ModuleList([LSQQuantizer(num_bits=b, init_s=0.1/(2**(b/4))) if b>0 else None for b in [0, 4, 8, 12, 16]])

    def forward(self, v, H, y, local_snr, tau=1.0):
        B, L, K, _ = v.shape
        h_pow = (H ** 2).sum(dim=(2, 4))
        sir = torch.log1p(h_pow / (h_pow.sum(dim=2, keepdim=True) - h_pow + 1e-10)).unsqueeze(-1)
        y_pow = torch.log1p((y ** 2).sum(dim=(2, 3))).view(B, L, 1, 1).expand(B, L, K, 1)
        policy_in = torch.cat([v, local_snr, h_pow.unsqueeze(-1), sir, y_pow, torch.norm(v, dim=-1, keepdim=True)], dim=-1)
        
        logits = self.policy(policy_in)
        w_y = F.gumbel_softmax(logits[..., :5].mean(dim=2), tau=tau, hard=True)
        w_H = F.gumbel_softmax(logits[..., 5:10], tau=tau, hard=True)
        w_d = F.gumbel_softmax(logits[..., 10:], tau=tau, hard=True)

        def apply_q(x, w, dims_for_w):
            stacked = torch.stack([(q(x) if q else torch.zeros_like(x)) for q in self.qs], dim=-1)
            return (stacked * w.view(*dims_for_w, 5)).sum(dim=-1)

        y_q = apply_q(y, w_y, [B, L, 1, 1])
        H_q = apply_q(H, w_H, [B, L, K, 1, 1])
        v_q = apply_q(v, w_d, [B, L, K, 1])

        bit_vals = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=v.device)
        bits_y = (2 * self.N * (w_y * bit_vals).sum(-1, keepdim=True) / K).expand(B, L, K)
        bits_H = 2 * self.N * (w_H * bit_vals).sum(-1)
        bits_d = 2 * (w_d * bit_vals).sum(-1)
        return v_q, H_q, y_q, bits_y + bits_H + bits_d, bits_y, bits_H, bits_d

class JointModelV10(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4):
        super().__init__()
        self.quantizer = TripleAdaptiveQuantizerV10(N)
        self.detector = GNNTransformerDetectorV10(L, N, K, hidden_dim, num_gnn_layers, num_transformer_layers, num_heads)

    def forward(self, v, H, y, snr, tau=1.0, use_quantization=True, noise_std=0.0):
        if self.training and noise_std > 0:
            v, H, y = v + torch.randn_like(v)*noise_std, H + torch.randn_like(H)*noise_std, y + torch.randn_like(y)*noise_std
        
        if use_quantization:
            v_q, H_q, y_q, exp_bits, b_y, b_H, b_d = self.quantizer(v, H, y, snr, tau)
            bw_feat = torch.stack([b_y/(32*self.quantizer.N/8), b_H/(32*self.quantizer.N), b_d/32.0], dim=-1)
        else:
            v_q, H_q, y_q, exp_bits = v, H, y, torch.zeros(v.shape[0], v.shape[1], v.shape[2], device=v.device)
            bw_feat = torch.ones(v.shape[0], v.shape[1], v.shape[2], 3, device=v.device)
            
        logits = self.detector(v_q, H_q, y_q, bw_feat, snr)
        return logits, exp_bits

# ============================================================
# 5. Main Loop
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    test_p_tx = [-10, -5, 0, 5, 10, 15, 20]
    test_set = {}
    print("Generating Test Set...")
    for p in test_p_tx:
        s_h, s, H, snr, y, lbl = generate_data_batch_v10(sys_model, args.test_samples, p, args.tau_est)
        test_set[p] = {
            'v': torch.tensor(np.stack([s_h.real, s_h.imag], -1), dtype=torch.float32).to(device),
            'H': torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device),
            'snr': torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device),
            'y': torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device),
            'lbl': torch.tensor(lbl, dtype=torch.long).to(device),
            's_np': s, 's_h_np': s_h, 'H_np': H, 'y_np': y
        }

    model = JointModelV10(args.L, args.N, args.K, args.hidden_dim, args.num_gnn_layers, args.num_transformer_layers, args.num_heads).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Pre-training
    print(f"\nPhase 1: Detector Pre-training ({args.epochs_phase1} epochs)")
    opt1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    for epoch in range(args.epochs_phase1):
        model.train(); loss_total = 0
        for _ in range(args.batches_per_epoch):
            p_train = np.random.uniform(5, 15)
            sh, s, H, snr, y, lbl = generate_data_batch_v10(sys_model, args.batch_size, p_train, args.tau_est)
            v_t = torch.tensor(np.stack([sh.real, sh.imag], -1), dtype=torch.float32).to(device)
            H_t = torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device)
            snr_t = torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device)
            y_t = torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device)
            lbl_t = torch.tensor(lbl, dtype=torch.long).to(device)
            
            opt1.zero_grad()
            logits, _ = model(v_t, H_t, y_t, snr_t, use_quantization=False, noise_std=args.noise_injection_std)
            loss = criterion(logits.view(-1, 4), lbl_t.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt1.step()
            loss_total += loss.item()
        print(f"Phase 1 Epoch {epoch+1}/{args.epochs_phase1} Loss: {loss_total/args.batches_per_epoch:.4f}")

    # Phase 2: Joint QAT
    print(f"\nPhase 2: Joint QAT ({args.epochs_phase2} epochs)")
    opt2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    for epoch in range(args.epochs_phase2):
        model.train(); ce_total, bit_total = 0, 0
        tau = max(args.tau_end, args.tau_start * (args.tau_end/args.tau_start)**(epoch/(args.epochs_phase2-1)))
        for _ in range(args.batches_per_epoch):
            p_train = np.random.uniform(5, 15)
            sh, s, H, snr, y, lbl = generate_data_batch_v10(sys_model, args.batch_size, p_train, args.tau_est)
            v_t = torch.tensor(np.stack([sh.real, sh.imag], -1), dtype=torch.float32).to(device)
            H_t = torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device)
            snr_t = torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device)
            y_t = torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device)
            lbl_t = torch.tensor(lbl, dtype=torch.long).to(device)

            opt2.zero_grad()
            logits, exp_bits = model(v_t, H_t, y_t, snr_t, tau=tau, use_quantization=True)
            loss_ce = criterion(logits.view(-1, 4), lbl_t.view(-1))
            loss_bit = args.lambda_penalty * ((exp_bits.mean() - args.c_target)**2)
            (loss_ce + loss_bit).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt2.step()
            ce_total += loss_ce.item(); bit_total += exp_bits.mean().item()
        
        if (epoch+1)%5 == 0 or epoch==0:
            model.eval(); td = test_set[10]
            with torch.no_grad():
                lgt, eb = model(td['v'], td['H'], td['y'], td['snr'], tau=args.tau_end)
                ber = compute_ber(td['s_np'], (torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], device=device)[lgt.argmax(-1)]/math.sqrt(2)).cpu().numpy())
            print(f"Phase 2 Epoch {epoch+1:02d} Loss CE: {ce_total/args.batches_per_epoch:.4f} Bits: {bit_total/args.batches_per_epoch:.1f} Val@10 BER: {ber:.5f}")

    # Evaluation
    print(f"\n{'='*100}\nFinal Evaluation\n{'='*100}")
    print(f"{'Ptx':<5} | {'Dist-Full':<10} | {'CMMSE-MxQ':<10} | {'Proposed':<10} | {'Bits':<5}")
    model.eval()
    qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / math.sqrt(2)
    for p in test_p_tx:
        td = test_set[p]
        ber_df = compute_dist_full_ber(td['s_h_np'], td['s_np'])
        ber_mq = compute_cmmse_mixed_q_detection(td['H_np'], td['y_np'], td['s_np'], sys_model.noise_w, args.c_target, args.K, args.N)
        with torch.no_grad():
            lgt, eb = model(td['v'], td['H'], td['y'], td['snr'], tau=args.tau_end)
            ber_prop = compute_ber(td['s_np'], qpsk[lgt.argmax(-1).cpu().numpy()])
        print(f"{p:<5} | {ber_df:<10.5f} | {ber_mq:<10.5f} | {ber_prop:<10.5f} | {eb.mean():.1f}")
    
    torch.save(model.state_dict(), "new_joint_model_v10.pth")

if __name__ == "__main__":
    main()
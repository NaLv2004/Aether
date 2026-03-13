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
    parser = argparse.ArgumentParser(description="GNN+Transformer Hybrid Detector V12 (MRC-Enhanced, Residual Logits, Triple Quant V12 Logic)")
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
# 1. Data Generation (Vectorized)
# ============================================================
def generate_data_batch_v12(sys_model, batch_size, p_tx_dbm, tau_est=0.0):
    p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
    beta_w = p_tx_w * sys_model.beta  # (L, K)
    h_small = (np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K) +
               1j * np.random.randn(batch_size, sys_model.L, sys_model.N, sys_model.K)) / np.sqrt(2)
    H = np.sqrt(beta_w[np.newaxis, :, np.newaxis, :]) * h_small
    if tau_est > 0:
        H_est = H + (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)) / np.sqrt(2) * np.sqrt(tau_est)
    else:
        H_est = H.copy()
    labels = np.random.randint(0, 4, size=(batch_size, sys_model.K))
    mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: 1 - 1j}
    s = np.vectorize(mapping.get)(labels) / np.sqrt(2)
    y = np.einsum('blnk,bk->bln', H, s) + (np.random.randn(batch_size, sys_model.L, sys_model.N) + 
         1j * np.random.randn(batch_size, sys_model.L, sys_model.N)) / np.sqrt(2) * np.sqrt(sys_model.noise_w)
    
    H_H = H_est.conj().transpose(0, 1, 3, 2)
    R_y_inv = np.linalg.inv(H_est @ H_H + sys_model.noise_w * np.eye(sys_model.N))
    s_hat = (H_H @ R_y_inv @ y[..., np.newaxis]).squeeze(-1)
    local_snr = 10 * np.log10(np.sum(np.abs(H_est)**2, axis=2) / sys_model.noise_w + 1e-12) / 10.0
    return s_hat, s, H_est, local_snr, y, labels

def compute_ber(s_true, s_pred_complex):
    s_true_real, s_true_imag = np.sign(s_true.real), np.sign(s_true.imag)
    s_pred_real, s_pred_imag = np.sign(s_pred_complex.real), np.sign(s_pred_complex.imag)
    s_pred_real[s_pred_real == 0] = 1; s_pred_imag[s_pred_imag == 0] = 1
    err = (s_true_real != s_pred_real).sum() + (s_true_imag != s_pred_imag).sum()
    return err / (s_true.size * 2)

# ============================================================
# 2. Triple Policy & Quantizer (V12 Logic)
# ============================================================
class TriplePolicyNetworkV12(nn.Module):
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

class TripleAdaptiveQuantizerV12(nn.Module):
    def __init__(self, N, policy_hidden=64):
        super().__init__()
        self.N = N
        self.policy = TriplePolicyNetworkV12(input_dim=9, policy_hidden=policy_hidden)
        self.bit_options = [0, 4, 8, 12, 16]
        # Quantizers for y, H, demod
        self.qs_y = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])
        self.qs_H = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])
        self.qs_d = nn.ModuleList([LSQQuantizer(b, 0.1/(2**(b/4))) if b>0 else None for b in self.bit_options])

    def forward(self, v, H, y, local_snr, tau=1.0):
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

        logits_y, logits_H, logits_demod = self.policy(policy_input)
        w_y = F.gumbel_softmax(logits_y.mean(dim=2), tau=tau, hard=True)
        w_H = F.gumbel_softmax(logits_H, tau=tau, hard=True)
        w_d = F.gumbel_softmax(logits_demod, tau=tau, hard=True)

        def apply_q(x, w, qs, dims_for_w):
            stacked = torch.stack([(q(x) if q else torch.zeros_like(x)) for q in qs], dim=-1)
            return (stacked * w.view(*dims_for_w, 5)).sum(dim=-1)

        y_q = apply_q(y, w_y, self.qs_y, [B, L, 1, 1])
        # Fix: correctly align H's shape (B, L, N, K, 2) by viewing w_H into [B, L, 1, K, 1, 5]
        H_q = apply_q(H, w_H, self.qs_H, [B, L, 1, K, 1])
        v_q = apply_q(v, w_d, self.qs_d, [B, L, K, 1])

        bit_vals = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=v.device)
        b_y_link = (2 * self.N * (w_y * bit_vals).sum(-1, keepdim=True) / K).expand(B, L, K)
        b_H_link = 2 * self.N * (w_H * bit_vals).sum(-1)
        b_d_link = 2 * (w_d * bit_vals).sum(-1)
        
        return v_q, H_q, y_q, b_y_link + b_H_link + b_d_link, b_y_link, b_H_link, b_d_link

# ============================================================
# 3. Detector (V12 Mechanisms: MRC + Residual Logits)
# ============================================================
class MeanFieldGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, h):
        mean_msg = self.message_mlp(h).mean(dim=1, keepdim=True).expand_as(h)
        return self.layer_norm(h + self.update_mlp(torch.cat([h, mean_msg], dim=-1)))

class GNNTransformerDetectorV12(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128, num_gnn_layers=3, num_transformer_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.demod_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.channel_mlp = nn.Sequential(nn.Linear(2 * N, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, hidden_dim // 4))
        self.mrc_mlp = nn.Sequential(nn.Linear(2, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim))
        
        fusion_dim = 2 * (hidden_dim // 4) + 3 + 2 + 1 
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        self.gnn_layers = nn.ModuleList([MeanFieldGNNLayer(hidden_dim) for _ in range(num_gnn_layers)])
        self.ap_agg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
                                               batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        self.output_head = nn.Linear(hidden_dim, 4)
        self.base_scale = nn.Parameter(torch.tensor(10.0)); self.alpha = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('constellation', torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32) / math.sqrt(2))

    def forward(self, s_hat_q, H_q, y_q, bw_feat, local_snr):
        B, L, K = s_hat_q.shape[:3]; N = H_q.size(2)
        # MRC
        H_c = torch.complex(H_q[..., 0], H_q[..., 1]); y_c = torch.complex(y_q[..., 0], y_q[..., 1])
        s_mrc_c = torch.matmul(H_c.reshape(B, L*N, K).conj().transpose(-1, -2), y_c.reshape(B, L*N, 1)).squeeze(-1)
        mrc_embed = self.mrc_mlp(torch.stack([s_mrc_c.real, s_mrc_c.imag], dim=-1))
        
        # Features
        h_pow = (H_q ** 2).sum(dim=(2, 4))
        interf = torch.log1p(torch.stack([h_pow, h_pow.sum(dim=2, keepdim=True) - h_pow], dim=-1))
        node_feat = self.fusion_mlp(torch.cat([self.demod_mlp(s_hat_q), self.channel_mlp(H_q.permute(0, 1, 3, 2, 4).reshape(B, L, K, N*2)),
                                              bw_feat, interf, local_snr], dim=-1))
        h = self.fusion_bn(node_feat.reshape(-1, self.hidden_dim)).reshape(B, L, K, self.hidden_dim)
        for gnn in self.gnn_layers: h = gnn(h)
        
        # Aggregation
        attn = torch.softmax(self.ap_agg(h), dim=1)
        user_feat = (h * attn).sum(dim=1) + mrc_embed
        ic_out = self.transformer(user_feat)
        
        # Base Logits
        dist_sq = ((s_hat_q.mean(dim=1).unsqueeze(2) - self.constellation.unsqueeze(0).unsqueeze(0))**2).sum(dim=-1)
        return -dist_sq * self.base_scale + self.alpha * self.output_head(ic_out)

class JointModelV12(nn.Module):
    def __init__(self, L, N, K, hidden_dim=128):
        super().__init__()
        self.quantizer = TripleAdaptiveQuantizerV12(N)
        self.detector = GNNTransformerDetectorV12(L, N, K, hidden_dim)

    def forward(self, v, H, y, snr, tau=1.0, use_quantization=True, noise_std=0.0):
        if self.training and noise_std > 0:
            v, H, y = v + torch.randn_like(v)*noise_std, H + torch.randn_like(H)*noise_std, y + torch.randn_like(y)*noise_std
        if use_quantization:
            v_q, H_q, y_q, exp_bits, b_y, b_H, b_d = self.quantizer(v, H, y, snr, tau)
            bw_feat = torch.stack([b_y/(32*self.quantizer.N/8), b_H/(32*self.quantizer.N), b_d/32.0], dim=-1)
        else:
            v_q, H_q, y_q, exp_bits = v, H, y, torch.zeros(v.size(0), v.size(1), v.size(2), device=v.device)
            bw_feat = torch.ones(v.size(0), v.size(1), v.size(2), 3, device=v.device)
        return self.detector(v_q, H_q, y_q, bw_feat, snr), exp_bits

# ============================================================
# 4. Main Script
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys_model = CellFreeSystem(args.L, args.N, args.K, args.area_size, args.bandwidth, args.noise_psd, 0.0)
    sys_model.generate_scenario()

    test_p_tx = [-10, -5, 0, 5, 10, 15, 20]; test_set = {}
    print("Generating Test Set...")
    for p in test_p_tx:
        sh, s, H, snr, y, lbl = generate_data_batch_v12(sys_model, args.test_samples, p, args.tau_est)
        test_set[p] = {
            'v': torch.tensor(np.stack([sh.real, sh.imag], -1), dtype=torch.float32).to(device),
            'H': torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device),
            'snr': torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device),
            'y': torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device),
            'lbl': torch.tensor(lbl, dtype=torch.long).to(device), 's_np': s
        }

    model = JointModelV12(args.L, args.N, args.K, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nPhase 1: Detector Pre-training ({args.epochs_phase1} epochs)")
    opt1 = optim.Adam(model.detector.parameters(), lr=args.lr_phase1)
    for epoch in range(args.epochs_phase1):
        model.train(); loss_total = 0
        for _ in range(args.batches_per_epoch):
            sh, s, H, snr, y, lbl = generate_data_batch_v12(sys_model, args.batch_size, np.random.uniform(5, 15), args.tau_est)
            v_t = torch.tensor(np.stack([sh.real, sh.imag], -1), dtype=torch.float32).to(device)
            H_t = torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device)
            snr_t = torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device)
            y_t = torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device)
            lbl_t = torch.tensor(lbl, dtype=torch.long).to(device)
            opt1.zero_grad(); logits, _ = model(v_t, H_t, y_t, snr_t, use_quantization=False, noise_std=args.noise_injection_std)
            loss = criterion(logits.view(-1, 4), lbl_t.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt1.step(); loss_total += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs_phase1} Loss: {loss_total/args.batches_per_epoch:.4f}")

    print(f"\nPhase 2: Joint QAT ({args.epochs_phase2} epochs)")
    opt2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    for epoch in range(args.epochs_phase2):
        model.train(); ce_total, bit_total = 0, 0
        tau = max(args.tau_end, args.tau_start * (args.tau_end/args.tau_start)**(epoch/(args.epochs_phase2-1) if args.epochs_phase2 > 1 else 1))
        for _ in range(args.batches_per_epoch):
            sh, s, H, snr, y, lbl = generate_data_batch_v12(sys_model, args.batch_size, np.random.uniform(5, 15), args.tau_est)
            v_t = torch.tensor(np.stack([sh.real, sh.imag], -1), dtype=torch.float32).to(device)
            H_t = torch.tensor(np.stack([H.real, H.imag], -1), dtype=torch.float32).to(device)
            snr_t = torch.tensor(snr, dtype=torch.float32).unsqueeze(-1).to(device)
            y_t = torch.tensor(np.stack([y.real, y.imag], -1), dtype=torch.float32).to(device)
            lbl_t = torch.tensor(lbl, dtype=torch.long).to(device)
            opt2.zero_grad(); logits, eb = model(v_t, H_t, y_t, snr_t, tau=tau, use_quantization=True)
            loss_ce = criterion(logits.view(-1, 4), lbl_t.view(-1))
            loss_bit = args.lambda_penalty * ((eb.mean() - args.c_target)**2)
            (loss_ce + loss_bit).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt2.step()
            ce_total += loss_ce.item(); bit_total += eb.mean().item()
        if (epoch+1)%5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} CE Loss: {ce_total/args.batches_per_epoch:.4f} Avg Bits: {bit_total/args.batches_per_epoch:.1f}")

    print(f"\nFinal Evaluation:")
    model.eval(); qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / math.sqrt(2)
    for p in test_p_tx:
        td = test_set[p]
        with torch.no_grad():
            lgt, eb = model(td['v'], td['H'], td['y'], td['snr'], tau=args.tau_end)
            ber = compute_ber(td['s_np'], qpsk[lgt.argmax(-1).cpu().numpy()])
        print(f"Ptx: {p:>3} dBm | BER: {ber:.5f} | Bits: {eb.mean():.1f}")
    torch.save(model.state_dict(), "new_joint_model_v12.pth")

if __name__ == "__main__":
    main()
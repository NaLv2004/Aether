import logging
import colorlog
import os
import shutil
def setup_logger(log_file_path):
    """
    初始化全局 Logger，同时输出到控制台(带颜色)和文件(纯文本)
    """
    # 确保日志文件夹存在
    # os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("AgentLogger")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 Handler 导致日志打印多次
    if not logger.handlers:
        # 1. 控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
            log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red'}
        ))
        logger.addHandler(console_handler)

        # 2. 文件
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger
    
logger = setup_logger("experiment_run.log")
import http.client
import json
import base64
import os

class PDFReader:
    def __init__(self, api_key, system_prompt, context_window_size=1, host="jeniya.top", model="gemini-3-flash-preview"):
        """
        初始化 PDFReader
        :param api_key: 你的 API 密钥 / Token
        :param system_prompt: 系统提示词，用于设定模型的身份和行为
        :param context_window_size: 上下文窗口大小。1表示只记得当前对话，2表示记得上一次问答+当前提问，以此类推。
        :param host: API代理地址，默认为图片中的 jeniya.top
        :param model: 模型名称
        """
        self.api_key = api_key
        self.system_prompt = system_prompt
        # 内部保证 context_window_size 至少为 1
        self.context_window_size = max(1, context_window_size) 
        self.host = host
        self.model = model
        self.history = [] # 用于存储对话上下文

    def _encode_pdf_to_base64(self, pdf_path):
        """将本地 PDF 文件读取并转换为 Base64 字符串"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到指定的 PDF 文件: {pdf_path}")
            
        with open(pdf_path, "rb") as file:
            binary_data = file.read()
            return base64.b64encode(binary_data).decode('utf-8')

    def read_pdf(self, pdf_path, output_txt_path, user_prompt="Summarize this document"):
        """
        解析 PDF，请求 Gemini 模型，并将结果追加到文本文件
        :param pdf_path: 要读取的 PDF 文件路径
        :param output_txt_path: 结果追加写入的 txt 文件路径
        :param user_prompt: 用户本次的具体提问
        """
        print(f"正在处理 PDF: {pdf_path}...")
        
        # 1. 将 PDF 转换为 Base64
        b64_data = self._encode_pdf_to_base64(pdf_path)

        # 2. 构建本次用户的请求 Part
        current_user_parts = [
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": b64_data
                }
            },
            {
                "text": user_prompt
            }
        ]

        # 3. 更新历史记录并维护上下文窗口
        self.history.append({"role": "user", "parts": current_user_parts})
        
        # 计算需要保留的消息数量: (窗口大小 * 2) - 1 
        # 例如 size=1, 保留 1 条(当前问题)
        # 例如 size=2, 保留 3 条(上一次问题, 上一次回答, 当前问题)
        keep_messages = (self.context_window_size * 2) - 1
        if len(self.history) > keep_messages:
            self.history = self.history[-keep_messages:]
            # 确保历史记录始终以 user 角色开头 (Gemini 的强制要求)
            if self.history[0]["role"] != "user":
                self.history = self.history[1:]

        # 4. 构建完整的请求 Payload
        payload_dict = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": self.history
        }
        payload = json.dumps(payload_dict)

        # 5. 设置请求头（参照图片格式，使用 Bearer Token）
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # 6. 发起 HTTP 请求
        print("正在向模型发送请求...")
        conn = http.client.HTTPSConnection(self.host)
        # 拼接 URL，注意图片中 URL 也带了 key 参数，如果你的中转站需要，可保留
        request_url = f"/v1beta/models/{self.model}:generateContent"
        
        try:
            conn.request("POST", request_url, payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            if res.status != 200:
                print(f"请求失败! 状态码: {res.status}, 响应: {data.decode('utf-8')}")
                return

            # 7. 解析 JSON 响应
            response_json = json.loads(data.decode("utf-8"))
            
            # 提取模型生成的文本内容
            # Gemini 的响应结构通常为: candidates -> [0] -> content -> parts -> [0] -> text
            answer_text = response_json.get("candidates", [{}])[0] \
                                       .get("content", {}) \
                                       .get("parts", [{}])[0] \
                                       .get("text", "")

            if not answer_text:
                print("解析失败：未能从返回结果中提取到文本内容。完整返回：", response_json)
                return

            print("获取回答成功！")

            # 8. 将模型的回答追加保存到上下文中，为后续多轮对话做准备
            self.history.append({
                "role": "model", 
                "parts": [{"text": answer_text}]
            })

            # 9. 将结果追加写入指定的 txt 文件
            # 使用 'a' 模式 (append) 追加内容
            with open(output_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"--- 对 PDF: {os.path.basename(pdf_path)} 的提问 ---\n")
                f.write(f"用户: {user_prompt}\n")
                f.write(f"模型: {answer_text}\n\n")
            print(f"结果已成功追加到: {output_txt_path}\n")

        except Exception as e:
            print(f"发生异常: {str(e)}")
        finally:
            conn.close()
            
            
# if __name__ == "__main__":
#     # 1. 准备你的配置
#     YOUR_API_KEY = os.getenv("JIANYI_API_KEY") 
#     # YOUR_API_KEY = "sk-xxxxxxxxxxxxxxxxx" # 替换为你的真实 API Key / Token
#     SYS_PROMPT = "请总结以下论文。"
    
#     # 2. 初始化类，上下文窗口设置为 2（记忆上一轮对话）
#     # host 默认是 jeniya.top，如果你用官方或其他代理，可以在此处修改 host="generativelanguage.googleapis.com"
#     reader = PDFReader(
#         api_key=YOUR_API_KEY, 
#         system_prompt=SYS_PROMPT, 
#         context_window_size=2
#     )

#     # 3. 第一次请求：要求总结
#     pdf_file = r"pdfs_ieee\\10.1109_TCCN.2017.2758370.pdf"      # 本地的 PDF 路径
#     output_file = "test.txt"   # 准备写入的文本文件路径
    
#     reader.read_pdf(
#         pdf_path=pdf_file, 
#         output_txt_path=output_file, 
#         user_prompt="请帮详细总结这篇论文，并写出式（4）是什么。"
#     )

#     # 4. 第二次请求：基于上一次的记忆追问 (测试上下文)
#     # 此时因为 context_window_size 设定为 2，它会记得上一次它回答的那 3 个要点
#     # reader.read_pdf(
#     #     pdf_path=pdf_file, 
#     #     output_txt_path=output_file, 
#     #     user_prompt="针对你刚才列出的第2个要点，在这份文档中有什么具体的数据支撑吗？"
#     # )
    

# compile given latex project to pdf   
import os
import subprocess

def compile_latex_project(directory, main_filename="main.tex", texlive_bin_dir=None):
    """
    强制编译 LaTeX 项目，并解决 MiKTeX / TeX Live 冲突问题。
    
    参数:
        directory: 项目工作目录
        main_filename: 主文件名
        texlive_bin_dir: (可选) 强制指定 TeX Live 的 bin 目录绝对路径
                         例如: r"C:\texlive\2023\bin\windows"
    """
    main_file_path = os.path.join(directory, main_filename)
    if not os.path.exists(main_file_path):
        print(f"❌ 错误: 找不到主文件 {os.path.abspath(main_file_path)}")
        return False

    base_name = os.path.splitext(main_filename)[0]

    # ================= 核心修复：处理环境变量冲突 =================
    # 复制当前系统的环境变量
    custom_env = os.environ.copy()
    
    # 获取当前的 PATH 列表
    current_paths = custom_env.get("PATH", "").split(os.pathsep)
    
    # 1. 自动过滤掉所有包含 "MiKTeX" 的路径，防止鸠占鹊巢
    cleaned_paths = [p for p in current_paths if "miktex" not in p.lower()]
    
    # 2. 如果用户指定了 TeX Live 的路径，将其强行插入到 PATH 的最前面！
    if texlive_bin_dir:
        if os.path.exists(texlive_bin_dir):
            cleaned_paths.insert(0, texlive_bin_dir)
            print(f"🔧 已强制置顶 TeX Live 路径: {texlive_bin_dir}")
        else:
            print(f"⚠️  警告: 指定的 TeX Live 路径不存在: {texlive_bin_dir}")

    # 将清洗后的 PATH 重新组装回环境变量中
    custom_env["PATH"] = os.pathsep.join(cleaned_paths)
    # =============================================================

    steps = [
        ["pdflatex", "-interaction=nonstopmode", "-file-line-error", main_filename],
        ["bibtex", base_name],
        ["pdflatex", "-interaction=nonstopmode", "-file-line-error", main_filename],
        ["pdflatex", "-interaction=nonstopmode", "-file-line-error", main_filename]
    ]

    print(f"🚀 开始强制编译 [{directory}] 下的 {main_filename} ...\n")

    for i, cmd in enumerate(steps, 1):
        print(f"--- 正在执行步骤 {i}/{len(steps)}: {' '.join(cmd)} ---")
        try:
            # 注意这里传入了 env=custom_env
            result = subprocess.run(
                cmd, 
                cwd=directory,
                env=custom_env,          # <--- 关键参数：使用清洗过的环境变量
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                print(f"⚠️  警告: 步骤 '{cmd[0]}' 遇到错误，强制继续！")
                errors = [line for line in result.stdout.split('\n') if "Error" in line or "!" in line]
                if errors:
                    print("   捕获到的主要报错片段:")
                    for err in errors[:3]: 
                        print(f"     {err.strip()}")
                print("   (忽略错误，继续...)\n")
                
        except FileNotFoundError:
            print(f"❌ 致命错误: 找不到命令 '{cmd[0]}'。")
            print("   说明系统 PATH 中找不到 TeX Live。请尝试使用 texlive_bin_dir 参数直接指定路径。")
            return False

    # 检查 PDF 是否生成
    pdf_file_path = os.path.join(directory, f"{base_name}.pdf")
    if os.path.exists(pdf_file_path):
        print(f"\n✅ 强制编译完成！已生成 PDF: {os.path.abspath(pdf_file_path)}")
        return True
    else:
        print("\n❌ 致命错误导致未能生成 PDF 文件。")
        return False
        
        
# compile_latex_project(directory=r"papers\\20260308_194954")
def remove_file(file_path):
    for file in os.listdir(file_path):
            if os.path.isfile(os.path.join(file_path, file)):
                try:
                    os.remove(os.path.join(file_path, file))
                    logger.info(f"Removed {file} from {file_path}")
                except:
                    logger.error(f"Failed to remove {file} from {file_path}")



# use shutil to copy files
def move_files(src, dst):
    for file in os.listdir(src):
        if os.path.isfile(os.path.join(src, file)):
            try:
                shutil.copy2(os.path.join(src, file), os.path.join(dst, file))
                logger.info(f"Copied {file} from {src} to {dst}")
            except:
                logger.error(f"Failed to copy {file} from {src} to {dst}")
    pass
    
    
# test json output
json_data = """
```json
{
    "Thoughts": "Let me analyze the current situation. The reviewer has 6 main comments:\n\n1. **Network structure incorrect**: Replace Transformer-only with GNN+Transformer hybrid ✓ (new_detector.py created with GNNLayer + TransformerEncoder)\n2. **Expand quantization to include channel coefficients**: ✓ (DualAdaptiveQuantizer quantizes both demod results and channel coefficients)\n3. **Joint training, must beat C-MMSE**: ✗ The current results show the GNN-Trans BER is WORSE than Dist-Full at all SNR points! At 0dBm: Dist-Full=3.69% vs GNN-Trans=5.65%. This is a serious problem - the model is not performing well enough.\n4. **Modify paper**: Not yet done\n5. **Paper length >10 pages, 5-6 figures with 5+ curves each**: Not yet done\n6. **Ablation experiments**: Not yet done\n\nThe critical issue right now is that the proposed method performs WORSE than even the simple Dist-Full baseline. This needs to be fixed before anything else. Let me analyze why:\n\n1. The GNN layer with L×L pairwise attention is very memory-intensive (16×16=256 pairs per user per batch). With B=256, L=16, K=8, D=64, the GNN creates tensors of shape (256, 16, 16, 8, 64) which is huge.\n\n2. The model seems to be learning to allocate mostly channel bits (18 out of 20) and very few demod bits. This might be suboptimal.\n\n3. The Phase 2 training shows the validation BER actually getting WORSE over time (from 0.0435 at epoch 10 to 0.0553 at epoch 80). The quantization is hurting performance.\n\n4. The 20 bits/link target might be too aggressive. With N=4, even 2-bit channel quantization costs 2*4*2=16 bits per link. \n\nLet me think about how to fix this:\n- The GNN is too complex and might be overfitting or training instability. I should simplify it.\n- Need more Phase 1 epochs to get a better initialization\n- The c_target of 20 might need adjustment - let me compute: C-MMSE uses full precision which is 320 bits/link. Even at 20 bits/link we're at 16x compression. Maybe we should try a higher bit budget like 24 or 32 bits to get better performance first.\n- The Phase 2 training seems to degrade the detector. This could be because the quantization noise is too severe.\n\nLet me modify the code to:\n1. Simplify the GNN (use simpler aggregation instead of L×L pairwise attention which is O(L²))\n2. Train longer in Phase 1 (40 epochs) \n3. Try a higher bit budget (24 bits/link)\n4. Use a warmup strategy in Phase 2 where we gradually increase quantization noise\n\nActually, let me first check what C-MMSE achieves from the recorded baselines data to set a proper target:\n- C-MMSE at 0dBm: 0.02388, at 10dBm: 0.00025\n- Dist-Full at 0dBm: 0.03037, at 10dBm: 0.00825\n\nSo C-MMSE is much better than Dist-Full. Our proposed method needs to beat C-MMSE which is a high bar. The GNN+Transformer with quantized inputs should ideally leverage the channel information to do better MMSE-like detection at the CPU.\n\nThe key insight is that with quantized H and quantized ŝ, the CPU can actually perform a better detection than just averaging local LMMSE estimates. If the CPU has access to (quantized) H, it can do a more intelligent combining.\n\nLet me redesign the approach to be more practical:\n1. Simplify the GNN to avoid the O(L²) complexity issue\n2. Make the detector use the channel information more directly (like learned MMSE)\n3. Increase training epochs and fix the training instability",
    "Action": "PROMPT_CODER",
    "Action_Params": {
        "instruction": "We are redesigning the Cell-Free MIMO detection system. The current new_detector.py has a performance problem - the proposed GNN+Transformer method performs WORSE than the simple Dist-Full baseline. The main issues are:\n1. The GNN layer creates O(L²) pairwise attention which is too memory-intensive and causes training instability\n2. Phase 2 QAT training degrades the detector instead of improving it\n3. The detector doesn't leverage the channel information effectively enough\n\n**Existing code context:**\n- `system_model.py`: Cell-Free MIMO system with L=16, N=4, K=8\n- `lsq_quantizer.py`: LSQ differentiable quantizer\n- `new_detector.py`: Current implementation (needs major fixes)\n- `baselines.py`: Reference baselines including C-MMSE (BER ~0.024 at 0dBm, ~0.00025 at 10dBm)\n\n**Task: Completely rewrite `new_detector.py` with the following critical changes:**\n\n### Key Design Changes:\n\n1. **Replace the expensive GNN with a more efficient design:**\n   Instead of L×L pairwise attention (which is O(L²)), use a simpler but effective GNN:\n   - Each AP node has features (B, L, K, D)\n   - Message passing: compute mean/max of neighbor features, concatenate with self, and project\n   - Or better: use a single attention-based aggregation similar to GAT but without explicit pairwise computation:\n     * Compute node importance scores: score_l = MLP(h_l) -> scalar\n     * Softmax over L dimension\n     * Weighted sum: aggregated = sum_l(softmax(score_l) * h_l)\n   - This is O(L) instead of O(L²)\n   - Still do 2 layers of this\n\n2. **Better use of channel information - LMMSE-inspired architecture:**\n   The key insight is that with H and local estimates, the CPU can compute a better combining.\n   Add a \"learned MMSE\" branch:\n   - From quantized H (B,L,N,K,2), compute gram matrix features: H^H @ H approximation\n   - Use this as additional features for the detector\n   - The network should learn to mimic/improve upon centralized MMSE\n\n3. **Training stability improvements:**\n   - Phase 1: 30 epochs with only detector (no quantization)\n   - Phase 2: 70 epochs with joint training\n   - In Phase 2, use a linear warmup of lambda (bit penalty weight) from 0 to lambda_val over first 20 epochs\n   - Use gradient clipping (max_norm=1.0)\n   - Lower initial tau to 1.0, decay to 0.1\n   - Use a smaller learning rate for the quantizer (0.01) vs detector (0.001)\n\n4. **Better data generation:**\n   - Keep generate_data_batch_v2 as is (it works fine)\n   - But also compute and pass the centralized MMSE estimate as a training signal reference\n\n### Detailed Architecture for GNNTransformerDetector:\n\n```\nInput:\n  s_hat_q: (B, L, K, 2) - quantized local LMMSE estimates\n  H_q: (B, L, N, K, 2) - quantized channel coefficients  \n  bitwidth_features: (B, L, K, 2) - demod and channel bitwidths\n\nStep 1: Per-AP Feature Extraction\n  demod_feat = MLP(s_hat_q)  # (B, L, K, D)\n  # For channel: per user k, take H_q[:,:,:,k,:] which is (B,L,N,2)\n  # Reshape to (B, L, K, 2*N) then MLP -> (B, L, K, D)\n  channel_feat = MLP(H_q reshaped)  # (B, L, K, D)\n  # Also compute channel gram features for LMMSE approximation\n  # H_q_complex approx -> compute ||h_{l,k}||^2 as additional feature\n  combined = cat[demod_feat, channel_feat, bitwidth_features] -> MLP -> (B, L, K, D)\n\nStep 2: Efficient GNN (2 layers)\n  For each layer:\n    # Compute importance score per AP\n    score = MLP(h)  # (B, L, K, 1)\n    attn = softmax(score, dim=L)  # (B, L, K, 1)\n    # Global message = weighted sum of all nodes\n    global_msg = sum(attn * MLP_msg(h), dim=L)  # (B, K, D)\n    # Expand back and combine with local\n    global_msg_exp = global_msg.unsqueeze(1).expand(B, L, K, D)\n    h = LayerNorm(h + MLP_update(cat[h, global_msg_exp]))  # (B, L, K, D)\n\nStep 3: AP Aggregation\n  # Learned attention-based pooling over L dimension\n  attn_pool = softmax(MLP(h), dim=L)  # (B, L, K, 1)  \n  user_feat = sum(h * attn_pool, dim=L)  # (B, K, D)\n  # Also compute weighted soft output for residual\n  base_out = sum(s_hat_q * attn_pool, dim=L)  # (B, K, 2)\n\nStep 4: Transformer IC (2 layers, 4 heads)\n  ic_out = Transformer(user_feat)  # (B, K, D)\n\nStep 5: Output\n  residual = Linear(ic_out)  # (B, K, 2), zero-initialized\n  output = base_out + residual\n```\n\n### DualAdaptiveQuantizer:\nKeep the same as current but fix the policy network input to be more informative:\n- Input: [v_real, v_imag, snr, channel_norm, channel_mean_power] -> 5 dims\n- Two heads: demod bitwidth {0,2,4}, channel bitwidth {0,2,4}\n- Bit calculation: demod uses 2*b bits (complex), channel uses 2*N*b bits\n\n### Training Script:\n- Phase 1: 30 epochs, lr=1e-3, train detector only with full precision data\n- Phase 2: 70 epochs, lr=5e-4 for detector and 5e-3 for quantizer\n  - Lambda warmup: linearly increase from 0 to lambda_val over first 20 epochs of phase 2\n  - tau: exponential decay from 1.0 to 0.1\n  - c_target: 20 bits/link (this means ~16x compression from 320 bits full precision)\n- Evaluation at: [-10, -5, 0, 5, 10, 15, 20] dBm with 500 samples\n- Also compute and print C-MMSE BER for comparison during evaluation\n\n### For C-MMSE computation during evaluation:\nAdd a function that computes centralized MMSE BER from the test data:\n```python\ndef compute_cmmse_ber(H_np, s_hat_np, s_np, noise_w):\n    \"\"\"Compute centralized MMSE BER.\"\"\"\n    B, L, N, K = H_np.shape\n    total_err = 0\n    for b in range(B):\n        H_all = H_np[b].reshape(L*N, K)  # (L*N, K)\n        # We need y, but we don't have it directly.\n        # Actually, from s_hat (local LMMSE), we can't easily reconstruct y.\n        # Instead, let's compute MMSE from H and s directly.\n        # R = H^H H + sigma^2 I\n        R = H_all.conj().T @ H_all + noise_w * np.eye(K)\n        # But we need y = H_all @ s + noise\n        # Let's just generate it\n    # Actually this is complex. Let me just use the baselines.py approach.\n    # Better: modify generate_data_batch_v2 to also return y\n```\n\nActually, let's modify `generate_data_batch_v2` to also return the received signal y so we can compute C-MMSE.\n\n### Modified generate_data_batch_v2 returns:\n- s_hat: (B, L, K) complex\n- s: (B, K) complex\n- H: (B, L, N, K) complex\n- local_snr: (B, L, K) real\n- y: (B, L, N) complex (received signal)\n\n### C-MMSE evaluation function:\n```python\ndef compute_cmmse_ber_from_data(H_np, y_np, s_np, noise_w):\n    B, L, N, K = H_np.shape\n    total_err = 0\n    total_bits = B * K * 2\n    for b in range(B):\n        H_all = H_np[b].reshape(L*N, K)\n        y_all = y_np[b].reshape(L*N)\n        R = H_all.conj().T @ H_all + noise_w * np.eye(K)\n        s_hat = np.linalg.inv(R) @ H_all.conj().T @ y_all\n        err_re = np.sign(s_np[b].real) != np.sign(s_hat.real)\n        err_im = np.sign(s_np[b].imag) != np.sign(s_hat.imag)\n        total_err += err_re.sum() + err_im.sum()\n    return total_err / total_bits\n```\n\nThis is slow but OK for evaluation.\n\n### CRITICAL print requirements:\n- Print phase 1 training progress every 5 epochs (loss, val BER)\n- Print phase 2 training progress every 5 epochs (loss, MSE, bits, val BER, tau)\n- Print final evaluation table with columns: p_tx, Dist-Full BER, C-MMSE BER, Proposed BER, Improve vs C-MMSE (%), Avg Total Bits\n- Print bit allocation statistics for 3 power points\n\nPlease completely rewrite new_detector.py. Make sure the code is correct and efficient.",
        "summary": "The first run of new_detector.py showed that the proposed GNN+Transformer method performs WORSE than even the simple Dist-Full baseline at all SNR points. For example, at 0dBm: Dist-Full BER=3.69% vs GNN-Trans BER=5.65%, at 10dBm: 0.84% vs 2.19%. This is because: (1) the O(L²) GNN pairwise attention is too memory-intensive and causes training instability, (2) Phase 2 QAT training degrades the detector (val BER went from 0.0435 at epoch 10 to 0.0553 at epoch 80), (3) the detector doesn't leverage channel information effectively. Now prompting Coder to completely rewrite new_detector.py with a more efficient GNN design (O(L) instead of O(L²)), better LMMSE-inspired architecture, and improved training stability measures including lambda warmup and separate learning rates for quantizer vs detector."
    }
}
```
"""
import re
import json_repair

# loaded = json_repair.loads(json_data)
def robust_extract_json(text):
        """鲁棒的 JSON 提取器，完美处理 LLM 输出的各类 LaTeX 公式和非法转义符"""
        try:
            return json_repair.loads(text)
        except Exception as e:
            pass
        
        json_pattern = r"```json(.*?)```"
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not matches:
            json_pattern = r"\{.*?\}"
            matches = re.findall(json_pattern, text, re.DOTALL)
            
        for json_string in matches:
            
            json_string = json_string.strip()
            
            # 【终极修复】逐个检查所有的反斜杠及其后面的字符
            def fix_escape(m):
                val = m.group(0)
                # 如果是合法的 JSON 转义序列，原样保留
                if val in ['\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t']:
                    return val
                if val.startswith('\\u') and len(val) == 6:
                    return val
                # 如果是非法的（比如 \p, \l, \m, \| 等），额外添加一个反斜杠将其转义为字面量
                return '\\' + val

            # 正则匹配 \uXXXX 或者 \ 加任意单个字符，或者在结尾的 \
            json_string = re.sub(r'\\u[0-9a-fA-F]{4}|\\.|\\$', fix_escape, json_string)
            try: 
                return json_repair.loads(text)
            except Exception as e:
                try:
                    # strict=False 允许字符串内部直接包含物理换行符
                    return json.loads(json_string, strict=False)
                except json.JSONDecodeError:
                    try:
                        # 兜底：清除非法的 ASCII 控制字符
                        json_string_clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", json_string)
                        return json.loads(json_string_clean, strict=False)
                    except json.JSONDecodeError:
                        continue
        return None
        
        
# loaded =robust_extract_json(json_data)
# print(loaded)
# action = loaded.get('Action')
# action_params = loaded.get('Action_Params')
# print(action)
# print(action_params.get('instruction'))
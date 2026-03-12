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
    "Thoughts": "Let me analyze the reviewer's comments and what has been done:\n\n**Reviewer Comments:**\n1. The new_detector_v3.py only transmits quantized local demod results and channel coefficients, but NOT the AP received signal y. Need to create new code that transmits y, H, and local demod results with triple adaptive quantization. The detector must significantly outperform Dist-Full (close to one order of magnitude BER reduction) and approach full-precision C-MMSE.\n2. Need results for both perfect CSI and imperfect CSI.\n3. Need stronger baselines (hybrid quantization + centralized detection).\n4. Need to update all paper sections and figures (at least 6 figures with 5+ curves each).\n5. Ensure all tex code compiles correctly.\n\n**What has been done so far in this rebuttal:**\n- Read new_detector_v3.py, system_model.py, and lsq_quantizer.py to understand the existing architecture.\n\n**What needs to be done:**\n1. Design and implement new_detector_v4.py that transmits y, H, and local demod with triple adaptive quantization\n2. Train and evaluate it\n3. Get imperfect CSI results\n4. Add stronger baselines\n5. Update all paper tex files\n\nLet me start by designing the new detector. The key insight from the reviewer is that transmitting the raw received signal y is critical because it contains information that local LMMSE processing may lose. With y, H, and local demod all available at the CPU, the CPU can perform much more sophisticated detection.\n\nThe system has L=16 APs, N=4 antennas/AP, K=8 users. For each AP-user link:\n- y: N complex values = 2N=8 real values (shared across users, so per AP it's 8 real values)\n- H: N×K complex values = 2NK real values per AP (but per user link it's 2N=8 real values)\n- local demod: K complex values = 2K real values per AP (per user it's 2 real values)\n\nWait, y is per AP (not per user-link), so the quantization of y should be done per AP, not per AP-user link. Let me think about this more carefully.\n\nActually, for the fronthaul, each AP transmits:\n- y_l: N complex = 2N real values (this is the same for all users)\n- H_l: N×K complex = 2NK real values\n- s_hat_l: K complex = 2K real values\n\nSo total real values per AP = 2N + 2NK + 2K = 2N(1+K) + 2K = 2*4*(1+8) + 2*8 = 72 + 16 = 88 real values.\n\nWith 4 bits each, that's 88*4 = 352 bits per AP. Per user link: 352/8 = 44 bits.\n\nThe reviewer suggests at least 192 bits per AP-user link for first run. With 8 users, that's 192*8 = 1536 bits per AP.\n\nLet me think about the bit budget differently. Per AP-user link:\n- y contribution: 2N/K per user? No, y is shared. \n\nActually, let me reconsider. The quantization policy should be:\n- For y: per AP, choose a bit width for quantizing the 2N real values of y_l\n- For H: per AP-user link, choose a bit width for quantizing the 2N real values of H_{l,k}\n- For demod: per AP-user link, choose a bit width for quantizing the 2 real values of s_hat_{l,k}\n\nTotal bits per AP = 2N * b_y + K * (2N * b_H_k + 2 * b_s_k)\n\nOr we can simplify and have per-AP policies for all three.\n\nLet me design this carefully. I'll instruct the coder to create new_detector_v4.py.",
    "Action": "PROMPT_CODER",
    "Action_Params": {
        "instruction": "We are building an improved Cell-Free MIMO fronthaul compression and detection system. The key improvement over the previous version (new_detector_v3.py) is that we now transmit THREE types of information from each AP to the CPU: (1) the raw received signal y, (2) channel coefficients H, and (3) local LMMSE demodulation results, all with adaptive quantization. The CPU then uses a powerful GNN+Transformer detector to perform joint detection.\n\n**Existing code files you should reference:**\n- `system_model.py`: Contains `CellFreeSystem` class for generating scenarios and channels\n- `lsq_quantizer.py`: Contains `LSQQuantizer` class for learned step-size quantization\n- `new_detector_v3.py`: Previous version that only transmits H and demod results (use as a starting template)\n\n**Create a new file: `new_detector_v4.py`**\n\n## System Parameters:\n- L=16 APs, N=4 antennas/AP, K=8 single-antenna users\n- QPSK modulation\n- 3GPP Dense Urban channel model (from CellFreeSystem)\n\n## Architecture Design:\n\n### 1. Data Generation (modify `generate_data_batch_v2`):\nReturn: s_hat (B,L,K) complex, s (B,K) complex, H (B,L,N,K) complex, local_snr (B,L,K), y (B,L,N) complex\n(Same as in v3, just make sure y is returned)\n\n### 2. Triple Adaptive Quantizer:\nFor each AP l, the quantizer decides bit widths for three types of data:\n\n**a) Received signal y_l:** Shape (N,) complex = 2N real values per AP. Bit options: {0, 2, 4, 6, 8} bits per real value. So bits for y per AP = 2N * b_y. This is shared across all users.\n\n**b) Channel H_{l,:,k}:** Shape (N,) complex per user = 2N real values per (AP, user) link. Bit options: {0, 2, 4, 6, 8}. Bits per link = 2N * b_H.\n\n**c) Local demod s_hat_{l,k}:** Shape (1,) complex per user = 2 real values per (AP, user) link. Bit options: {0, 2, 4, 6, 8}. Bits per link = 2 * b_s.\n\nThe policy network takes as input features: [local_snr, channel_norm, interference_features, signal_power, etc.] and outputs Gumbel-Softmax probabilities for each of the three quantizers.\n\nFor y, since it's per-AP (not per-user), the policy can be based on AP-level features (e.g., total received power, average SNR across users).\n\nFor H and demod, the policy is per (AP, user) link.\n\n**Total bits per AP** = 2N * b_y + sum_k(2N * b_H_k + 2 * b_s_k)\n\n**Bit budget constraint:** Average total bits per AP <= C_target_per_ap. Set C_target_per_ap as a command-line argument. For the first run, use C_target_per_ap = 1536 (which gives each of the 88 real values about 17.5 bits... that's too much). \n\nActually, let me reconsider. Let's define the constraint as average bits per AP-user link:\ntotal_bits_per_ap / K = c_target_per_link\n\nFor the first run with generous budget: each AP-user link has:\n- y contribution: 2N * b_y / K = 8 * b_y / 8 = b_y per real value of y (shared)\n- H: 2N * b_H = 8 * b_H per link  \n- demod: 2 * b_s per link\n\nWith 4 bits average: 2N*4/K + 2N*4 + 2*4 = 8*4/8 + 32 + 8 = 4 + 32 + 8 = 44 bits per link.\n\nLet's use c_target = 192 bits per AP-user link for the first generous run. Full precision would be: 2N*32/K + 2N*32 + 2*32 = 32 + 256 + 64 = 352 bits per link.\n\nSo compression ratio = 352/192 ≈ 1.83x. That's quite generous, good for first run.\n\n### 3. CPU-side Detector (GNN + Transformer):\n\nThe detector receives quantized versions of y, H, and demod from all APs, plus bitwidth metadata.\n\n**Input processing:**\n- For each AP l and user k, construct a feature vector that includes:\n  - Quantized demod: s_hat_q_{l,k} (2 real values)\n  - Quantized channel: H_q_{l,:,k} (2N real values) \n  - Quantized received signal: y_q_l (2N real values, shared across users)\n  - Bitwidth features: normalized bit widths for y, H, demod (3 values)\n  - Local SNR feature (1 value)\n  - Interference features (2 values)\n\nTotal input per (l,k) node: 2 + 2N + 2N + 3 + 1 + 2 = 2 + 8 + 8 + 3 + 1 + 2 = 24 values (for N=4)\n\n**Architecture:**\n1. Input MLP: 24 -> hidden_dim (128)\n2. Mean-field GNN layers (3 layers) for AP-level message passing\n3. Attention-based AP aggregation\n4. Transformer IC layers (2 layers) for inter-user interference cancellation\n5. Output head: hidden_dim -> 2 (real, imag of detected symbol)\n\nIMPORTANT: Add a residual connection from quantized demod to output (weighted sum baseline).\n\nAlso add a **re-estimation branch**: use quantized y and H to compute a refined LMMSE estimate at the CPU:\n- For each user k, concatenate all APs' quantized y and H, and use a small MLP to produce a refined estimate\n- This refined estimate is combined with the GNN output\n\n### 4. Training Strategy:\n\n**Phase 1 (50 epochs):** Pre-train detector only (no quantization, full precision inputs). Use MSE loss. LR=1e-3 with cosine annealing.\n\n**Phase 2 (150 epochs, 3 sub-phases):** Joint QAT training.\n- Sub-phase 1 (30 epochs): lambda=0.001, tau: 5.0->2.0 (warm start)\n- Sub-phase 2 (60 epochs): lambda=0.01, tau: 2.0->0.5\n- Sub-phase 3 (60 epochs): lambda=0.1, tau: 0.5->0.1\n\nLoss = MSE(detected, true_symbols) + lambda * (avg_bits - c_target)^2\n\n### 5. Baselines to compute:\n- **Dist-Full**: Mean pooling of local LMMSE (no quantization)\n- **C-MMSE**: Full centralized MMSE (full precision H and y)\n- **C-MMSE-Q**: Centralized MMSE with uniform quantization at same total bit budget\n- **Dist-Q**: Local LMMSE with uniform quantization, then mean pooling\n- **LSFD-Q**: Local LMMSE with uniform quantization, then LSFD weighting\n\nFor C-MMSE-Q: uniformly allocate the same total bits across all quantities (y, H) using the same per-real bit width.\n\nFor LSFD-Q: use large-scale fading decoding weights: w_lk = beta_lk / sum_l(beta_lk), then weighted sum of quantized local LMMSE.\n\n### 6. Evaluation:\nTest at P_tx = [-10, -5, 0, 5, 10, 15, 20] dBm with 500 samples each.\nPrint detailed BER comparison table.\nPrint bit allocation statistics.\n\n### 7. Command-line arguments:\n```\n--L 16 --N 4 --K 8 --epochs_phase1 50 --phase2_sub1_epochs 30 --phase2_sub2_epochs 60 --phase2_sub3_epochs 60\n--batches_per_epoch 100 --batch_size 256 --hidden_dim 128 --num_gnn_layers 3 --num_transformer_layers 2 --num_heads 4\n--c_target 192.0 --lr_phase1 1e-3 --lr_phase2 5e-4 --test_samples 500 --seed 42\n--grad_clip 5.0 --dropout 0.1\n```\n\n### 8. Key implementation details:\n\n- The c_target is in bits per AP-user link. Total bits per AP = c_target * K.\n- For y quantization, since y is shared across users, the policy can output one decision per AP (not per AP-user). The bits for y are then divided equally among K users when computing per-link budget.\n- Use 5 bit options: {0, 2, 4, 6, 8} for all three quantizers (0 means don't transmit).\n- The LSQ quantizers from lsq_quantizer.py should be used for each bit level.\n- For 0-bit quantization, just output zeros.\n- Gumbel-Softmax with hard=True for forward pass, soft gradients for backward.\n\n### 9. CRITICAL for performance:\n- The detector MUST be able to use raw y and H to perform re-estimation at the CPU. This is the key advantage over v3.\n- Include a centralized re-estimation module that takes quantized y and H from ALL APs and computes a refined MMSE-like estimate using learned weights.\n- The hidden_dim should be 128 or larger.\n- Use batch normalization after key layers.\n- The model should save as 'new_joint_model_v4.pth'.\n\n### 10. Print statements:\nMake sure to print:\n- Training progress every 5 epochs (loss, BER at 0/10/20 dBm, avg bits, bit distribution)\n- Final BER table for all power points and all baselines\n- Bit allocation statistics at multiple power points\n- Model parameter count\n- Compression ratio\n\nPlease create the complete `new_detector_v4.py` file.",
        "summary": "Designing and instructing coder to create new_detector_v4.py that implements triple adaptive quantization (y, H, demod) with a GNN+Transformer detector at the CPU. This addresses reviewer comment 1 about transmitting the raw received signal y in addition to H and demod results. The initial bit budget is set generously at 192 bits per AP-user link (vs 352 full precision) to ensure the detector can achieve significant BER gains over Dist-Full."
    }
}
```  
\n
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
        
# with open('resp_temp.txt','r',encoding='utf-8',errors='ignore') as f:
#      json_data = f.read()
# loaded =robust_extract_json(json_data)
# print(isinstance(loaded,list))
# print(isinstance(loaded[0],dict))
# print(isinstance(loaded[1],dict))
# print(loaded)
# action = loaded[1].get('Action')
# action_params = loaded[1].get('Action_Params')
# print(action)
# print(action_params.get('instruction'))
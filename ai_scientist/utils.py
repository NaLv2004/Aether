import logging
import colorlog
import os

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
    
    
import http.client
import json
import base64
import os

class PDFReader:
    def __init__(self, api_key, system_prompt, context_window_size=1, host="jeniya.top", model="gemini-3-pro-preview"):
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
            
            
if __name__ == "__main__":
    # 1. 准备你的配置
    YOUR_API_KEY = os.getenv("JIANYI_API_KEY") 
    # YOUR_API_KEY = "sk-xxxxxxxxxxxxxxxxx" # 替换为你的真实 API Key / Token
    SYS_PROMPT = "请总结以下论文。"
    
    # 2. 初始化类，上下文窗口设置为 2（记忆上一轮对话）
    # host 默认是 jeniya.top，如果你用官方或其他代理，可以在此处修改 host="generativelanguage.googleapis.com"
    reader = PDFReader(
        api_key=YOUR_API_KEY, 
        system_prompt=SYS_PROMPT, 
        context_window_size=2
    )

    # 3. 第一次请求：要求总结
    pdf_file = r"pdfs_ieee\\10.1109_TCCN.2017.2758370.pdf"      # 本地的 PDF 路径
    output_file = "test.txt"   # 准备写入的文本文件路径
    
    reader.read_pdf(
        pdf_path=pdf_file, 
        output_txt_path=output_file, 
        user_prompt="请帮详细总结这篇论文，并写出式（4）是什么。"
    )

    # 4. 第二次请求：基于上一次的记忆追问 (测试上下文)
    # 此时因为 context_window_size 设定为 2，它会记得上一次它回答的那 3 个要点
    # reader.read_pdf(
    #     pdf_path=pdf_file, 
    #     output_txt_path=output_file, 
    #     user_prompt="针对你刚才列出的第2个要点，在这份文档中有什么具体的数据支撑吗？"
    # )
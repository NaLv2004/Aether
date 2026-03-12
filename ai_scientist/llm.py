import json
import os
import re
import datetime

import anthropic
import backoff
import openai
import json_repair
MAX_NUM_TOKENS = 65000

class LLMAgent:
    def __init__(self, model, temperature=1, log_file="llm_agent.log"):
        """
        初始化 LLMAgent 实例。
        
        :param model: 所选用的 LLM 模型名称。
        :param temperature: 生成文本的 temperature。
        :param log_file: 记录日志的文件路径。
        """
        self.raw_model_name = model
        self.temperature = temperature
        self.log_file = log_file
        self.msg_history = []
        self.context_window_len = -1  # 默认不限制上下文长度
        
        # 创建客户端及获取实际的 client_model 名称
        self.client, self.client_model = self._create_client(model)
        
        # 初始化日志文件
        self._log_event(f"LLMAgent initialized with model: {self.raw_model_name} (Client Model: {self.client_model})")

    def set_context_len(self, length):
        """
        设置上下文窗口大小。
        
        :param length: 保留最近的对话轮数。
                       -1 表示保留所有历史对话（默认）；
                       >0 表示保留最近的 length 轮对话。
        """
        self.context_window_len = length
        self._log_event(f"Context window length set to: {self.context_window_len}")

    def _trim_history(self):
        """根据 context_window_len 裁剪 msg_history"""
        if self.context_window_len > 0:
            # 一轮对话通常包含 User 和 Assistant 两条消息
            max_msgs = self.context_window_len * 2
            if len(self.msg_history) > max_msgs:
                self.msg_history = self.msg_history[-max_msgs:]

    def _log_event(self, content):
        """向指定的日志文件追加写入日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {content}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def _log_interaction(self, role, message):
        """记录对话交互"""
        self._log_event(f"--- {role.upper()} ---\n{message}\n")

    def clear_history(self):
        """清空当前的对话上下文"""
        self.msg_history = []
        self._log_event("Message history cleared.")

    def _create_client(self, model):
        """根据模型名称创建对应的 API Client"""
        if model.startswith("claude-"):
            print(f"Using CLAUDE API with {model}.")
            return openai.OpenAI(
                api_key = os.environ.get("CLAUDE_API_KEY"),
                #base_url="https://newapi.baosiapi.com/v1"
                base_url = "https://jeniya.top/v1"
            ), model
        elif model.startswith("bedrock") and "claude" in model:
            client_model = model.split("/")[-1]
            print(f"Using Amazon Bedrock with model {client_model}.")
            return anthropic.AnthropicBedrock(), client_model
        elif model.startswith("vertex_ai") and "claude" in model:
            client_model = model.split("/")[-1]
            print(f"Using Vertex AI with model {client_model}.")
            return anthropic.AnthropicVertex(), client_model
        elif 'gpt' in model or "o1" in model or "o3" in model:
            print(f"Using OpenAI API with model {model}.")
            return openai.OpenAI(), model
        elif model in ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]:
            print(f"Using OpenAI API with {model}.")
            return openai.OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            ), model
        elif model == "llama3.1-405b":
            print(f"Using OpenAI API with {model}.")
            return openai.OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            ), "meta-llama/llama-3.1-405b-instruct"
        elif "gemini" in model:
            print(f"Using OpenAI API with {model}.")
            return openai.OpenAI(
                api_key = os.environ.get("JIANYI_API_KEY"),
                #base_url="https://newapi.baosiapi.com/v1"
                base_url = "https://jeniya.top/v1"
            ), model
        else:
            raise ValueError(f"Model {model} not supported.")

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
    def get_response(self, msg, system_message, print_debug=False):
        """
        发送单条消息并获取响应，自动维护上下文历史和日志记录。
        """
        # 在添加新消息前，先根据上下文窗口裁剪历史记录
        self._trim_history()
        
        self._log_interaction("SYSTEM MESSAGE", system_message)
        self._log_interaction("USER", msg)

        if "xxxxx" in self.client_model:
            self.msg_history.append({
                "role": "user",
                "content": [{"type": "text", "text": msg}]
            })
            response = self.client.messages.create(
                model=self.client_model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=self.temperature,
                system=system_message,
                messages=self.msg_history,
            )
            content = response.content[0].text
            self.msg_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": content}]
            })

        elif 'gpt' in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                seed=0,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        elif "o1" in self.client_model or "o3" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "user", "content": system_message},
                    *self.msg_history,
                ],
                temperature=1,
                max_completion_tokens=MAX_NUM_TOKENS,
                n=1,
                seed=0,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["deepseek-chat", "deepseek-coder"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["deepseek-reasoner"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        elif "gemini" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})
            
        elif "claude" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
            )
            content = response.choices[0].message.content
            self.msg_history.append({"role": "assistant", "content": content})

        else:
            raise ValueError(f"Model {self.client_model} not supported.")

        self._log_interaction("ASSISTANT", content)

        if print_debug:
            print("\n" + "*" * 20 + " LLM START " + "*" * 20)
            for j, m in enumerate(self.msg_history):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21 + "\n")

        return content, self.msg_history
        
        
    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
    def get_response_stream(self, msg, system_message, print_debug=False):
        """
        发送单条消息并获取响应，自动维护上下文历史和日志记录。
        增加了 timeout 限制和流式 (stream) 控制台输出。
        """
        # 在添加新消息前，先根据上下文窗口裁剪历史记录
        self._trim_history()

        self._log_interaction("SYSTEM MESSAGE", system_message)
        self._log_interaction("USER", msg)

        if "xxxxx" in self.client_model:
            self.msg_history.append({
                "role": "user",
                "content": [{"type": "text", "text": msg}]
            })
            response = self.client.messages.create(
                model=self.client_model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=self.temperature,
                system=system_message,
                messages=self.msg_history,
                stream=True,       # 开启流式输出
                timeout=600.0      # 增加 Timeout 限制 (600秒)
            )
            content = ""
            for chunk in response:
                if chunk.type == "content_block_delta":
                    text = chunk.delta.text
                    print(text, end="", flush=True)  # 控制台实时打印
                    content += text
            print() # 输出完毕后换行
            
            self.msg_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": content}]
            })

        elif 'gpt' in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                seed=0,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        elif "o1" in self.client_model or "o3" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "user", "content": system_message},
                    *self.msg_history,
                ],
                temperature=1,
                max_completion_tokens=MAX_NUM_TOKENS,
                n=1,
                seed=0,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["deepseek-chat", "deepseek-coder"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        elif self.client_model in ["deepseek-reasoner"]:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                n=1,
                stop=None,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    # 对于 reasoner 模型，我们依然只提取 content（不提取 reasoning_content），以保持原逻辑一致
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        elif "gemini" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})
            
            
        elif "claude" in self.client_model:
            self.msg_history.append({"role": "user", "content": msg})
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *self.msg_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stream=True,
                timeout=600.0
            )
            content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        content += delta_content
            print()
            self.msg_history.append({"role": "assistant", "content": content})

        else:
            raise ValueError(f"Model {self.client_model} not supported.")

        self._log_interaction("ASSISTANT", content)

        if print_debug:
            print("\n" + "*" * 20 + " LLM START " + "*" * 20)
            for j, m in enumerate(self.msg_history):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21 + "\n")
        
        try:
            with open('resp_temp.txt', 'w', encoding='utf-8') as f:
                f.write(content)
                f.close()
            with open('resp_temp.txt','r', encoding='utf-8',errors='ignore') as ff:
                content = ff.read()
                ff.close()
        except:
            pass
        return content, self.msg_history

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
    def get_batch_responses(self, msg, system_message, n_responses=1, print_debug=False):
        """
        获取批量的 LLM 响应 (Ensembling)。
        注意：这会分叉产生多个返回上下文，不会自动修改主进程的 `self.msg_history` 状态以防止冲突。
        """
        # 在处理前，先根据上下文窗口裁剪历史记录
        self._trim_history()

        self._log_interaction(f"USER (BATCH request, n={n_responses})", msg)
        
        if 'gpt' in self.client_model:
            temp_history = self.msg_history + [{"role": "user", "content": msg}]
            response = self.client.chat.completions.create(
                model=self.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *temp_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
                seed=0,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                temp_history + [{"role": "assistant", "content": c}] for c in content
            ]
        elif self.client_model == "llama-3-1-405b-instruct":
            temp_history = self.msg_history + [{"role": "user", "content": msg}]
            response = self.client.chat.completions.create(
                model="meta-llama/llama-3.1-405b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    *temp_history,
                ],
                temperature=self.temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
            )
            content = [r.message.content for r in response.choices]
            new_msg_history = [
                temp_history + [{"role": "assistant", "content": c}] for c in content
            ]
        else:
            # Fallback：逐个请求
            content, new_msg_history = [], []
            for _ in range(n_responses):
                # Temporarily save history to restore later, to prevent accumulating n copies
                saved_history = list(self.msg_history)
                c, hist = self.get_response(msg, system_message, print_debug=False)
                content.append(c)
                new_msg_history.append(hist)
                self.msg_history = saved_history

        self._log_interaction("ASSISTANT (BATCH responses)", str(content))

        if print_debug:
            print("\n" + "*" * 20 + " LLM BATCH START " + "*" * 20)
            for j, m in enumerate(new_msg_history[0]):
                print(f'{j}, {m["role"]}: {m["content"]}')
            print(content)
            print("*" * 21 + " LLM BATCH END " + "*" * 21 + "\n")

        return content, new_msg_history

    @staticmethod
    def extract_json_between_markers(llm_output):
        """
        静态辅助方法：从大模型输出中提取被 ```json 和 ``` 包裹的 JSON 内容
        """
        json_pattern = r"```json(.*?)```"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

        if not matches:
            # Fallback: Try to find any JSON-like content in the output
            json_pattern = r"\{.*?\}"
            matches = re.findall(json_pattern, llm_output, re.DOTALL)

        for json_string in matches:
            json_string = json_string.strip()
            try:
                parsed_json = json.loads(json_string)
                return parsed_json
            except json.JSONDecodeError:
                # Attempt to fix common JSON issues
                try:
                    # Remove invalid control characters
                    json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                    parsed_json = json.loads(json_string_clean)
                    return parsed_json
                except json.JSONDecodeError:
                    continue  # Try next match

        return None
        
    @staticmethod
    def robust_extract_json_list(text):
        """
        专门用于提取 JSON List 的鲁棒提取器。
        解决原 robust_extract_json 在没遇到 markdown 标记时只提取第一个 dict 的问题。
        """
        # 1. 优先尝试提取 Markdown 代码块
        json_pattern = r"```json(.*?)```"
        matches = re.findall(json_pattern, text, re.DOTALL)

        # 2. 如果没有 Markdown，尝试找最外层的 [...] 结构
        if not matches:
            # 使用 find/rfind 寻找第一个 [ 和最后一个 ]，这比正则能更好地捕获跨行的大列表
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # 提取整个可能的列表字符串
                matches = [text[start_idx : end_idx + 1]]
            else:
                return None

        for json_string in matches:
            json_string = json_string.strip()

            # --- 复用原有的修复逻辑 (核心) ---
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
                # strict=False 允许字符串内部直接包含物理换行符
                parsed = json.loads(json_string, strict=False)
                # 关键：只有当解析结果确实是 list 时才返回
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                try:
                    # 兜底：清除非法的 ASCII 控制字符
                    json_string_clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", json_string)
                    parsed = json.loads(json_string_clean, strict=False)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None
        
   
    @staticmethod
    def robust_extract_json(text):
        """鲁棒的 JSON 提取器，完美处理 LLM 输出的各类 LaTeX 公式和非法转义符"""
        try:
            loaded = json_repair.loads(text)
            if isinstance(loaded, dict):
                return loaded
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict):
                        return item
                        break
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
    # def robust_extract_json(text):
    #     """鲁棒的 JSON 提取器，完美处理 LLM 输出的各类 LaTeX 公式和非法转义符"""
    #     json_pattern = r"```json(.*?)```"
    #     matches = re.findall(json_pattern, text, re.DOTALL)
        
    #     if not matches:
    #         json_pattern = r"\{.*?\}"
    #         matches = re.findall(json_pattern, text, re.DOTALL)
            
    #     for json_string in matches:
    #         json_string = json_string.strip()
            
    #         # 【终极修复】逐个检查所有的反斜杠及其后面的字符
    #         def fix_escape(m):
    #             val = m.group(0)
    #             # 如果是合法的 JSON 转义序列，原样保留
    #             if val in ['\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t']:
    #                 return val
    #             if val.startswith('\\u') and len(val) == 6:
    #                 return val
    #             # 如果是非法的（比如 \p, \l, \m, \| 等），额外添加一个反斜杠将其转义为字面量
    #             return '\\' + val

    #         # 正则匹配 \uXXXX 或者 \ 加任意单个字符，或者在结尾的 \
    #         json_string = re.sub(r'\\u[0-9a-fA-F]{4}|\\.|\\$', fix_escape, json_string)
            
    #         try:
    #             # strict=False 允许字符串内部直接包含物理换行符
    #             return json.loads(json_string, strict=False)
    #         except json.JSONDecodeError:
    #             try:
    #                 # 兜底：清除非法的 ASCII 控制字符
    #                 json_string_clean = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", json_string)
    #                 return json.loads(json_string_clean, strict=False)
    #             except json.JSONDecodeError:
    #                 continue
    #     return None
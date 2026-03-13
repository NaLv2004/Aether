# accepts review opinions, and updates the entire project

import json
import os
import re
import argparse
import datetime
import subprocess
import time
import threading
import queue

from llm import LLMAgent
from utils import setup_logger

# 导入文献搜索和阅读功能
try:
    from generate_ideas import (
        search_for_papers,
        format_search_results_and_update_map,
        process_papers_to_read,
        read_knowledge_base
    )
except ImportError:
    pass

logger = setup_logger("experiment_run.log")

# ==========================================
# 1. 配置与全局常量
# ==========================================
CONDA_ENV_NAME = "AutoGenOld"  # 指定的 Conda 环境名称
MAX_RETRIES = 500            # 每一步最大失败重试次数，超过则触发 Git 回退机制

# ==========================================
# 2. 系统提示词定义 (System Prompts)
# ==========================================

# 你是一个科研项目管家 (Orchestrator Agent)。你的任务是浏览当前工作目录下的科研项目（当前文件夹下包含项目的idea、plan，前期编写好的代码和运行结果，以及已经写好的论文的tex文件，和审稿人的意见），并根据审稿人的意见修改论文
# 内容（如果涉及获取新的数据，添加新的baseline，改进现有方法等，则需要提示coder编写新的代码来获取新数据）。最终，你需要完成论文修改。

# PreviousSummary.txt中包含了所有代码的readme文件，曾经AI编写这些代码时所用到的idea和实验计划。
# 之前论文的所有实验数据，在execute_history.txt中记录。
# 你执行后产生的数据，在data.txt中记录。
with open('ai_scientist\\pompts\\update_orchestrator.md', encoding='utf-8', errors='ignore') as f:
    ORCHESTRATOR_SYSTEM_PROMPT = f.read()


with open('ai_scientist\\pompts\\update_coder.md', encoding='utf-8', errors='ignore') as f:
    CODER_SYSTEM_PROMPT = f.read()
# CODER_SYSTEM_PROMPT = """


MONITOR_SYSTEM_PROMPT = """
你是一个科研实验的实时监控助手。
你的任务是观察当前正在运行的进程的控制台最新输出，判断程序是否处于异常状态（例如：死循环、长时间卡死无有效输出、Loss发散出现NaN、严重报错等）。
请根据提供的最新输出内容，严格返回以下 JSON 格式：

```json
{
    "Thoughts": "简短分析当前控制台输出是否表明程序运行正常。",
    "Action": "CONTINUE 或者 KILL",
    "Feedback": "如果Action为KILL，请给出详细的中断原因及提供给Coder修复代码的建议；如果Action为CONTINUE，请留空。"
}

```

注意：你提供的代码修复建议最好针对 run.bat 中的参数修改，或者明确指出 python 代码的哪个环节可能死锁或发散。
"""

# ==========================================

# 3. 现场保护与 Git 版本控制

# ==========================================

def save_state(workspace_dir, step_idx, past_summaries, base_readme, tool_calls_history):
    """保护现场：将 Orchestrator 所有状态信息存为断点文件"""
    state = {
    "step_idx": step_idx,
    "past_summaries": past_summaries,
    "base_readme": base_readme,
    "tool_calls_history": tool_calls_history
    }
    path = os.path.join(workspace_dir, "experiment_state.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.warning(f"[State Save] 保存保护文件失败: {e}")

def load_state(workspace_dir):
    """保护现场：加载断点文件"""
    path = os.path.join(workspace_dir, "experiment_state.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[State Load] 加载保护文件失败: {e}")
            return None

def git_init(workspace_dir, remote_repo=None):
    if not os.path.exists(os.path.join(workspace_dir, ".git")):
        subprocess.run(["git", "init"], cwd=workspace_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"[Git] 初始化 Git 仓库...")
    with open(os.path.join(workspace_dir, ".gitignore"), "w") as f:
        f.write("**pycache**/\n*.pyc\npdfs/\n*.log\n")
    subprocess.run(["git", "add", "."], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if remote_repo:
        try:
            subprocess.run(["git", "remote", "add", "origin", remote_repo], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "branch", "-M", "main"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[Git] 已关联远程仓库: {remote_repo}")

        except Exception as e:
            logger.warning(f"[Git] 关联远程仓库失败: {e}")
        
def git_commit_and_push(workspace_dir, step_idx, remote_repo=None):
    subprocess.run(["git", "add", "."], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=workspace_dir, capture_output=True, text=True).stdout
    if status.strip():
        subprocess.run(["git", "commit", "-m", f"Step {step_idx} passed"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"[Git] 提交 Git 仓库...")
    if remote_repo:
        try:
             subprocess.run(["git", "push", "origin", "main", "-f"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             logger.info(f"[Git] 推送到远程仓库...")
        except Exception as e:
            logger.warning(f"[Git] 推送到远程仓库失败")
            
            
def git_commit_and_push_with_msg(workspace_dir, commit_msg, remote_repo=None, review_round=0):
    subprocess.run(["git", "add", "."], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=workspace_dir, capture_output=True, text=True).stdout
    if status.strip():
        subprocess.run(["git", "commit", "-m", f"[R{review_round}]{commit_msg}"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"[Git] 提交 Git 仓库...")
    if remote_repo:
        try:
             subprocess.run(["git", "push", "origin", "main", "-f"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             logger.info(f"[Git] 推送到远程仓库...")
        except Exception as e:
            logger.warning(f"[Git] 推送到远程仓库失败")


def git_rollback(workspace_dir, remote_repo=None):
    if remote_repo:
        try:
            subprocess.run(["git", "fetch", "origin"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[Git] 回滚 Git 仓库...")
        except Exception as e:
            logger.warning(f"[Git] 从远程仓库回滚失败: {e}")
    else:
        try:
            # 强制重置清除污染，恢复至上一次正确的 Commit 节点
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "clean", "-fd"], cwd=workspace_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[Git] 回滚本地 Git 仓库...")
        except Exception as e:
            logger.warning(f"[Git] 从本地仓库回滚失败: {e}")
    

# ==========================================

# 4. 系统环境与底层监控函数

# ==========================================

def get_hardware_status():
    status_info = "【当前硬件资源状态】\n"
    try:
        smi_output = subprocess.check_output("nvidia-smi", shell=True, encoding="utf-8", errors="replace", timeout=5)
        status_info += f"--- nvidia-smi 专用显存与GPU利用率 ---\n{smi_output}\n"
    except Exception:
        pass
    try:
        mem_output = subprocess.check_output("wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", shell=True, encoding="utf-8", errors="ignore", timeout=5)
        status_info += f"--- 系统物理内存 ---\n{mem_output.strip()}\n"
    except Exception:
        pass
    return status_info

def run_command_with_monitoring(script_path, cwd, monitor_agent, orchestrator_prompt = ""):
    cmd = f'conda run -n {CONDA_ENV_NAME} --no-capture-output cmd.exe /c "{script_path} & exit"'
    logger.info(f"\n[System] 执行脚本: {script_path} ...")


    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding='utf-8', errors='replace', text=True, env=env
    )

    output_lines = []
    q = queue.Queue()

    def reader_thread(proc, queue):
        for line in iter(proc.stdout.readline, ''):
            queue.put(line)
        proc.stdout.close()

    t = threading.Thread(target=reader_thread, args=(process, q))
    t.daemon = True
    t.start()

    start_time = time.time()
    last_monitor_time = start_time
    killed_by_monitor = False
    monitor_feedback = ""

    while True:
        while not q.empty():
            try:
                line = q.get_nowait()
                output_lines.append(line)
            except queue.Empty: break

        if process.poll() is not None:
            while not q.empty():
                try:
                    line = q.get_nowait()
                    output_lines.append(line)
                except queue.Empty: break
            break

        current_time = time.time()
        if current_time - start_time > 36000:
            killed_by_monitor = True
            monitor_feedback = "进程运行超过10小时硬性超时限制，已被系统强制杀死。"
            subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break

        if current_time - last_monitor_time >= 400:
            last_monitor_time = current_time
            recent_output = "".join(output_lines[-150:])
            if recent_output.strip():
                status_info = get_hardware_status()
                monitor_prompt = f"【实时监控最新控制台输出片段】代码已运行{int(current_time - start_time)}秒，当前输出：\n{recent_output}\n\n宿主机状态：\n{status_info}\n\n请判定程序是否正常，是否需要提前终止(KILL)？"
                
                monitor_agent.clear_history()
                try:
                      resp, _ = monitor_agent.get_response_stream(monitor_prompt, ORCHESTRATOR_SYSTEM_PROMPT)
                except Exception as e:
                      logger.warning(f"实时监控获取失败")
                      continue
                monitor_json = LLMAgent.robust_extract_json(resp)
                
                
                if monitor_json and monitor_json.get("Action") == "KILL":
                    logger.info("\n[Orchestrator Monitor] 判定异常，下达指令：终止进程！")
                    killed_by_monitor = True
                    monitor_feedback = monitor_json.get("Feedback", "被根据实时输出强制终止。")
                    action_json = monitor_json.get("Action_Params",{})
                    monitor_feedback = action_json.get()
                    subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
        time.sleep(1)

    full_stdout = "".join(output_lines)
    return (not killed_by_monitor and process.returncode == 0), full_stdout, "", monitor_feedback

def get_installed_packages():
    cmd = f'conda run -n {CONDA_ENV_NAME} pip list'
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
        return result.stdout if result.returncode == 0 else "Failed to get pip list."
    except: return "Failed to get pip list."

def get_workspace_state(cwd, include_content=False):
    if not os.path.exists(cwd): return "Workspace empty."
    files = [f for f in os.listdir(cwd) if not f.startswith('.') and f not in ['**pycache**', 'pdfs']]
    if not files: return "Workspace is empty."

    state = "【当前文件结构】\n"
    for f in files: state += f"- {f}\n"
        
    if include_content:
        state += "\n【文件具体内容】\n"
        has_content = False
        for file in files:
            if file.endswith((".py", ".md", ".bat", ".txt")):
                path = os.path.join(cwd, file)
                if os.path.isfile(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            state += f"\n--- {file} ---\n{f.read()}\n"
                        has_content = True
                    except Exception: pass
        if not has_content:
            state += "暂无代码文件。\n"
    return state



# def extract_files_from_coder(text):
#     pattern = r"###\s*File:\s*([^\\n]+)\n`\w*\n(.*?)`"
#     matches = re.findall(pattern, text, re.DOTALL)
#     return {filename.strip(): content.strip() for filename, content in matches}
import re

def extract_files_from_coder(text):
    # 使用 \s* 来容忍文件名和代码块之间的任意空行/空格
    # 使用 ``` 匹配完整的 Markdown 代码块语法
    # [^\n]* 用于匹配 ``` 后面的语言标识符（如 python, bat, markdown 等）
    pattern = r"###\s*File:\s*([^\n]+)\s*```[^\n]*\n(.*?)```"
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    # .strip() 可以清理掉文件名或代码内容首尾多余的空格和换行
    return {filename.strip(): content.strip() for filename, content in matches}

def save_files_to_workspace(files, cwd, base_readme=""):
    saved_list = []
    for filename, content in files.items():
        filepath = os.path.join(cwd, filename)
        final_content = f"{base_readme}\n\n## [Current Step]\n{content}".strip() if filename.lower() == "readme.md" else content
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_content)
            saved_list.append(filename)
        except Exception as e:
            logger.warning(f"[System Error] 无法保存文件 {filename}: {e}")
    return saved_list

# ==========================================
# 5. 可拓展的 Tool 类封装 (ToolManager)
# ==========================================
class ToolManager:
    def __init__(self, workspace_dir, orchestrator, coder, monitor_agent):
        self.workspace_dir = workspace_dir
        self.orchestrator = orchestrator
        self.coder = coder
        self.monitor_agent = monitor_agent
        self.doi_url_map = {}
        self.kb_txt_path = os.path.join(workspace_dir, "knowledge_base.txt")


    def search_literature(self, queries):
        if not queries: return "未提供查询词。"
        try:
            res = format_search_results_and_update_map(queries, self.doi_url_map)
            return res
        except Exception as e: return f"文献搜索出错: {e}"

    def read_paper(self, dois):
        if not dois: return "未提供 DOI。"
        try:
            process_papers_to_read(dois, self.doi_url_map, self.kb_txt_path)
            return read_knowledge_base(self.kb_txt_path)
        except Exception as e: return f"阅读文献出错: {e}"

    def read_code(self, filename):
        if not filename: return "未提供文件名。"
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path): return f"File '{filename}' does not exist in the workspace."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def run_code(self, run_script, orch_prompt):
        if not run_script: return "脚本内容为空。"
        script_path = os.path.join(self.workspace_dir, "run_tool.bat")
        with open(script_path, "w", encoding="utf-8") as f: f.write(run_script)
        
        success, stdout, stderr, monitor_feedback = run_command_with_monitoring(script_path, self.workspace_dir, self.orchestrator, orchestrator_prompt=orch_prompt)
        res = f"Execute Success: {success}\nConsole Output:\n{stdout}\n"
        if monitor_feedback: res += f"Monitor Feedback: {monitor_feedback}\n"
        return res

    def prompt_coder(self, instruction, base_readme=""):
        """此方法独立接管了与 Coder Agent 之间的自迭代死循环，并将最终结果返回给主控。"""
        coder_history = []
        for i in range(10): # 允许 Coder 内循环自行迭代并测试最多 10 次
            self.coder.clear_history()
            
            pip_list = get_installed_packages()
            workspace_files = get_workspace_state(self.workspace_dir, include_content=False)
            
            prompt = f"【Orchestrator 指令】\n{instruction}\n\n"
            prompt += f"【当前工作空间文件结构】\n{workspace_files}\n\n"
            prompt += f"【Pip 依赖包 (截断)】\n{pip_list[:1000]}\n\n"
            
            if coder_history:
                prompt += "【你本轮已执行的 Tool 历史】\n"
                for h in coder_history:
                    prompt += f"Action: {h['action']}, Params: {json.dumps(h['params'], ensure_ascii=False)}\nResult:\n{h['result']}\n\n"
            prompt += "请决定你的下一步 Action (READ_CODE, RUN_CODE, 或 SUBMIT_CODE)。"
            
            prompt += """
            【交互格式】
             你的回复必须严格包含以下 JSON 结构（被 ```json 和 ```json 包裹）(注意：切勿漏掉```json!!!)：：

            ```json
           {
                "Thoughts": "你的思考过程，是否需要读取现存代码、自己运行测试还是准备提交代码。",
                "Action": "READ_CODE | RUN_CODE | SUBMIT_CODE",
                "Action_Params": {
                    "filename": "如果 READ_CODE，需要读取的文件名",
                    "run_script": "如果 RUN_CODE，运行用于测试的 bat 脚本内容(必须命名为run.bat)。注意："
                }
           }
            ```

            **【强制要求】**：当且仅当 Action 为 `SUBMIT_CODE` 时，你必须在 JSON 结构**之外**的下方，以 Markdown 代码块的形式输出你需要创建或修改的文件，并且必须在代码块上方标明文件名！例如：
            当你选择输出python脚本时，你必须确保脚本实现的功能完整且可运行。即便你只是被要求修改代码，你也必须保证返回的代码是完整的（而不是片段）。
            ### File: main.py

            ```python
            import numpy as np
            import argparse
            print("Hello World")

            ```

            ### File: run.bat

            ```bat
            pip install numpy
            python main.py --epochs 10

            ```

            ### File: readme.md

            ```markdown
            # 功能说明

            ```
            """
            
            try:
                # CODER_SYSTEM_PROMPT.format(conda_env=CONDA_ENV_NAME)
                # 直接替换，原 CODER_SYSTEM_PROMPT 字符串一个字都不用改
                formatted_prompt = CODER_SYSTEM_PROMPT.replace("{conda_env}", CONDA_ENV_NAME)
            
                resp, _ = self.coder.get_response_stream(prompt, formatted_prompt)
            except Exception as e:
                logger.error(f"[Coder Error] API 调用出错：{e}")
                time.sleep(5)
                continue
                
            json_resp = LLMAgent.robust_extract_json(resp)
            if not json_resp:
                files = extract_files_from_coder(resp)
                if files:
                    saved_files = save_files_to_workspace(files, self.workspace_dir, base_readme)
                    # 容错：如果 JSON 解析失败，但成功提取了代码
                    return f"Coder 已成功完成任务并隐式提交了文件: {', '.join(saved_files)}。\nCoder输出说明:\n{resp}"
                else:
                    coder_history.append({"action": "PARSE_ERROR", "params": "", "result": "JSON解析失败且未发现Markdown代码块。"})
                    continue
                
            action = json_resp.get("Action")
            params = json_resp.get("Action_Params", {})
            
            if action == "READ_CODE":
                result = self.read_code(params.get("filename", ""))
                coder_history.append({"action": action, "params": params, "result": result})
                logger.info(f"[Coder Action] 读取文件 {params.get('filename', '')}。")
            elif action == "RUN_CODE":
                files = extract_files_from_coder(resp)
                if files:
                    saved_files = save_files_to_workspace(files, self.workspace_dir, base_readme)
                result = self.run_code(params.get("run_script", ""))
                coder_history.append({"action": action, "params": params, "result": result})
                logger.info(f"[Coder Action] 运行脚本 {params.get('run_script', '')}。")
            elif action == "SUBMIT_CODE":
                files = extract_files_from_coder(resp)
                if not files:
                    coder_history.append({"action": action, "params": params, "result": "错误: 选择了 SUBMIT_CODE 但未按格式输出 Markdown 代码块。"})
                    continue
                saved_files = save_files_to_workspace(files, self.workspace_dir, base_readme)
                # logger.info(f"[Coder Action] 提交文件。")
                # Append last run result if available
                last_run = next((item for item in reversed(coder_history) if item["action"] == "RUN_CODE"), None)
                run_res = ""
                if last_run:
                    run_res = f"\nCoder自测结果:\n{last_run['result']}"
                
                return f"Coder 已成功完成任务并提交了文件: {', '.join(saved_files)}。\n{run_res}"
            else:
                coder_history.append({"action": action, "params": params, "result": f"Unknown Action: {action}"})
                
        return "Coder 达到了自我调试上限，未能成功提交最终代码。"



# ==========================================
# 6. 主控工作流 (Agentic Workflow)
# ==========================================

def load_plan(plan_file):
    with open(plan_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list): data = data[0]
    return data.get("Detailed_Plan", []), data.get("Original_Idea", {})

def run_experiment(experiment_dir, log_dir, model_orchestrator, model_coder, include_all_files=False, repo_url=None, review_round = 1):
    # plan_steps, idea = load_plan(plan_file)
    # if not plan_steps:
    #     logger.info("[System] 未能在计划文件中找到步骤！")
    #     return

    isFirstRound = False
    workspace_dir = os.path.abspath(experiment_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    orch_log = os.path.join(log_dir, "orchestrator.log")
    coder_log = os.path.join(log_dir, "coder.log")
    monitor_log = os.path.join(log_dir, "monitor.log")
    summary_txt = os.path.join(workspace_dir, "rebuttal_summary.txt")
    if not os.path.exists(summary_txt):
       isFirstRound = True
    rebuttal_summary_txt = os.path.join(workspace_dir, "rebuttal_summary.txt")
    data_record_txt = os.path.join(workspace_dir,"recorded_data.txt")
    review_txt = os.path.join(workspace_dir, "review.txt")
    review_raw_txt = ""
    with open(review_txt, "r", encoding="utf-8",errors='replace') as f: review_raw_txt=f.read()
    orchestrator = LLMAgent(model=model_orchestrator, log_file=orch_log)
    coder = LLMAgent(model=model_coder, log_file=coder_log)
    monitor = LLMAgent(model=model_orchestrator, log_file=monitor_log) # Monitor 需要独立的实例以防污染历史
    monitor.set_context_len(5)

    tool_manager = ToolManager(workspace_dir, orchestrator, coder, monitor)
    git_init(workspace_dir, repo_url)

    state = load_state(workspace_dir)
    state = None
    if state:
        step_idx = state["step_idx"]
        past_summaries = state["past_summaries"]
        base_readme = state["base_readme"]
        tool_calls_history = state.get("tool_calls_history", [])
        logger.info(f"=== 发现现场保护配置文件，系统从第 {step_idx + 1} 步热启动恢复执行 ===")
    else:
        step_idx, past_summaries, base_readme, tool_calls_history = 0, [], "", []
        logger.info(f"=== 实验启动，工作空间: {workspace_dir} ===")
    
    past_rebuttal_summaries = ""
    if not isFirstRound:
        with open(data_record_txt, "r", encoding="utf-8",errors='ignore') as f:
             past_rebuttal_summaries = f.read()
    # 没有执行计划的概念
    past_summaries = []
    while step_idx < 1:
        step = {}
        step_name = step.get("name", "")
        step_content = step.get("content", "")
        step_expected = step.get("expected_outcome", "")
        
        # logger.info(f"\n>>> 开始攻关计划 第 {step_idx + 1} 步: {step_name} <<<")
        step_passed = False
        attempts = len(tool_calls_history) 
        
        while attempts < MAX_RETRIES and not step_passed:
            attempts += 1
            logger.info(f"\n--- [Step {step_idx + 1}] Orchestrator 第 {attempts}/{MAX_RETRIES} 轮交互 ---")
            
            # 使用 LLMAgent 的 clear_history()，由于我们已经在下面显式附带历史信息了，这使得我们严格隔绝了历史爆发的隐患
            orchestrator.clear_history() 
            
            workspace_files = get_workspace_state(workspace_dir, include_content=include_all_files)
            
            # context_prompt = f"【产生这篇论文的完整研究计划(已经执行并生成了你看到的这些代码）】\n背景: {idea.get('Background','')}\n方法: {idea.get('Methodology','')}\n\n"
            context_prompt = f""
            context_prompt += f"【当前工作目录状态】\n{workspace_files}\n\n"
            context_prompt += f"【之前几次rebuttal中你已经完成的工作】：\n {past_rebuttal_summaries}\n"
            context_prompt += f"【来自审稿人的审稿意见】\n{review_raw_txt}\n"
            if True:
                context_prompt += f"【已经完成的工作总结】\n"
                context_prompt += "\n".join(past_summaries) + "\n"
                
            context_prompt += "【你已执行的最近10轮 Tool Call 及结果】\n"
            if not tool_calls_history:
                context_prompt += "暂无。这是第一轮操作。\n"
            else:
                if len(tool_calls_history) > 20:
                    for th in tool_calls_history[-20:]:
                        # res_trunc = th['result'] if len(str(th['result'])) < 3000 else str(th['result'])[:3000] + "...(truncated)"
                        res_trunc = th['result']
                        context_prompt += f"Action: {th['action']}, Params: {json.dumps(th.get('params', {}), ensure_ascii=False)}\nResult:\n{res_trunc}\n\n"
                else:
                    for th in tool_calls_history:
                        # res_trunc = th['result'] if len(str(th['result'])) < 3000 else str(th['result'])[:3000] + "...(truncated)"
                        res_trunc = th['result']
                        context_prompt += f"Action: {th['action']}, Params: {json.dumps(th.get('params', {}), ensure_ascii=False)}\nResult:\n{res_trunc}\n\n"
                    
            context_prompt += "请仔细阅读以上内容，并输出 JSON 指令。如确认已经针对用户的所有意见完成对应的修改和操作，调用 PASS_STEP。"
            context_prompt += """
           【交互格式】
你的回复必须严格包含以下 JSON 结构（被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "思考当前处于什么阶段，已经完成了用户的哪些需求，还有用户的哪些需求没有完成，分析上一个工具的返回结，决定下一步调用什么工具",
    "Action": "SEARCH_LITERATURE | READ_PAPER | READ_CODE | PROMPT_CODER | RUN_CODE |KILL| CONTINUE|MODIFY_ARTICLE| RECORD_DATA|PASS_STEP",
    "Action_Params": {
        "queries": ["如果调用 SEARCH_LITERATURE，在此提供搜索关键词"],
        "dois": ["如果调用 READ_PAPER，在此提供 doi"],
        "filename": "如果调用 READ_CODE，在此提供文件名",
        "instruction": "如果调用 PROMPT_CODER，在此写明具体编程或修改指令",
        "run_script": "如果调用 RUN_CODE，在此写你需要系统执行的完整 bat 脚本内容",
        "summary": "在此写当前步骤完成情况的详细总结（只要写当前步骤干了什么即可！不需要一起总结前面的内容）。重点包含目前在处理用户需求中的具体哪一点，并重点描述当前的仿真场景，仿真参数和详细的仿真结果(例如详细的BER-SNR数据),以及你采取当前操作的原因和意图"，
        "data":"如果是 RECORD_DATA，请完整提供：（完整地记录数据本身，你还要包括：这些数据对应的详细仿真场景，产生这些数据的python文件名，这些数据是为了回应用户的哪一条意见）"
    }
}
```
            """
            try:
                resp, _ = orchestrator.get_response_stream(context_prompt, ORCHESTRATOR_SYSTEM_PROMPT)
            except Exception as e:
                logger.error(f"[Orchestrator Error] API connection failed")
                continue
            action_json = LLMAgent.robust_extract_json(resp)
            
            if not action_json:
                tool_calls_history.append({"action": "PARSE_ERROR", "params": {}, "result": "Failed to parse JSON."})
                save_state(workspace_dir, step_idx, past_summaries, base_readme, tool_calls_history)
                continue
                
            action = action_json.get("Action")
            params = action_json.get("Action_Params", {})
            thoughts_curr = action_json.get("Thoughts", "")
            
            summary_curr = params.get("summary", "")
            if summary_curr:
                past_summaries.append(f"【第 {step_idx + 1} 步: {step_name} 总结】: {summary_curr}")
                with open(summary_txt, "a", encoding="utf-8") as f:
                    f.write(f"--- 第 {attempts + 1}次工具调用总结 ---\n{summary_curr}\n\n")
            else:
                logger.warning(f"[Orchestrator] 未能获取到当前步骤的总结。")
            if action == "PASS_STEP":
                summary = params.get("summary", f"步骤 {step_idx + 1} 已成功完成。")
                step_passed = True
                
                past_summaries.append(f"【第 {attempts + 1} 步: {step_name} 总结】: {summary}")
                with open(summary_txt, "a", encoding="utf-8") as f:
                    f.write(f"--- 步骤 {attempts + 1}: {step_name} ---\n{summary}\n\n")
                    
                logger.info(f"[Orchestrator] 验收通过！总结: {summary[:100]}...")
            elif action == "SEARCH_LITERATURE":
                res = tool_manager.search_literature(params.get("queries", []))
                tool_calls_history.append({"action": action, "params": params, "result": res, "Thoughts": thoughts_curr})
                logger.info(f"[Orchestrator]请求文献搜索： {params.get('queries', [])}"[:70])
            elif action == "READ_PAPER":
                res = tool_manager.read_paper(params.get("dois", []))
                tool_calls_history.append({"action": action, "params": params, "result": res, "Thoughts": thoughts_curr})
                logger.info(f"[Orchestrator]请求阅读论文： {params.get('dois', [])}"[:70])
            elif action == "READ_CODE":
                res = tool_manager.read_code(params.get("filename", ""))
                tool_calls_history.append({"action": action, "params": params, "result": res, "Thoughts": thoughts_curr})
                logger.info(f"[Orchestrator]请求阅读代码： {params.get('filename', '')}"[:70])
            elif action == "PROMPT_CODER":
                res = tool_manager.prompt_coder(params.get("instruction", ""), base_readme)
                path = os.path.join(workspace_dir, "readme.md")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        base_readme = f.read()
                tool_calls_history.append({"action": action, "params": params, "result": res, "Thoughts": thoughts_curr})
                logger.info(f"[Orchestrator]请求 Coder 编写代码： {params.get('instruction', '')}"[:70])
            elif action == "RUN_CODE":
                res = tool_manager.run_code(params.get("run_script", ""),context_prompt)
                tool_calls_history.append({"action": action, "params": params, "result": res, "Thoughts": thoughts_curr})
                logger.info(f"[Orchestrator]请求运行脚本： {params.get('run_script', '')}"[:70])
            elif action == "MODIFY_ARTICLE":
                files = extract_files_from_coder(resp)
                if files:
                    saved_files = save_files_to_workspace(files, experiment_dir, base_readme)
                    # 容错：如果 JSON 解析失败，但成功提取了代码
                    
                    logger.info( f"[Orchestrator]修改了论文并提交了文件: {', '.join(saved_files)}。")
                    tool_calls_history.append({"action": action, "params": params, "result": f"Modified article and submitted files: {', '.join(saved_files)}", "Thoughts": thoughts_curr})
                    commit_msg = f"Updated {', '.join(saved_files)}"[:30]
                    git_commit_and_push_with_msg(workspace_dir, commit_msg, repo_url, review_round=review_round)
                    
                else:
                    tool_calls_history.append({"action": "PARSE_ERROR", "params": "", "result": "JSON解析失败且未发现Markdown代码块。"})
                    continue
            elif action == "RECORD_DATA":
                data_record = params.get("data", "")
                with open(data_record_txt, "a", encoding="utf-8") as f:
                    f.write(f"--- 第 {attempts + 1} 步数据记录 ---\n{data_record}\n\n")
                    tool_calls_history.append({"action": action, "params": params, "result": f"Recorded data: {data_record}","Thoughts": thoughts_curr})
                commit_msg = f"Recorded Data"
                git_commit_and_push_with_msg(workspace_dir, commit_msg, repo_url, review_round=review_round)
            else:
                tool_calls_history.append({"action": action, "params": params, "result": f"Unknown action: {action}", "Thoughts": thoughts_curr})
                logger.warning(f"[Orchestrator]未知指令： {action}"[:70])
                
            if not step_passed:
                # 无论结果如何，每次行动都保存，保证断电不丢数据
                save_state(workspace_dir, step_idx, past_summaries, base_readme, tool_calls_history)

        # ====== 步进机制与 Git 线上回溯联动 ======
        if step_passed:
            commit_msg = f"Finished Review Round {review_round}"
            git_commit_and_push_with_msg(workspace_dir, commit_msg, repo_url, review_round=review_round)
            step_idx += 1
            tool_calls_history = []
            save_state(workspace_dir, step_idx, past_summaries, base_readme, tool_calls_history)
        else:
            logger.warning(f"\n[System Error] 第 {step_idx + 1} 步在 {MAX_RETRIES} 次尝试后仍然失败。触发 Git 回退机制！")
            if step_idx > 0:
                # 1. 从线上仓库拉取上一步 push 的结果，重置本地一切脏乱改动
                git_rollback(workspace_dir, repo_url) 
                
                # 2. 退回上一步
                step_idx -= 1
                if past_summaries:
                    past_summaries.pop()
                tool_calls_history = []
                
                path = os.path.join(workspace_dir, "readme.md")
                base_readme = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""
                
                # 重新重写 summary 以保持干净
                with open(summary_txt, "w", encoding="utf-8") as f:
                    for s in past_summaries:
                        f.write(s + "\n\n")

                save_state(workspace_dir, step_idx, past_summaries, base_readme, tool_calls_history)
                logger.info(f"[System] 已从仓库恢复上一步push的结果重置项目，并退回第 {step_idx + 1} 步重试。")
            else:
                logger.error("[System Fatal] 第一步就彻底失败，无法回退。实验终止。")
                break

    logger.info("\n=== 科研代码全流程执行完毕 ===")
    logger.info(f"最终结果与代码保存在: {workspace_dir}")



def update_from_review(args):
    """主入口函数，可通过传参灵活执行控制"""
    # if not os.path.exists(args.plan_file):
    #     logger.info(f"找不到计划文件: {args.plan_file}")
    #     return
    
    include_all_files = getattr(args, 'include_all_files', False)
    repo_url = getattr(args, 'repo_url', None)
       
    run_experiment(
        model_orchestrator=args.model_orchestrator,
        model_coder=args.model_coder,
        experiment_dir=args.experiment_dir,
        log_dir=args.experiment_log_dir,
        include_all_files=include_all_files,
        repo_url=repo_url
    )
    
class MyArgs:
    def __init__(self):
        self.model_orchestrator = None
        self.model_coder = None
        self.experiment_dir = None
        self.experiment_log_dir = None
        self.include_all_files = False
        self.repo_url = None
    pass
args = MyArgs()
args.model_orchestrator = 'gemini-3.1-pro-preview'
args.model_coder = args.model_orchestrator
args.experiment_dir = r"D:\\ChannelCoding\\RCOM\\ROCM-main\\ROCM-main"
args.experiment_log_dir = r"D:\\ChannelCoding\\RCOM\\ROCM-main\\ROCM-main\\auto_log"
args.include_all_files = False
args.repo_url = "https://github.com/NaLv2004/NewDarts.git"

update_from_review(args)
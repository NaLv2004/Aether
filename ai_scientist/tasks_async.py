
import os
import re
import argparse
import time
import threading
import queue
import subprocess
from collections import deque

from llm import LLMAgent
from utils import setup_logger

logger = setup_logger("agent_workspace.log")

# ==========================================
# 1. 默认系统提示词模板 (与具体任务解耦)
# ==========================================

DEFAULT_ORCHESTRATOR_PROMPT = """
你是一个高级科研/开发项目管家 (Orchestrator Agent)。你的任务是根据用户的需求（在 review.txt 或 request.txt 中），分析当前工作目录的代码和文件，并完成最终目标。

你可以并发执行多个任务。你可以执行的操作（Tools）包括：
【同步工具】(立刻返回结果)：
1. `READ_FILE`: 读取某个文件的内容。参数: "filename"
2. `WRITE_FILE`: 创建或覆写一个文件。只能是txt文件，tex文件，用来汇总你的实验过程，不可以是代码（你不可以自己写代码，只能通过SPAWN_CODER来让Coder写代码）。参数: "filename"
3. `SEARCH_LITERATURE`: 查找文献。参数: "queries" (列表)
4. `KILL_TASK`: 终止当前正在运行的某个异步任务（如果发现报错死循环、Loss发散、或迟迟没有结果）。参数: "task_id"
5. `WAIT`: 等待一段时间。如果你发现当前有任务正在运行，且你需要等它们产生更多日志或运行结束才能进行下一步，请调用此工具。参数: "wait_seconds" (整数)
6. `FINISH`: 确认所有用户要求已完成，结束工作流。参数: "summary"
7. `RECORD_DATA`:当前程序输出了对论文撰写有意义的实验数据，你必须采取此行动，将当前仿真场景、完整数据详细写入JSON的"data"字段。 

【异步工具】(下发后会分配一个 task_id 在后台运行，你可以继续执行其他动作)：
7. `SPAWN_CODER`: 分配一个 Coder 智能体去编写/修改代码并自行测试。参数: "instruction" (详细指令。建议采取总分结构：先介绍任务背景，涉及的文件等，再介绍具体的编程方案)。
8. `SPAWN_RUN`: 直接运行一个系统 bat 脚本。参数: "run_script" (完整的bat命令，不要包含pause，不要在bat脚本中内嵌python代码，不要在一个bat脚本中运行多个或者多次运行一个python文件，防止时间过长)。

【并发与监控机制说明】
- 你的上下文中会看到【当前正在运行的任务 (Active Tasks)】及其最新控制台输出片段。
- 如果输出显示异常（如死锁、报错停滞、性能不达标），你必须果断调用 `KILL_TASK` 结束它，然后可能需要重新修改代码。
- 如果并发任务数未达上限，你可以连续调用 `SPAWN_*` 工具开启多个实验。
- 如果你在等待某个任务的结果，且暂时不需要开启新任务，请调用 `WAIT`。

【交互格式】严格遵守JSON格式 (包含在 ```json 中)：
```json
{
    "Thoughts": "分析用户的需求，当前状态、运行任务的日志；当前解决了哪些任务，哪些任务还没有解决。决定是下发新任务、杀死任务还是等待。",
    "Action": "READ_FILE | WRITE_FILE | KILL_TASK | WAIT | SPAWN_CODER | SPAWN_RUN | FINISH",
    "Action_Params": {
        "instruction":"如果是SPAWN_CODER,提供详细指令",
        "run_script":"如果是SPAWN_RUN,提供完整的bat命令",
        "filename":"如果是READ_FILE或WRITE_FILE,提供文件名",
        "queries":"如果是SEARCH_LITERATURE,提供搜索词列表",
        "task_id":"如果是KILL_TASK,提供任务ID",
        "data":"如果是RECORD_DATA,在此提供详细的实验数据和仿真场景,以及数据是为了回应用户的哪一点需求",
        "summary":"对当前步骤的详细总结。无论你的动作是什么，必须提供此字段"
    }
}
```
当且仅当 Action 为 WRITE_FILE 时，在 JSON 外用 markdown 代码块提供文件内容：
### File: filename.txt
```
[内容]
```
"""

DEFAULT_CODER_PROMPT = """
你是一个顶级的 AI 程序员。你的任务是根据主管的需求编写、修改并测试代码。
你可以执行：
1. `READ_CODE`: 读取文件，参数 "filename"
3. `SUBMIT_CODE`: 提交最终代码完成任务。

返回格式：
```json
{
    "Thoughts": "思考过程",
    "Action": "READ_CODE | SUBMIT_CODE",
    "Action_Params": {
        "filename": "如果READ_CODE,提供文件名",
    }
}
```
当为 SUBMIT_CODE  时，可在外部附带代码块：
### File: main.py
```python
print("Hello")
```

注意：无论你是被要求修改还是编写新的代码，你都必须提交完整的代码。除非你被要求，否则不得删除原有代码的任何功能。
"""



# ==========================================
# 2. 文件系统与工具辅助函数
# ==========================================

def get_workspace_state_recursive(dir_path, max_files_per_dir=10, prefix=""):
    """递归获取文件树，限制单层显示文件数量"""
    if not os.path.exists(dir_path):
        return "Workspace empty."
    
    state = ""
    try:
        items = sorted(os.listdir(dir_path))
    except PermissionError:
        return prefix + "[Permission Denied]\n"

    # 过滤掉隐藏文件和缓存
    items = [f for f in items if not f.startswith('.') and f not in ['__pycache__', 'pdfs']]
    
    files = [f for f in items if os.path.isfile(os.path.join(dir_path, f))]
    dirs = [d for d in items if os.path.isdir(os.path.join(dir_path, d))]

    # 处理文件
    for i, f in enumerate(files):
        if i < max_files_per_dir:
            state += f"{prefix}- {f}\n"
        elif i == max_files_per_dir:
            state += f"{prefix}- ... and {len(files) - max_files_per_dir} more files.\n"
            break
            
    # 处理文件夹
    for d in dirs:
        state += f"{prefix}+ [DIR] {d}/\n"
        state += get_workspace_state_recursive(os.path.join(dir_path, d), max_files_per_dir, prefix + "  ")

    return state if state else f"{prefix}(Empty Directory)\n"

def extract_files_from_response(text):
    pattern = r"###\s*File:\s*([^\n]+)\s*```[^\n]*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return {filename.strip(): content.strip() for filename, content in matches}

# ==========================================
# 3. 异步并发任务管理器 (AsyncTask & TaskManager)
# ==========================================

class AsyncTask:
    def __init__(self, task_id, task_type, args, workspace_dir):
        self.task_id = task_id
        self.task_type = task_type  # "CODER" or "RUN"
        self.args = args
        self.workspace_dir = workspace_dir
        
        self.status = "RUNNING"  # RUNNING, KILLED, FINISHED, ERROR
        self.output_queue = queue.Queue()
        self.log_history = deque(maxlen=200)  # 只保留最近200行用于实时监控
        self.full_log = []
        self.result_summary = ""
        
        self.process = None
        self.thread = None
        self._stop_event = threading.Event()

    def log(self, msg):
        line = msg.strip() + "\n"
        self.output_queue.put(line)
        self.log_history.append(line)
        self.full_log.append(line)

    def kill(self):
        self._stop_event.set()
        if self.process:
            try:
                subprocess.run(f"taskkill /F /T /PID {self.process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
        self.status = "KILLED"
        self.log("\n[System] Task was KILLED by Orchestrator.")

class TaskManager:
    def __init__(self, max_concurrent, workspace_dir, coder_model_name, conda_env):
        self.max_concurrent = max_concurrent
        self.workspace_dir = workspace_dir
        self.coder_model_name = coder_model_name
        self.conda_env = conda_env
        self.tasks = {}       # id -> AsyncTask
        self.task_counter = 0

    def get_active_tasks(self):
        return {tid: t for tid, t in self.tasks.items() if t.status == "RUNNING"}

    def get_finished_tasks_and_clear(self):
        """获取已完成/被杀死的任务状态，并从池中移除以释放并发槽位"""
        finished = {}
        for tid in list(self.tasks.keys()):
            if self.tasks[tid].status in ["FINISHED", "KILLED", "ERROR"]:
                finished[tid] = self.tasks.pop(tid)
        return finished

    def spawn_run(self, run_script):
        if len(self.get_active_tasks()) >= self.max_concurrent:
            return None, "Max concurrency reached. Please WAIT or KILL_TASK."
        
        self.task_counter += 1
        tid = f"Task-Run-{self.task_counter}"
        task = AsyncTask(tid, "RUN", {"script": run_script}, self.workspace_dir)
        self.tasks[tid] = task

        def _run_worker():
            script_path = os.path.join(self.workspace_dir, f"run_{tid}.bat")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(run_script)
            
            cmd = f'conda run -n {self.conda_env} --no-capture-output cmd.exe /c "{script_path} & exit"'
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            task.process = subprocess.Popen(
                cmd, shell=True, cwd=self.workspace_dir, stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', text=True, env=env
            )
            
            for line in iter(task.process.stdout.readline, ''):
                if task._stop_event.is_set():
                    break
                if line:
                    task.log(line)
                    
            task.process.stdout.close()
            task.process.wait()
            
            if not task._stop_event.is_set():
                task.status = "FINISHED" if task.process.returncode == 0 else "ERROR"
                task.result_summary = f"Process exited with code {task.process.returncode}."

        task.thread = threading.Thread(target=_run_worker, daemon=True)
        task.thread.start()
        return tid, "Spawned successfully."

    def spawn_coder(self, instruction):
        if len(self.get_active_tasks()) >= self.max_concurrent:
            return None, "Max concurrency reached. Please WAIT or KILL_TASK."
        
        self.task_counter += 1
        tid = f"Task-Coder-{self.task_counter}"
        task = AsyncTask(tid, "CODER", {"instruction": instruction}, self.workspace_dir)
        self.tasks[tid] = task

        def _coder_worker():
            # 这里简化了原有的 Coder 逻辑为异步函数
            coder_agent = LLMAgent(model=self.coder_model_name)
            task.log(f"[Coder] Started task: {instruction[:50]}...")
            
            # Coder 的自循环逻辑 (至多10次)
            for i in range(10):
                if task._stop_event.is_set(): break
                
                ws_state = get_workspace_state_recursive(self.workspace_dir, 10)
                prompt = f"指令: {instruction}\n环境:\n{ws_state}\n"
                
                try:
                    resp, _ = coder_agent.get_response_stream(prompt, DEFAULT_CODER_PROMPT)
                except Exception as e:
                    task.log(f"[Coder Error] API fail: {e}")
                    time.sleep(2)
                    continue
                
                action_json = LLMAgent.robust_extract_json(resp)
                if not action_json:
                    files = extract_files_from_response(resp)
                    if files:
                        for fn, cont in files.items():
                            with open(os.path.join(self.workspace_dir, fn), "w", encoding="utf-8") as f: f.write(cont)
                        task.log("[Coder] Implied submission. Files written.")
                        task.status = "FINISHED"
                        return
                    continue
                
                action = action_json.get("Action")
                params = action_json.get("Action_Params", {})
                
                if action == "READ_CODE":
                    task.log(f"[Coder] Read file {params.get('filename')}")
                elif action == "RUN_CODE":
                    task.log(f"[Coder] Wants to run test script (skipped in minimal async demo, treating as mock success)")
                elif action == "SUBMIT_CODE":
                    files = extract_files_from_response(resp)
                    for fn, cont in files.items():
                        with open(os.path.join(self.workspace_dir, fn), "w", encoding="utf-8") as f: f.write(cont)
                    task.log(f"[Coder] Finished. Wrote {len(files)} files.")
                    task.status = "FINISHED"
                    return
            
            if not task._stop_event.is_set():
                task.status = "ERROR"
                task.result_summary = "Coder failed to finish in 10 loops."

        task.thread = threading.Thread(target=_coder_worker, daemon=True)
        task.thread.start()
        return tid, "Coder spawned successfully."

    def kill_task(self, task_id):
        if task_id in self.tasks:
            self.tasks[task_id].kill()
            return f"Task {task_id} kill signal sent."
        return f"Task {task_id} not found."

# ==========================================
# 4. 主控工作流
# ==========================================

def update(args):
    """
    通用、解耦、并发的主入口函数
    """
    workspace_dir = os.path.abspath(args.experiment_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    
    logger.info(f"=== 启动并发实验管家, 工作空间: {workspace_dir} ===")

    # 1. 组装 Prompt
    if hasattr(args, 'orchestrator_prompt_file') and os.path.exists(args.orchestrator_prompt_file):
        with open(args.orchestrator_prompt_file, 'r', encoding='utf-8') as f:
            orchestrator_sys_prompt = f.read()
    else:
        orchestrator_sys_prompt = DEFAULT_ORCHESTRATOR_PROMPT

    # 2. 读取用户请求 (类似 review.txt)
    request_text = "No specific request found."
    req_file = os.path.join(workspace_dir, getattr(args, 'request_file', 'review.txt'))
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            request_text = f.read()
    summary_txt = os.path.join(workspace_dir, "experiment_summary.txt")
    summaries = ""
    if os.path.exists(summary_txt):
        with open(summary_txt, 'r', encoding='utf-8',errors='ignore') as f:
            summaries = f.read()
    
    # 3. 初始化 Agents & TaskManager
    orchestrator = LLMAgent(model=args.orchestrator)
    task_manager = TaskManager(
        max_concurrent=getattr(args, 'max_concurrent_tasks', 2),
        workspace_dir=workspace_dir,
        coder_model_name=args.coder,
        conda_env=getattr(args, 'conda_env', 'base')
    )

    action_history = []
    MAX_ROUNDS = getattr(args, 'max_rounds', 100)
    rounds = 0

    # 4. 核心事件循环
    while rounds < MAX_ROUNDS:
        rounds += 1
        orchestrator.clear_history()  # 严格上下文管理：每轮重组 Prompt

        # ================= 准备上下文 =================
        workspace_tree = get_workspace_state_recursive(workspace_dir, getattr(args, 'max_files_per_dir', 10))
        
        # 提取活动任务日志
        active_tasks = task_manager.get_active_tasks()
        active_tasks_info = ""
        if not active_tasks:
            active_tasks_info = "当前没有正在运行的任务。你可以使用 SPAWN_ 工具发起新任务。"
        else:
            for tid, t in active_tasks.items():
                recent_logs = "".join(list(t.log_history)[-30:]) # 取最后30行
                active_tasks_info += f"\n--- [运行中] {tid} ({t.task_type}) ---\n"
                active_tasks_info += f"最新日志片段:\n{recent_logs}\n"

        # 提取刚刚完成的任务结果
        finished_tasks = task_manager.get_finished_tasks_and_clear()
        finished_tasks_info = ""
        for tid, t in finished_tasks.items():
            finished_tasks_info += f"任务 {tid} 结束。状态: {t.status}. 结果: {t.result_summary}\n"
            action_history.append(f"Async Task {tid} finished with status {t.status}")

        context_prompt = f"【用户的核心请求/意见】\n{request_text}\n\n"
        context_prompt += f"【工作目录结构 (限制展示数)】\n{workspace_tree}\n\n"
        context_prompt += f"【当前运行中的任务监控 (最大并发:{task_manager.max_concurrent})】\n{active_tasks_info}\n\n"
        
        if finished_tasks_info:
            context_prompt += f"【刚刚结束的任务】\n{finished_tasks_info}\n\n"
            
        context_prompt += "【近期执行过的历史动作】\n"
        context_prompt += "\n".join(action_history[-15:]) + "\n\n"
        context_prompt += f"最近执行历史的概述\n{summaries}\n\n"
        
        context_prompt += "请根据上述监控状态和请求，返回你的 JSON 决策。如果你需要时间等待日志输出，请选择 WAIT。"

        # ================= 调用大模型 =================
        try:
            resp, _ = orchestrator.get_response_stream(context_prompt, orchestrator_sys_prompt)
        except Exception as e:
            logger.error(f"Orchestrator API 失败: {e}")
            time.sleep(5)
            continue

        action_json = LLMAgent.robust_extract_json(resp)
        if not action_json:
            logger.warning("未能解析JSON指令。")
            action_history.append("Error: Failed to parse JSON in last turn.")
            continue

        action = action_json.get("Action")
        params = action_json.get("Action_Params", {})
        thoughts = action_json.get("Thoughts", "")
        current_summary = params.get('summary',"")
        summaries += f"Round {rounds}\n: {current_summary}\n"

        logger.info(f"\n[Round {rounds}] Orchestrator: {action} | Thoughts: {thoughts[:100]}...")

        # ================= 执行决策 =================
        if action == "FINISH":
            logger.info("=== 管家确认任务完成 ===")
            logger.info(f"总结: {params.get('summary', '')}")
            break
            
        elif action == "WAIT":
            wait_time = int(params.get("wait_seconds", 10))
            if wait_time < 200:
                wait_time = 200
            logger.info(f"Orchestrator decided to wait for {wait_time}s...")
            time.sleep(min(wait_time, 600))  # 限制最大单次等待时间
            action_history.append(f"Action: WAIT {wait_time}s")
            
        elif action == "KILL_TASK":
            tid = params.get("task_id", "")
            res = task_manager.kill_task(tid)
            action_history.append(f"Action: KILL_TASK {tid} -> {res}")
            
        elif action == "SPAWN_CODER":
            tid, msg = task_manager.spawn_coder(params.get("instruction", ""))
            action_history.append(f"Action: SPAWN_CODER -> {msg} (ID: {tid})")
            
        elif action == "SPAWN_RUN":
            tid, msg = task_manager.spawn_run(params.get("run_script", ""))
            action_history.append(f"Action: SPAWN_RUN -> {msg} (ID: {tid})")
            
        elif action == "READ_FILE":
            fn = params.get("filename", "")
            path = os.path.join(workspace_dir, fn)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                action_history.append(f"Action: READ_FILE {fn}\n {content}\n.")
            else:
                action_history.append(f"Action: READ_FILE {fn} -> File Not Found.")
                
        elif action == "WRITE_FILE":
            files = extract_files_from_response(resp)
            cont_tmp = ""
            if files:
                for fn, cont in files.items():
                    with open(os.path.join(workspace_dir, fn), "w", encoding="utf-8") as f:
                        f.write(cont)
                        cont_tmp += f"{fn}: {cont}\n"
                action_history.append(f"Action: WRITE_FILE -> Wrote {cont_tmp}")
            else:
                action_history.append(f"Action: WRITE_FILE -> Error: Markdown code blocks missing.")

        # 防死循环保护机制：如果所有动作瞬间完成且没有 WAIT，适当休眠1秒避免 token 耗尽
        if action not in ["WAIT", "SPAWN_CODER", "SPAWN_RUN"]:
            time.sleep(1)

    # 收尾：杀死所有依然活着的任务
    for tid in list(task_manager.tasks.keys()):
        task_manager.kill_task(tid)
        
    logger.info("=== 工作流退出 ===")

# ==========================================
# 5. 命令行启动入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent Agentic Orchestrator")
    parser.add_argument("--experiment_dir", type=str, default="./workspace", help="工作目录")
    parser.add_argument("--request_file", type=str, default="review.txt", help="包含用户请求的文件名")
    parser.add_argument("--orchestrator", type=str, default="gemini-3-flash-preview", help="Orchestrator 模型")
    parser.add_argument("--coder", type=str, default="gemini-3-flash-preview", help="Coder 模型")
    parser.add_argument("--max_concurrent_tasks", type=int, default=3, help="最大并发任务数")
    parser.add_argument("--max_files_per_dir", type=int, default=20, help="目录树单层最大显示文件数")
    parser.add_argument("--conda_env", type=str, default="AutoGenOld", help="运行代码的 Conda 环境")
    parser.add_argument("--max_rounds", type=int, default=500, help="管家最多迭代轮数")
    parser.add_argument("--orchestrator_prompt_file", type=str, default="", help="可选：外部传入的Prompt文件路径")
    
    args = parser.parse_args()
    args.experiment_dir = r"D:\\ChannelCoding\\RCOM\\ROCM-float16\\DartsMIMODetector.py"
    
    update(args)


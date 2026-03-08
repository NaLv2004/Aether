import os
import ast
import argparse
from collections import defaultdict, deque

# 从 llm.py 中导入 LLMAgent
from llm import LLMAgent
from utils import setup_logger

logger = setup_logger("experiment_run.log")

def get_python_files(directory):
    """获取指定目录下所有的 .py 文件（忽略隐藏文件和子目录）"""
    py_files = []
    for f in os.listdir(directory):
        if f.endswith('.py') and not f.startswith('.'):
            py_files.append(f)
    return py_files

def parse_dependencies(directory, py_files):
    """
    解析 Python 文件之间的依赖关系。
    如果 A 导入了 B，说明 A 依赖于 B。
    返回:
        dependencies: dict, 记录每个文件依赖了哪些本地文件 {A: [B, C]}
        reverse_dependencies: dict, 记录每个文件被哪些文件依赖 {B: [A], C: [A]}
    """
    # 提取模块名（去掉 .py 后缀）
    module_to_file = {f[:-3]: f for f in py_files}
    local_modules = set(module_to_file.keys())
    
    dependencies = defaultdict(set)
    reverse_dependencies = defaultdict(set)
    
    for file in py_files:
        filepath = os.path.join(directory, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file)
        except Exception as e:
            logger.info(f"Warning: 无法解析 {file} 的语法结构, 错误: {e}")
            continue
            
        for node in ast.walk(tree):
            # 处理 import X
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in local_modules and base_module != file[:-3]:
                        dependencies[file].add(module_to_file[base_module])
                        reverse_dependencies[module_to_file[base_module]].add(file)
            
            # 处理 from X import Y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in local_modules and base_module != file[:-3]:
                        dependencies[file].add(module_to_file[base_module])
                        reverse_dependencies[module_to_file[base_module]].add(file)
                        
    return dependencies, reverse_dependencies

def get_processing_order(py_files, dependencies):
    """
    使用拓扑排序生成处理顺序：从叶子节点（不依赖任何本地文件的文件）开始。
    """
    in_degree = {f: len(dependencies.get(f, [])) for f in py_files}
    
    # 队列中存放的是入度为 0 的文件（即没有依赖任何其他本地文件的叶子节点）
    queue = deque([f for f in py_files if in_degree[f] == 0])
    order = []
    
    # 构建依赖的反向图，便于在叶子节点被处理后，减少依赖它的文件的入度
    # graph: {被依赖文件: [依赖它的文件]}
    graph = defaultdict(list)
    for f, deps in dependencies.items():
        for dep in deps:
            graph[dep].append(f)
            
    while queue:
        current = queue.popleft()
        order.append(current)
        
        # current 已经被处理，依赖 current 的文件入度减 1
        for neighbor in graph.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    # 处理循环依赖的情况
    if len(order) != len(py_files):
        logger.info("警告：检测到循环依赖！剩余文件将按照字母顺序强行处理。")
        remaining = sorted(list(set(py_files) - set(order)))
        order.extend(remaining)
        
    return order

def generate_file_readme(agent, filepath, filename, research_plan, overview):
    """
    调用 agent 为单个 Python 文件生成 README 描述。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        return f"读取 {filename} 失败: {e}"

    system_message = (
        "你是一个顶级的通信领域AI科研助手。你的任务是为一个刚刚生成的Python代码文件编写README文档。\n"
        "请务必使用平实、清晰的语言进行描述。"
    )

    user_message = (
        f"下面是一份【科研计划】、一份【执行概述】以及当前需要你分析的【Python代码文件】。\n\n"
        f"【科研计划】:\n{research_plan}\n\n"
        f"【执行概述】:\n{overview}\n\n"
        f"【当前代码文件 ({filename})】:\n```python\n{code_content}\n```\n\n"
        f"请根据上述信息，为这个Python文件 `{filename}` 生成一份详细的说明文档。你的回答必须严格包含以下三个部分：\n"
        f"1. **计划步骤映射**：描述该文件实现了【科研计划】中的哪一个（或哪些）步骤，它是如何服务于这个步骤的。\n"
        f"2. **总体功能描述**：详细说明这个文件的总体逻辑、在通信系统/仿真中扮演的角色及其核心实现。\n"
        f"3. **命令行调用方式**：严格按照以下要求执行，不要给出任何其他信息：如果该文件具有main函数和argParser进行命令行参数解析，必须给出该文件的命令行调用方式（调用示例中必须准确无误地包含每一个参数），以及命令行调用中每个参数的物理意义；如果该文件没有main函数和ArgParser，则写“该文件不支持命令行调用”\n\n"
        f"4. **期待的输出**：通过在命令行中运行该文件，可以得到哪些输入输出。必须详细，对照代码中的所有print语句逐条列出。如果该文件不可通过命令行调用，这一项填“无”"
        f"注意：只需输出上述三个部分的Markdown内容，不要输出多余的寒暄或开场白。"
    )

    # 关键：每次清空上下文，确保 Agent 上下文中只有当前文件、计划和概述
    agent.clear_history()
    
    logger.info(f"\n========== 开始生成 {filename} 的说明 ==========")
    # 使用带流式输出的接口
    response, _ = agent.get_response_stream(user_message, system_message, print_debug=False)
    logger.info(f"========== {filename} 生成完毕 ==========\n")
    
    return response

def generate_readme(args):
    # work_dir = r"experiments\\20260301_211831"
    # args = parser.parse_args()

    work_dir = args.work_dir
    log_dir = args.log_dir
    
    # 1. 读取科研计划和概述文件
    try:
        with open(args.plan_file, 'r', encoding='utf-8') as f:
            research_plan = f.read()
        with open(args.overview_file, 'r', encoding='utf-8') as f:
            overview = f.read()
    except Exception as e:
        logger.info(f"读取计划或概述文件失败: {e}")
        return

    # 2. 解析目录下所有的 Python 文件及依赖树
    py_files = get_python_files(work_dir)
    if not py_files:
        logger.info(f"在工作目录 {work_dir} 中未找到任何 .py 文件。")
        return
        
    dependencies, _ = parse_dependencies(work_dir, py_files)
    processing_order = get_processing_order(py_files, dependencies)
    
    logger.info(f"依赖解析完成。文件处理顺序(从叶子到根):")
    for idx, f in enumerate(processing_order, 1):
        logger.info(f"  {idx}. {f}")

    # 3. 初始化 LLMAgent
    agent = LLMAgent(model=args.model, temperature=0.3, log_file=os.path.join(log_dir, "readme_agent.log"))

    # 4. 遍历处理所有文件并收集汇总内容
    all_summaries = []
    
    for filename in processing_order:
        filepath = os.path.join(work_dir, filename)
        
        # 针对每个文件调用 LLM
        file_md = generate_file_readme(agent, filepath, filename, research_plan, overview)
        
        # 格式化汇总结果
        formatted_summary = f"## 文件: `{filename}`\n\n{file_md}\n\n" + "-"*80 + "\n"
        all_summaries.append(formatted_summary)

    # 5. 生成最终汇总文档并保存
    final_txt_path = os.path.join(work_dir, "PreviousSummary.txt")
    
    with open(final_txt_path, 'w', encoding='utf-8') as f:
        f.write("# 通信科研项目综合总结与说明文档\n\n")
        
        f.write("## 1. 科研计划 (Research Plan)\n\n")
        f.write(research_plan + "\n\n")
        f.write("="*80 + "\n\n")
        
        f.write("## 2. 执行概述 (Execution Overview)\n\n")
        f.write(overview + "\n\n")
        f.write("="*80 + "\n\n")
        
        f.write("## 3. 代码文件详细说明 (File Details)\n\n")
        f.write("以下文档基于代码依赖关系从底层向上递推生成：\n\n")
        for summary in all_summaries:
            f.write(summary)
            f.write("\n")
            
    logger.info(f"\n>>> 任务完成！完整的综合文档已保存至: {final_txt_path}")

# if __name__ == "__main__":
#     main()
    
    



### `perform_experiments.py` 完整代码：
import os
import re
import json
import time
import queue
import threading
import subprocess
import logging

# 引入您写好的 LLMAgent
from llm import LLMAgent
from utils import setup_logger

logger = setup_logger("experiment_run.log")

# ================= 配置参数与系统提示词 =================

MAX_RETRIES = 30  # 每一步计划最大允许的 Tool Call 轮数

MONITOR_SYSTEM_PROMPT = """你是一个负责监控长期运行任务的 Orchestrator Agent。
你需要观察终端输出和硬件状态，判断程序是否陷入死循环、报错卡死或占用异常。
如果你认为程序正常运行，请返回：{"Action": "CONTINUE"}
如果你认为程序出现严重异常必须立刻杀死，或者当前程序无法在2个小时执行完，请返回：{"Action": "KILL", "Feedback": "你的理由"}
请严格以 JSON 格式返回。"""

EXECUTOR_SYSTEM_PROMPT = """你是一个通信科研团队的高级AI实验执行员(Executor Agent)。
你的任务是根据预先制定的实验计划，通过多次运行当前工作目录下的现有 Python 代码来收集充分的实验数据，以供撰写 IEEE TCOM 级别的论文。
所有的代码都将在特定的 Conda 环境中通过 bat 脚本执行。你只能通过命令行参数运行，绝不允许修改原有的 Python 代码或自行编写新代码。

你可以执行的操作（Tools）包括：
1. `READ_CODE`: 读取特定的代码文件全文，以弄清楚应该传递什么样的命令行参数。
2. `RUN_CODE`: 编写并运行 bat 脚本测试。你可以并且必须传入不同的命令行超参数（如SNR, 天线数, Epoch等）获取多组对比数据。
3. `RECORD_DATA`: 记录并保存当前已经获取的有价值的中间数据。因为单步计划可能需要多次 RUN_CODE，为了防止中途崩溃导致数据丢失，当你通过 RUN_CODE 获取到部分非常好的结果数据时，必须调用此工具总结并写入历史记录。
4. `PASS_STEP`: 当你认为本步骤要求的所有场景和参数的对比数据都已充分获取、记录，且结果能够体现出显著优势、有利于论文发表时，生成该步的最终总结并进入下一步。

【交互格式】
你的回复必须严格包含以下 JSON 结构（被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "思考当前进度：是需要阅读代码了解参数、还是编写 bat 运行测试、还是记录刚才的好数据、或是所有目标达成准备进入下一步。",
    "Action": "READ_CODE | RUN_CODE | RECORD_DATA | PASS_STEP",
    "Action_Params": {
        "filename": "如果 READ_CODE，提供需要读取的文件名",
        "run_script": "如果 RUN_CODE，提供可在 Windows cmd 下运行的完整 bat 脚本内容（不要带路径，假设在当前工作目录执行）",
        "data_summary": "如果 RECORD_DATA，在此详细整理并记录刚刚跑出的关键科研数据（例如不同SNR下的BER、Loss下降曲线数值等，必须具备直接用于画图/制表的完整性）。",
        "final_summary": "如果 PASS_STEP，在此对这一整步的所有实验数据和结论做一个终局总结。"
    }
}
```

【核心要求】：
1. 每次只允许调用一个工具！
2. 涉及AI模型时，如果当前结果不好是因为 epoch、batch_size 等太小，请在 RUN_CODE 时加大这些参数。
3. RUN_CODE 提供的 bat 脚本中绝对禁止包含 `pause` 等会卡死进程的命令。
4. 你需要真正获取足够多的数据（多换几种场景、多扫一些参数点）。不要只跑一次就 PASS_STEP。如果数据不够，请继续换参数 RUN_CODE。
"""

# ================= 底层辅助函数 =================

def get_hardware_status():
    status_info = "【当前硬件资源状态】\n"
    try:
        smi_output = subprocess.check_output("nvidia-smi", shell=True, encoding="utf-8", errors="replace", timeout=5)
        status_info += f"--- nvidia-smi 专用显存与GPU利用率 ---\n{smi_output}\n"
    except Exception as e:
        status_info += f"获取 nvidia-smi 失败: {e}\n"
    try:
        mem_output = subprocess.check_output("wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", shell=True, encoding="utf-8", errors="ignore", timeout=5)
        status_info += f"--- 系统物理内存 (用作共享GPU内存池) ---\n{mem_output.strip()}\n"
    except Exception: pass
    return status_info

def run_command_with_monitoring(script_path, cwd, orchestrator_agent, conda_env_name):
    cmd = f'conda run -n {conda_env_name} --no-capture-output cmd.exe /c "{script_path} & exit"'
    logger.info(f"\n[System] 执行脚本: {script_path} ...")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding='utf-8', errors='replace', text=True, env=env
    )

    output_lines = []
    q = queue.Queue()

    def reader_thread(proc, que):
        for line in iter(proc.stdout.readline, ''): que.put(line)
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

        if current_time - last_monitor_time >= 200:
            last_monitor_time = current_time
            recent_output = "".join(output_lines[-150:])
            status_info = get_hardware_status()
            if recent_output.strip():
                monitor_prompt = f"【实时监控最新控制台输出片段】代码已运行{int(current_time - start_time)}秒，当前输出：\n{recent_output}\n\n宿主机状态：\n{status_info}\n\n请判定程序是否正常运行，是否需要提前终止(KILL)？另外，如果根据当前的执行时间和进度，你预计该程序执行时间将超过2小时，你也必须给出杀死进程（KILL）的指示，并且在FeedBack字段中说明“我们希望程序在2小时以内执行完，当前预估时间为xxx，同时总结当前代码占用的显存资源情况（当前占用量/总量），以指导其他agent修改执行脚本（绝对不能让agent减小epoch数量）”"
                
                orchestrator_agent.clear_history()
                resp, _ = orchestrator_agent.get_response_stream(monitor_prompt, MONITOR_SYSTEM_PROMPT)
                monitor_json = LLMAgent.robust_extract_json(resp)
                
                if monitor_json and monitor_json.get("Action") == "KILL":
                    logger.info("\n[Orchestrator Monitor] 判定异常或超时，下达指令：终止进程！")
                    killed_by_monitor = True
                    monitor_feedback = monitor_json.get("Feedback", "被根据实时输出强制终止。")
                    subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
        time.sleep(1)

    full_stdout = "".join(output_lines)
    success = (not killed_by_monitor) and (process.returncode == 0)
    return success, full_stdout, "", monitor_feedback

def get_directory_structure(rootdir):
    structure = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        level = dirpath.replace(rootdir, '').count(os.sep)
        indent = ' ' * 4 * level
        structure.append(f"{indent}{os.path.basename(dirpath)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in filenames: structure.append(f"{subindent}{f}")
    return "\n".join(structure)

def read_file_content(filepath):
    if not os.path.exists(filepath): return ""
    with open(filepath, "r", encoding="utf-8") as f: return f.read()

# ================= 现场保护与断点机制 =================

def save_state(workspace_dir, step_idx, plan, intermediate_records, tool_calls_history):
    state = {
        "step_idx": step_idx,
        "plan": plan,
        "intermediate_records": intermediate_records,
        "tool_calls_history": tool_calls_history
    }
    path = os.path.join(workspace_dir, "experiment_state.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.warning(f"[State Save] 保存保护文件失败: {e}")

def load_state(workspace_dir):
    path = os.path.join(workspace_dir, "experiment_state.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: pass
    return None

def write_to_history(file_path, content):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ================= ToolManager 工具类 =================

class ToolManager:
    def __init__(self, workspace_dir, conda_env, monitor_agent, history_file):
        self.workspace_dir = workspace_dir
        self.conda_env = conda_env
        self.monitor_agent = monitor_agent
        self.history_file = history_file

    def read_code(self, filename):
        if not filename: return "未提供文件名。"
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path): return f"File '{filename}' does not exist."
        return read_file_content(path)

    def run_code(self, run_script):
        if not run_script: return "脚本为空。"
        bat_path = os.path.join(self.workspace_dir, "run.bat")
        with open(bat_path, "w", encoding="utf-8") as f: f.write(run_script)
        
        success, stdout, _, monitor_feedback = run_command_with_monitoring(bat_path, self.workspace_dir, self.monitor_agent, self.conda_env)
        
        # 截取最新结果防止上下文爆炸，但尽可能多留（尤其是尾部数据）
        out_str = stdout if len(stdout) < 4000 else f"{stdout[:1000]}\n...\n{stdout[-3000:]}"
        res = f"Execute Success: {success}\nConsole Output:\n{out_str}\n"
        if monitor_feedback: res += f"Monitor Feedback: {monitor_feedback}\n"
        return res

    def record_data(self, data_summary, step_info):
        content = f"\n>>> [INTERMEDIATE DATA RECORD] {step_info} <<<\n{data_summary}\n"
        write_to_history(self.history_file, content)
        return "数据已成功记录至 execute_history.txt 中"

# ================= 核心工作流 =================

def plan_and_execute_experiments(args):
    WORKSPACE_DIR = os.path.abspath(args.workspace_dir)
    CONDA_ENV_NAME = args.conda_env_name
    MODEL_NAME = args.model
    MONITOR_MODEL_NAME = args.model
    LOG_DIR = args.log_dir
    PREVIOUS_SUMMARY_FILE = os.path.join(WORKSPACE_DIR, "PreviousSummary.txt")
    EXECUTE_HISTORY_FILE = os.path.join(WORKSPACE_DIR, "execute_history.txt")
    
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    executor_agent = LLMAgent(model=MODEL_NAME, log_file=os.path.join(LOG_DIR, "executor_agent.log"))
    monitor_agent = LLMAgent(model=MONITOR_MODEL_NAME, log_file=os.path.join(LOG_DIR, "monitor_agent.log"))
    monitor_agent.set_context_len(5)
    
    tool_manager = ToolManager(WORKSPACE_DIR, CONDA_ENV_NAME, monitor_agent, EXECUTE_HISTORY_FILE)
    logger.info(f"[System] 初始化完成。工作目录: {WORKSPACE_DIR}")

    dir_structure = get_directory_structure(WORKSPACE_DIR)
    previous_summary = read_file_content(PREVIOUS_SUMMARY_FILE)

    # =============== 尝试热启动加载断点 ===============
    state = load_state(WORKSPACE_DIR)
    # state = None
    if state:
        step_idx = state["step_idx"]
        plan = state["plan"]
        intermediate_records = state.get("intermediate_records", "")
        tool_calls_history = state.get("tool_calls_history", [])
        logger.info(f"=== 发现现场保护配置文件，系统从第 {step_idx + 1} 步热启动恢复执行 ===")
    else:
        step_idx = 0
        intermediate_records = ""
        tool_calls_history = []
        
        # 制定计划
        system_prompt_plan = "你是一个通信科研团队的高级AI实验执行员(Executor Agent)。"
        plan_prompt = f"""
请基于以下工作目录结构和先前的研究总结(PreviousSummary.txt)...
【工作目录结构】：
{dir_structure}

【先前研究总结】：
{previous_summary}

【要求】：
1. 观察之前AI编写的代码及得到的结果，指出不足，决定要进行哪些对比。
2. 至少进行 4 种对比（至少包含复杂度和性能两个层面），每种对比涉及多个不同的仿真场景。
3. 一步（对应一个idx）可以包含多点仿真。
4. 对比对象必须非常明确，且调用的物理量必须是现有代码中规定的。
5. **你只能通过命令行参数运行现有的 Python 文件（通过bat脚本的形式），不能修改 Python 文件本身，不能自行编写新代码。**
6. 严格以 JSON List 格式返回你的计划，不要包含任何 markdown 以外的额外文本。 
7. json的content字段中，还需要严格规定满足什么要求，这个计划才被成功执行。
8. 格式示例：
[
    {{"idx": 1, "content": "比较A和B算法..."}},
    {{"idx": 2, "content": "比较A和B算法..."}}
]
"""
        logger.info("\n[System] 正在请求 Executor Agent 生成实验计划...")
        plan_response, _ = executor_agent.get_response_stream(plan_prompt, system_prompt_plan)
        plan = LLMAgent.robust_extract_json_list(plan_response)
        
        if not plan or not isinstance(plan, list):
            logger.error("\n[错误] Executor 未能返回合法的 JSON 计划，系统退出。")
            return

        logger.info(f"\n[System] 成功解析实验计划，共 {len(plan)} 步。")
        write_to_history(EXECUTE_HISTORY_FILE, "=== 实验执行历史记录 ===\n\n")
        save_state(WORKSPACE_DIR, step_idx, plan, intermediate_records, tool_calls_history)

    # =============== Agentic 执行流 ===============
    while step_idx < len(plan):
        step_item = plan[step_idx]
        actual_idx = step_item.get("idx", step_idx + 1)
        step_content = step_item.get("content", "")
        
        logger.info(f"\n=============================================")
        logger.info(f" 开始执行/恢复计划步骤 {actual_idx}: {step_content}")
        logger.info(f"=============================================")

        step_passed = False
        attempts = len(tool_calls_history)

        while attempts < MAX_RETRIES and not step_passed:
            attempts += 1
            executor_agent.clear_history() # 严控上下文污染
            
            context_prompt = f"""
你正在执行实验计划的第 {actual_idx} 步。
【当前计划要求】: {step_content}
【工作目录结构】:
{dir_structure}

【先前研究总结 (背景参考)】:
{previous_summary}

【本步已记录的安全中间数据】:
{intermediate_records if intermediate_records else "暂无记录的数据。"}

【本步你已执行的动作历史】:
"""
            if not tool_calls_history:
                context_prompt += "暂无。这是本步骤第一轮操作。\n"
            else:
                for th in tool_calls_history:
                    # res_trunc = th['result'] if len(str(th['result'])) < 1500 else str(th['result'])[-1500:] + "\n...(truncated)"
                    res_trunc = th['result']
                    context_prompt += f"Action: {th['action']}, Params: {json.dumps(th.get('params', {}), ensure_ascii=False)}\nResult:\n{res_trunc}\n\n"

            context_prompt += "请根据上述信息决定下一步 Action。如果获取了较多好数据请及时 RECORD_DATA；如果目标全部完成请 PASS_STEP。"
            
            logger.info(f"\n--- [Step {actual_idx}] Executor 主动权第 {attempts}/{MAX_RETRIES} 轮 ---")
            resp, _ = executor_agent.get_response_stream(context_prompt, EXECUTOR_SYSTEM_PROMPT)
            action_json = LLMAgent.robust_extract_json(resp)
            
            if not action_json:
                tool_calls_history.append({"action": "PARSE_ERROR", "params": {}, "result": "Failed to parse JSON."})
                save_state(WORKSPACE_DIR, step_idx, plan, intermediate_records, tool_calls_history)
                continue

            action = action_json.get("Action")
            params = action_json.get("Action_Params", {})

            if action == "READ_CODE":
                res = tool_manager.read_code(params.get("filename", ""))
                tool_calls_history.append({"action": action, "params": params, "result": res})
                logger.info(f"[Executor] 读取了代码文件: {params.get('filename', '')}")
                
            elif action == "RUN_CODE":
                logger.info(f"[Executor] 运行 bat 脚本测试...")
                res = tool_manager.run_code(params.get("run_script", ""))
                tool_calls_history.append({"action": action, "params": params, "result": res})
                
            elif action == "RECORD_DATA":
                logger.info(f"[Executor] 主动总结并记录中间数据至历史...")
                summary = params.get("data_summary", "")
                res = tool_manager.record_data(summary, f"Step {actual_idx} (Attempt {attempts})")
                intermediate_records += f"\n- {summary}\n"
                tool_calls_history.append({"action": action, "params": params, "result": res})
                
            elif action == "PASS_STEP":
                logger.info(f"[Executor] 步骤验收通过！")
                final_summary = params.get("final_summary", f"步骤 {actual_idx} 已完成。")
                
                # 落盘最终总结
                content = f"\n=== Plan Step {actual_idx} PASS ===\n【本步最终总结】:\n{final_summary}\n"
                write_to_history(EXECUTE_HISTORY_FILE, content)
                
                step_passed = True
                
            else:
                tool_calls_history.append({"action": action, "params": params, "result": f"Unknown Action: {action}"})
                logger.warning(f"[Executor] 未知指令： {action}")

            # 无脑落盘保存状态，防止下一秒断电
            if not step_passed:
                save_state(WORKSPACE_DIR, step_idx, plan, intermediate_records, tool_calls_history)

        # 步进处理
        if step_passed:
            step_idx += 1
            tool_calls_history = []
            intermediate_records = ""
            save_state(WORKSPACE_DIR, step_idx, plan, intermediate_records, tool_calls_history)
        else:
            logger.error(f"\n[System Error] 步骤 {actual_idx} 在 {MAX_RETRIES} 次尝试后仍然未能通过。强行跳过。")
            write_to_history(EXECUTE_HISTORY_FILE, f"\n=== Plan Step {actual_idx} (FAILED/SKIPPED) ===\n")
            step_idx += 1
            tool_calls_history = []
            intermediate_records = ""
            save_state(WORKSPACE_DIR, step_idx, plan, intermediate_records, tool_calls_history)

    logger.info("\n[System] =========================================")
    logger.info("[System] 所有实验计划已执行完毕，完整总结已存入:")
    logger.info(f"[System] {EXECUTE_HISTORY_FILE}")
    logger.info("[System] =========================================")

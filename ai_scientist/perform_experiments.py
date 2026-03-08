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
    
    

import os
import re
import json
import time
import queue
import threading
import subprocess

# 引入您写好的 LLMAgent
from llm import LLMAgent

# ================= 配置参数 =================


# 监控系统的 System Prompt
MONITOR_SYSTEM_PROMPT = """你是一个负责监控长期运行任务的 Orchestrator Agent。
你需要观察终端输出和硬件状态，判断程序是否陷入死循环、报错卡死或占用异常。
如果你认为程序正常运行，请返回：{"Action": "CONTINUE"}
如果你认为程序出现严重异常必须立刻杀死，或者当前程序无法在2个小时执行完，请返回：{"Action": "KILL", "Feedback": "你的理由"}
请严格以 JSON 格式返回。"""

# ================= 辅助函数 =================

def get_hardware_status():
    """获取 Windows 系统的 GPU 和 内存状态"""
    status_info = "【当前硬件资源状态】\n"
    try:
        smi_output = subprocess.check_output(
            "nvidia-smi", shell=True, encoding="utf-8", errors="replace", timeout=5
        )
        status_info += f"--- nvidia-smi 专用显存与GPU利用率 ---\n{smi_output}\n"
    except Exception as e:
        status_info += f"获取 nvidia-smi 失败: {e}\n"
        
    try:
        mem_output = subprocess.check_output(
            "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", 
            shell=True, encoding="utf-8", errors="ignore", timeout=5
        )
        status_info += f"--- 系统物理内存 (用作共享GPU内存池) ---\n{mem_output.strip()}\n"
    except Exception:
        pass
    return status_info

def run_command_with_monitoring(script_path, cwd, orchestrator_agent, CONDA_ENV_NAME):
    """
    在指定工作目录下的 Conda 环境中执行 bat 脚本，并每隔 200 秒交由Orchestrator监控。
    支持Orchestrator中断死循环/错误进程。
    返回: success(bool), stdout(str), stderr(str), monitor_feedback(str)
    """
    cmd = f'conda run -n {CONDA_ENV_NAME} --no-capture-output cmd.exe /c "{script_path} & exit"'
    logger.info(f"\n[System] 执行脚本: {script_path} ...")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding='utf-8', errors='replace', text=True, env=env
    )

    output_lines = []
    q = queue.Queue()

    def reader_thread(proc, que):
        for line in iter(proc.stdout.readline, ''):
            que.put(line)
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
            except queue.Empty:
                break

        if process.poll() is not None:
            while not q.empty():
                try:
                    line = q.get_nowait()
                    output_lines.append(line)
                except queue.Empty:
                    break
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
                logger.info(f"\n[System] 代码执行已历时 {int(current_time - start_time)} 秒。正在截取最新输出交由 Orchestrator 实时监控...")
                monitor_prompt = f"【实时监控最新控制台输出片段】代码运行已经历时{int(current_time - start_time)} 秒，当前输出：\n{recent_output}\n\n宿主机硬件状态：\n{status_info}\n\n请判定程序是否正常运行，是否需要提前终止(KILL)？另外，如果根据当前的执行时间和进度，你预计该程序执行时间将超过2小时，你也必须给出杀死进程（KILL）的指示，并且在FeedBack字段中说明“我们希望程序在2小时以内执行完，当前预估时间为xxx，同时总结当前代码占用的显存资源情况（当前占用量/总量），以指导其他agent修改执行脚本（绝对不能让agent减小epoch数量）”"
                
                resp, _ = orchestrator_agent.get_response_stream(monitor_prompt, MONITOR_SYSTEM_PROMPT)
                monitor_json = LLMAgent.robust_extract_json(resp)
                
                if monitor_json and monitor_json.get("Action") == "KILL":
                    logger.info("\n[Orchestrator Monitor] 判定程序运行异常，下达指令：终止进程！")
                    killed_by_monitor = True
                    monitor_feedback = monitor_json.get("Feedback", "Orchestrator 根据实时输出强制终止了运行。")
                    subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                else:
                    logger.info("[Orchestrator Monitor] 判定程序正常，允许继续运行。")
            else:
                logger.info(f"[System] 代码执行历时 {int(current_time - start_time)} 秒。暂无新输出。")

        time.sleep(1)

    full_stdout = "".join(output_lines)
    
    if killed_by_monitor:
        return False, full_stdout, "", monitor_feedback
    else:
        success = process.returncode == 0
        return success, full_stdout, "", ""


def get_directory_structure(rootdir):
    """获取目录结构的字符串表示"""
    structure = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        # 排除隐藏文件夹和 pycache
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        level = dirpath.replace(rootdir, '').count(os.sep)
        indent = ' ' * 4 * level
        structure.append(f"{indent}{os.path.basename(dirpath)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in filenames:
            structure.append(f"{subindent}{f}")
    return "\n".join(structure)

def read_file_content(filepath):
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def extract_python_files_from_bat(bat_content):
    """通过正则从 bat 脚本内容中提取出调用的 python 文件名"""
    py_files = set(re.findall(r'(\b\w+\.py\b)', bat_content))
    return list(py_files)


# ================= 核心执行流程 =================

def plan_and_execute_experiments(args):
    # args = parser.parse_args()
    WORKSPACE_DIR = args.workspace_dir    # 指定的工作目录（请根据需要修改）
    CONDA_ENV_NAME = args.conda_env_name            # 运行代码所在的 Conda 环境名称
    MODEL_NAME = args.model             # Executor 主节点使用的模型
    MONITOR_MODEL_NAME = args.model # Orchestrator 监控节点使用的模型
    LOG_DIR = args.log_dir
    PREVIOUS_SUMMARY_FILE = os.path.join(WORKSPACE_DIR, "PreviousSummary.txt")
    EXECUTE_HISTORY_FILE = os.path.join(WORKSPACE_DIR, "execute_history.txt")
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    # 1. 实例化 Agent
    executor_agent = LLMAgent(model=MODEL_NAME, log_file=os.path.join(LOG_DIR, "executor_agent.log"))
    orchestrator_agent = LLMAgent(model=MONITOR_MODEL_NAME, log_file=os.path.join(LOG_DIR, "orchestrator_agent.log"))
    orchestrator_agent.set_context_len(5)
    # orchestrator_agent = LLMAgent(model=MONITOR_MODEL_NAME, log_file="orchestrator_agent.log")

    logger.info(f"[System] 初始化完成。工作目录: {WORKSPACE_DIR}")

    # 2. 读取文件结构和 PreviousSummary
    dir_structure = get_directory_structure(WORKSPACE_DIR)
    previous_summary = read_file_content(PREVIOUS_SUMMARY_FILE)

    if not previous_summary:
        logger.info(f"[警告] 未在工作目录找到 {PREVIOUS_SUMMARY_FILE}，或者文件为空。请确认。")

    # 3 & 4. 制定仿真计划
    system_prompt_plan = "你是一个通信科研团队的高级AI实验执行员(Executor Agent)。"
    
    plan_prompt = f"""
请基于以下工作目录结构和先前的研究总结(PreviousSummary.txt)（其中包含该研究的idea,该研究的详细计划，以及基于该计划AI编写的python程序和程序执行日志），思考哪些地方对当前idea的论文发表不利。
然后，重新设计仿真参数，获取新的数据。注意，虽然之前已经有一些初步的结果，但是你所设计的仿真组数必须足够多，以支撑一篇IEEE TCOM需要的数据量。
之前的仿真结果中，一些数据可能受训练轮数或其他参数影响而效果不好，你必须重新考虑如何设计仿真场景（执行那些代码的AI对自己过于宽松，还常常用“完美”等极端词语掩盖不好的地方，请谨慎甄别）

【工作目录结构】：
{dir_structure}

【先前研究总结】：
{previous_summary}

【要求】：
1. 观察之前AI编写的代码及得到的结果，指出不足，决定要进行哪些对比。
2. 至少进行 4 种对比（至少包含复杂度和性能两个层面），每种对比涉及多个不同的仿真场景。
3. 一步（对应一个idx）可以包含多点仿真（不仅仅是SNR变化，还可以是SNR变化的同时其他参数变化。因为可以编写bat脚本，所以可以通过脚本语言，控制一个参数取多个值（大于等于3种取值）时，多次循环运行程序）。
   如果某一步要研究参数A的影响，建议研究多个不同的A的影响，而不是只改变一次A。
4. 对比对象必须非常明确，且调用的物理量必须是现有代码中规定的。
5. **你只能通过命令行参数运行现有的 Python 文件（通过bat脚本的形式），不能修改 Python 文件本身，不能自行编写新代码。**
6. 严格以 JSON List 格式返回你的计划，不要包含任何 markdown 以外的额外文本。 
7. json的content字段中，还需要严格规定满足什么要求，这个计划才被成功执行。
8. 科研计划中，创新点往往是逐层递进的。对比图中除了将最终形态和baseline对比，还可以将中间状态和baseline对比
9. 有的文件本身包含了多组对比，但往往写的比较死。除了运行这些文件之外，强烈建议你多运行中间文件，通过汇总多组运行结果形成对比
10. 科研计划中，不需要显式包含要执行的命令。写清楚场景即可。
11.涉及AI模型时，除了训练集大小，epoch数量等可调，如果模型规模被暴露为命令行参数，那也是可以调整的。
12.涉及AI模型时，注意很多当前的结果不好是因为epoch，batch_size，网络规模等设置太小。凡是涉及AI模型，你提出的多组参数设定必须比当前更好（而且各模型自身的规模要严格统一（即：如果模型A选了3种不同规模进行测试，模型B也至少要包含这3种不同规模）），以期获得更好的效果。
13.测试集的大小必须足够大。每一步中，都要给出每个参数的详细取值方式，不得存在类似于“某些参数第二步取值方式和第一步对齐”的说法，必须直接给出定义。
14.为了避免执行时间太长，模型训练集大小不要超过20000
格式示例：
[
    {{"idx": 1, "content": "比较A和B算法的复杂度，天线比为16 x 8, SNR取-5~15dB，参数M取0，1，2，3"}},
    {{"idx": 2, "content": "比较A和B算法的性能，天线比为16 x 8, SNR取-5~15dB，参数M取0，1，2，3"}}
]
"""

    logger.info("\n[System] 正在请求 Executor Agent 生成实验计划...")
    plan_response, _ = executor_agent.get_response_stream(plan_prompt, system_prompt_plan)
    
    plan = LLMAgent.robust_extract_json_list(plan_response)
    logger.info(plan)
    if not plan or not isinstance(plan, list):
        logger.info("\n[错误] Executor 未能返回合法的 JSON 计划，系统退出。")
        return

    logger.info(f"\n[System] 成功解析实验计划，共 {len(plan)} 步。")
    
    # 清空之前的执行历史文件
    with open(EXECUTE_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write("=== 实验执行历史记录 ===\n\n")

    # 5. 逐点执行计划
    for step_item in plan:
        executor_agent.clear_history()
        step_idx = step_item.get("idx")
        # if (step_idx == 1 or step_idx == 2 or step_idx == 3) :
        #     continue
        step_content = step_item.get("content")
        logger.info(f"\n=============================================")
        logger.info(f" 开始执行计划步骤 {step_idx}: {step_content}")
        logger.info(f"=============================================")

        step_passed = False
        attempt_count = 0
        run_bat_content = ""

        # 单步执行循环（处理重试或错误修复）
        while not step_passed and attempt_count < 10: # 防止死循环，最大尝试3次
            attempt_count += 1
            executor_agent.clear_history() # 每次对话前清理历史，避免上下文爆炸

            execute_history = read_file_content(EXECUTE_HISTORY_FILE)

            action_prompt = f"""

你正在执行实验计划的第 {step_idx} 步。
【当前计划要求】: {step_content}
【完整计划列表】: {json.dumps(plan, ensure_ascii=False)}

【之前积累的执行历史】:
{execute_history}
【先前研究总结】:
{previous_summary}

任务：
你是一个通信科研团队的高级AI实验执行员(Executor Agent)。
我们正在为一篇准备投稿至IEEE TCOM的论文运行实验，并提供数据，目标是获取有利于idea发表的数据。之前，你通过阅读idea和对之前工作的总结，形成了一份获取仿真数据的计划
现在，你需要根据该计划编写可执行的bat脚本。
你正在执行实验计划的第 {step_idx} 步。
【当前计划要求】: {step_content}
请编写一个可以在 Windows 的 cmd 下运行的批处理脚本内容 (run.bat)，通过传入合适的命令行参数调用工作目录下的 python 文件来完成第 {step_idx} 步所需的仿真。
注意：
1. 你不能修改原本的 python 文件。
2. 将你的批处理内容放在 ```bat 和 ``` 之间。例如：
3. 不要包含任何路径信息，假设当前目录就是工作目录。运行python文件时，直接用文件名。例如，python main.py 而不是python workspace\\main.py。
```bat
python main_simulation.py --algo A --antennas 16x8
```
4. bat脚本中禁止包含任何pause指令，或功能相似的会使得进程卡死的指令。我们会通过程序自动捕捉输出，你不需要操心。
5. 涉及AI模型时，除了训练集大小，epoch数量等可调，如果模型规模被暴露为命令行参数，那也是可以调整的。
6. 每步可以包含多点仿真（不仅仅是SNR变化，还可以是SNR变化的同时其他参数变化）。要保证实验数据尽可能充足。
"""
            logger.info(f"\n[System] 步骤 {step_idx} (尝试 {attempt_count}): 正在生成 run.bat 脚本...")
            bat_response, _ = executor_agent.get_response_stream(action_prompt, "请按照要求编写或者修改脚本")
            
            # 提取 bat 内容
            bat_match = re.search(r"```bat(.*?)```", bat_response, re.DOTALL | re.IGNORECASE)
            if not bat_match:
                logger.info("[警告] 未匹配到 bat 脚本内容，退回重试。")
                continue
            
            run_bat_content = bat_match.group(1).strip()
            bat_file_path = os.path.join(WORKSPACE_DIR, "run.bat")
            with open(bat_file_path, "w", encoding="utf-8") as f:
                f.write(run_bat_content)

            # 运行命令
            success, stdout, _, monitor_feedback = run_command_with_monitoring(bat_file_path, WORKSPACE_DIR, orchestrator_agent, CONDA_ENV_NAME)

            if not success:
                # 运行报错或被终止，收集相关 Python 代码以供 Executor 修复
                logger.info(f"\n[System] 步骤 {step_idx} 执行失败。正在让 Executor 修复...")
                error_msg = stdout if not monitor_feedback else (stdout + "\n[监控器干预]: " + monitor_feedback)
                
                # 提取它刚刚运行了哪些 python 文件
                py_files = extract_python_files_from_bat(run_bat_content)
                py_contents_str = ""
                for py_file in py_files:
                    path = os.path.join(WORKSPACE_DIR, py_file)
                    py_contents_str += f"\n--- {py_file} ---\n{read_file_content(path)}\n"

                fix_prompt = f"""
你刚才生成的 run.bat 执行失败了(可能是因为本身报错，也可能是因为监视其执行的agent认为其执行需要的时间太长)。
【你的脚本】:
{run_bat_content}

【控制台报错与输出】:
{error_msg}

【脚本中涉及的 Python 代码实际内容】:
{py_contents_str}

请仔细阅读报错和 Python 代码内容。不要修改 Python 代码。
调整 run.bat 中的命令行调用参数来修复这个问题，并将修复后的完整脚本放在 ```bat 和 ``` 之间返回。
"""    
                # executor_agent.clear_history()
                fix_resp, _ = executor_agent.get_response_stream(fix_prompt, "你是 Debug 专家，请根据报错和源码调整命令行参数。")
                
                # 更新 bat 内容供下一次尝试
                bat_match = re.search(r"```bat(.*?)```", fix_resp, re.DOTALL | re.IGNORECASE)
                if bat_match:
                    run_bat_content = bat_match.group(1).strip()
                    with open(bat_file_path, "w", encoding="utf-8") as f:
                        f.write(run_bat_content)

                    logger.info(f"[System]提取出修复的bat文件，正在重新运行")
                    success, stdout, _, monitor_feedback = run_command_with_monitoring(bat_file_path, WORKSPACE_DIR, orchestrator_agent, CONDA_ENV_NAME)
            if not success:      
                continue # 进入下一次 Attempt 进行重跑

            else:
                # 运行成功，提取数据并判断是否通过
                logger.info(f"\n[System] 步骤 {step_idx} 执行成功。正在请求 Executor 提取数据并评估结果...")
                eval_prompt = f"""
你刚才生成的 run.bat 成功执行了。
【你的脚本】:
{run_bat_content}

【执行输出内容】 (可能包含大量仿真日志，请仅提取具有的论文中适合展示的，结果性的仿真性能/复杂度结果用于最终论文撰写):
{stdout}

请执行以下任务：
1. 提取本次执行中有用的结果性数据。
2. 判断当前的运行结果是否达到了该步的实验目的（例如优势是否显著，参数是否合理）。
3. 决定是直接结束当前计划(PASS)，还是换一组参数重新执行(RETRY)（标准是是否达到了计划中说明的这一步的执行目标，是否有利于论文发表）。
注意：
涉及AI模型时，除了训练集大小，epoch数量等可调，如果模型规模被暴露为命令行参数，那也是可以调整的。
每步可以包含多点仿真（不仅仅是SNR变化，还可以是SNR变化的同时其他参数变化）。要保证实验数据尽可能充足。

请严格遵守 JSON 格式返回结果：
{{
    "status": "PASS",  // 或者是 "RETRY"
    "extracted_data": "你提取的对论文撰写有用的数据。（先根据你看到的执行结果，以列表形式返回完整的，可以直接在论文中被做成图表的数据（loss等中间结果不需要）,再对数据作简要解释。重点是返回的数据列表清晰完整）",
    "reason": "你做出上述判断的理由"
    
}}
"""
                executor_agent.clear_history()
                eval_resp, _ = executor_agent.get_response_stream(eval_prompt, "你是数据分析和论文撰写专家。")
                eval_json = LLMAgent.robust_extract_json(eval_resp)
                
                if not eval_json:
                    logger.info("[警告] 解析结果评估 JSON 失败，默认通过并保存。")
                    eval_json = {"status": "PASS", "extracted_data": "未能成功结构化提取，请检查日志", "reason": "解析异常"}

                if eval_json.get("status", "").upper() == "PASS":
                    logger.info(f"\n[System] Executor 评估满意 (PASS)。提取数据并记录历史...")
                    # 写入执行历史
                    with open(EXECUTE_HISTORY_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n=== Plan Step {step_idx}: {step_content} ===\n")
                        f.write(f"【执行脚本】:\n{run_bat_content}\n")
                        f.write(f"【提取的结果与结论】:\n{eval_json.get('extracted_data', '')}\n")
                    step_passed = True # 退出当前步骤的循环，进入下个计划点
                else:
                    logger.info(f"\n[System] Executor 评估不满意 (RETRY)。理由: {eval_json.get('reason')}")
                    # 不改变 step_passed，继续循环，让其重试
                    
        if not step_passed:
            logger.info(f"\n[警告] 步骤 {step_idx} 达到最大重试次数仍未能成功，系统强行记录并跳过。")
            with open(EXECUTE_HISTORY_FILE, "a", encoding="utf-8") as f:
                 f.write(f"\n=== Plan Step {step_idx} (FAILED/SKIPPED) ===\n内容: {step_content}\n")

    logger.info("\n[System] =========================================")
    logger.info("[System] 所有实验计划已执行完毕，完整总结已存入:")
    logger.info(f"[System] {EXECUTE_HISTORY_FILE}")
    logger.info("[System] =========================================")


# if __name__ == "__main__":
#     main()
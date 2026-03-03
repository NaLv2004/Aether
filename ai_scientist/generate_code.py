
import json
import os
import re
import argparse
import datetime
import subprocess
import shutil
import time
import threading
import queue

# 导入已经实现的 LLMAgent
from llm import LLMAgent

# ==========================================
# 1. 配置与全局常量
# ==========================================
CONDA_ENV_NAME = "AutoGenOld"  # 指定的 Conda 环境名称
MAX_RETRIES = 10        # 每一步最大失败重试次数，超过则回退

# ==========================================
# 2. 系统提示词定义 (System Prompts)
# ==========================================

ORCHESTRATOR_SYSTEM_PROMPT = """
你是一个科研项目管家 (Orchestrator Agent)。你的任务是管理并推进一个预先制定好的研究计划。
为了完成任务，你可以指导一个AI程序员 (Coding Agent) 编写代码，也可以自行调整参数运行代码。

你可以通过输出特定的 JSON 格式来执行以下四种【操作/决策】：
1. 【指导编程】: 输出提示词让 Coding Agent 开始写代码或修改逻辑。 (Action: PROMPT_CODER)
2. 【自行运行代码】: 当代码已经写好，你想要换一组参数重新运行现有的代码，或者需要亲自执行某条命令测试时，你可以自行编写 bat 脚本执行。系统会直接运行它并将结果交给你验收。 (Action: RUN_CODE)
3. 【验收通过并总结】: 当代码运行成功，并且你查看代码和运行结果后认为“这一步的目标已完美实现”时，生成总结并进入下一步。 (Action: PASS_STEP)
4. 【打回重做】: 虽然代码没有报错，但逻辑不对或没有达到本步骤预期成果，你需要给出修改意见让 Coder 继续修改。 (Action: REJECT_STEP)

【交互格式】
你的回复必须严格包含以下 JSON 结构（被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "思考当前处于什么阶段，分析代码运行结果或代码逻辑是否满足当前步骤的预期。",
    "Action": "PROMPT_CODER 或者 RUN_CODE 或者 PASS_STEP 或者 REJECT_STEP",
    "Coder_Prompt": "如果Action是PROMPT_CODER或REJECT_STEP，这里写给Coding Agent的具体编程/修改指令；否则留空。",
    "Run_Script": "如果Action是RUN_CODE，这里写你需要系统执行的完整 bat 脚本内容（例如：python main.py --lr 0.01 --epochs 10）；否则留空。",
    "Summary": "如果Action是PASS_STEP，这里写对这一步完成情况的详细总结（将用于压缩上下文）；否则留空。"
}
```
【核心要求】：
1.你只需要关注逻辑和最终结果，语法错误系统会自动让Coder去修复。此外，因为我们只会返回给你代码运行的控制台输出。
2.你必须在Coder_Prompt中明确要求Coder在代码中增加足够的 print() 语句，把关键结果输出到控制台，以便你进行验收判断。同时注意：你让Coder输出的内容必须具体，具有实际的物理意义（对你的判断和后续研究有指导意义）。
3.绝对不允许存在将中间结果保存到文件中再在后续步骤读取的情况。你必须明确指示 Coder 从前面生成的代码中直接 import 需要的函数/模块来获取数据或执行计算。不允许要求Coder在代码中有任何文件读写操作（即便给你的计划中这么说了）
4.忽略科研计划中关于需要输出其他中间结果（如pth）的建议，你必须从控制台输出判断一切。
5.你不能完全信任给你的代码执行计划（因为那些计划也是由AI给出的）。你必须独立思考，确认该步骤的目标（不局限于计划中的目标）确实完成。你必须确保以下几点：
  a) 如果涉及和之前的运行结果对比来观察当前结果是否合理，你必须保证仿真场景参数完全相同，如果Coder给出的run.bat的仿真参数和之前并非完全相同，你需要使用你运行代码的权限，重新定义运行脚本（其中包含仿真参数）
  b) 你必须做到足够严格和谨慎，因为Coder是一个AI，你不能在完全确定当前结果正常之前跳过当前步骤。如果你认为当前结果因为各种原因效果不够好，必须重新定义仿真脚本（如果你认为是某些仿真参数设置不合理导致）或者让Coder修改代码
  c) Summary中，重点包含目前已实现的所有代码并简述其作用，并重点描述当前的仿真场景，仿真参数和详细的仿真结果。
"""

CODER_SYSTEM_PROMPT = """
你是一个顶级的 AI 程序员 (Coding Agent)。你的任务是根据主管 (Orchestrator) 的需求编写功能完善的 Python 代码和执行脚本。
所有的代码都将在一个名为 `{conda_env}` 的 Conda 环境中执行。

【强制输出格式】
你必须以 Markdown 代码块的形式输出你需要创建或修改的文件，并且必须在代码块上方标明文件名，例如：

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
实现了输出 Hello World。
## 运行示例与参数说明
`python main.py --epochs 10`
- `--epochs`: 训练的轮数。
```

【核心要求】
1. 每次必须输出完整、可运行的代码，不要省略任何部分（不要使用"// 此处省略"）。如果你被要求修改某个文件，你也必须返回那个文件的完整、可运行的版本。
2. 必须输出 `run.bat`，里面包含安装所需依赖 (pip install) 和运行主程序的命令。
3. 必须输出 `readme.md` 详细解释当前代码的功能。
4. 尽可能在代码中增加 `print()` 语句，把关键的中间结果输出到控制台，以便 Orchestrator 验收。
5. 在给你的环境中已经有torch和numpy（不论你是否看到），切勿重复安装！
6. bat脚本中，只允许出现pip install和用python运行某个文件，切勿使用pause等使得进程卡死。我们会自动捕捉输出，你不需要操心。
7. 所有文件名和`run.bat`中涉及的文件，不要使用任何具体的绝对或者相对路径，只能出现文件名(例如，写python main.py)。
8. 【参数暴露要求】：你必须在代码中尽可能多且必要地使用 argparser 暴露参数（例如学习率，epoch数，隐藏层维度，批次大小等）到命令行。同时，必须在 readme.md 中给出详细的运行示例，并逐一解释每个参数的具体意义，以便 Orchestrator 后续通过修改启动参数来控制实验。
9. 切记：如果涉及AI模型训练，必须明确分割训练集和测试集，且每个epoch训练的测试集必须相同（在所有epoch之前预先计算好），切勿出现每个epoch使用不同训练集的情况。
10. 不能因为这次代码比某次非常不成功的运行好了很多就通过验收，必须达到你心中认为达到目标的客观标准才可以通过。
"""

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
注意：你给Coder提供的代码修复建议最好是针对run.bat脚本中部分参数的修改建议，最好不要涉及python代码的修改，除非你很确定问题是由python代码本身导致的。
```
"""

# ==========================================
# 3. 辅助功能函数
# ==========================================
def get_hardware_status():
    """获取 Windows 系统的 GPU 和 内存状态"""
    status_info = "【当前硬件资源状态】\n"
    
    # 1. 获取常规 GPU 状态 (包含专用显存 VRAM 和 GPU 利用率)
    try:
        smi_output = subprocess.check_output(
            "nvidia-smi", 
            shell=True, 
            encoding="utf-8", 
            errors="replace",
            timeout=5
        )
        status_info += f"--- nvidia-smi 专用显存与GPU利用率 ---\n{smi_output}\n"
    except Exception as e:
        status_info += f"获取 nvidia-smi 失败: {e}\n"
        
    # 2. 获取系统物理内存状态 (Windows 的'共享GPU内存'是由物理内存分配的，监控此项可反映共享显存占用情况)
    try:
        mem_output = subprocess.check_output(
            "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", 
            shell=True, 
            encoding="utf-8", 
            errors="ignore",
            timeout=5
        )
        # wmic 返回的单位是 KB
        status_info += f"--- 系统物理内存 (用作共享GPU内存池) ---\n{mem_output.strip()}\n"
    except Exception as e:
        pass
        
    return status_info

def run_command_with_monitoring(script_path, cwd, orchestrator_agent):
    """
    在指定工作目录下的 Conda 环境中执行 bat 脚本，并每隔1分钟交由Orchestrator监控。
    支持Orchestrator中断死循环/错误进程。
    返回: success(bool), stdout(str), stderr(str), monitor_feedback(str)
    """
    cmd = f'conda run -n {CONDA_ENV_NAME} --no-capture-output cmd.exe /c "{script_path} & exit"'
    print(f"\n[System] 执行脚本: {script_path} ...")
    
    # 启动子进程，将 stderr 合并到 stdout 以便监控
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        errors='replace',
        text=True,
        env = env
    )

    output_lines = []
    q = queue.Queue()

    # 后台线程读取输出，防止阻塞
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
        # 收集当前可用的输出
        while not q.empty():
            try:
                line = q.get_nowait()
                output_lines.append(line)
            except queue.Empty:
                break

        # 检查进程是否结束
        if process.poll() is not None:
            # 读取剩余所有输出
            while not q.empty():
                try:
                    line = q.get_nowait()
                    output_lines.append(line)
                except queue.Empty:
                    break
            break

        # 检查是否触发生命周期超时 (硬超时防死锁)
        current_time = time.time()
        if current_time - start_time > 3600: # 1小时硬超时
            killed_by_monitor = True
            monitor_feedback = "进程运行超过1小时硬性超时限制，已被系统强制杀死。"
            subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break

        # 实时监控逻辑：每隔 60 秒触发 Orchestrator 检查
        if current_time - last_monitor_time >= 200: # 200s 监控一次
            last_monitor_time = current_time
            # 获取最近的输出，避免上下文过长
            recent_output = "".join(output_lines[-150:])
            # 获取当前硬件状态，提供给 Orchestrator 以辅助判断
            status_info = get_hardware_status()
            if recent_output.strip():
                print(f"\n[System] 代码执行已历时 {int(current_time - start_time)} 秒。正在截取最新输出交由 Orchestrator 实时监控...")
                monitor_prompt = f"【实时监控最新控制台输出片段】代码运行已经历时{int(current_time - start_time)} 秒，当前输出：\n{recent_output}\n\n宿主机硬件状态：\n{status_info}\n\n请判定程序是否正常运行，是否需要提前终止(KILL)？"
                
                resp, _ = orchestrator_agent.get_response_stream(monitor_prompt, MONITOR_SYSTEM_PROMPT)
                monitor_json = LLMAgent.robust_extract_json(resp)
                
                if monitor_json and monitor_json.get("Action") == "KILL":
                    print("\n[Orchestrator Monitor] 判定程序运行异常，下达指令：终止进程！")
                    killed_by_monitor = True
                    monitor_feedback = monitor_json.get("Feedback", "Orchestrator 根据实时输出强制终止了运行。")
                    # 在 Windows 下强杀进程树
                    subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                else:
                    print("[Orchestrator Monitor] 判定程序正常，允许继续运行。")
            else:
                print(f"[System] 代码执行历时 {int(current_time - start_time)} 秒。暂无新输出。")

        time.sleep(1) # 降低CPU占用

    full_stdout = "".join(output_lines)
    
    if killed_by_monitor:
        return False, full_stdout, "", monitor_feedback
    else:
        success = process.returncode == 0
        return success, full_stdout, "", ""


def get_installed_packages():
    """获取 conda 环境中的 pip 包列表"""
    cmd = f'conda run -n {CONDA_ENV_NAME} pip list'
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
        return result.stdout if result.returncode == 0 else "Failed to get pip list."
    except:
        return "Failed to get pip list."


def get_workspace_state(cwd):
    """获取工作目录下的文件结构以及所有 py 和 md 文件内容"""
    if not os.path.exists(cwd):
        return "Workspace empty."
        
    files = os.listdir(cwd)
    if not files:
        return "Workspace is empty."
        
    state = "【文件结构】\n"
    for f in files:
        state += f"- {f}\n"
        
    state += "\n【文件内容】\n"
    has_content = False
    for file in files:
        if file.endswith(".py") or file.endswith(".md"):
            path = os.path.join(cwd, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                state += f"\n--- {file} ---\n{content}\n"
                has_content = True
            except Exception:
                pass
                
    if not has_content:
        state += "暂无 .py 或 .md 文件。\n"
        
    return state


def extract_files_from_coder(text):
    """解析 Coder 输出的 Markdown 文件块"""
    pattern = r"###\s*File:\s*([^\n]+)\n```\w*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    files = {}
    for filename, content in matches:
        filename = filename.strip()
        files[filename] = content.strip()
    return files


def save_files_to_workspace(files, cwd, base_readme=""):
    """将解析出的文件保存到工作目录"""
    saved_list = []
    for filename, content in files.items():
        filepath = os.path.join(cwd, filename)
        
        # 处理 Readme 的附加逻辑
        if filename.lower() == "readme.md":
            final_content = f"{base_readme}\n\n## [Current Step]\n{content}".strip()
        else:
            final_content = content
            
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_content)
            saved_list.append(filename)
        except Exception as e:
            print(f"[System Error] 无法保存文件 {filename}: {e}")
            
    return saved_list


# ==========================================
# 4. 主控工作流 (Agentic Workflow)
# ==========================================

def load_plan(plan_file):
    with open(plan_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return data.get("Detailed_Plan", []), data.get("Original_Idea", {})


def run_experiment(plan_file, model_orchestrator="gemini-3.1-pro-preview", model_coder="gemini-3.1-pro-preview"):
    
    plan_steps, idea = load_plan(plan_file)
    if not plan_steps:
        print("未能在计划文件中找到步骤 (Detailed_Plan为空)！")
        return

    # 1. 创建工作空间
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = os.path.abspath(os.path.join("experiments", timestamp))
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"=== 实验启动 ===")
    print(f"工作空间: {workspace_dir}")
    print(f"共发现 {len(plan_steps)} 个步骤。")
    print("================\n")

    # 2. 初始化 Agents
    orch_log = os.path.join(workspace_dir, "orchestrator.log")
    coder_log = os.path.join(workspace_dir, "coder.log")
    summary_txt = os.path.join(workspace_dir, "experiment_summary.txt")
    
    orchestrator = LLMAgent(model=model_orchestrator, log_file=orch_log)
    coder = LLMAgent(model=model_coder, log_file=coder_log)

    # 状态变量
    past_summaries = []
    base_readme = ""
    step_idx = 0
    
    # 开始遍历步骤
    while step_idx < len(plan_steps):
        step = plan_steps[step_idx]
        step_name = step.get("name", "Unknown Step")
        step_content = step.get("content", "")
        step_expected = step.get("expected_outcome", "")
        
        print(f"\n>>> 开始执行 第 {step_idx + 1} 步: {step_name} <<<")
        
        # -----------------------------------------------------
        # 上下文压缩与注入
        # -----------------------------------------------------
        orchestrator.clear_history()
        
        workspace_files = get_workspace_state(workspace_dir)
        
        context_prompt = f"【完整研究计划摘要】\n背景: {idea.get('Background','')}\n方法: {idea.get('Methodology','')}\n\n"
        if past_summaries:
            context_prompt += "【之前步骤的总结 (已完成)】\n" + "\n".join(past_summaries) + "\n\n"
            
        context_prompt += f"【当前工作目录文件结构及代码/文档内容】\n{workspace_files}\n\n"
        
        context_prompt += f"""【当前任务: 计划第 {step_idx + 1} 步】
步骤名称: {step_name}
具体内容: {step_content}
预期成果: {step_expected}

请仔细阅读以上内容，并输出 JSON 指令。当前步骤刚开始，通常请输出 Action: PROMPT_CODER 指导 Coder 开始编程。
请再次注意以下要求：
1.你只需要关注逻辑和最终结果，语法错误系统会自动让Coder去修复。此外，因为我们只会返回给你代码运行的控制台输出。
2.你必须在Coder_Prompt中明确要求Coder在代码中增加足够的 print() 语句，把关键结果输出到控制台，以便你进行验收判断。同时注意：你让Coder输出的内容必须具体，具有实际的物理意义（对你的判断和后续研究有指导意义）。
3.绝对不允许存在将中间结果保存到文件中再在后续步骤读取的情况。你必须明确指示 Coder 从前面生成的代码中直接 import 需要的函数/模块来获取数据或执行计算。不允许要求Coder在代码中有任何文件读写操作（即便给你的计划中这么说了）
4.忽略科研计划中关于需要输出其他中间结果（如pth）的建议，你必须从控制台输出判断一切。
5.你不能完全信任给你的代码执行计划（因为那些计划也是由AI给出的）。你必须独立思考，确认该步骤的目标（不局限于计划中的目标）确实完成。你必须确保以下几点：
  a) 如果涉及和之前的运行结果对比来观察当前结果是否合理，你必须保证仿真场景参数完全相同，如果Coder给出的run.bat的仿真参数和之前并非完全相同，你需要使用你运行代码的权限，重新定义运行脚本（其中包含仿真参数）
  b) 你必须做到足够严格和谨慎，因为Coder是一个AI，你不能在完全确定当前结果正常之前跳过当前步骤。如果你认为当前结果因为各种原因效果不够好，必须重新定义仿真脚本（如果你认为是某些仿真参数设置不合理导致）或者让Coder修改代码
  c) Summary中，重点包含目前已实现的所有代码并简述其作用，并重点描述当前的仿真场景，仿真参数和详细的仿真结果(例如详细的BER-SNR数据)。
6. 你必须做到足够严格，如果你不够严苛，错误的代码可能会导致后续步骤持续出错，带来非常严重的经济损失。
"""
        
        response, _ = orchestrator.get_response_stream(context_prompt, ORCHESTRATOR_SYSTEM_PROMPT)
        orch_json = LLMAgent.robust_extract_json(response)
        
        if not orch_json:
            print("[System] Orchestrator 初始输出解析失败，重试...")
            continue
            
        current_orch_action = orch_json.get("Action", "PROMPT_CODER")
        current_orch_instruction = orch_json.get("Coder_Prompt", "")
        current_run_script = orch_json.get("Run_Script", "")
        
        # -----------------------------------------------------
        # 子循环：执行与验收，直至通过或超时
        # -----------------------------------------------------
        attempts = 0
        step_passed = False
        coder.clear_history()
        
        while attempts < MAX_RETRIES and not step_passed:
            attempts += 1
            print(f"\n--- [Step {step_idx + 1}] 尝试 {attempts}/{MAX_RETRIES} ---")
            
            files = {}
            run_script_path = ""
            
            # 路径分配：Orchestrator直接运行代码 还是 交给Coder编程
            if current_orch_action == "RUN_CODE" and current_run_script:
                print("[Orchestrator] 选择绕过 Coder，直接带参数运行代码...")
                run_script_path = os.path.join(workspace_dir, "orch_run.bat")
                try:
                    with open(run_script_path, "w", encoding="utf-8") as f:
                        f.write(current_run_script)
                except Exception as e:
                    print(f"[System Error] 无法保存运行脚本: {e}")
                    current_orch_action = "PROMPT_CODER"
                    continue
            else:
                # 给 Coder 编写代码的路径
                pip_list = get_installed_packages()
                workspace_files_coder = get_workspace_state(workspace_dir)
                
                coder_prompt = f"""
【Orchestrator 的指令/反馈】
{current_orch_instruction}

【当前工作空间文件内容】
{workspace_files_coder}

【当前虚拟环境 Pip 包列表】
{pip_list[:1000]} ... (截断)

【系统强制要求】
请返回完整的，功能完善的，可运行的python代码以及运行此代码需要的bat脚本(必须命名为run.bat)。
务必通过 argparser 暴露出各项超参，并在 readme.md 中写明参数意义及完整运行命令示例。
"""
                print("[Coder] 正在思考并编写代码...")
                try:
                    coder_resp, _ = coder.get_response_stream(coder_prompt, CODER_SYSTEM_PROMPT.format(conda_env=CONDA_ENV_NAME))
                except Exception as e:
                    print(f"[System Error] 无法获取 Coder 输出: {e},等待10秒后重试... ")
                    time.sleep(10)
                    continue
                
                files = extract_files_from_coder(coder_resp)
                if not files:
                    current_orch_instruction = "系统提示：没有检测到任何符合要求的代码块(### File: ...)，请严格按照Markdown格式重新输出代码。"
                    current_orch_action = "PROMPT_CODER"
                    print("[System] Coder 没有输出文件格式。")
                    continue
                    
                saved = save_files_to_workspace(files, workspace_dir, base_readme)
                print(f"[System] 成功保存文件: {', '.join(saved)}")
                
                if "run.bat" not in files:
                    current_orch_instruction = "系统提示：你没有生成 run.bat 脚本，请必须提供该脚本以供执行！"
                    current_orch_action = "PROMPT_CODER"
                    print("[System] 未找到 run.bat，打回。")
                    continue
                    
                run_script_path = os.path.join(workspace_dir, "run.bat")

            # ==== 运行 bat 脚本 (统一使用监控接口) ====
            success, stdout, stderr, monitor_feedback = run_command_with_monitoring(run_script_path, workspace_dir, orchestrator)
            
            # 运行失败或被强制中断处理
            if not success:
                print(f"[System] 脚本执行异常！")
                if monitor_feedback:
                    # 被 Orchestrator 实时监控终止
                    current_orch_instruction = f"【系统提示：程序被Orchestrator强制中断】\n中断原因及修改建议：\n{monitor_feedback}\n\n中断前的最后部分控制台输出：\n{stdout[-1500:]}"
                else:
                    # 原生报错崩溃
                    print(f"Error Detail: {stdout[:500]}...")
                    current_orch_instruction = f"运行批处理脚本发生严重错误/崩溃，请根据以下输出信息修复代码：\n{stdout}"
                
                current_orch_action = "PROMPT_CODER"  # 发生错误后必须强行回退给 Coder 处理
                continue
                
            # 运行成功 -> 汇报给 Orchestrator 进行逻辑验收
            print(f"[System] 脚本执行成功！返回：\n {stdout[:1000]}... (截断)\n")
            print(f"[System] 正在交由 Orchestrator 验收结果...")
            
            eval_prompt = f"""
【系统报告：代码已成功运行完毕，退出码为 0】
以下是控制台输出内容:
{stdout}

以下是当前工作区最新文件状态:
{get_workspace_state(workspace_dir)}

请判定：
1. 计划的这一步是否达到预期？如果达到，输出 Action: PASS_STEP 并总结。
2. 如果未达到预期（需要修改代码逻辑），输出 Action: REJECT_STEP 并在 Coder_Prompt 给出指导。
3. 如果代码逻辑基本正确，但你希望亲自换一组参数重新运行代码来观察结果，请输出 Action: RUN_CODE 并在 Run_Script 中提供你需要执行的完整 bat 脚本内容。
"""
            orch_resp, _ = orchestrator.get_response_stream(eval_prompt, ORCHESTRATOR_SYSTEM_PROMPT)
            orch_json = LLMAgent.robust_extract_json(orch_resp)
            
            if not orch_json:
                 current_orch_instruction = "Orchestrator 返回的JSON格式错误，请重新验证。"
                 current_orch_action = "PROMPT_CODER"
                 continue
                 
            action = orch_json.get("Action", "REJECT_STEP")
            
            if action == "PASS_STEP":
                step_passed = True
                summary = orch_json.get("Summary", f"步骤 {step_name} 已成功完成。")
                print(f"[Orchestrator] 验收通过！总结: {summary[:100]}...")
                
                # 记录总结
                past_summaries.append(f"【第 {step_idx + 1} 步: {step_name} 总结】: {summary}")
                with open(summary_txt, "a", encoding="utf-8") as f:
                    f.write(f"--- 步骤 {step_idx + 1}: {step_name} ---\n{summary}\n\n")
                    
                # 更新全局 README 基础
                if "readme.md" in files:
                    base_readme += f"\n\n## 步骤 {step_idx + 1}: {step_name}\n" + files["readme.md"]
                    
            elif action == "RUN_CODE":
                current_orch_action = "RUN_CODE"
                current_run_script = orch_json.get("Run_Script", "")
                print(f"[Orchestrator] 决定使用自定义参数亲自重新运行代码: {current_run_script[:100]}...")
            
            else: # REJECT_STEP
                current_orch_action = "PROMPT_CODER"
                current_orch_instruction = orch_json.get("Coder_Prompt", "Orchestrator 认为逻辑不符，请重新检查需求并修改。")
                print(f"[Orchestrator] 验收驳回，修改意见: {current_orch_instruction[:100]}...")
                
        # -----------------------------------------------------
        # 处理重试限制与回溯逻辑 (Backtracking)
        # -----------------------------------------------------
        if step_passed:
            step_idx += 1 # 进入下一步
        else:
            print(f"\n[System Error] 第 {step_idx + 1} 步在 {MAX_RETRIES} 次尝试后仍然失败。触发回退机制！")
            if step_idx > 0:
                step_idx -= 1
                if past_summaries:
                    past_summaries.pop()
                print(f"[System] 已回退到第 {step_idx + 1} 步。")
            else:
                print("[System Fatal] 第一步就彻底失败，无法回退。实验终止。")
                break

    print("\n=== 实验执行完毕 ===")
    print(f"最终结果与代码保存在: {workspace_dir}")


def main():
    parser = argparse.ArgumentParser(description="AI Scientist - Experiment Performer")
    parser.add_argument("--plan_file", type=str, default=r"final_research_plans\single_plan.json", help="之前生成的包含计划的JSON文件路径")
    parser.add_argument("--orchestrator", type=str, default="gemini-3.1-pro-preview", help="Orchestrator 使用的模型")
    parser.add_argument("--coder", type=str, default="gemini-3.1-pro-preview", help="Coder 使用的模型")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.plan_file):
        print(f"找不到计划文件: {args.plan_file}")
        return
        
    run_experiment(
        plan_file=args.plan_file,
        model_orchestrator=args.orchestrator,
        model_coder=args.coder
    )

if __name__ == "__main__":
    main()

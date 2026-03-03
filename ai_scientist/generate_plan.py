
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入您已经实现的 LLMAgent
from llm import LLMAgent

# ==========================================
# 1. 提示词定义 (Prompts)
# ==========================================

STUDENT_SYSTEM_PROMPT = """
你是一个严谨细致的通信领域研究员。
你的任务是根据一个由其他AI给出的初步的科研Idea，制定出一套极其详尽、循序渐进的研究与仿真落地计划。
因为后续的所有代码编写和仿真都将由能力有限的AI助手来完成，所以你的计划必须做到“手把手”级别，不能有跨度过大的跳跃。

你的计划必须严格按照以下4个模块展开：
1. 系统模型 (System Model)：用非常具体、严谨的文字和数学语言描述系统模型如何建立（如信道模型、信号传输过程、目标函数等）。
2. 基线方法实施 (Baselines)：必须包含先实施一个传统通信方法（如迫零、MMSE、传统优化算法等），然后再实施一个与Idea相关的基础AI方法(即idea中所提AI方法的改进对象)作为Baseline。
3. 创新方案拆解 (Innovative Scheme)：针对Idea涉及的创新部分，必须将其拆分成**至少3个递进的步骤**。建议先对Baseline中的AI方法进行微调，验证通过后再一步步叠加复杂度，最终完整实现Idea。
4. 对比方式与预期 (Comparison & Outcomes)：明确需要和哪些Baseline对比，以及评估指标是什么。
5. 如果你发现这个idea本身不够合理，可以计划实现他的变体
6. 从相对简单但合理的仿真模型开始（但是该模型应该能够覆盖Innovative Scheme中第一步对应的场景）。如果后续Innovative Scheme需要更复杂的模型，可在后续步骤中补充（作为单独步骤）

对于通信领域的研究，一个好的Plan必须满足以下要求：
1、仿真场景能够用明确的数学语言描述
2、expected_outcome包含可以量化的标准
3、各变量/输入输出之间具有清晰的关系，即计划中每出现一个新变量，都要指明他的来源（他是系统本身的固有变量，还是依赖与其他变量，是哪些变量经过什么样的操作得到的？）
4、各步骤之间的递进关系尽可能清晰，衔接紧密，后面的步骤可以由前面的步骤经过较小的改动得到
5、想象你的计划将要交给一个非常incompetent的AI执行者/程序员，你的计划需要细致到每一步都非常清晰明确，不能有任何模糊的描述或者跳跃性的思维。你需要把每一步都拆解得非常细。
6、准备编写计划前，你需要先仔细想想，计划中的每个部分是否都确实可以执行（容易被编写成代码），以及从你的知识和逻辑上判断，执行是否会有好的效果。否则，请修改你的计划。

请输出如下JSON格式（必须被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "这里简述你的整体规划思路。",
    "Plan": [
        {
            "idx": 1,
            "name": "步骤名称 (如: 系统模型数学建模)",
            "content": "这一步要执行的具体内容，包含必要的数学公式定义、参数设置、具体实现逻辑等，越详细越好。",
            "expected_outcome": "这一步执行的预期成果，必须非常具体且容易衡量是否做对了(如: 成功生成瑞利衰落信道矩阵且均值为0方差为1；或者代码跑通无报错)。"
        },
        ...
    ]
}
```
注意：Plan列表中的每个对象必须严格包含 idx, name, content, expected_outcome 四个字段。
"""

STUDENT_FIRST_PROMPT = """
这是你需要落地实施的科研Idea：
标题: {title}
背景: {background}
假设: {hypothesis}
方法与实际场景考量: {methodology}

请根据该Idea，输出你的第一版详细研究计划。请严格遵守系统提示词中的4个模块和模块3至少拆分3步的要求。
"""

STUDENT_REFINE_PROMPT = """
这是你需要落地实施的科研Idea：
标题: {title}

这是你上一版的计划：
{current_plan}

这是资深导师(Teacher Planner)给你的修改意见：
{teacher_feedback}

请认真反思并在上一版的基础上进行修改完善。确保计划步骤粒度足够细，对较弱的AI执行者足够友好。
请输出修改后的完整JSON计划。
"""

TEACHER_SYSTEM_PROMPT = """
你是一位苛刻且经验丰富的通信领域资深教授（Teacher Planner）。
你的任务是审查学生(Student Planner)根据科研Idea提交的研究与仿真计划。
你需要重点审查以下几点：
1. 步骤是否太宽泛？(由于后续由能力一般的AI执行代码，跨度太大的步骤会导致AI写不出代码或疯狂报错)。
2. 系统模型是否有具体的数学语言描述？
3. 是否严格遵循了“传统基线 -> 基础AI基线 -> 逐步叠加创新的AI方案”的原则？
4. 创新方案是否被细致地拆分为了至少3步递进操作？
5. 每个步骤的预期成果(expected_outcome)是否具体且可衡量？

请输出如下JSON格式（必须被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "你对计划的详细评价、找出的缺陷以及具体的修改建议。",
    "Decision": "Pass 或者 Refine" 
}
```
如果你认为计划已经足够完美、颗粒度极细，可以将Decision设为"Pass"，否则设为"Refine"。
"""

TEACHER_EVAL_PROMPT = """
原始科研Idea如下：
标题: {title}
假设: {hypothesis}

学生提交的计划如下：
{student_plan}

请作为导师严格审查这份计划，并给出你的评审结果和修改建议。如果认为可以了，Decision设为Pass；否则设为Refine。
"""


# ==========================================
# 2. 配置与类定义
# ==========================================

class PlannerConfig:
    """参数配置类"""
    def __init__(self, input_file, output_file, log_dir, max_iters, model_student, model_teacher, max_workers):
        self.input_file = input_file          # 输入的idea json文件路径
        self.output_file = output_file        # 输出的最终plan json文件路径
        self.log_dir = log_dir                # 日志文件夹路径
        self.max_iters = max_iters            # Student和Teacher之间的最大迭代轮数
        self.model_student = model_student    # Student使用的LLM模型
        self.model_teacher = model_teacher    # Teacher使用的LLM模型
        self.max_workers = max_workers        # 并发处理的Idea数量

        # 确保日志文件夹存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


class PlannerStudent:
    def __init__(self, task_id, config: PlannerConfig):
        self.task_id = task_id
        log_file = os.path.join(config.log_dir, f"student_task_{task_id}.log")
        self.agent = LLMAgent(model=config.model_student, log_file=log_file)
        self.current_plan = []
        self.thoughts = ""

    def generate_plan(self, idea, teacher_feedback=None):
        """生成或根据反馈修改研究计划"""
        title = idea.get("Title", "Unknown Idea")
        
        if teacher_feedback is None:
            # 第一次生成
            prompt = STUDENT_FIRST_PROMPT.format(
                title=title,
                background=idea.get("Background", ""),
                hypothesis=idea.get("Hypothesis", ""),
                methodology=idea.get("Methodology", "")
            )
        else:
            # 根据导师反馈修改
            prompt = STUDENT_REFINE_PROMPT.format(
                title=title,
                current_plan=json.dumps(self.current_plan, indent=2, ensure_ascii=False),
                teacher_feedback=teacher_feedback
            )

        response, _ = self.agent.get_response(prompt, STUDENT_SYSTEM_PROMPT)
        parsed_json = LLMAgent.extract_json_between_markers(response)

        if parsed_json and "Plan" in parsed_json:
            self.current_plan = parsed_json.get("Plan", [])
            self.thoughts = parsed_json.get("Thoughts", "")
            return True
        else:
            # 容错处理
            self.agent._log_event("[Error] Failed to parse valid JSON from Student output.")
            return False


class PlannerTeacher:
    def __init__(self, task_id, config: PlannerConfig):
        self.task_id = task_id
        log_file = os.path.join(config.log_dir, f"teacher_task_{task_id}.log")
        self.agent = LLMAgent(model=config.model_teacher, log_file=log_file)

    def review_plan(self, idea, student_plan):
        """审查学生提交的计划并给出建议和判定"""
        prompt = TEACHER_EVAL_PROMPT.format(
            title=idea.get("Title", "Unknown Idea"),
            hypothesis=idea.get("Hypothesis", ""),
            student_plan=json.dumps(student_plan, indent=2, ensure_ascii=False)
        )

        response, _ = self.agent.get_response(prompt, TEACHER_SYSTEM_PROMPT)
        parsed_json = LLMAgent.extract_json_between_markers(response)

        if parsed_json:
            decision = parsed_json.get("Decision", "Refine")
            feedback = parsed_json.get("Thoughts", "No detailed feedback provided.")
            return decision, feedback
        else:
            self.agent._log_event("[Error] Failed to parse valid JSON from Teacher output.")
            return "Refine", "请仔细检查你的输出格式，必须包含严谨的Plan列表，并且符合系统要求。"


# ==========================================
# 3. 核心执行逻辑
# ==========================================

def run_planner_pipeline(task_id, idea, config: PlannerConfig):
    """单对 Student-Teacher 协同处理一个 Idea 的流水线"""
    print(f"[Task {task_id}] Started planning for Idea: {idea.get('Title', 'Unknown')[:30]}...")
    
    student = PlannerStudent(task_id, config)
    teacher = PlannerTeacher(task_id, config)

    teacher_feedback = None
    final_plan = None
    final_decision = "Timeout"

    for i in range(config.max_iters):
        print(f"[Task {task_id}] Iteration {i+1}/{config.max_iters} - Student generating plan...")
        
        # 1. 学生生成/修改计划
        success = student.generate_plan(idea, teacher_feedback)
        if not success:
            teacher_feedback = "上一轮你的输出不符合JSON规范或缺少Plan字段，请严格按格式重新生成。"
            continue
            
        # 2. 导师审查计划
        print(f"[Task {task_id}] Iteration {i+1}/{config.max_iters} - Teacher reviewing plan...")
        decision, feedback = teacher.review_plan(idea, student.current_plan)
        
        if decision.upper() == "PASS":
            print(f"[Task {task_id}] Iteration {i+1}: Teacher PASSED the plan.")
            final_plan = student.current_plan
            final_decision = "Pass"
            break
        else:
            print(f"[Task {task_id}] Iteration {i+1}: Teacher asked to REFINE.")
            teacher_feedback = feedback
            final_plan = student.current_plan # 保留最新的一版
            final_decision = "Max_Iters_Reached"

    # 如果达到最大迭代次数依然没有Pass，我们仍然保留最后一版Plan
    print(f"[Task {task_id}] Completed. Final Status: {final_decision}.")
    
    # 组装返回结果
    result = {
        "Task_ID": task_id,
        "Original_Idea_Title": idea.get("Title", ""),
        "Original_Idea": idea,
        "Final_Decision": final_decision,
        "Teacher_Final_Feedback": teacher_feedback if final_decision != "Pass" else "Perfect plan.",
        "Detailed_Plan": final_plan
    }
    return result


def generate_plan(parser):
    args = parser.parse_args()

    # 初始化配置 (为保证兼容性，仍将 output_dir 赋给原配置类的 output_file 属性)
    config = PlannerConfig(
        input_file=args.input_file,
        output_file=args.output_dir, 
        log_dir=args.log_dir,
        max_iters=args.max_iters,
        model_student=args.model_student,
        model_teacher=args.model_teacher,
        max_workers=args.max_workers
    )

    print("=== 启动 AI Scientist: Planner ===")
    print(f"输入文件: {config.input_file}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"Student 模型: {config.model_student} | Teacher 模型: {config.model_teacher}")
    print(f"每个 Idea 分配: {args.k_agents} 组 Agent 同步进行")
    print(f"最大迭代: {config.max_iters} 轮 | 线程池大小: {config.max_workers}")
    print("==================================\n")

    # 读取 Ideas
    if not os.path.exists(config.input_file):
        print(f"找不到输入文件: {config.input_file}。请先运行Idea Generator生成Idea。")
        return
        
    try:
        with open(config.input_file, "r", encoding="utf-8") as f:
            ideas = json.load(f)
    except Exception as e:
        print(f"读取输入文件失败，请确认文件是否为合法的JSON格式。错误: {e}")
        return

    if not isinstance(ideas, list):
        # 兼容单条idea的情况
        ideas = [ideas]
        
    print(f"共读取到 {len(ideas)} 个 Ideas，准备开始制定计划...\n")

    # 确保输出文件夹存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 结果按 idea_idx 进行归类分组字典，存储每个 idea 的所有 K 个 plan 
    idea_plans = {i: [] for i in range(len(ideas))}

    # 并发执行 Planner
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_task = {}
        
        # 提交所有的Idea到线程池：每个 Idea 提交 K 次
        for idx, idea in enumerate(ideas):
            for k in range(args.k_agents):
                # 生成唯一 task_id 避免日志和过程冲突，如 "idea_1_agent_2"
                task_id = f"idea_{idx + 1}_agent_{k + 1}"
                
                # 提交任务，并且把 (idx, task_id) 作为凭据存起来
                future = executor.submit(run_planner_pipeline, task_id, idea, config)
                future_to_task[future] = (idx, task_id)
        
        # 收集结果
        for future in as_completed(future_to_task):
            idea_idx, task_id = future_to_task[future]
            try:
                result = future.result()
                # 记录一下当前这个 plan 是由几号 agent 生成的
                result["Agent_Task_ID"] = task_id 
                
                # 将结果追加到对应 Idea 的列表里
                idea_plans[idea_idx].append(result)
            except Exception as exc:
                print(f"[Task {task_id}] generated an exception: {exc}")

    # 将归类好的规划结果保存为每个 Idea 对应的独立 JSON 文件
    try:
        for idea_idx, plans in idea_plans.items():
            if not plans:
                continue
            
            # 拼接该 idea 的文件保存路径，例如 "final_research_plans/idea_1_plans.json"
            idea_file_name = f"idea_{idea_idx + 1}_plans.json"
            idea_file_path = os.path.join(args.output_dir, idea_file_name)
            
            with open(idea_file_path, "w", encoding="utf-8") as f:
                json.dump(plans, f, indent=4, ensure_ascii=False)
                
        print(f"\n全部计划制定完成！各想法的并行汇总计划已成功保存至文件夹: {args.output_dir}")
        return idea_file_path
    except Exception as e:
        print(f"保存最终结果失败: {e}")

if __name__ == "__main__":
    generate_plan()
    
    

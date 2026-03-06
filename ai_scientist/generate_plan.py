
import json
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入基础组件
from llm import LLMAgent
from utils import setup_logger

# ==========================================
# 从 generate_ideas.py 导入必要的工具函数
# 请确保 generate_ideas.py 在同一目录下，或者在 PYTHONPATH 中
# ==========================================
from generate_ideas import (
    search_for_papers,
    download_paper_pdf,
    process_papers_to_read,
    format_search_results_and_update_map,
    read_knowledge_base
)

logger = setup_logger("experiment_run.log")

# ==========================================
# 1. 更新后的提示词 (支持搜索与阅读)
# ==========================================

STUDENT_SYSTEM_PROMPT = """
你是一个严谨细致的通信领域研究员（Student Planner）。
你的任务是根据一个初步的科研Idea，制定出一套极其详尽、循序渐进的研究与仿真落地计划。

你拥有**文献检索**和**全文阅读**的能力。在制定计划前或制定过程中，如果你觉得某些步骤的数学原理不清晰、或者不知道具体的实现算法，请务必使用搜索工具查找相关文献，甚至阅读全文以获取具体的系统模型公式和算法流程。

你的回复必须包含如下JSON格式（必须被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "这里简述你的思考过程。如果你决定搜索，请说明你想查什么；如果你决定写计划，请说明思路。",
    "SearchQueries": ["query1", "query2"], 
    "PapersToRead": ["https://doi.org/10.xxxx/xxxx"],
    "Plan": [
        {
            "idx": 1,
            "name": "步骤名称",
            "content": "具体的实施细节、数学公式、算法步骤。",
            "expected_outcome": "可量化的预期结果。"
        }
    ]
}
```
**关键要求：**
1. **Plan结构**：必须包含4个模块：(1)系统模型(System Model, 必须有详细数学描述)；(2)基线方法(Baselines, 包含传统与基础AI方法)；(3)创新方案(Innovative Scheme, 必须拆分为至少3个递进步骤)；(4)对比与评估(Comparison)。
2. **SearchQueries**：如果你觉得当前知识不足以写出详细公式，请填入搜索词。
3. **PapersToRead**：如果你需要系统阅读某篇论文的全文（特别是提取公式时），请将其DOI填入此列表。
4. **Plan字段**：每次都必须生成完整的Plan（即使现在的信息不完全充足）。每次生成Plan时，必须是完整的计划（包含所有步骤）。
"""


STUDENT_ITERATION_PROMPT = """
这是你需要落地实施的科研Idea：
标题: {title}
背景: {background}
假设: {hypothesis}
方法: {methodology}

--- 状态与知识更新 ---
【当前最新的知识库（精读论文笔记）】（注意：这是当前收集到的所有公式与细节，请充分利用）：
{knowledge_base}

【上一轮检索得到的文献搜索结果】（注意：仅展示最新一次搜索的结果）：
{search_results}

【导师反馈/初始指引】：
{teacher_feedback}

【之前每轮迭代给出的Plan历史记录】：
{plan_history}

请基于上述所有的最新知识和反馈，继续你的工作。如果你觉得某些步骤的数学原理不清晰，可以继续使用 PapersToRead 请求阅读全文，或者使用 SearchQueries 查找具体实现。如果信息已经充分，请在 Thoughts 中反思后直接给出更新后的详细 Plan。
注意：哪怕只是在上一版基础上微调，也请在 Plan 字段输出完整的计划结构。
"""

# STUDENT_ITERATION_PROMPT = """
# 这是你需要落地实施的科研Idea：
# 标题: {title}
# 背景: {background}
# 假设: {hypothesis}
# 方法: {methodology}

# --- 状态更新 ---
# 这是你上一轮的文献搜索结果：
# {search_results}

# 这是你要求精读的论文的全文总结笔记（知识库）：
# {knowledge_base}

# 这是导师（或上一轮你自己）给出的反馈/计划草稿：
# {feedback_or_previous_plan}

# 请根据最新的文献知识，继续你的工作。你可以继续搜索，也可以完善并输出最终的详细计划。
# 如果你的计划已经非常完善且不需要再查阅资料，请确保Plan字段不为空，并在Thoughts中表明已完成。
# """

TEACHER_SYSTEM_PROMPT = """
你是一位苛刻且经验丰富的通信领域资深教授（Teacher Planner）。
你的任务是审查学生提交的研究计划。
重点审查：
1. 步骤是否太宽泛？(由于后续由能力一般的AI执行代码，跨度太大的步骤会导致AI写不出代码或疯狂报错)。
2. 系统模型是否有具体的数学语言描述？
3. 是否严格遵循了“传统基线 -> 基础AI基线 -> 逐步叠加创新的AI方案”的原则？
4. 创新方案是否被细致地拆分为了至少3步递进操作？
5. 每个步骤的预期成果(expected_outcome)是否具体且可衡量？

请输出如下JSON格式（必须被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "详细的评价和修改建议。",
    "Decision": "Pass" 或 "Refine"
}
```
"""

USER_REFINE_SYSTEM_PROMPT = """
你是一个协助用户修改科研计划的AI助手。
你的任务是根据用户的具体指令，修改已有的科研计划。
你可以利用搜索工具来验证用户的想法或寻找具体的实现细节。
建议：由于我们使用的OpenAlex对模糊搜索支持较差，所以建议搜索较短的关键词，同时用OR或者AND来连接：如："deep learning" OR "neural network"。
请严格将输出格式为JSON：
你的输出中必须包含以下字段："Thoughts","SearchQueries,"PapersToRead","Plan"
输出示例：
```json
{
    "Thoughts": "这里简述你的思考过程。如果你决定搜索，请说明你想查什么；如果你决定写计划，请说明思路。",
    "SearchQueries": ["query1", "query2"], 
    "PapersToRead": ["https://doi.org/10.xxxx/xxxx"],
    "Plan": [
        {
            "idx": 1,
            "name": "步骤名称",
            "content": "具体的实施细节、数学公式、算法步骤。",
            "expected_outcome": "具体的、可量化的预期结果。"
        }
    ]
}
```
"""

# USER_REFINE_START_PROMPT = """
# 这是当前的计划版本：
# {current_plan}

# 用户指令：
# 【{user_instruction}】

# 请根据用户指令修改计划。如果需要，你可以先进行文献搜索。
# """
USER_REFINE_START_PROMPT = """
你是一个协助用户修改科研计划的AI助手。
这是当前正在修改的科研Idea：
标题: {title}

--- 交互历史记录 ---
【之前所有交互中用户的修改意见，以及每次交互后生成的计划版本】：
{interaction_history}

--- 状态与知识更新 ---
【当前最新的知识库（精读论文笔记）】：
{knowledge_base}

【上一轮检索得到的文献搜索结果】：
{search_results}

--- 本次任务 ---
【当前用户的最新修改指令】：
{current_instruction}

请根据用户的最新指令，修改并完善计划。你可以利用搜索工具来验证想法或寻找具体实现细节。
输出格式要求与之前保持一致（包含 Thoughts, SearchQueries, PapersToRead, Plan）。必须返回完整的 Plan。
"""
# ==========================================
# 2. 辅助类与配置
# ==========================================
class PlannerConfig:
    def __init__(self, input_file, output_file, log_dir, max_iters, max_inner_iters,model_student, model_teacher, max_workers):
        self.input_file = input_file
        self.output_file = output_file
        self.log_dir = log_dir
        self.max_iters = max_iters
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.max_workers = max_workers
        self.max_inner_iters = max_inner_iters
        self.search_params = { # 默认搜索配置
            "open_access": True,
            "has_pdf_url": True,
            "from_year": 2018
        }
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

# ==========================================
# 3. 核心逻辑实现
# ==========================================

# ==========================================
# 核心业务逻辑函数
# ==========================================

def run_planner_pipeline(task_id, idea, config):
    """
    单条 Idea 的规划流水线：
    严格控制上下文：Student 每轮重新构造 Prompt，包含历史 Plan、最新 KB 和最新一次的 Search。
    """
    logger.info(f"[Task {task_id}] Started planning...")
    
    student_log = os.path.join(config.log_dir, f"student_{task_id}.log")
    teacher_log = os.path.join(config.log_dir, f"teacher_{task_id}.log")
    
    kb_txt_path = os.path.join(config.log_dir, f"kb_{task_id}.txt")
    doi_url_map = {}
    
    plan_history = []  # 记录该 Student 自己迭代产生的各个版本的 Plan
    teacher_feedback = "这是第一轮，请开始你的工作。建议先搜索相关文献确定系统模型，再起草计划。"
    final_decision = "Timeout"
    
    max_review_rounds = 3 # 设定 Teacher 最多审查 3 次
    
    for review_round in range(max_review_rounds):
        logger.info(f"[Task {task_id}] --- Review Round {review_round + 1} ---")
        
        # --- Student Phase: 自我迭代 (搜索 & 起草) ---
        search_feedback = "暂无新搜索结果。"
        student_sub_iters = config.max_inner_iters 
        
        for i in range(student_sub_iters):
            logger.info(f"[Task {task_id}] Student Step {i+1}/{student_sub_iters}")
            
            # 【关键】每次重新实例化 Agent，彻底抛弃框架自带的会话历史积累，防止 Context 爆炸
            student_agent = LLMAgent(model=config.model_student, log_file=student_log)
            
            kb_content = read_knowledge_base(kb_txt_path)
            
            # 组装 plan_history 字符串
            history_str = ""
            if not plan_history:
                history_str = "暂无历史计划。"
            else:
                for idx, p in enumerate(plan_history):
                    history_str += f"--- 第 {idx+1} 次起草的 Plan ---\n{json.dumps(p, indent=2, ensure_ascii=False)}\n\n"
            
            prompt = STUDENT_ITERATION_PROMPT.format(
                title=idea.get("Title", ""),
                background=idea.get("Background", ""),
                hypothesis=idea.get("Hypothesis", ""),
                methodology=idea.get("Methodology", ""),
                search_results=search_feedback,
                knowledge_base=kb_content,
                plan_history=history_str,
                teacher_feedback=teacher_feedback if i == 0 else "无最新导师反馈，请继续自我完善。"
            )
            
            # 【注意】这里假设 STUDENT_SYSTEM_PROMPT 已在全局定义（要求输出 JSON 等）
            response, _ = student_agent.get_response_stream(prompt, STUDENT_SYSTEM_PROMPT)
            parsed_json = LLMAgent.robust_extract_json(response)
            
            if not parsed_json:
                search_feedback = "错误：你的输出不符合JSON格式，请修正并重新输出完整的JSON。"
                continue
            
            queries = parsed_json.get("SearchQueries", [])
            papers_to_read = parsed_json.get("PapersToRead", [])
            plan_draft = parsed_json.get("Plan", [])
            
            if plan_draft:
                plan_history.append(plan_draft)
                
            has_action = False
            
            # 1. 阅读 PDF
            if papers_to_read:
                logger.info(f"[Task {task_id}] Student requesting read: {len(papers_to_read)} papers")
                process_papers_to_read(papers_to_read, doi_url_map, kb_txt_path)
                has_action = True
                
            # 2. 搜索文献
            if queries:
                logger.info(f"[Task {task_id}] Student searching: {queries}")
                search_feedback = format_search_results_and_update_map(
                    queries, doi_url_map,
                    open_access=config.search_params['open_access'],
                    has_pdf_url=config.search_params['has_pdf_url'],
                    from_year=config.search_params['from_year']
                )
                has_action = True
            else:
                search_feedback = "本轮未进行搜索。"

            # 如果已经输出了Plan，并且当前不需要搜索/阅读来获取额外知识，提前结束起草阶段
            # if plan_draft and not has_action:
            #     logger.info(f"[Task {task_id}] Student is ready for review.")
            #     break
        
        # --- Teacher Phase: 审查 ---
        if not plan_history:
            logger.warning(f"[Task {task_id}] Student failed to generate a plan.")
            teacher_feedback = "你还没有生成任何 Plan 字段，请立即生成完整结构。"
            continue

        teacher_agent = LLMAgent(model=config.model_teacher, log_file=teacher_log)
        current_plan = plan_history[-1]
        
        prompt_teacher = f"Idea: {idea.get('Title')}\nStudent's Plan:\n{json.dumps(current_plan, indent=2, ensure_ascii=False)}"
        
        # 【注意】这里假设 TEACHER_SYSTEM_PROMPT 已在全局定义
        response_t, _ = teacher_agent.get_response_stream(prompt_teacher, TEACHER_SYSTEM_PROMPT)
        parsed_t = LLMAgent.robust_extract_json(response_t)
        
        if parsed_t:
            decision = parsed_t.get("Decision", "Refine")
            feedback = parsed_t.get("Thoughts", "")
            
            if decision == "Pass":
                final_decision = "Pass"
                teacher_feedback = feedback
                logger.info(f"[Task {task_id}] Teacher PASSED the plan.")
                break
            else:
                final_decision = "Refine"
                teacher_feedback = f"Teacher Feedback (Round {review_round+1}): {feedback}"
                logger.info(f"[Task {task_id}] Teacher requires REFINEMENT.")
        else:
            teacher_feedback = "Teacher 输出格式错误，这是系统故障导致的无反馈，请自行反思当前计划。"

    return {
        "Task_ID": task_id,
        "Original_Idea": idea,
        "Final_Decision": final_decision,
        "Teacher_Feedback": teacher_feedback,
        "Detailed_Plan": plan_history[-1] if plan_history else [],
        "KB_Path": kb_txt_path,
        "DOI_Map": doi_url_map
    }


def refine_plan_interactive(interaction_history, current_instruction, allow_search, idea, config, existing_kb_path, existing_doi_map):
    """
    交互模式下的 Plan 修改器。
    严格控制上下文：注入之前的历史交互对、最新的 KB 和最近一轮搜索结果。
    """
    refine_id = f"interactive_{int(time.time())}"
    kb_txt_path = existing_kb_path
    doi_url_map = existing_doi_map
    
    search_feedback = "用户开启了搜索权限，你可以先搜索相关知识。" if allow_search else "用户未开启搜索权限，请仅依据已有知识修改。"
    
    # 组装交互历史字符串
    inter_str = ""
    if not interaction_history:
        inter_str = "暂无历史交互。这是第一次针对原始计划的修改。"
    else:
        for idx, item in enumerate(interaction_history):
            inter_str += f"--- 第 {idx+1} 次交互 ---\n"
            inter_str += f"用户指令: {item['instruction']}\n"
            inter_str += f"生成的计划结果: {json.dumps(item['plan'], indent=2, ensure_ascii=False)}\n\n"

    updated_plan = []
    if interaction_history:
        updated_plan = interaction_history[-1]["plan"]
        
    for i in range(3): # Agent 可以有多轮自我思考/搜索来响应用户的单次指令
        logger.info(f"[Interactive Refine] Loop {i+1}/3")
        
        # 【关键】重新实例化 Agent
        agent = LLMAgent(model=config.model_student, log_file=os.path.join(config.log_dir, f"log_{refine_id}.log"))
        kb_content = read_knowledge_base(kb_txt_path)
        
        prompt = USER_REFINE_START_PROMPT.format(
            title=idea.get('Title', 'Unknown'),
            interaction_history=inter_str,
            knowledge_base=kb_content,
            search_results=search_feedback,
            current_instruction=current_instruction
        )
        
        # 【注意】这里假设 USER_REFINE_SYSTEM_PROMPT 已在全局定义
        response, _ = agent.get_response_stream(prompt, USER_REFINE_SYSTEM_PROMPT)
        parsed = LLMAgent.robust_extract_json(response)
        
        if not parsed:
            search_feedback = "你的输出不符合JSON格式，请重试。"
            continue
            
        queries = parsed.get("SearchQueries", [])
        papers = parsed.get("PapersToRead", [])
        new_plan = parsed.get("Plan", [])
        
        if new_plan:
            updated_plan = new_plan
            
        has_action = False
        if allow_search:
            if papers:
                process_papers_to_read(papers, doi_url_map, kb_txt_path)
                has_action = True
            if queries:
                search_feedback = format_search_results_and_update_map(
                    queries, doi_url_map,
                    open_access=config.search_params['open_access'],
                    has_pdf_url=config.search_params['has_pdf_url'],
                    from_year=config.search_params['from_year']
                )
                has_action = True
            else:
                search_feedback = "本轮未进行搜索。"
        
        # 若成功更新了 Plan 且不需要查资料，则跳出Agent自我反思循环
        if new_plan and (not has_action or not allow_search):
            break
            
    return updated_plan


def generate_plan(args):
    """
    主入口：并行生成初步计划，并支持交互式选择与持续迭代细化 (Refine)
    """
    config = PlannerConfig(
        input_file=args.input_file,
        output_file=args.output_dir, 
        log_dir=args.log_dir,
        max_iters=args.max_iters,
        max_inner_iters = args.max_inner_iters,
        model_student=args.model_student,
        model_teacher=args.model_teacher,
        max_workers=args.max_workers
    )

    logger.info("=== 启动 AI Scientist: Planner (Interactive + Context Controlled) ===")
    
    if not os.path.exists(config.input_file):
        logger.error("找不到输入文件。")
        return
    with open(config.input_file, "r", encoding="utf-8") as f:
        ideas = json.load(f)
    if not isinstance(ideas, list): ideas = [ideas]

    all_results = []
    
    logger.info(f"为 {len(ideas)} 个 Idea 制定计划，每个分配 {args.k_agents} 组 Agent 并行执行...")
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_task = {}
        for idx, idea in enumerate(ideas):
            for k in range(args.k_agents):
                task_id = f"idea_{idx+1}_agent_{k+1}"
                future = executor.submit(run_planner_pipeline, task_id, idea, config)
                future_to_task[future] = (idx, k)
        
        for future in as_completed(future_to_task):
            idx, k = future_to_task[future]
            try:
                res = future.result()
                res["Idea_Index"] = idx
                res["Agent_Index"] = k
                all_results.append(res)
            except Exception as e:
                logger.error(f"任务执行异常: {e}")

    # 分组汇总
    grouped_results = {i: [] for i in range(len(ideas))}
    for res in all_results:
        # 初始录入时，如果想开启交互模式，我们为每个计划初始化一个交互历史字段
        res['Interaction_History'] = [
            {"instruction": "【初始 Teacher 审查基线版本】", "plan": res['Detailed_Plan']}
        ]
        grouped_results[res["Idea_Index"]].append(res)
        
    initial_output_path = os.path.join(args.output_dir, "initial_plans.json")
    with open(initial_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    logger.info(f"所有初版并行计划已保存至 {initial_output_path}")

    # ===== 交互模式 =====
    if getattr(args, 'interactive', True):
        print("\n" + "="*60)
        print("   INTERACTIVE PLAN SELECTION & REFINEMENT")
        print("="*60)
        
        while True:
            print(f"\n>>> 找到 {len(ideas)} 个 Idea：")
            for i, idea in enumerate(ideas):
                print(f"  [{i+1}] {idea.get('Title', 'Untitled')[:60]}...")
            
            choice = input("\n请选择要查看/修改的 Idea 编号 (输入 q 退出): ").strip()
            if choice.lower() == 'q': break
            
            try:
                idea_idx = int(choice) - 1
                if idea_idx not in grouped_results: raise ValueError
            except ValueError:
                print("无效编号，请重新输入。")
                continue
                
            current_idea_plans = grouped_results[idea_idx]
            
            while True:
                print(f"\n--- Idea {idea_idx+1} 拥有的 Plan 版本 ---")
                for j, p in enumerate(current_idea_plans):
                    status = p.get('Final_Decision', 'Unknown')
                    steps_cnt = len(p.get('Detailed_Plan', []))
                    print(f"  版本 [{j+1}] 状态: {status} | 步骤数: {steps_cnt} | 导师反馈: {p.get('Teacher_Feedback', '')[:40]}...")
                
                v_choice = input("\n请选择要详细查看/微调的版本 (输入 b 返回上级): ").strip()
                if v_choice.lower() == 'b': break
                
                try:
                    v_idx = int(v_choice) - 1
                    selected_res = current_idea_plans[v_idx]
                except ValueError:
                    print("无效版本。")
                    continue
                
                # 打印详细计划
                print("\n" + "#"*50)
                print(json.dumps(selected_res['Detailed_Plan'], indent=2, ensure_ascii=False))
                print("#"*50)
                print(f"【导师/历史反馈】: {selected_res.get('Teacher_Feedback', '')}")
                
                action = input("\n[s] 满意，采纳此版本并保存\n[r] 不满意，输入指令进行修改(Refine)\n[b] 返回上一级\n您的选择: ").lower().strip()
                
                if action == 's':
                    final_path = os.path.join(args.output_dir, f"final_plan_idea_{idea_idx+1}.json")
                    with open(final_path, "w", encoding="utf-8") as f:
                        json.dump(selected_res['Detailed_Plan'], f, indent=4, ensure_ascii=False)
                    print(f"最终选定的科研计划已保存至: {final_path}")
                    return final_path
                    
                elif action == 'r':
                    instruction = input("请输入具体的修改建议 (如: '增加基于深度强化学习的 baseline'): ").strip()
                    enable_search = input("是否允许修改时检索文献以补充公式/算法细节？(y/n): ").lower() == 'y'
                    
                    print(">>> 正在启动 Agent 进行修改优化...")
                    try:
                        history = selected_res.get('Interaction_History', [])
                        
                        new_plan = refine_plan_interactive(
                            interaction_history=history,
                            current_instruction=instruction,
                            allow_search=enable_search,
                            idea=ideas[idea_idx],
                            config=config,
                            existing_kb_path=selected_res.get('KB_Path'),
                            existing_doi_map=selected_res.get('DOI_Map')
                        )
                        
                        # 创建一个新分支保存修改后的结果，追加到列表末尾
                        new_res = selected_res.copy()
                        new_res['Detailed_Plan'] = new_plan
                        new_res['Final_Decision'] = "Refined_by_User"
                        new_res['Teacher_Feedback'] = f"基于用户指令修改: {instruction[:20]}..."
                        # 重点：维护完整的历史
                        new_res['Interaction_History'] = history + [{"instruction": instruction, "plan": new_plan}]
                        
                        current_idea_plans.append(new_res)
                        print("\n[Success] 修改完成！新的版本已追加至列表末尾。请在菜单中选择最新版本查看效果。")
                    except Exception as e:
                        logger.error(f"交互修改发生异常: {e}")
                        print("修改发生错误，请查看日志。")
    if not args.interactive:
        return initial_output_path
    else:
        return final_path
            
        
    return initial_output_path
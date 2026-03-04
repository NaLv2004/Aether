
import json
import os
import random
import time
import argparse
import requests
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import setup_logger

logger = setup_logger("experiment_run.log")

# 导入你刚刚重构的 LLMAgent
from llm import LLMAgent

# ==========================================
# 1. 辅助函数：OpenAlex 文献检索
# ==========================================
def on_backoff(details):
    logger.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff)
def search_for_papers(query, result_limit=10, engine="openalex"):
    if not query:
        return None
        
    if engine == "openalex":
        import pyalex
        from pyalex import Works
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", "jiayanxu@seu.edu.cn")
        pyalex.config.email = mail

        def extract_info_from_work(work, max_abstract_length=1000):
            venue = "Unknown"
            if work.get("locations"):
                for location in work["locations"]:
                    if location.get("source"):
                        venue = location["source"].get("display_name", "Unknown")
                        if venue:
                            break
            title = work.get("title", "No Title")
            
            # 解析 abstract_inverted_index
            abstract = ""
            abstract_inverted_idx = work.get("abstract_inverted_index")
            if abstract_inverted_idx:
                max_index = max(pos for positions in abstract_inverted_idx.values() for pos in positions)
                abstract_words = [""] * (max_index + 1)
                for word, positions in abstract_inverted_idx.items():
                    for pos in positions:
                        abstract_words[pos] = word
                abstract = " ".join(abstract_words)
            else:
                abstract = work.get("abstract") or ""
                
            if len(abstract) > max_abstract_length:
                abstract = abstract[:max_abstract_length]
            
            authorships = work.get("authorships", [])
            authors_list = [a["author"]["display_name"] for a in authorships if a.get("author")]
            authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            
            return {
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": work.get("publication_year"),
                "abstract": abstract,
                "citationCount": work.get("cited_by_count", 0),
            }

        try:
            works = Works().search(query).get(per_page=result_limit)
            papers = [extract_info_from_work(work) for work in works]
            return papers
        except Exception as e:
            logger.info(f"[OpenAlex Search Error] {e}")
            return None
    else:
        raise NotImplementedError(f"{engine} not supported in this script!")

# ==========================================
# 2. Prompts 提示词定义
# ==========================================
IDEA_GENERATOR_SYSTEM_PROMPT = """
你是一个充满雄心壮志且富有创造力的通信领域AI科学家。
你的目标是提出具有高度创新性、跨学科（或跨细分领域）的科研Idea。
请遵循以下原则：
1. 鼓励大胆假设，并充分考虑实际通信场景的复杂性（如信道衰落、硬件损伤、动态拓扑等）。
2. 提出的假设必须非常具体，避免假大空。
3. 鼓励进行广泛的文献调研。当你需要搜索文献时，请使用逻辑词或模糊查询词（例如 ""semantic communication" AND "dynamic resource allocation"" 或 ""MIMO detection" OR "machine learning""），而不是仅搜索某篇具体的文章。
4. 你可以同时执行多项操作：生成新的Idea、优化（Refine）之前的Idea、发起新的文献搜索。
5. 当你认为已经生成了非常满意的Idea，不需要再进行迭代或搜索时，请在你的思考(Thoughts)中包含 "I'm done" 这句话以结束迭代。
6. 生成新的idea时，需要在返回的结果中包含之前的所有idea。
6. 虽然鼓励进行跨学科的研究，但是不鼓励研究过于天马行空、脱离实际、无法落地的idea，例如，和语义通信、量子计算、脉冲神经网络、类脑计算等结合，都是严重脱离实际的坏idea。
7. 研究对象不应该过于复杂，不应该堆砌过多技术名词，而是要聚焦于某一个具体问题，给出原理性的创新。

你的回复必须包含如下JSON格式（可以包含在 ```json 和 ``` 之间）：
```json
{
    "Thoughts": "这里写下你的思考过程、对当前结果的分析以及你接下来的计划。如果结束，请在这里包含 'I'm done'。",
    "SearchQueries": ["query1", "query2"], 
    "Ideas": [
        {
            "Name": "简短的Idea英文代号",
            "Title": "Idea的完整标题",
            "Background": "研究背景与动机",
            "Hypothesis": "具体的大胆假设",
            "Methodology": "具体的研究方法与实际复杂场景考量"
        }
    ]
}

```
如果当前轮次只是在调研文献尚未形成Idea，Ideas可为空列表。
除了在第一轮对话，在每轮对话中，你都需要提出新的idea，适当修改之前的idea，并且进行新的文献调研来确认现有idea是否具有创新性，并寻求潜在的新的创新方向。
"""

IDEA_GENERATOR_FIRST_PROMPT = """
我们正在探索以下粗略的研究主题：
【{theme}】

请根据该主题，提出你初步的想法，或者直接给出一系列广泛的文献检索Query以帮助你构思。请输出符合系统提示词要求的JSON。
"""

IDEA_GENERATOR_ITERATION_PROMPT = """
这是你在上一轮中提交的文献搜索Query的结果：
{search_results}

这是你之前已经生成的Ideas（供你参考，你可以选择Refine它们，或者提出全新的Idea）：
{previous_ideas}

请根据最新的文献结果，继续你的研究设想。如果发现你的Idea已经被前人做过，请大修你的假设。
请输出符合系统要求的JSON。如果你的Idea已经打磨完毕且创新性足够，请在Thoughts中包含 "I'm done"。
"""

NOVELTY_CHECK_SYSTEM_PROMPT = """
你是一位顶尖通信学术会议（如 GLOBECOM, ICC, INFOCOM, SIGCOMM）或知名学术期刊的资深审稿人 (Area Chair)。
你的任务是严格审查一个新提交的科研Idea是否具有真正的学术创新价值。
请注意这些idea均由AI生成，其正确性，合理性，创新性都无法保证，你必须谨慎评估。你必须做到非常严格。
在做决定之前，你必须充分利用OpenAlex API进行文献调研，确保该Idea没有与已有文献严重撞车，并判断这些idead的逻辑性、可实现性如何（后续所有工作都会由AI完成，因此必须判断仿真层面的可实现性）
由于OpenAlex对模糊检索支持有限，所以如果检索对象包含多个关键词，建议使用逻辑词连接search query，例如："Massive MIMO" OR "CHANNEL CODING"
你可以进行多轮搜索。
请在每轮回复中输出如下JSON：

```json
{
    "Thoughts": "你的审查思路、对Idea的评价或对文献搜索结果的分析。",
    "SearchQueries": ["查找该idea相关文献的Query"],
    "Decision": "Pending",
    "Score": null
}

```

当你在若干轮检索后有了明确结论，请将 "Decision" 设置为 "Finished"，并给出具体的 "Score" (1到10分，10分为最高分)，并在 "Thoughts" 中给出详细的评审意见和得分理由。
"""

NOVELTY_CHECK_EVAL_PROMPT = """
你需要评估以下科研Idea：
标题: {title}
背景: {background}
假设: {hypothesis}
方法与实际场景考量: {methodology}

当前搜索结果反馈：
{search_results}

请继续你的审查。如果还需要搜索文献，请输出 "SearchQueries" 并在 "Decision" 中填 "Pending"。如果审查完毕，请在 "Decision" 中填 "Finished"
无论你是否审查完毕，都需要给出"Thoughts"和"Score"。即便你认为搜索到的信息不足以判定，也应该结合你自己对相关学术领域已有的了解给出相应的字段。
"""

# ==========================================

# 3. Agents 工作流实现

# ==========================================

def format_search_results(queries, engine="openalex"):
    """执行批量检索并将结果格式化为字符串"""
    if not queries:
        return "没有进行文献检索。"

    results_str = ""
    for q in queries:
        papers = search_for_papers(q, result_limit=50, engine=engine)
        results_str += f"\n--- Query: [{q}] 的搜索结果 ---\n"
        if not papers:
            results_str += "未找到相关文献。\n"
        else:
            # logger.info(f"p['abstract'] ")
            for i, p in enumerate(papers):
                results_str += f"{i+1}. {p['title']} ({p['year']}) - {p['venue']}\n   Authors: {p['authors']}\n   Abstract: {p['abstract'][:300]}...\n"
    return results_str


def run_student_agent(student_id, theme, max_iters, model, log_dir):
    """Idea Generator Agent 的运行逻辑"""
    agent = LLMAgent(model=model, log_file=os.path.join(log_dir, f"log_student_{student_id}.log"))
    logger.info(f"[Student {student_id}] Started working on theme...")

    current_prompt = IDEA_GENERATOR_FIRST_PROMPT.format(theme=theme)
    current_ideas = []

    for i in range(max_iters):
        logger.info(f"[Student {student_id}] Iteration {i+1}/{max_iters}")
        response, _ = agent.get_response(current_prompt, IDEA_GENERATOR_SYSTEM_PROMPT)
        
        parsed_json = LLMAgent.extract_json_between_markers(response)
        if not parsed_json:
            logger.info(f"[Student {student_id}] Failed to parse JSON. Retrying...")
            current_prompt = "你的输出不符合JSON格式要求，请修正并重新输出。"
            continue
            
        thoughts = parsed_json.get("Thoughts", "")
        queries = parsed_json.get("SearchQueries", [])
        ideas = parsed_json.get("Ideas", [])
        
        if ideas:
            current_ideas = ideas # 更新为最新的 Ideas
            
        if "i'm done" in thoughts.lower():
            logger.info(f"[Student {student_id}] Finished early: {thoughts[:50]}...")
            break
            
        if i < max_iters - 1:
            search_feedback = format_search_results(queries)
            current_prompt = IDEA_GENERATOR_ITERATION_PROMPT.format(
                search_results=search_feedback,
                previous_ideas=json.dumps(current_ideas, indent=2, ensure_ascii=False)
            )
            
    logger.info(f"[Student {student_id}] Completed. Generated {len(current_ideas)} ideas.")
    return current_ideas


def run_teacher_agent(teacher_id, idea, max_iters, model, log_dir):
    """Novelty Check Agent 的运行逻辑"""
    agent = LLMAgent(model=model, log_file=os.path.join(log_dir, f"log_teacher_{teacher_id}.log"))
    logger.info(f"[Teacher {teacher_id}] Start reviewing idea: {idea.get('Title', 'Unknown')}")

    search_feedback = "目前尚未进行任何搜索。"
    final_score = None
    review_comments = ""

    for i in range(max_iters):
        current_prompt = NOVELTY_CHECK_EVAL_PROMPT.format(
            title=idea.get('Title', ''),
            background=idea.get('Background', ''),
            hypothesis=idea.get('Hypothesis', ''),
            methodology=idea.get('Methodology', ''),
            search_results=search_feedback
        )
        
        response, _ = agent.get_response(current_prompt, NOVELTY_CHECK_SYSTEM_PROMPT)
        parsed_json = LLMAgent.extract_json_between_markers(response)
        
        if not parsed_json:
            search_feedback = "请严格按照要求的JSON格式输出你的评估和搜索Query。"
            continue
            
        decision = parsed_json.get("Decision", "Pending")
        if decision == "Finished":
            final_score = parsed_json.get("Score")
            review_comments = parsed_json.get("Thoughts", "")
            logger.info(f"[Teacher {teacher_id}] Review finished. Score: {final_score}")
            break
        else:
            final_score = parsed_json.get("Score")
            review_comments = parsed_json.get("Thoughts", "")
        
            
        queries = parsed_json.get("SearchQueries", [])
        search_feedback = format_search_results(queries)
        
    return {
        "Idea": idea,
        "Reviewer_ID": teacher_id,
        "Score": final_score,
        "Review_Comments": review_comments
    }

# ==========================================

# 4. 主控流程
def generate_ideas(args):
    # args = parser.parse_args()

    # 1. 读取主题文件 
    if not os.path.exists(args.theme_file):
        logger.info(f"找不到主题文件: {args.theme_file}。请先创建该文件并写入研究主题。")
        return
    with open(args.theme_file, "r", encoding="utf-8") as f:
        theme = f.read().strip()

    logger.info(f"=== 启动 AI Scientist ===")
    logger.info(f"Theme: {theme}")
    logger.info(f"Students: {args.n_students}, Teachers: {args.n_teachers}, Model: {args.model}")
    logger.info("=========================\n")

    # 2. 阶段一：并行启动 Idea Generators
    all_ideas = []
    logger.info(">>> 阶段一：Idea Generation (并发多 Agent) <<<")
    with ThreadPoolExecutor(max_workers=args.n_students) as executor:
        future_to_student = {
            executor.submit(run_student_agent, i+1, theme, args.max_student_iters, args.model, args.log_dir): i+1 
            for i in range(args.n_students)
        }
        for future in as_completed(future_to_student):
            student_ideas = future.result()
            all_ideas.extend(student_ideas)
            
    # 将所有的idea记录到单独的txt文件中
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_ideas, f, indent=4, ensure_ascii=False)
    logger.info(f"阶段一结束。共生成 {len(all_ideas)} 个Ideas，已保存至 {args.output_file}。\n")

    if not all_ideas:
        logger.info("未生成任何Idea，程序退出。")
        return

    # 3. 打乱 Idea 顺序
    random.shuffle(all_ideas)

    # 4. 阶段二：并行启动 Novelty Checkers (Teachers) 进行评估
    # 为了简化，我们将所有打乱后的idea分配给 N 个 teacher 组成的线程池进行评估。
    logger.info(">>> 阶段二：Novelty Check (并发严苛审稿) <<<")
    evaluated_results = []

    # 控制并发量为 n_teachers，处理所有的 ideas
    with ThreadPoolExecutor(max_workers=args.n_teachers) as executor:
        future_to_idea = {
            executor.submit(run_teacher_agent, idx % args.n_teachers + 1, idea, args.max_teacher_iters, args.model, args.log_dir): idea
            for idx, idea in enumerate(all_ideas)
        }
        for future in as_completed(future_to_idea):
            result = future.result()
            evaluated_results.append(result)

    # 5. 输出审查结果到控制台和独立的Log文件
    logger.info("\n>>> 最终审查结果汇总 <<<")
    with open(args.review_log, "w", encoding="utf-8") as f:
        for idx, res in enumerate(evaluated_results):
            idea_title = res["Idea"].get("Title", "Unknown Title")
            score = res["Score"]
            comments = res["Review_Comments"]
            
            output_str = f"--- Idea {idx+1} ---\n"
            output_str += f"Title: {idea_title}\n"
            output_str += f"Score: {score}/10\n"
            output_str += f"Review Comments:\n{comments}\n"
            output_str += "="*40 + "\n"
            
            logger.info(output_str)
            f.write(output_str)
            
    logger.info(f"\n所有评估结束，详细评分已保存至 {args.review_log}。")
    return args.output_file # 返回生成的idea文件路径
    pass
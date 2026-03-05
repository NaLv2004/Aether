
import json
import os
import random
import time
import argparse
import requests
import backoff
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import setup_logger
from utils import PDFReader

logger = setup_logger("experiment_run.log")

# 导入你刚刚重构的 LLMAgent
from llm import LLMAgent

# ==========================================
# 1. 辅助函数：OpenAlex 文献检索与 PDF 下载
# ==========================================
def on_backoff(details):
    logger.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff)
def search_for_papers(query, result_limit=10, engine="openalex", open_access=True, has_pdf_url=True, from_year=2020):
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
            doi = work.get("doi", "No DOI")
            
            # 获取下载链接：优先从 best_oa_location 获取 pdf_url
            pdf_url = None
            best_oa = work.get("best_oa_location")
            if best_oa:
                pdf_url = best_oa.get("pdf_url")
            
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
                "doi": doi,
                "pdf_url": pdf_url
            }

        try:
            # 构建查询并应用过滤条件
            search_query = Works().search(query)
            if open_access:
                search_query = search_query.filter(is_oa=True)
            if has_pdf_url:
                search_query = search_query.filter(has_pdf_url=True)
            if from_year:
                search_query = search_query.filter(from_publication_date=f"{from_year}-01-01")
                
            works = search_query.get(per_page=result_limit)
            papers = [extract_info_from_work(work) for work in works]
            return papers
        except Exception as e:
            logger.info(f"[OpenAlex Search Error] {e}")
            return None
    else:
        raise NotImplementedError(f"{engine} not supported in this script!")


def download_paper_pdf(pdf_url, doi, save_dir="pdfs"):
    """
    下载论文 PDF，更加鲁棒，不限于 arXiv。
    """
    if not pdf_url:
        return None
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 用 DOI 构造安全的文件名
    safe_name = urllib.parse.quote_plus(doi.replace("https://doi.org/", ""))
    filename = f"{safe_name[:50]}.pdf"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        logger.info(f"[Download Info] PDF 已经存在: {filename}")
        return save_path

    logger.info(f"[Download] 尝试下载 PDF: {pdf_url}")
    try:
        # 伪装 User-Agent，防止被简单的反爬拦截
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        # 允许重定向，设置超时时间
        response = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            # 严格验证是否返回了 PDF 文件
            if "application/pdf" in content_type or "binary/octet-stream" in content_type:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[Success] PDF 下载成功: {filename}")
                return save_path
            else:
                logger.debug(f"[Warning] 下载失败：返回的内容类型不是 PDF ({content_type})")
        else:
             logger.debug(f"[Error] 下载失败：HTTP 状态码 {response.status_code}")
             
    except Exception as e:
        logger.debug(f"[Error] PDF 下载过程发生异常: {e}")
        
    return None


# ==========================================
# 2. Prompts 提示词定义 (新增了 PapersToRead 字段)
# ==========================================
IDEA_GENERATOR_SYSTEM_PROMPT = """
你是一个充满雄心壮志且富有创造力的通信领域AI科学家。
你的目标是提出具有高度创新性、跨学科（或跨细分领域）的科研Idea。
请遵循以下原则：
1. 鼓励大胆假设，并充分考虑实际通信场景的复杂性（如信道衰落、硬件损伤、动态拓扑等）。
2. 提出的假设必须非常具体，避免假大空。
3. 鼓励进行广泛的文献调研。当你需要搜索文献时，请使用逻辑词或模糊查询词。
4. 你可以同时执行多项操作：生成新的Idea、优化（Refine）之前的Idea、发起新的文献搜索、选择阅读某篇文献的全文。
6. 生成新的idea时，需要在返回的结果中包含之前的所有idea。
7. 研究对象不应该过于复杂，不应该堆砌过多技术名词，而是要聚焦于某一个具体问题，给出原理性的创新。
8。每轮对话中，你都需要refine之前的idea（更加具体、可行、具有合理性），或者努力根据你已有的知识，或者搜索到的论文，提出新的insight和idea(或使用新的insights修改当前idea)。

如果当前搜索结果中的摘要不足以判断，你可以要求阅读全文。将你想要阅读的论文的 DOI 填入 "PapersToRead" 列表中。系统会自动下载、阅读并把核心总结返回给你。
此外，你需要通过阅读其他论文和摘要产生新的insights，而不是仅仅依据这些内容来鉴定创新性。此外，需要保证你提出的idea不仅仅是简单的排列组合。最好能够有较长，较完整的创新逻辑链条。

你的回复必须包含如下JSON格式（可以包含在 ```json 和 ``` 之间）：
```json
{
    "Thoughts": "这里写下你的思考过程、对当前idea的分析以及你接下来的计划。",
    "SearchQueries": ["query1", "query2"], 
    "PapersToRead": ["https://doi.org/10.xxxx/xxxx", "https://doi.org/10.yyyy/yyyy"],
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
SearchQueries 不可为空，Ideas字段不可以为空。；如果不需要读全文，PapersToRead 可为空。
"""

# IDEA_GENERATOR_SYSTEM_PROMPT = """
# 你是一个充满雄心壮志且富有创造力的通信领域AI科学家。
# 你的目标是根据给定的主题，提出能够被顶级通信期刊接收的idea。
# 请遵循以下原则：
# 1. 鼓励大胆假设，提出的idea需要能够被TCOM等期刊接收。
# 2. 提出的假设必须非常具体，避免假大空。
# 3. 鼓励进行广泛的文献调研。当你需要搜索文献时，请使用逻辑词或模糊查询词。
# 4. 你可以同时执行多项操作：生成新的Idea、优化（Refine）之前的Idea、发起新的文献搜索、选择阅读某篇文献的全文。
# 6. 生成新的idea时，必须在返回的结果中包含之前的所有idea。
# 7. 研究对象不应该过于复杂，不应该堆砌过多技术名词，而是要聚焦于某一个具体问题，给出原理性的创新。
# 8。每轮对话中，你都需要refine之前的idea（更加具体、可行、具有合理性），或者努力根据你已有的知识，或者搜索到的论文，提出新的insight和idea(或使用新的insights修改当前idea)。

# 如果当前搜索结果中的摘要不足以判断，你可以要求阅读全文。将你想要阅读的论文的 DOI 填入 "PapersToRead" 列表中。系统会自动下载、阅读并把核心总结返回给你。
# 此外，你需要通过阅读其他论文和摘要产生新的insights，而不是仅仅依据这些内容来鉴定创新性。

# 你的回复必须包含如下JSON格式（可以包含在 ```json 和 ``` 之间）：
# ```json
# {
#     "Thoughts": "这里写下你的思考过程、对当前idea的分析以及你接下来的计划。",
#     "SearchQueries": ["query1", "query2"], 
#     "PapersToRead": ["https://doi.org/10.xxxx/xxxx", "https://doi.org/10.yyyy/yyyy"],
#     "Ideas": [
#         {
#             "Name": "简短的Idea英文代号",
#             "Title": "Idea的完整标题",
#             "Background": "研究背景与动机",
#             "Hypothesis": "具体的大胆假设",
#             "Methodology": "具体的研究方法与实际复杂场景考量"
#         }
#     ]
# }
# ```
# 如果不需要搜文献，SearchQueries 可为空；如果不需要读全文，PapersToRead 可为空；如果是第一轮只负责搜文献，Ideas 可为空。
# """
IDEA_GENERATOR_FIRST_PROMPT = """
我们正在探索以下粗略的研究主题：
【{theme}】

请根据该主题，提出你初步的想法和idea，并且给出一系列广泛的文献检索Query以帮助你构思。请输出符合系统提示词要求的JSON。
"""

IDEA_GENERATOR_ITERATION_PROMPT = """
这是你在上一轮中提交的文献搜索Query的结果：
{search_results}

这是你（或其它评审）之前要求精读的论文的全文总结笔记（知识库）：
{knowledge_base}

这是你之前已经生成的Ideas（供你参考，你可以选择Refine它们，或者提出全新的Idea）.注意，每次json中返回的Ideas列表中，至少要包含之前提出的所有Ideas，也必须追加新的Ideas,并refine之前的。
{previous_ideas}

请根据最新的文献结果和精读笔记，继续你的研究设想。如果发现你的Idea已经被前人做过，请大修你的假设。如果需要阅读新搜索出的文献全文，请填入 PapersToRead。
请输出符合系统要求的JSON。每轮搜索中，你都必须合理地refine你的idea。同时，你被鼓励多阅读全文。
"""

NOVELTY_CHECK_SYSTEM_PROMPT = """
你是一位顶尖通信学术会议（如 GLOBECOM, ICC）或知名学术期刊的资深审稿人 (Area Chair)。
你的任务是严格审查一个新提交的科研Idea是否具有真正的学术创新价值。
请注意这些idea均由AI生成，其正确性，合理性，创新性都无法保证，你必须谨慎评估。
在做决定之前，你必须充分利用文献调研，确保该Idea没有与已有文献严重撞车。

如果你怀疑某篇已发表的论文已经做过了这个 Idea，但仅仅看摘要无法确定，你可以将该论文的 DOI 放入 "PapersToRead" 列表中，要求系统阅读全文并提供核心总结。

请在每轮回复中输出如下JSON：
```json
{
    "Thoughts": "你的审查思路、对Idea的评价或对文献搜索结果的分析。",
    "SearchQueries": ["查找该idea相关文献的Query"],
    "PapersToRead": ["https://doi.org/10.xxxx/xxxx"],
    "Decision": "Pending",
    "Score": null
}
```

若检索轮数超过8轮而且你在若干轮检索/阅读后有了明确结论，请将 "Decision" 设置为 "Finished"，并给出具体的 "Score" (1到10分)，并在 "Thoughts" 中给出详细的评审意见和得分理由。
"""

NOVELTY_CHECK_EVAL_PROMPT = """
你需要评估以下科研Idea：
标题: {title}
背景: {background}
假设: {hypothesis}
方法与实际场景考量: {methodology}

当前搜索结果反馈：
{search_results}

当前精读文献的全文笔记（知识库）：
{knowledge_base}

请继续你的审查。如果还需要搜索/阅读，请在Decision中填 "Pending"。如果审查完毕，请填 "Finished"。
"""

PDFReader_PROMPT = """
你是一个高级学术助理。你的任务是仔细阅读提供的PDF文献，并总结出其核心创新点(Key Takeaway)、使用的方法、以及它解决了什么具体问题。
你需要详细解释：
0) 论文标题（放在第一行）
1）这篇文章想解决什么问题
2）这篇文章采用了什么样的系统模型（给出具体文字描述和准确的公式描述）
3）详细解释这篇文章的所有创新点，对关键创新点给出具体公式
4）这篇文章的关键结论，取得的增益等
5）概括这篇文章中总结的本领域之前的研究进展
"""
# ==========================================
# 3. Agents 工作流实现与工具封装
# ==========================================

def format_search_results_and_update_map(queries, doi_url_map, engine="openalex", open_access=True, has_pdf_url=True, from_year=2020):
    """执行检索，格式化字符串，并更新 doi_to_url 的映射表以便后续下载"""
    if not queries:
        return "没有进行文献检索。"

    results_str = ""
    for q in queries:
        papers = search_for_papers(q, result_limit=20, engine=engine, 
                                   open_access=open_access, has_pdf_url=has_pdf_url, from_year=from_year)
        results_str += f"\n--- Query: [{q}] 的搜索结果 ---\n"
        if not papers:
            results_str += "未找到相关文献。\n"
        else:
            for i, p in enumerate(papers):
                # 记录 DOI 到 PDF URL 的映射
                if p['doi'] and p['pdf_url']:
                    doi_url_map[p['doi']] = p['pdf_url']
                
                results_str += f"{i+1}. {p['title']} ({p['year']}) - {p['venue']}\n"
                results_str += f"   DOI: {p['doi']}\n"  # 必须展示 DOI，让 Agent 知道填什么
                results_str += f"   Authors: {p['authors']}\n   Abstract: {p['abstract'][:300]}...\n"
    return results_str


def process_papers_to_read(papers_to_read, doi_url_map, kb_txt_path):
    """处理下载并阅读 PDF 的逻辑"""
    if not papers_to_read:
        return
    
    # 获取 Gemini API Key 供 PDFReader 使用
    gemini_api_key = os.environ.get("JIANYI_API_KEY") 
    if not gemini_api_key:
        logger.warning("未设置 GEMINI_API_KEY 环境变量，跳过 PDF 全文阅读！")
        return

    # 初始化用于阅读长文的 Reader Agent (每次阅读用全新实例，防止上下文污染)
    pdf_reader = PDFReader(
        api_key=gemini_api_key,
        system_prompt=PDFReader_PROMPT,
        context_window_size=1
    )

    for doi in papers_to_read:
        pdf_url = doi_url_map.get(doi)
        if not pdf_url:
            logger.debug(f"无法找到 DOI: {doi} 对应的 PDF 下载链接，跳过阅读。")
            continue
            
        pdf_path = download_paper_pdf(pdf_url, doi)
        if pdf_path:
            logger.info(f"正在交由 AI 深入阅读: {pdf_path}")
            # 调用 PDFReader 获取并追加知识到 txt
            pdf_reader.read_pdf(
                pdf_path=pdf_path, 
                output_txt_path=kb_txt_path, 
                user_prompt=f"Please read this paper (DOI: {doi}) and summarize its core methodology and key takeaways."
            )
        else:
             logger.debug(f"PDF 下载失败，跳过阅读 DOI: {doi}")


def read_knowledge_base(txt_path):
    """读取已存储的全文阅读笔记"""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else "暂无精读笔记。"
    return "暂无精读笔记。"


def run_student_agent(student_id, theme, max_iters, model, log_dir, search_params):
    """Idea Generator Agent 的运行逻辑"""
    agent = LLMAgent(model=model, log_file=os.path.join(log_dir, f"log_student_{student_id}.log"))
    logger.info(f"[Student {student_id}] Started working on theme...")
    agent.set_context_len(4)
    # 每个 Student 独享一个 Knowledge Base 文件，用于存储它读过的论文笔记
    kb_txt_path = os.path.join(log_dir, f"kb_student_{student_id}.txt")
    
    # 存储本轮搜索出的文献供下载查找
    doi_url_map = {} 
    
    current_prompt = IDEA_GENERATOR_FIRST_PROMPT.format(theme=theme)
    current_ideas = []

    for i in range(max_iters):
        logger.info(f"[Student {student_id}] Iteration {i+1}/{max_iters}")
        response, _ = agent.get_response_stream(current_prompt, IDEA_GENERATOR_SYSTEM_PROMPT)
        
        parsed_json = LLMAgent.robust_extract_json(response)
        if not parsed_json:
            logger.debug(f"[Student {student_id}] Failed to parse JSON. Retrying...")
            current_prompt = "你的输出不符合JSON格式要求，请修正并重新输出。"
            continue
            
        thoughts = parsed_json.get("Thoughts", "")
        queries = parsed_json.get("SearchQueries", [])
        papers_to_read = parsed_json.get("PapersToRead", [])
        ideas = parsed_json.get("Ideas", [])
        
        if ideas:
            current_ideas = ideas
            
        if "i'm done" in thoughts.lower():
            logger.info(f"[Student {student_id}] Finished early: {thoughts[:50]}...")
            break
            
        if i < max_iters - 1:
            # 1. 触发全文精读并写入 kb_txt_path
            if papers_to_read:
                logger.info(f"[Student {student_id}] 要求阅读全文: {papers_to_read}")
                process_papers_to_read(papers_to_read, doi_url_map, kb_txt_path)
                
            # 2. 执行新的检索，更新 doi_url_map 供下一轮下载用
            search_feedback = format_search_results_and_update_map(
                queries, doi_url_map, 
                open_access=search_params['open_access'], 
                has_pdf_url=search_params['has_pdf_url'], 
                from_year=search_params['from_year']
            )
            
            # 3. 读取最新的精读知识库
            kb_content = read_knowledge_base(kb_txt_path)
            
            # 4. 构建下一轮 Prompt
            current_prompt = IDEA_GENERATOR_ITERATION_PROMPT.format(
                search_results=search_feedback,
                knowledge_base=kb_content,
                previous_ideas=json.dumps(current_ideas, indent=2, ensure_ascii=False)
            )
            
    logger.info(f"[Student {student_id}] Completed. Generated {len(current_ideas)} ideas.")
    return current_ideas


def run_teacher_agent(teacher_id, idea, max_iters, model, log_dir, search_params):
    """Novelty Check Agent 的运行逻辑"""
    agent = LLMAgent(model=model, log_file=os.path.join(log_dir, f"log_teacher_{teacher_id}.log"))
    logger.info(f"[Teacher {teacher_id}] Start reviewing idea: {idea.get('Title', 'Unknown')}")

    kb_txt_path = os.path.join(log_dir, f"kb_teacher_{teacher_id}.txt")
    doi_url_map = {}
    
    search_feedback = "目前尚未进行任何搜索。"
    final_score = None
    review_comments = ""

    for i in range(max_iters):
        # 读取此 Reviewer 看过的全文笔记
        kb_content = read_knowledge_base(kb_txt_path)
        
        current_prompt = NOVELTY_CHECK_EVAL_PROMPT.format(
            title=idea.get('Title', ''),
            background=idea.get('Background', ''),
            hypothesis=idea.get('Hypothesis', ''),
            methodology=idea.get('Methodology', ''),
            search_results=search_feedback,
            knowledge_base=kb_content
        )
        
        response, _ = agent.get_response_stream(current_prompt, NOVELTY_CHECK_SYSTEM_PROMPT)
        parsed_json = LLMAgent.robust_extract_json(response)
        
        if not parsed_json:
            search_feedback = "请严格按照要求的JSON格式输出你的评估和搜索Query。"
            continue
            
        decision = parsed_json.get("Decision", "Pending")
        final_score = parsed_json.get("Score")
        review_comments = parsed_json.get("Thoughts", "")
        
        if decision == "Finished":
            logger.info(f"[Teacher {teacher_id}] Review finished. Score: {final_score}")
            break
            
        # 处理全文精读请求
        papers_to_read = parsed_json.get("PapersToRead", [])
        if papers_to_read:
            logger.info(f"[Teacher {teacher_id}] 要求阅读全文核查 Novelty: {papers_to_read}")
            process_papers_to_read(papers_to_read, doi_url_map, kb_txt_path)
            
        # 执行新的检索更新 Feedback
        queries = parsed_json.get("SearchQueries", [])
        search_feedback = format_search_results_and_update_map(
            queries, doi_url_map, 
            open_access=search_params['open_access'], 
            has_pdf_url=search_params['has_pdf_url'], 
            from_year=search_params['from_year']
        )
        
    return {
        "Idea": idea,
        "Reviewer_ID": teacher_id,
        "Score": final_score,
        "Review_Comments": review_comments
    }



IDEA_REFINER_SYSTEM_PROMPT = """
你是一个专业的通信领域科研助手。你的任务是根据用户的反馈和指令，修改并完善一个已有的科研Idea。
请遵循以下原则：
1. 严格针对用户提出的修改意见进行调整。如果用户认为某部分不合理，请根据你的专业知识进行修正。
2. 如果用户允许搜索文献，请积极利用搜索工具验证新的假设或寻找解决方案。
3. 保持Idea的完整性，输出格式必须严格遵守JSON结构。
4. 你输出的 "Ideas" 列表中应当只包含当前正在修改的这一个Idea（即修改后的版本）。

你的回复必须包含如下JSON格式（可以包含在 ```json 和 ``` 之间）：
```json
{
    "Thoughts": "你对用户反馈的理解，以及你修改Idea的思路。",
    "SearchQueries": ["query1", "query2"], 
    "PapersToRead": ["https://doi.org/10.xxxx/xxxx"],
    "Ideas": [
        {
            "Name": "Idea代号(保持不变)",
            "Title": "修改后的标题",
            "Background": "修改后的背景",
            "Hypothesis": "修改后的假设",
            "Methodology": "修改后的方法"
        }
    ]
}
```
如果不需要搜文献或用户禁止搜索，SearchQueries 应为空列表；如果不需要读全文，PapersToRead 应为空列表。
"""

IDEA_REFINER_START_PROMPT = """
这是当前版本的Idea：
{current_idea}

用户对该Idea的修改反馈/指令如下：
【{user_feedback}】

请根据用户的指令，对Idea进行修改和完善。
"""

IDEA_REFINER_ITERATION_PROMPT = """
这是你在上一轮中提交的文献搜索Query的结果：
{search_results}

这是你要求精读的论文的全文总结笔记（知识库）：
{knowledge_base}

这是上一轮修改后的Idea版本：
{previous_idea}

请根据最新的文献结果和之前的思路，继续完善Idea。确保回应了用户的初始反馈。
"""
def refine_idea(idea, user_instructions, allow_search, max_iters, model, log_dir, search_params):
    """
    根据用户指令Refine特定的Idea。
    """
    # 为Refine过程创建一个临时的Agent
    refine_id = int(time.time()) # 使用时间戳作为ID避免冲突
    agent = LLMAgent(model=model, log_file=os.path.join(log_dir, f"log_refiner_{refine_id}.log"))
    logger.info(f"[Refiner] Started refining idea: {idea.get('Name', 'Unknown')}")
    agent.set_context_len(4)
    
    # 独立的知识库文件
    kb_txt_path = os.path.join(log_dir, f"kb_refiner_{refine_id}.txt")
    doi_url_map = {} 
    
    current_idea = idea
    # 初始 Prompt
    current_prompt = IDEA_REFINER_START_PROMPT.format(
        current_idea=json.dumps(current_idea, indent=2, ensure_ascii=False),
        user_feedback=user_instructions
    )

    for i in range(max_iters):
        logger.info(f"[Refiner] Iteration {i+1}/{max_iters}")
        
        # 获取 LLM 响应
        response, _ = agent.get_response_stream(current_prompt, IDEA_REFINER_SYSTEM_PROMPT)
        parsed_json = LLMAgent.robust_extract_json(response)
        
        if not parsed_json:
            logger.debug(f"[Refiner] Failed to parse JSON. Retrying...")
            current_prompt = "你的输出不符合JSON格式要求，请修正并重新输出。"
            continue
            
        thoughts = parsed_json.get("Thoughts", "")
        queries = parsed_json.get("SearchQueries", [])
        papers_to_read = parsed_json.get("PapersToRead", [])
        refined_ideas_list = parsed_json.get("Ideas", [])
        
        # 更新当前的 Idea
        if refined_ideas_list and len(refined_ideas_list) > 0:
            current_idea = refined_ideas_list[0]
            
        # 如果 LLM 认为完成了（可以通过thoughts判断，也可以跑满轮数），这里简化为跑满轮数或由用户判断
        # 但为了让 Agent 有机会利用搜索结果，我们需要继续循环
        
        # 处理搜索和阅读逻辑
        search_feedback = "用户未开启搜索权限或本轮未进行搜索。"
        kb_content = "暂无新笔记。"

        if allow_search:
            # 1. 阅读全文
            if papers_to_read:
                logger.info(f"[Refiner] 要求阅读全文: {papers_to_read}")
                process_papers_to_read(papers_to_read, doi_url_map, kb_txt_path)
            
            # 2. 搜索文献
            if queries:
                search_feedback = format_search_results_and_update_map(
                    queries, doi_url_map, 
                    open_access=search_params['open_access'], 
                    has_pdf_url=search_params['has_pdf_url'], 
                    from_year=search_params['from_year']
                )
            
            # 3. 读取知识库
            kb_content = read_knowledge_base(kb_txt_path)
        else:
            # 如果不允许搜索，且已经生成了新的Idea，其实可以提前结束，或者让模型自省一轮
            if i == 0: 
                logger.info("[Refiner] Search disabled. Only using internal knowledge.")
            
        # 构建下一轮 Prompt
        current_prompt = IDEA_REFINER_ITERATION_PROMPT.format(
            search_results=search_feedback,
            knowledge_base=kb_content,
            previous_idea=json.dumps(current_idea, indent=2, ensure_ascii=False)
        )
        
    logger.info(f"[Refiner] Refinement completed.")
    return current_idea


# ==========================================
# 4. 主控流程
# ==========================================
def generate_ideas(args, open_access=True, has_pdf_url=True, from_year=2020, interactive=True):
    """
    修改了函数签名，支持将搜索参数暴露出来
    """
    # 包装搜索参数字典传递给 Agent
    search_params = {
        "open_access": open_access,
        "has_pdf_url": has_pdf_url,
        "from_year": from_year
    }

    if not os.path.exists(args.theme_file):
        logger.info(f"找不到主题文件: {args.theme_file}。请先创建该文件并写入研究主题。")
        return
        
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    with open(args.theme_file, "r", encoding="utf-8") as f:
        theme = f.read().strip()

    logger.info(f"=== 启动 AI Scientist ===")
    logger.info(f"Theme: {theme}")
    logger.info(f"Search Config - OA:{open_access}, PDF:{has_pdf_url}, Year>={from_year}")
    logger.info(f"Students: {args.n_students}, Teachers: {args.n_teachers}, Model: {args.model}")
    logger.info("=========================\n")

    # >>> 阶段一：并行启动 Idea Generators <<<
    all_ideas = []
    logger.info(">>> 阶段一：Idea Generation (并发多 Agent) <<<")
    with ThreadPoolExecutor(max_workers=args.n_students) as executor:
        future_to_student = {
            executor.submit(run_student_agent, i+1, theme, args.max_student_iters, args.model, args.log_dir, search_params): i+1 
            for i in range(args.n_students)
        }
        for future in as_completed(future_to_student):
            student_ideas = future.result()
            all_ideas.extend(student_ideas)
            
    # 保存所有初步生成的 idea
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_ideas, f, indent=4, ensure_ascii=False)
    logger.info(f"阶段一结束。共生成 {len(all_ideas)} 个Ideas，已保存至 {args.output_file}。\n")

    if not all_ideas:
        logger.info("未生成任何Idea，程序退出。")
        return

    # 打乱 Idea 顺序
    random.shuffle(all_ideas)

    # >>> 阶段二：并行启动 Novelty Checkers (Teachers) <<<
    logger.info(">>> 阶段二：Novelty Check (并发严苛审稿) <<<")
    evaluated_results = []

    with ThreadPoolExecutor(max_workers=args.n_teachers) as executor:
        future_to_idea = {
            executor.submit(run_teacher_agent, idx % args.n_teachers + 1, idea, args.max_teacher_iters, args.model, args.log_dir, search_params): idea
            for idx, idea in enumerate(all_ideas)
        }
        for future in as_completed(future_to_idea):
            result = future.result()
            evaluated_results.append(result)

    # 汇总输出
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
    if interactive:
        print("\n" + "="*60)
        print("   INTERACTIVE MODE ENABLED")
        print("="*60)
        
        # --- 1. 整合所有 Ideas 和 评审结果 ---
        # 建立一个 ID 到 评审结果 的映射，方便查找
        # 注意：all_ideas 是原始列表，evaluated_results 是评审过的列表
        # 我们使用对象的内存 id 或者标题来匹配（假设标题唯一，或者直接引用对象）
        
        review_map = {}
        if evaluated_results:
            for res in evaluated_results:
                # 这里的 res['Idea'] 是 all_ideas 中某个字典的引用
                # 我们可以用 id(res['Idea']) 作为 key
                review_map[id(res['Idea'])] = {
                    "score": res["Score"],
                    "comments": res["Review_Comments"]
                }
        
        # 准备显示列表：包含 all_ideas 中的每一个
        display_list = []
        for idx, idea in enumerate(all_ideas):
            review_info = review_map.get(id(idea))
            if review_info:
                score = review_info['score']
                comments = review_info['comments']
            else:
                score = "Pending/Not Reviewed"
                comments = "该 Idea 尚未被 Teacher Agent 评审。"
            
            display_list.append({
                "id": idx + 1,
                "idea": idea,
                "score": score,
                "comments": comments
            })

        user_satisfied = False
        
        # --- 2. 主交互循环 ---
        while True:
            print(f"\n>>> 总共生成了 {len(display_list)} 个 Idea。列表如下：\n")
            
            # 完整展示所有 Idea 的内容
            for item in display_list:
                print(f"Option [{item['id']}] (Score: {item['score']})")
                print("-" * 40)
                # 使用 json.dumps 格式化打印完整内容
                print(json.dumps(item['idea'], indent=4, ensure_ascii=False))
                print("-" * 40)
                if item['score'] != "Pending/Not Reviewed":
                    print(f"Review Comments: {item['comments'][:200]}..." + (" (more)" if len(item['comments']) > 200 else ""))
                print("=" * 60 + "\n")

            # 用户选择
            try:
                choice = input("\n请输入你想 **进一步处理** 的 Idea 编号 (输入 q 退出): ").strip()
                if choice.lower() == 'q':
                    logger.info("用户退出交互模式。")
                    return args.output_file
                
                selected_idx = int(choice) - 1
                if not (0 <= selected_idx < len(display_list)):
                    print("无效的编号，请重新输入。")
                    continue
            except ValueError:
                print("请输入数字。")
                continue

            # 选中了某个 Idea，进入该 Idea 的子循环
            current_item = display_list[selected_idx]
            current_idea = current_item['idea']
            
            while True:
                print("\n" + "#"*50)
                print(f"   当前选中: Option [{current_item['id']}]")
                print("#"*50)
                print(json.dumps(current_idea, indent=4, ensure_ascii=False))
                print("\n>>> 评审意见:")
                print(current_item['comments'])
                print("#"*50)

                confirm = input("\n请选择操作:\n [y] 满意此版本 (保存并结束)\n [n] 不满意 (进入修改/Refine模式)\n [b] 返回 Idea 列表\n [q] 退出程序\n您的选择: ").lower().strip()
                
                if confirm == 'q':
                    logger.info("用户退出。")
                    return args.output_file
                
                elif confirm == 'b':
                    break # 跳出子循环，回到列表展示
                
                elif confirm == 'y':
                    # 虽然已经保存过，但这里可以确认最终版本
                    final_path = args.output_file.replace(".json", "_final_selected.json")
                    with open(final_path, "w", encoding="utf-8") as f:
                        json.dump(current_idea, f, indent=4, ensure_ascii=False)
                    print(f"最终选定的 Idea 已保存至: {final_path}")
                    return final_path
                
                elif confirm == 'n':
                    # 进入 Refine 流程
                    instructions = input("\n请输入你的具体修改指令 (例如: '增加对低轨卫星场景的考虑'): ").strip()
                    if not instructions:
                        print("指令为空，取消修改。")
                        continue
                    
                    search_choice = input("允许 Agent 搜索新文献吗? (y/n, 默认 n): ").lower()
                    allow_search = (search_choice == 'y')
                    
                    print(f"\n>>> 正在启动 Refiner Agent 对 Idea 进行优化 (Search={allow_search})...")
                    
                    try:
                        refined_idea_result = refine_idea(
                            idea=current_idea,
                            user_instructions=instructions,
                            allow_search=allow_search,
                            max_iters=3,  # 设定为3轮，防止等待过久
                            model=args.model,
                            log_dir=args.log_dir,
                            search_params=search_params
                        )
                        
                        # 更新内存中的 Idea
                        current_idea = refined_idea_result
                        current_item['idea'] = current_idea # 更新列表中的引用
                        
                        # 自动备份这个 Refined 版本
                        refined_filename = f"refined_idea_{current_item['id']}_{int(time.time())}.json"
                        # 优先存 output_dir，没有则存 log_dir
                        save_dir = getattr(args, 'output_dir', args.log_dir)
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        
                        refined_path = os.path.join(save_dir, refined_filename)
                        with open(refined_path, "w", encoding="utf-8") as f:
                            json.dump(current_idea, f, indent=4, ensure_ascii=False)
                            
                        print(f"\n[Success] 修改完成！新版本已保存至 {refined_path}")
                        print("请查看上方的新版本内容。")
                        # 循环会回到子循环开头，展示新的 JSON 内容
                        
                    except Exception as e:
                        logger.error(f"Refine 过程出错: {e}")
                        print("修改过程中发生错误，请检查日志。")
                else:
                    print("无效输入，请重新选择。")

    if not interactive:
        return args.output_file
    else:
        return refined_path




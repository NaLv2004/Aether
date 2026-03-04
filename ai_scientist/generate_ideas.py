
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



import os
import requests
import http.cookiejar
import re
from urllib.parse import urlparse

def load_cookies_from_netscape(cookie_file):
    """
    从 Netscape 格式的 txt 文件加载 cookie (兼容 wget/curl 格式)
    """
    cj = http.cookiejar.MozillaCookieJar(cookie_file)
    try:
        cj.load()
        return cj
    except Exception as e:
        print(f"[Cookie Error] Failed to load cookies: {e}")
        return None


# get pdf from IEEEXplore. 
def download_paper_pdf(paper_info, save_dir="pdfs", ieee_cookie_path="ieee_cookies.txt"):
    """
    下载论文 PDF。
    策略：
    1. 如果有 OpenAlex 提供的 OA 链接 (通常是 arXiv)，优先下载。
    2. 如果是 IEEE 的 DOI，尝试使用本地 Cookie 下载。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    title = paper_info.get("title", "untitled").replace("/", "_").replace(":", "-")
    # 截断文件名防止过长
    filename = f"{title[:50]}.pdf"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        print(f"[Info] File already exists: {filename}")
        return save_path

    # ---------------------------
    # 策略 A: 尝试 Open Access (ArXiv 等)
    # ---------------------------
    oa_url = paper_info.get("oa_url")
    if oa_url and "arxiv.org" in oa_url:
        print(f"[Download] Trying arXiv for: {title}")
        try:
            # ArXiv 需要特殊的 User-Agent，否则会拒绝
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(oa_url, headers=headers, timeout=30)
            if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"[Success] Downloaded from OA: {filename}")
                return save_path
        except Exception as e:
            print(f"[Error] OA download failed: {e}")

    # ---------------------------
    # 策略 B: 尝试 IEEE Xplore (带 Cookie)
    # ---------------------------
    doi = paper_info.get("doi")
    if doi and ("10.1109" in doi or "IEEE" in paper_info.get("venue", "").upper()):
        print(f"[Download] Trying IEEE Xplore for: {title}")
        
        # 1. 构造 IEEE 下载链接
        # OpenAlex 的 DOI 通常是 https://doi.org/10.1109/XXX.2023.1234567
        # 我们需要先访问 DOI 获取重定向后的 IEEE 真实链接，或者直接解析
        
        session = requests.Session()
        # 加载你的机构 Cookie
        if os.path.exists(ieee_cookie_path):
            session.cookies = load_cookies_from_netscape(ieee_cookie_path)
        else:
            print("[Warning] No IEEE cookie file found! Download will likely fail.")

        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Referer": "https://ieeexplore.ieee.org/"
        })

        try:
            # 第一步：访问 DOI 链接，允许重定向，让 IEEE 验证 Cookie 并跳转到文章页
            # 注意：这里我们直接构造 stamp 地址可能更直接，但先通过 DOI 跳转更稳健
            response = session.get(doi, allow_redirects=True, timeout=15)
            final_url = response.url
            
            # 从 URL 中提取 arnumber (IEEE 的文章 ID)
            # URL 可能是 https://ieeexplore.ieee.org/document/10380315
            arnumber_match = re.search(r"document/(\d+)", final_url)
            
            if arnumber_match:
                arnumber = arnumber_match.group(1)
                # 构造直接下载接口
                # 这种链接通常会触发下载，或者再次重定向到 CDN
                pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
                
                print(f"[Info] Found IEEE arnumber: {arnumber}, requesting PDF...")
                
                # 请求 PDF (再次允许重定向)
                pdf_response = session.get(pdf_url, allow_redirects=True, stream=True, timeout=60)
                
                # 检查是否真的拿到了 PDF (如果 Cookie 失效，这里会返回 HTML 网页)
                content_type = pdf_response.headers.get("Content-Type", "")
                if "application/pdf" in content_type:
                    with open(save_path, "wb") as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"[Success] Downloaded from IEEE: {filename}")
                    return save_path
                else:
                    print(f"[Failed] IEEE returned {content_type} instead of PDF. Cookies might be expired.")
                    # 可以在这里记录日志，提醒你更新 cookie.txt
            else:
                print(f"[Error] Could not extract arnumber from URL: {final_url}")

        except Exception as e:
            print(f"[Error] IEEE download failed: {e}")

    return None

    

# --- 使用示例 ---
# if __name__ == "__main__":
#     # 模拟从 search_for_papers 返回的一个结果
#     mock_paper = {
#         "title": "Deep Learning for Massive MIMO",
#         "doi": "https://doi.org/10.1109/TWC.2019.2946123", # 这是一个 IEEE 的 DOI
#         "oa_url": None, # 假设没有 OA
#         "venue": "IEEE Transactions on Wireless Communications"
#     }
    
#     # 请确保同目录下有 ieee_cookies.txt
#     download_paper_pdf(mock_paper)
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
    
# test functions for pdf download
import os
import shutil
# 假设你的下载函数保存在 downloader.py 文件中，或者直接粘贴在同一个文件里
# from downloader import download_paper_pdf 

def test_arxiv_download():
    """
    测试用例 1: ArXiv 免费下载
    目标：验证是否能自动识别 oa_url 并下载，无需 Cookie。
    """
    print("\n" + "="*40)
    print("Test Case 1: Downloading from ArXiv (Open Access)")
    print("="*40)

    # 这是一个真实的 ArXiv 论文数据结构 (模拟 OpenAlex 返回)
    # 论文: "Attention Is All You Need"
    paper_info = {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani et al.",
        "venue": "ArXiv",
        "year": 2017,
        "doi": "https://doi.org/10.48550/arXiv.1706.03762",
        # OpenAlex 通常会提供这个字段
        "oa_url": "https://arxiv.org/pdf/1706.03762.pdf", 
        "is_oa": True
    }

    # 执行下载
    save_path = download_paper_pdf(paper_info, save_dir="./test_pdfs")

    # 验证结果
    if save_path and os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"✅ [PASS] ArXiv download successful!")
        print(f"   Saved at: {save_path}")
        print(f"   File size: {file_size / 1024:.2f} KB")
        
        # 简单验证文件头是否为 PDF
        with open(save_path, 'rb') as f:
            header = f.read(4)
            if header == b'%PDF':
                print("   File header check: Valid PDF")
            else:
                print(f"   ⚠️ File header check: Invalid ({header})")
    else:
        print("❌ [FAIL] ArXiv download failed.")


def test_ieee_download():
    """
    测试用例 2: IEEE Xplore 下载
    目标：验证 Cookie 是否生效，能否穿透权限墙下载。
    注意：需要目录下存在有效的 ieee_cookies.txt
    """
    print("\n" + "="*40)
    print("Test Case 2: Downloading from IEEE Xplore (Auth Required)")
    print("="*40)

    # cookie_file = "coockies\\ieee_coockies.txt"
    cookie_file = r"D:\\ChannelCoding\\Aether\\coockies\\ieee_cookies.txt"
    if not os.path.exists(cookie_file):
        print(f"⚠️ [SKIP] {cookie_file} not found. Skipping IEEE test.")
        print("   Please export cookies using the browser extension first.")
        return

    # 这是一个真实的 IEEE 通信领域论文 (Deep Learning for Massive MIMO)
    # DOI: 10.1109/TWC.2019.2946123
    paper_info = {
        "title": "An Introduction to Deep Learning for the Physical Layer",
        "authors": "T. O'Shea et al.",
        "venue": "IEEE Transactions on Cognitive Communications and Networking",
        "year": 2017,
        "doi": "https://doi.org/10.1109/TCCN.2017.2758370",
        # 模拟没有 OA 链接的情况，强制走 IEEE 渠道
        "oa_url": None, 
        "is_oa": False
    }

    # 执行下载
    save_path = download_paper_pdf(
        paper_info, 
        save_dir="./test_pdfs", 
        ieee_cookie_path=cookie_file
    )

    # 验证结果
    if save_path and os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        
        # 关键检查：如果没有权限，IEEE 可能会返回一个 100KB 左右的 HTML 登录页
        # 真正的 PDF 通常大于 200KB
        if file_size < 150 * 1024: 
            print(f"⚠️ [WARNING] File size is suspiciously small ({file_size/1024:.2f} KB).")
            print("   It might be an HTML login page. Check your cookies.")
            
            # 检查文件头
            with open(save_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    print("❌ [FAIL] Content is NOT a PDF (likely HTML). Cookie invalid.")
                    return

        print(f"✅ [PASS] IEEE download successful!")
        print(f"   Saved at: {save_path}")
        print(f"   File size: {file_size / 1024:.2f} KB")
    else:
        print("❌ [FAIL] IEEE download failed.")

if __name__ == "__main__":
    # 清理之前的测试目录（可选）
    if os.path.exists("pdfs"):
        shutil.rmtree("pdfs")
    
    # 运行测试
    test_arxiv_download()
    test_ieee_download()
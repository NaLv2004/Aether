import os
import glob
import json
import time
import datetime
import requests
import backoff

# 从已有的 llm.py 导入 LLMAgent 及其静态方法
from llm import LLMAgent

# =====================================================================
# 提供的 search_for_papers 函数 (添加了回退所需的回调函数)
# =====================================================================
def on_backoff(details):
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
          f"calling function {details['target'].__name__} with args {details['args']} and kwargs "
          f"{details['kwargs']}")

@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff)
def search_for_papers(query, result_limit=10, engine="openalex"):
    if not query:
        return None
        
    if engine == "openalex":
        import pyalex
        from pyalex import Works
        # 可以通过环境变量设置，也可以填写真实的邮箱
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", "jiayanxu@seu.edu.cn")
        pyalex.config.email = mail

        def extract_info_from_work(work, max_abstract_length=3000):
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
            print(f"[OpenAlex Search Error] {e}")
            return None
    else:
        raise NotImplementedError(f"{engine} not supported in this script!")

# =====================================================================
# 辅助函数：读取文件
# =====================================================================
def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def read_python_codes(exp_dir):
    py_files = glob.glob(os.path.join(exp_dir, "**", "*.py"), recursive=True)
    code_content = ""
    for pf in py_files:
        code_content += f"\n--- File: {os.path.basename(pf)} ---\n"
        code_content += read_file(pf)
    return code_content

# =====================================================================
# 主 Agent 流程类
# =====================================================================
class PaperWriterSystem:
    def __init__(self, exp_dir, model="gemini-3.1-pro-high"):
        self.exp_dir = exp_dir
        self.model = model
        
        # 文件路径映射
        self.file_A_path = os.path.join(exp_dir, "idea.txt")
        self.file_B_path = os.path.join(exp_dir, "plan.txt")
        self.file_D_path = os.path.join(exp_dir, "PreviousSummary.txt")
        self.file_E_path = os.path.join(exp_dir, "execute_history.txt")
        
        # 读取内容
        self.file_A = read_file(self.file_A_path)
        self.file_B = read_file(self.file_B_path)
        self.file_C = read_python_codes(exp_dir)
        self.file_D = read_file(self.file_D_path)
        self.file_E = read_file(self.file_E_path)
        
        # 创建论文输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paper_dir = os.path.join("papers", timestamp)
        os.makedirs(self.paper_dir, exist_ok=True)
        print(f"[*] Created paper directory: {self.paper_dir}")
        
        self.reference_path = os.path.join(self.paper_dir, "reference.bib")
        self.accumulated_latex = {} # 存储已生成的章节 latex

    def do_literature_search(self, rounds=10):
        print("\n=== Starting Literature Search ===")
        agent = LLMAgent(model=self.model, temperature=0.5, log_file=os.path.join(self.paper_dir, "lit_search.log"))
        
        # 初始关键词
        current_keywords = "MIMO detection" 
        
        for r in range(rounds):
            print(f"[*] Literature Search Round {r+1}/{rounds}")
            # 获取检索结果
            raw_papers = search_for_papers(current_keywords, result_limit=25)
            if not raw_papers:
                print("[-] No papers found or search error. Skipping round.")
                continue
                
            papers_json = json.dumps(raw_papers, indent=2)
            
            sys_prompt = (
            """
                "You are an expert communications researcher conducting a literature review. "
                "I will provide you with an original Idea and a set of searched metadata from OpenAlex. "
                "Your task is to select the most relevant papers (the selected paper must be very relevant to the idea, do not select the remotely related ones), generate BibTeX formatting for them, and propose new search keywords. "
                "Return a valid JSON list. Each dictionary must have exactly these keys: "
                "'title', 'reference' (containing the full BibTeX string, including the abstract), and 'new_keywords' (USE search logics such as AND, OR to find more relevant papers, for instance, \"Communications\" OR \"Wireless\""
            """
            )
            
            msg = f"Original Idea:\n{self.file_A}\n\nSearch Results:\n{papers_json}\n"
            
            output, _ = agent.get_response_stream(msg, sys_prompt)
            json_res = LLMAgent.robust_extract_json_list(output)
            
            if json_res:
                with open(self.reference_path, "a", encoding="utf-8") as f:
                    for item in json_res:
                        bibtex = item.get("reference", "")
                        f.write(bibtex + "\n\n")
                        current_keywords = item.get("new_keywords", current_keywords) # 更新下一轮搜索词
                print(f"[+] Added {len(json_res)} references.")
            else:
                print("[-] Failed to parse BibTeX JSON in this round.")
        
        print(f"[+] Literature search complete. References saved to {self.reference_path}")

    def orchestrate_plan(self):
        print("\n=== Starting Orchestrator ===")
        agent = LLMAgent(model=self.model, temperature=0.4, log_file=os.path.join(self.paper_dir, "orchestrator.log"))
        
        sys_prompt = (
            "You are the Lead Author Orchestrator for a communications research paper. "
            "Based on the provided PreviousSummary and ExecutionHistory, generate a comprehensive paper outline. "
            "You MUST output a JSON list where each element represents a section. "
            "Each element MUST have the keys: "
            "'name' (Must be EXACTLY one of: abstract, introduction, system model, Proposed Method, Numerical Results, Conclusion), "
            "'plan' (Detailed paragraph-by-paragraph instructions for the writer agent), "
            "'figure' (Instructions for any figures needed in this section using pgfplots. If none, leave empty. "
            "For Numerical Results, mandate at least 2-3 specific pgfplot performance/complexity comparison figures)."
        )
        
        msg = f"PreviousSummary (File D):\n{self.file_D}\n\nExecute_history (Results and data):\n{self.file_E}\n"
        
        output, _ = agent.get_response_stream(msg, sys_prompt)
        plan = LLMAgent.robust_extract_json_list(output)
        
        if not plan:
            print("[-] Critical Error: Orchestrator failed to return valid JSON plan. Using fallback.")
            plan = [
                {"name": "abstract", "plan": "Summarize the paper briefly.", "figure": ""},
                {"name": "introduction", "plan": "Introduce the background and contribution.", "figure": ""},
                {"name": "system model", "plan": "Describe system model and variables.", "figure": ""},
                {"name": "Proposed Method", "plan": "Detail the algorithms.", "figure": ""},
                {"name": "Numerical Results", "plan": "Analyze data.", "figure": "Generate 2 pgfplot figures showing performance."},
                {"name": "Conclusion", "plan": "Conclude the paper.", "figure": ""}
            ]
            
        # 强制格式化名称并写入文件备份
        with open(os.path.join(self.paper_dir, "paper_plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4)
            
        print(f"[+] Orchestrator generated {len(plan)} sections.")
        return plan

    def write_section(self, section_name, section_plan, figure_plan, refine_times=1):
        print(f"\n=== Writing Section: {section_name} ===")
        agent = LLMAgent(model=self.model, temperature=0.3, log_file=os.path.join(self.paper_dir, f"{section_name.replace(' ', '_')}.log"))
        
        # 构建各个 Agent 专属的上下文
        # 基础公共设定：平实、客观的英文，无论好坏如实反映
        sys_prompt = (
            f"You are the dedicated writer for the '{section_name}' section of an IEEE TCOM paper. "
            "RULES: "
            "1. Write exclusively in plain, objective, academic English. "
            "2. Present all results (favorable or unfavorable) honestly and analytically. "
            "3. Use LaTeX formatting. Do not output markdown blocks like ```latex, just pure text or the required JSON format. "
            "4. For regular sections, output a valid JSON list with ONE dictionary: [{'filename': 'section_name.tex', 'content': 'latex code'}]. "
            "5. If you are writing 'Numerical Results', you MUST output MULTIPLE dictionaries in the JSON list: "
            "one for the main text, and others for pgfplot figures (e.g., [{'filename': 'fig1.tex', 'content': '\\begin{tikzpicture}...'}, ... , {'filename': 'numerical_results.tex', 'content': 'text using \input{fig1.tex}'}]). "
        )
        sys_prompt_specific = f"""\n
        Depending on the specific section you are assigned to write, you must strictly adhere to the following TCOM-standard guidelines:
        If you are asked to write the Abstract, you should write a highly concise summary (150–250 words) containing absolutely no citations, footnotes, or mathematical equations if possible. You must immediately state the core communication problem being addressed, briefly define the proposed system or algorithmic methodology, and explicitly highlight the most significant quantitative results (e.g., specific percentage improvements in spectral efficiency, bit error rate, or computational complexity) derived from the execute history.
        If you are asked to write the Introduction, you should construct a logical "funnel". Begin by establishing the broad motivation and practical importance of the specific wireless/communication scenario. Next, comprehensively review the provided literature, explicitly identifying the technical gaps or limitations in existing works. Follow this by clearly stating the motivation of this paper to bridge that gap. You must then provide a clear, bulleted list of the paper's explicit novel contributions. Finally, end with a standard paragraph outlining the organization of the remainder of the paper. You should cite at least 10 literatures from the reference.bib provided. You should never make up literatures on your own.
        If you are asked to write the System Model, you should rigorously and systematically define the physical communication environment, network topology, transceiver architecture, and signal models. Use standard IEEE LaTeX math formatting (e.g., bold lowercase for vectors, bold uppercase for matrices). You must explicitly state and justify all mathematical assumptions (e.g., fading channel distributions, AWGN variances, perfect/imperfect CSI). Define every mathematical variable immediately upon its first use. You should conclude this section by formally defining the overarching mathematical problem the paper aims to solve (e.g., a specific optimization formulation like sum-rate maximization or transmit power minimization).
        If you are asked to write the Proposed Method, you should provide a logical, step-by-step detailing of the algorithm, mathematical derivations, or analytical framework used to solve the problem formulated in the System Model. You must objectively justify your design choices and clearly explain the physical or mathematical rationale behind each step. You should include a rigorous theoretical analysis of the proposed method, which must include a computational complexity analysis (using Big-O notation) and, if applicable, convergence guarantees. Ensure smooth, readable transitions between inline/display equations and the explanatory text.
        If you are asked to write the Numerical Results, you should first clearly define the simulation setup, listing all key system parameters, channel conditions, and baseline schemes used for comparison. You MUST generate the LaTeX code for 2 to 3 pgfplots figures that plot performance, convergence, or complexity tradeoffs against the baselines. In the accompanying text, you must systematically reference these figures and deeply analyze the physical meaning behind the trends (e.g., why a curve saturates at high SNR). You must maintain absolute scientific objectivity: if the proposed method underperforms or exhibits unfavorable results in specific regimes, you must report this honestly and provide a scientifically rigorous explanation for why the degradation occurs.
        If you are asked to write the Conclusion, you should concisely summarize the paper’s original objectives, the proposed methodology, and the core engineering/physical insights obtained from the numerical results. You must not simply copy and paste the Abstract. Do not include any equations, cross-references to figures, or citations in this section. Conclude with one or two sentences suggesting highly specific and realistic directions for future research based on the limitations of the current work.
        """
        sys_prompt += sys_prompt_specific
        # 组装 msg 依赖
        context_msg = f"--- PLAN ---\n{section_plan}\n\n--- FIGURE REQ ---\n{figure_plan}\n\n"
        
        name_lower = section_name.lower()
        
        # 引入先前生成的 latex 代码
        prev_latex_str = ""
        for k, v in self.accumulated_latex.items():
            prev_latex_str += f"\n--- Previous Section: {k} ---\n{v}\n"
            
        context_msg += f"--- PREVIOUS SECTIONS ---\n{prev_latex_str}\n\n"
        
        # 按照题目要求的权限配置输入文件
        if "abstract" in name_lower:
            context_msg += f"Original Idea (Idea):\n{self.file_A}\nResearch Plan (Plan):\n{self.file_B}\nResults and data (Results):\n{self.file_E}\n"
        elif "introduction" in name_lower:
            bib = read_file(self.reference_path)
            context_msg += f"Original Idea:\n{self.file_A}\nResearch Plan:\n{self.file_B}\nResults and data:\n{self.file_E}\nReference.bib:\n{bib}\n"
        elif "system model" in name_lower:
            context_msg += f"Original Idea:\n{self.file_A}\nResearch Plan:\n{self.file_B}\nExperiment Code (Code):\n{self.file_C}\n"
        elif "proposed method" in name_lower:
            context_msg += f"Original Idea:\n{self.file_A}\nResearch Plan:\n{self.file_B}\nExperiment Code (Code):\n{self.file_C}\n"
        elif "numerical results" in name_lower:
            context_msg += f"Original Idea:\n{self.file_A}\nResearch Plan:\n{self.file_B}\nResults and data (Results):\n{self.file_E}\n"
        elif "conclusion" in name_lower:
            pass # 结论仅需要前面的 sections，上面已经加了
        
        # 第一轮写作
        output, _ = agent.get_response_stream(context_msg, sys_prompt)
        
        # Refine 循环
        for i in range(refine_times):
            print(f"[*] Refining {section_name} (Round {i+1}/{refine_times})...")
            refine_sys = (
                "You are an expert editor reviewing the LaTeX code for this section. "
                "Improve academic tone, ensure absolute objectivity, fix any LaTeX syntax errors (especially pgfplots), "
                "and ensure smooth transitions. Return the full updated LaTeX code in the EXACT SAME JSON list format as requested originally."
            )
            refine_msg = f"Original Output to fix:\n{output}\n\nPlease output the fully corrected JSON list."
            output, _ = agent.get_response_stream(refine_msg, refine_sys)

        # 解析输出
        json_out = LLMAgent.robust_extract_json_list(output)
        
        main_tex_content = ""
        
        if json_out:
            for item in json_out:
                fname = item.get("filename", "unknown.tex")
                # 清洗文件名，防止带路径
                fname = os.path.basename(fname)
                content = item.get("content", "")
                
                # 写入具体的文件
                out_path = os.path.join(self.paper_dir, fname)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[+] Saved: {fname}")
                
                # 将主文本加入累积历史中（如果不是图片文件）
                if "fig" not in fname.lower() and fname.endswith(".tex"):
                    main_tex_content += content
        else:
            print(f"[-] Failed to parse JSON for {section_name}. Saving raw output.")
            fname = f"{section_name.replace(' ', '_')}.tex"
            with open(os.path.join(self.paper_dir, fname), "w", encoding="utf-8") as f:
                f.write(output)
            main_tex_content = output
            
        self.accumulated_latex[section_name] = main_tex_content
        return main_tex_content

    def generate_main_tex(self, plan):
        print("\n=== Generating main.tex ===")
        main_path = os.path.join(self.paper_dir, "main.tex")
        
        # 固定的 IEEE Trans 模板头
        latex_template = r"""\documentclass[journal]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Automated Communications Research Paper}

\author{Automated LLM Agent System}

\maketitle

"""
        # 动态插入 abstract
        has_abstract = False
        for sec in plan:
            name = sec['name'].lower()
            fname = sec.get('saved_filename', f"{name.replace(' ', '_')}.tex")
            if "abstract" in name:
                latex_template += f"\\begin{{abstract}}\n\\input{{{fname}}}\n\\end{{abstract}}\n\n"
                has_abstract = True
                break
                
        # 动态插入其它章节
        for sec in plan:
            name = sec['name'].lower()
            if "abstract" in name:
                continue
            
            fname = sec.get('saved_filename', f"{name.replace(' ', '_')}.tex")
            
            if "introduction" in name:
                latex_template += f"\n\\input{{{fname}}}\n\n"
            elif "system model" in name:
                latex_template += f"\n\\input{{{fname}}}\n\n"
            elif "proposed method" in name:
                latex_template += f"\n\\input{{{fname}}}\n\n"
            elif "numerical results" in name:
                latex_template += f"\n\\input{{{fname}}}\n\n"
            elif "conclusion" in name:
                latex_template += f"\n\\input{{{fname}}}\n\n"

        # 插入参考文献
        latex_template += r"""
\bibliographystyle{IEEEtran}
\bibliography{reference}

\end{document}
"""
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(latex_template)
            
        print(f"[+] main.tex generated at {main_path}")
        print("[*] Write-up process completed successfully!")


# =====================================================================
# 启动入口
# =====================================================================
def perform_writeup(exp_dir):
    """
    暴露给外部调用的主接口
    """
    # 这里选择使用的 LLM 模型，推荐使用 o1 或者 claude-3-5-sonnet，这里用 gpt-4o 作为示例
    system = PaperWriterSystem(exp_dir=exp_dir, model="gemini-3.1-pro-high")
    
    # 1. 生成引用文献库
    system.do_literature_search(rounds=10)
    
    # 2. 统筹计划
    plan = system.orchestrate_plan()
    
    # 3. 严格按顺序生成每个 section
    for section in plan:
        sec_name = section['name']
        sec_plan = section['plan']
        fig_plan = section['figure']
        
        # 允许 refine 1 次
        system.write_section(sec_name, sec_plan, fig_plan, refine_times=1)
        
        # 记录生成的默认主文件名，便于后续 main.tex 拼接
        # （注：真实环境如果 JSON 解析出了其他名字，这里为了简化，依然根据 name 拼接）
        section['saved_filename'] = f"{sec_name.lower().replace(' ', '_')}.tex"
        
    # 4. 汇总生成 main.tex
    system.generate_main_tex(plan)

if __name__ == "__main__":
    # 测试运行（请确保当前目录下有名为 'experiment_data' 的文件夹并包含了所需的文件）
    test_dir = "experiments\\20260301_211831"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        print(f"Created dummy directory {test_dir}. Please populate it with files (idea.txt, plan.txt, etc.) before full run.")
    else:
        perform_writeup(test_dir)

import os
import json
import argparse
import logging

from llm import LLMAgent
from utils import setup_logger, PDFReader

# 导入文献搜索功能（如果环境中有提供）
try:
    from generate_ideas import format_search_results_and_update_map
except ImportError:
    pass

logger = setup_logger("review_run.log")

# ==========================================
# 1. 提示词定义 (System Prompts)
# ==========================================

# 针对 PDF 初审的 Prompt
PDF_COMMENTATOR_PROMPT = (
    "请以 IEEE TVT 的标准，对以下完全由 AI 生成的论文给出审稿意见"
)

# 针对深度全面审稿人的系统提示词
COMPREHENSIVE_REVIEWER_SYSTEM_PROMPT = """
你是一个严苛且专业的学术论文深度审稿人 (Comprehensive Reviewer)。
以下论文代码和论文的tex源文件完全由AI给出，请对其进行严苛审核。
前置的 PDF 初审（PDF Commentator）已经给出了初步的宏观审稿意见。你的任务是在它的基础上，结合项目当前工作目录下的论文源码（.tex, .bib）和实验代码（.py），给出更进一步的、细致到代码实现与论文描述是否一致的终审意见。

你可以执行的操作（Tools）包括：
1. `READ_CODE`: 申请阅读当前工作目录下的某份论文源码或代码文件的全文。建议先阅读主 tex 文件和主要的模型/环境 python 脚本。
2. `SEARCH_LITERATURE`: 如果对某项技术的新颖性存疑，或需要查找是否遗漏了重要的 Baseline，通过查找文献来确认（建议使用较短的关键词进行组合，如 "MIMO" AND "Deep Learning"）。
3. `FINISH_REVIEW`: 在完成了充分的审查后，输出最终、全面、细致的审稿意见（格式参考正规顶刊审稿意见，包含 Major Comments 和 Minor Comments），结束审查工作。

【审查重点】
1. 论文（tex文件）中描述的实验参数、方法逻辑是否与 Python 源代码完全一致？
2. 论文是否缺少必要的、前沿的对比基线（Baselines）？
3. 结合前置 PDF 初审的意见，指出具体在代码或 tex 的哪个文件中进行修改才能解决这些问题。

【交互格式】
你的回复必须严格包含以下 JSON 结构（被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "思考当前审查到了哪一步，还需要阅读什么文件，或者分析上一个工具的返回结果。",
    "Action": "READ_CODE | SEARCH_LITERATURE | FINISH_REVIEW",
    "Action_Params": {
        "filename": "如果调用 READ_CODE，在此提供需要读取的文件名（如 main.tex 或 train.py）",
        "queries": ["如果调用 SEARCH_LITERATURE，在此提供搜索关键词"],
        "review_content": "当且仅当调用 FINISH_REVIEW 时，在此填入最终详细的综合审稿意见全文"
    }
}
```

注意：
1. 每次只允许调用一个工具！如果对论文内容不清晰，务必先多次调用 `READ_CODE` 阅读具体的 tex 和 py 文件。
2. 你是一个专业的审稿人，不需要对文件进行任何修改，只需给出详细的审稿意见文本。
3. 请尽可能挖掘潜在的缺陷（包括：所提方法优势不明显，对比不公平，代码和论文讲述不匹配等）。
4.论文的代码是分阶段实现的！你必须充分阅读整篇论文和所有对应代码之后，才给出最终的评审意见！
5. 最终的review不要过于冗长。
"""

# ==========================================
# 2. 审查工具封装 (ReviewToolManager)
# ==========================================
class ReviewToolManager:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.doi_url_map = {}

    def read_code(self, filename):
        if not filename:
            return "未提供文件名。"
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path):
            return f"Error: 文件 '{filename}' 不存在于当前工作目录。"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return f"--- {filename} 内容 ---\n{content}\n"
        except Exception as e:
            return f"读取文件出错: {e}"

    def search_literature(self, queries):
        if not queries:
            return "未提供查询词。"
        try:
            # 假设环境中已有原先代码里的 format_search_results_and_update_map
            res = format_search_results_and_update_map(queries, self.doi_url_map)
            return res
        except Exception as e:
            logger.warning(f"文献搜索模块出错 (可能未导入): {e}")
            return "文献搜索不可用或发生错误。"

# ==========================================
# 3. 辅助函数
# ==========================================
def get_separated_workspace_files(cwd):
    """分开显示论文文件(.tex, .bib)和代码文件(.py)及其他"""
    if not os.path.exists(cwd):
        return "Workspace does not exist."
    files = [f for f in os.listdir(cwd) if not f.startswith('.') and f not in ['__pycache__', 'pdfs']]
    
    tex_files = [f for f in files if f.endswith('.tex')]
    bib_files = [f for f in files if f.endswith('.bib')]
    py_files = [f for f in files if f.endswith('.py')]
    other_files = [f for f in files if not f.endswith(('.tex', '.bib', '.py'))]

    state = "【当前工作目录文件列表】\n"
    state += f"- LaTeX 文件: {', '.join(tex_files) if tex_files else '无'}\n"
    state += f"- BibTeX 文件: {', '.join(bib_files) if bib_files else '无'}\n"
    state += f"- Python 源代码: {', '.join(py_files) if py_files else '无'}\n"
    state += f"- 其他文件 (PDF/TXT等): {', '.join(other_files) if other_files else '无'}\n"
    return state


# ==========================================
# 4. 主控工作流
# ==========================================
def run_review_workflow(workspace_dir, pdf_api_key, model_comprehensive, model_read_pdf):
    os.makedirs(workspace_dir, exist_ok=True)
    pdf_path = os.path.join(workspace_dir, "main.pdf")
    review_output_path = os.path.join(workspace_dir, "review.txt")
    temp_pdf_review_path = os.path.join(workspace_dir, "temp_pdf_review.txt")
    
    # ---------------------------------------------------------
    # 阶段 1: PDF Commentator 初审 (单轮)
    # ---------------------------------------------------------
    logger.info("=== 阶段 1: 启动 PDF Commentator 初审 ===")
    pdf_review_text = ""
    
    if not os.path.exists(pdf_path):
        logger.error(f"找不到 PDF 文件: {pdf_path}，跳过 PDF 初审阶段。")
        pdf_review_text = "未找到 main.pdf 文件，无法进行 PDF 初审。"
    else:
        try:
            pdf_reader = PDFReader(
                api_key=pdf_api_key,
                system_prompt="你是一个严苛的学术审稿专家。",
                context_window_size=1,
                model = model_read_pdf
            )
            logger.info("正在使用 PDFReader 解析并获取意见...")
            pdf_reader.read_pdf(
                pdf_path=pdf_path, 
                output_txt_path=temp_pdf_review_path, 
                user_prompt=PDF_COMMENTATOR_PROMPT
            )
            # 提取单轮调用的回答
            if pdf_reader.history and pdf_reader.history[-1]["role"] == "model":
                pdf_review_text = pdf_reader.history[-1]["parts"][0]["text"]
            else:
                # 兼容备份读取方式
                with open(temp_pdf_review_path, 'r', encoding='utf-8') as f:
                    pdf_review_text = f.read()
            logger.info("PDF 初审完成。")
        except Exception as e:
            logger.error(f"PDF 初审发生错误: {e}")
            pdf_review_text = f"[PDF 解析异常] {e}"

    # ---------------------------------------------------------
    # 阶段 2: Comprehensive Reviewer 深度审稿 (多轮 Tool_Call)
    # ---------------------------------------------------------
    logger.info("=== 阶段 2: 启动 Comprehensive Reviewer 深度审稿 ===")
    comprehensive_reviewer = LLMAgent(model=model_comprehensive, log_file=os.path.join(workspace_dir, "reviewer.log"))
    tool_manager = ReviewToolManager(workspace_dir)
    
    tool_calls_history = []
    MAX_ITERATIONS = 50
    final_comprehensive_review = ""
    
    for attempt in range(MAX_ITERATIONS):
        logger.info(f"--- Comprehensive Reviewer 第 {attempt + 1}/{MAX_ITERATIONS} 轮探测 ---")
        
        comprehensive_reviewer.clear_history()
        workspace_files_str = get_separated_workspace_files(workspace_dir)
        
        context_prompt = f"【来自 PDF Commentator 的初步审稿意见】\n{pdf_review_text}\n\n"
        context_prompt += f"{workspace_files_str}\n\n"
        
        context_prompt += "【你已执行的 Tool Call 历史及摘要】\n"
        if not tool_calls_history:
            context_prompt += "暂无。这是第一轮操作。\n"
        else:
            for i, th in enumerate(tool_calls_history):
                # res_trunc = th['result'] if len(str(th['result'])) < 1500 else str(th['result'])[:1500] + "...(内容过长截断)"
                res_trunc = th['result']
                context_prompt += f"[{i+1}] Action: {th['action']}, Params: {json.dumps(th.get('params', {}), ensure_ascii=False)}\nResult:\n{res_trunc}\n\n"
                
        context_prompt += "请根据上述内容，决定下一步是要 `READ_CODE` 阅读具体源码，还是 `SEARCH_LITERATURE` 查重，如果所有疑点均已理清，调用 `FINISH_REVIEW` 输出最终审稿意见全文。"
        
        context_prompt += """
        【交互格式】
你的回复必须严格包含以下 JSON 结构（被 ```json 和 ``` 包裹）：
```json
{
    "Thoughts": "思考当前审查到了哪一步，还需要阅读什么文件，或者分析上一个工具的返回结果。",
    "Action": "READ_CODE | SEARCH_LITERATURE | FINISH_REVIEW",
    "Action_Params": {
        "filename": "如果调用 READ_CODE，在此提供需要读取的文件名（如 main.tex 或 train.py）",
        "queries": ["如果调用 SEARCH_LITERATURE，在此提供搜索关键词"],
        "review_content": "当且仅当调用 FINISH_REVIEW 时，在此填入最终详细的综合审稿意见全文"
    }
}
```
        """
        

        try:
            resp, _ = comprehensive_reviewer.get_response_stream(context_prompt, COMPREHENSIVE_REVIEWER_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"[Reviewer Error] API 调用失败: {e}")
            continue
            
        action_json = LLMAgent.robust_extract_json(resp)
        if not action_json:
            if len(resp)<1000:
                tool_calls_history.append({"action": "PARSE_ERROR", "params": {}, "result": "无法解析 JSON，请严格遵守 JSON 格式输出。"})
                continue
            else:
                logger.info("[Action] 综合审稿完毕，正在生成最终 review.txt ...")
                final_comprehensive_review = resp
                break
        action = ""
        params = {}
        if action_json:      
            action = action_json.get("Action")
            params = action_json.get("Action_Params", {})
        
        if action == "READ_CODE":
            filename = params.get("filename", "")
            logger.info(f"[Action] 正在阅读文件: {filename}")
            res = tool_manager.read_code(filename)
            tool_calls_history.append({"action": action, "params": params, "result": res})
            
        elif action == "SEARCH_LITERATURE":
            queries = params.get("queries", [])
            logger.info(f"[Action] 正在搜索文献: {queries}")
            res = tool_manager.search_literature(queries)
            tool_calls_history.append({"action": action, "params": params, "result": res})
            
        elif action == "FINISH_REVIEW" or len(resp)>1000:
            logger.info("[Action] 综合审稿完毕，正在生成最终 review.txt ...")
            if action_json:
               final_comprehensive_review = params.get("review_content", "未提取到 review_content。")
            else: final_comprehensive_review = resp
            break
            
        else:
            logger.warning(f"未知 Action: {action}")
            tool_calls_history.append({"action": action, "params": params, "result": f"Unknown action: {action}"})
        
            
    if not final_comprehensive_review:
        logger.warning("Comprehensive Reviewer 达到最大迭代次数，未能正常输出 FINISH_REVIEW。将使用最后一步的 Thoughts 替代。")
        final_comprehensive_review = action_json.get("Thoughts", "Review failed to complete.")

    # ---------------------------------------------------------
    # 阶段 3: 合并结果并写入 review.txt
    # ---------------------------------------------------------
    logger.info("=== 阶段 3: 合并双阶段意见并写入 ===")
    with open(review_output_path, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("             STAGE 1: PDF 初审意见 (PDF Commentator)  \n")
        f.write("====================================================\n\n")
        f.write(pdf_review_text + "\n\n")
        f.write("====================================================\n")
        f.write("      STAGE 2: 深度代码与文本审查 (Comprehensive Reviewer)\n")
        f.write("====================================================\n\n")
        f.write(final_comprehensive_review + "\n")
        
    logger.info(f"审稿结束！完整意见已保存至: {review_output_path}")
    
    # 清理临时文件
    if os.path.exists(temp_pdf_review_path):
        os.remove(temp_pdf_review_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Auto Paper Review Agent")
#     parser.add_argument("--workspace_dir", type=str, default="./", help="目标科研项目的工作目录")
#     parser.add_argument("--comprehensive_model", type=str, default="gpt-4o", help="Comprehensive Reviewer 使用的模型")
#     parser.add_argument("--pdf_api_key", type=str, default=os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE"), help="PDFReader 需要的 API 密钥")
#     args = parser.parse_args()

#     run_review_workflow(
#         workspace_dir=args.workspace_dir,
#         pdf_api_key=args.pdf_api_key,
#         model_comprehensive=args.comprehensive_model
#     )


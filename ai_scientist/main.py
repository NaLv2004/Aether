import argparse
import datetime
import os
from generate_ideas import generate_ideas
from generate_plan import generate_plan
from generate_code import generate_code
from perform_experiments import generate_readme, plan_and_execute_experiments
from perform_writeup import perform_writeup
from update_from_reviews import update_from_review
from utils import setup_logger
from utils import PDFReader
from review import run_review_workflow
import shutil
from utils import remove_file
from utils import move_files
from utils import compile_latex_project

logger = setup_logger("experiment_run.log")

def main():
    # =========================================Parameter Configurations==============================================
    # Parameters for idea generation
    # MODEL = 'claude-opus-4-6'
    MODEL = 'gemini-3.1-pro-preview'
    # MODEL = 'claude-opus-4-6'
    THEME_FILE_PATH = 'theme_idea_gen.txt'
    N_PARALLEL_IDEA_GENERATOR = 3
    MAX_STUDENT_ITERS = 2
    N_PARALLEL_TEACHER_CHECKER = 3
    MAX_TEACHER_ITERS = 2
    # parameters for plan generation
    MAX_PLAN_REVIEW_ITER = 3
    MAX_STUDENT_SELF_REFINE_ITER = 4
    N_PARALLEL_PLAN_GENERATOR = 3
    N_GENERATOR_PER_IDEA = 2
    REPO_URL = "https://github.com/NaLv2004/rebuttal.git"
    MAX_REBUTTAL_TURNS = 10
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = "20260308_025855"
    LOG_ROOT_DIR = 'logs'
    LOG_DIR = os.path.join(LOG_ROOT_DIR, timestamp)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_DIR_SUB = ['idea_gen','plan_gen','code_gen','perform_experiments','perform_writeup','rebuttal','review']
    for sub_dir in LOG_DIR_SUB:
        os.makedirs(os.path.join(LOG_DIR, sub_dir), exist_ok=True)
    LOG_PATH_SUB = dict()
    for sub_dir in LOG_DIR_SUB:
        LOG_PATH_SUB[sub_dir] = os.path.join(LOG_DIR, sub_dir)
    OUTPUT_DIR = os.path.join("products",timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH_SUB = dict()
    for sub_dir in LOG_DIR_SUB:
        OUTPUT_PATH_SUB[sub_dir] = os.path.join(OUTPUT_DIR, f'{sub_dir}')
        os.makedirs(OUTPUT_PATH_SUB[sub_dir], exist_ok=True)
        
    parser_idea_gen = argparse.ArgumentParser(description="通信领域 AI Scientist - Idea 生成与审查")
    parser_idea_gen.add_argument("--theme_file", type=str, default=THEME_FILE_PATH, help="存放研究主题的txt文件")
    parser_idea_gen.add_argument("--n_students", type=int, default=N_PARALLEL_IDEA_GENERATOR, help="并发运行的Idea Generator Agent数量")
    parser_idea_gen.add_argument("--n_teachers", type=int, default=N_PARALLEL_TEACHER_CHECKER, help="并发运行的Novelty Check Agent数量")
    parser_idea_gen.add_argument("--max_student_iters", type=int, default=MAX_STUDENT_ITERS, help="Student最大迭代次数")
    parser_idea_gen.add_argument("--max_teacher_iters", type=int, default=MAX_TEACHER_ITERS, help="Teacher最大检索评估次数")
    parser_idea_gen.add_argument("--model", type=str, default=MODEL, help="使用的LLM模型名称")
    parser_idea_gen.add_argument("--output_file", type=str, default=os.path.join(OUTPUT_PATH_SUB['idea_gen'], "all_generated_ideas.txt"), help="所有生成的Idea保存位置")
    parser_idea_gen.add_argument("--output_dir", type=str, default=OUTPUT_PATH_SUB['idea_gen'], help="输出文件夹中")
    parser_idea_gen.add_argument("--review_log", type=str, default=os.path.join(LOG_PATH_SUB['idea_gen'], "review.log"), help="审查结果的输出位置")
    parser_idea_gen.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['idea_gen'], help="审查结果的输出位置")
    
    parser_plan_gen = argparse.ArgumentParser(description="AI Scientist - Research Planner")
    # 将输出文件修改为输出文件夹
    parser_plan_gen.add_argument("--output_dir", type=str, default=OUTPUT_PATH_SUB['plan_gen'], help="输出完整Plan的汇总文件夹路径")
    parser_plan_gen.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['plan_gen'], help="存放Log文件的文件夹路径")
    parser_plan_gen.add_argument("--max_iters", type=int, default=MAX_PLAN_REVIEW_ITER, help="每个Idea的师生最大讨论轮数")
    parser_plan_gen.add_argument("--max_inner_iters", type=int, default=MAX_STUDENT_SELF_REFINE_ITER, help="每个Idea的师生最大讨论轮数")
    parser_plan_gen.add_argument("--model_student", type=str, default= MODEL, help="Student Planner使用的模型")
    parser_plan_gen.add_argument("--model_teacher", type=str, default= MODEL, help="Teacher Planner使用的模型")
    parser_plan_gen.add_argument("--max_workers", type=int, default=N_PARALLEL_PLAN_GENERATOR, help="并发处理的总线程数")
    # 新增参数：每个 idea 分配的 agent 数量
    parser_plan_gen.add_argument("--k_agents", type=int, default=N_GENERATOR_PER_IDEA, help="每个Idea分配几对并行的Student和Teacher")
    parser_plan_gen.add_argument("--interactive", type=bool, default=True, help="是否启用交互模式")
    
    parser_code_gen = argparse.ArgumentParser(description="AI Scientist - Experiment Performer")
    parser_code_gen.add_argument("--orchestrator", type=str, default=MODEL, help="Orchestrator 使用的模型")
    parser_code_gen.add_argument("--coder", type=str, default=MODEL, help="Coder 使用的模型")
    parser_code_gen.add_argument("--experiment_log_dir", type=str, default=LOG_PATH_SUB['code_gen'], help="实验log的输出目录")
    parser_code_gen.add_argument("--experiment_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="实验log的输出目录")
    parser_code_gen.add_argument("--include_all_files", type=bool, default=True, help="Orchestrator的context中是否包含所有文件")
    parser_code_gen.add_argument("--repo_url", type=str, default=REPO_URL, help="Orchestrator的context中是否包含所有文件")
    
    parser_readme_gen = argparse.ArgumentParser(description="根据代码依赖关系流式生成全面的科研项目 README")
    parser_readme_gen.add_argument("--work_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="Python文件所在的工作目录")
    parser_readme_gen.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['code_gen'], help="Python文件所在的工作目录")
    parser_readme_gen.add_argument("--overview_file", type=str,default=os.path.join(OUTPUT_PATH_SUB['code_gen'], "experiment_summary.txt"), help="概述文件(txt/md)的绝对或相对路径")
    parser_readme_gen.add_argument("--model", type=str, default=MODEL, help="要使用的LLM模型名称 (默认: gemini-3.1-pro-preview)")
    
    
    parser_experiment = argparse.ArgumentParser(description="计划并执行详细实验")
    parser_experiment.add_argument("--workspace_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="实验工作目录")
    parser_experiment.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['perform_experiments'], help="实验log的输出目录")
    parser_experiment.add_argument("--model", type=str, default=MODEL, help="使用的模型")
    parser_experiment.add_argument("--conda_env_name", type=str, default="AutoGenOld", help="使用的Conda环境名称")
    
    parser_review_update = argparse.ArgumentParser(description="AI Scientist - Update review")
    parser_review_update.add_argument("--orchestrator", type=str, default=MODEL, help="Orchestrator 使用的模型")
    parser_review_update.add_argument("--coder", type=str, default=MODEL, help="Coder 使用的模型")
    parser_review_update.add_argument("--experiment_log_dir", type=str, default=LOG_PATH_SUB['rebuttal'], help="实验log的输出目录")
    parser_review_update.add_argument("--experiment_dir", type=str, default=OUTPUT_PATH_SUB['rebuttal'], help="实验log的输出目录")
    parser_review_update.add_argument("--include_all_files", type=bool, default=False, help="Orchestrator的context中是否包含所有文件")
    parser_review_update.add_argument("--repo_url", type=str, default=REPO_URL, help="Orchestrator的context中是否包含所有文件")
    
        
    logger.info(f"Starting Research. Experiment Run Log Dir: {LOG_DIR}")
    # ========================================Research Starts=========================================================
    # arguments for idea generator 
    
    # run idea generator
    
    logger.info(f"Starting Idea Generation...")
    # output_ideas_path = generate_ideas(parser_idea_gen.parse_args(),interactive=True)
    output_ideas_path = r"products\\20260305_195128\\idea_gen\\refined_idea_4_1772712730.json"
    logger.info(f"Idea Generation Finished. Output File: {output_ideas_path}")
    # run plan generator
    logger.info(f"Starting Plan Generation...")
    parser_plan_gen.add_argument("--input_file", type=str, default=output_ideas_path, help="包含Idea的JSON文件路径")
    # plan_file_path = generate_plan(parser_plan_gen.parse_args())
    plan_file_path = r"products\\20260306_120355\\plan_gen\\initial_plans.json"
    logger.info(f"Plan Generation Finished. Output File: {plan_file_path}")
    # run code generator
    logger.info(f"Starting Code Generation...")
    parser_code_gen.add_argument("--plan_file", type=str, default=plan_file_path, help="之前生成的包含计划的JSON文件路径")
    # generate_code(parser_code_gen.parse_args())
    logger.info(f"Code Generation Finished.")
    # generate readme
    # run readme generator
    logger.info(f"Starting README Generation...")
    parser_readme_gen.add_argument("--plan_file", type=str, default=plan_file_path, help="科研计划文件(txt/md)的绝对或相对路径")
    # generate_readme(parser_readme_gen.parse_args())
    logger.info(f"README Generation Finished.")
    
    # generate and execute detailed experiment plans
    # run experiment
    logger.info(f"Starting Experiment Execution...")
    # plan_and_execute_experiments(parser_experiment.parse_args())
    logger.info(f"Experiment Execution Finished.")
    logger.info(f"Starting Writeup Generation...")
    parser_writeup = argparse.ArgumentParser(description="生成科研文章")
   
    
    logger.info(f"Writeup Generation Finished.")
    # perform_writeup(exp_dir=OUTPUT_PATH_SUB['code_gen'],paper_dir=OUTPUT_PATH_SUB['perform_writeup'],model=MODEL,idea_path=output_ideas_path,plan_path=plan_file_path)
    
    papers_path =  r"papers\\20260308_194954"
    logger.info(f"Starting Rebuttal Generation...")
    # copy all files in paper path to OUTPUT_PATH_SUB['rebuttal']
    # remove_file(OUTPUT_PATH_SUB['rebuttal'])
    # move_files(papers_path, OUTPUT_PATH_SUB['rebuttal'])
    # move_files(OUTPUT_PATH_SUB['code_gen'],OUTPUT_PATH_SUB['rebuttal'])
    parser_review_update.add_argument("--plan_file", type=str, default=plan_file_path, help="之前生成的包含计划的JSON文件路径")

    
    for i in range(MAX_REBUTTAL_TURNS):
        logger.info(f"Starting Rebuttal Round {i+1}...")
        # clear all files in rebuttal path
        remove_file(OUTPUT_PATH_SUB['review'])
        pdf_path = OUTPUT_PATH_SUB['rebuttal']
        try:
           compile_latex_project(pdf_path, "main.tex")    
           logger.info(f"pdf compiled generated successfully")     
        except Exception as e:
           logger.error(f"Failed to compile pdf: {e}")
        if i>=1:
            logger.info(f"Starting Rebuttal Review {i+1}/MAX_REBUTTAL_TURNS...")
            move_files(OUTPUT_PATH_SUB['rebuttal'], OUTPUT_PATH_SUB['review'])
            run_review_workflow(workspace_dir=OUTPUT_PATH_SUB['review'], pdf_api_key=os.environ['JIANYI_API_KEY'],model_comprehensive='claude-opus-4-6',model_read_pdf='gemini-3.1-pro-preview' )
            logger.info(f"Rebuttal Review {i+1}/MAX_REBUTTAL_TURNS Finished.")
            logger.info(f"Starting Rebuttal Update {i+1}/MAX_REBUTTAL_TURNS...")
            shutil.copy2(os.path.join(OUTPUT_PATH_SUB['review'], "review.txt"), os.path.join(OUTPUT_PATH_SUB['rebuttal'], "review.txt"))
        update_from_review(parser_review_update.parse_args())
        logger.info(f"Rebuttal Update {i+1}/MAX_REBUTTAL_TURNS Finished.")
        pass

main()



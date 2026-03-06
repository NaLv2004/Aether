import argparse
import datetime
import os
from generate_ideas import generate_ideas
from generate_plan import generate_plan
from generate_code import generate_code
from perform_experiments import generate_readme, plan_and_execute_experiments
from perform_writeup import perform_writeup
from utils import setup_logger

logger = setup_logger("experiment_run.log")

def main():
    # =========================================Parameter Configurations==============================================
    # Parameters for idea generation
    MODEL = 'gemini-3-flash-preview'
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_ROOT_DIR = 'logs'
    LOG_DIR = os.path.join(LOG_ROOT_DIR, timestamp)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_DIR_SUB = ['idea_gen','plan_gen','code_gen','perform_experiments','perform_writeup']
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
        
    logger.info(f"Starting Research. Experiment Run Log Dir: {LOG_DIR}")
    # ========================================Research Starts=========================================================
    # arguments for idea generator 
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
    # run idea generator
    
    logger.info(f"Starting Idea Generation...")
    # output_ideas_path = generate_ideas(parser_idea_gen.parse_args(),interactive=True)
    output_ideas_path = r"products\\20260305_195128\\idea_gen\\refined_idea_4_1772712730.json"
    logger.info(f"Idea Generation Finished. Output File: {output_ideas_path}")
    # to do : let the user select one of the ideas.
    # parser.add_argument()
    # generate research plan
    parser_plan_gen = argparse.ArgumentParser(description="AI Scientist - Research Planner")
    parser_plan_gen.add_argument("--input_file", type=str, default=output_ideas_path, help="包含Idea的JSON文件路径")
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
    # run plan generator
    logger.info(f"Starting Plan Generation...")
    plan_file_path = generate_plan(parser_plan_gen.parse_args())
    logger.info(f"Plan Generation Finished. Output File: {plan_file_path}")
    
    parser_code_gen = argparse.ArgumentParser(description="AI Scientist - Experiment Performer")
    parser_code_gen.add_argument("--plan_file", type=str, default=plan_file_path, help="之前生成的包含计划的JSON文件路径")
    parser_code_gen.add_argument("--orchestrator", type=str, default=MODEL, help="Orchestrator 使用的模型")
    parser_code_gen.add_argument("--coder", type=str, default=MODEL, help="Coder 使用的模型")
    parser_code_gen.add_argument("--experiment_log_dir", type=str, default=LOG_PATH_SUB['code_gen'], help="实验log的输出目录")
    parser_code_gen.add_argument("--experiment_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="实验log的输出目录")
    
    
    # run code generator
    logger.info(f"Starting Code Generation...")
    generate_code(parser_code_gen.parse_args())
    logger.info(f"Code Generation Finished.")
    # generate readme
    parser_readme_gen = argparse.ArgumentParser(description="根据代码依赖关系流式生成全面的科研项目 README")
    parser_readme_gen.add_argument("--work_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="Python文件所在的工作目录")
    parser_readme_gen.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['code_gen'], help="Python文件所在的工作目录")
    parser_readme_gen.add_argument("--plan_file", type=str, default=plan_file_path, help="科研计划文件(txt/md)的绝对或相对路径")
    parser_readme_gen.add_argument("--overview_file", type=str,default=os.path.join(OUTPUT_PATH_SUB['code_gen'], "experiment_summary.txt"), help="概述文件(txt/md)的绝对或相对路径")
    parser_readme_gen.add_argument("--model", type=str, default=MODEL, help="要使用的LLM模型名称 (默认: gemini-3.1-pro-preview)")
    
    # run readme generator
    logger.info(f"Starting README Generation...")
    generate_readme(parser_readme_gen.parse_args())
    logger.info(f"README Generation Finished.")
    
    # generate and execute detailed experiment plans
    parser_experiment = argparse.ArgumentParser(description="计划并执行详细实验")
    parser_experiment.add_argument("--workspace_dir", type=str, default=OUTPUT_PATH_SUB['code_gen'], help="实验工作目录")
    parser_experiment.add_argument("--log_dir", type=str, default=LOG_PATH_SUB['perform_experiments'], help="实验log的输出目录")
    parser_experiment.add_argument("--model", type=str, default=MODEL, help="使用的模型")
    
    # run experiment
    logger.info(f"Starting Experiment Execution...")
    plan_and_execute_experiments(parser_experiment.parse_args())
    logger.info(f"Experiment Execution Finished.")
    # to do: create plan.txt, idea.txt, etc.
    # compose article
    logger.info(f"Starting Writeup Generation...")
    parser_writeup = argparse.ArgumentParser(description="生成科研文章")
    logger.info(f"Writeup Generation Finished.")
    perform_writeup(OUTPUT_PATH_SUB['code_gen'])

main()
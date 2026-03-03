import argparse
import datetime
import os



def main():
    # Parameters for idea generation
    MODEL = 'gemini-3.1-pro-high'
    THEME_FILE_PATH = 'theme_idea_gen.txt'
    N_PARALLEL_IDEA_GENERATOR = 9
    MAX_STUDENT_ITERS = 3
    N_PARALLEL_TEACHER_CHECKER = 5
    MAX_TEACHER_ITERS = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_ROOT_DIR = 'logs'
    LOG_DIR = os.path.join(LOG_ROOT_DIR, timestamp)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_DIR_SUB = ['idea_gen','plan_gen','code_gen','perform_experiments','perform_writeup']
    for sub_dir in LOG_DIR_SUB:
        os.makedirs(os.path.join(LOG_DIR, sub_dir), exist_ok=True)
    LOG_PATH_SUB = dict()
    for sub_dir in LOG_DIR_SUB:
        LOG_PATH_SUB[sub_dir] = os.path.join(LOG_DIR, sub_dir, f'{sub_dir}.log')
        
    parser = argparse.ArgumentParser(description="通信领域 AI Scientist - Idea 生成与审查")
    parser.add_argument("--theme_file", type=str, default="THEME_FILE_PATH", help="存放研究主题的txt文件")
    parser.add_argument("--n_students", type=int, default=N_PARALLEL_IDEA_GENERATOR, help="并发运行的Idea Generator Agent数量")
    parser.add_argument("--n_teachers", type=int, default=N_PARALLEL_TEACHER_CHECKER, help="并发运行的Novelty Check Agent数量")
    parser.add_argument("--max_student_iters", type=int, default=MAX_STUDENT_ITERS, help="Student最大迭代次数")
    parser.add_argument("--max_teacher_iters", type=int, default=MAX_TEACHER_ITERS, help="Teacher最大检索评估次数")
    parser.add_argument("--model", type=str, default="MODEL", help="使用的LLM模型名称")
    parser.add_argument("--output_file", type=str, default="all_generated_ideas.txt", help="所有生成的Idea保存位置")
    parser.add_argument("--review_log", type=str, default="novelty_scores.log", help="审查结果的输出位置")
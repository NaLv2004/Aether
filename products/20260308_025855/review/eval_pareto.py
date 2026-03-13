import subprocess
import re
import sys
import os
import argparse

# 全局缓存字典，用于避免重复运行相同的实验参数
experiment_cache = {}

def run_experiment(c_avg, lambda_val, args):
    """
    运行一次 joint_qat.py 并在输出中提取关键性能指标。
    如果该组合已经运行过，则直接返回缓存的结果。
    """
    key = (c_avg, lambda_val)
    if key in experiment_cache:
        print(f"\n>>> Using Cached Result for C_avg={c_avg}, Lambda={lambda_val} ...")
        return experiment_cache[key]
        
    print(f"\n>>> Running Experiment: C_avg={c_avg}, Lambda={lambda_val} ...")
    cmd = [
        sys.executable, "joint_qat.py",
        "--epochs", str(args.epochs),
        "--batches_per_epoch", str(args.batches_per_epoch),
        "--batch_size", str(args.batch_size),
        "--c_avg", str(c_avg),
        "--lambda_val", str(lambda_val)
    ]
    
    # 执行子进程并实时读取输出以供监控，同时保存完整输出用于解析
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    full_output = []
    for line in process.stdout:
        print(line, end="")
        full_output.append(line)
    process.wait()
    
    output_str = "".join(full_output)
    
    # 使用正则表达式提取 p_tx (dBm) = 10 时的 BER 和 Actual Average Bits
    # 匹配行示例: 10           | 1.3484%         | 0.8407%         | 37.65 % | 2.50
    pattern = re.compile(r"10\s+\|\s+[\d\.]+%?\s+\|\s+([\d\.]+%?)\s+\|\s+[\d\.]+\s*%\s+\|\s+([\d\.]+)")
    match = pattern.search(output_str)
    
    if match:
        qat_ber = match.group(1)
        actual_bits = match.group(2)
        res = (qat_ber, actual_bits)
    else:
        res = ("N/A", "N/A")
        
    experiment_cache[key] = res
    return res

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pareto Frontier for QAT-GNN.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per experiment.")
    parser.add_argument("--batches_per_epoch", type=int, default=30, help="Number of batches per epoch.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    args = parser.parse_args()

    # 实验 1: 固定 Lambda 0.1，改变 Target C_avg
    print("="*80)
    print("EXPERIMENT 1: C_avg Sensitivity (Pareto Frontier Analysis)")
    print("Fixed Lambda = 0.1")
    print("="*80)
    
    # 精简搜索空间
    c_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
    results1 = []
    for c in c_vals:
        ber, bits = run_experiment(c, 0.1, args)
        results1.append({
            "target_c": c,
            "lambda": 0.1,
            "actual_bits": bits,
            "ber_10dbm": ber
        })
        
    # 实验 2: 固定 C_avg 2.5，改变 Lambda
    print("\n\n" + "="*80)
    print("EXPERIMENT 2: Lambda Sensitivity Analysis")
    print("Fixed Target C_avg = 2.5")
    print("="*80)
    
    # 精简搜索空间
    l_vals = [0.01, 0.1, 0.5, 1.0]
    results2 = []
    for l in l_vals:
        ber, bits = run_experiment(2.5, l, args)
        results2.append({
            "target_c": 2.5,
            "lambda": l,
            "actual_bits": bits,
            "ber_10dbm": ber
        })
        
    # 最终结果展示
    print("\n" + "#"*80)
    print("FINAL SUMMARY TABLES")
    print("#"*80)
    
    print("\nTable 1: Experiment 1 (C_avg Sensitivity / Pareto Frontier, Lambda=0.1)")
    print(f"{'Target C_avg':<15} | {'Lambda':<10} | {'Actual Avg Bits':<20} | {'QAT-GNN BER (10dBm)':<20}")
    print("-" * 75)
    for r in results1:
        print(f"{r['target_c']:<15.1f} | {r['lambda']:<10.2f} | {r['actual_bits']:<20} | {r['ber_10dbm']:<20}")
        
    print("\nTable 2: Experiment 2 (Lambda Sensitivity, Target C_avg=2.5)")
    print(f"{'Target C_avg':<15} | {'Lambda':<10} | {'Actual Avg Bits':<20} | {'QAT-GNN BER (10dBm)':<20}")
    print("-" * 75)
    for r in results2:
        print(f"{r['target_c']:<15.1f} | {r['lambda']:<10.2f} | {r['actual_bits']:<20} | {r['ber_10dbm']:<20}")

if __name__ == "__main__":
    main()
import subprocess
import os

def parse_ieee_paper_with_marker(pdf_path, output_dir):
    """
    使用 Marker 解析 IEEE 双栏论文
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Marker 的单文件解析命令
    # --force_ocr 针对部分字体嵌合奇怪的IEEE早期论文很有用，如果是新论文可省略
    command = [
        "marker_single",
        pdf_path,
        output_dir,
        "--batch_multiplier", "2" # 如果显存够可以调大，加快速度
    ]
    
    print(f"🚀 开始解析 IEEE 论文: {pdf_path}")
    print("⏳ 包含双栏分析和公式识别，请耐心等待...")
    
    try:
        # 捕获输出以显示进度
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ 转换成功！结果已保存在: {output_dir}")
        else:
            print(f"\n❌ 转换失败，错误信息:\n{process.stderr.read()}")
            
    except Exception as e:
        print(f"发生异常: {e}")

# 使用示例
# 解析后会在 ieee_output 目录下生成一个与原 PDF 同名的文件夹，内含 .md 和 提取的图片
parse_ieee_paper_with_marker("pdfs_ieee\\10.1109_TCCN.2017.2758370.pdf", "ieee_output")
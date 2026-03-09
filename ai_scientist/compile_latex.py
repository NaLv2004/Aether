import os
import subprocess

def compile_latex_project(main_filename="main.tex"):
    """
    使用 subprocess 调用 TeX Live 编译 LaTeX 项目。
    支持参考文献 (pdflatex -> bibtex -> pdflatex -> pdflatex)
    """
    if not os.path.exists(main_filename):
        print(f"错误: 找不到主文件 {main_filename}！")
        return False

    # 去除扩展名，用于 bibtex 命令 (例如 'main')
    base_name = os.path.splitext(main_filename)[0]

    # 定义编译步骤
    # 使用 -interaction=nonstopmode 防止编译时遇到小错误卡住等待用户输入
    steps = [
        ["pdflatex", "-interaction=nonstopmode", main_filename],
        ["bibtex", base_name],
        ["pdflatex", "-interaction=nonstopmode", main_filename],
        ["pdflatex", "-interaction=nonstopmode", main_filename]
    ]

    print(f"开始编译 {main_filename} ...\n")

    for i, cmd in enumerate(steps, 1):
        print(f"--- 正在执行步骤 {i}/{len(steps)}: {' '.join(cmd)} ---")
        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                encoding='utf-8',
                errors='ignore' # 防止非utf-8字符导致报错
            )
            
            # pdflatex/bibtex 遇到严重错误时通常返回非 0 状态码
            if result.returncode != 0:
                print(f"❌ 步骤 '{' '.join(cmd)}' 执行失败！")
                print("以下是错误日志最后30行:")
                # 只打印最后几行日志避免刷屏
                lines = result.stdout.strip().split('\n')
                print('\n'.join(lines[-30:]))
                return False
                
        except FileNotFoundError:
            print(f"❌ 找不到命令 '{cmd[0]}'。请确保已经安装 TeX Live 并将其添加到了系统环境变量 (PATH) 中。")
            return False

    # 检查最终的 PDF 文件是否生成
    pdf_file = f"{base_name}.pdf"
    if os.path.exists(pdf_file):
        print(f"\n✅ 编译成功！已生成 PDF 文件: {os.path.abspath(pdf_file)}")
        return True
    else:
        print("\n❌ 编译流程结束，但未找到预期的 PDF 文件。")
        return False


def generate_test_files():
    """
    生成用于测试的 main.tex 和 refs.bib 文件 (IEEEtran 模板)
    """
    print("正在生成测试用的 IEEEtran LaTeX 文件...")
    
    tex_content = r"""\documentclass[journal]{IEEEtran}
\usepackage[utf8]{inputenc}

\begin{document}

\title{Test Document for Python Subprocess}
\author{Your Name}

\maketitle

\begin{abstract}
This is a minimal working example to test compiling an IEEEtran document with citations using Python.
\end{abstract}

\section{Introduction}
Here is a citation to Claude Shannon's classic paper \cite{shannon1948}. 
We are using Python's \texttt{subprocess} module to compile this.

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
"""

    bib_content = """@article{shannon1948,
  title={A mathematical theory of communication},
  author={Shannon, Claude Elwood},
  journal={The Bell system technical journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948},
  publisher={Nokia Bell Labs}
}
"""

    with open("main.tex", "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    with open("refs.bib", "w", encoding="utf-8") as f:
        f.write(bib_content)
        
    print("测试文件 (main.tex, refs.bib) 生成完毕。\n")


if __name__ == "__main__":
    # 1. 生成测试文件
    generate_test_files()
    
    # 2. 调用函数进行编译
    compile_latex_project("main.tex")
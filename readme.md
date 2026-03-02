这是一份为您量身定制的 `Aether` 项目中 `IdeaGenerator` 模块的 README 文档。文档采用专业的技术语言编写，详细拆解了代码的实现逻辑，并配有易于理解的 Mermaid 流程图。

---

# 🧠 Aether: IdeaGenerator 模块详解

## 📖 模块简介
**Aether** 是一个专注于通信领域的虚拟 AI 科学家。`IdeaGenerator` 是 Aether 的“大脑”与“灵感引擎”，负责从零开始构思具有高度创新性、跨学科且具备落地可行性的科研 Idea，并对其进行严苛的学术审查。

该模块采用 **“学生-老师” (Student-Teacher) 双智能体架构**：
1. **Student Agent (Idea Generator)**：扮演极具创造力的 AI 科学家，负责提出假设、检索文献、迭代打磨科研 Idea。
2. **Teacher Agent (Novelty Checker)**：扮演顶会（如 GLOBECOM, INFOCOM）的资深审稿人 (Area Chair)，负责通过检索前沿文献对生成的 Idea 进行查重、评估与打分。

---

## ⚙️ 核心工作流程图

以下是 `IdeaGenerator` 模块的完整执行逻辑流程图：

```mermaid
graph TD
    Start(["启动主程序 main"]) --> ReadTheme["读取研究主题 txt"]
    
    subgraph Phase1 ["阶段一：Idea 生成与打磨 (Student Agents)"]
        InitStudents["并发启动 N 个 Student Agent"]
        StudentLoop{"迭代满或满意?"}
        Generate["生成/优化 Idea & 提取 Queries"]
        OpenAlex1["调用 OpenAlex API 检索文献"]
        Feedback1["将文献摘要反馈给 LLM"]
        OutputIdeas["保存该 Agent 的 Ideas"]
        
        ReadTheme --> InitStudents
        InitStudents --> StudentLoop
        StudentLoop -- "未结束" --> Generate
        Generate --> OpenAlex1
        OpenAlex1 --> Feedback1
        Feedback1 --> StudentLoop
        StudentLoop -- "I'm done / 达上限" --> OutputIdeas
    end

    MergeIdeas["合并并打乱所有候选 Ideas"]
    OutputIdeas --> MergeIdeas
    
    subgraph Phase2 ["阶段二：新颖性审查与打分 (Teacher Agents)"]
        InitTeachers["并发启动 M 个 Teacher Agent"]
        TeacherLoop{"Decision == Finished?"}
        Review["审阅 Idea & 提出查重 Queries"]
        OpenAlex2["调用 OpenAlex API 查重"]
        Feedback2["将检索结果反馈给审稿 LLM"]
        MakeScore["给出评审意见及 Score (1-10)"]
        
        MergeIdeas --> InitTeachers
        InitTeachers --> TeacherLoop
        TeacherLoop -- "Pending" --> Review
        Review --> OpenAlex2
        OpenAlex2 --> Feedback2
        Feedback2 --> TeacherLoop
        TeacherLoop -- "Finished" --> MakeScore
    end
    
    SaveResult(["输出结果至 Log 与 JSON"])
    MakeScore --> SaveResult

    %% 样式定义
    classDef student fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef teacher fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#000;
    class InitStudents,StudentLoop,Generate,OpenAlex1,Feedback1,OutputIdeas student;
    class InitTeachers,TeacherLoop,Review,OpenAlex2,Feedback2,MakeScore teacher;
```

---

## 🛠️ 详细实现解析

### 1. 外部知识库接入 (OpenAlex API)
为了确保 AI 生成的 Idea 具有时代前沿性且不与现有研究冲突，模块实现了 `search_for_papers` 函数，深度集成了学术数据库 OpenAlex：
* **摘要解析**：智能解析 OpenAlex 返回的 `abstract_inverted_index`（倒排索引），将其还原为可读的摘要文本。
* **容错与重试机制**：使用 `@backoff` 装饰器，在遭遇 HTTP 请求限制或网络波动时，采用指数退避（Exponential Backoff）策略自动重试，保证长期运行的稳定性。
* **数据瘦身**：提取标题、作者、发表平台、年份、引用量和摘要，截断过长摘要，防止 LLM 上下文超载。

### 2. Phase 1: 学生智能体 (Idea Generator)
* **角色设定**：富有雄心的通信领域学者。被限制不能提出“天马行空、无法落地”（如强行结合量子计算、脑机接口）的伪需求，必须聚焦具体通信场景（信道衰落、硬件损伤等）。
* **迭代机制 (Self-Refinement)**：
  * **第一轮**：接收粗略主题 (`theme_idea_gen.txt`)，进行初步思考并输出文献搜索词。
  * **后续轮次**：吸收上一轮的文献搜索结果，修改已被前人研究过的设想，提出新 Idea。
  * **提前终止**：当模型在 `Thoughts` 字段中输出 `"I'm done"` 时，认为 Idea 已足够完善，主动结束迭代，节省 Token。
* **并发执行**：通过 `ThreadPoolExecutor` 并发启动多个 Student，极大地提高了灵感发散的广度和生成效率。

### 3. Phase 2: 老师智能体 (Novelty Check)
* **角色设定**：极其严格的顶级学术会议 Area Chair。被告知 Idea 由 AI 生成，要求其必须秉持怀疑态度进行审查。
* **动态多轮查重**：
  * Teacher 不会被强制要求一次性给出评分。它可以通过输出 `"Decision": "Pending"` 并附带 `SearchQueries`，不断要求系统去 OpenAlex 查阅文献。
  * 只有当收集了足够证据，确认 Idea 的创新性和仿真可行性后，才会输出 `"Decision": "Finished"`，并给出 `Score` (1-10分) 和详细评价。

### 4. 稳健的 JSON 结构化输出
为保证代码解析的稳定性，通过提示词（Prompt）严格要求 LLM 输出特定格式的 JSON，配合 `LLMAgent.extract_json_between_markers` 实现精准的数据提取。
Student 提取结构：`Thoughts` -> `SearchQueries` -> `Ideas[Name, Title, Background, Hypothesis, Methodology]`。
Teacher 提取结构：`Thoughts` -> `SearchQueries` -> `Decision` -> `Score`。

---

## 🚀 使用指南

### 1. 环境准备
确保已安装所需的 Python 依赖库，并在环境变量中配置好您的大模型 API 密钥以及 OpenAlex 邮箱：
```bash
pip install requests backoff pyalex concurrent.futures
export OPENALEX_MAIL_ADDRESS="your_email@domain.com"
```

### 2. 准备主题文件
在根目录下创建或编辑 `theme_idea_gen.txt`，输入您想要探索的通信领域粗略方向。例如：
> B5G/6G 网络中的语义通信与动态资源分配优化

### 3. 运行主程序
可以直接运行 Python 脚本，支持通过命令行参数动态调整并发数、迭代次数和使用的模型：

```bash
python idea_generator.py \
    --theme_file theme_idea_gen.txt \
    --n_students 9 \
    --n_teachers 5 \
    --max_student_iters 3 \
    --max_teacher_iters 10 \
    --model "gemini-3-pro-high" \
    --output_file all_generated_ideas.txt \
    --review_log novelty_scores.log
```

### 4. 参数说明
| 参数名 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `--theme_file` | `theme_idea_gen.txt` | 输入文件，包含研究的主题方向。 |
| `--n_students` | 9 | 并发执行的 Student Agent 数量（决定 Idea 的广度）。 |
| `--n_teachers` | 5 | 并发执行的 Teacher Agent 数量（决定评估的速度）。 |
| `--max_student_iters` | 3 | Student 打磨一个 Idea 的最大轮数。 |
| `--max_teacher_iters` | 10 | Teacher 为了查重所能允许的最大检索轮数。 |
| `--model` | `gemini-3-pro-high` | 所调用的底层大语言模型。 |
| `--output_file` | `all_generated_ideas.txt` | 阶段一输出的所有未经筛选的原始 Idea (JSON 格式)。 |
| `--review_log` | `novelty_scores.log` | 阶段二输出的带状元评分与评审意见的最终结果。 |

---

## 📂 输出示例

执行结束后，查看 `novelty_scores.log`，您将获得类似如下的高质量反馈：

```text
--- Idea 1 ---
Title: Semantic-Aware Dynamic Resource Allocation for Low-Latency Massive MIMO Systems
Score: 8/10
Review Comments:
[评审意见] 该设想将语义通信与大规模MIMO的物理层资源分配结合，具有一定的创新性。通过检索 "Semantic communication" AND "Massive MIMO"，发现大多数现有工作集中在信源编码，而非天线与功率的联合动态分配。然而，其 Methodology 中对信道状态信息(CSI)获取的延迟假设过于理想，建议在仿真中加入非完美CSI的考量。整体具有较高的落地价值。
========================================
```

这是一份为您量身定制的 `Aether` 项目中 `Research Planner` (研究计划生成) 模块的 README 文档。结构和风格与上一个模块保持一致，使用了标准且安全的 Mermaid 语法以确保完美渲染。

---

# 📐 Aether: Research Planner 模块详解

## 📖 模块简介
**Research Planner** 是 Aether 的“架构师”与“施工指导”。它的核心职责是将 `IdeaGenerator` 提出的抽象、高阶的科研想法（Idea），转化为**极其详尽、循序渐进、数学严谨且包含具体基线（Baselines）的可执行代码编写与仿真计划**。

由于后续的仿真代码将由能力受限的代码生成 AI 来完成，因此该模块生成的 Plan 必须达到“手把手”的颗粒度，严禁跳跃性思维。

该模块同样采用 **“学生-导师” (Student-Teacher) 协同对抗架构**：
1. **Student Planner (研究员)**：负责拆解 Idea，建立数学系统模型，并严格按照“传统基线 -> AI基线 -> 创新方案分步叠加”的逻辑编写执行计划。
2. **Teacher Planner (资深教授)**：扮演极其严苛的导师，专门挑刺。审查步骤跨度是否过大、数学表达是否清晰、验收标准（expected outcome）是否可量化。

---

## ⚙️ 核心工作流程图

以下是 `Research Planner` 模块的并发执行逻辑图：

```mermaid
graph TD
    Start(["启动 Planner 主程序"]) --> LoadIdeas["读取阶段一生成的 Ideas (JSON)"]
    
    LoadIdeas --> ForkK["为每个 Idea 并发启动 K 对师生 Agent"]
    
    subgraph PlanningPhase ["核心迭代：计划制定与审查 (Student-Teacher 协同)"]
        InitStudent["Student: 接收 Idea (或 Teacher 反馈)"]
        StudentGen["生成 / 修改细粒度研究计划 (严格4模块)"]
        TeacherReview["Teacher: 严苛审查计划 (颗粒度/数学/基线/可衡量性)"]
        Eval{"Decision == Pass?"}
        
        InitStudent --> StudentGen
        StudentGen --> TeacherReview
        TeacherReview --> Eval
        Eval -- "Refine (附带修改建议)" --> InitStudent
        Eval -- "Pass 或 达最大迭代" --> SavePlan["截取该 Agent 的最终 Plan"]
    end

    ForkK --> PlanningPhase
    SavePlan --> Aggregate["按 Idea 分组聚合 K 份不同的实现路线"]
    Aggregate --> WriteFiles(["输出至 final_research_plans 文件夹 (独立 JSON)"])

    %% 样式定义
    classDef student fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef teacher fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#000;
    class InitStudent,StudentGen student;
    class TeacherReview,Eval teacher;
```

---

## 🛠️ 详细实现解析

### 1. 极其严格的 4 模块标准化输出
为了保证后续代码执行者不会“迷失方向”，Student Agent 被 Prompt 严格限制，必须按照以下 4 个模块输出计划：
*   **模块 1：System Model (系统模型)**：必须使用具体的数学语言定义变量（如信道矩阵、发送信号、目标优化函数等）。变量来源必须清晰。
*   **模块 2：Baselines (基线方法)**：强制要求先实现传统通信方法（如 ZF, MMSE），再实现基础 AI 方法作为对比参照物，确保科研的严谨性。
*   **模块 3：Innovative Scheme (创新方案拆解)**：强制要求将 AI 创新部分**至少拆分成 3 个递进的微小步骤**（例如：先微调基础模型 -> 再引入新损失函数 -> 最后应用到复杂拓扑场景）。
*   **模块 4：Comparison & Outcomes (对比与预期)**：明确评估指标。每个步骤的 `expected_outcome` 必须是可衡量的（如“成功生成均值为0方差为1的瑞利信道并无报错”）。

### 2. 师生对抗机制 (Actor-Critic 思想)
*   **迭代打磨**：Student 初次提交计划后，Teacher 会根据“步骤是否太宽泛？”、“是否有数学描述？”等核心准则进行苛刻打分。
*   **反馈修正**：如果 Teacher 判定为 `Refine`，会将详细的批评意见（`Thoughts`）打回给 Student。Student 必须在下一轮上下文中吸收这些批评，细化颗粒度。
*   **兜底机制**：如果达到 `max_iters` 仍未拿到 `Pass`，系统会自动保留最后一版的计划，防止死循环。

### 3. 多路并行探索 ($K$-Agents 并发)
通信仿真往往有多种实现路径。该脚本引入了 `--k_agents` 机制：
*   对于**每一个**输入的 Idea，系统会并发启动 $K$ 组完全独立的 `Student-Teacher` 组合。
*   这意味着同一个 Idea 会得到 $K$ 种不同的落地计划（Plan），极大增加了后续代码生成成功的容错率。最后会将这 $K$ 份计划统一汇总到一个独立的 JSON 文件中（如 `idea_1_plans.json`）。

---

## 🚀 使用指南

### 1. 运行主程序
在完成 Idea 生成后，使用以下命令启动 Planner。系统会自动读取上一步的输出文件，并发生成执行计划：

```bash
python research_planner.py \
    --input_file all_generated_ideas.txt \
    --output_dir final_research_plans \
    --log_dir planner_logs \
    --max_iters 3 \
    --model_student "gemini-3-pro-high" \
    --model_teacher "gemini-3-pro-high" \
    --max_workers 5 \
    --k_agents 3
```

### 2. 参数说明
| 参数名 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `--input_file` | `all_generated_ideas.txt` | 阶段一输出的 Idea 汇总 JSON 文件。 |
| `--output_dir` | `final_research_plans` | **存放最终计划的输出文件夹**。每个 Idea 将在此处生成一个独立的 `.json` 文件。 |
| `--log_dir` | `planner_logs` | 存放各个智能体推理过程日志的文件夹。 |
| `--max_iters` | 3 | 每个 Idea 师生之间最大讨论、修改的轮数。 |
| `--model_student` | `gemini-3-pro-high` | Student (计划制定者) 使用的 LLM 模型。 |
| `--model_teacher` | `gemini-3-pro-high` | Teacher (计划审查者) 使用的 LLM 模型。 |
| `--max_workers` | 3 | 整个程序的并发线程池大小（控制 API 并发请求量）。 |
| `--k_agents` | 3 | **核心参数**：为每个 Idea 并行分配几组独立的师生 Agent（生成几份不同的备选计划）。 |

---

## 📂 输出示例

执行结束后，进入 `final_research_plans/idea_1_plans.json`，您将看到结构严谨的执行计划。每一步都被切分到了可以直接用来指导编写代码的程度：

```json
[
    {
        "Task_ID": "idea_1_agent_1",
        "Final_Decision": "Pass",
        "Teacher_Final_Feedback": "Perfect plan.",
        "Detailed_Plan": [
            {
                "idx": 1,
                "name": "System Model: 构建基础多天线传输信道",
                "content": "使用 Python/NumPy 编写。设基站天线数 Nt=64, 用户单天线 Nr=1。生成复高斯瑞利衰落信道矩阵 H (大小为 1x64)，H ~ CN(0, I)。发送符号 s 采用 QPSK 调制...",
                "expected_outcome": "能够成功实例化 H 矩阵和 s 向量，矩阵维度完全匹配，打印输出无报错。"
            },
            {
                "idx": 2,
                "name": "Baselines: 实现传统迫零(ZF)预编码",
                "content": "基于步骤 1 的信道矩阵 H，计算 ZF 预编码矩阵 W_zf = H^H (H H^H)^-1。对发送信号进行预编码 x = W_zf * s。添加高斯白噪声 n...",
                "expected_outcome": "计算出接收信噪比(SNR)并在不同噪声方差下绘制出传统的误码率(BER)曲线，作为后续 AI 方法的对比基准。"
            }
            // ... 递进的创新AI步骤 3, 4, 5 等等
        ]
    }
]
```
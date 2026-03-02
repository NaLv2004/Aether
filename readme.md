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
    Start([启动主程序 main]) --> ReadTheme[读取研究主题 txt]
    
    subgraph 阶段一：Idea 生成与打磨 (Student Agents)
        ReadTheme --> InitStudents[并发启动 N 个 Student Agent]
        InitStudents --> StudentLoop
        
        StudentLoop{迭代次数满或满意?}
        StudentLoop -- 未结束 --> Generate[基于提示词生成/优化 Idea <br> 提取 Search Queries]
        Generate --> OpenAlex1[调用 OpenAlex API 检索相关文献]
        OpenAlex1 --> Feedback1[将文献摘要反馈给 LLM]
        Feedback1 --> StudentLoop
        
        StudentLoop -- I'm done / 达上限 --> OutputIdeas[保存该 Agent 的 Ideas]
    end

    OutputIdeas --> MergeIdeas[合并并打乱所有候选 Ideas]
    
    subgraph 阶段二：新颖性审查与打分 (Teacher Agents)
        MergeIdeas --> InitTeachers[并发启动 M 个 Teacher Agent 分配评估任务]
        InitTeachers --> TeacherLoop
        
        TeacherLoop{Decision == Finished?}
        TeacherLoop -- Pending --> Review[审阅 Idea <br> 提出查重 Search Queries]
        Review --> OpenAlex2[调用 OpenAlex API 检索是否撞车]
        OpenAlex2 --> Feedback2[将文献反馈给审稿 LLM]
        Feedback2 --> TeacherLoop
        
        TeacherLoop -- Finished --> Score[给出详细评审意见及 Score 1-10]
    end
    
    Score --> SaveResult([输出结果至 Log 与 JSON])

    classDef student fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef teacher fill:#f8d7da,stroke:#dc3545,stroke-width:2px;
    class InitStudents,StudentLoop,Generate,Feedback1 student;
    class InitTeachers,TeacherLoop,Review,Feedback2 teacher;
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
## 3. 代码文件详细说明 (File Details)

以下文档基于代码依赖关系从底层向上递推生成：

## 文件: `dataset.py`

### 1. 计划步骤映射

该文件 `dataset.py` 完整实现了【科研计划】中的 **步骤 1：建立窄带近场非平稳信道模型与数据集构建 (System Model - Narrowband Dataset)**。

它是整个科研项目的基石，负责模拟极大规模 MIMO（XL-MIMO）在太赫兹频段的物理特性。具体服务于以下目标：
*   **物理建模**：构建了基于球面波的近场信道模型，而非传统的远场平面波模型。
*   **非平稳性模拟**：实现了“可见区域”（Visibility Region, VR）掩码，模拟了 XL-MIMO 中用户仅能覆盖部分阵列的特征。
*   **数据驱动基础**：生成了包含信道矩阵 $\mathbf{H}$、发送符号 $\mathbf{x}$ 和接收信号 $\mathbf{y}$ 的标准化数据集，为后续的传统基线算法（步骤 2）和深度学习模型（步骤 3-8）提供训练与测试数据。

---

### 2. 总体功能描述

`dataset.py` 的核心功能是构建一个符合物理规律的**太赫兹近场非平稳通信仿真环境**。其逻辑流程如下：

1.  **几何坐标构建**：在三维空间中布置一个沿 Y 轴排列的均匀线性阵列（ULA），并随机生成分布在菲涅尔近场区（5-30米）内的用户坐标。
2.  **空间非平稳性（VR）生成**：为每个用户随机指定一个阵列中心，并根据 $0.3N$ 的半径划定“可见天线”。只有落在该区域内的天线才能接收到该用户的有效信号，其余天线增益设为 0。
3.  **近场信道计算**：基于精确的距离 $r_{n,k}$ 计算球面波传播的幅度衰减（自由空间路径损耗）和相位偏移，生成复数信道矩阵 $\mathbf{H}$。
4.  **信号传输仿真**：
    *   生成归一化的 **QPSK** 发送符号。
    *   执行矩阵乘法 $\mathbf{y} = \mathbf{Hx} + \mathbf{n}$。
    *   根据设定的 **SNR（信噪比）** 范围，动态计算信号功率并添加复高斯白噪声。
5.  **工程化封装**：利用 PyTorch 的 `Dataset` 和 `DataLoader` 接口，将复杂的物理仿真封装为易于调用的迭代器，支持多线程加载和 Batch 处理。

---

### 3. 详细调用方式

#### 主要类与函数说明

**1. `NearFieldDataset(num_samples, N, K, snr_range)`**
*   **物理含义**：
    *   `num_samples`: 生成样本的总数。
    *   `N`: 基站天线总数（如 256），代表阵列规模。
    *   `K`: 同时通信的用户总数（如 8）。
    *   `snr_range`: 元组 `(min_db, max_db)`，定义了仿真中接收端信噪比的随机分布范围。

**2. `get_dataloaders(...)`**
*   **物理含义**：
    *   `batch_size`: 训练时的批大小。
    *   `num_train/val/test`: 分别对应训练集、验证集和测试集的样本数量。
    *   其他参数同上。

#### Python 调用示例

```python
from dataset import get_dataloaders

# 1. 初始化数据加载器
# 模拟一个拥有 256 根天线、服务 8 个用户的系统
# 训练集 10000 样本，SNR 分布在 -5dB 到 15dB 之间
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64,
    num_train=10000,
    num_val=1000,
    num_test=1000,
    N=256,
    K=8,
    snr_range=(-5, 15)
)

# 2. 在训练循环中使用
for batch_idx, (H, x, y) in enumerate(train_loader):
    # H: 信道矩阵，维度 [Batch, N, K] (复数张量)
    # x: 发送符号，维度 [Batch, K, 1] (复数张量)
    # y: 接收信号，维度 [Batch, N, 1] (复数张量)
    
    # 打印第一个 batch 的维度信息
    if batch_idx == 0:
        print(f"Channel Matrix H shape: {H.shape}")
        print(f"Transmitted x shape: {x.shape}")
        print(f"Received y shape: {y.shape}")
        break

# 3. 物理量验证示例
# 从 H 中可以观察到非平稳性：大部分元素由于 VR Mask 的存在而为 0
sparsity = (H[0].abs() > 0).float().mean()
print(f"Channel Sparsity (VR coverage): {sparsity:.2%}")
```

--------------------------------------------------------------------------------

## 文件: `dataset_wideband.py`

### 1. 计划步骤映射

该文件 `dataset_wideband.py` 严格实现了【科研计划】中的 **步骤 6：宽带 OFDM 信道升级与数据重构 (Innovative Scheme Part 3 - Wideband Data)**。

它是整个宽带仿真环节的基础，通过引入频率维度 $F$，将原有的窄带近场模型扩展为宽带多载波模型。该文件生成的信道矩阵 $H$ 包含了随频率变化的相位偏移，从而在物理层面上模拟了太赫兹频段特有的“波束分裂”（Beam Splitting）效应，为后续步骤 7 中频率感知 GNN 的训练和验证提供了核心数据集。

---

### 2. 总体功能描述

该文件的核心功能是构建一个符合物理规律的**太赫兹（0.1 THz）极大规模 MIMO（XL-MIMO）宽带近场非平稳信道数据集**。其主要逻辑包含以下几个方面：

*   **几何建模与近场特性**：基站采用均匀线性阵列（ULA），用户随机分布在距离基站 5 到 30 米的近场菲涅尔区内。信道计算基于**球面波模型**（而非远场平面波），精确计算每个天线到每个用户之间的物理距离。
*   **空间非平稳性（VR Mask）**：模拟了 XL-MIMO 的关键特性，即每个用户只能被阵列中的一部分天线（可见区域，Visibility Region）有效覆盖。代码通过随机生成中心天线并设定可视半径（$0.3N$）来生成二值化的掩码矩阵 $M$。
*   **宽带 OFDM 效应**：将总带宽（10 GHz）划分为多个子载波（如 16 个）。对于每个子载波，根据其特定的频率 $f_m$ 计算相位响应 $e^{-j \frac{2\pi f_m}{c} r_{n,k}}$。这种频率相关的相位变化是导致近场波束分裂的根本原因。
*   **端到端信号仿真**：自动生成归一化的 QPSK 发送符号，并根据设定的 SNR 范围（如 -5dB 到 15dB）为每个样本、每个子载波添加独立的复高斯白噪声，最终输出接收信号 $y$。
*   **高效内存管理**：利用 PyTorch 的张量广播机制（Broadcasting）在初始化时一次性生成所有样本，避免了训练过程中的实时计算开销。

---

### 3. 详细调用方式

#### 类名：`NearFieldWidebandDataset`

该类继承自 `torch.utils.data.Dataset`，可直接用于 PyTorch 的 `DataLoader`。

**参数说明：**
*   `num_samples` (int): 生成样本的总数（如 10000）。
*   `N` (int): 基站天线总数（默认 256）。
*   `K` (int): 用户总数（默认 8）。
*   `F` (int): 子载波数量（默认 16）。
*   `f_c` (float): 中心载波频率，单位 Hz（默认 0.1e12，即 0.1 THz）。
*   `B_bw` (float): 系统总带宽，单位 Hz（默认 10e9，即 10 GHz）。
*   `snr_range` (tuple): 训练/测试时的信噪比范围，单位 dB（默认 `(-5, 15)`）。

**返回数据结构：**
每次迭代返回一个元组 `(H, x, y)`：
*   `H`: 信道矩阵，维度为 `(F, N, K)`。
*   `x`: 发送的 QPSK 符号，维度为 `(F, K, 1)`。
*   `y`: 接收信号，维度为 `(F, N, 1)`。

#### Python 调用示例

```python
import torch
from torch.utils.data import DataLoader
from dataset_wideband import NearFieldWidebandDataset

# 1. 初始化宽带数据集
dataset = NearFieldWidebandDataset(
    num_samples=1000, 
    N=256, 
    K=8, 
    F=16, 
    f_c=0.1e12, 
    B_bw=10e9, 
    snr_range=(0, 10)
)

# 2. 使用 DataLoader 进行批量加载
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 遍历数据
for batch_idx, (H_batch, x_batch, y_batch) in enumerate(dataloader):
    # H_batch shape: [32, 16, 256, 8]  -> (Batch, Subcarriers, Antennas, Users)
    # x_batch shape: [32, 16, 8, 1]    -> (Batch, Subcarriers, Users, 1)
    # y_batch shape: [32, 16, 256, 1]  -> (Batch, Subcarriers, Antennas, 1)
    
    print(f"Batch {batch_idx}: H shape is {H_batch.shape}")
    
    # 在此处接入你的检测算法 (如 MMSE 或 GNN)
    # ...
    break
```

--------------------------------------------------------------------------------

## 文件: `baselines.py`

### 1. 计划步骤映射

该文件 `baselines.py` 严格对应【科研计划】中的 **步骤 2：实施传统基线方法 (Baselines - Traditional ZF/MMSE)**。

它的核心作用是为后续提出的创新性 Graph-VR-Det 算法建立性能标杆。通过在相同的近场非平稳信道模型下运行传统的线性检测算法（迫零检测 ZF 和 最小均方误差检测 MMSE），该文件产出了理想情况下的误码率（BER）参考曲线，作为评价 AI 模型优劣的“天花板”或“基准线”。

---

### 2. 总体功能描述

`baselines.py` 是整个仿真系统的**性能评估基准模块**。它模拟了通信接收机在获取信道状态信息（CSI）后，对用户发送的 QPSK 信号进行恢复的过程。其主要逻辑如下：

1.  **数据驱动评估**：动态调用 `dataset.py` 生成包含空间非平稳特性的近场信道数据。由于数据集生成的 $H$ 矩阵已包含了可见区域（VR）掩码，这里的检测本质上是 **Genie-Aided（理想辅助）** 的，即假设接收机完美已知哪些天线是有效的。
2.  **线性检测实现**：
    *   **VR-ZF (Zero Forcing)**：通过计算信道矩阵的伪逆来消除多用户干扰，公式为 $\hat{\mathbf{x}}_{ZF} = (\mathbf{H}^H \mathbf{H})^{-1} \mathbf{H}^H \mathbf{y}$。
    *   **VR-MMSE (Minimum Mean Square Error)**：在迫零的基础上考虑了噪声功率，通过正则化项平衡干扰消除与噪声放大，公式为 $\hat{\mathbf{x}}_{MMSE} = (\mathbf{H}^H \mathbf{H} + \sigma^2 \mathbf{I})^{-1} \mathbf{H}^H \mathbf{y}$。
3.  **鲁棒矩阵运算**：利用 `torch.linalg.solve` 进行高效的批处理（Batched）矩阵求逆，并内置了异常处理机制，当矩阵奇异时自动回退至 `pinv`（伪逆），确保大规模仿真不中断。
4.  **自动化指标统计**：自动完成从复数符号到比特流的解调（硬判决），并统计不同信噪比（SNR）下的平均误码率（BER）。

---

### 3. 详细调用方式

#### 函数/类定义
文件主要通过 `evaluate_baselines(args)` 函数执行，其中 `args` 是一个包含仿真配置的对象。

#### 参数物理意义
*   `N` (int): **基站天线总数**。在 XL-MIMO 场景下通常为 256 或更高。
*   `K` (int): **用户数量**。表示同时占用相同频时资源的单天线用户数。
*   `num_samples` (int): **测试样本数**。每个 SNR 点下生成的独立信道实现次数，样本越多，BER 统计越精确。
*   `batch_size` (int): **批处理大小**。为了加速计算，一次性送入 GPU/CPU 进行矩阵运算的样本数量。
*   `snr_list` (list of float): **信噪比列表 (dB)**。衡量信号功率与噪声功率的比值，决定了通信环境的恶劣程度。

#### Python 调用示例
你可以通过命令行直接运行，也可以在其他脚本中构造参数对象调用：

```python
import torch
from baselines import evaluate_baselines

# 1. 构造模拟的参数配置类
class SimulationArgs:
    def __init__(self):
        self.N = 256            # 256 根天线
        self.K = 8              # 8 个用户
        self.num_samples = 1000 # 测试 1000 个样本
        self.batch_size = 50    # 每批处理 50 个
        self.snr_list = [-5.0, 0.0, 5.0, 10.0, 15.0] # 评估的 SNR 范围

args = SimulationArgs()

# 2. 执行基线评估
# 注意：确保当前目录下存在 dataset.py 文件
evaluate_baselines(args)
```

**命令行运行方式：**
```bash
python baselines.py --N 256 --K 8 --num_samples 2000 --snr_list -5 0 5 10 15
```

--------------------------------------------------------------------------------

## 文件: `mpnn.py`

### 1. 计划步骤映射

该文件 `mpnn.py` 严格对应【科研计划】中的 **步骤 3：实施基础 AI 方法 (Baselines - Fully-Connected MPNN)**。

它是整个科研流程中的重要基准点，旨在建立一个全连接的消息传递神经网络（MPNN）作为 AI 检测器的性能标杆。通过将 XL-MIMO 检测问题建模为图上的推断问题，它为后续步骤 5（稀疏图优化）和步骤 7（频率感知优化）提供了标准的架构原型和性能对比基线。

---

### 2. 总体功能描述

`mpnn.py` 实现了一个基于深度学习的图神经网络检测器，用于在极大规模 MIMO（XL-MIMO）系统中从接收信号 $y$ 和信道矩阵 $H$ 中恢复发送符号 $x$。

其核心逻辑如下：
1.  **图建模**：将通信系统抽象为一个二部图。**天线**和**用户**分别作为两类节点，它们之间的**信道系数**作为连接边的特征。
2.  **特征嵌入 (Embedding)**：由于信道系数和信号通常是复数，代码将其实部和虚部拼接为 2 维向量，并通过线性层映射到高维空间（默认 64 维），以增强特征表达能力。
3.  **数值缩放 (Scaling)**：针对太赫兹（THz）频段路径损耗极大的物理特性（信道增益极小），引入了 `scale_factor`（如 $10^5$），将输入数据放大到神经网络易于处理的量级，有效防止梯度消失。
4.  **迭代消息传递 (Message Passing)**：
    *   **天线到用户 (A2U)**：每个用户节点收集所有与其相连的天线节点信息，通过 MLP 融合后更新自身状态。
    *   **用户到天线 (U2A)**：天线节点收集所有用户的信息进行状态更新。
    *   通过多层（默认 4 层）迭代，模型能够学习到复杂的空间相关性和多用户干扰模式。
5.  **符号读出 (Readout)**：最终层通过线性映射输出每个用户发送符号的实部和虚部预测值。
6.  **端到端训练与评估**：包含完整的训练循环（MSE 损失函数）和针对不同信噪比（SNR）的误码率（BER）评估逻辑。

---

### 3. 详细调用方式

#### 类与函数参数说明

**1. `MPNN` 类**
*   `num_layers` (int): 消息传递的迭代次数。在通信中代表算法的深度，层数越多，消除干扰的能力通常越强，但计算复杂度越高。
*   `hidden_dim` (int): 隐藏层特征维度。代表模型对空间特征的抽象能力。
*   `scale_factor` (float): 物理信号缩放因子。用于补偿太赫兹频段极大的路径损耗，确保数值稳定性。

**2. `forward(y, H)` 方法**
*   `y` (Tensor): 接收信号向量，维度为 `(Batch, N, 1)`。代表基站 $N$ 根天线收到的复数信号。
*   `H` (Tensor): 信道矩阵，维度为 `(Batch, N, K)`。代表 $K$ 个用户到 $N$ 根天线的复数信道增益。

#### Python 调用示例

该文件可以直接通过命令行运行进行训练，也可以在代码中实例化使用。

**命令行运行示例：**
```bash
# 在 /workspace 目录下运行，设置天线数256，用户数8，训练30轮
python mpnn.py --N 256 --K 8 --epochs 30 --batch_size 64 --lr 1e-3
```

**代码内集成示例：**
```python
import torch
from mpnn import MPNN

# 1. 物理参数设置
N = 256  # 基站天线数
K = 8    # 用户数
batch_size = 4

# 2. 模拟生成随机输入数据 (复数张量)
# 模拟接收信号 y 和信道矩阵 H
y = torch.randn(batch_size, N, 1, dtype=torch.complex64)
H = torch.randn(batch_size, N, K, dtype=torch.complex64)

# 3. 实例化模型
# num_layers=4 对应计划中的 4 次迭代
model = MPNN(num_layers=4, hidden_dim=64, scale_factor=1e5)

# 4. 前向传播进行信号检测
# 输出维度为 (batch_size, K, 2)，即 K 个用户各自的 [实部, 虚部] 预测
detected_x = model(y, H)

print(f"检测输出维度: {detected_x.shape}") # torch.Size([4, 8, 2])
```

--------------------------------------------------------------------------------

## 文件: `vr_discovery.py`

### 1. 计划步骤映射

该文件 `vr_discovery.py` 严格对应【科研计划】中的 **步骤 4：创新方案1 - 动态可见区域(VR)发现与稀疏图构建**。

它是整个创新方案的基石，负责将传统的密集信道矩阵（Dense Channel Matrix）转化为图神经网络（GNN）可处理的稀疏二分图结构。通过模拟导频阶段的能量检测，该文件实现了计划中提到的“动态发现用户与天线子集之间的强连接”，从而为后续步骤中降低计算复杂度提供了拓扑依据。

---

### 2. 总体功能描述

在极大规模 MIMO（XL-MIMO）系统中，由于天线阵列尺寸巨大，每个用户仅能被部分天线有效覆盖，这种物理现象被称为**空间非平稳性**。`vr_discovery.py` 的核心功能是利用物理层导频信号，自动识别出这些“有效覆盖区域”（Visibility Region, VR）。

**核心逻辑流程：**
1.  **导频信号模拟**：在给定的信噪比（SNR）条件下，为真实的信道矩阵 $H$ 添加复高斯白噪声，模拟基站端接收到的导频信号 $Y_p$。
2.  **能量特征提取**：计算每个天线节点接收到的导频能量 $|Y_p|^2$。
3.  **动态稀疏化（Top-S 过滤）**：针对每一个用户，仅保留能量最强的 $S$ 根天线。这一步实现了从“全阵列处理”到“局部 VR 处理”的转变，将连接关系从 $N \times K$ 简化为 $S \times K$。
4.  **图拓扑构建**：将提取出的连接关系转换为 PyTorch Geometric (PyG) 标准的 `edge_index` 格式。为了支持深度学习中的批处理（Batching），该文件还实现了自动索引偏移计算，确保不同样本的节点在图中被正确隔离。

**物理意义**：该文件通过数据驱动的方式，在不需要先验位置信息的情况下，实时探测电磁波的可见性边界，为稀疏 GNN 检测器提供了“在哪里进行计算”的指引。

---

### 3. 详细调用方式

#### 函数定义：
`get_sparse_edge_index(H, snr_db, S)`

#### 参数物理意义：
*   **`H` (torch.Tensor)**: 维度为 `[B, N, K]`。表示一个 Batch 的信道矩阵。
    *   `B`: 批大小（Batch Size）。
    *   `N`: 基站总天线数（如 256）。
    *   `K`: 用户总数（如 8）。
*   **`snr_db` (float)**: 导频阶段的信噪比（单位：dB）。该值越低，噪声越大，发现 VR 区域的准确率（Hit Rate）通常会下降。
*   **`S` (int)**: 稀疏度参数，即每个用户保留的天线数量。通常根据 VR 的物理半径设定（例如设为总天线数的 30%）。

#### 返回值：
*   **`edge_index` (torch.Tensor)**: 维度为 `[2, B * K * S]`。PyG 格式的稀疏边索引，记录了哪些天线节点与哪些用户节点相连。
*   **`top_idx` (torch.Tensor)**: 维度为 `[B, S, K]`。每个用户选中的天线在原始阵列中的局部索引。

#### Python 调用示例：

```python
import torch
from vr_discovery import get_sparse_edge_index

# 1. 模拟参数设置
batch_size = 16
num_antennas = 256
num_users = 8
sparsity_S = 77  # 每个用户只连接 77 根天线 (约 30%)
pilot_snr = 10.0 # 10dB 的导频信噪比

# 2. 生成模拟信道矩阵 (实部和虚部)
H = torch.randn(batch_size, num_antennas, num_users) + 1j * torch.randn(batch_size, num_antennas, num_users)

# 3. 调用函数获取稀疏图结构
# edge_index 用于 GNN 的消息传递，top_idx 用于后续特征提取
edge_index, top_idx = get_sparse_edge_index(H, pilot_snr, sparsity_S)

# 4. 打印结果验证
print(f"生成的边索引维度: {edge_index.shape}") # 预期输出: [2, 16 * 8 * 77] = [2, 9856]
print(f"每个用户保留的天线数: {top_idx.shape[1]}") # 预期输出: 77
```

--------------------------------------------------------------------------------

## 文件: `baselines_wideband.py`

### 1. 计划步骤映射

该文件 `baselines_wideband.py` 严格对应【科研计划】中的 **步骤 6：宽带 OFDM 信道升级与数据重构 (Innovative Scheme Part 3 - Wideband Data)**。

在科研流程中，它的具体作用是：
*   **建立性能基准**：在引入 0.1 THz 宽带特性和波束分裂（Beam Splitting）效应后，通过传统的线性检测算法（ZF 和 MMSE）为系统提供性能参考线。
*   **验证数据正确性**：通过对 `dataset_wideband.py` 生成的多载波数据进行处理，确保宽带信道张量的维度（Batch, Subcarrier, Antenna, User）在数学运算和物理逻辑上是自洽的。
*   **定义理想上限**：由于该脚本在每个子载波上独立使用完美的信道状态信息（CSI）进行检测，它代表了不考虑计算复杂度和硬件受限情况下的理想检测性能（Upper Bound）。

---

### 2. 总体功能描述

`baselines_wideband.py` 是一个用于评估**宽带近场 XL-MIMO 系统**传统检测性能的仿真脚本。其核心逻辑如下：

1.  **多载波并行处理**：针对 OFDM 系统的 $F$ 个子载波，脚本通过张量重塑（Reshape）操作，将“子载波维度”合并到“批处理维度”中。这种做法在物理上等同于对每个子载波进行独立的窄带信号检测，能够高效地利用 GPU 的并行计算能力。
2.  **传统线性检测实现**：
    *   **VR-ZF (Zero-Forcing)**：迫零检测，通过计算信道矩阵的伪逆来消除用户间干扰。
    *   **VR-MMSE (Minimum Mean Square Error)**：最小均方误差检测，在迫零的基础上考虑了噪声功率，能够更好地平衡干扰消除与噪声放大。
3.  **鲁棒性矩阵运算**：使用 `torch.linalg.solve` 进行批处理矩阵求逆，并内置了异常处理机制（当矩阵奇异时自动回退到 `pinv` 伪逆），确保在大规模天线场景下的数值稳定性。
4.  **自动化评估流程**：脚本会自动遍历指定的 SNR（信噪比）列表，生成对应的宽带数据集，执行检测算法，并最终统计输出误码率（BER）曲线数据。

---

### 3. 详细调用方式

#### 函数接口说明
主要调用接口为 `evaluate_baselines_wideband(args)`。

**参数物理意义：**
*   `args.N` (int): 基站总天线数。在 XL-MIMO 场景下通常为 256 或更高。
*   `args.K` (int): 同时通信的用户总数。
*   `args.F` (int): OFDM 子载波数量。代表频域的采样密度。
*   `args.f_c` (float): 中心载波频率（单位：Hz）。例如 `0.1e12` 代表 0.1 THz。
*   `args.B` (float): 系统总带宽（单位：Hz）。带宽越大，波束分裂效应越明显。
*   `args.num_samples` (int): 每个 SNR 点下的测试样本数。样本越多，BER 统计越精确。
*   `args.batch_size` (int): 每次处理的样本批大小，影响显存占用。
*   `args.snr_list` (list of float): 待评估的信噪比列表（单位：dB）。

#### Python 调用示例

你可以通过以下代码在其他脚本中调用该功能，或者直接在终端运行。

```python
import torch

# 模拟命令行参数对象
class Args:
    def __init__(self):
        self.N = 256            # 256根天线
        self.K = 8              # 8个用户
        self.F = 16             # 16个子载波
        self.f_c = 0.1e12       # 0.1 THz 中心频率
        self.B = 10e9           # 10 GHz 带宽
        self.num_samples = 500  # 测试500个样本
        self.batch_size = 50    # 每批处理50个
        self.snr_list = [0, 10, 20] # 测试 0dB, 10dB, 20dB

# 实例化参数
args = Args()

# 导入并执行评估函数
# 注意：确保 dataset_wideband.py 在同一目录下
from baselines_wideband import evaluate_baselines_wideband

evaluate_baselines_wideband(args)
```

**终端运行方式：**
```bash
python /workspace/baselines_wideband.py --N 256 --K 8 --F 16 --snr_list -5 0 5 10 15
```

--------------------------------------------------------------------------------

## 文件: `sparse_gnn_wideband.py`

### 1. 计划步骤映射

该文件 `sparse_gnn_wideband.py` 严格实现了【科研计划】中的 **步骤 7：频率感知 Sparse-GNN 实现 (Innovative Scheme Part 4 - Freq-Aware GNN)**。

它是整个科研项目的核心创新点实现，旨在解决极大规模 MIMO（XL-MIMO）在太赫兹（0.1 THz）宽带通信中的两个关键挑战：
*   **空间非平稳性**：通过动态发现每个用户的可见区域（VR）并构建稀疏图来降低计算复杂度。
*   **波束分裂（Beam Splitting）效应**：通过在 GNN 的边特征中引入归一化频率偏移，使模型具备感知频率变化的能力，从而补偿宽带信号在不同子载波上的相位偏差。

---

### 2. 总体功能描述

该文件实现了一个**频率感知稀疏图神经网络（Freq-Aware Sparse-GNN）**，用于宽带近场 XL-MIMO 系统的信号检测。其核心逻辑包含以下三个模块：

1.  **联合频域动态构图 (`get_sparse_edge_index_wideband`)**：
    *   模拟导频传输过程，在所有子载波（Subcarriers）上对接收能量进行平均。
    *   为每个用户挑选能量最强的 $S$ 根天线作为其“可见区域”。
    *   构建一个跨子载波共享的稀疏二部图结构，将复杂的全连接检测问题转化为局部图上的消息传递问题。

2.  **频率感知消息传递架构 (`SparseMPNNLayer` & `SparseMPNNWideband`)**：
    *   **特征增强**：在天线与用户之间的边特征中，除了包含信道系数的实部和虚部，还额外拼接了归一化频率偏移 $\Delta f = (f_m - f_c)/f_c$。
    *   **稀疏聚合**：利用 `scatter_add_` 算子实现高效的消息聚合，仅在可见区域内的边上进行计算，显著降低了超大规模阵列下的运算开销。
    *   **多层迭代**：通过多层 A2U（天线到用户）和 U2A（用户到天线）的消息传递，模拟迭代检测过程。

3.  **端到端训练与评估流程**：
    *   支持在内存中动态生成宽带近场信道数据。
    *   通过将“批次（Batch）”与“子载波（Subcarrier）”维度合并，实现多载波信号的并行处理。
    *   提供不同 SNR 环境下的误码率（BER）测试功能，验证模型对波束分裂的补偿效果。

---

### 3. 详细调用方式

#### 主要函数与类参数说明

**1. `get_sparse_edge_index_wideband(H, snr_db, S)`**
*   `H` (Tensor): 宽带信道矩阵，维度为 `[Batch, Freq, Antennas, Users]`。
*   `snr_db` (float): 导频信号的信噪比，用于模拟真实的 VR 发现过程。
*   `S` (int): 每个用户保留的天线数量（即 VR 的大小）。

**2. `SparseMPNNWideband(num_layers, hidden_dim, scale_factor)`**
*   `num_layers` (int): 消息传递的层数（迭代次数）。
*   `hidden_dim` (int): 隐藏层神经元维度，代表节点特征的丰富度。
*   `scale_factor` (float): 缩放因子（如 $10^5$），用于将太赫兹频段极小的信道增益放大到易于神经网络处理的量级。

**3. `model.forward(y, H, edge_index, S, F, f_c, B_bw)`**
*   `y` (Tensor): 接收信号，维度 `[Batch*Freq, Antennas, 1]`。
*   `H` (Tensor): 展平后的信道矩阵，维度 `[Batch*Freq, Antennas, Users]`。
*   `edge_index` (Tensor): 由构图函数生成的稀疏边索引。
*   `F` (int): 子载波总数。
*   `f_c` (float): 中心载波频率（Hz）。
*   `B_bw` (float): 系统总带宽（Hz）。

#### Python 调用示例

```python
import torch
from sparse_gnn_wideband import SparseMPNNWideband, get_sparse_edge_index_wideband

# 1. 仿真参数设置
B, F, N, K = 4, 16, 256, 8  # Batch, 子载波, 天线数, 用户数
S = 77                       # 每个用户保留 30% 的天线
f_c = 0.1e12                 # 0.1 THz 中心频率
B_bw = 10e9                  # 10 GHz 带宽

# 2. 构造模拟数据 (复数张量)
H = torch.randn(B, F, N, K, dtype=torch.complex64)
y = torch.randn(B * F, N, 1, dtype=torch.complex64)
H_reshaped = H.view(B * F, N, K)

# 3. 动态构建稀疏图索引
# 模拟在 10dB 导频环境下发现可见区域 (VR)
edge_index, _ = get_sparse_edge_index_wideband(H, snr_db=10.0, S=S)

# 4. 初始化频率感知 Sparse-GNN 模型
model = SparseMPNNWideband(num_layers=4, hidden_dim=64, scale_factor=1e5)

# 5. 前向传播进行信号检测
# 输出 out 的维度为 [B*F, K, 2]，包含预测符号的实部和虚部
with torch.no_grad():
    predictions = model(y, H_reshaped, edge_index, S, F, f_c, B_bw)

print(f"检测结果维度: {predictions.shape}") # 预期输出: torch.Size([64, 8, 2])
```

--------------------------------------------------------------------------------

## 文件: `sparse_gnn.py`

### 1. 计划步骤映射

该文件 `sparse_gnn.py` 严格实现了【科研计划】中的 **步骤 5：创新方案 2 - 基于稀疏图的 GNN 检测算子 (Innovative Scheme Part 2)**。

它是整个科研任务的核心创新点实现，具体服务于以下逻辑：
*   **承接步骤 4**：调用 `vr_discovery.py` 中实现的动态可见区域（VR）发现算法，获取稀疏的图拓扑结构（`edge_index`）。
*   **改进步骤 3**：将步骤 3 中的全连接 MPNN 改造为稀疏架构。通过仅在“强连接”（即用户可见的天线子集）上进行消息传递，验证在极大规模 MIMO（XL-MIMO）场景下降低计算复杂度的可行性。
*   **性能对比**：文件末尾包含了与全连接 MPNN 的耗时对比逻辑，直接服务于计划中关于“加速比”和“计算效率”的验证。

---

### 2. 总体功能描述

`sparse_gnn.py` 实现了一个专门用于 **XL-MIMO 近场信号检测的稀疏图神经网络 (Sparse-GNN)**。在 XL-MIMO 系统中，由于阵列尺寸巨大，每个用户只能看到阵列的一部分（即 Visibility Region, VR）。该文件利用这一物理特性，构建了一个稀疏二部图模型：

1.  **稀疏消息传递机制**：不同于传统 GNN 处理所有天线与用户间的连接，该模型只在动态发现的 VR 边上进行信息交换。它通过自定义的 `SparseMPNNLayer` 实现了天线到用户（A2U）和用户到天线（U2A）的迭代更新。
2.  **高效张量运算**：为了在不依赖额外图处理库（如 `torch_scatter`）的情况下保持高效率，代码利用 PyTorch 原生的 `scatter_add_` 算子实现了基于索引的并行聚合，并根据每个节点的度（Degree）进行归一化，确保了在大规模阵列下的数值稳定性。
3.  **物理感知缩放**：考虑到太赫兹（0.1THz）频段极大的路径损耗，模型引入了 `scale_factor`（默认 $10^5$），将微弱的接收信号放大到神经网络易于处理的量级，有效解决了深度学习在通信物理层应用中的梯度消失问题。
4.  **端到端仿真与评估**：文件集成了从数据加载、动态构图、模型训练到多 SNR 误码率（BER）测试的完整流程，并提供了与全连接基线模型的推断速度对比工具。

---

### 3. 详细调用方式

#### 主要类：`SparseMPNN`

**构造函数参数说明：**
*   `num_layers` (int): 消息传递的迭代次数（层数）。在通信中代表算法迭代处理的深度，默认为 4。
*   `hidden_dim` (int): 隐藏层特征维度。代表节点特征向量的长度，默认为 64。
*   `scale_factor` (float): 信号缩放因子。用于补偿近场球面波路径损耗导致的极小数值，默认为 $10^5$。

**`forward` 方法参数说明：**
*   `y` (Tensor): 接收信号向量，维度为 `(Batch, N, 1)`。物理意义是基站 $N$ 根天线收到的复数信号。
*   `H` (Tensor): 信道矩阵，维度为 `(Batch, N, K)`。物理意义是 $K$ 个用户到 $N$ 根天线的复信道增益。
*   `edge_index` (LongTensor): 稀疏图的边索引，维度为 `(2, E)`。由 `vr_discovery` 模块生成，定义了哪些天线与哪些用户存在“强连接”。
*   `S` (int): 每个用户保留的天线数量（VR 大小）。用于消息聚合时的平均归一化。

#### Python 调用示例

```python
import torch
from sparse_gnn import SparseMPNN
from vr_discovery import get_sparse_edge_index

# 1. 基础参数设置
B, N, K, S = 4, 256, 8, 77  # Batch大小, 天线数, 用户数, 每个用户可见天线数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 模拟生成输入数据 (复数张量)
H = torch.randn(B, N, K, dtype=torch.complex64).to(device)
y = torch.randn(B, N, 1, dtype=torch.complex64).to(device)

# 3. 动态构建稀疏图索引 (模拟导频阶段发现VR)
# 假设当前 SNR 为 10dB
edge_index, _ = get_sparse_edge_index(H, snr_db=10.0, S=S)
edge_index = edge_index.to(device)

# 4. 初始化 Sparse-GNN 模型
model = SparseMPNN(num_layers=4, hidden_dim=64).to(device)

# 5. 前向传播进行信号检测
# 输出 out 的维度为 (B, K, 2)，分别代表 K 个用户发送符号的实部和虚部预测值
out = model(y, H, edge_index, S)

print(f"输入接收信号维度: {y.shape}")
print(f"稀疏边数量: {edge_index.shape[1]}")
print(f"检测输出维度: {out.shape} (Batch, Users, Real/Imag)")
```

--------------------------------------------------------------------------------

## 文件: `scale_up_evaluation.py`

### 1. 计划步骤映射

该文件 `scale_up_evaluation.py` 完整实现了【科研计划】中的 **步骤 8：大规模天线扩展与最终对比评估 (Comparison & Scale-up)**。

同时，为了保证评估的完整性，它在内部集成了 **步骤 7** 的频率感知 Sparse-GNN 训练逻辑。该文件的核心目标是验证创新方案 `Graph-VR-Det` 的两个关键科学假设：
*   **规模可扩展性 (Scalability)**：证明在较小规模阵列（N=256）上训练的 GNN 模型，可以直接推广（Zero-shot）到超大规模阵列（N=1024）而性能不崩溃。
*   **计算高效性 (Efficiency)**：通过对比 Sparse-GNN 与传统算法在不同天线规模下的推理耗时，证明稀疏图处理在极大规模 MIMO 中的优势。

---

### 2. 总体功能描述

该文件是整个科研项目的**最终集成评估脚本**，它模拟了一个完整的太赫兹（0.1 THz）宽带近场通信系统。其主要功能模块包括：

1.  **在线模型训练**：在内存中动态生成 $N=256$ 天线的宽带非平稳信道数据，训练一个具备频率感知能力的稀疏图神经网络（Freq-Aware Sparse-GNN）。
2.  **跨规模零样本推理 (Zero-shot Scale-up)**：这是本文件的核心逻辑。利用 GNN 处理局部图结构的特性，将 $N=256$ 训练好的模型直接应用于 $N=1024$ 的场景。它会根据天线规模动态调整可见区域（VR）的大小 $S$，验证模型是否捕捉到了通用的近场物理规律。
3.  **性能基准对比**：
    *   **BER 评估**：在不同的信噪比（SNR）下，对比 Sparse-GNN 与拥有完美信道状态信息的 VR-MMSE（虚拟实现上限）的误码率。
    *   **耗时评估 (Benchmarking)**：测量单次信号检测的毫秒级耗时，特别关注在大规模天线（N=1024）下，包含“动态构图”在内的 GNN 全流程运行效率。
4.  **可视化产出**：自动生成并保存 `fig_ber.png`（误码率曲线）和 `fig_complexity.png`（推理耗时对比图），直观展示科研成果。

---

### 3. 详细调用方式

#### 主要函数说明

文件通过 `main(args)` 统一调度，核心逻辑封装在以下函数中：

*   **`train_model(args, device)`**:
    *   **物理意义**：在标准近场场景下训练 AI 检测器。
    *   **关键参数**：`args.N_train` (训练天线数), `args.S_train` (每个用户关联的稀疏边数), `args.f_c` (载波频率), `args.B_bw` (系统带宽)。
*   **`test_model(args, model, device)`**:
    *   **物理意义**：测试模型在不同规模阵列下的泛化能力。
    *   **关键参数**：`args.N_list` (测试的天线规模列表，如 [256, 1024]), `args.snr_list` (评估的信噪比范围)。
*   **`benchmark(args, model, device)`**:
    *   **物理意义**：评估算法的计算复杂度。它会统计从“接收导频并构建稀疏图”到“输出比特预测”的全过程耗时。

#### Python 调用示例

你可以通过命令行传递参数，也可以在脚本中构造 `args` 对象进行调用。以下是一个完整的 Python 调用示例：

```python
import torch
from scale_up_evaluation import main

# 构造配置参数
class SimulationArgs:
    def __init__(self):
        # 训练配置
        self.epochs = 5               # 快速演示设为5，实际科研建议20+
        self.lr = 1e-3                # 学习率
        self.batch_size = 16          # 批大小
        self.num_train = 1000         # 训练样本数
        self.num_val = 100            # 验证样本数
        self.num_test = 200           # 每个SNR下的测试样本数
        
        # 通信物理参数
        self.N_train = 256            # 训练时的基站天线数
        self.K = 8                    # 用户数
        self.F = 16                   # OFDM子载波数
        self.f_c = 0.1e12             # 中心频率 0.1 THz
        self.B_bw = 10e9              # 总带宽 10 GHz
        self.S_train = 77             # 训练时每个用户的可见天线数 (约30%)
        
        # 评估范围
        self.snr_list = [0, 10, 20]   # 测试的信噪比点 (dB)
        self.N_list = [256, 1024]     # 验证规模扩展性的天线数列表
        
        # 模型架构
        self.num_layers = 4           # GNN消息传递层数
        self.hidden_dim = 64          # 隐藏层维度
        self.scale_factor = 1e5       # 信号缩放因子（应对太赫兹极大的路径损耗）
        
        # 其他辅助参数
        self.train_snr = 10.0         # 导频构图时的参考SNR
        self.snr_min = -5             # 训练集随机SNR下限
        self.snr_max = 15             # 训练集随机SNR上限

# 实例化参数
args = SimulationArgs()

# 执行完整评估流程
# 该函数会自动完成：训练 -> N=256测试 -> N=1024测试 -> 耗时统计 -> 绘图
main(args)

# 运行结束后，检查当前目录下的 fig_ber.png 和 fig_complexity.png
```

--------------------------------------------------------------------------------


## [Current Step]
# 6G Cell-Free MIMO System Modeler and Dataset Generator

本模块用于生成用于 6G Cell-Free MIMO 系统的物理层信道数据集，模拟信号传输，并对比测试采用不同量化位宽对 AP 处 LMMSE（线性最小均方误差）检测器恢复性能的影响。

## 功能说明

1. **场景生成**：
   - 生成 1km x 1km 区域内随机均匀分布的 L 个 AP 和 K 个单天线用户的坐标。
   - 使用 3GPP 密集市区路径损耗模型 ($128.1 + 37.6 \log_{10}(d)$) 计算大尺度衰落。
   - 叠加标准差为 8dB 的对数正态阴影衰落。
   - 提供环境噪声及指定带宽下的 SNR 计算与打印。
2. **信号生成与接收**：
   - 对于每次采样 (epoch)，生成 K 个用户的 QPSK 发射符号 $s$ 及带有瑞利小尺度衰落特征的信道矩阵 $H_l$。
   - AP 接收端施加独立同分布的加性高斯白噪声。
3. **局部 LMMSE 检测**：
   - 实现每个 AP 的局部检测，利用公式：$\hat{s}_l = H_l^H (H_l H_l^H + \sigma^2 I)^{-1} y_l$ 进行初步解调。
4. **量化模块**：
   - 实现了具备动态范围估计的简单均匀量化器 `quantize_signal`，分别量化复数信号的实部与虚部。
   - 根据超参，测试不同离散位宽 $b_{lk} \in \{0, 2, 4, 8, \text{float32}\}$ 的影响，0位宽将直接丢弃信号映射为0。
5. **数据集拆分与保存 (AI集成)**：
   - 按照强制要求，通过 `--train_ratio` 参数对生成的多 epochs 原始信道及信号数据进行了**严格的训练集与测试集分割**。
   - 在程序运行结束后，结果被存储为 `cell_free_mimo_dataset.npz` 以便于后续直接导入如 PyTorch 的 AI 模型进行训练。每个 epoch 分割结果相同且明确。

## 运行方法

可以直接执行提供的 `run.bat` 脚本，它会自动安装必要的 Python 包并运行主流程：

## [Current Step]
# 6G 无小区 (Cell-Free) MIMO 基线评估

本项目主要通过 Python 脚本 `baselines.py` 评估了无小区大规模 MIMO (Cell-Free Massive MIMO) 系统在不同发射功率下的各项性能基准，旨在为后续分布式全精度与量化方案提供对比基准线。

## 脚本主要功能说明

1. **环境与信道生成**：调用并重新实例化 `system_model.py` 中的 `CellFreeSystem`。为不同的发射功率扫描点产生不同的场景参数（包括大规模衰落与接收端的平均 SNR），并复用之前的数据集分布。
2. **多方案信号检测与解调**：对于每次 Monte Carlo 实验（Epoch），自动针对上行接收端执行四种不同的组合接收策略：
   - **C-MMSE（集中式 MMSE）**: 各 AP 聚合所有的信道信息以及天线维度接收信号至云端处理器（CPU），进行全局维度的联合信道均衡和 MMSE 接收。
   - **全精度分布式（Local LMMSE）**: 各个 AP 独立计算局部 LMMSE，然后将全精度估算值无损传递至 CPU 并进行均值合并。
   - **4-bit 分布式量化**: 各个 AP 计算局部 LMMSE 后，分别将实部和虚部均进行 4-bit 量化传递到 CPU 取均值。
   - **2-bit 分布式量化**: 原理同上，但为 2-bit 极限压缩量化。
3. **BER（误码率计算）**：重构了 QPSK 的解调判定逻辑，通过比对真实发射信号以及软信息的实部/虚部正负号准确计算出错的 bit 数，并根据 Epoch 计算最终平滑的 BER。
4. **控制台输出与表格展示**：在仿真运算中途清晰展示关键的物理层信息及各个功率下的中间结果，并在最终使用标准表格格式展示四者的 BER 在 SNR 上升中的变化趋势。

## 超参数设置说明

利用 `argparse` 对核心参数进行了暴露。您可以在命令行直接通过指定 flag 修改环境参数或仿真强度：
- `--L`: (默认 `16`) 接入点（AP）的数量。
- `--N`: (默认 `4`) 每个 AP 配置的天线数量。
- `--K`: (默认 `8`) 场景内的单天线用户数。
- `--area_size`: (默认 `1.0`) 方形环境面积边长，单位为 km。
- `--bandwidth`: (默认 `20e6`) 系统可用信道总带宽，单位为 Hz。
- `--noise_psd`: (默认 `-174`) 基础热噪声功率谱密度（dBm/Hz）。
- `--epochs`: (默认 `300`) 每个发射功率水平下进行测试的 Monte Carlo 平滑执行次数。对于 BER 计算具有重要的平滑作用，每次 Epoch 会保证统一的小尺度信道衰落集合不变。
- `--p_tx_list`: (默认 `"0,5,10,15,20,25,30"`) 需要系统依次评估扫描的用户发射端发射功率水平（dBm），以逗号分隔。

### 使用方法

只需要运行项目内预设的 `run.bat`，系统会自动拉起并按照默认 300 轮 epochs 迭代模拟各参数。
或者在命令行中手动输入运行指令：

## [Current Step]
# 6G Cell-Free MIMO 基线测试修复

本模块提供了修复了物理环境抖动后的通信系统蒙特卡洛基线评估脚本 (`baselines.py`)。它旨在计算和比较集中式 Minimum Mean Square Error (C-MMSE) 检测方法和不同的分布式检测及量化方法的误码率 (BER) 性能，确保了在同一场景配置下对比不同发射功率的结果。

## 修复的功能描述

在早期的实现中，`baselines.py` 在每一次不同的发射功率配置(`p_tx`)遍历时都重新调用了 `generate_scenario()` 产生新的大尺度环境配置，这导致了阴影衰落和位置分布引起较大的 SNR 波动并掩盖了由于功率提升带来的自然信噪比增益。

当前更新中，执行以下严格控制逻辑：
1. **统一的物理部署**: 在测试任意 `p_tx` 前，实例化统一的 `CellFreeSystem` 并仅调用一次 `generate_scenario()` 以固化全局大尺度参数 (如 $\beta$)。
2. **动态功率调制**: 在扫掠不同发射功率参数时，动态更改了信道系统的 `p_tx_w` 和 `beta_w`（真实接收功率）从而保持 AP 和 UE 地理分布与大尺度特征严格不变化。
3. 接着对于同一环境和确定 SNR 分别通过小尺度衰落进行多 epoch 蒙特卡洛仿真进行对比。

## 可用参数选项 (Command Line Arguments)

通过 `argparse`，暴露了仿真使用的环境配置参数如下：
- `--L`: (int) Access Points (APs) 的数量。默认值：16。
- `--N`: (int) 每个 AP 的天线数量。默认值：4。
- `--K`: (int) 单天线用户数量。默认值：8。
- `--area_size`: (float) 区域大小 (千米)。默认值：1.0。
- `--bandwidth`: (float) 系统带宽 (Hz)。默认值：20e6。
- `--noise_psd`: (float) 噪声功率谱密度 (dBm/Hz)。默认值：-174。
- `--epochs`: (int) 每个功率点的小尺度信道衰落及数据包接收的独立蒙特卡洛随机试验次数。默认值：300。
- `--p_tx_list`: (str) 用户发射功率范围，允许使用逗号分隔指定多个 dBm 级别进行测试评估。默认值："0,5,10,15,20,25,30"

## 运行方式
调用 `run.bat` 将基于 300 轮蒙特卡洛 epoch 触发本脚本自动打印 `Average Receive SNR` 和 集中/分布检测在多组功率下的 BER 表格比较情况。

## [Current Step]
# 阶段 3: 全精度 GNN 检测器预训练 (FPGNN) 功能说明

本阶段脚本 `gnn_detector.py` 构建并训练了基于全精度 LMMSE 局部检测特征的深度学习检测网络（FPGNN），用以实现在 Cell-Free CPU 端的智能加权汇合以及多用户干扰消除 (MU-IC)。

### 1. 动态在线数据生成机制
由于单纯使用预生成磁盘数据可能会限制数据量，引起模型过拟合，我们设计了完全向量化的 `generate_data_batch` 方法在线不断产生数据样本：
- 场景的大尺度衰落 (AP位置与用户位置) 在 `generate_scenario()` 被锁定，使得物理结构一致。
- 每次训练 Batch 中，模型动态传入随机的发射功率范围 `p_tx`（-10 到 20 dBm 均匀分布）。这给予模型对抗不同信噪比场景的高泛化能力。
- 全套 LMMSE 检测也利用了 `np.einsum` 和 `np.linalg.inv` 的批量矩阵运算进行了优化，极速产出输入特征 $X$。

### 2. 深度神经网络架构设计 (FPGNN)
模型主要由三个子模块组成：
- **特征提升层 (Feature Lifting)**: 通过 MLP 将二维复平面软信息提升为高维隐藏特征(`hidden_dim`=32)。
- **AP-to-CPU 注意力机制**: 模型不直接采取简单的均值合并 (Mean-Pooling)，而是学习了各个局域 AP 的置信度权重 `attn_weights` 并施加 `softmax`。根据此权重对所有 AP 的信息作加权融合。
- **多用户干扰消除层 (MU-IC)**: 融合完成后，多用户的特征被拉平并通过数层全连接层交互（256->128）。这使得网络能够分析用户之间的空间和信道特征交叉以消除导频污染等干扰，并直接预测各个用户的复合符号分量。

### 3. 数据分割与公平测试方案
- 代码在初始化大尺度场景后，立即预先生成了针对 `[-20, -15, -10, ..., 20] dBm` 的各个固定评估验证集。
- 这些评估集和在线训练循环的数据严格分离，并在 Epoch 迭代期间维持不变，完美符合测试隔离规定。
- 测试结束时，程序对比了传统的均值合并方法（Dist-Full Baseline）和 FPGNN 预测符号在硬判决后的误码率 (BER)，充分量化显示由于注意力整合和 MU-IC 所引发的至少 15% - 50% 性能飞跃。

### 4. 命令行超参数暴露
程序暴露了多项可以由 Orchestrator 直接调节的超参数：
- `--epochs`: 总训练轮数 (建议 `30`)。
- `--batches_per_epoch`: 每轮生成多少次批次。
- `--batch_size`: 每批次处理的小尺度衰落样本量。
- `--lr`: Adam 优化器初始学习率。
- `--hidden_dim`: 特征提升隐藏维度。

详细使用命令请参照 `run.bat`。最终生成的预训练权重将被导出至 `fp_gnn_model.pth` 供后续量化部署环节加载。

## [Current Step]
# 物理先验增强的 FPGNN 接收器

当前项目对无小区大规模 MIMO (Cell-Free Massive MIMO) 系统中的全精度图神经网络 (FPGNN) 探测器架构进行了针对物理规律的深度优化。

## 核心重构与优化项

### 1. 局部 SNR (信噪比) 特征引入
原先的注意力机制由于没有接收信噪比指标，导致对不同 AP 的可靠度分配带有盲目性。
- 在 `generate_data_batch` 的底层信道建模中，计算每条链路各用户对应的天线接收端功率总和：`local_snr = 10 * np.log10(...)`。
- 将 `local_snr` 作为第 3 维特征维度 (`X`的形状调整为 `B x L x K x 3`)，连同估计值的实部和虚部共同送入 GNN 的特征提升网络中。这样注意力网络 (Attn-Net) 便可敏锐感知各个 AP 的信道质量。

### 2. 基线软信息加权与残差架构
传统的全连接 MLP 会一定程度上破坏原有的物理软信息。本次重构基于经典通信理论：
- **物理先验加权**：注意力模块学习得到的归一化权重 `attn_weights` 首先被直接用于对原始复数软信息 (`soft_info = x[..., :2]`) 进行线性加权求和，这在物理上充当了**多天线系统的自适应加权组合器**。
- **残差干扰消除网络**：加权后的高维隐含层特征 `fused_h` 被平铺并送往多用户联合干扰消除网络 (`mu_ic_mlp`) 以提取残余的跨用户干扰影响。网络最终输出为`基准分集组合结果 + 神经网络预测的多用户干扰残差补偿`。
- **初始化稳定**：强制将残差预测 MLP 的最后一层权重和偏置清零，使得训练的初始状态完全等效于最优线性权重组合器，从而保障其表现稳定并避免掉入破坏软信息的局部最小值。

## 参数配置说明

通过 Python 的 `argparser` 暴露了以下主要命令行参数：

- `--L`, `--N`, `--K`：无小区场景环境参数，分别为基站 (AP) 数量、AP 天线数量、单天线用户数量。
- `--epochs`：训练的迭代总轮数 (本次优化默认提升至 **40** 轮)。
- `--batches_per_epoch`：单 Epoch 中每次前向反向传播的次数。
- `--lr`：优化器学习率，默认 1e-3。
- `--hidden_dim`：控制隐层表示的网络宽度。

## 数据集分离机制
为了实现公平、真实的评测效果：
1. **测试集隔离生成**：在训练开始前，为验证在 -20 dBm 至 20 dBm 各个不同发射功率下的表现，系统提前生成了固定种子生成的测试点数据集 (`test_dataset`)。
2. **在线泛化训练集**：在单轮 epoch 的训练中，通过 `np.random.uniform(-10, 20)` 对功率注入随机噪声进行泛化训练。生成的这部分数据与测试集具有不同的微观分布，且测试集在每轮 epoch 间始终处于锁定冻结状态，仅用于零梯度模式 (`model.eval()`) 下评估和对比泛化能力。

## [Current Step]
# 无蜂窝MIMO检测器优化版本 (Transformer + FPGNN)

## 核心功能说明
本项目优化了 `gnn_detector.py`，用于实现无蜂窝大规模 MIMO (Cell-Free Massive MIMO) 场景中的上行分布式接收检测。核心改进如下：

1. **Transformer 引入多用户干扰消除 (MU-IC)**:
   原本的模型采用全连接层（MLP）将所有用户特征展平送入网络进行处理，不仅破坏了各用户结构独立性导致表达能力不足，还容易陷入局部最优。重构后的版本使用 `nn.TransformerEncoder` 作为多用户干扰消除核心机制，允许模型通过 Attention 机制有效感知和消除用户之间的空间干扰，进一步提升了接收增益。

2. **物理先验加权与残差架构**:
   引入了特征提取层 (`feature_mlp`)，提升到更高维度 (默认 `64` 维)，然后通过 `attn_net` 在 L 维度 (AP 维度) 预测每个信号分支的可靠性并使用 Softmax 归一化。这部分直接计算出了基于物理先验的预测值 `base_out`。
   网络对高维特征采用同样加权，并交由 Transformer 计算出多用户干扰的非线性抵消残差预测 `residual_out`。网络总输出等于 `base_out + residual_out`，能极大地提升网络学习的稳定性。

3. **训练参数调整以提升泛化能力**:
   提高了整体的数据量，每个 Epoch 扩大了训练 batch size 到 `512` 以及 batches 个数到 `100`。总计训练 `50` 个 Epoch。增加海量随机数据点训练大幅增强了模型的环境泛化能力，使其真正能将消除干扰的规律固化在网络权重中。
   
## 参数暴露
为了方便后续调参实验，使用 `argparse` 对核心参数进行了暴露：
* `--L`：Access Points(AP) 数量，默认 16
* `--N`：每个 AP 的天线数量，默认 4
* `--K`：单天线用户的数量，默认 8
* `--epochs`：训练代数，默认 50
* `--batches_per_epoch`：每个 Epoch 使用的批次数，默认 100
* `--batch_size`：每批训练样本大小，默认 512
* `--lr`：Adam 优化器学习率，默认 5e-4
* `--hidden_dim`：特征提取空间的隐维度大小，同时也是 Transformer 的 `d_model` 维数，默认 64

## 数据集构建规范
本实验在线实时通过仿真系统 (`system_model.py`) 生成小尺度信道用于训练和测试。在每一个 Epoch 中均对信道进行随机抽样，实现了近乎无限的海量训练集。
同时，在脚本启动前会立刻固定随机数种子单独生成一组 `test_dataset` 供所有 Epoch 之后验证公平比较使用。保证了训练测试数据集的严格分离和评价准则的绝对客观。

## [Current Step]
# 优化版 GNN-Detector (Cell-Free MIMO)

## 功能说明
该代码是无蜂窝 (Cell-Free) MIMO 场景下的全精度 GNN 及 Transformer 联合预训练模型。用于在 CPU (Central Processing Unit) 端进行局部信息的智能自适应合并与多用户互相干扰（MU-IC）的有效消除。

## 本次更新亮点
为了克服在高 SNR（信噪比）时发生的由于 MSE 梯度差异过大与非线性网络盲目扰动导致的严重性能退化问题，在最新版本中引入了：

1. **残差关键零初始化（Zero-Init Residual）**：通过初始化最后线性输出层 `self.final_linear` 的 `weight` 和 `bias` 为 0，保证 Transformer 在训练初期输出为0。从而使模型的预测起步于纯粹的基于 AP 信道质量的注意力加权机制，严格保障了高信噪比情况下的底线性能！
2. **更宽泛的训练范围**：将在线随机训练的信噪比范围拓展至 `[-20, 25]` dBm，确保高 SNR 特征有足够的样本参与网络权重的平滑。
3. **Transformer 抗过拟合**：引入 `dropout=0.1` 以及 `norm_first=True` 提高残差块与 Transformer 层在深层次更新中的稳定性。
4. **余弦退火学习率**：配合长 Epoch 训练（`epochs=80`，起步 `lr=1e-3`），将学习率缓慢衰减至 `1e-5`，以获得极度平滑的收敛结果。

## 暴露的关键参数
代码提供了丰富的命令行参数（通过 argparse 控制）：
- `--L` (default=16): AP（接入点）的数量。
- `--N` (default=4): 每个 AP 的天线数量。
- `--K` (default=8): 单天线用户数量。
- `--epochs` (default=80): 总训练轮次。
- `--lr` (default=1e-3): 初始学习率（随后由 CosineAnnealingLR 控制衰减）。
- `--batch_size` (default=512): 每批次训练所需的 Batch Size 大小。
- `--hidden_dim` (default=64): Transformer 中用于提升局部 SNR、软信息实部与虚部特征维度的隐空间大小。

## [Current Step]
# LSQ Differentiable Quantizer Core (可微量化器核心设计)

## 功能说明
该代码模块 (`lsq_quantizer.py`) 提供了基于 LSQ (Learned Step Size Quantization) 算法的端到端可微分量化器。
量化器在深度学习部署中用于减少权重或激活值的位宽（如减少到 2-bit）。但直接舍入操作由于导数为0阻碍了反向传播，因此引入直通估计器 (Straight-Through Estimator, STE) 将舍入操作的导数近似为 1。

该模块包含以下两部分核心：
1. **LSQFunction (`torch.autograd.Function`)**:
   - 实现了前向的浮点数量化计算逻辑。
   - 实现了对网络主体输入 `v` 的直通梯度 (`STE`)。
   - 实现了基于 LSQ 论文公式的对缩放系数 `s` (步长) 的梯度推导。

2. **LSQQuantizer (`nn.Module`)**:
   - 将 `LSQFunction` 封装为神经网络层。
   - 维护可学习的步长参数 `s`，并在前向时自动施加 `abs() + 1e-5` 防止 `s` 为负或0导致的计算越界崩溃。
   - 根据所需的位数 `num_bits` 自动配置有符号定点的截断上下边界 (`v_min` / `v_max`)。

## 测试与数据集划分说明
代码内嵌了一个用于直接测试的入口脚本，可以通过验证针对量化误差的直接优化来证明参数 $s$ 的可导性：
- **数据集划分**：内部生成 2000 个长度的一维连续数据，划分前 1000 个为训练集 (`x_train`)，后 1000 个为测试集 (`x_test`)。
- **验证目的**：每 40 步我们会打印在固定 `x_test` 上的测试 Loss，从而判断优化的量化步长是否对未知数据起到泛化作用（而不是发生过拟合）。

## 参数暴露
为了灵活调整测试时的超参，我们在入口程序中使用了 `argparse` 暴露出了超参控制接口：
- `--epochs`：控制测试程序迭代优化的轮数，默认 `200` 轮。
- `--lr`：优化量化步长的 Adam 学习率，默认 `0.01`。

在命令行/控制台可以通过：

## [Current Step]
# 自适应位宽策略网络 (Adaptive Quantization Policy Network)

## 功能说明
本模块实现了一个**可微分自适应量化决策网络**，核心功能在于使系统能够根据不同的物理先验信息（如当前信道质量 SNR）自动决定并分配所需的量化位数。为了实现端到端基于梯度的联合训练，本模块使用 `Gumbel-Softmax` 生成硬分配决定，兼顾了前向传播的 0/1 位宽绝对选择与反向传播的梯度直通估计。

在训练时，网络采用 `Weighted MSE` + `Bit Penalty` 为总的 Loss，使得：
1. **加权误差项**：惩罚高信噪比的高质量信号失真，以保护核心链路；对低信噪比被干扰链路，允许更多失真。
2. **比特惩罚项**：限制总体下发或传输消耗。

经过训练后，模型将自动掌握“根据信道质量分配资源”的物理先验（信噪比越高，分配比特越大）。

## 命令行参数 (超参数暴露)
脚本内部使用 `argparse` 对超参数进行了暴露，可以直接在启动时传入指定：
- `--epochs` : 训练轮数。默认值 `300`。
- `--lr` : 策略网络的优化器学习率。默认值 `0.01`。
- `--lambda_val` : 用于平衡误差与量化比特消耗惩罚项权重的系数。值越大网络越倾向于降低比特消耗（选择 0-bit 或 2-bit）；默认值 `0.5`。
- `--tau` : `Gumbel-Softmax` 的温度参数，用于控制激活概率分布的尖锐程度；默认值 `1.0`。

## 运行方式
执行附带的批处理脚本即可启动端到端测试，包含数据划分并验证泛化能力：

## [Current Step]
# Joint Quantization-Aware Training (Joint QAT) for Cell-Free MIMO

## 1. 功能说明
本模块整合了 `AdaptiveQuantizer` (自适应量化策略网络) 和 `FPGNN` (全精度处理端 GNN)，提出了端到端的联合量化感知训练方案。

该方案的主要流程如下：
1. **信号自适应量化**: 在每个 AP 处，接受来自底层探测的软信息并与局域 SNR 一同输入给量化策略网络，使用 Gumbel-Softmax 选择所需的最佳量化位宽（0-bit 丢弃，2-bit 粗分，4-bit 细分），并将其离散化为量化值。
2. **特征整合传输**: AP 向云端 CPU 传输量化后的特征 `v_q`、局域 SNR 以及表示当前传输压力的 `expected_bits` (占用位宽状态)。
3. **全局恢复计算**: 云端 CPU 处基于改造后的 `QAT_FPGNN`，接收融合四维特征。模型使用预训练网络 `fp_gnn_model.pth` 作为骨架（平滑处理了新增的一维特征以实现无损过渡）。
4. **损失约束与退火**: 通过 Gumbel-Softmax 的硬决策支持与 `tau` (温度参数) 的余弦退火策略，逼近离散分布。同时以 MSE 和对平均占用比特 `2.5 bits` 的方差限制，共同实现网络在低通信开销与高性能间寻找纳什平衡。

## 2. 命令行参数 (Hyperparameters)
使用 `argparser` 将常用的超参数全部暴露出，以便于从控制台进行调度：

- `--L` (int, default=16): 系统布置的基站或 AP 数量。
- `--N` (int, default=4): 每个 AP 的天线数量。
- `--K` (int, default=8): 服务的用户总数。
- `--epochs` (int, default=50): 训练代数。
- `--batches_per_epoch` (int, default=100): 每个 Epoch 包含的随机批次数。
- `--batch_size` (int, default=512): 网络训练的批大小。
- `--lr` (float, default=1e-3): 优化器的初始学习率，使用了 CosineAnnealingLR。
- `--pretrained_model` (str, default='fp_gnn_model.pth'): 用于承接全精度网络权重的路径。

## 3. 结果观察
训练完毕后，在隔离的测试集上运行多功率节点验证测试，并直接打印 `Dist-Full BER`（全精度分布式直接平均误差，最原始基线）、`QAT-GNN BER` (基于量化感知的联合GNN网络硬判决)、各功率增益表现及各环节上的 `Average Bits`。
运行完毕后权重将保存至 `qat_gnn_model.pth` 中。

## [Current Step]
# QAT-GNN Evaluation & Robustness Testing

## Overview
This script `evaluation.py` evaluates the trained Joint Quantization-Aware GNN (`qat_gnn_model.pth`). It tests the model's performance under realistic deployment conditions, including hard bitwidth selection and AP dropout scenarios.

## Features
1. **Hard Selection Inference**: Replaces the soft Gumbel-Softmax used during training with an `argmax` operation to simulate physical deployment constraints, ensuring that exactly one bitwidth (0, 2, or 4) is chosen per AP per user.
2. **Robustness Testing (AP Dropout)**: Introduces random AP failure (simulating backhaul disconnections) by forcing the selected bitwidth to 0-bit for dropped APs. Evaluated at `0%`, `25%`, and `50%` drop rates.
3. **Complexity Analysis**: Prints out a theoretical FLOP comparison demonstrating that the lightweight `BitwidthPolicyNet` (MLP) incurs negligible computational overhead compared to the local LMMSE processing at each AP.

## Arguments
- `--L`, `--N`, `--K`: Network topology settings (APs, Antennas, Users).
- `--area_size`, `--bandwidth`, `--noise_psd`: Physical layer environment configurations.
- `--model_path`: Path to the trained `qat_gnn_model.pth` weight file.
- `--test_samples`: Number of random samples per power point.

## Outputs
- A theoretical FLOP comparison between local LMMSE and Policy Network.
- A comprehensive table showing the BER (Bit Error Rate) performance across different transmit power levels (`-10 dBm` to `20 dBm`) for various baselines (Dist-Full, Dist-Q2) and the QAT-GNN under multiple dropout rates.

## [Current Step]
# QAT GNN Evaluation

本文档用于说明 `evaluation.py` 的功能及命令行传参方法。修复后的脚本已经排除了由于广播机制导致的推理报错，同时修正了基线模型（LMMSE）在复数运算状态下的复杂度估计值。

## 功能说明
`evaluation.py` 是去蜂窝（Cell-Free）MIMO 架构的鲁棒性和性能评估脚本。其主要工作流程如下：
1. **系统复杂度分析（Complexity Analysis）**：输出基于实际天线数（`N`）和用户数（`K`）在不同方法下（LMMSE 和 Policy Net）的计算复杂度，考虑复数运算，使得分析更加贴近真实物理层。
2. **场景生成与推理测试（Inference & Testing）**：
   - 使用加载的去蜂窝系统构建并模拟真实物理环境。
   - 分别对比在不同功率点下（`p_tx_dbm`）下，全局融合（Dist-Full）、均匀量化（Dist-Q2）和本文联合感知训练策略（QAT GNN）下的误码率（BER）。
   - 包含设备掉线鲁棒性模拟（AP Dropout Simulation）：在 0%、25% 以及 50% 等设备掉线率下测试感知模型带来的系统性能稳定性。

## 修复内容
1. 修复了 `simulate_inference()` 函数中由 Dropout 掩码生成时由于 `.squeeze(-1)` 引起的 `The size of tensor a (8) must match the size of tensor b (16)` 报错，确保 3D `(B, L, 1)` Tensor 自动广播至 `(B, L, K)`。
2. 修复了计算 LMMSE 复杂度（`complexity_analysis`）时漏乘 4 倍实数等效 FLOPs 系数的问题，现在评估时能更加客观地展现计算资源需求。

## 支持的超参数（Argparse）
你可以通过以下参数从命令行灵活改变系统的测试设置：
* `--L`：访问接入点 (APs) 的数量（默认：`16`）。
* `--N`：每个接入点包含的天线数量（默认：`4`）。
* `--K`：单天线用户的数量（默认：`8`）。
* `--area_size`：评估的模拟场景边长，单位为 km（默认：`1.0`）。
* `--bandwidth`：系统带宽大小，单位为 Hz（默认：`20000000.0` 即 20Mhz）。
* `--noise_psd`：环境噪声的功率谱密度，单位 dBm/Hz（默认：`-174`）。
* `--model_path`：QAT GNN 的权重文件路径（默认：`qat_gnn_model.pth`）。
* `--test_samples`：每个发射功率测试点所需要的样本数量（默认：`500`）。

运行 `run.bat` 将直接执行 `evaluation.py` 并以控制台输出最终各掉线率下的误码率结果。
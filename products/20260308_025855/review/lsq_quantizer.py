import torch
import torch.nn as nn
import argparse

class LSQFunction(torch.autograd.Function):
    """
    LSQ核心计算的前向与反向传播自定义 Function。
    利用 STE 传播被截断的输入梯度，同时提供可微分的缩放参数(scale s)梯度。
    """
    @staticmethod
    def forward(ctx, v, s, v_min, v_max):
        # 1. 除以步长 s
        v_div_s = v / s
        
        # 2. 截断到量化区间
        clipped = torch.clamp(v_div_s, v_min, v_max)
        
        # 3. 舍入并恢复尺度
        v_q = torch.round(clipped) * s
        
        # 4. 存储反向传播所需要的变量
        ctx.save_for_backward(v_div_s, s)
        ctx.v_min = v_min
        ctx.v_max = v_max
        
        return v_q

    @staticmethod
    def backward(ctx, grad_output):
        v_div_s, s = ctx.saved_tensors
        v_min = ctx.v_min
        v_max = ctx.v_max
        
        # 计算量化区间的掩码
        mask_in = (v_div_s >= v_min) & (v_div_s <= v_max)
        mask_below = (v_div_s < v_min)
        mask_above = (v_div_s > v_max)
        
        # 1. 计算对 v 的梯度 (STE)
        grad_v = grad_output * mask_in.float()
        
        # 2. 计算对 s 的梯度
        grad_s_elementwise = (torch.round(v_div_s) - v_div_s) * mask_in.float() + \
                             v_min * mask_below.float() + \
                             v_max * mask_above.float()
        # 将逐元素的梯度缩放相加并调整到与 s 相同的形状
        grad_s = (grad_output * grad_s_elementwise).sum().view_as(s)
        
        # 因为前向传播接受 4 个参数：v, s, v_min, v_max，
        # 所以我们需要返回对应 4 个输入的梯度。v_min 和 v_max 不需要梯度，返回 None。
        return grad_v, grad_s, None, None


class LSQQuantizer(nn.Module):
    """
    基于 LSQ 的可学习量化器层
    """
    def __init__(self, num_bits, init_s=1.0):
        super(LSQQuantizer, self).__init__()
        self.num_bits = num_bits
        # 定义可学习的步长参数 s
        self.s = nn.Parameter(torch.tensor([init_s], dtype=torch.float32))
        
        # 定义量化上下界，针对有符号整型
        self.v_min = -(2 ** (num_bits - 1))
        self.v_max = (2 ** (num_bits - 1)) - 1

    def forward(self, v):
        # 增加绝对值和极小数保护，防止 s 为负数或 0 造成不稳定的计算
        s_abs = torch.abs(self.s) + 1e-5
        
        # 调用自定义的 Function
        return LSQFunction.apply(v, s_abs, self.v_min, self.v_max)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test script for LSQ Differentiable Quantizer")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for scale parameter s')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (iterations) to run')
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(42)

    print("Generating random dataset...")
    # 随机生成输入数据 (模拟连续信号) 并进行数据集划分
    x_full = torch.randn(2000) * 2
    
    # 【核心要求：分割训练集和测试集，每个epoch训练的测试集必须相同】
    x_train = x_full[:1000]
    x_test = x_full[1000:]

    # 实例化一个 2-bit 量化器
    quantizer = LSQQuantizer(num_bits=2, init_s=1.0)
    print(f"Quantizer initialized: v_min = {quantizer.v_min}, v_max = {quantizer.v_max}")

    # 使用 MSE 作为损失函数，Adam作为优化器来优化步长参数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=args.lr)

    print("\n--- Starting optimization for step size s ---")
    for epoch in range(args.epochs):
        # 训练前向过程
        x_q = quantizer(x_train)
        loss = criterion(x_q, x_train)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 40 次迭代打印中间状态，观察优化效果
        if (epoch + 1) % 40 == 0:
            with torch.no_grad():
                # 计算并在固定的测试集上验证泛化性能
                test_q = quantizer(x_test)
                test_loss = criterion(test_q, x_test)
                
            grad_val = quantizer.s.grad.item() if quantizer.s.grad is not None else 0.0
            print(f"Iteration {epoch + 1}/{args.epochs}: "
                  f"Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}, "
                  f"s = {quantizer.s.item():.4f}, s.grad = {grad_val:.4f}")

    print("--- Testing complete ---")
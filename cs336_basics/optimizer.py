"""
Optimizers and Learning Rate Schedulers

本模块实现了训练神经网络所需的优化器和学习率调度器：
- AdamW: Adam 优化器的改进版本，使用解耦的权重衰减
- get_lr_cosine_schedule: 带预热的余弦学习率调度器

Author: PSX
Date: 2025-11-05
"""

import torch
from torch.optim import Optimizer
from typing import Iterable, Callable
import math

class AdamW(Optimizer):
    """
    AdamW 优化器
    
    AdamW 是 Adam 的改进版本，主要改进了权重衰减的实现方式。
    与 Adam 在损失函数中添加 L2 正则化不同，AdamW 直接在参数更新时
    应用权重衰减，这对于自适应学习率优化器更有效。
    
    算法流程（对于每个参数 θ）：
        1. 计算梯度: g_t = ∇L(θ_t)
        
        2. 更新一阶矩（动量）:
           m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        
        3. 更新二阶矩（自适应学习率）:
           v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        
        4. 偏差修正:
           m̂_t = m_t / (1 - β₁^t)
           v̂_t = v_t / (1 - β₂^t)
        
        5. 更新参数（包含权重衰减）:
           θ_{t+1} = θ_t - lr * [m̂_t / (√v̂_t + ε) + λ * θ_t]
                                                       ↑
                                                   权重衰减
    
    参数:
        params: 可迭代的参数或参数组字典
        lr: 学习率（默认: 1e-3）
        betas: (β₁, β₂) 用于计算梯度及其平方的移动平均的系数（默认: (0.9, 0.999)）
        eps: 添加到分母以提高数值稳定性的项（默认: 1e-8）
        weight_decay: 权重衰减系数（L2 惩罚）（默认: 0.01）
    
    状态变量（每个参数）:
        step: 当前时间步（从 1 开始）
        exp_avg: 梯度的指数移动平均（一阶矩 m_t）
        exp_avg_sq: 梯度平方的指数移动平均（二阶矩 v_t）
    
    示例:
        >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    
    参考:
        Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
        https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01
    ):
        """
        初始化 AdamW 优化器
        
        参数:
            params: 要优化的参数
            lr: 学习率
            betas: (β₁, β₂) 一阶和二阶矩的衰减率
            eps: 数值稳定性常数
            weight_decay: 权重衰减系数
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} - should be >= 0.0")
        
        defaluts = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        # 初始化父类 Optimizer，设置默认参数
        super().__init__(params, defaluts)
    
    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        执行单个优化步骤

        参数:
            closure: 重新计算模型输出并返回损失的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 获取超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历参数组中的每个参数
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # 状态初始化
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩
                    state["exp_avg"] = torch.zeros_like(p)
                    # 二阶矩
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                # 获取状态变量
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 更新一阶矩
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新二阶矩
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # 数学便捷运算, 避免了中间变量的创建
                # 计算步长
                step_size = lr / bias_correction1

                # 计算分母
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)


                # 权重衰减
                if weight_decay != 0:
                    p.add_(p, alpha=-weight_decay * lr)

                # 更新参数
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss




def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
) -> float:
    """
    带预热的余弦学习率调度
    """

    # =============================
    # 预热阶段
    # =============================
    if warmup_iters > 0 and it < warmup_iters:
        # 线性增长
        return max_learning_rate * it / warmup_iters
    # =============================
    # 余弦衰退
    # =============================
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        denom = cosine_cycle_iters - warmup_iters
        if denom <= 0:
            return min_learning_rate
        progress = (it - warmup_iters) / denom
        cos_decay = (1 + math.cos(math.pi * progress)) / 2
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cos_decay

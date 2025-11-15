# -*- coding: utf-8 -*-
# 文件：nn.utils.py
# 作者：PSX
# 描述：softmax, cross-entropy, gradient_clipping
# 日期：2025-10-25
from jaxtyping import Float, Int
from typing import Iterable
import torch
from torch import Tensor

#实现softmax函数
def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:

    # 保持稳定，减去指定维度上的最大值
    max_vals = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]

    # 计算指数
    exp_features = torch.exp(max_vals)

    # 计算和
    sum_exp_features = torch.sum(exp_features, dim=dim, keepdim=True)

    # 计算softmax
    softmax_features = exp_features / sum_exp_features

    return softmax_features


#
def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    
    max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
    shifted = inputs - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=1, keepdim=True))
    log_softmax = shifted - log_sum_exp

    correct_log_probs = log_softmax[torch.arange(inputs.size(0)), targets]
    cross_entropy_loss = -torch.mean(correct_log_probs)


    return cross_entropy_loss



def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:

    # 计算所有参数的L2范数的平方
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += (p.grad ** 2).sum().item()

    # 计算总L2范数
    total_norm = total_norm ** 0.5

    # 计算缩放因子
    if total_norm > max_l2_norm:
        scale_factor = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale_factor)
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional


@torch.no_grad()
def sample_next_token_from_logits(
    logits: Tensor,  # shape (batch_size,vocab_size)
    temperature: float = 1.0,
    top_p: Optional[float] = None,  # 可选的top-p采样,核采样
) -> Tensor:
    """
    对单步 logits 进行温度缩放 + nucleus(top-p) 采样，返回采样到的 token id (batch,)

    - temperature <= 0: 退化为贪心（argmax）

    - 0 < top_p < 1: nucleus 抽样；top_p>=1 则等价于普通按概率抽样

    参考：
      - Nucleus Sampling (top-p): Holtzman et al., 2020
      - torch.multinomial 用于从概率分布采样
    """

    # 贪心
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)  
    
    # 温度缩放
    logits = logits / float(temperature)
    
    # 转为概率
    probs = torch.softmax(logits, dim=-1)  

    if top_p is None or top_p >= 1.0:
        # 直接从整个词表抽样
        return torch.multinomial(probs, num_samples=1).squeeze(-1)  

    # 核采样
    # 1)降序排序
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  
    # 2)累计和
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 3)找到第一个累计和大于top_p的位置
    cutoff = (cumsum > top_p)
    cutoff[..., 1:] = cutoff[..., :-1].clone()  # 每个位置都大于top_p
    cutoff[..., 0] = False  # 第一个位置一定小于top_p
    nucleus_probs = sorted_probs.masked_fill(cutoff, 0.0) 
    # 4)归一化
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)  
    # 5)从核采样的概率分布中抽样,然后映射回原索引
    next_sorted = torch.multinomial(nucleus_probs, num_samples=1).squeeze(-1) 
    next_token = sorted_idx.gather(-1, next_sorted.unsqueeze(-1)).squeeze(-1)  
    return next_token # (batch_size,)



@torch.no_grad()
def generate(
    model: torch.nn.Module,  # 模型
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,  # 可选的top-p采样
    eos_id: Optional[int] = None,  # 可选的结束符id
    device: Optional[torch.device] = None,  # 可选的设备
) -> Tensor:  # (batch_size, context_length + max_new_tokens)
    """
    自回归生成：把每次模型最后一个位置的 logits 取出 → 经过温度+top-p 采样 → 追加。
    返回生成后的完整序列 (batch, seq_len + max_new_tokens)

    约定：model(input) → logits: (batch, seq_len, vocab_size)
    """
    if device is None:
        device = next(model.parameters()).device
    
    x = input_ids.to(device)
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(x)
        last_logits = logits[:, -1, :]  # (batch_size, context_length, vocab_size) -> (batch_size, vocab_size)

        next_token = sample_next_token_from_logits(last_logits, temperature, top_p)  # (batch_size, vocab_size) -> (batch_size,)

        x = torch.cat([x, next_token.unsqueeze(-1)], dim=-1)  # (batch_size, context_length) -> (batch_size, context_length + 1)

        if eos_id is not None:
            if torch.all(next_token == eos_id):  # 所有样本都生成了结束符，提前结束
                break
    
    return x


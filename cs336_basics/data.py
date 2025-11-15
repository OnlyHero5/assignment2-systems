# -*- coding: utf-8 -*-
# 文件：data.py
# 作者：PSX
# 描述：实现get_batch函数
# 日期：2025-10-26


from jaxtyping import Float, Int
from typing import Iterable
import torch
from torch import Tensor
import numpy as np
from numpy import typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # # 确保数据集的长度大于或等于批次大小乘以上下文长度
    # if len(dataset) < batch_size * context_length:
    #     raise ValueError("数据集长度不足，无法生成批次")
    
    # 确定最后一个x可能的起始索引
    last_possible_start_index = dataset.shape[0] - context_length - 1

    # 随机选择起始索引
    rng = np.random.default_rng()
    starts: npt.NDArray[np.int64] = rng.integers(0, last_possible_start_index + 1, size=batch_size, dtype=np.int64)

    # 构造索引
    offsets: npt.NDArray[np.int64] = np.arange(context_length, dtype=np.int64)[None, :]
    x_indices: npt.NDArray[np.int64] = starts[:, None] + offsets # shape (batch_size, context_length) 广播机制
    y_indices: npt.NDArray[np.int64] = x_indices + 1

    # 从数据集中获取x和y
    x: npt.NDArray[np.int64] = dataset[x_indices]
    y: npt.NDArray[np.int64] = dataset[y_indices]

    # 将x和y转换为PyTorch张量
    x_tensor: torch.Tensor = torch.as_tensor(x, dtype=torch.long, device=device)
    y_tensor: torch.Tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    return x_tensor, y_tensor
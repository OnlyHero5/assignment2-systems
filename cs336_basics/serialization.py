# -*- coding: utf-8 -*-
# 文件：serialization.py
# 作者：PSX
# 描述：模型的保存和读取
# 日期：2025-11-06
#
import os
import torch
from typing import BinaryIO, IO

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    """
    保存模型的检查点。
    Args:
        model (torch.nn.Module): 要保存的模型。
        optimizer (torch.optim.Optimizer): 要保存的优化器。
        iteration (int): 当前迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 保存路径或文件对象。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
    ) -> int:
    """
    从检查点加载模型和优化器的状态。
    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 加载路径或文件对象。
        model (torch.nn.Module): 要加载的模型。
        optimizer (torch.optim.Optimizer): 要加载的优化器。
    Returns:
        int: 加载的迭代次数。
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration
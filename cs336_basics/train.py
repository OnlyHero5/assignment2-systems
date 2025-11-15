from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式训练

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint, load_checkpoint

#========== 配置结构体 ===============
@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    num_heads: int
    d_ff : int

@dataclass
class OptimConfig:
    max_iters: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    min_lr: float
    warmup_iters: int
    cosine_iters: int
    grad_accum_steps: int   # 梯度累积步数
    weight_decay: float
    grad_clip: float

@dataclass
class DatasetPreset:
    name: str
    train_file: Path
    valid_file: Path
    tokenizer_dir: Path
    context_length: int
    batch_size: int
    model: ModelConfig
    optim: OptimConfig

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

PRESETS: Dict[str, DatasetPreset] = {
    "tinystories": DatasetPreset(
        name="tinystories",
        train_file=DATA_DIR / "TinyStoriesV2/train.npy",
        valid_file=DATA_DIR / "TinyStoriesV2/valid.npy",
        tokenizer_dir=DATA_DIR / "TinyStoriesV2",
        context_length=256,
        batch_size=64,
        model=ModelConfig(d_model=384, num_layers=4, num_heads=6, d_ff=1536),
        optim=OptimConfig(
            max_iters=20_000,
            eval_interval=200,
            eval_iters=200,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=500,
            cosine_iters=20_000,
            grad_accum_steps=1,
            weight_decay=0.1,
            grad_clip=1.0,
        )
    ),
    "owt": DatasetPreset(
        name="owt",
        train_file=DATA_DIR / "OWT/train.npy",
        valid_file=DATA_DIR / "OWT/valid.npy",
        tokenizer_dir=DATA_DIR / "OWT",
        context_length=512,
        batch_size=24,
        model=ModelConfig(d_model=768, num_layers=12, num_heads=12, d_ff=3072),
        optim=OptimConfig(
            max_iters=60_000,
            eval_interval=500,
            eval_iters=100,
            learning_rate=2e-4,
            min_lr=2e-5,
            warmup_iters=2000,
            cosine_iters=60_000,
            grad_accum_steps=6,
            weight_decay=0.1,
            grad_clip=1.0,
        )
    )
}

def _load_vocab_size(tokenizer_dir: Path) -> int:
    vocab_path = tokenizer_dir / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)

def _load_tokenizer(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"未找到{path}, 请先运行prepare_dataset_fast.py生成数据流")
    return np.load(path, mmap_mode="r")

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.memmap,
    device_str: str,
    preset: DatasetPreset
) -> float:
    model.eval()
    losses = []
    for _ in range(preset.optim.eval_iters):
        xb, yb = get_batch(dataset, preset.batch_size, preset.context_length, device_str)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))

#============训练主流程=================
def run_training(args: argparse.Namespace) -> None:
    preset = PRESETS[args.dataset]

    # -----------[DDP] 初始化分布式-----------
    distributed = args.distributed
    if distributed:
        rank = int(os.environ["RANK"])  # 获取当前进程的 rank（进程编号）
        world_size = int(os.environ["WORLD_SIZE"])  # 获取训练任务总进程数
        local_rank = int(os.environ["LOCAL_RANK"])  # 获取当前节点的 rank（节点内进程编号）
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU 设备
        device = torch.device(f"cuda:{local_rank}")  # 使用当前节点的 GPU 设备
        is_main = (rank == 0)
    else:
        rank, world_size, local_rank = 0, 1, 0  # 单节点训练
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        is_main = True
    
    device_str = str(device)
    _set_seed(args.seed + rank)

    train_tokens = _load_tokenizer(preset.train_file)
    valid_tokens = _load_tokenizer(preset.valid_file)
    vocab_size = _load_vocab_size(preset.tokenizer_dir)

   
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=preset.context_length,
        d_model=preset.model.d_model,
        num_layers=preset.model.num_layers,
        num_heads=preset.model.num_heads,
        d_ff=preset.model.d_ff,
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    optimizer = AdamW(
        model.parameters(),
        lr=preset.optim.learning_rate,
        weight_decay=preset.optim.weight_decay,
    )

    device_type = device.type
    # TODO: 兼容性
    amp_enabled = args.use_amp and device_type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    except (AttributeError, TypeError):
        # 方式 2: 兼容旧版本（PyTorch < 2.0）
        if device_type == "cuda":
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=False)
    def autocast_ctx():
        if device_type == "cuda":
            return torch.cuda.amp.autocast(enabled=amp_enabled)
        else:
            return nullcontext()
    
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.run_name}.pt"

    start_iter = 0
    best_val = float("inf")
    if args.resume and Path(args.resume).exists():
        start_iter = load_checkpoint(args.resume, model, optimizer)
        if is_main:
            print(f"从{args.resume}恢复训练, 继续迭代 {start_iter}")
    
    if args.eval_only:
        if is_main:
            val_loss = evaluate(model, valid_tokens, device_str, preset)
            print(f"[eval_only] val_loss={val_loss:.4f}, val_ppl={math.exp(val_loss):.2f}")
        
        if distributed:
            dist.destroy_process_group()  # 销毁进程组
        return
    
    t0 = time.time()
    for it in range(start_iter, preset.optim.max_iters):
        lr = get_lr_cosine_schedule(
            it,
            preset.optim.learning_rate,
            preset.optim.min_lr,
            preset.optim.warmup_iters,
            preset.optim.cosine_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        optimizer.zero_grad()

        for _ in range(preset.optim.grad_accum_steps):
            xb, yb = get_batch(train_tokens, preset.batch_size, preset.context_length, device_str)
            with autocast_ctx():
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                loss = loss / preset.optim.grad_accum_steps
            scaler.scale(loss).backward()
        
        if preset.optim.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), preset.optim.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        if is_main and it % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"iter {it:06d}/{preset.optim.max_iters} world_size={world_size} | lr={lr:.2e} | loss={loss.item()* preset.optim.grad_accum_steps:.4f} | {elapsed:.1f}s")
        
        if is_main and it % preset.optim.eval_interval == 0 or it == preset.optim.max_iters - 1:
            train_loss = evaluate(model, train_tokens, device_str, preset)
            val_loss = evaluate(model, valid_tokens, device_str, preset)
            print(f"[eval] it={it} train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                to_save = model.module if distributed else model
                save_checkpoint(to_save, optimizer, it, ckpt_path)
                print(f"[save] 保存到{ckpt_path}")
    
    if is_main:
        print(f"训练完成, 最佳验证集损失={best_val:.4f}")
    
    if distributed:
        dist.destroy_process_group()  # 销毁进程组
        
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PRESETS.keys(), default="owt")
    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--distributed", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(args)


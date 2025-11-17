"""
端到端的benchmark
"""
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from cs336_systems.models import get_basics_transformer, count_parameters



def benchmark_model(
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_warmup: int = 5,
        num_iters: int = 10,
        backward: bool = False,
        device: str = "cuda",
) -> Dict[str, float]:
    """
    Args:
        model: 要测试的模型
        batch_size: 批次大小
        seq_len: 序列长度
        vocab_size: 词汇表大小
        num_warmup: 预热迭代次数
        num_iters: 要测量的迭代次数
        backward: 是否包含后向传递
        device: 运行设备

    Returns:
        包含计时统计信息的字典
    """
    model.eval()

    inputs_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if backward:
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        criterion = nn.CrossEntropyLoss()
        model.train()
    
    # 热启动
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad() if not backward else torch.enable_grad():
            logits = model(inputs_ids)

            if backward:
                loss = criterion(logits.view(-1, vocab_size), target.view(-1))
                model.zero_grad()
                loss.backward()
        
        torch.cuda.synchronize()
    # 正式测量阶段
    print(f"Running {num_iters} measurement iterations...")
    forward_times = []
    backward_times = []
    total_times = []

    for i in range(num_iters):
        if backward:
            model.zero_grad()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = model(inputs_ids)

        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_time
        forward_times.append(forward_time)

        if backward:
            loss = criterion(logits.view(-1, vocab_size), target.view(-1))

            torch.cuda.synchronize()
            backward_start = time.perf_counter()

            loss.backward()

            torch.cuda.synchronize()
            backward_time = time.perf_counter() - backward_start
            backward_times.append(backward_time)

            total_time = forward_time + backward_time
            total_times.append(total_time)
        
        if (i+1) % 5 == 0:
            print(f" Completed {i+1}/{num_iters} iterations")
    
    results = {
        "forward_mean": np.mean(forward_times) * 1000,
        "forward_std": np.std(forward_times) * 1000,
    }
    if backward:
        results.update({
            "backward_mean": np.mean(backward_times) * 1000,
            "backward_std": np.std(backward_times) * 1000,
            "total_mean": np.mean(total_times) * 1000,
            "total_std": np.std(total_times) * 1000
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models")

    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "2.7B"],
        help="Model size configuration"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch-size"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="vocabulary size"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Precision for computation"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of measurement iterations"
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Include backward pass in benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    dtype = torch.float32 if args.precision == "fp32" else ( torch.bfloat16 if args.precision == "bf16" else torch.float16)

    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print(f"Model size:          {args.model_size}")
    print(f"Sequence length:     {args.seq_len}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Vocab size:          {args.vocab_size}")
    print(f"Precison:            {args.precision}")
    print(f"Warmup iters:        {args.num_warmup}")
    print(f"Measure iters:       {args.num_iters}")
    print(f"Backward pass:       {args.backward}")
    print(f"Device:              {args.device}")
    print("="*60)
    print()

    print("Creating model ...")
    model = get_basics_transformer(
        size=args.model_size,
        context_length=args.seq_len,
        vocab_size=args.vocab_size,
        dtype=dtype,
        device=args.device
    )

    total_params = count_parameters(model)['total']
    print(f"Total parameters: {total_params:,}")
    print()

    results = benchmark_model(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        backward=args.backward,
        device=args.device
    )

    print()
    print("="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Forward: {results['forward_mean']:.2f} ms ± {results['forward_std']:.2f} ms")

    if args.backward:
        print(f"Backward: {results['backward_mean']:.2f} ms ± {results['backward_std']:.2f} ms")
        print(f" Total:  {results['total_mean']:.2f} ms ± {results['total_std']:.2f} ms")
    
    print("="*60)


if __name__ == "__main__":
    main()





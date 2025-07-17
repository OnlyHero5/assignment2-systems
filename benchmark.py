from contextlib import contextmanager, nullcontext
import dataclasses
import itertools
import statistics
import timeit
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

from jaxtyping import Float

from optimizers import AdamW, AdamWParams
import pandas as pd

import torch
import torch.cuda.nvtx as nvtx

from llm import attention, LMHyperparams, TransformerLM
from train import CachingTextFileLoader, load_params, load_sweep_params, TokenizerLoader
from training import cross_entropy

@dataclasses.dataclass
class BenchmarkParams:
    operation: str
    device: str
    n_warmup: int
    n_runs: int
    batch_size: int = 32
    include_backward: bool = True
    include_optimizer: bool = False
    autocast_dtype: Optional[str] = None

compiled_attention = torch.compile(attention)
MetricsList = List[Tuple[str, float]]

@contextmanager
def timer_context(operation: str, metrics: Optional[MetricsList] = None, nvtx_range: Optional[str] = None):
    if nvtx_range is not None:
        nvtx_context = nvtx.range(nvtx_range)
    else:
        nvtx_context = nullcontext()

    with nvtx_context:
        try:
            start = timeit.default_timer()
            yield None
        finally:
            torch.cuda.synchronize()
            end = timeit.default_timer()
            if metrics is not None:
                metrics.append((operation, end - start))


def run_single_benchmark(benchmark_params: BenchmarkParams, run: Callable[[], MetricsList]) -> pd.DataFrame:
    with nvtx.range("warmup"):
        for _ in range(benchmark_params.n_warmup):
            run()

    with nvtx.range("real runs"):
        metrics = itertools.chain(*(run() for _ in range(benchmark_params.n_runs)))

    return pd.DataFrame.from_records({
        "action": action,
        "t": t,
    } for action, t in metrics)


class FullModelRun:
    def __init__(self, benchmark_params, hp, optimizer_params, data_loader):
        self.hyperparams = hp
        self.optimizer_params = optimizer_params
        self.data_loader = data_loader

        self.model = TransformerLM(self.hyperparams)
        self.optimizer = None
        if self.optimizer_params is not None:
            self.optimizer = AdamW(self.model.parameters(), self.optimizer_params)

    def __call__(self) -> MetricsList:
        metrics = []
        samples, targets = data_loader.create_training_batch(params.batch_size, self.model.params.context_length, params.device)
        self.model.train()
        self.model.zero_grad()

        with nvtx.range("full run"):
            with timer_context("forward", metrics, "forward pass"):
                predictions = self.model(samples)

            if params.include_backward:
                with timer_context("backward", metrics, "backward pass"):
                    loss = cross_entropy(predictions, targets).mean()
                    loss.backward()

                if self.optimizer is not None:
                    with timer_context("optimize", metrics, "optimizer step"):
                        self.optimizer.step()

        return metrics


class AttentionRun:
    def __init__(self, benchmark_params, hyperparams: LMHyperparams):
        self.benchmark_params = benchmark_params
        self.hyperparams = hyperparams

        self.attention_fn = compiled_attention if compile else attention

    def __call__(self) -> MetricsList:
        metrics = []

        def create_input():
            return torch.rand(self.hyperparams.context_length, self.hyperparams.d_model, requires_grad=self.benchmark_params.include_backward)

        with timer_context("forward", metrics):
            loss = self.attention_fn(*(create_input() for _ in range(3))).mean()
        
        if self.benchmark_params.include_backward:
            with timer_context("backward", metrics, "attention backward"):
                loss.backward()

        return metrics


def run_benchmarks(benchmark_params: BenchmarkParams, all_hyperparams: Iterable[LMHyperparams], swept_fields: List[str], optimizer_params, data_loader):
    all_runs = []

    dtype_context = nullcontext()
    if benchmark_params.autocast_dtype is not None:
        dtype_context = torch.autocast(device_type=benchmark_params.device, dtype=getattr(torch, benchmark_params.autocast_dtype))

    with dtype_context:
        for hp in all_hyperparams:
            annotations = ", ".join(f"{field}={getattr(hp, field)}" for field in swept_fields)

            if benchmark_params.operation == "full":
                runner = FullModelRun(benchmark_params, hp, optimizer_params, data_loader)
            elif benchmark_params.operation == "attention":
                runner = AttentionRun(benchmark_params, hp)
            else:
                raise ValueError("unknown benchmark type")

            try:
                with nvtx.range(f"benchmark ({annotations})"):
                    hp_runs = run_single_benchmark(benchmark_params, runner)
            except torch.OutOfMemoryError:
                print(f"configuration {annotations} caused CUDA OOM")
            else:
                for field in swept_fields:
                    hp_runs[field] = getattr(hp, field)

                all_runs.append(hp_runs)

    df = pd.concat(all_runs)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("operation", choices=["full", "attention"], default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--hyperparams", nargs="?", default="hyperparams.json")
    parser.add_argument("--optimizer-params", nargs="?", default="optimizer.json")
    parser.add_argument("--training-data", default="TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--tokenizer-path", default="tokenizer.pkl")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--include-optimizer", action="store_true")
    parser.add_argument("--autocast")
    parser.add_argument("--device")

    args = parser.parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_default_device(device)

    tokenizer = TokenizerLoader.load(args.tokenizer_path, args.training_data, 10_000)
    data_loader = CachingTextFileLoader(tokenizer, args.training_data)

    all_hyperparams = list(load_sweep_params(LMHyperparams, args.hyperparams))

    optimizer_params = None
    if args.include_optimizer:
        optimizer_params = load_params(AdamWParams, args.optimizer_params)

    params = BenchmarkParams(
        operation=args.operation,
        device=device,
        n_warmup=args.warmup,
        n_runs=args.n_run,
        include_backward=args.include_backward,
        include_optimizer=args.include_optimizer,
        autocast_dtype=args.autocast,
    )

    swept_fields = [
        field.name for field in dataclasses.fields(LMHyperparams) if
        len(set(getattr(h, field.name) for h in all_hyperparams)) > 1
    ]

    results = run_benchmarks(params, all_hyperparams, swept_fields, optimizer_params, data_loader)

    print(results.pivot_table(index=swept_fields, columns=["action"], values=["t"]))
    print(results.pivot_table(index=swept_fields, columns=["action"], values=["t"], aggfunc=statistics.stdev))
from collections import defaultdict
import dataclasses
import statistics
import timeit

import pandas as pd

import torch
import torch.cuda.nvtx as nvtx

from llm import LMHyperparams, TransformerLM
from train import CachingTextFileLoader, load_sweep_params, TokenizerLoader
from training import cross_entropy

@dataclasses.dataclass
class BenchmarkParams:
    device: str
    n_warmup: int
    n_runs: int
    batch_size: int = 32
    include_backward: bool = True

def run_single_benchmark(benchmark_params: BenchmarkParams, hyperparams, data_loader) -> pd.DataFrame:
    model = TransformerLM(hyperparams)

    with nvtx.range("warmup"):
        for _ in range(benchmark_params.n_warmup):
            samples, _ = data_loader.create_training_batch(benchmark_params.batch_size, hyperparams.context_length, benchmark_params.device)
            model(samples)
        torch.cuda.synchronize()

    metrics = []
    for _ in range(benchmark_params.n_runs):
        model.train()
        model.zero_grad()

        samples, targets = data_loader.create_training_batch(benchmark_params.batch_size, hyperparams.context_length, benchmark_params.device)
        start = timeit.default_timer()
        predictions = model(samples)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        metrics.append(("forward", end - start))

        if benchmark_params.include_backward:
            start = timeit.default_timer()
            loss = cross_entropy(predictions, targets).mean()
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            metrics.append(("backward", end - start))

    return pd.DataFrame.from_records({
        "action": action,
        "t": t,
    } for action, t in metrics)

def run_benchmarks(benchmark_params: BenchmarkParams, all_hyperparams, data_loader):
    swept_fields = [
        field.name for field in dataclasses.fields(LMHyperparams) if
        len(set(getattr(h, field.name) for h in all_hyperparams)) > 1
    ]

    all_runs = []
    for hp in all_hyperparams:
        hp_runs = run_single_benchmark(benchmark_params, hp, data_loader)
        for field in swept_fields:
            hp_runs[field] = getattr(hp, field)
        all_runs.append(hp_runs)
    df = pd.concat(all_runs)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--hyperparams", nargs="?", default="hyperparams.json")
    parser.add_argument("--training-data", default="TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--tokenizer-path", default="tokenizer.pkl")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--device")

    args = parser.parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_default_device(device)

    tokenizer = TokenizerLoader.load(args.tokenizer_path, args.training_data, 10_000)
    data_loader = CachingTextFileLoader(tokenizer, args.training_data)

    all_hyperparams = list(load_sweep_params(LMHyperparams, args.hyperparams))
    params = BenchmarkParams(
        device=device,
        n_warmup=args.warmup,
        n_runs=10,
        include_backward=args.include_backward,
    )

    results = run_benchmarks(params, all_hyperparams, data_loader)

    print(results.pivot_table(index=["d_model", "n_layers"], columns=["action"], values=["t"]))
    print(results.pivot_table(index=["d_model", "n_layers"], columns=["action"], values=["t"], aggfunc=statistics.stdev))
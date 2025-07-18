from abc import ABC, abstractmethod
import dataclasses
import hashlib
import itertools
import json
import logging
import os
from statistics import mean
import sys
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, TypeVar

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import wandb

from generation import generate_text
from llm import LMHyperparams, TransformerLM
from optimizers import AdamW, AdamWParams, clip_gradients
from tokenizer import Tokenizer, BPETokenizer, TokenizerTrainer
from training import load_checkpoint, save_checkpoint, create_training_batch, cross_entropy

class DataLoader(ABC):
    @abstractmethod
    def create_training_batch(self, batch_size: int, context_length: int, device: str) -> Tuple[Tensor, Tensor]:
        pass

class TokenizerLoader:
    def __init__(self, cache_path: os.PathLike, training_path: os.PathLike, vocab_size: int):
        self.cache_path = cache_path
        self.training_path = training_path
        self.vocab_size = vocab_size

    @classmethod
    def load(cls, cache_path: os.PathLike, training_path: os.PathLike, vocab_size: int):
        loader = cls(cache_path, training_path, vocab_size)
        return loader.get()

    def get(self) -> Tokenizer:
        tokenizer: Tokenizer
        if os.path.exists(self.cache_path):
            print(f"loading tokenizer from {self.cache_path}")
            tokenizer = BPETokenizer.from_disk(self.cache_path)
            if len(tokenizer.vocab) != self.vocab_size:
                print(f"cached tokenizer has vocab size {len(tokenizer.vocab)}, expected {self.vocab_size}")
        else:
            print(f"training new tokenizer with vocab size {self.vocab_size} on training data {self.training_path}")
            special_tokens: Set[str] = { "<|endoftext|>" }
            trainer = TokenizerTrainer(self.vocab_size, special_tokens=special_tokens)
            tokenizer_params = trainer.train_on_file(self.training_path)
            print("tokenizer training complete")
            tokenizer = BPETokenizer(tokenizer_params, special_tokens=special_tokens)
            tokenizer.to_disk(self.cache_path)
            print(f"tokenizer saved to {self.cache_path}")
        tokenizer.silent = True

        return tokenizer

class CachingTextFileLoader(DataLoader):
    def __init__(self, tokenizer: Tokenizer, text_path: str, np_cache_path: Optional[str] = None):
        self.tokenizer = tokenizer
        self.text_path = text_path
        self.np_cache_path = np_cache_path or f"{self.text_path}.tokens.npy"
        self._data: Optional[np.typing.NDArray] = None

    def create_training_batch(self, batch_size: int, context_length: int, device: str) -> Tuple[Tensor, Tensor]:
        if self._data is None:
            self._data = self._load()
        return create_training_batch(self._data, batch_size, context_length, device)

    def _build_cache(self) -> None:
        print(f"tokenizing data from {self.text_path} to {self.np_cache_path}...")
        with open(self.text_path, "r") as text_file:
            # XXX stream rather than read into memory
            tokens: List[int] = self.tokenizer.encode(text_file.read())
            cache_file = np.memmap(self.np_cache_path, dtype=np.int32, mode="w+", shape=(len(tokens),))
            cache_file[:] = tokens
            del cache_file
        print(f"saved token cache to {self.np_cache_path}")

    def _load(self) -> np.typing.NDArray:
        if not os.path.exists(self.np_cache_path):
            self._build_cache()
        return np.memmap(self.np_cache_path, dtype=np.int32, mode="r")


@dataclasses.dataclass
class TrainingParams:
    device: str
    batch_size: int
    steps_per_checkpoint: int
    steps_per_validation: int
    steps_per_sample_output: int
    checkpoint_root: os.PathLike
    training_loader: DataLoader
    validation_loader: DataLoader
    debugging: bool = False
    custom_run_id: Optional[str] = None
    sample_prompt: str = "Once upon a time"
    temperature: Optional[float] = None
    nucleus_size: Optional[float] = None

class TrainingRun:
    def __init__(self,
                 logger: logging.Logger,
                 tokenizer: Tokenizer,
                 hyperparams: LMHyperparams,
                 optimizer_params: AdamWParams,
                 training_params: TrainingParams,
                 code_version: Optional[str] = None,
                 ):
        self.logger = logger


        self.tokenizer = tokenizer
        self.hyperparams = hyperparams
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.code_version = code_version or "0"

        self.logger.info(f"using run id {self._get_run_id()}")

        self.model: TransformerLM = TransformerLM(self.hyperparams).to(self.training_params.device)
        if "mps" in self.training_params.device:
            self.model = torch.compile(self.model, backend="aot_eager") # type: ignore

        self.optimizer = AdamW(self.model.parameters(), self.optimizer_params)
        self.checkpoint_dir = os.path.join(self.training_params.checkpoint_root, self._get_run_id())
        self.logger.info(f"using checkpoint directory {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            self.logger.info(f"creating checkpoint directory {self.checkpoint_dir}")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger.info(f"loading from checkpoint directory {self.checkpoint_dir}")
        self.last_checkpoint = self._get_latest_checkpoint_file()
        if self.last_checkpoint is None:
            self.logger.warning("no checkpoint file found")
            self.next_iteration = 1
        else:
            self.next_iteration = self._get_checkpoint_number(self.last_checkpoint) + 1
            self.logger.info(f"loading iteration checkpoint from {self.last_checkpoint}, {self.next_iteration=}")
            load_checkpoint(self.last_checkpoint, self.model, self.optimizer)

    # XXX: refactor all checkpoint loading/saving into interface so that we can replace with other datastores
    def _params_key(self, params) -> str:
        return "|".join(
            f"{param}={str(getattr(params, param))}"
            for param in
            sorted(field.name for field in dataclasses.fields(params))
        )

    def _params_path(self) -> str:
        return "||".join(self._params_key(p) for p in (self.hyperparams, self.optimizer_params))

    def _get_run_id(self) -> str:
        if self.training_params.custom_run_id is not None:
            return self.training_params.custom_run_id
        run_details: str = f"{self._params_path()}/{self.code_version}"
        return hashlib.sha256(run_details.encode(encoding="utf-8")).hexdigest()

    def _get_checkpoint_number(self, path: str) -> int:
        return int(os.path.basename(path).split(".")[0])

    def _get_checkpoint_file(self, n: int) -> str:
        return os.path.join(self.checkpoint_dir, f"{n}.ckpt")

    def _get_checkpoint_files(self) -> Iterator[str]:
        # XXX use explicit metadata file for this instead
        for entry in os.listdir(self.checkpoint_dir):
            full_path = os.path.join(self.checkpoint_dir, entry)
            if not os.path.isfile(full_path):
                continue

            try:
                _ = self._get_checkpoint_number(full_path)
            except ValueError:
                continue

            yield full_path

    def _get_latest_checkpoint_file(self) -> Optional[str]:
        latest_checkpoint: Optional[int] = None
        latest_checkpoint_file: Optional[str] = None
        for checkpoint_file in self._get_checkpoint_files():
            try:
                checkpoint = self._get_checkpoint_number(checkpoint_file)
            except ValueError:
                continue

            if latest_checkpoint is None or checkpoint > latest_checkpoint:
                latest_checkpoint = checkpoint
                latest_checkpoint_file = checkpoint_file

        return latest_checkpoint_file

    def _save_checkpoint(self) -> None:
        next_checkpoint: str = self._get_checkpoint_file(self.next_iteration)
        self.logger.info(f"writing checkpoint {self.next_iteration} to {next_checkpoint}")
        save_checkpoint(self.model, self.optimizer, self.next_iteration, next_checkpoint)

    def generate_text(self, prompt: str, max_tokens: int) -> str:
        kwargs = {}

        if self.training_params.temperature is not None:
            kwargs["temperature"] = self.training_params.temperature
        if self.training_params.nucleus_size is not None:
            kwargs["top_p"] = self.training_params.nucleus_size

        completion: str = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens,
            device=self.training_params.device,
            **kwargs)
        return f"{prompt} {completion}"

    def _do_eval(self) -> Dict[str, Any]:
        self.model.eval()

        with torch.no_grad():
            samples, targets = self.training_params.validation_loader.create_training_batch(
                self.training_params.batch_size,
                self.hyperparams.context_length,
                self.training_params.device,
            )

            predictions = self.model(samples)
            loss = cross_entropy(predictions, targets).mean().item()

        return {
            "validation_loss": loss,
        }

    def _get_project_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "code_version": self.code_version,
        }

        for c in (self.hyperparams, self.optimizer_params, self.training_params):
            for field, value in dataclasses.asdict(c).items():
                config[field] = value

        return config

    def _get_model_stats(self) -> Dict[str, float]:
        stats = {}

        ff_params = list(itertools.chain.from_iterable(
            (layer.feed_forward.w1, layer.feed_forward.w2, layer.feed_forward.w3) for layer in self.model.transformers
        ))
        param_groups = {
            "q": [layer.attention.weights_q for layer in self.model.transformers],
            "k": [layer.attention.weights_k for layer in self.model.transformers],
            "v": [layer.attention.weights_v for layer in self.model.transformers],
            "ff": ff_params,
        }

        for group, params in param_groups.items():
            stats.update({
                f"{group}_norm": mean(torch.norm(p).item() for p in params),
                f"{group}_grad": mean(p.grad.abs().mean().item() for p in params),
                f"{group}_grad_max": mean(p.grad.abs().max().item() for p in params),
            })

        return stats

    def train_epochs(self, target_steps: int) -> None:
        if self.training_params.debugging:
            self._train_epochs_impl(target_steps, None)
        else:
            resume_behavior = "must" if self.next_iteration > 1 else "never"
            with wandb.init(project="cs336-a1", config=self._get_project_config(), resume=resume_behavior, id=self._get_run_id()) as run:
                run.define_metric("*", step_metric="total_tokens_trained")
                self._train_epochs_impl(target_steps, run)

    def _train_epochs_impl(self, target_steps: int, wandb_run) -> None:
        batch_tokens_trained = self.training_params.batch_size * self.hyperparams.context_length
        n_steps = (target_steps - self.next_iteration) + 1
        for _ in range(n_steps):
            start_time = perf_counter()

            # XXX log this in checkpoints so that we don't have to rely on the assumption of a constant
            # batch size throughout the entire run
            total_tokens_trained = batch_tokens_trained * (self.next_iteration + 1)

            step_stats: Dict[str, Any] = {
                "step": self.next_iteration,
                "total_tokens_trained": total_tokens_trained,
            }

            self.model.train()
            self.optimizer.zero_grad()

            samples, targets = self.training_params.training_loader.create_training_batch(
                self.training_params.batch_size,
                self.hyperparams.context_length,
                self.training_params.device
            )

            predictions = self.model(samples)

            batch_loss = cross_entropy(predictions, targets).mean()
            batch_loss.backward()

            step_stats.update(self._get_model_stats())

            # TODO: record weights before optimizer step, and log l2 norm of deltas for each param type?
            self.optimizer.step()

            end_time = perf_counter()

            seconds_elapsed = end_time - start_time
            
            step_stats.update({
                "training_loss": batch_loss.item(),
                "seconds_elapsed": seconds_elapsed,
                "tokens_per_second": batch_tokens_trained / seconds_elapsed,
                "learning_rate": self.optimizer.get_learning_rate(self.next_iteration),
            })

            if self.next_iteration % self.training_params.steps_per_validation == 0:
                step_stats.update(self._do_eval())

            if wandb_run is not None:
                wandb_run.log(step_stats)

            self.logger.info(f"finished step {self.next_iteration}")
            if self.next_iteration % self.training_params.steps_per_checkpoint == 0:
                self._save_checkpoint()

            if self.training_params.steps_per_sample_output > 0 and self.next_iteration % self.training_params.steps_per_sample_output == 0:
                if "validation_loss" in step_stats:
                    self.logger.info(f"validation loss = {step_stats['validation_loss']}")
                tokens: int = 100
                self.logger.info(f"generating {tokens} tokens of sample text:")
                self.logger.info(self.generate_text(self.training_params.sample_prompt, tokens))

            self.next_iteration += 1

    def output_eval(self) -> None:
        pass


T = TypeVar("T")
def load_params(cls: Type[T], path: str) -> T:
    with open (path, "r") as param_file:
        return cls(**dict(json.load(param_file))) # type: ignore


T = TypeVar("T")
def load_sweep_params(cls: Type[T], path: str) -> Iterator[T]:
    with open (path, "r") as param_file:
        values = json.load(param_file)
        options = []
        for k, vs in values.items():
            if isinstance(vs, dict):
                raise NotImplementedError()
            elif isinstance(vs, list):
                # XXX assume this is a list of scalar values all of the same type
                raw_values = vs
            else:
                raw_values = [vs]

            options.append([(k, v) for v in raw_values])

        for arg_list in itertools.product(*options):
            kwargs = dict(arg_list)
            yield cls(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # XXX change these to be subparsers
    parser.add_argument("action", default="train", choices=["train", "generate"])
    parser.add_argument("n_step", type=int)
    parser.add_argument("hyperparams_path", nargs="?", default="hyperparams.json")
    parser.add_argument("optimizer_params_path", nargs="?", default="optimizer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-data", default="TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--validation-data", default="TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--tokenizer-path", default="tokenizer.pkl")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=1)
    parser.add_argument("--sample-output-steps", type=int, default=0)
    parser.add_argument("--run-id", nargs="?")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--nucleus-size", type=float)

    # one-off param overrides
    parser.add_argument("--learning-rate", type=float)

    args = parser.parse_args()

    # https://github.com/pytorch/pytorch/issues/149184
    # dynamo instantiation is broken on mps
    if "mps" in args.device:
        torch._dynamo.disable()

    torch.set_default_device(args.device)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    hyperparams = load_params(LMHyperparams, args.hyperparams_path)
    optimizer_params = load_params(AdamWParams, args.optimizer_params_path)

    if args.learning_rate is not None:
        optimizer_params.learning_rate = args.learning_rate

    tokenizer = TokenizerLoader.load(args.tokenizer_path, args.training_data, hyperparams.vocab_size)

    training_params = TrainingParams(
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_root=args.checkpoint_root,
        training_loader=CachingTextFileLoader(
            tokenizer,
            text_path=args.training_data,
            # XXX allow cache path to be configured separately
            np_cache_path=f"{args.training_data}.tokens.npy",
        ),
        validation_loader=CachingTextFileLoader(
            tokenizer,
            text_path=args.validation_data,
            np_cache_path=f"{args.validation_data}.tokens.npy",
        ),
        steps_per_checkpoint=args.checkpoint_steps,
        steps_per_validation=args.validation_steps,
        steps_per_sample_output=args.sample_output_steps,
        custom_run_id=args.run_id,
        sample_prompt=args.prompt,
        temperature=args.temperature,
        nucleus_size=args.nucleus_size,
        debugging=args.debugging,
    )

    trainer = TrainingRun(logger, tokenizer, hyperparams, optimizer_params, training_params, code_version="28")

    if args.action == "train":
        trainer.train_epochs(args.n_step)
    elif args.action == "generate":
        print(trainer.generate_text(args.prompt, 100))

import dataclasses
import math
from typing import Any, Callable, Dict, Iterable, Optional, Self, Tuple, TypedDict, Union
import torch

class CosineScheduleParams(TypedDict):
    lr_min: float
    lr_max: float
    n_warmup: int
    n_anneal: int

@dataclasses.dataclass
class AdamWParams:
    learning_rate: float
    b1: float
    b2: float
    weight_decay: float
    eps: float
    gradient_clip: Optional[float] = None
    cosine_schedule: Optional[CosineScheduleParams] = None


def get_cosine_lr_schedule(t: int, lr_range: Tuple[float, float], n_warmup: int, n_anneal: int) -> float:
    lr_min, lr_max = lr_range
    if t < n_warmup:
        return (t / n_warmup) * lr_max
    elif t <= n_anneal:
        cos_steps: int = t - n_warmup
        total_cos_steps: int = n_anneal - n_warmup
        cos_progress: float = cos_steps / total_cos_steps
        scale_factor: float = (1 + math.cos(cos_progress * math.pi)) / 2
        return lr_min + scale_factor * (lr_max - lr_min)
    else:
        return lr_min


def clip_gradients(params: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    total_squared_gradient: float = 0.

    for param in params:
        if param.grad is not None:
            total_squared_gradient += param.grad.pow(2).sum().item()

    total_l2_norm: float = math.sqrt(total_squared_gradient)
    if total_l2_norm <= max_l2_norm:
        return

    scaling_factor: float = max_l2_norm / (total_l2_norm + eps)
    for param in params:
        if param.grad is not None:
            param.grad *= scaling_factor

class AdamW(torch.optim.Optimizer):
    def __init__(self, model_params, optimizer_params: AdamWParams):
        defaults: Dict[str, Any] = {
            "lr": optimizer_params.learning_rate,
            "b1": optimizer_params.b1,
            "b2": optimizer_params.b2,
            "wdr": optimizer_params.weight_decay,
            "eps": optimizer_params.eps,
        }

        if optimizer_params.gradient_clip is not None:
            defaults["gradient_clip"] = optimizer_params.gradient_clip

        if optimizer_params.cosine_schedule is not None:
            defaults["cosine_params"] = optimizer_params.cosine_schedule

        super().__init__(model_params, defaults)

    def _get_learning_rate_for_group(self, step: int, group) -> float:
        cosine_schedule: Optional[CosineScheduleParams] = group.get("cosine_params", None)

        if cosine_schedule is None:
            lr = group["lr"]
        else:
            lr = get_cosine_lr_schedule(step, (cosine_schedule["lr_min"], cosine_schedule["lr_max"]),
                cosine_schedule["n_warmup"], cosine_schedule["n_anneal"])

        return lr

    def get_learning_rate(self, step: int) -> float:
        return self._get_learning_rate_for_group(step, self.defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            wdr = group["wdr"]
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]
            gradient_clip = group.get("gradient_clip", None)

            param: torch.nn.Parameter
            for param in group["params"]:
                if param.grad is None:
                    continue

                if gradient_clip is not None:
                    clip_gradients([param], gradient_clip)
                    
                param_state = self.state[param]
                t = param_state.get("t", 1)
                grad = param.grad.data

                lr = self._get_learning_rate_for_group(t, group)

                param_state["m1"] = b1 * param_state.get("m1", 0) + (1 - b1) * grad
                param_state["m2"] = b2 * param_state.get("m2", 0) + (1 - b2) * (grad ** 2)

                # time step learning rate
                lr_t = lr * math.sqrt(1 - math.pow(b2, t)) / (1 - math.pow(b1, t))

                # core param update
                param.data -= lr_t * param_state["m1"] / (torch.sqrt(param_state["m2"]) + eps)

                # apply weight decay
                param.data -= param.data * lr * wdr
                
                # increment time step
                param_state["t"] = t + 1

        return loss
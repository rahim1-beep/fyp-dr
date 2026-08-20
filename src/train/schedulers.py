"""Optimiser and LR schedule: AdamW + cosine decay with linear warmup (CLAUDE.md §5).

WHY WARMUP. The backbone is pretrained and the head is random. In the first few hundred
steps the head produces large, badly-directed gradients, and at full LR those flow
straight into pretrained features and destroy them — the model then spends the rest of
training recovering to somewhere worse than where it started. A linear ramp over the first
`warmup_epochs` keeps that from happening.

WHY PER-STEP AND NOT PER-EPOCH. A cosine stepped once per epoch is a staircase, not a
cosine, and with 30 epochs it is a very coarse one. This schedules per optimiser step.

WHY NO LR ON THE METRIC. `ReduceLROnPlateau` reads the validation metric and changes
training in response to it. That is a legitimate technique, but it makes the validation
set part of the optimisation loop in a way that is easy to lose track of, and this project
already spends its validation budget on early stopping and model selection (R3). Cosine
needs nothing from val.
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

OPTIMIZERS = {"adamw", "sgd"}


def build_optimizer(model: torch.nn.Module, kind: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 1e-4, momentum: float = 0.9) -> Optimizer:
    """AdamW over the parameters that require grad.

    No-decay on norm layers and biases: weight decay on a BatchNorm scale or a bias is
    regularising a parameter that has no capacity to overfit, and on small datasets it
    measurably hurts. This is standard practice and worth having rather than leaving as a
    detail someone rediscovers.
    """
    if kind not in OPTIMIZERS:
        raise ValueError(f"optimizer={kind!r}; expected one of {sorted(OPTIMIZERS)}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    if not decay and not no_decay:
        raise ValueError("no trainable parameters — the model is entirely frozen")

    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]

    if kind == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    return AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def cosine_with_warmup(
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: float = 2.0,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Linear warmup then cosine decay to `min_lr_ratio` of the peak, stepped per batch.

    `min_lr_ratio` is 0.01 rather than 0: a schedule that reaches exactly zero spends its
    last epoch not training, which wastes the epoch and makes the final checkpoint
    identical to the one before it.
    """
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch={steps_per_epoch}; the loader is empty")
    if epochs < 1:
        raise ValueError(f"epochs={epochs}")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            f"warmup_epochs={warmup_epochs} must be >= 0 and < epochs={epochs}; a warmup "
            "as long as the run means the LR never decays"
        )

    total = steps_per_epoch * epochs
    warmup = int(round(steps_per_epoch * warmup_epochs))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            # step+1 so the very first step is not exactly zero LR, which would make the
            # first batch a no-op.
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def lr_of(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def preview_schedule(steps_per_epoch: int, epochs: int, warmup_epochs: float = 2.0,
                     lr: float = 3e-4, min_lr_ratio: float = 0.01) -> list[float]:
    """The LR at each epoch boundary, for the run log.

    Cheap, and it turns "the schedule is cosine with warmup" from a claim in a config
    into a printed curve that someone can look at and see is wrong.
    """
    model = torch.nn.Linear(1, 1)
    opt = build_optimizer(model, "adamw", lr=lr)
    sched = cosine_with_warmup(opt, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio)

    out = []
    for _ in range(epochs):
        out.append(lr_of(opt))
        for _ in range(steps_per_epoch):
            opt.step()
            sched.step()
    return out

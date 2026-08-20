"""Losses for arms A-F. One factory, selected by `imbalance.loss` in the arm config.

    A  cross_entropy, label smoothing 0.1, no class weights
    B  cross_entropy + balanced sampler
    C  cross_entropy + class weights from the TRAIN split only
    D  focal + balanced sampler
    E  smooth_l1 on a single continuous output (ordinal)
    F  cross_entropy + balanced sampler, pooled training data

CLASS WEIGHTS COME FROM THE TRAIN SPLIT AND NOWHERE ELSE (R2). `class_weights_from` takes
a training frame, not a dataset; computing them over train+val would let the validation
distribution steer the loss, which is the same leak as balancing val, wearing a different
hat.

Arms C and D both fight imbalance, one through the loss and one through the sampler.
Combining them is possible and is *not* the default anywhere: doing both double-counts the
correction, and then no one can say which mechanism produced the result. If an arm config
ever sets `class_weights: true` alongside `sampler: weighted_random`, `build_loss` says so
loudly rather than silently obeying.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.manifest import class_counts

LOSSES = {"cross_entropy", "focal", "smooth_l1", "mse"}


def class_weights_from(train_df: pd.DataFrame, scheme: str = "inverse_frequency",
                       normalise: bool = True) -> torch.Tensor:
    """Per-class loss weights, computed on the TRAINING SPLIT ONLY (R2).

    Normalised to mean 1.0 so the weights change the BALANCE between classes without
    changing the overall gradient scale — otherwise arm C would effectively be running a
    different learning rate from arm A and the comparison would confound the two.
    """
    counts = class_counts(train_df)
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f"train split has no examples of class(es) {missing}")

    if scheme == "inverse_frequency":
        w = counts.sum() / counts
    elif scheme == "effective_number":
        # Cui et al. 2019: (1 - beta) / (1 - beta^n), beta close to 1. Gentler than raw
        # inverse frequency on a 36:1 imbalance, where inverse frequency hands grade 4 a
        # weight ~36x grade 0 and the loss becomes dominated by a few hundred images.
        beta = 0.9999
        w = (1.0 - beta) / (1.0 - beta ** counts.to_numpy())
        w = pd.Series(w, index=counts.index)
    else:
        raise ValueError(f"unknown class_weight_scheme={scheme!r}")

    t = torch.tensor(w.to_numpy(), dtype=torch.float32)
    return t / t.mean() if normalise else t


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017): (1 - p_t)^gamma * CE.

    Down-weights examples the model already gets right, so training attention moves to
    the hard ones. In arm D the sampler has already balanced the classes, so `alpha` is
    None by default — adding per-class alpha on top of a balanced sampler applies the
    same correction twice.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"focal gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.alpha if self.alpha.numel() else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none",
                             label_smoothing=self.label_smoothing)
        # p_t via exp(-ce) is only exact without smoothing; with smoothing it is a close
        # and standard approximation, and label_smoothing defaults to 0 for focal arms.
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class OrdinalRegressionLoss(nn.Module):
    """Smooth L1 (or MSE) on a single continuous output — arm E.

    The grades are ORDERED, and softmax cross-entropy is blind to that: it treats
    "predicted 0, truth 4" and "predicted 3, truth 4" as equally wrong. A regression head
    is penalised by distance, which is the same thing QWK measures, so this arm frequently
    wins on the primary metric.

    The output is a real number. Turning it into a grade needs cut points, and those are
    optimised on VALIDATION ONLY (R3) — see configs/arm_e.yaml.
    """

    def __init__(self, kind: str = "smooth_l1", beta: float = 1.0):
        super().__init__()
        if kind not in {"smooth_l1", "mse"}:
            raise ValueError(f"ordinal loss must be smooth_l1 or mse, got {kind!r}")
        self.kind = kind
        self.beta = beta

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output.squeeze(-1).float()
        y = target.float()
        if self.kind == "mse":
            return F.mse_loss(pred, y)
        return F.smooth_l1_loss(pred, y, beta=self.beta)


def build_loss(
    arm_cfg: dict,
    train_df: pd.DataFrame | None = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """The loss for one arm. `arm_cfg` is the `imbalance:` block of the arm config.

    `train_df` is required only when the arm asks for class weights, and it must be the
    TRAIN split (R2).
    """
    kind = arm_cfg.get("loss", "cross_entropy")
    if kind not in LOSSES:
        raise ValueError(f"loss={kind!r}; expected one of {sorted(LOSSES)}")

    wants_weights = bool(arm_cfg.get("class_weights", False))
    uses_sampler = arm_cfg.get("sampler", "none") == "weighted_random"

    if wants_weights and uses_sampler:
        raise ValueError(
            f"arm {arm_cfg.get('arm', '?')} sets class_weights=true AND "
            "sampler=weighted_random. Both correct the same imbalance, so together they "
            "double-count it and no one can attribute the result to either. Arms C and D "
            "exist to compare them, not to combine them — pick one."
        )

    if kind in {"smooth_l1", "mse"}:
        return OrdinalRegressionLoss(kind)

    weights = None
    if wants_weights:
        if train_df is None:
            raise ValueError(
                "this arm asks for class weights, so the TRAIN split frame is required "
                "(R2 — weights come from train only)"
            )
        weights = class_weights_from(
            train_df, arm_cfg.get("class_weight_scheme", "inverse_frequency")
        )

    if kind == "focal":
        alpha = arm_cfg.get("focal_alpha")
        return FocalLoss(
            gamma=float(arm_cfg.get("focal_gamma", 2.0)),
            alpha=weights if alpha is not None else None,
        )

    # Label smoothing 0.1 on CE arms, per configs/base.yaml.
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def predictions_from(output: torch.Tensor, head: str,
                     thresholds: list[float] | None = None) -> torch.Tensor:
    """Model output -> integer grades 0-4, for whichever head the arm uses.

    Softmax arms take the argmax. Ordinal arms cut the continuous output at thresholds
    which, until Phase 4 optimises them on validation, default to the midpoints
    [0.5, 1.5, 2.5, 3.5]. Rounding is exactly those midpoints, so the default is the
    honest starting point rather than a tuned one.
    """
    if head == "softmax":
        return output.argmax(dim=1)

    cuts = torch.tensor(thresholds if thresholds is not None else [0.5, 1.5, 2.5, 3.5],
                        dtype=torch.float32, device=output.device)
    x = output.squeeze(-1).float()
    # right=True so a value sitting exactly ON a cut point rounds UP: with the default
    # midpoints that makes this exactly round-half-up, which is what "cut at the
    # midpoints" means. right=False would send 2.5 to grade 2 and quietly disagree with
    # the docstring.
    return torch.bucketize(x, cuts, right=True).clamp(0, 4)

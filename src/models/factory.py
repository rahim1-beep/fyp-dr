"""Model construction — timm only (R7), fully fine-tuned, never frozen-only.

`build_model` is the single place a network is created. Two things it will not let you
do, because both are project failure modes rather than style preferences:

**Frozen-only training is not expressible.** `configs/base.yaml` offers
`freeze_warmup_epochs` (0-2), and that is a *warmup* — the backbone comes back. A model
whose backbone never unfreezes is an ImageNet feature extractor with a linear probe on
top; on fundus images, whose statistics look nothing like ImageNet's, that is the
standard route to a model that sits near the majority-class rate. `unfreeze_all` is
called at the end of warmup and `assert_fully_trainable` is called before the first real
epoch.

**Normalisation is never invented.** `normalisation` reads timm's `default_cfg` for the
architecture actually built. Nothing here, and nothing in `configs/`, holds a mean/std
literal — a dataset-computed statistic would fold test pixels into training, and an
ImageNet literal is silently wrong for any model whose weights expect something else.

HEADS. Arms A-D and F use a 5-way softmax. Arm E uses a single-output regression head
(`num_outputs: 1`) because the grades are ordered; its predictions become grades via
thresholds optimised on VALIDATION (R3), which lives in `src/eval/thresholds.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs" / "base.yaml"

HEADS = {"softmax", "ordinal_regression"}


@dataclass(frozen=True)
class ModelConfig:
    """The `model:` block of configs/base.yaml, plus the arm's head."""

    arch: str = "resnet18"
    pretrained: bool = True
    freeze_warmup_epochs: int = 0
    drop_rate: float = 0.0
    head: str = "softmax"
    num_outputs: int = 5

    def __post_init__(self) -> None:
        if self.head not in HEADS:
            raise ValueError(f"model.head={self.head!r} is not one of {sorted(HEADS)}")
        if self.head == "softmax" and self.num_outputs != 5:
            raise ValueError(
                f"a softmax head needs num_outputs=5 (one per DR grade), got "
                f"{self.num_outputs}"
            )
        if self.head == "ordinal_regression" and self.num_outputs != 1:
            raise ValueError(
                f"the ordinal head is a single continuous output; got "
                f"{self.num_outputs}. See configs/arm_e.yaml"
            )
        if not 0 <= self.freeze_warmup_epochs <= 5:
            raise ValueError(
                f"freeze_warmup_epochs={self.freeze_warmup_epochs}; this is a short "
                "head-only WARMUP, not a training strategy. The backbone always "
                "unfreezes (R7 / failure mode 2)."
            )
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError(f"drop_rate={self.drop_rate} out of range")


def load_model_config(path: Path | str = BASE_CONFIG,
                      arm_path: Path | str | None = None) -> ModelConfig:
    """base.yaml's `model:` block, with an arm overlay on top.

    Same merge order as `load_preprocess_config`: base first, then the overlay, so an
    arm file only has to state what it changes.
    """
    def block(p: Path | str) -> dict:
        cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        b = dict(cfg.get("model") or {})
        unknown = set(b) - set(ModelConfig.__dataclass_fields__)
        if unknown:
            raise KeyError(f"{p} model block has unknown key(s): {sorted(unknown)}")
        return b

    merged = block(path)
    if arm_path is not None:
        merged.update(block(arm_path))
    return ModelConfig(**merged)


def build_model(cfg: ModelConfig) -> nn.Module:
    """Create the network. Every parameter trainable on return."""
    model = timm.create_model(
        cfg.arch,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_outputs,
        drop_rate=cfg.drop_rate,
    )

    # timm sets num_classes on the classifier it knows about; confirm it took, rather
    # than trusting it. A head left at 1000 outputs trains, converges to something, and
    # produces a confusion matrix that is quietly meaningless.
    n_out = _classifier_out_features(model)
    if n_out is not None and n_out != cfg.num_outputs:
        raise RuntimeError(
            f"{cfg.arch} was built with num_classes={cfg.num_outputs} but its classifier "
            f"has {n_out} outputs"
        )

    unfreeze_all(model)
    return model


def _classifier_out_features(model: nn.Module) -> int | None:
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(head, nn.Linear):
        return head.out_features
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    return None


def normalisation(model: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) from timm's config for THIS architecture. Never a literal."""
    cfg = getattr(model, "default_cfg", None) or getattr(model, "pretrained_cfg", None)
    if not cfg or "mean" not in cfg or "std" not in cfg:
        raise AttributeError(
            f"{type(model).__name__} exposes no default_cfg mean/std. Do not fall back "
            "to ImageNet constants silently — a wrong normalisation trains a model that "
            "converges and underperforms with no error anywhere."
        )
    return tuple(cfg["mean"]), tuple(cfg["std"])


def input_size(model: nn.Module) -> int:
    cfg = getattr(model, "default_cfg", None) or getattr(model, "pretrained_cfg", {})
    return int(cfg.get("input_size", (3, 224, 224))[-1])


# ----------------------------------------------------------------------------------
# Freezing — a warmup, never a strategy
# ----------------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> int:
    """Freeze everything except the classifier. Returns the number frozen.

    Only ever for `freeze_warmup_epochs` at the start of training, to stop a randomly
    initialised head from wrecking pretrained features with its first large gradients.
    """
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_params = {id(p) for p in head.parameters()} if head is not None else set()
    if not head_params:
        raise RuntimeError(
            f"{type(model).__name__} exposes no classifier, so a head-only warmup would "
            "freeze the entire network. Set freeze_warmup_epochs: 0 for this model."
        )

    n = 0
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad_(False)
            n += 1
    return n


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def trainable_fraction(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return live / total if total else 0.0


def assert_fully_trainable(model: nn.Module) -> None:
    """Called before the first unfrozen epoch. Failure mode 2, made loud.

    A run that spends thirty epochs with a frozen backbone looks completely normal in the
    logs — the loss falls, the metric rises a little, and the result is a model at 20-45%
    accuracy for a reason nothing reports.
    """
    if not any(True for _ in model.parameters()):
        raise RuntimeError(
            f"{type(model).__name__} has no parameters at all, so there is nothing to "
            "fine-tune. This is almost always a stub or a misconfigured head."
        )
    frac = trainable_fraction(model)
    if frac < 0.999:
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        raise RuntimeError(
            f"only {frac:.1%} of parameters are trainable; {len(frozen)} tensor(s) are "
            f"frozen, e.g. {frozen[:5]}. Full fine-tuning is required (R7 / failure "
            "mode 2) — frozen-only training is the standard route to a collapsed model."
        )


def describe(model: nn.Module, cfg: ModelConfig) -> str:
    """One block for the run log, so a run's architecture is in its own output."""
    total = sum(p.numel() for p in model.parameters())
    mean, std = normalisation(model)
    return (
        f"arch          : {cfg.arch}  (pretrained={cfg.pretrained})\n"
        f"head          : {cfg.head}, {cfg.num_outputs} output(s)\n"
        f"parameters    : {total / 1e6:.2f}M, {trainable_fraction(model):.1%} trainable\n"
        f"normalisation : mean={tuple(round(v, 4) for v in mean)} "
        f"std={tuple(round(v, 4) for v in std)}  [from timm default_cfg]\n"
        f"input size    : {input_size(model)}"
    )


@torch.no_grad()
def smoke_forward(model: nn.Module, image_size: int = 224, batch: int = 2) -> torch.Size:
    """One forward pass on noise. Catches a head/loss shape mismatch in a second rather
    than in the first epoch on Kaggle."""
    model.eval()
    return model(torch.randn(batch, 3, image_size, image_size)).shape


# The key `src/train/loop.py` saves the weights under. There is exactly one convention
# and it lives here so that call sites cannot invent their own — which is precisely what
# happened: `notebooks/phase5_gradcam.py` guessed `"model"`, a key that has never existed
# in this project, and the guess survived local testing because no test loaded a real
# checkpoint written by the real training loop.
CHECKPOINT_STATE_KEY = "state_dict"


def load_checkpoint(path: Path | str, model: nn.Module | None = None, *,
                    map_location: str = "cpu") -> dict:
    """Load a checkpoint written by `src/train/loop.py`, optionally into `model`.

    Returns the whole dict — callers also want `epoch` and `val_qwk`.

    Raises with the keys it ACTUALLY found rather than letting `load_state_dict` report
    a wall of unexpected-key noise. Passing the outer dict (epoch, val_qwk, state_dict)
    straight to `load_state_dict` produces a RuntimeError that reads like a model
    architecture mismatch, which is a long way from "you used the wrong key".
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or CHECKPOINT_STATE_KEY not in ckpt:
        found = sorted(ckpt) if isinstance(ckpt, dict) else type(ckpt).__name__
        raise KeyError(
            f"{path} has no {CHECKPOINT_STATE_KEY!r} key; found {found}. Checkpoints in "
            "this project are written by src/train/loop.py as "
            "{'epoch', 'val_qwk', 'state_dict'}."
        )
    if model is not None:
        model.load_state_dict(ckpt[CHECKPOINT_STATE_KEY])
    return ckpt

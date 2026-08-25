"""Grad-CAM and Grad-CAM++ for the DR models, ordinal head included.

THREE THINGS HERE ARE EASY TO GET SILENTLY WRONG, and all three would produce a picture
rather than an error.

1. **Arm E has ONE output, not five.** Textbook Grad-CAM backpropagates from a class
   logit chosen by `argmax`. The ordinal-regression head emits a single scalar, so
   `argmax` over a length-1 vector is always 0 and the code "works" while meaning nothing.
   Here the gradient is taken from **the scalar itself**, and the map answers *what raised
   the predicted severity*.

2. **Softmax heads are scored on the expected grade**, not the argmax logit, so that the
   explanation matches the quantity every ranking claim uses (DECISION-035). Explaining
   `argmax` while ranking on `sum(p_i * i)` would be explaining a different model than the
   one being reported.

3. **The map is 7x7 before upsampling.** EfficientNet-B0 at 224 ends on a 7x7 grid, so one
   CAM cell covers ~32x32 input pixels. **This cannot localise microaneurysms** — they are
   sub-pixel at 224. It resolves gross attention only: optic disc, macula, arcades, and
   the rim. That is a limitation to state in the write-up, not to imply away; it is also
   exactly the resolution the border-artefact question needs (DECISION-046).

A CAM is a *correlational* statement about where activation and gradient coincide. It is
not evidence that the model's decision depends on that region — high activation at a
high-contrast boundary is partly an edge effect. `src/xai/border_check.py` carries the
occlusion test that turns this into a causal claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Layer names tried in order for the final convolutional feature map. `conv_head` is
# EfficientNet's 1x1 projection to 1280 channels and is the conventional Grad-CAM target
# for that family; `blocks[-1]` is the fallback, and `layer4` covers ResNet.
TARGET_LAYER_NAMES = ("conv_head", "blocks", "layer4", "features")


def pick_target_layer(model: nn.Module) -> nn.Module:
    """The last convolutional feature map, chosen by name and then VERIFIED by shape.

    Choosing by name alone is how this breaks on the next backbone: `timm` is not
    obliged to keep a name, and a wrong pick silently yields a 1x1 map that upsamples to
    a uniform heatmap — which looks like "the model attends everywhere" rather than like
    a bug.
    """
    for name in TARGET_LAYER_NAMES:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        if isinstance(mod, (nn.Sequential, nn.ModuleList)) and len(mod):
            mod = mod[-1]
        return mod
    raise ValueError(
        f"no target layer found among {TARGET_LAYER_NAMES} on {type(model).__name__}; "
        "pass one explicitly"
    )


def scalar_target(out: torch.Tensor) -> torch.Tensor:
    """Reduce a model output to ONE scalar per image, to differentiate.

    Ordinal head (1 output)  -> the scalar itself.
    Softmax head (5 outputs) -> the EXPECTED GRADE, sum(p_i * i), matching the quantity
                                every ranking claim in this project uses (DECISION-035).
    """
    if out.ndim == 1:
        return out
    if out.shape[1] == 1:
        return out[:, 0]
    p = out.softmax(dim=1)
    grades = torch.arange(out.shape[1], device=out.device, dtype=p.dtype)
    return (p * grades).sum(dim=1)


@dataclass
class CamResult:
    cam: np.ndarray            # (B, H, W) in [0, 1], upsampled to the input size
    score: np.ndarray          # (B,) the scalar that was differentiated


class GradCAM:
    """Grad-CAM over one target layer. `plus_plus=True` selects Grad-CAM++.

    Used as a context manager so the hooks are always removed; a leaked backward hook on
    a model that is later trained is a memory leak and a silent slowdown.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None,
                 *, plus_plus: bool = False) -> None:
        self.model = model
        self.layer = target_layer if target_layer is not None else pick_target_layer(model)
        self.plus_plus = plus_plus
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        self._handles = [
            self.layer.register_forward_hook(self._save_acts),
            self.layer.register_full_backward_hook(self._save_grads),
        ]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _save_acts(self, _m, _i, out) -> None:
        self._acts = out

    def _save_grads(self, _m, _gi, gout) -> None:
        self._grads = gout[0]

    def __call__(self, x: torch.Tensor) -> CamResult:
        if not self._handles:
            raise RuntimeError("use GradCAM as a context manager: `with GradCAM(m) as g:`")
        was_training = self.model.training
        self.model.eval()
        # enable_grad explicitly: this is called from evaluation code that is very often
        # already inside torch.no_grad(), where the backward pass would fail with an
        # unhelpful message about tensors not requiring grad.
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            out = self.model(x)
            score = scalar_target(out)
            self.model.zero_grad(set_to_none=True)
            score.sum().backward()

        acts, grads = self._acts, self._grads
        if acts is None or grads is None:
            raise RuntimeError("the target layer produced no activations or gradients")
        if acts.ndim != 4 or min(acts.shape[2:]) < 2:
            raise ValueError(
                f"target layer output is {tuple(acts.shape)}; Grad-CAM needs a 4D feature "
                "map with spatial extent > 1. A 1x1 map upsamples to a uniform heatmap, "
                "which looks like a result and is a bug."
            )

        if self.plus_plus:
            g2, g3 = grads.pow(2), grads.pow(3)
            denom = 2 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[:, 0]

        # Per-image min-max. A CAM that is entirely zero (no positive evidence anywhere)
        # stays zero rather than becoming uniform noise through a divide-by-epsilon.
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        span = (hi - lo)
        cam = torch.where(span > 0, (cam - lo) / span.clamp(min=1e-12),
                          torch.zeros_like(cam))

        if was_training:
            self.model.train()
        return CamResult(cam=cam.detach().cpu().numpy(),
                         score=score.detach().cpu().numpy())


def randomise_last_block(model: nn.Module, seed: int = 0) -> nn.Module:
    """A DEEP COPY of `model` with the target layer's parameters re-initialised.

    For the randomisation gate in `border_check.py`. Deep-copied because randomising in
    place would destroy the checkpoint being analysed, and the caller almost never wants
    that.
    """
    rng = torch.Generator().manual_seed(seed)
    clone = copy.deepcopy(model)
    layer = pick_target_layer(clone)
    n = 0
    for p in layer.parameters():
        with torch.no_grad():
            p.copy_(torch.empty_like(p).normal_(0.0, 0.1, generator=rng)
                    if p.dim() > 0 else torch.zeros_like(p))
        n += 1
    if n == 0:
        raise ValueError("the target layer has no parameters to randomise, so the "
                         "randomisation gate would trivially and meaninglessly pass")
    return clone


def cam_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two CAMs, flattened. NaN-safe.

    Two constant maps are perfectly correlated in the sense that matters here — neither
    carries information — so they return 1.0 rather than NaN, which would otherwise be
    silently dropped from a median and turn a failing gate into a passing one.
    """
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if x.std() == 0 and y.std() == 0:
        return 1.0
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

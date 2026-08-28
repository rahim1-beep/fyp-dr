"""The one way images reach a model: manifest CSV -> Dataset -> DataLoader (R5).

Reads the preprocessed cache written by `src/data/preprocess.py`. Never the raw images,
never a monolithic array.

THREE THINGS HERE ARE LOAD-BEARING AND EASY TO GET SILENTLY WRONG.

1. **Channel order.** The cache is BGR on disk, because cv2 wrote it. timm's
   `default_cfg` normalisation assumes **RGB**. The conversion happens exactly once, in
   `load_cached_image`, and `tests/test_dataset.py` asserts it. Getting this backwards
   produces a model that trains, converges, and quietly underperforms with no error
   anywhere — the failure mode that lands a DR project at 20-45% accuracy.

2. **Augmentation is train-only** (R2, proposal §5.3). `train=False` builds a Dataset with
   no augmentation at all, not "milder" augmentation. A flip on the validation set is a
   different kind of leak: it changes what the early-stopping metric measures.

3. **Normalisation constants come from the model**, via `timm`'s `default_cfg`, and are
   passed in. `configs/base.yaml` says explicitly not to put mean/std in the config, and
   nothing here computes them from the data — a dataset-wide mean would fold test pixels
   into training.

A missing cache file RAISES. It is never skipped: a skipped row is a silently shorter
epoch and a val set that quietly changed size between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import cache_relpath

# ImageNet defaults, used ONLY when no model config is available (a smoke test, a
# transform inspected in isolation). Real runs pass timm's `model.default_cfg`.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentConfig:
    """Exactly the `augment:` block of configs/base.yaml."""

    enabled: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation_degrees: float = 20.0
    brightness: float = 0.15
    contrast: float = 0.15
    scale: tuple[float, float] = (0.9, 1.1)
    shift: float = 0.05
    # DECISION-063 — the surround-randomisation remedy. Probability that one training
    # image has its masked-out surround replaced by a per-image random constant colour.
    # DEFAULT 0.0 = OFF, so every run predating the remedy reproduces bit-for-bit.
    surround_randomisation: float = 0.0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "AugmentConfig":
        block = dict(cfg.get("augment") or {})
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"augment block has unknown key(s): {sorted(unknown)}")
        if "scale" in block:
            block["scale"] = tuple(block["scale"])
        return cls(**block)


def load_cached_image(path: Path) -> np.ndarray:
    """Decode one cached JPEG to an **RGB** uint8 array.

    THE ONE PLACE BGR BECOMES RGB. np.fromfile rather than cv2.imread because cv2.imread
    returns None on a non-ASCII path on Windows and reports no error.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{path} did not decode as an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DRDataset(Dataset):
    """One split of the manifest, backed by the preprocessed cache.

    `__getitem__` returns `(image, label, index)`. The index is not decoration: arm F
    reports metrics broken down by source dataset (DECISION-007), the eval code needs to
    know which manifest row each prediction came from, and reconstructing that from a
    shuffled DataLoader is a classic silent misalignment. Training loops ignore it.

    **The index is positional. Join it as `loader.dataset.df.iloc[idx]`** — never `.loc`
    on a frame the caller kept a reference to. `__init__` refuses a non-RangeIndex frame
    so that rule cannot be broken quietly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_root: Path | str,
        *,
        train: bool,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        augment: AugmentConfig | None = None,
        image_size: int = 224,
    ) -> None:
        missing = {"image_path", "label", "dataset"} - set(df.columns)
        if missing:
            raise ValueError(f"manifest is missing column(s): {sorted(missing)}")
        if len(df) == 0:
            raise ValueError("manifest has no rows")

        # The returned index is POSITIONAL in this frame. If the caller's frame carries a
        # shuffled or gapped index, `.loc[idx]` on the caller's side returns a different
        # row than the one the model saw, and a pooled frame built with a bare pd.concat
        # has duplicate labels so `.loc[idx]` returns SEVERAL rows. Both are silent.
        # Refuse the ambiguity instead of documenting it: the join is
        # `loader.dataset.df.iloc[idx]`.
        if not df.index.equals(pd.RangeIndex(len(df))):
            raise ValueError(
                "manifest index must be a clean RangeIndex — pass "
                "df.reset_index(drop=True) (or pd.concat(..., ignore_index=True)). The "
                "index returned by __getitem__ is positional, and joining it against a "
                "shuffled or duplicated index silently returns the wrong row."
            )

        # NaN counts as a violation, not as something to drop. A NaN label survives
        # construction, becomes a NaN sampler weight, and fails at the first draw of the
        # first epoch — on Kaggle, forty minutes in, rather than here.
        if df["label"].isna().any():
            n = int(df["label"].isna().sum())
            raise ValueError(f"{n} row(s) have a NaN label")
        bad = sorted(set(pd.unique(df["label"])) - {0, 1, 2, 3, 4})
        if bad:
            raise ValueError(f"labels outside 0-4: {bad}")

        # §2.1 forbids duplicated rows in any manifest. tests/test_no_leakage.py enforces
        # that on the committed CSVs; nothing enforced it on the frame actually handed to
        # a DataLoader, so physical oversampling by pd.concat was accepted here in silence.
        dup = df["image_path"].duplicated()
        if dup.any():
            examples = df.loc[dup, "image_path"].head(3).tolist()
            raise ValueError(
                f"{int(dup.sum())} duplicate image_path row(s), e.g. {examples}. "
                "Physical oversampling is forbidden (CLAUDE.md §2.1) — balance with "
                "WeightedRandomSampler via src/data/sampler.py."
            )

        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.train = train
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        # R2/§5.3: val and test get no augmentation, ever. Not milder — none. Passing an
        # AugmentConfig alongside train=False is a mistake worth refusing rather than
        # silently ignoring, because the silent version is invisible in every log.
        if augment is not None and not train and augment.enabled:
            raise ValueError(
                "augmentation was requested for a non-training split. Augmentation is "
                "train-only (R2); val/test must be evaluated on the images as cached."
            )
        self.augment = augment if train else None

        self.cache_paths = [
            self.cache_root / cache_relpath(p, d)
            for p, d in zip(self.df["image_path"], self.df["dataset"])
        ]

    def __len__(self) -> int:
        return len(self.df)

    def check_cache(self, limit: int | None = None) -> list[Path]:
        """Cache files this split expects but does not have. Cheap; call it once at
        construction time in a training script rather than discovering row 20,000 is
        missing forty minutes into an epoch."""
        out = []
        for p in self.cache_paths:
            if not p.exists():
                out.append(p)
                if limit and len(out) >= limit:
                    break
        return out

    def _augment(self, img: torch.Tensor,
                 retina: torch.Tensor | None = None) -> torch.Tensor:
        """Geometry then photometry, on a [3,H,W] float tensor in [0,1].

        Rotation and shift are safe here in a way they would not be on a raw fundus: the
        image is already square-cropped and centred on the retina with a black surround,
        so a 20-degree rotation moves black into black. Fill is 0 for the same reason —
        it matches the masked surround the model already sees everywhere else.

        `retina` is an optional [H,W] float indicator of the retinal disc, supplied only
        when `surround_randomisation` is on. It rides through the SAME geometry as the
        image, stacked as a fourth channel, rather than being recomputed afterwards:
        re-thresholding a rotated, scaled, colour-jittered image would disagree with the
        mask at exactly the boundary the remedy is about. Riding along also means the
        affine's own zero-fill wedges land OUTSIDE the indicator and get filled too —
        otherwise the remedy would leave fresh black corners for the model to read, which
        is the cue it exists to remove.
        """
        from torchvision.transforms import v2
        from torchvision.transforms.v2 import functional as F

        a = self.augment
        assert a is not None
        if not a.enabled:
            return img

        x = img if retina is None else torch.cat([img, retina[None]], 0)

        if torch.rand(1).item() < a.horizontal_flip:
            x = F.horizontal_flip(x)
        if torch.rand(1).item() < a.vertical_flip:
            x = F.vertical_flip(x)

        if a.rotation_degrees or a.shift or a.scale != (1.0, 1.0):
            affine = v2.RandomAffine(
                degrees=a.rotation_degrees,
                translate=(a.shift, a.shift) if a.shift else None,
                scale=a.scale if a.scale else None,
                fill=0.0,
            )
            x = affine(x)

        # ColorJitter is photometric and must not see the indicator channel.
        img, mask = (x, None) if retina is None else (x[:3], x[3] > 0.5)

        if a.brightness or a.contrast:
            img = v2.ColorJitter(brightness=a.brightness, contrast=a.contrast)(img)
        img = img.clamp_(0.0, 1.0)

        # Surround randomisation goes LAST, so the fill is exactly the sampled colour and
        # not a jittered version of it — the point is that surround appearance carries no
        # stable signal, and a jitter applied on top would reintroduce a weak one.
        if mask is not None and torch.rand(1).item() < a.surround_randomisation:
            img = torch.where(mask.unsqueeze(0), img, torch.rand(3, 1, 1))
        return img

    def _retina_indicator(self, rgb: np.ndarray) -> torch.Tensor | None:
        """[H,W] float 1.0-inside-retina indicator, or None when it is not needed or not
        defined.

        Returns None — leaving the image untouched — for the four `ok:no-retina` cache
        images of DECISION-018, one of which (1986_left) is in the validation split. An
        empty mask would make `~mask` the WHOLE FRAME, and the remedy would replace the
        entire image with a flat random colour and train on it. That is a silent,
        catastrophic version of the augmentation, and it is why this returns None rather
        than trusting the mask to be non-empty.
        """
        a = self.augment
        if a is None or not a.enabled or a.surround_randomisation <= 0:
            return None
        from src.data.preprocess import retina_mask

        m = retina_mask(np.ascontiguousarray(rgb[:, :, ::-1])) > 0
        if not m.any():
            return None
        return torch.from_numpy(m.astype(np.float32))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        path = self.cache_paths[i]
        if not path.exists():
            row = self.df.iloc[i]
            raise FileNotFoundError(
                f"cache miss: {path} (manifest row {i}, image_path={row['image_path']!r}, "
                f"dataset={row['dataset']!r}). The cache and the split CSVs disagree — "
                "run `python -m src.data.reconcile_cache` rather than skipping the row."
            )

        rgb = load_cached_image(path)
        if rgb.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(
                f"{path} is {rgb.shape[:2]}, expected "
                f"({self.image_size}, {self.image_size}). The cache was built at a "
                "different image_size than this run's config."
            )

        img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        if self.augment is not None:
            img = self._augment(img, self._retina_indicator(rgb))
        img = (img - self.mean) / self.std

        return img, int(self.df.at[i, "label"]), i


def normalisation_from_model(model) -> tuple[tuple, tuple]:
    """timm's per-model normalisation. Never hardcode these, never compute them from the
    data — a dataset-wide mean/std folds test pixels into training (base.yaml, §model)."""
    cfg = getattr(model, "default_cfg", None) or getattr(model, "pretrained_cfg", None)
    if not cfg:
        raise AttributeError(
            f"{type(model).__name__} exposes no default_cfg/pretrained_cfg; pass mean/std "
            "explicitly rather than falling back to ImageNet silently"
        )
    return tuple(cfg["mean"]), tuple(cfg["std"])

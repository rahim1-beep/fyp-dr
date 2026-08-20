"""Tests for src/data/dataset.py and src/data/sampler.py.

The two questions worth the most here are the ones no loss curve would ever answer:
whether the pixels reaching the model are RGB, and whether balancing stayed inside the
training split.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import AugmentConfig, DRDataset, load_cached_image
from src.data.manifest import class_counts
from src.data.sampler import build_loaders, build_sampler, describe_balance

REPO = Path(__file__).resolve().parents[1]


def _cache(tmp_path: Path, n: int = 40, size: int = 224, labels=None) -> tuple[pd.DataFrame, Path]:
    """A synthetic cache in the real layout: {root}/eyepacs/{patient}_{eye}.jpg."""
    root = tmp_path / "cache"
    (root / "eyepacs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    labels = labels if labels is not None else [i % 5 for i in range(n)]

    rows = []
    for i, lab in enumerate(labels):
        stem = f"{i}_left"
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "eyepacs" / f"{stem}.jpg"), img)
        rows.append({
            "image_path": f"data/data/{stem}.jpeg",
            "patient_id": str(i),
            "eye": "left",
            "label": int(lab),
            "dataset": "eyepacs",
            "split": "train",
        })
    return pd.DataFrame(rows), root


# ----------------------------------------------------------------------------------
# Channel order — the silent one
# ----------------------------------------------------------------------------------

def test_cached_bgr_is_delivered_as_rgb(tmp_path):
    """The cache is BGR on disk because cv2 wrote it; timm's default_cfg normalisation
    assumes RGB. A swap here trains a model that runs, converges, and underperforms with
    no error anywhere."""
    root = tmp_path / "c"
    root.mkdir()
    bgr = np.zeros((224, 224, 3), np.uint8)
    bgr[..., 0] = 200          # BLUE channel in cv2's ordering
    cv2.imwrite(str(root / "x.jpg"), bgr)

    rgb = load_cached_image(root / "x.jpg")
    assert rgb[..., 2].mean() > 180, "blue must arrive in channel index 2 (RGB)"
    assert rgb[..., 0].mean() < 40, "channel 0 is red and should be near zero"


def test_dataset_tensor_is_chw_float_and_normalised(tmp_path):
    df, root = _cache(tmp_path, n=4)
    ds = DRDataset(df, root, train=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    img, label, idx = ds[0]

    assert img.shape == (3, 224, 224) and img.dtype == torch.float32
    assert idx == 0 and isinstance(label, int)
    # with mean=std=0.5 a [0,1] image maps into [-1,1]
    assert -1.01 <= float(img.min()) and float(img.max()) <= 1.01


def test_index_is_returned_so_predictions_can_be_joined_back(tmp_path):
    """Arm F reports metrics by source dataset (DECISION-007). Recovering which row a
    prediction came from by re-deriving the shuffle order is a classic silent
    misalignment; the index removes the need."""
    df, root = _cache(tmp_path, n=8)
    ds = DRDataset(df, root, train=False)
    assert [ds[i][2] for i in range(8)] == list(range(8))


# ----------------------------------------------------------------------------------
# Augmentation is train-only
# ----------------------------------------------------------------------------------

def test_augmentation_is_refused_on_a_non_training_split(tmp_path):
    """Silently ignoring it would be worse than raising: a flip on the validation set
    changes what early stopping is measuring, and nothing in the log would show it."""
    df, root = _cache(tmp_path, n=4)
    with pytest.raises(ValueError, match="train-only"):
        DRDataset(df, root, train=False, augment=AugmentConfig(enabled=True))


def test_val_split_is_deterministic_across_reads(tmp_path):
    df, root = _cache(tmp_path, n=4)
    ds = DRDataset(df, root, train=False)
    assert torch.equal(ds[1][0], ds[1][0])


def test_training_augmentation_actually_changes_the_image(tmp_path):
    """An augment config that quietly does nothing would make arms look identical."""
    df, root = _cache(tmp_path, n=4)
    ds = DRDataset(df, root, train=True, augment=AugmentConfig())
    torch.manual_seed(0)
    a = ds[0][0]
    torch.manual_seed(1)
    b = ds[0][0]
    assert not torch.equal(a, b)


def test_augment_config_rejects_an_unknown_key():
    with pytest.raises(KeyError, match="unknown key"):
        AugmentConfig.from_yaml({"augment": {"horizontal_flip": 0.5, "hflip": 0.5}})


def test_base_config_augment_block_loads():
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
    a = AugmentConfig.from_yaml(cfg)
    assert a.enabled and a.scale == (0.9, 1.1)


# ----------------------------------------------------------------------------------
# A missing cache file is an error, never a skip
# ----------------------------------------------------------------------------------

def test_a_missing_cache_file_raises_rather_than_shortening_the_epoch(tmp_path):
    df, root = _cache(tmp_path, n=4)
    (root / "eyepacs" / "2_left.jpg").unlink()

    ds = DRDataset(df, root, train=False)
    assert ds.check_cache() == [root / "eyepacs" / "2_left.jpg"]
    with pytest.raises(FileNotFoundError, match="cache miss"):
        _ = ds[2]


def test_wrong_cache_size_is_caught(tmp_path):
    df, root = _cache(tmp_path, n=2, size=160)
    ds = DRDataset(df, root, train=False, image_size=224)
    with pytest.raises(ValueError, match="different image_size"):
        _ = ds[0]


def test_labels_outside_0_4_are_rejected(tmp_path):
    df, root = _cache(tmp_path, n=4)
    df.loc[0, "label"] = 7
    with pytest.raises(ValueError, match="outside 0-4"):
        DRDataset(df, root, train=False)


# ----------------------------------------------------------------------------------
# Balancing — CLAUDE.md §2.1
# ----------------------------------------------------------------------------------

def _imbalanced(tmp_path):
    labels = [0] * 300 + [1] * 60 + [2] * 90 + [3] * 20 + [4] * 12
    return _cache(tmp_path, n=len(labels), labels=labels)


def test_weighted_sampler_draws_the_classes_roughly_evenly(tmp_path):
    df, _ = _imbalanced(tmp_path)
    s = build_sampler(df, "weighted_random", seed=42)
    table = describe_balance(df, s, draws=40_000)

    drawn = table["drawn"].to_numpy()
    assert drawn.min() > 0.15 and drawn.max() < 0.25, f"not balanced: {drawn}"
    # and the natural distribution really was skewed, or the test proves nothing
    assert table["natural"].max() > 0.6


def test_sampler_none_leaves_the_distribution_alone(tmp_path):
    df, _ = _imbalanced(tmp_path)
    assert build_sampler(df, "none") is None


def test_sampler_draws_exactly_one_epoch_worth(tmp_path):
    """Two arms that see different amounts of data per epoch are not comparable, and the
    A-E comparison is the point."""
    df, _ = _imbalanced(tmp_path)
    s = build_sampler(df, "weighted_random")
    assert s.num_samples == len(df)
    assert s.replacement is True


def test_sampler_is_reproducible(tmp_path):
    df, _ = _imbalanced(tmp_path)
    a = list(build_sampler(df, "weighted_random", seed=42))
    b = list(build_sampler(df, "weighted_random", seed=42))
    c = list(build_sampler(df, "weighted_random", seed=7))
    assert a == b and a != c


def test_sampler_is_refused_on_val_or_test(tmp_path):
    """The forbidden operation of §2.1, made structurally impossible rather than
    documented. Rebalancing val makes the early-stopping metric a number about a
    distribution that does not exist."""
    df, _ = _imbalanced(tmp_path)
    for name in ("val", "test", "aptos_val"):
        bad = df.assign(split=name)
        with pytest.raises(ValueError, match="train-only"):
            build_sampler(bad, "weighted_random")


def test_arm_f_style_pooled_train_is_allowed(tmp_path):
    """Arm F pools train + aptos_train — several split values, all of them training."""
    df, _ = _imbalanced(tmp_path)
    df.loc[df.index[:50], "split"] = "aptos_train"
    assert build_sampler(df, "weighted_random") is not None


def test_sampler_refuses_a_split_with_a_missing_class(tmp_path):
    """Inverse frequency over a zero count is a division by zero that would otherwise
    surface as an all-zero confusion-matrix column much later (CLAUDE.md §2)."""
    labels = [0] * 50 + [1] * 20 + [2] * 20 + [3] * 10      # no grade 4
    df, _ = _cache(tmp_path, n=len(labels), labels=labels)
    with pytest.raises(ValueError, match="no examples of class"):
        build_sampler(df, "weighted_random")


def test_unknown_sampler_kind_raises(tmp_path):
    df, _ = _cache(tmp_path, n=5)
    with pytest.raises(ValueError, match="expected one of"):
        build_sampler(df, "oversample")


# ----------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------

def test_loaders_balance_train_and_leave_val_and_test_natural(tmp_path):
    """The whole of R2 in one assertion: the train loader's draw is balanced, and the
    val/test loaders deliver exactly the manifest, in order, unbalanced."""
    df, root = _imbalanced(tmp_path)
    val = df.iloc[:100].copy().assign(split="val")
    test = df.iloc[100:200].copy().assign(split="test")

    loaders = build_loaders(df, val, test, root, sampler_kind="weighted_random",
                            batch_size=16)

    assert loaders["train"].sampler is not None
    assert isinstance(loaders["val"].sampler, torch.utils.data.SequentialSampler)
    assert isinstance(loaders["test"].sampler, torch.utils.data.SequentialSampler)

    seen = [int(l) for _, lb, _ in loaders["val"] for l in lb]
    assert seen == val["label"].tolist(), "val must be the manifest, in order"

    natural = class_counts(test) / len(test)
    got = pd.Series([int(l) for _, lb, _ in loaders["test"] for l in lb])
    got = got.value_counts().reindex(range(5), fill_value=0).sort_index() / len(test)
    assert np.allclose(got.to_numpy(), natural.to_numpy())


def test_loaders_never_augment_val_or_test(tmp_path):
    df, root = _cache(tmp_path, n=8)
    loaders = build_loaders(df, df.assign(split="val"), None, root,
                            augment=AugmentConfig(), batch_size=4)
    assert loaders["val"].dataset.augment is None
    assert loaders["train"].dataset.augment is not None


def test_train_loader_delivers_one_epoch_of_batches(tmp_path):
    df, root = _cache(tmp_path, n=40)
    loaders = build_loaders(df, df.assign(split="val"), None, root,
                            sampler_kind="weighted_random", batch_size=8)
    n = sum(len(lb) for _, lb, _ in loaders["train"])
    assert n == len(df)


def test_describing_the_balance_does_not_advance_the_real_sampler(tmp_path):
    """R6. If logging the balance table changed the epoch, removing the log would change
    the run — and the table exists precisely so every run logs it."""
    df, _ = _imbalanced(tmp_path)
    s = build_sampler(df, "weighted_random", seed=42)
    describe_balance(df, s, draws=5_000)
    assert list(s) == list(build_sampler(df, "weighted_random", seed=42))


def test_the_balance_table_matches_a_real_epoch(tmp_path):
    """describe_balance draws from a clone, so it has to be checked against the sampler
    it claims to describe."""
    df, _ = _imbalanced(tmp_path)
    s = build_sampler(df, "weighted_random", seed=42)
    table = describe_balance(df, s, draws=40_000)

    # Accumulate 40 epochs. One epoch is 482 draws, where a proportion around 0.2 has a
    # standard error of ~0.018 — wide enough that a tolerance loose enough to pass would
    # not distinguish a balanced sampler from a mildly broken one.
    idx = np.concatenate([np.fromiter(s, dtype=np.int64) for _ in range(40)])
    labels = df["label"].to_numpy()[idx]
    real = (pd.Series(labels).value_counts().reindex(range(5), fill_value=0).sort_index()
            / len(labels))
    assert np.allclose(table["drawn"].to_numpy(), real.to_numpy(), atol=0.01), (
        f"table says {table['drawn'].tolist()}, 40 epochs are {real.round(4).tolist()}"
    )

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


def _write_split(root: Path, n: int = 40, size: int = 224, labels=None,
                 split: str = "train", first_patient: int = 0,
                 dataset: str = "eyepacs") -> pd.DataFrame:
    """Write `n` synthetic cache files into `root` and return the matching manifest.

    `first_patient` exists so callers can build train/val/test over DISJOINT patients.
    An earlier version of these tests reused the same patients across all three, which is
    the exact R1 violation the loaders are supposed to refuse — and the suite was green.
    """
    (root / dataset).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(first_patient)
    labels = labels if labels is not None else [i % 5 for i in range(n)]

    rows = []
    for i, lab in enumerate(labels):
        pid = first_patient + i
        stem = f"{pid}_left" if dataset == "eyepacs" else f"aptos_{pid:06d}"
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(root / dataset / f"{stem}.jpg"), img)
        src = (f"data/data/{stem}.jpeg" if dataset == "eyepacs"
               else f"train_images/train_images/{pid:06d}.png")
        rows.append({
            "image_path": src,
            "patient_id": str(pid) if dataset == "eyepacs" else f"aptos_{pid:06d}",
            "eye": "left",
            "label": int(lab),
            "dataset": dataset,
            "split": split,
        })
    return pd.DataFrame(rows)


def _cache(tmp_path: Path, n: int = 40, size: int = 224, labels=None) -> tuple[pd.DataFrame, Path]:
    """A synthetic cache in the real layout: {root}/eyepacs/{patient}_{eye}.jpg."""
    root = tmp_path / "cache"
    return _write_split(root, n=n, size=size, labels=labels), root


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
    for name in ("val", "test", "aptos_val", "aptos_test"):
        bad = df.assign(split=name)
        with pytest.raises(ValueError, match="train-only"):
            build_sampler(bad, "weighted_random")


def test_sampler_guard_fails_closed_not_open(tmp_path):
    """A substring blocklist passed `holdout`, `development`, `tuning` and
    `screening_2026`, and skipped entirely when the column was absent or all-NaN — so a
    frame that was a valid DRDataset input was a frame with no guard at all."""
    df, _ = _imbalanced(tmp_path)

    for name in ("holdout", "development", "tuning", "screening_2026", "eval_2026"):
        with pytest.raises(ValueError, match="train-only"):
            build_sampler(df.assign(split=name), "weighted_random")

    with pytest.raises(ValueError, match="no `split` column"):
        build_sampler(df.drop(columns=["split"]), "weighted_random")

    with pytest.raises(ValueError, match="NaN split"):
        build_sampler(df.assign(split=np.nan), "weighted_random")


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

def _partition(tmp_path: Path):
    """train/val/test over DISJOINT patients, in one shared cache root."""
    root = tmp_path / "cache"
    imbal = [0] * 300 + [1] * 60 + [2] * 90 + [3] * 20 + [4] * 12
    train = _write_split(root, n=len(imbal), labels=imbal, split="train",
                         first_patient=0)
    val = _write_split(root, n=100, split="val", first_patient=10_000)
    test = _write_split(root, n=100, split="test", first_patient=20_000)
    return train, val, test, root


def test_loaders_balance_train_and_leave_val_and_test_natural(tmp_path):
    """The whole of R2 in one assertion: the train loader's draw is balanced, and the
    val/test loaders deliver exactly the manifest, in order, unbalanced."""
    train, val, test, root = _partition(tmp_path)

    loaders = build_loaders(train, val, test, root, sampler_kind="weighted_random",
                            include_test=True, batch_size=16)

    assert loaders["train"].sampler is not None
    assert isinstance(loaders["val"].sampler, torch.utils.data.SequentialSampler)
    assert isinstance(loaders["test"].sampler, torch.utils.data.SequentialSampler)

    seen = [int(l) for _, lb, _ in loaders["val"] for l in lb]
    assert seen == val["label"].tolist(), "val must be the manifest, in order"

    natural = class_counts(test) / len(test)
    got = pd.Series([int(l) for _, lb, _ in loaders["test"] for l in lb])
    got = got.value_counts().reindex(range(5), fill_value=0).sort_index() / len(test)
    assert np.allclose(got.to_numpy(), natural.to_numpy())


def test_loaders_refuse_a_partition_that_shares_patients(tmp_path):
    """build_loaders is the one function that sees all three frames at once.
    load_arm_splits checks this, but every path that calls load_split directly bypasses
    it — a debug subset, a filtered run, a re-split."""
    train, val, _, root = _partition(tmp_path)
    leaking_val = pd.concat([val, train.iloc[:5]], ignore_index=True)

    with pytest.raises(AssertionError, match="LEAKAGE"):
        build_loaders(train, leaking_val, None, root, batch_size=8)


def test_loaders_do_not_hand_out_a_test_loader_unasked(tmp_path):
    """R3 — the test set is touched once. That should be an act, not an inheritance:
    load_arm_splits returns three frames, so the natural call passes three."""
    train, val, test, root = _partition(tmp_path)
    assert "test" not in build_loaders(train, val, test, root, batch_size=8)
    assert "test" in build_loaders(train, val, test, root, include_test=True,
                                   batch_size=8)


def test_loaders_never_augment_val_or_test(tmp_path):
    train, val, _, root = _partition(tmp_path)
    loaders = build_loaders(train, val, None, root, augment=AugmentConfig(),
                            batch_size=4)
    assert loaders["val"].dataset.augment is None
    assert loaders["train"].dataset.augment is not None


def test_train_loader_delivers_one_epoch_of_batches(tmp_path):
    train, val, _, root = _partition(tmp_path)
    loaders = build_loaders(train, val, None, root, sampler_kind="weighted_random",
                            batch_size=8)
    n = sum(len(lb) for _, lb, _ in loaders["train"])
    assert n == len(train)


def test_half_specified_normalisation_is_refused(tmp_path):
    """Passing mean without std silently reverted BOTH to ImageNet, discarding the
    caller's explicit value with no warning."""
    train, val, _, root = _partition(tmp_path)
    with pytest.raises(ValueError, match="both mean and std"):
        build_loaders(train, val, None, root, mean=(0.5, 0.5, 0.5), std=None)


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


# ----------------------------------------------------------------------------------
# Regressions from the leakage audit
# ----------------------------------------------------------------------------------

def test_a_shuffled_caller_index_is_refused(tmp_path):
    """The returned index is positional. Against a caller frame with a shuffled index,
    `.loc[idx]` silently returns a DIFFERENT row than the model saw — measured: ds[5]
    reported label 0 for 5_left.jpeg while the row actually loaded was 142_left.jpeg."""
    df, root = _cache(tmp_path, n=20)
    shuffled = df.sample(frac=1.0, random_state=0)          # index no longer 0..n-1
    with pytest.raises(ValueError, match="RangeIndex"):
        DRDataset(shuffled, root, train=False)

    ok = shuffled.reset_index(drop=True)
    ds = DRDataset(ok, root, train=False)
    assert ds.df.iloc[7]["image_path"] == ok.iloc[7]["image_path"]


def test_a_pooled_frame_with_duplicate_index_labels_is_refused(tmp_path):
    """pd.concat without ignore_index — the natural way to hand-roll arm F's pool — gives
    duplicate index labels, and `.loc[50]` then returns two rows with different grades."""
    df, root = _cache(tmp_path, n=20)
    pooled = pd.concat([df.iloc[:10], df.iloc[10:]])        # index 0..9, 10..19 is fine
    pooled = pd.concat([df.iloc[:10], df.iloc[:10]])        # this is not
    with pytest.raises(ValueError, match="duplicate image_path|RangeIndex"):
        DRDataset(pooled.reset_index(drop=True), root, train=False)


def test_physical_oversampling_is_refused_at_the_dataset(tmp_path):
    """§2.1: no duplicated rows in any manifest. test_no_leakage.py enforces that on the
    committed CSVs; nothing enforced it on the frame actually handed to a DataLoader, so
    `pd.concat([train] + [rare] * 8)` was accepted in silence."""
    df, root = _cache(tmp_path, n=20)
    rare = df[df["label"] == 4]
    over = pd.concat([df] + [rare] * 8, ignore_index=True)
    with pytest.raises(ValueError, match="oversampling is forbidden"):
        DRDataset(over, root, train=False)


def test_nan_label_is_refused_at_construction(tmp_path):
    """It used to survive construction, become a NaN sampler weight, and fail at the
    first draw of the first epoch — on Kaggle, forty minutes in."""
    df, root = _cache(tmp_path, n=8)
    df.loc[3, "label"] = np.nan
    with pytest.raises(ValueError, match="NaN label"):
        DRDataset(df, root, train=False)


def test_balance_table_exposes_dataset_origin_for_a_pooled_arm(tmp_path):
    """Arm F's central risk: origin correlates with severity, and the sampler amplifies
    the payoff for using it. DECISION-007 records the measured correlation; the run log
    has to show it too, or nothing ever will."""
    root = tmp_path / "cache"
    eye_labels = [0] * 150 + [1] * 10 + [2] * 20 + [3] * 10 + [4] * 10
    ap_labels = [0] * 10 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 35
    eye = _write_split(root, n=len(eye_labels), labels=eye_labels, split="train")
    ap = _write_split(root, n=len(ap_labels), labels=ap_labels, split="aptos_train",
                      first_patient=90_000, dataset="aptos")
    pooled = pd.concat([eye, ap], ignore_index=True)

    s = build_sampler(pooled, "weighted_random")
    table = describe_balance(pooled, s, draws=20_000)

    assert "p_aptos" in table.columns
    assert table.loc[4, "p_aptos"] > table.loc[0, "p_aptos"] + 0.3, (
        f"origin/severity correlation is invisible in the table:\n{table}"
    )


def test_arm_f_pooled_train_is_still_accepted(tmp_path):
    root = tmp_path / "cache"
    eye = _write_split(root, n=50, split="train")
    ap = _write_split(root, n=20, split="aptos_train", first_patient=90_000,
                      dataset="aptos")
    pooled = pd.concat([eye, ap], ignore_index=True)
    assert build_sampler(pooled, "weighted_random") is not None

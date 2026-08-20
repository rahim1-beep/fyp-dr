"""Class balancing — `WeightedRandomSampler`, train split only. CLAUDE.md §2.1.

The mechanism is fixed and must not drift:

    HOW    torch.utils.data.WeightedRandomSampler, weights = inverse class frequency
           computed from the TRAIN SPLIT ONLY
    WHEN   at batch-draw time, per epoch; nothing is materialised
    WHERE  the training split, only ever the training split

Explicitly forbidden by §2.1: physical oversampling, undersampling, duplicating or
deleting files on disk, and rebalancing val or test by any mechanism at all.

What this module actually enforces, as opposed to merely documenting:

- a sampler is refused on any split outside `TRAIN_SPLIT_NAMES` (allowlist), and refused
  outright if the frame cannot prove which split it is
- `build_loaders` asserts patient-level disjointness across train/val/test
- `DRDataset` refuses a frame with duplicate `image_path` rows, which is what physical
  oversampling looks like by the time it reaches a DataLoader

**Undersampling is not detectable here** and is not claimed to be: a train frame with
rows removed is indistinguishable from a legitimate debug subset. That one rests on
review and on `tests/test_no_leakage.py` over the committed CSVs.

The val/test refusal matters most because the mistake is invisible in a training log —
the val metric simply becomes a number about a distribution that does not exist.

Val and test keep the natural distribution permanently. That is what real screening looks
like and it is the only distribution on which a reported metric means anything.

Arms A, C and E draw naturally (`sampler: none`); B, D and F use the sampler. Which one
wins is a thesis chapter and is decided on validation QWK (R3) — so nothing here
hardcodes one arm's behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import DRDataset
from src.data.manifest import class_counts, sampler_weights

SAMPLER_KINDS = {"none", "weighted_random"}

# An ALLOWLIST, not a blocklist. A substring blocklist ("val"/"test") fails OPEN on every
# name nobody thought of - `holdout`, `development`, `tuning`, `screening_2026` all passed
# it - and fails open entirely when the column is absent or all-NaN. This fails closed.
TRAIN_SPLIT_NAMES = {"train", "aptos_train"}


def assert_patient_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    """R1: a patient appears in exactly one split. Raises naming the overlap.

    Checked on `patient_id`, never on `image_path` or on cache paths. The failure this
    catches is two DIFFERENT images of the SAME patient landing either side of the
    boundary, and those have different filenames, different cache files, and produce no
    duplicate anywhere - counting files will never find it.
    """
    for name, df in frames.items():
        if "patient_id" not in df.columns:
            raise ValueError(
                f"{name} frame has no patient_id column, so R1 cannot be checked. "
                "Load splits with src.data.manifest.load_split."
            )

    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(frames[a]["patient_id"]) & set(frames[b]["patient_id"])
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: {len(overlap)} patient(s) in both {a} and {b}, e.g. "
                    f"{sorted(map(str, overlap))[:5]}. R1 - a patient appears in exactly "
                    "one split."
                )


def build_sampler(
    train_df: pd.DataFrame,
    kind: str,
    *,
    seed: int = 42,
    num_samples: int | None = None,
) -> WeightedRandomSampler | None:
    """The sampler for ONE TRAINING SPLIT, or None for the natural distribution.

    `num_samples` defaults to `len(train_df)` so an epoch is the same number of gradient
    steps whether or not balancing is on. Two arms that see different amounts of data per
    epoch are not comparable, and the comparison is the point.

    `replacement=True` is what makes this reversible: rare classes are drawn more often
    within an epoch, and nothing on disk changes. With replacement=False the sampler would
    degenerate into a shuffle of the natural distribution.
    """
    if kind not in SAMPLER_KINDS:
        raise ValueError(f"sampler={kind!r}; expected one of {sorted(SAMPLER_KINDS)}")
    if kind == "none":
        return None

    if "split" not in train_df.columns:
        # The column being optional was the live hole: DRDataset requires only
        # {image_path, label, dataset}, so a frame that is a valid Dataset input was a
        # frame with no sampler guard at all.
        raise ValueError(
            "train_df has no `split` column, so the train-only rule cannot be checked. "
            "Load splits with src.data.manifest.load_split, which always provides it."
        )
    # NaN is a violation, not something to drop: an all-NaN column used to empty the set
    # and skip the check entirely.
    if train_df["split"].isna().any():
        raise ValueError(
            f"{int(train_df['split'].isna().sum())} row(s) have a NaN split value; a "
            "sampler cannot be built over a frame whose split membership is unknown"
        )
    forbidden = set(train_df["split"].unique()) - TRAIN_SPLIT_NAMES
    if forbidden:
        raise ValueError(
            f"refusing to build a sampler over {sorted(forbidden)}. Balancing is "
            f"train-only (CLAUDE.md §2.1); the only balanceable splits are "
            f"{sorted(TRAIN_SPLIT_NAMES)}. Val and test keep the natural distribution."
        )

    weights = sampler_weights(train_df)          # raises if a class is absent
    g = torch.Generator()
    g.manual_seed(seed)                          # R6

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights.to_numpy(), dtype=torch.double),
        num_samples=int(num_samples if num_samples is not None else len(train_df)),
        replacement=True,
        generator=g,
    )


def describe_balance(train_df: pd.DataFrame, sampler: WeightedRandomSampler | None,
                     draws: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Natural vs expected-drawn class distribution, for the run log and the write-up.

    Printed at the top of every training run: an arm whose sampler silently did nothing
    would otherwise look identical to arm A in every log it produces.

    On a POOLED frame (arm F) the table gains a `p_<dataset>` column: the share of each
    grade's rows that come from the minority dataset. That number is the arm's central
    risk and it is not small. On the committed splits P(aptos | grade) runs 0.065 at
    grade 0 to 0.291 at grade 4, and the sampler lifts grade 4 from 2.6% of the gradient
    to about 20% - so "APTOS camera implies severe" is a shortcut worth several times more
    under arm F WITH the sampler than under arm F alone. DECISION-007 records the measured
    correlation; this column is what keeps it visible in every run log.
    """
    counts = class_counts(train_df)
    natural = counts / counts.sum()

    if sampler is None:
        drawn = natural
    else:
        # Draw from a CLONE of the sampler, not from `sampler` itself. Iterating the real
        # one would advance its generator, so merely logging the balance table would
        # change the first epoch — a reproducibility bug (R6) that only appears when
        # someone removes the logging.
        g = torch.Generator()
        g.manual_seed(seed)
        clone = WeightedRandomSampler(
            weights=torch.as_tensor(sampler.weights, dtype=torch.double),
            num_samples=max(int(draws), 1),
            replacement=True,
            generator=g,
        )
        labels = train_df["label"].to_numpy()[np.fromiter(clone, dtype=np.int64)]
        drawn = (pd.Series(labels).value_counts().reindex(range(5), fill_value=0)
                 .sort_index() / len(labels))

    table = pd.DataFrame({
        "n": counts,
        "natural": natural.round(4),
        "drawn": drawn.round(4),
        "ratio": (drawn / natural.replace(0, pd.NA)).round(2),
    })

    if "dataset" in train_df.columns and train_df["dataset"].nunique() > 1:
        counts_by = train_df["dataset"].value_counts()
        minority = str(counts_by.index[-1])
        share = (train_df.assign(_m=(train_df["dataset"] == minority).astype(float))
                 .groupby("label")["_m"].mean()
                 .reindex(range(5)))
        table[f"p_{minority}"] = share.round(4)
    return table


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    cache_root: Path | str,
    *,
    sampler_kind: str = "none",
    include_test: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    mean=None,
    std=None,
    augment=None,
    image_size: int = 224,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """train/val/test DataLoaders with the partitioning rules enforced structurally.

    R3: `include_test` defaults to False. `load_arm_splits` returns three frames, so the
    natural call passes three, and a test loader that merely exists for thirty epochs is
    an invitation. Touching the test set should be an act, not an inheritance.

    Windows note (CLAUDE.md §7): `num_workers > 0` spawns processes, so any caller must
    be guarded by `if __name__ == "__main__":`. Default 0 for that reason.
    """
    # R1, at the only place all three frames are visible at once. manifest.load_arm_splits
    # checks this, but every path that calls load_split directly - a debug subset, a
    # filtered run, a re-split - bypasses it, and this function is downstream of all of
    # them.
    assert_patient_disjoint({"train": train_df, "val": val_df,
                             **({"test": test_df} if test_df is not None else {})})

    # Exactly one of mean/std silently reverted BOTH to ImageNet, discarding the caller's
    # explicit value with no warning.
    if (mean is None) != (std is None):
        raise ValueError("pass both mean and std, or neither")
    kw = {}
    if mean is not None and std is not None:
        kw = {"mean": tuple(mean), "std": tuple(std)}

    train_ds = DRDataset(train_df, cache_root, train=True, augment=augment,
                         image_size=image_size, **kw)
    # augment is NOT passed to val/test. Not a milder version — none.
    val_ds = DRDataset(val_df, cache_root, train=False, image_size=image_size, **kw)

    sampler = build_sampler(train_df, sampler_kind, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),   # torch forbids both; this is the correct pairing
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False, generator=g,
        ),
        # shuffle=False on val/test: the order is the manifest order, so predictions line
        # up with rows without a join.
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory),
    }

    if test_df is not None and include_test:
        test_ds = DRDataset(test_df, cache_root, train=False, image_size=image_size, **kw)
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
    return loaders

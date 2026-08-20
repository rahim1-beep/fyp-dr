"""Class balancing — `WeightedRandomSampler`, train split only. CLAUDE.md §2.1.

The mechanism is fixed and must not drift:

    HOW    torch.utils.data.WeightedRandomSampler, weights = inverse class frequency
           computed from the TRAIN SPLIT ONLY
    WHEN   at batch-draw time, per epoch; nothing is materialised
    WHERE  the training split, only ever the training split

Explicitly forbidden, and none of it is expressible through this module: physical
oversampling, undersampling, duplicating or deleting files on disk, and rebalancing val or
test by any mechanism at all. `build_loaders` refuses a sampler on val or test rather than
trusting the caller, because that mistake is invisible in a training log — the val metric
simply becomes a number about a distribution that does not exist.

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

    if "split" in train_df.columns:
        splits = set(train_df["split"].dropna().unique())
        # Arm F pools train + aptos_train, so more than one value is legitimate; a val or
        # test value in here is not, under any arm.
        forbidden = {s for s in splits if "val" in str(s) or "test" in str(s)}
        if forbidden:
            raise ValueError(
                f"refusing to build a sampler over {sorted(forbidden)}. Balancing is "
                "train-only (CLAUDE.md §2.1); val and test keep the natural distribution."
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

    return pd.DataFrame({
        "n": counts,
        "natural": natural.round(4),
        "drawn": drawn.round(4),
        "ratio": (drawn / natural.replace(0, pd.NA)).round(2),
    })


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    cache_root: Path | str,
    *,
    sampler_kind: str = "none",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    mean=None,
    std=None,
    augment=None,
    image_size: int = 224,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """train/val/test DataLoaders with the balancing rules enforced structurally.

    Windows note (CLAUDE.md §7): `num_workers > 0` spawns processes, so any caller must
    be guarded by `if __name__ == "__main__":`. Default 0 for that reason.
    """
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

    if test_df is not None:
        test_ds = DRDataset(test_df, cache_root, train=False, image_size=image_size, **kw)
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
    return loaders

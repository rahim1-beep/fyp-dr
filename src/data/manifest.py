"""The single supported way to read a split CSV.

Split CSVs carry a '#' provenance header (seed, generator, git SHA, fractions, class
distribution). `pandas.read_csv` does NOT skip those lines by default — it parses the first
one as the header row, silently producing a one-column DataFrame whose only column is
named "# fyp-dr train split — EyePACS".

That failure is quiet: the load succeeds, `len(df)` looks plausible, and everything
downstream breaks in confusing ways. So nothing in this project calls `read_csv` on a split
file directly. Use `load_split()`.

    from src.data.manifest import load_split, load_arm_splits

    train = load_split("train")                  # EyePACS train
    train, val, test = load_arm_splits("F")      # arm F: pooled train/val, EyePACS test
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO / "data" / "splits"

MANIFEST_COLUMNS = ["image_path", "patient_id", "eye", "label", "dataset", "split"]

# Arm F pools APTOS into train and val, but tests on EyePACS ONLY so it stays directly
# comparable to arms A-E on an identical test set. See DECISION-007.
ARM_F_SPLITS = {
    "train": ["train", "aptos_train"],
    "val": ["val", "aptos_val"],
    "test": ["test"],
}


def split_path(name: str) -> Path:
    return SPLIT_DIR / f"{name}.csv"


def read_header(name: str) -> list[str]:
    """The provenance comment block, for logging into a run config (R6)."""
    lines = split_path(name).read_text(encoding="utf-8").splitlines()
    return [l for l in lines if l.startswith("#")]


def load_split(name: str) -> pd.DataFrame:
    """Load one split CSV. `name` is e.g. 'train', 'val', 'test', 'aptos_train'."""
    path = split_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Generate splits with:\n"
            f"    python -m src.data.split --config configs/base.yaml"
        )

    # comment='#' is the whole point of this module.
    df = pd.read_csv(path, comment="#", dtype={"patient_id": "string"})

    if list(df.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"{path} has columns {list(df.columns)}, expected {MANIFEST_COLUMNS}. "
            "If the first column name starts with '#', the provenance header was parsed "
            "as data — read this file with comment='#'."
        )
    return df


def load_arm_splits(arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) for an ablation arm.

    Arms A-E use EyePACS alone. Arm F pools APTOS into train and val only.
    """
    arm = arm.upper()

    if arm == "F":
        parts = {k: pd.concat([load_split(n) for n in names], ignore_index=True)
                 for k, names in ARM_F_SPLITS.items()}
    elif arm in set("ABCDE"):
        parts = {k: load_split(k) for k in ("train", "val", "test")}
    else:
        raise ValueError(f"Unknown arm {arm!r}; expected one of A B C D E F")

    train, val, test = parts["train"], parts["val"], parts["test"]

    # R1 is enforced by tests/test_no_leakage.py, but a pooled arm builds its splits at
    # runtime, so re-check the invariant here rather than trusting the concat.
    for a, b, an, bn in ((train, val, "train", "val"),
                         (train, test, "train", "test"),
                         (val, test, "val", "test")):
        overlap = set(a["patient_id"]) & set(b["patient_id"])
        if overlap:
            raise AssertionError(
                f"LEAKAGE in arm {arm}: {len(overlap)} patients in both {an} and {bn}"
            )

    return train, val, test


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Label counts, always reindexed 0-4 so a missing class shows as 0 rather than
    vanishing from the index (failure mode 1)."""
    return df["label"].value_counts().reindex(range(5), fill_value=0).sort_index()


def sampler_weights(train: pd.DataFrame) -> pd.Series:
    """Per-sample weights for WeightedRandomSampler — inverse class frequency.

    R2: computed from the TRAINING SPLIT ONLY, and applied at batch-draw time. This is the
    only balancing mechanism in the project. Nothing is duplicated on disk, nothing is
    discarded, and val/test are never touched. See CLAUDE.md §2.1.
    """
    counts = class_counts(train)
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f"Training split has no examples of class(es) {missing}")
    return train["label"].map(1.0 / counts)

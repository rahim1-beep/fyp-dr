"""Select the Phase 2 visual-QA sample.

Produces the fixed list of images used for the before/after contact sheet, so the sheet is
reproducible and the same images can be re-rendered after any preprocessing change.

Design constraints (set by the user):
  - ~20 pairs, original and processed side by side at the same display size
  - WEIGHTED TOWARD GRADES 1 AND 2 (at least 6). Those are where circle-crop and Ben
    Graham either preserve microaneurysms and small haemorrhages or destroy them. Grade
    0 and 4 examples cannot answer that question.
  - 2-3 deliberately poor-quality inputs, because the web app will receive exactly that.

All QA images are drawn from the TRAIN split. Visual inspection is not model selection, so
this is not an R3 issue, but keeping val/test unopened is free hygiene.

Poor-quality selection is by file size, which is a proxy for "low detail" — a blank, very
dark, or failed capture compresses far smaller than a normal fundus photo. Only 21 of
35,126 files fall under 50 KB against a median of ~1,100 KB. True dark / over-exposed /
off-centre detection needs pixel statistics and is done in Phase 2 once the images are
readable on Kaggle; see `scan_quality()`.

Usage:
    python -m src.data.qa_sample --out docs/phase2_qa_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Clean examples per grade. Grades 1+2 get 8 of the 17 clean slots.
PER_GRADE = {0: 3, 1: 4, 2: 4, 3: 3, 4: 3}

# Deliberately poor-quality inputs, chosen from the size distribution.
# All are in the TRAIN split.
POOR_QUALITY = [
    ("3829_left", "smallest file in the dataset at 8.1 KB, grade 2"),
    ("39106_left", "16.6 KB, grade 3 — a rare-class image that may be degraded"),
    ("15942_right", "15.9 KB, grade 0"),
]

SEED = 42


def build(listing: Path, labels: Path, splits_dir: Path) -> pd.DataFrame:
    L = pd.read_csv(listing, header=None, names=["name", "size", "created"])
    L = L[L["name"].str.startswith("data/data/")].copy()
    L["image"] = L["name"].str.rsplit("/", n=1).str[-1].str.replace(".jpeg", "", regex=False)

    lab = pd.read_csv(labels)
    d = L.merge(lab, on="image", validate="one_to_one")
    d["kb"] = (d["size"] / 1024).round(1)

    train = pd.read_csv(splits_dir / "train.csv", comment="#")
    train["image"] = (
        train["image_path"].str.rsplit("/", n=1).str[-1].str.replace(".jpeg", "", regex=False)
    )
    d = d.merge(train[["image", "patient_id", "eye", "split"]], on="image", how="inner")

    poor_ids = [i for i, _ in POOR_QUALITY]
    reasons = dict(POOR_QUALITY)

    # Clean examples: exclude the poor-quality picks and the whole low-size tail, so a
    # "clean" slot never accidentally draws a degraded image.
    clean_pool = d[(~d["image"].isin(poor_ids)) & (d["kb"] >= 100)]

    rng = np.random.default_rng(SEED)
    picks = []
    for grade, n in PER_GRADE.items():
        pool = clean_pool[clean_pool["level"] == grade]
        idx = rng.choice(len(pool), size=n, replace=False)
        sel = pool.iloc[np.sort(idx)].copy()
        sel["category"] = "clean"
        sel["reason"] = f"representative grade {grade}"
        picks.append(sel)

    poor = d[d["image"].isin(poor_ids)].copy()
    poor["category"] = "poor_quality"
    poor["reason"] = poor["image"].map(reasons)
    picks.append(poor)

    out = pd.concat(picks, ignore_index=True)
    out = out.rename(columns={"level": "label"})
    out["image_path"] = "data/data/" + out["image"] + ".jpeg"
    return out[["image_path", "image", "patient_id", "eye", "label",
                "kb", "split", "category", "reason"]].sort_values(
        ["category", "label", "image"], ascending=[False, True, True]
    ).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", type=Path,
                    default=REPO / "data/raw/eyepacs/remote_listing.csv")
    ap.add_argument("--labels", type=Path,
                    default=REPO / "data/raw/eyepacs/trainLabels.csv")
    ap.add_argument("--splits", type=Path, default=REPO / "data/splits")
    ap.add_argument("--out", type=Path, default=REPO / "docs/phase2_qa_sample.csv")
    args = ap.parse_args()

    df = build(args.listing, args.labels, args.splits)

    header = [
        "# Phase 2 visual-QA sample — fixed list for the before/after contact sheet",
        f"# seed: {SEED}   generator: src/data/qa_sample.py",
        "# all images drawn from the TRAIN split",
        f"# {len(df)} images: "
        + ", ".join(f"grade {g}: {n}" for g, n in df['label'].value_counts().sort_index().items()),
        f"# grades 1+2: {int(df['label'].isin([1,2]).sum())} (user requirement: >= 6)",
        f"# poor-quality: {int((df['category']=='poor_quality').sum())}",
        "# NOTE: readers must use pandas.read_csv(..., comment='#')",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, index=False, lineterminator="\n")

    print("\n".join(header))
    print()
    print(df.to_string(index=False))
    print(f"\nwritten to {args.out}")

    total_kb = df["kb"].sum()
    print(f"\ndownload cost for the QA sample: {total_kb/1024:.1f} MB "
          f"({len(df)} images) — trivial, safe to pull locally or on Kaggle")


if __name__ == "__main__":
    main()

"""Patient-level stratified splitting. This file implements R1 and nothing else.

The rule (R1): a patient appears in EXACTLY ONE of train / val / test. We split the
PATIENT LIST, never the image list, and never before grouping. In EyePACS 2,240 patients
(12.8%) have eyes with different grades, so the two eyes are correlated but not duplicates
— an image-level split would leak genuine per-patient signal.

Stratification is on each patient's MAX grade, computed before splitting. A patient with a
grade-0 left eye and a grade-3 right eye is a grade-3 patient for stratification purposes,
because that is the severity a screening system must not miss.

Balancing is NOT done here and never will be (R2). Balancing is train-only and happens at
batch-draw time via WeightedRandomSampler. No file is duplicated or deleted, and val/test
keep their natural distribution permanently.

Usage:
    python -m src.data.split --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MANIFEST_COLUMNS = ["image_path", "patient_id", "eye", "label", "dataset", "split"]
VALID_EYES = ("left", "right")


# --------------------------------------------------------------------------------------
# Manifest builders
# --------------------------------------------------------------------------------------

def build_eyepacs_manifest(labels_csv: Path, image_subdir: str, image_ext: str) -> pd.DataFrame:
    """EyePACS: filenames are {patientID}_{left|right}, labels from the official CSV.

    Note the join trap: CSV `image` values carry NO extension ("10_left") while the files
    on disk do ("10_left.jpeg"). We build the path; we never join on a bare filename.
    """
    df = pd.read_csv(labels_csv, dtype={"image": "string"})

    cols = list(df.columns)
    if len(cols) != 2:
        raise SystemExit(f"Expected 2 columns in {labels_csv}, found {cols}")
    image_col, label_col = cols

    df[label_col] = df[label_col].astype(int)

    distinct = sorted(df[label_col].unique().tolist())
    if distinct != [0, 1, 2, 3, 4]:
        raise SystemExit(f"Expected labels [0,1,2,3,4], found {distinct}")

    parts = df[image_col].str.partition("_")
    patient_id = parts[0]
    eye = parts[2]

    bad = df[~eye.isin(VALID_EYES)]
    if len(bad):
        raise SystemExit(f"{len(bad)} filenames did not parse as patient_eye: "
                         f"{bad[image_col].head(10).tolist()}")

    subdir = image_subdir.rstrip("/")
    return pd.DataFrame({
        "image_path": subdir + "/" + df[image_col] + image_ext,
        "patient_id": patient_id,
        "eye": eye,
        "label": df[label_col],
        "dataset": "eyepacs",
        "split": pd.NA,
    })


def build_aptos_manifest(aptos_dir: Path) -> pd.DataFrame:
    """APTOS: pooled across the author-provided split files (DECISION-004).

    APTOS `id_code` values are anonymised hashes with NO patient linkage, so each image is
    treated as its own patient. This is a documented limitation, not an assumption.

    Directory layout is double-nested: train_images/train_images/{id}.png
    """
    sources = {"train_1.csv": "train_images", "valid.csv": "val_images", "test.csv": "test_images"}

    frames = []
    for csv_name, img_dir in sources.items():
        path = aptos_dir / csv_name
        if not path.exists():
            raise SystemExit(f"Missing APTOS labels file: {path}")
        d = pd.read_csv(path, dtype={"id_code": "string"})
        d["diagnosis"] = d["diagnosis"].astype(int)
        d["image_path"] = f"{img_dir}/{img_dir}/" + d["id_code"] + ".png"
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)

    if df["id_code"].duplicated().any():
        dupes = df.loc[df["id_code"].duplicated(), "id_code"].tolist()
        raise SystemExit(f"Duplicate APTOS id_code across split files: {dupes[:10]}")

    distinct = sorted(df["diagnosis"].unique().tolist())
    if distinct != [0, 1, 2, 3, 4]:
        raise SystemExit(f"Expected APTOS labels [0,1,2,3,4], found {distinct}")

    return pd.DataFrame({
        "image_path": df["image_path"],
        # one image == one patient. No linkage exists to do better.
        "patient_id": "aptos_" + df["id_code"],
        "eye": "unknown",
        "label": df["diagnosis"],
        "dataset": "aptos",
        "split": pd.NA,
    })


# --------------------------------------------------------------------------------------
# The split itself
# --------------------------------------------------------------------------------------

def assign_splits(manifest: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Group by patient -> max grade -> stratify the PATIENT list -> 70/15/15.

    Every patient lands in exactly one split. Returns the manifest with `split` filled.
    """
    total = fractions["train"] + fractions["val"] + fractions["test"]
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"Split fractions must sum to 1.0, got {total}")

    patients = (
        manifest.groupby("patient_id", sort=True)["label"]
        .max()
        .rename("max_grade")
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}

    # Stratify by shuffling WITHIN each max-grade stratum, then cutting. This keeps the
    # per-stratum proportions exact rather than approximately right.
    for grade in sorted(patients["max_grade"].unique()):
        ids = patients.loc[patients["max_grade"] == grade, "patient_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]

        n = len(ids)
        n_train = int(round(fractions["train"] * n))
        n_val = int(round(fractions["val"] * n))
        # test takes the remainder so the three always sum to n exactly
        n_test = n - n_train - n_val

        # A stratum too small to populate all three splits produces an empty val or test
        # class — which downstream becomes an all-zero confusion-matrix column, i.e. the
        # exact silent failure CLAUDE.md's detection rule exists to catch. At 70/15/15 this
        # bites for n < 4. Harmless on the full dataset (smallest real stratum is 445
        # patients) but a filtered subset or debug sample would hit it quietly.
        if n_val == 0 or n_test == 0:
            raise SystemExit(
                f"Stratum max_grade={grade} has only {n} patients — too few to populate "
                f"all three splits at {fractions} (train={n_train}, val={n_val}, "
                f"test={n_test}). This would produce an empty class in val or test and an "
                f"all-zero confusion-matrix column. Refusing to write a silently broken "
                f"split."
            )

        for pid in ids[:n_train]:
            assignment[pid] = "train"
        for pid in ids[n_train:n_train + n_val]:
            assignment[pid] = "val"
        for pid in ids[n_train + n_val:]:
            assignment[pid] = "test"

    out = manifest.copy()
    out["split"] = out["patient_id"].map(assignment)

    if out["split"].isna().any():
        raise SystemExit(f"{int(out['split'].isna().sum())} images were left unassigned")

    return out


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, shell=(sys.platform == "win32"))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_split_csv(df: pd.DataFrame, path: Path, *, seed: int, split_name: str,
                    fractions: dict[str, float], source: str) -> None:
    """Write one split CSV with a provenance header.

    IMPORTANT: the header lines start with '#'. Every reader of these files must pass
    `comment='#'` to pandas.read_csv, or the header will be parsed as data.
    """
    counts = df["label"].value_counts().sort_index()
    dist = ", ".join(f"{g}:{c}" for g, c in counts.items())

    header = [
        f"# fyp-dr {split_name} split — {source}",
        f"# RNG seed: {seed}   (recorded in configs/base.yaml as split.seed)",
        f"# generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# generator: src/data/split.py @ git {git_sha()}",
        # numpy's Generator stream is not contractually stable across versions (NEP 19),
        # so the version that produced the permutation is part of the provenance.
        f"# python: {sys.version.split()[0]}  numpy: {np.__version__}  pandas: {pd.__version__}",
        f"# fractions: train={fractions['train']} val={fractions['val']} test={fractions['test']}",
        "# stratified on: each patient's MAX grade, computed before splitting",
        "# R1: patient-level — a patient appears in exactly one of train/val/test",
        "# R2: NOT balanced. Balancing is train-only, at batch-draw time via",
        "#     WeightedRandomSampler. Val/test keep the natural distribution permanently.",
        f"# images: {len(df)}   patients: {df['patient_id'].nunique()}",
        f"# class distribution: {dist}",
        "# NOTE: readers must use pandas.read_csv(..., comment='#')",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        df[MANIFEST_COLUMNS].to_csv(fh, index=False, lineterminator="\n")


def summarise(manifest: pd.DataFrame, title: str) -> str:
    """Side-by-side class distribution and patient counts per split."""
    L = [f"\n{title}", "=" * len(title)]

    splits = ["train", "val", "test"]
    present = [s for s in splits if (manifest["split"] == s).any()]

    grades = sorted(manifest["label"].unique())
    natural = manifest["label"].value_counts(normalize=True).sort_index() * 100

    L.append("\nIMAGE-LEVEL CLASS DISTRIBUTION")
    head = f"  {'grade':<7}" + "".join(f"{s:>20}" for s in present) + f"{'NATURAL':>12}"
    L.append(head)
    L.append("  " + "-" * (len(head) - 2))
    for g in grades:
        row = f"  {g:<7}"
        for s in present:
            sub = manifest[manifest["split"] == s]
            n = int((sub["label"] == g).sum())
            pct = 100 * n / len(sub) if len(sub) else 0.0
            row += f"{n:>12,} {pct:>6.2f}%"
        row += f"{natural.get(g, 0.0):>11.2f}%"
        L.append(row)

    total_row = f"  {'TOTAL':<7}"
    for s in present:
        sub = manifest[manifest["split"] == s]
        total_row += f"{len(sub):>12,} {100.0:>6.2f}%"
    L.append("  " + "-" * (len(head) - 2))
    L.append(total_row)

    L.append("\nPATIENT COUNTS")
    L.append(f"  {'split':<10}{'patients':>12}{'images':>12}{'img/patient':>14}")
    for s in present:
        sub = manifest[manifest["split"] == s]
        np_ = sub["patient_id"].nunique()
        L.append(f"  {s:<10}{np_:>12,}{len(sub):>12,}{len(sub)/np_:>14.2f}")
    L.append(f"  {'TOTAL':<10}{manifest['patient_id'].nunique():>12,}{len(manifest):>12,}")

    L.append("\nPATIENT-LEVEL MAX-GRADE DISTRIBUTION (what was stratified)")
    pat = manifest.groupby("patient_id").agg(max_grade=("label", "max"), split=("split", "first"))
    head2 = f"  {'max grade':<11}" + "".join(f"{s:>20}" for s in present)
    L.append(head2)
    for g in sorted(pat["max_grade"].unique()):
        row = f"  {g:<11}"
        for s in present:
            sub = pat[pat["split"] == s]
            n = int((sub["max_grade"] == g).sum())
            pct = 100 * n / len(sub) if len(sub) else 0.0
            row += f"{n:>12,} {pct:>6.2f}%"
        L.append(row)

    return "\n".join(L)


# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/raw/eyepacs/trainLabels.csv"))
    ap.add_argument("--aptos-dir", type=Path, default=Path("data/raw/aptos"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--skip-aptos", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = cfg["split"]["seed"]
    fractions = {k: cfg["split"][k] for k in ("train", "val", "test")}
    ep_cfg = cfg["data"]["eyepacs"]

    print(f"seed={seed}  fractions={fractions}")

    # ---- EyePACS (primary; arms A-E) ----
    ep = build_eyepacs_manifest(args.labels_csv, ep_cfg["image_subdir"], ep_cfg["image_ext"])
    ep = assign_splits(ep, fractions, seed)

    for s in ("train", "val", "test"):
        write_split_csv(ep[ep["split"] == s], args.out_dir / f"{s}.csv",
                        seed=seed, split_name=s, fractions=fractions, source="EyePACS")
    print(summarise(ep, "EyePACS — patient-level split (arms A-E)"))

    # ---- APTOS (arm F training pool; DECISION-007) ----
    if not args.skip_aptos:
        ap_m = build_aptos_manifest(args.aptos_dir)
        # Same procedure, same seed. Each image is its own patient (no linkage exists),
        # so this reduces to a stratified image split — which is correct ONLY because
        # there is genuinely nothing to group by.
        ap_m = assign_splits(ap_m, fractions, seed)
        for s in ("train", "val", "test"):
            write_split_csv(ap_m[ap_m["split"] == s], args.out_dir / f"aptos_{s}.csv",
                            seed=seed, split_name=f"aptos_{s}", fractions=fractions,
                            source="APTOS 2019 (pooled; author split discarded)")
        print(summarise(ap_m, "APTOS — one image per patient (arm F)"))

    print(f"\nwritten to {args.out_dir}/")


if __name__ == "__main__":
    main()

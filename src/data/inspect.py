"""EyePACS inspection and reconciliation.

Phase 1 step 2. Produces a report, not a split. Nothing here writes a split CSV.

Design notes:
  - The labels CSV schema is DISCOVERED and reported, never assumed (failure mode 1).
  - The remote file listing comes from the Kaggle API (`kaggle datasets files`), which
    returns filenames + sizes without transferring a single image byte. This lets the
    full CSV-to-disk reconciliation run on a laptop that will never hold the 35.3 GB
    dataset.
  - Mismatches are REPORTED in both directions. Nothing is dropped silently.

Usage:
    python -m src.data.inspect --listing <files.csv> --labels <trainLabels.csv> \
        [--json-out <report.json>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# EyePACS filenames are "{patientID}_{eye}". Split on the FIRST underscore only.
VALID_EYES = ("left", "right")


@dataclass
class Report:
    """Everything the reconciliation learns. Serialised verbatim to JSON."""

    schema: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    class_distribution: dict[str, Any] = field(default_factory=dict)
    patients: dict[str, Any] = field(default_factory=dict)
    mismatches: dict[str, Any] = field(default_factory=dict)
    anomalies: dict[str, Any] = field(default_factory=dict)
    assertions: dict[str, Any] = field(default_factory=dict)


def load_listing(path: Path) -> pd.DataFrame:
    """Load the Kaggle remote file listing (name,size,creationDate; no header)."""
    df = pd.read_csv(path, header=None, names=["name", "size", "created"])
    df["name"] = df["name"].astype("string").str.strip()
    return df


def load_labels(path: Path) -> pd.DataFrame:
    # dtype=str first so we can inspect the raw label values before coercing.
    return pd.read_csv(path, dtype="string")


def parse_stem(stem: str) -> tuple[str | None, str | None]:
    """'10_left' -> ('10', 'left'). Returns (None, None) if it does not parse."""
    if "_" not in stem:
        return None, None
    patient_id, _, eye = stem.partition("_")
    if not patient_id or eye not in VALID_EYES:
        return None, None
    return patient_id, eye


def reconcile(listing: pd.DataFrame, labels: pd.DataFrame, image_subdir: str) -> Report:
    rep = Report()

    # ---- 1. Schema, discovered not assumed -------------------------------------
    cols = list(labels.columns)
    rep.schema = {
        "columns": cols,
        "n_columns": len(cols),
        "raw_dtypes": {c: str(labels[c].dtype) for c in cols},
    }
    if len(cols) != 2:
        raise SystemExit(f"Expected 2 columns in the labels CSV, found {len(cols)}: {cols}")

    image_col, label_col = cols[0], cols[1]
    rep.schema["image_column"] = image_col
    rep.schema["label_column"] = label_col

    # Label column must be cleanly integer-valued. Report before coercing.
    raw_label_values = sorted(labels[label_col].dropna().unique().tolist())
    non_integer = [v for v in raw_label_values if not str(v).strip().lstrip("-").isdigit()]
    rep.schema["distinct_label_values_raw"] = raw_label_values
    rep.schema["non_integer_label_values"] = non_integer
    if non_integer:
        raise SystemExit(f"Non-integer label values present: {non_integer}")

    labels = labels.assign(**{label_col: labels[label_col].astype(int)})
    distinct_labels = sorted(labels[label_col].unique().tolist())
    rep.schema["distinct_label_values"] = distinct_labels
    rep.schema["inferred_label_dtype"] = "int64"

    # ---- 2. Counts --------------------------------------------------------------
    prefix = image_subdir.rstrip("/") + "/"
    images = listing[listing["name"].str.startswith(prefix, na=False)].copy()
    non_images = listing[~listing["name"].str.startswith(prefix, na=False)]

    images["stem"] = images["name"].str.rsplit("/", n=1).str[-1].str.replace(
        r"\.jpeg$", "", regex=True
    )

    rep.counts = {
        "listing_entries_total": int(len(listing)),
        "listing_image_files": int(len(images)),
        "listing_non_image_entries": non_images["name"].tolist(),
        "labels_csv_rows": int(len(labels)),
        "labels_csv_unique_ids": int(labels[image_col].nunique()),
        "image_bytes_total": int(images["size"].sum()),
        "image_bytes_mean": float(images["size"].mean()),
        "image_bytes_min": int(images["size"].min()),
        "image_bytes_max": int(images["size"].max()),
    }

    # ---- 3. Mismatches, BOTH directions ----------------------------------------
    csv_ids = set(labels[image_col].tolist())
    disk_ids = set(images["stem"].tolist())

    in_csv_not_on_disk = sorted(csv_ids - disk_ids)
    on_disk_not_in_csv = sorted(disk_ids - csv_ids)

    rep.mismatches = {
        "in_csv_not_on_disk_count": len(in_csv_not_on_disk),
        "in_csv_not_on_disk": in_csv_not_on_disk[:100],
        "on_disk_not_in_csv_count": len(on_disk_not_in_csv),
        "on_disk_not_in_csv": on_disk_not_in_csv[:100],
        "matched_count": len(csv_ids & disk_ids),
        "truncated_to_first_100": (
            len(in_csv_not_on_disk) > 100 or len(on_disk_not_in_csv) > 100
        ),
    }

    # ---- 4. Anomalies -----------------------------------------------------------
    dup_csv = labels[labels[image_col].duplicated(keep=False)]
    dup_disk = images[images["stem"].duplicated(keep=False)]

    parsed = labels[image_col].map(parse_stem)
    labels = labels.assign(
        patient_id=[p for p, _ in parsed],
        eye=[e for _, e in parsed],
    )
    unparseable = labels[labels["patient_id"].isna()][image_col].tolist()

    zero_byte = images[images["size"] == 0]["name"].tolist()

    rep.anomalies = {
        "duplicate_csv_ids_count": int(dup_csv[image_col].nunique()),
        "duplicate_csv_ids": sorted(dup_csv[image_col].unique().tolist())[:50],
        "duplicate_disk_stems_count": int(dup_disk["stem"].nunique()),
        "duplicate_disk_stems": sorted(dup_disk["stem"].unique().tolist())[:50],
        "unparseable_filenames_count": len(unparseable),
        "unparseable_filenames": unparseable[:50],
        "zero_byte_files_count": len(zero_byte),
        "zero_byte_files": zero_byte[:50],
    }

    # ---- 5. Class distribution --------------------------------------------------
    counts = labels[label_col].value_counts().sort_index()
    total = int(counts.sum())
    rep.class_distribution = {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "percent": {int(k): round(100.0 * v / total, 3) for k, v in counts.items()},
        "total": total,
        "majority_class": int(counts.idxmax()),
        "majority_class_rate_percent": round(100.0 * counts.max() / total, 3),
        "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 2),
    }

    # ---- 6. Patients ------------------------------------------------------------
    valid = labels[labels["patient_id"].notna()]
    per_patient = valid.groupby("patient_id")
    images_per_patient = per_patient.size()
    ipp_dist = Counter(images_per_patient.tolist())

    max_grade = per_patient[label_col].max()
    mg_counts = max_grade.value_counts().sort_index()

    # Patients whose two eyes disagree on grade — clinically normal, worth reporting.
    grade_spread = per_patient[label_col].agg(lambda s: int(s.max() - s.min()))

    eyes_per_patient = per_patient["eye"].apply(lambda s: tuple(sorted(set(s))))
    not_exactly_two_eyes = images_per_patient[images_per_patient != 2]

    rep.patients = {
        "n_patients": int(images_per_patient.size),
        "images_per_patient_distribution": {int(k): int(v) for k, v in sorted(ipp_dist.items())},
        "patients_without_exactly_two_images_count": int(len(not_exactly_two_eyes)),
        "patients_without_exactly_two_images": sorted(not_exactly_two_eyes.index.tolist())[:50],
        "patients_missing_an_eye_count": int(
            sum(1 for t in eyes_per_patient if set(t) != {"left", "right"})
        ),
        "max_grade_distribution": {int(k): int(v) for k, v in mg_counts.items()},
        "max_grade_percent": {
            int(k): round(100.0 * v / int(mg_counts.sum()), 3) for k, v in mg_counts.items()
        },
        "patients_with_asymmetric_grades": int((grade_spread > 0).sum()),
        "grade_spread_distribution": {
            int(k): int(v) for k, v in grade_spread.value_counts().sort_index().items()
        },
    }

    # ---- 7. Assertions (fail loudly) -------------------------------------------
    n_distinct = len(distinct_labels)
    rep.assertions = {
        "exactly_5_distinct_labels": n_distinct == 5,
        "labels_are_0_to_4": distinct_labels == [0, 1, 2, 3, 4],
        "no_duplicate_csv_ids": rep.anomalies["duplicate_csv_ids_count"] == 0,
        "no_unparseable_filenames": rep.anomalies["unparseable_filenames_count"] == 0,
        "csv_rows_equal_disk_files": (
            rep.counts["labels_csv_rows"] == rep.counts["listing_image_files"]
        ),
        "perfect_bidirectional_match": (
            rep.mismatches["in_csv_not_on_disk_count"] == 0
            and rep.mismatches["on_disk_not_in_csv_count"] == 0
        ),
        "no_zero_byte_files": rep.anomalies["zero_byte_files_count"] == 0,
    }
    return rep


def render(rep: Report) -> str:
    """Human-readable report for the sign-off gate."""
    L: list[str] = []
    a = L.append

    a("=" * 78)
    a("EyePACS RECONCILIATION REPORT — Phase 1 step 2")
    a("=" * 78)

    a("\n-- 1. LABELS CSV SCHEMA (discovered, not assumed) --")
    a(f"  columns          : {rep.schema['columns']}")
    a(f"  image column     : '{rep.schema['image_column']}'  (dtype: string)")
    a(f"  label column     : '{rep.schema['label_column']}'  (dtype: {rep.schema['inferred_label_dtype']})")
    a(f"  rows             : {rep.counts['labels_csv_rows']:,}")
    a(f"  distinct labels  : {rep.schema['distinct_label_values']}")

    a("\n-- 2. FILE COUNTS --")
    a(f"  listing entries total     : {rep.counts['listing_entries_total']:,}")
    a(f"  image files               : {rep.counts['listing_image_files']:,}")
    a(f"  labels CSV rows           : {rep.counts['labels_csv_rows']:,}")
    a(f"  unique IDs in CSV         : {rep.counts['labels_csv_unique_ids']:,}")
    gb = rep.counts["image_bytes_total"] / 1024**3
    a(f"  total image bytes         : {gb:.2f} GB")
    a(
        f"  per-image size            : min {rep.counts['image_bytes_min']/1024:.0f} KB"
        f" / mean {rep.counts['image_bytes_mean']/1024:.0f} KB"
        f" / max {rep.counts['image_bytes_max']/1024:.0f} KB"
    )

    a("\n-- 3. CSV <-> DISK RECONCILIATION (both directions) --")
    a(f"  matched                   : {rep.mismatches['matched_count']:,}")
    a(f"  in CSV but NOT on disk    : {rep.mismatches['in_csv_not_on_disk_count']:,}")
    if rep.mismatches["in_csv_not_on_disk"]:
        a(f"      {rep.mismatches['in_csv_not_on_disk'][:20]}")
    a(f"  on disk but NOT in CSV    : {rep.mismatches['on_disk_not_in_csv_count']:,}")
    if rep.mismatches["on_disk_not_in_csv"]:
        a(f"      {rep.mismatches['on_disk_not_in_csv'][:20]}")

    a("\n-- 4. ANOMALIES --")
    an = rep.anomalies
    a(f"  duplicate CSV ids         : {an['duplicate_csv_ids_count']}")
    a(f"  duplicate disk stems      : {an['duplicate_disk_stems_count']}")
    a(f"  unparseable filenames     : {an['unparseable_filenames_count']}")
    a(f"  zero-byte files           : {an['zero_byte_files_count']}")

    a("\n-- 5. CLASS DISTRIBUTION (image level) --")
    cd = rep.class_distribution
    a(f"  {'grade':<8}{'count':>10}{'percent':>10}")
    for k in sorted(cd["counts"]):
        a(f"  {k:<8}{cd['counts'][k]:>10,}{cd['percent'][k]:>9.2f}%")
    a(f"  {'TOTAL':<8}{cd['total']:>10,}")
    a(f"  majority class {cd['majority_class']} at {cd['majority_class_rate_percent']:.2f}%"
      f"  |  imbalance ratio {cd['imbalance_ratio_max_over_min']}:1")

    a("\n-- 6. PATIENTS --")
    p = rep.patients
    a(f"  distinct patients         : {p['n_patients']:,}")
    a("  images per patient        :")
    for k, v in p["images_per_patient_distribution"].items():
        a(f"      {k} image(s): {v:,} patients")
    a(f"  patients without exactly 2: {p['patients_without_exactly_two_images_count']:,}")
    a(f"  patients missing an eye   : {p['patients_missing_an_eye_count']:,}")
    a("\n  MAX-GRADE distribution (this is what the split stratifies on):")
    a(f"  {'max grade':<12}{'patients':>10}{'percent':>10}")
    for k in sorted(p["max_grade_distribution"]):
        a(f"  {k:<12}{p['max_grade_distribution'][k]:>10,}{p['max_grade_percent'][k]:>9.2f}%")
    a(f"\n  patients whose eyes disagree on grade: {p['patients_with_asymmetric_grades']:,}")
    a("  grade spread (max-min) within a patient:")
    for k, v in p["grade_spread_distribution"].items():
        a(f"      spread {k}: {v:,} patients")

    a("\n-- 7. ASSERTIONS --")
    for k, v in rep.assertions.items():
        a(f"  [{'PASS' if v else 'FAIL'}] {k}")

    a("\n" + "=" * 78)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--image-subdir", default="data/data")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rep = reconcile(load_listing(args.listing), load_labels(args.labels), args.image_subdir)
    print(render(rep))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rep.__dict__, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()

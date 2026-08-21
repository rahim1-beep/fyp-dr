"""Reconcile the preprocessed cache against the split CSVs, row for row.

The `leakage-auditor` requires this before the cache is used or published. The cache is
flat and split-agnostic (DECISION-012), which is the right design but means nothing about
the directory listing tells you whether it matches the splits. This module is what makes
that check explicit rather than assumed.

Five questions, all of which have to be answered "yes" before anything trains:

  1. Is every PATIENT in exactly one split?           (R1 - the leakage question itself)
  2. Does every split row have a cache file?          (missing -> a silently short epoch)
  3. Does every cache file belong to a split row?     (orphan  -> an image from nowhere)
  4. Does every file decode, at the configured size?  (corrupt -> a crash mid-epoch, or
                                                       worse, a black image that trains)
  5. Does any image map to more than one cache file?  (collision -> one file shared across
                                                       the train/test boundary)

(1) IS THE ONE THAT MATTERS AND IT IS NOT (5). A patient with a left eye in train and a
right eye in test produces two different filenames, two different cache files, zero
collisions and zero orphans - and (5) reports PASSED. That is the textbook R1 violation
this project exists to avoid, and an earlier version of this file certified it with exit
code 0. Patient identity is checked directly, on `patient_id`.

`tests/test_no_leakage.py` checks the committed CSVs. This runs on Kaggle against
whatever CSVs were uploaded, so it cannot assume those tests ever ran here.

Nothing here drops, moves, or rewrites anything. It reports and exits non-zero.

Usage:
    python -m src.data.reconcile_cache --cache-root /kaggle/working/processed \
        --stats /kaggle/working/processed_stats.csv \
        --stats /kaggle/working/aptos_stats.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.manifest import load_split, read_header
from src.data.preprocess import cache_relpath, load_preprocess_config

REPO = Path(__file__).resolve().parents[2]
EYEPACS_SPLITS = ["train", "val", "test"]
APTOS_SPLITS = ["aptos_train", "aptos_val", "aptos_test"]

# Which splits are the same side of the partition. Arm F pools eyepacs+aptos within a
# role, so `train` and `aptos_train` may legitimately share nothing but must both be
# disjoint from every val and test split.
ROLES = {"train": "train", "aptos_train": "train",
         "val": "val", "aptos_val": "val",
         "test": "test", "aptos_test": "test"}

PROVENANCE_FILE = "_cache_provenance.json"


def expected_rows(splits: list[str]) -> pd.DataFrame:
    """Every split row with the cache path it should have produced."""
    frames = []
    for name in splits:
        df = load_split(name)
        df = df.assign(
            split_file=name,
            cache_path=[
                cache_relpath(p, d) for p, d in zip(df["image_path"], df["dataset"])
            ],
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def patient_overlaps(exp: pd.DataFrame) -> list[str]:
    """R1, on patient identity. Returns one message per offending pair of roles."""
    if "patient_id" not in exp.columns:
        return ["split CSVs have no patient_id column, so R1 cannot be checked"]

    roles = exp["split_file"].map(ROLES)
    unknown = sorted(set(exp.loc[roles.isna(), "split_file"]))
    problems = []
    if unknown:
        problems.append(
            f"split file(s) {unknown} have no known role (train/val/test); refusing to "
            "guess which side of the partition they are on"
        )

    by_role = {r: set(g["patient_id"]) for r, g in exp.assign(_r=roles).groupby("_r")}
    names = sorted(by_role)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = by_role[a] & by_role[b]
            if overlap:
                problems.append(
                    f"R1 VIOLATION: {len(overlap)} patient(s) appear in both {a} and {b}, "
                    f"e.g. {sorted(map(str, overlap))[:5]}"
                )
    return problems


def check_provenance(root: Path, cfg) -> list[str]:
    """Compare the cache's provenance sidecar against the config being reconciled.

    Without this the check is content-blind: a cache half-built before DECISION-016 and
    half after - two different `mask_erode_frac` values, two different image domains -
    reconciles perfectly clean, because every count matches and every file decodes.
    """
    path = root / PROVENANCE_FILE
    if not path.exists():
        return [f"no {PROVENANCE_FILE} in the cache; it was built by a version of "
                "preprocess.py that did not record its config, so the pipeline that "
                "produced these files cannot be verified"]

    entries = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    if not entries:
        return [f"{PROVENANCE_FILE} records no runs"]

    want = asdict(cfg)
    problems = []
    for e in entries:
        got = e.get("preprocess", {})
        diff = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if diff:
            problems.append(
                f"cache run {e.get('started', '?')} ({e.get('src_root', '?')}) was built "
                f"with a different preprocess config: {diff} (got, wanted)"
            )
    return problems


def decode_check(paths: list[Path], size: int, sample: int, seed: int = 42) -> list[str]:
    """Decode `sample` randomly chosen files (all of them if sample <= 0).

    Full decoding of 38,788 JPEGs costs minutes of a Kaggle session for a check whose
    failure mode is systematic — a broken writer breaks every file, not one. A fixed-seed
    sample catches that; `--decode-all` exists for the paranoid final pass.
    """
    if not paths:
        return []
    if 0 < sample < len(paths):
        # Stratified by dataset directory. An unstratified 400-of-38,788 sample gives
        # APTOS about 9% of the checks purely because it is 9% of the cache, and APTOS is
        # the half whose source format differs (.png, double-nested, different camera).
        rng = np.random.default_rng(seed)
        groups: dict[str, list[Path]] = {}
        for p in paths:
            groups.setdefault(p.parent.name, []).append(p)
        per = max(1, sample // len(groups))
        chosen = []
        for g in sorted(groups):
            gp = groups[g]
            k = min(per, len(gp))
            chosen += [gp[i] for i in rng.choice(len(gp), k, replace=False)]
    else:
        chosen = paths

    bad = []
    for p in chosen:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            bad.append(f"{p.name}: does not decode")
        elif img.shape != (size, size, 3):
            bad.append(f"{p.name}: shape {img.shape}, expected ({size}, {size}, 3)")
        elif int(img.max()) == 0:
            bad.append(f"{p.name}: decodes to an entirely black image")
    return bad


def build_parser() -> argparse.ArgumentParser:
    """The parser, separate from main(), so an invocation can be checked without
    running it (DECISION-022)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    ap.add_argument("--stats", type=Path, action="append", default=[],
                    help="a stats CSV from preprocess.py; repeatable")
    ap.add_argument("--splits", action="append", default=[],
                    help="split names to reconcile; defaults to all six")
    ap.add_argument("--decode-sample", type=int, default=400,
                    help="how many files to fully decode (0 = all)")
    ap.add_argument("--decode-all", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    cfg = load_preprocess_config(args.config)
    splits = args.splits or (EYEPACS_SPLITS + APTOS_SPLITS)
    root = args.cache_root

    print(f"cache root : {root}")
    print(f"splits     : {', '.join(splits)}")
    print(f"image size : {cfg.image_size}\n")

    exp = expected_rows(splits)
    print(f"split rows : {len(exp)}")

    problems: list[str] = []

    # (1) R1, FIRST, on patient identity. Everything below counts files, and no count of
    # files can see a patient straddling the boundary.
    problems += patient_overlaps(exp)
    n_pat = exp["patient_id"].nunique() if "patient_id" in exp.columns else -1
    print(f"patients   : {n_pat}")

    # (5) One cache file claimed by rows in two different splits - a stem collision, not
    # a patient overlap. Both are fatal; they are different failures.
    owners: dict[str, set[str]] = defaultdict(set)
    for cp, sp in zip(exp["cache_path"], exp["split_file"]):
        owners[cp].add(sp)
    crossing = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
    if crossing:
        problems.append(
            f"R1 VIOLATION: {len(crossing)} cache file(s) are claimed by more than one "
            f"split, e.g. {list(crossing.items())[:5]}"
        )
    dupes = exp["cache_path"].duplicated().sum()
    if dupes:
        problems.append(f"{dupes} duplicate cache_path value(s) within the splits")

    # (1) missing
    on_disk = {
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*.jpg")
    }
    wanted = set(exp["cache_path"])
    missing = sorted(wanted - on_disk)
    if missing:
        problems.append(f"{len(missing)} split row(s) have no cache file, "
                        f"e.g. {missing[:5]}")

    # (2) orphans
    orphan = sorted(on_disk - wanted)
    if orphan:
        problems.append(f"{len(orphan)} cache file(s) belong to no split row, "
                        f"e.g. {orphan[:5]}")

    print(f"on disk    : {len(on_disk)}")
    print(f"matched    : {len(wanted & on_disk)}")
    print(f"missing    : {len(missing)}")
    print(f"orphaned   : {len(orphan)}\n")

    # (3) decode + measured size
    present = [root / c for c in sorted(wanted & on_disk)]
    sizes = np.array([p.stat().st_size for p in present], dtype=np.int64)
    if len(sizes):
        empty = int((sizes == 0).sum())
        if empty:
            problems.append(f"{empty} cache file(s) are zero bytes")
        print("MEASURED cache size")
        print(f"  files    : {len(sizes)}")
        print(f"  total    : {sizes.sum() / 1024**3:.3f} GB "
              f"({sizes.sum() / 1024**2:.1f} MB)")
        print(f"  mean     : {sizes.mean() / 1024:.1f} KB/image")
        print(f"  min/max  : {sizes.min() / 1024:.1f} / {sizes.max() / 1024:.1f} KB")
        for ds in sorted({c.split("/")[0] for c in wanted & on_disk}):
            sub = np.array([p.stat().st_size for p in present
                            if p.parent.name == ds], dtype=np.int64)
            print(f"  {ds:<8} : {len(sub)} files, {sub.sum() / 1024**3:.3f} GB")
        print()

    bad = decode_check(present, cfg.image_size,
                       0 if args.decode_all else args.decode_sample)
    n_dec = len(present) if args.decode_all or args.decode_sample <= 0 else min(
        args.decode_sample, len(present))
    print(f"decoded    : {n_dec} file(s) checked, {len(bad)} bad")
    if bad:
        problems.append(f"{len(bad)} cache file(s) failed the decode check: {bad[:5]}")

    # Non-'ok' statuses. These are NOT dropped — CLAUDE.md §4 and DECISION-013. They need
    # a docs/DECISIONS.md entry and they stay in their split, because removing a val or
    # test row silently rebalances that split, an R2 violation by omission.
    frames: list[pd.DataFrame] = []
    if args.stats:
        frames = [pd.read_csv(s, comment="#") for s in args.stats if s.exists()]
        if frames:
            st = pd.concat(frames, ignore_index=True)
            notok = st[st["status"].astype(str) != "ok"]
            print(f"\nstatus != 'ok': {len(notok)} of {len(st)}")
            for status, grp in notok.groupby("status"):
                print(f"  {status:<24} {len(grp)}")
                for r in grp.head(10).itertuples(index=False):
                    print(f"      {r.image_path}")
                if len(grp) > 10:
                    print(f"      ... and {len(grp) - 10} more")
            if len(notok):
                print("\n  ^ every one of these needs a docs/DECISIONS.md entry and "
                      "STAYS IN ITS SPLIT.")
        missing_stats = [str(s) for s in args.stats if not s.exists()]
        if missing_stats:
            problems.append(f"stats CSV not found: {missing_stats}")

    # Provenance: was this cache built by the pipeline this config describes?
    problems += check_provenance(root, cfg)

    # The stats CSVs must describe exactly the split rows, no more and no less. Free, and
    # it catches a stats file from a different run being read for the status column.
    if args.stats and frames:
        described = set()
        for s_df, s_path in zip(frames, [s for s in args.stats if s.exists()]):
            if "image_path" not in s_df.columns:
                problems.append(f"{s_path} has no image_path column")
                continue
            described |= set(s_df["image_path"])
        wanted_src = set(exp["image_path"])
        if described and described != wanted_src:
            problems.append(
                f"the stats CSVs describe {len(described)} images but the splits hold "
                f"{len(wanted_src)}: {len(wanted_src - described)} split row(s) have no "
                f"stats row, {len(described - wanted_src)} stats row(s) are in no split"
            )

    # Split provenance (DECISION-008): every CSV must come from the same generator run.
    try:
        seeds = {n: [l for l in read_header(n) if "seed" in l.lower()] for n in splits}
        distinct = {tuple(v) for v in seeds.values()}
        if len(distinct) > 1:
            problems.append(
                f"the split CSVs do not share a seed header: {seeds}. They were not "
                "generated together, so reconciling against them proves nothing."
            )
    except FileNotFoundError as exc:
        problems.append(f"could not read a split header: {exc}")

    print()
    if problems:
        print(f"RECONCILIATION FAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("RECONCILIATION PASSED - every split row has exactly one cache file, every "
          "cache file belongs to exactly one split row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

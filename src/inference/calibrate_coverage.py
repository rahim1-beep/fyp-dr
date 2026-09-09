"""Measure the training split's retina-coverage distribution. Produces the ONLY numbers
`coverage_guard` is allowed to judge against (R4).

    python -m src.inference.calibrate_coverage --cache-root <cache> --split train

Runs where the preprocessed cache lives — i.e. Kaggle. About 30 s for 24,586 images at
the measured 1.19 ms/image.

WHY p1/p99 AND NOT min/max. The guard's question is "did the training set contain images
framed like this one", and a single freak image would stretch a min/max range far enough
to wave everything through. The 1st and 99th percentiles answer the same question while
tolerating a handful of outliers, and both bounds are recorded in the artefact so a reader
can see which rule produced them.

NO-RETINA IMAGES ARE COUNTED, NOT DROPPED (DECISION-052). They have no coverage, so they
cannot enter a percentile; the count goes in the artefact so the denominator is honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset import DRDataset
from src.data.manifest import load_split
from src.xai.border_check import NoRetinaError
from src.inference.coverage_guard import DEFAULT_CALIBRATION, retina_coverage

REPO = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--split", default="train",
                    help="the split the model LEARNED from; 'train' unless you have a "
                         "specific reason, and record the reason if not")
    ap.add_argument("--low-pct", type=float, default=1.0)
    ap.add_argument("--high-pct", type=float, default=99.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--limit", type=int,
                    help="first N images only — a SMOKE run, not a calibration")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    df = load_split(args.split)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"!! --limit {args.limit}: this is a SMOKE run, not a calibration")

    ds = DRDataset(df, args.cache_root, train=False, image_size=224)
    missing = ds.check_cache(limit=3)
    if missing:
        raise SystemExit(f"cache is incomplete, e.g. {missing}")

    print(f"measuring retina coverage over {len(ds)} image(s) of split {args.split!r} ...")
    covs, n_no_retina = [], 0
    for i, path in enumerate(ds.cache_paths):
        bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"{path} did not decode")
        try:
            covs.append(retina_coverage(bgr))
        except NoRetinaError:
            n_no_retina += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    if len(covs) < 100:
        raise SystemExit(
            f"only {len(covs)} image(s) yielded a coverage value; refusing to write a "
            "calibration a guard would then trust"
        )

    covs = np.asarray(covs, float)
    low = float(np.percentile(covs, args.low_pct))
    high = float(np.percentile(covs, args.high_pct))

    # Fingerprint the inputs so a calibration can be traced to the cache and split that
    # produced it, rather than to a filename nobody can verify later.
    h = hashlib.sha256()
    h.update(str(args.split).encode())
    h.update(str(len(ds)).encode())
    h.update(f"{covs.sum():.6f}".encode())
    payload = {
        "split": args.split,
        "n_images": int(len(ds)),
        "n_no_retina": int(n_no_retina),
        "low": low,
        "high": high,
        "low_pct": float(args.low_pct),
        "high_pct": float(args.high_pct),
        "median": float(np.median(covs)),
        "cache_fingerprint": h.hexdigest()[:16],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print()
    print(f"  n images       {payload['n_images']}")
    print(f"  no retina      {payload['n_no_retina']}  (counted, excluded from percentiles)")
    print(f"  median         {payload['median']:.4f}")
    print(f"  p{args.low_pct:g} / p{args.high_pct:g}    {low:.4f} / {high:.4f}")
    print(f"  min / max      {covs.min():.4f} / {covs.max():.4f}")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

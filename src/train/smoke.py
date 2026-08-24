"""Run the REAL `src.train.train.main()` end to end, locally, on a synthetic dataset.

WHY THIS EXISTS (DECISION-025). Three consecutive Kaggle sessions died inside
`train.py` on code that only executes in a real run:

    1. describe_balance   'RandomSampler' object has no attribute 'weights'
    2. yaml.safe_dump     cannot represent an object '2.10.0+cu128'
    3. (this file exists so there is no third entry from the same cause)

Every one was in `main()`, past the point unit tests reach, and every one would have been
caught in seconds by executing `main()` once. The unit tests cover `fit`, `evaluate`,
`build_loaders`, `build_loss`, the metrics — all of it green while `main()` itself had
never run start to finish.

WHAT IT DOES. Builds a throwaway cache of a few dozen 224x224 images and a
patient-disjoint split trio with real provenance headers, then calls the real `main()`
through its real argv, and checks that `config.yaml` and `metrics.json` come out
well-formed. About twenty seconds on CPU, no network, no GPU, no real data.

WHAT IT DELIBERATELY DOES NOT DO. It does not assert anything about the SCORE. Twelve
images per class cannot produce a meaningful QWK and pretending otherwise would make this
a flaky test instead of a wiring check. It asserts the pipeline runs and its artefacts are
readable — nothing else.

    python -m src.train.smoke                 # arm A, the default
    python -m src.train.smoke --arm E         # the ordinal head
    python -m src.train.smoke --all-arms      # every arm, ~2 minutes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Small enough to be fast, big enough that every class exists in every split and the
# sampler has something to rebalance.
PATIENTS_PER_CLASS = {"train": 6, "val": 3, "test": 3}


def build_fixture(root: Path, image_size: int = 224) -> tuple[Path, Path]:
    """Write a synthetic cache and a patient-disjoint split trio. Returns (cache, splits).

    The splits are REAL split CSVs: the same six columns, the same `#` provenance header,
    patient-disjoint by construction. `load_split` validates the column set, so a fixture
    that drifts from the real schema fails here rather than passing vacuously.
    """
    import cv2

    cache = root / "cache"
    splits = root / "splits"
    (cache / "eyepacs").mkdir(parents=True)
    (cache / "aptos").mkdir(parents=True)
    splits.mkdir(parents=True)

    rng = np.random.default_rng(0)

    def write(name: str, rows: list[str]) -> None:
        header = (
            f"# fyp-dr {name} split - SYNTHETIC FIXTURE, not real data\n"
            f"# generated: src/train/smoke.py\n"
            f"# RNG seed: 0\n"
            f"# rows: {len(rows)}\n"
        )
        (splits / f"{name}.csv").write_text(
            header + "image_path,patient_id,eye,label,dataset,split\n"
            + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    def image(path: Path) -> None:
        cv2.imwrite(str(path),
                    rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8))

    pid = 0
    for split, per_class in PATIENTS_PER_CLASS.items():
        rows = []
        for label in range(5):
            for _ in range(per_class):
                pid += 1
                for eye in ("left", "right"):
                    stem = f"{pid}_{eye}"
                    image(cache / "eyepacs" / f"{stem}.jpg")
                    rows.append(
                        f"data/data/{stem}.jpeg,{pid},{eye},{label},eyepacs,{split}"
                    )
        write(split, rows)

    # APTOS too, or arm F cannot run - and arm F has the least coverage and the most
    # machinery: pooled splits, a per-origin metrics block, and an origin/severity
    # correlation DECISION-007 requires to stay visible. One image per patient,
    # hash-like ids, no eye linkage (DECISION-004).
    for split, per_class in PATIENTS_PER_CLASS.items():
        rows = []
        for label in range(5):
            for _ in range(max(2, per_class - 1)):
                pid += 1
                ident = f"{pid:012x}"
                image(cache / "aptos" / f"aptos_{ident}.jpg")
                rows.append(
                    f"train_images/train_images/{ident}.png,aptos_{ident},"
                    f"unknown,{label},aptos,aptos_{split}"
                )
        write(f"aptos_{split}", rows)

    return cache, splits


def run_arm(arm: str, workdir: Path, *, epochs: int = 1, keep: bool = False,
            arch: str | None = None) -> dict:
    """One arm, through the real main(). Returns the parsed metrics.json.

    `arch` exists so a backbone can be exercised end to end BEFORE an hour of GPU is
    spent on it. Every architecture switch so far has been made by passing `--arch` to
    a Kaggle run whose first execution of that code path was the real one; a head that
    does not attach, or a classifier attribute timm names differently for that family,
    fails at minute zero of an hour-long cell instead of here in twelve seconds.
    """
    from src.train.train import main as train_main

    cache, splits = build_fixture(workdir / arm)
    runs = workdir / arm / "runs"
    run_id = f"smoke_{arm.lower()}"

    argv = [
        "--arm", arm,
        "--cache-root", str(cache),
        "--splits-root", str(splits),
        "--run-id", run_id,
        "--runs-root", str(runs),
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--num-workers", "0",
        "--device", "cpu",
        "--no-pretrained",          # no network, and the weights are irrelevant here
        "--skip-gate",              # the gate is over the COMMITTED splits, not these
    ]
    if arch:
        argv += ["--arch", arch]

    old_argv = sys.argv
    sys.argv = ["src.train.train"] + argv
    try:
        rc = train_main()
    finally:
        sys.argv = old_argv

    if rc != 0:
        raise RuntimeError(f"arm {arm}: train.main() returned {rc}")

    run_dir = runs / run_id
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # The artefacts have to be readable and honest about what they are (R4/R6).
    assert cfg["arm"] == arm.upper(), cfg["arm"]
    assert cfg["is_smoke_run"] is True, "a synthetic run must say so in its own config"
    assert cfg["versions"]["torch"], "versions did not survive the yaml dump"
    assert metrics["is_smoke_test"] is True
    assert len(metrics["confusion_matrix"]) == 5
    assert (run_dir / "train_log.csv").exists()

    if not keep:
        shutil.rmtree(workdir / arm, ignore_errors=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="A")
    ap.add_argument("--arch", help="backbone to exercise, e.g. efficientnet_b2; default is whatever the config says")
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="leave the fixture on disk")
    ap.add_argument("--workdir", type=Path,
                    help="where to build the fixture (default: a temp dir)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    arms = list("ABCDEF") if args.all_arms else [args.arm.upper()]

    tmp = None
    if args.workdir:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        workdir = Path(tmp.name)

    failures = []
    try:
        for arm in arms:
            print("\n" + "#" * 78)
            print(f"# SMOKE: arm {arm}"
                  + (f" on {args.arch}" if args.arch else ""))
            print("#" * 78, flush=True)
            t0 = time.time()
            try:
                m = run_arm(arm, workdir, epochs=args.epochs, keep=args.keep,
                            arch=args.arch)
                print(f"\n[arm {arm} OK in {time.time() - t0:.0f}s  "
                      f"qwk={m['qwk']:.3f} acc={m['accuracy']:.3f} "
                      f"(scores are meaningless at this size)]")
            except Exception as exc:               # noqa: BLE001 - reported, not hidden
                import traceback
                traceback.print_exc()
                failures.append((arm, f"{type(exc).__name__}: {exc}"))
    finally:
        if tmp is not None:
            tmp.cleanup()

    print("\n" + "=" * 78)
    if failures:
        print(f"SMOKE FAILED - {len(failures)} arm(s):")
        for arm, why in failures:
            print(f"  arm {arm}: {why}")
        print("=" * 78)
        return 1
    print(f"SMOKE PASSED - {len(arms)} arm(s) ran end to end through the real main()"
          + (f" on {args.arch}." if args.arch else "."))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

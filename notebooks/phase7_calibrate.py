"""Phase 7 — measure the coverage guard's bounds. ONE cell, ~2 minutes. No GPU needed.

    python -m notebooks.phase7_calibrate 1

This is the only thing standing between the coverage guard and being live. The guard
(DECISION-068) refuses to run without a measured calibration, because a guard with
invented bounds presents as evidence. This produces that measurement over the TRAINING
split — what the model was actually fitted on.

WHY IT HAS TO RUN ON KAGGLE. It reads the preprocessed cache, which is 38,788 images and
does not exist locally (CLAUDE.md §3).

WHY THE OUTPUT PATH IS /kaggle/working/coverage_guard AND NOT INSIDE THE REPO COPY.
`fetch_run --artefacts coverage_guard` looks for a directory of that name in the kernel
output. Writing it under `/kaggle/working/fyp-dr/analysis/...` would bury it inside the
copied repo where the fetcher does not look.

SESSION SETTINGS
    Accelerator : None (CPU is fine — no model is loaded)
    Internet    : ON
    Inputs      : fyp-dr-eyepacs-224, fyp-dr-code

HOW TO RUN
    Cell 1, by COMMIT (Save & Run All). It is short enough to run interactively too, but
    only a committed run saves an output you can fetch.

AFTERWARDS, LOCALLY
    python -m src.data.fetch_run --kernel rah098/<this-notebook-slug> \\
        --artefacts coverage_guard --force
    git add analysis/coverage_guard/calibration.json && git commit
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- measure the training split's retina-coverage distribution ------
# COMMIT THIS. ~2 min, CPU only.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
OUT = WORK / "coverage_guard"

# Same resolution as phase6: resolve_input, never a first-hit rglob over all of
# /kaggle/input, because attached notebook outputs carry their own copy of the repo.
_boot = [h for h in Path("/kaggle/input").rglob("src/data/kaggle_paths.py")
         if "notebooks" not in h.parts]
if not _boot:
    raise SystemExit("fyp-dr-code not found. + Add Input -> Your Datasets.")
_boot_root = str(_boot[0].parents[2])
sys.path.insert(0, _boot_root)
from src.data.kaggle_paths import resolve_input

CODE = resolve_input("fyp-dr-code", owner="rah098", must_contain="src/data/preprocess.py")
print("code:", CODE)

sys.path.remove(_boot_root)
for _m in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
    del sys.modules[_m]

if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
OUT.mkdir(parents=True, exist_ok=True)

# The splits vanish mid-run on this platform and the cause is still unknown
# (DECISION-064). This job reads train.csv, so check before spending the two minutes.
FYP_SPLITS = ["train", "val", "test", "aptos_train", "aptos_val", "aptos_test"]
_absent = [n for n in FYP_SPLITS if not (REPO / "data" / "splits" / f"{n}.csv").exists()]
if _absent:
    raise SystemExit(f"the copied repo is missing split CSV(s): {_absent}. Re-upload "
                     "fyp-dr-code as a New Version and confirm the notebook is pinned "
                     "to it.")
print(f"splits: all {len(FYP_SPLITS)} present")
os.environ["FYP_REQUIRE_SPLITS"] = "1"

from src.data.archive_cache import resolve_cache

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    return rc


assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_coverage_guard.py", "-q"]) == 0

# THE MEASUREMENT. Training split, because the guard's question is what the model was
# FITTED on; using validation would pull an evaluation split into a deployment decision.
assert run([sys.executable, "-m", "src.inference.calibrate_coverage",
            "--cache-root", CACHE, "--split", "train",
            "--out", OUT / "calibration.json"]) == 0

cal = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
print("\nMEASURED coverage calibration:")
for k, v in cal.items():
    print(f"  {k:<20} {v}")

# The guard must accept what was just written — check here, not after the fetch.
from src.inference.coverage_guard import Calibration, assess

c = Calibration.load(OUT / "calibration.json")
print(f"\nthe guard loads it: bounds {c.low:.4f}-{c.high:.4f} from {c.n_images} images")
if not (0.0 < c.low < c.median < c.high < 1.0):
    raise SystemExit(f"implausible calibration: {c}")

# Sanity: a typical cached training image must PASS its own calibration. If the median
# image is flagged, the guard would flag almost everything and is useless.
import cv2
import numpy as np

from src.data.dataset import DRDataset
from src.data.manifest import load_split

_ds = DRDataset(load_split("train").head(50), CACHE, train=False, image_size=224)
_status = []
for p in _ds.cache_paths:
    bgr = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    _status.append(assess(bgr, c)["status"])
_ok = _status.count("ok")
print(f"self-check: {_ok}/50 training images pass their own calibration")
if _ok < 45:
    raise SystemExit(
        f"only {_ok}/50 training images pass the calibration measured FROM the training "
        "split. The guard would flag most real images; do not commit this."
    )

from src.data.fetch_run import write_manifest

write_manifest(OUT)
print(f"\nartefacts + MANIFEST.json in {OUT}")
for q in sorted(OUT.iterdir()):
    print(f"  {q.name}  {q.stat().st_size / 1024:.1f} KB")
print("""
Commit this notebook, then locally:

  python -m src.data.fetch_run --kernel rah098/<slug> --artefacts coverage_guard --force
  git add analysis/coverage_guard/calibration.json && git commit
""")
'''

CELLS = [CELL_1]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1"]):
        print(CELLS[int(w) - 1])

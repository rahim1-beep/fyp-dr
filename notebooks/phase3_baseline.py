"""Phase 3 baseline - ResNet18, arm A, short schedule. Five Kaggle cells.

The point of this run is a CORRECT pipeline, not a good score (PROGRESS.md, Phase 3
acceptance): a validation QWK whose confidence interval excludes zero, and a confusion
matrix that is not collapsed into one column.

    python -m notebooks.phase3_baseline 1     # or 2, 3, 4, 5

SESSION SETTINGS - different from the Phase 2 build in every respect:

    Accelerator : GPU T4 x2      (P100 also fine; NOT "None" - this is the training run)
    Internet    : ON             REQUIRED. timm downloads pretrained weights from
                                 HuggingFace. Needs a phone-verified Kaggle account.
                                 Cell 1 fails fast and says so if it is off.
    Inputs      : fyp-dr-eyepacs-224   (the Phase 2 cache)
                  fyp-dr-code          (the code bundle)
                  The raw 35.3 GB EyePACS and APTOS datasets are NOT attached. Training
                  reads the preprocessed cache and never opens a source image.

HOW TO RUN IT (this is the part worth getting right):

    Cells 1-2  INTERACTIVELY, by hand. Cell 2 is a five-minute smoke run whose only job
               is to fail fast on a plumbing mistake before the real run spends GPU quota.
    Cells 3-5  by COMMIT: Save Version -> Save & Run All. A commit re-executes the whole
               notebook from a clean session, so set RUN_SMOKE = False in cell 2 first.

DECISION-020: Kaggle mounts inputs at /kaggle/input/datasets/{owner}/{slug}/ now, and
/kaggle/input is read-only. Both cell 1 lookups go through src.data.kaggle_paths, which
tries the known layouts and raises with the real directory listing if none matches.

DECISION-021 / DECISION-023: the cache is published as ONE verified zip, because Kaggle
caps notebook output at 500 files and a loose tree was silently truncated to 499 - but
Kaggle then AUTO-EXTRACTS that zip when the dataset is created, so what mounts is a folder
tree, not an archive. Cell 1 calls resolve_cache, which handles both and verifies the
image count whichever it finds.
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, mount resolution, versions, leakage gate -----------------
# RUN INTERACTIVELY. About one minute.
import os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

print("mounted under /kaggle/input:")
for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
    if root.is_dir():
        print(f"  {root}: {sorted(p.name for p in root.iterdir())[:20]}")

# The code bundle has to be found before its own resolver can be imported, so this one
# lookup is done by hand - with the same candidate list kaggle_paths uses.
CODE = None
for cand in (Path("/kaggle/input/datasets/rah098/fyp-dr-code"),
             Path("/kaggle/input/fyp-dr-code"),
             Path("/kaggle/working/inputs/fyp-dr-code")):
    if (cand / "src").is_dir():
        CODE = cand
        break
if CODE is None:
    hits = list(Path("/kaggle/input").rglob("src/data/preprocess.py"))
    if not hits:
        raise SystemExit(
            "fyp-dr-code not found. Attach it: + Add Input -> Your Datasets -> "
            "fyp-dr-code. What is mounted is listed above."
        )
    CODE = hits[0].parents[2]
print("\ncode:", CODE)

# /kaggle/input is read-only; pytest and __pycache__ need to write. Copy it out.
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --- the preprocessed cache: FOUND BY SHAPE, never by path (DECISION-023) -----
# Kaggle AUTO-EXTRACTS an uploaded archive when it publishes a dataset, so the single
# verified .zip from DECISION-021 is not a .zip by the time it is mounted. The published
# dataset came back as a `fyp-dr-eyepacs-224/` folder holding 38.8k files with the stats
# CSVs beside it - neither MOUNT/*.zip nor MOUNT/processed, which is what this cell used
# to look for.
#
# Three layouts for one artefact across two platform behaviours. resolve_cache stops
# predicting: it takes the zip path if a zip is there, otherwise it searches for the
# SHAPE - a directory with eyepacs/ and aptos/ holding images - and verifies the count
# either way.
from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
print("dataset:", MOUNT)
print("        ", sorted(q.name for q in MOUNT.iterdir())[:10])

CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("\ncache:", CACHE)
print("      ", sorted(q.name for q in CACHE.iterdir())[:10])
print(f"       {sum(1 for _ in CACHE.rglob('*.jpg'))} images, verified")

# --- versions: the GPU image may differ from the CPU image (DECISION-019) -----
import cv2, numpy, pandas, timm, torch, torchvision
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  torchvision {torchvision.__version__}  |  timm {timm.__version__}")
print(f"numpy {numpy.__version__}  |  pandas {pandas.__version__}  |  cv2 {cv2.__version__}")
print("^ RECORD these. If they differ from requirements.txt, add them there as the")
print("  GPU-image block (DECISION-019) - never edit a pin from memory.")

if not torch.cuda.is_available():
    raise SystemExit(
        "no GPU. Session sidebar -> Accelerator -> GPU T4 x2, then re-run this cell."
    )
print(f"\ncuda: {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")

# --- pretrained weights need Internet ON --------------------------------------
# Checked here with the smallest possible download. Discovering this at epoch 0 of the
# committed run wastes the queue wait and the session.
try:
    _ = timm.create_model("resnet18", pretrained=True, num_classes=5)
    print("pretrained weights: OK")
except Exception as e:
    raise SystemExit(
        f"could not fetch pretrained weights: {e}\n"
        "Session sidebar -> Internet -> On (needs a phone-verified account). Training "
        "from scratch is NOT a substitute: it changes what this baseline measures."
    )


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    return rc


# DECISION-022: every command line this notebook will run, checked against the real
# parsers, plus a real pack/verify/unpack on a toy directory. Seconds, and it is the
# check that three lost sessions paid for.
assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"]) == 0

# DECISION-025: run the REAL train.main() end to end on a synthetic 90-image fixture,
# on CPU, before spending anything. Two sessions died inside main() on code no unit test
# reaches -- a sampler attribute and a yaml dump -- and both would have failed here in
# under a minute. Arm A and arm F between them touch every branch in main().
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "A"]) == 0
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "F"]) == 0

# CLAUDE.md S7. src/train/train.py runs it again itself before it touches a weight.
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0
'''

CELL_2 = r'''
# -- Cell 2 -- SMOKE TEST: 2,000 train rows, 2 epochs, about 5 minutes ---------
# RUN INTERACTIVELY, then set RUN_SMOKE = False before you commit. A commit re-runs the
# whole notebook from a clean session, and this has already done its job by then.
#
# A smoke run is NOT a result. --limit-train goes into the run config and metrics.json
# carries is_smoke_test: true, so it cannot be quoted as one by accident.
RUN_SMOKE = True

if RUN_SMOKE:
    rc = run([
        sys.executable, "-m", "src.train.train",
        "--arm", "A",
        "--config", "configs/kaggle.yaml",
        "--cache-root", CACHE,
        "--run-id", "phase3_smoke",
        "--runs-root", RUNS,
        "--epochs", "2",
        "--limit-train", "2000",
        "--num-workers", "2",
        "--log-every", "20",
    ])
    print("smoke exit:", rc)
    print("\nWhat to look for: the loss moves, val QWK is a real number, and the run "
          "reaches the confusion matrix. The SCORE does not matter here.")
else:
    print("smoke skipped (RUN_SMOKE = False)")
'''

CELL_3 = r'''
# -- Cell 3 -- the baseline: arm A, ResNet18, 8 epochs, roughly 60-90 min ------
# COMMIT THIS ONE. Save Version -> Save & Run All.
rc = run([
    sys.executable, "-m", "src.train.train",
    "--arm", "A",
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", "phase3_baseline_resnet18",
    "--runs-root", RUNS,
    "--epochs", "8",
    "--num-workers", "2",
    "--log-every", "100",
])
print("baseline exit:", rc)
'''

CELL_4 = r'''
# -- Cell 4 -- read the result and state plainly whether it passed ------------
import json

run_dir = RUNS / "phase3_baseline_resnet18"
m = json.loads((run_dir / "metrics.json").read_text())

print(f"run          : {m['run_id']}  (arm {m['arm']}, {m['epochs_run']} epochs, "
      f"best at epoch {m['best_epoch']})")
print(f"val QWK      : {m['qwk']:.4f}   95% CI [{m['qwk_ci']['lo']:.4f}, "
      f"{m['qwk_ci']['hi']:.4f}]")
print(f"accuracy     : {m['accuracy']:.4f}")
print(f"balanced acc : {m['balanced_accuracy']:.4f}")
print(f"referable    : sens {m['referable']['sensitivity']:.4f}  "
      f"spec {m['referable']['specificity']:.4f}")
print(f"\nper-class recall: "
      f"{[None if v is None else round(v, 3) for v in m['per_class_recall']]}")
print(f"support         : {m['support']}")
print(f"predicted counts: {m['predicted_counts']}")
print("\nconfusion matrix (rows = truth, cols = predicted):")
for i, row in enumerate(m["confusion_matrix"]):
    print(f"  {i}  " + " ".join(f"{v:>6}" for v in row))

print("\nrare-class recall with intervals (DECISION-006 - never a bare point estimate):")
for g, ci in m["rare_class_ci"].items():
    print(f"  grade {g}: {ci['point']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]")

# Phase 3 acceptance, as a check rather than a vibe.
qwk_ok = m["qwk_ci"]["lo"] > 0.0
not_collapsed = not m["collapse"]["collapsed"]
print("\n" + "=" * 72)
print(f"val QWK CI excludes zero       : {qwk_ok}")
print(f"confusion matrix not collapsed : {not_collapsed}")
for r in m["collapse"]["reasons"]:
    print(f"    {r}")
print("PHASE 3 ACCEPTANCE MET" if (qwk_ok and not_collapsed)
      else "NOT MET - stop and diagnose before Phase 4 (CLAUDE.md S2)")
print("=" * 72)
'''

CELL_5 = r'''
# -- Cell 5 -- keep the evidence ----------------------------------------------
# /kaggle/working is wiped between sessions. runs/ is small - config, metrics, the epoch
# log, one checkpoint - and it IS the record of this run (R4/R6).
import shutil

for p in sorted(RUNS.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUNS)}  {p.stat().st_size / 1024:.0f} KB")

# Drop the smoke run's checkpoint: it is a 2,000-row 2-epoch model and nothing should
# ever load it. Its config and metrics stay, because the run happened.
smoke_ckpt = RUNS / "phase3_smoke" / "best.pth"
if smoke_ckpt.exists():
    smoke_ckpt.unlink()
    print("\nremoved the smoke checkpoint (its config and metrics are kept)")

shutil.rmtree(REPO, ignore_errors=True)   # keep the output to runs/ alone
print("\nCommit this notebook, then download runs/ from the Output tab and commit it to "
      "the repo: runs/<run_id>/{config.yaml,metrics.json,train_log.csv} are tracked, "
      "*.pth is not.")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(CELLS[int(w) - 1])

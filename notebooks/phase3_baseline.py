"""Phase 3 baseline - ResNet18, arm A, short schedule. Four Kaggle cells.

The point of this run is a CORRECT pipeline, not a good score (PROGRESS.md, Phase 3
acceptance): validation QWK clearly above 0, and a confusion matrix that is not collapsed
into one column.

    python -m notebooks.phase3_baseline 1     # or 2, 3, 4

NOTEBOOK SETTINGS - different from the Phase 2 build:

    Accelerator : GPU T4 x2  (or P100)
    Internet    : ON         <- REQUIRED. timm downloads pretrained weights from
                                HuggingFace. Needs a phone-verified Kaggle account.
                                Without it, cell 1 fails fast and says so.
    Inputs      : fyp-dr-code  and  fyp-dr-eyepacs-224   (the Phase 2 cache)
                  the raw EyePACS/APTOS datasets are NOT needed - training reads the cache

DECISION-020: Kaggle mounts inputs at /kaggle/input/datasets/{owner}/{slug}/ now, and
/kaggle/input is read-only. Cell 1 resolves the location instead of assuming it.
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, mount resolution, versions, leakage gate -----------------
import os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

print("mounted under /kaggle/input:")
for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
    if root.is_dir():
        print(f"  {root}: {sorted(p.name for p in root.iterdir())[:20]}")

# --- code bundle --------------------------------------------------------------
CODE = None
for cand in (Path("/kaggle/input/datasets/rah098/fyp-dr-code"),
             Path("/kaggle/input/fyp-dr-code")):
    if (cand / "src").is_dir():
        CODE = cand
        break
if CODE is None:
    hits = list(Path("/kaggle/input").rglob("src/data/preprocess.py"))
    if not hits:
        raise SystemExit("fyp-dr-code not found - attach it as an input")
    CODE = hits[0].parents[2]
print("\ncode:", CODE)

if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --- the preprocessed cache (DECISION-020: resolve, never assume) -------------
from src.data.kaggle_paths import resolve_input

CACHE = resolve_input("fyp-dr-eyepacs-224", owner="rah098", must_contain="processed")
CACHE = CACHE / "processed"
print("cache:", CACHE)
print("       ", sorted(p.name for p in CACHE.iterdir())[:10])

# --- versions: the GPU image may differ from the CPU image (DECISION-019) -----
import cv2, numpy, pandas, timm, torch, torchvision
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  torchvision {torchvision.__version__}  |  timm {timm.__version__}")
print(f"numpy {numpy.__version__}  |  pandas {pandas.__version__}  |  cv2 {cv2.__version__}")
print(f"cuda available: {torch.cuda.is_available()}  "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'})")
print("^ record these in requirements.txt as the GPU-image block if they differ")

if not torch.cuda.is_available():
    raise SystemExit("no GPU - set Accelerator to GPU T4 x2 in the session sidebar")

# --- pretrained weights need internet -----------------------------------------
# timm downloads from HuggingFace. Finding this out at epoch 0 of a real run wastes the
# queue wait, so it is checked here with the smallest possible download.
try:
    _ = timm.create_model("resnet18", pretrained=True, num_classes=5)
    print("\npretrained weights: OK")
except Exception as e:
    raise SystemExit(
        f"could not fetch pretrained weights: {e}\n"
        "Turn Internet ON in the session sidebar (needs a phone-verified account). "
        "Training from scratch is NOT an acceptable substitute - it changes what the "
        "baseline measures."
    )


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    return rc


# CLAUDE.md S7 - and src/train/train.py runs it again itself.
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0
'''

CELL_2 = r'''
# -- Cell 2 -- SMOKE TEST first: 2000 train rows, 2 epochs, ~5 minutes ---------
# The point is to fail fast on a plumbing mistake rather than 90 minutes in. A smoke run
# is NOT a result: --limit-train is recorded in the run config and metrics.json carries
# is_smoke_test: true, so it can never be quoted as one.
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
'''

CELL_3 = r'''
# -- Cell 3 -- the baseline: arm A, ResNet18, 8 epochs, roughly 60-90 min ------
# Acceptance (PROGRESS.md Phase 3): val QWK clearly above 0, confusion matrix not
# collapsed. A good score is Phase 4's job.
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
# -- Cell 4 -- read the result, and say plainly whether it passed --------------
import json

run_dir = RUNS / "phase3_baseline_resnet18"
m = json.loads((run_dir / "metrics.json").read_text())

print(f"run          : {m['run_id']}  (arm {m['arm']}, {m['epochs_run']} epochs, "
      f"best at {m['best_epoch']})")
print(f"val QWK      : {m['qwk']:.4f}   95% CI [{m['qwk_ci']['lo']:.4f}, "
      f"{m['qwk_ci']['hi']:.4f}]")
print(f"accuracy     : {m['accuracy']:.4f}")
print(f"balanced acc : {m['balanced_accuracy']:.4f}")
print(f"referable    : sens {m['referable']['sensitivity']:.4f}  "
      f"spec {m['referable']['specificity']:.4f}")
print(f"\nper-class recall: {[None if v is None else round(v, 3) for v in m['per_class_recall']]}")
print(f"support         : {m['support']}")
print(f"predicted counts: {m['predicted_counts']}")
print("\nconfusion matrix (rows = truth, cols = predicted):")
for i, row in enumerate(m["confusion_matrix"]):
    print(f"  {i}  " + " ".join(f"{v:>6}" for v in row))

# Phase 3 acceptance, stated as a check rather than a vibe.
qwk_ok = m["qwk_ci"]["lo"] > 0.0
not_collapsed = not m["collapse"]["collapsed"]
print("\n" + "=" * 72)
print(f"val QWK CI excludes zero      : {qwk_ok}")
print(f"confusion matrix not collapsed: {not_collapsed}")
if m["collapse"]["reasons"]:
    for r in m["collapse"]["reasons"]:
        print(f"    {r}")
print("PHASE 3 ACCEPTANCE MET" if (qwk_ok and not_collapsed) else
      "NOT MET - stop and diagnose before Phase 4 (CLAUDE.md S2)")
print("=" * 72)

# runs/ is small (config, metrics, log, one checkpoint). Keep it: /kaggle/working is
# wiped between sessions, so commit the notebook and download runs/ or publish it.
for p in sorted(RUNS.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUNS)}  {p.stat().st_size / 1024:.0f} KB")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4"]):
        print(CELLS[int(w) - 1])

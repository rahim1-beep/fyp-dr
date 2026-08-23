"""Phase 4 follow-up: re-run arm C, and backfill validation outputs for the five arms
that already ran. Five Kaggle cells, about 45 minutes.

    python -m notebooks.phase4_armc_backfill 1     # or 2, 3, 4, 5

WHY BOTH IN ONE SESSION. Arm C needs a GPU and 36 minutes. The backfill needs the five
existing CHECKPOINTS, which exist only in the previous notebook's output. Doing them
together costs one queue wait instead of two, and after this session no threshold or
operating-point question needs a GPU ever again — `val_outputs.npz` is a ~150 KB tracked
artefact and everything downstream is local.

THE READ-ONLY TRAP. `/kaggle/input` is read-only, and a mounted notebook output is no
exception. `src.eval.predict` writes `val_outputs.npz` INTO the run directory, so the
mounted `runs/` is copied to `/kaggle/working/runs` first and the backfill runs against
the writable copy. Backfilling in place would fail at the first write, after loading the
checkpoint and running inference.

SESSION SETTINGS
    Accelerator : GPU T4 x2
    Internet    : ON              (arm C's re-run downloads pretrained weights)
    Inputs      : fyp-dr-eyepacs-224, fyp-dr-code,
                  AND the Phase 4 ablation notebook's OUTPUT
                  (+ Add Input -> Notebook Output -> fyp-dr phase4 ablation)

HOW TO RUN
    Cell 1      by hand  (~4 min: checks, smoke, gate, and it locates the checkpoints)
    Cells 2-5   by COMMIT
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, cache, checkpoints, and every pre-flight check ------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

ARCH = "resnet18"
EPOCHS = 30                  # identical to stage 1 - arm C must be comparable
SEED = 42
ARM_C_RUN_ID = f"phase4_arm_c_{ARCH}"
BACKFILL_ARMS = ["A", "B", "D", "E", "F"]        # the five that already ran

print("mounted under /kaggle/input:")
for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
    if root.is_dir():
        print(f"  {root}: {sorted(p.name for p in root.iterdir())[:20]}")

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
        raise SystemExit("fyp-dr-code not found. + Add Input -> Your Datasets.")
    CODE = hits[0].parents[2]
print("\ncode:", CODE)

if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --- the cache, by shape (DECISION-023) ---------------------------------------
from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

# --- the previous notebook's checkpoints, also by shape ------------------------
# Found by looking for best.pth under a phase4 run directory, wherever the mounted
# notebook output happens to sit. Same principle as DECISION-023: the platform's layout
# is not a stable interface, the artefact's shape is.
ckpts = sorted(Path("/kaggle/input").rglob(f"phase4_arm_*_{ARCH}/best.pth"))
if not ckpts:
    raise SystemExit(
        "no phase4 checkpoints found. Add the ablation notebook's output:\n"
        "  + Add Input -> Notebook Output -> the phase 4 ablation notebook\n"
        f"looked for: /kaggle/input/**/phase4_arm_*_{ARCH}/best.pth"
    )
SRC_RUNS = ckpts[0].parent.parent
print(f"\ncheckpoints: {SRC_RUNS}")
for c in ckpts:
    print(f"  {c.parent.name}  {c.stat().st_size / 1024**2:.0f} MB")

# /kaggle/input is READ-ONLY and predict.py writes val_outputs.npz into the run dir, so
# work on a writable copy. Backfilling in place fails after the inference, not before it.
RUNS.mkdir(parents=True, exist_ok=True)
for c in ckpts:
    dst = RUNS / c.parent.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(c.parent, dst)
print(f"\ncopied {len(ckpts)} run(s) to {RUNS} (writable)")

# --- versions, GPU, weights ---------------------------------------------------
import cv2, numpy, pandas, timm, torch, torchvision
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  timm {timm.__version__}  |  numpy {numpy.__version__}  |  cv2 {cv2.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("no GPU. Session sidebar -> Accelerator -> GPU T4 x2.")
print(f"cuda: {torch.cuda.get_device_name(0)}")

try:
    _ = timm.create_model(ARCH, pretrained=True, num_classes=5)
    print("pretrained weights: OK")
except Exception as e:
    raise SystemExit(f"could not fetch pretrained weights: {e}\nInternet -> On.")


def run(cmd, timeout_min=None):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=REPO,
                            timeout=timeout_min * 60 if timeout_min else None).returncode
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT after {timeout_min} min]", flush=True)
        return 124, (time.time() - t0) / 60
    mins = (time.time() - t0) / 60
    print(f"[exit {rc} in {mins:.1f} min]", flush=True)
    return rc, mins


# --- the gates (DECISION-022 / DECISION-025) ----------------------------------
assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"])[0] == 0
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "C"])[0] == 0   # the fixed arm
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "E"])[0] == 0   # ordinal head
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

print(f"\nplan: re-run arm C ({EPOCHS} epochs, ~36 min), then backfill "
      f"{len(BACKFILL_ARMS)} checkpoint(s) (~5 min)")
'''

CELL_2 = r'''
# -- Cell 2 -- arm C, the arm that died before epoch 1 -------------------------
# COMMIT THIS. ~36 min.
#
# It failed because fit() moved the model to the GPU and not the criterion, and arm C is
# the only arm whose loss carries class weights. Same budget, seed and arch as stage 1,
# because a differently-trained arm C would not be comparable with the other five.
rc_c, mins_c = run([
    sys.executable, "-m", "src.train.train",
    "--arm", "C",
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", ARM_C_RUN_ID,
    "--runs-root", RUNS,
    "--epochs", str(EPOCHS),
    "--seed", str(SEED),
    "--num-workers", "2",
    "--log-every", "200",
], timeout_min=90)

if rc_c == 0:
    m = json.loads((RUNS / ARM_C_RUN_ID / "metrics.json").read_text())
    print(f"\narm C: qwk {m['qwk']:.4f} [{m['qwk_ci']['lo']:.4f}, {m['qwk_ci']['hi']:.4f}]  "
          f"bal {m['balanced_accuracy']:.4f}  ref_sens {m['referable']['sensitivity']:.4f}  "
          f"({m['epochs_run']} epochs, best {m['best_epoch']})")
    print(f"val outputs: {(RUNS / ARM_C_RUN_ID / 'val_outputs.npz').exists()}")
else:
    print(f"\n*** ARM C FAILED AGAIN (exit {rc_c}). Do not re-run blind - read the "
          "traceback above. ***")
'''

CELL_3 = r'''
# -- Cell 3 -- backfill val_outputs.npz for the five arms that already ran -----
# COMMIT THIS. ~5 min: inference only, no training.
#
# Each run is reloaded from its OWN config.yaml and best.pth, so the model rebuilt is the
# model that was trained. predict.py also recomputes QWK and the confusion matrix from
# the raw outputs and compares them to the committed metrics.json - a reproducibility
# check on the whole evaluation path, obtained for free. metrics.json is never rewritten.
targets = sorted(d for d in RUNS.iterdir()
                 if d.is_dir() and (d / "best.pth").exists() and d.name != ARM_C_RUN_ID)
print("backfilling:", [d.name for d in targets])

# Built as one list and VALIDATED before it runs (DECISION-022). A splat cannot be
# checked statically, so it is checked here instead - the loop below is exactly the kind
# of thing that silently produces a malformed argv.
from src.data.notebook_check import validate_argv

cmd = [sys.executable, "-m", "src.eval.predict"]
for d in targets:
    cmd += ["--run-dir", str(d)]
cmd += ["--cache-root", str(CACHE), "--batch-size", "64", "--num-workers", "2"]

why = validate_argv(cmd)
if why:
    raise SystemExit(f"backfill command is malformed: {why}")
print("argv validated against src.eval.predict's own parser")

rc_b, mins_b = run(cmd, timeout_min=45)
print("backfill exit:", rc_b)
'''

CELL_4 = r'''
# -- Cell 4 -- the operating-point table, all six arms ------------------------
# COMMIT THIS. Seconds.
#
# Selection is on VALIDATION (R3). The floor is PROVISIONAL - DECISION-032 records that
# the applicable standard is not yet confirmed, and it must not be quoted in the write-up
# until it is.
have = sorted(d for d in RUNS.iterdir()
              if d.is_dir() and (d / "val_outputs.npz").exists())
print(f"{len(have)} run(s) carry val_outputs.npz: {[d.name for d in have]}\n")

if have:
    from src.data.notebook_check import validate_argv

    cmd = [sys.executable, "-m", "src.eval.thresholds"]
    for d in have:
        cmd += ["--run-dir", str(d)]

    why = validate_argv(cmd)
    if why:
        raise SystemExit(f"thresholds command is malformed: {why}")

    run(cmd)
else:
    print("nothing to analyse - cells 2 and 3 did not produce outputs")
'''

CELL_5 = r'''
# -- Cell 5 -- keep the evidence ----------------------------------------------
import shutil

for p in sorted(RUNS.rglob("*")):
    if p.is_file() and p.suffix != ".pth":
        print(f"  {p.relative_to(RUNS)}  {p.stat().st_size / 1024:.0f} KB")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)

print("\nCommit this notebook, then locally:")
print()
ids = f"--run-id {ARM_C_RUN_ID}"
print(f"  python -m src.data.fetch_run --kernel rah098/<this-notebook-slug> \\")
print(f"      {ids}")
print()
print("fetch_run pulls config.yaml, metrics.json, train_log.csv, and val_outputs.npz +")
print("thresholds.json when present. Pass --run-id for all six arms. Then:")
print("  python -m src.eval.thresholds --run-dir runs/phase4_arm_a_resnet18 ... (all six)")
print("  python -m notebooks.gen_experiments")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(CELLS[int(w) - 1])

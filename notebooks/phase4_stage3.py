"""Phase 4 stage 3 — the capacity hypothesis, plus arm C2. Five cells, about 2.6 hours.

    python -m notebooks.phase4_stage3 1     # or 2, 3, 4, 5

THIS IS A FALSIFIABLE TEST, NOT A SCALE-UP. The prediction is written down in
`docs/EXPERIMENTS.md` BEFORE the numbers land, so it is on record either way.

    Stage 1 showed the three sampler arms reaching train QWK 0.87-0.92 and losing ~0.30
    on validation, while arms A and E lost 0.07-0.09. That is memorisation, not
    underfitting, so the binding constraint is GENERALISATION and not capacity.

    PREDICTION: EfficientNet-B0 will not close the 0.14 referable-sensitivity gap. Arm B's
    train-val gap will stay wide or widen (>= 0.25); arms A and E will move by less than
    0.03 QWK.

    If that is wrong - if B0 closes the gap - then capacity WAS the limit and the
    resolution/regularisation route can be deprioritised. Either outcome is a result.

Arms A, E and B are the informative three: the two best (which barely overfit) and the
clearest overfitter, which is where the prediction is sharpest.

ARM C2 rides along on ResNet18. Arm C used inverse-frequency weights at a 36:1 ratio;
effective-number weighting is 17.1:1 on the same split. Without it, "class weighting does
not work" rests on one run of the most aggressive scheme available (DECISION-034).

SESSION SETTINGS
    Accelerator : GPU T4 x2      Internet : ON      Inputs : fyp-dr-eyepacs-224, fyp-dr-code
    (no notebook output needed this time - nothing here reloads a checkpoint)

HOW TO RUN
    Cell 1      by hand  (~4 min)
    Cells 2-5   by COMMIT
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, cache, and every pre-flight check ------------------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

EPOCHS = 30
SEED = 42
CAPACITY_ARMS = ["A", "E", "B"]       # two best, one clear overfitter
CAPACITY_ARCH = "efficientnet_b0"
C2_ARCH = "resnet18"                  # arm C2 must match arm C to be comparable

# MEASURED on ResNet18 in stage 1, from the committed train_log.csv files. The hypothesis
# in cell 4 is tested against these, so they are constants here rather than re-derived:
# runs/ is not in the code bundle.
RESNET18_GAP = {"A": 0.092, "E": 0.073, "B": 0.290}
RESNET18_QWK = {"A": 0.6823, "E": 0.7081, "B": 0.5841}     # as run
RESNET18_SECONDS_PER_EPOCH = 70.6
B0_SECONDS_PER_EPOCH = 76.0                                # measured, stage 1 cell 4

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

from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

import cv2, numpy, pandas, timm, torch, torchvision
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  timm {timm.__version__}  |  numpy {numpy.__version__}  |  cv2 {cv2.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("no GPU. Session sidebar -> Accelerator -> GPU T4 x2.")
print(f"cuda: {torch.cuda.get_device_name(0)}")

for arch in (CAPACITY_ARCH, C2_ARCH):
    try:
        _ = timm.create_model(arch, pretrained=True, num_classes=5)
        print(f"pretrained weights {arch}: OK")
    except Exception as e:
        raise SystemExit(f"could not fetch weights for {arch}: {e}\nInternet -> On.")


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


assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"])[0] == 0
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "C2"])[0] == 0   # the new arm
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "E"])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

budget = 3 * (148 + 29 * B0_SECONDS_PER_EPOCH) / 60 + (148 + 29 * RESNET18_SECONDS_PER_EPOCH) / 60
print(f"\nplan: {CAPACITY_ARMS} on {CAPACITY_ARCH}, then C2 on {C2_ARCH}")
print(f"      {EPOCHS} epochs each, seed {SEED}; worst case ~{budget/60:.1f} h "
      "(early stopping will cut it)")
print("\nPREDICTION ON RECORD (docs/EXPERIMENTS.md): B0 does NOT close the gap.")
print("  arm B train-val gap stays >= 0.25; arms A and E move < 0.03 QWK.")
'''

CELL_2 = r'''
# -- Cell 2 -- the capacity test: A, E, B on EfficientNet-B0 ------------------
# COMMIT THIS. ~2 h worst case.
#
# Same budget, seed, splits and cache as stage 1. The ONLY change is the backbone, which
# is what makes this a test of capacity rather than of anything else.
results = {}
STATUS = RUNS / "_stage3_status.json"
RUNS.mkdir(parents=True, exist_ok=True)


def save_status():
    STATUS.write_text(json.dumps(results, indent=2), encoding="utf-8")


for arm in CAPACITY_ARMS:
    run_id = f"phase4_stage3_arm_{arm.lower()}_{CAPACITY_ARCH}"
    print("\n" + "#" * 78)
    print(f"# {CAPACITY_ARCH}  arm {arm}  ->  {run_id}")
    print("#" * 78, flush=True)

    rc, mins = run([
        sys.executable, "-m", "src.train.train",
        "--arm", arm,
        "--config", "configs/kaggle.yaml",
        "--cache-root", CACHE,
        "--run-id", run_id,
        "--runs-root", RUNS,
        "--arch", CAPACITY_ARCH,
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--num-workers", "2",
        "--log-every", "200",
    ], timeout_min=90)

    entry = {"arm": arm, "arch": CAPACITY_ARCH, "run_id": run_id,
             "exit_code": rc, "minutes": round(mins, 1), "ok": rc == 0}
    if rc == 0:
        m = json.loads((RUNS / run_id / "metrics.json").read_text())
        entry.update({"qwk": m["qwk"], "epochs_run": m["epochs_run"],
                      "best_epoch": m["best_epoch"],
                      "referable_sens": m["referable"]["sensitivity"]})
        print(f"\n  arm {arm} on {CAPACITY_ARCH}: qwk {m['qwk']:.4f}  "
              f"ref_sens {m['referable']['sensitivity']:.4f}  "
              f"({m['epochs_run']} epochs, best {m['best_epoch']}, {mins:.1f} min)")
    else:
        print(f"\n  *** arm {arm} FAILED (exit {rc}) - continuing ***")
    results[arm] = entry
    save_status()

print("\n" + "=" * 78)
print(f"capacity test finished: {sum(1 for e in results.values() if e['ok'])}/"
      f"{len(CAPACITY_ARMS)} arms")
print("=" * 78)
'''

CELL_3 = r'''
# -- Cell 3 -- arm C2: effective-number weighting, on ResNet18 ----------------
# COMMIT THIS. ~36 min.
#
# Arm C used inverse frequency (36:1 grade 4 to grade 0). Effective-number weighting is
# 17.1:1 on the same split. ResNet18, so it is directly comparable with arm C.
C2_RUN_ID = f"phase4_arm_c2_{C2_ARCH}"

rc_c2, mins_c2 = run([
    sys.executable, "-m", "src.train.train",
    "--arm", "C2",
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", C2_RUN_ID,
    "--runs-root", RUNS,
    "--arch", C2_ARCH,
    "--epochs", str(EPOCHS),
    "--seed", str(SEED),
    "--num-workers", "2",
    "--log-every", "200",
], timeout_min=90)

if rc_c2 == 0:
    m = json.loads((RUNS / C2_RUN_ID / "metrics.json").read_text())
    print(f"\narm C2: qwk {m['qwk']:.4f}  ref_sens {m['referable']['sensitivity']:.4f}  "
          f"({m['epochs_run']} epochs, best {m['best_epoch']})")
    print("  arm C (inverse frequency) was qwk 0.4048, ref_sens 0.6725, 17 epochs")
    print("  NOTE: compare these at MATCHED decision rules (DECISION-035), not as-run.")
    results["C2"] = {"arm": "C2", "arch": C2_ARCH, "run_id": C2_RUN_ID,
                     "exit_code": 0, "minutes": round(mins_c2, 1), "ok": True,
                     "qwk": m["qwk"], "epochs_run": m["epochs_run"],
                     "best_epoch": m["best_epoch"],
                     "referable_sens": m["referable"]["sensitivity"]}
    save_status()
else:
    print(f"\n*** arm C2 FAILED (exit {rc_c2}) ***")
'''

CELL_4 = r'''
# -- Cell 4 -- does the capacity hypothesis survive? --------------------------
# COMMIT THIS. Seconds.
#
# The prediction was written down before the run: B0 does NOT close the gap; arm B's
# train-val gap stays >= 0.25 and arms A and E move < 0.03 QWK.
import pandas as pd

print("CAPACITY HYPOTHESIS - predicted before the numbers landed\n")
print(f"{'arm':<5}{'r18 QWK':>9}{'b0 QWK':>9}{'dQWK':>8}   "
      f"{'r18 gap':>9}{'b0 gap':>9}{'dgap':>8}")
print("-" * 62)

verdict = {}
for arm in CAPACITY_ARMS:
    e = results.get(arm, {})
    if not e.get("ok"):
        print(f"{arm:<5}  FAILED - no verdict")
        continue
    log = pd.read_csv(RUNS / e["run_id"] / "train_log.csv")
    m = json.loads((RUNS / e["run_id"] / "metrics.json").read_text())
    row = log[log.epoch == m["best_epoch"]].iloc[0]
    gap = float(row.train_qwk - row.val_qwk)
    d_qwk = m["qwk"] - RESNET18_QWK[arm]
    d_gap = gap - RESNET18_GAP[arm]
    verdict[arm] = {"qwk_b0": m["qwk"], "gap_b0": gap, "d_qwk": d_qwk, "d_gap": d_gap}
    print(f"{arm:<5}{RESNET18_QWK[arm]:>9.4f}{m['qwk']:>9.4f}{d_qwk:>+8.4f}   "
          f"{RESNET18_GAP[arm]:>9.3f}{gap:>9.3f}{d_gap:>+8.3f}")

print()
if "B" in verdict:
    b = verdict["B"]
    held = b["gap_b0"] >= 0.25
    print(f"arm B gap {b['gap_b0']:.3f} >= 0.25 ?  {held}   "
          f"-> {'AS PREDICTED' if held else 'PREDICTION WRONG'}")
small = [a for a in ("A", "E") if a in verdict and abs(verdict[a]["d_qwk"]) < 0.03]
if small:
    print(f"arms {small} moved < 0.03 QWK -> as predicted")
big = [a for a in ("A", "E") if a in verdict and abs(verdict[a]["d_qwk"]) >= 0.03]
if big:
    print(f"arms {big} moved >= 0.03 QWK -> PREDICTION WRONG for them")

print("\nIf the prediction HELD: capacity is not the limit. The route to the")
print("sensitivity gap is regularisation and input resolution (handoff), not a")
print("bigger backbone - and that is now evidence rather than an argument.")
print("If it FAILED: capacity mattered after all; re-plan before spending on 384px.")

# And the matched comparison over everything that now has val outputs (DECISION-035).
have = sorted(d for d in RUNS.iterdir()
              if d.is_dir() and (d / "val_outputs.npz").exists())
if have:
    from src.data.notebook_check import validate_argv

    cmd = [sys.executable, "-m", "src.eval.compare_arms",
           "--runs-root", str(RUNS), "--pattern", "*"]
    why = validate_argv(cmd)
    if why:
        raise SystemExit(f"compare_arms command is malformed: {why}")
    run(cmd)
'''

CELL_5 = r'''
# -- Cell 5 -- keep the evidence ----------------------------------------------
import shutil

for p in sorted(RUNS.rglob("*")):
    if p.is_file() and p.suffix != ".pth":
        print(f"  {p.relative_to(RUNS)}  {p.stat().st_size / 1024:.0f} KB")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)

ids = " ".join(f"--run-id {e['run_id']}" for e in results.values() if e.get("ok"))
print("\nCommit this notebook, then locally:")
print()
print(f"  python -m src.data.fetch_run --kernel rah098/<this-notebook-slug> \\")
print(f"      {ids}")
print()
print("Then, and this is the part that produces the ranking:")
print("  python -m src.eval.compare_arms --pattern 'phase4_*' --compare A E")
print("  python -m notebooks.gen_experiments")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(CELLS[int(w) - 1])

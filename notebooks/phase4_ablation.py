"""Phase 4 stage 1 - the A-F imbalance ablation. Five Kaggle cells.

Six arms, one committed session, ~3.8 h. Every arm gets the SAME 30-epoch budget, the
same splits, the same cache and the same seed, because an arm that trains longer is not a
better arm and the comparison is the thesis chapter.

    python -m notebooks.phase4_ablation 1     # or 2, 3, 4, 5

ARM ISOLATION - the point of cell 2's design
--------------------------------------------
Each arm runs as its OWN SUBPROCESS and a failure is recorded, not raised. If arm C dies
at epoch 20 the loop notes it and starts arm D; the session finishes with five arms and a
named failure instead of three arms and a traceback.

Three reasons it is a subprocess rather than an in-process call:

  * a CUDA OOM or a segfault kills that process, not the notebook kernel
  * GPU memory is returned when the process exits, so one arm's OOM does not cascade
  * `train.py` already writes its own artefacts, so a crashed arm still leaves whatever
    epochs it completed in `train_log.csv`

Each arm also gets a wall-clock timeout. An arm that hangs would otherwise eat the whole
session; at 36 min expected, 90 min is generous and still bounded.

`_phase4_status.json` is rewritten after EVERY arm, so a session killed at arm E still
leaves a record of A-D on disk.

SESSION SETTINGS
    Accelerator : GPU T4 x2   Internet : ON   Inputs : fyp-dr-eyepacs-224, fyp-dr-code

HOW TO RUN
    Cell 1      by hand  (~3 min: checks, smoke, gate)
    Cells 2-5   by COMMIT: Save Version -> Save & Run All. Cell 2 is the 3.8 h one.
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, cache, versions, and every pre-flight check ---------------
# RUN INTERACTIVELY. About three minutes, and it is the cheapest three minutes in the
# phase: three sessions were lost to things this cell now catches (DECISION-022/025).
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

# The plan, in one place, so cell 2 has nothing to decide.
ARMS = ["A", "B", "C", "D", "E", "F"]
EPOCHS = 30                  # identical for every arm - DECISION-026 / EXPERIMENTS.md
SEED = 42
ARCH = "resnet18"
ARM_TIMEOUT_MIN = 90         # expected ~36 min; a hung arm must not eat the session
RUN_ID = {a: f"phase4_arm_{a.lower()}_{ARCH}" for a in ARMS}

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

# --- the cache, found by shape (DECISION-023) ---------------------------------
from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)
print(f"       {sum(1 for _ in CACHE.rglob('*.jpg'))} images, verified")

# --- versions (DECISION-019) --------------------------------------------------
import cv2, numpy, pandas, timm, torch, torchvision
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  torchvision {torchvision.__version__}  |  timm {timm.__version__}")
print(f"numpy {numpy.__version__}  |  pandas {pandas.__version__}  |  cv2 {cv2.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("no GPU. Session sidebar -> Accelerator -> GPU T4 x2.")
print(f"cuda: {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")

# Provenance: the bundle carries the SHA because Kaggle has no git checkout (DECISION-027)
bundle = REPO / "BUNDLE.txt"
print("bundle:", bundle.read_text().strip().replace("\n", " | ") if bundle.exists()
      else "BUNDLE.txt missing - runs will record git: unknown")

# --- pretrained weights need Internet ON --------------------------------------
try:
    _ = timm.create_model(ARCH, pretrained=True, num_classes=5)
    print("pretrained weights: OK")
except Exception as e:
    raise SystemExit(f"could not fetch pretrained weights: {e}\n"
                     "Session sidebar -> Internet -> On.")


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


# --- every check, before any GPU time is spent --------------------------------
# DECISION-022: every command line, against the real parsers.
assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"])[0] == 0

# DECISION-025: the real train.main(), end to end, on a synthetic fixture. Arms A and F
# between them touch every branch in main().
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "A"])[0] == 0
assert run([sys.executable, "-m", "src.train.smoke", "--arm", "F"])[0] == 0

# CLAUDE.md S7.
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

print(f"\nplan: {len(ARMS)} arms {ARMS}, {EPOCHS} epochs each, seed {SEED}, arch {ARCH}")
print(f"      expected ~36 min/arm, ~{6 * 36 / 60:.1f} h total")
'''

CELL_2 = r'''
# -- Cell 2 -- the ablation: six arms, sequential, ISOLATED --------------------
# COMMIT THIS. ~3.8 h.
#
# A failing arm is RECORDED, not raised. Arm C dying at epoch 20 must not cost arms D, E
# and F - the session ends with five results and one named failure instead of a traceback
# and three hours of unused quota.
STATUS = WORK / "runs" / "_phase4_status.json"
STATUS.parent.mkdir(parents=True, exist_ok=True)
results = {}


def save_status():
    # Rewritten after EVERY arm: a session killed at arm E still leaves A-D on disk.
    STATUS.write_text(json.dumps({
        "arms": ARMS, "epochs": EPOCHS, "seed": SEED, "arch": ARCH,
        "results": results,
    }, indent=2), encoding="utf-8")


def read_metrics(run_id):
    f = RUNS / run_id / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


for arm in ARMS:
    run_id = RUN_ID[arm]
    print("\n" + "#" * 78)
    print(f"# ARM {arm}  ->  {run_id}")
    print("#" * 78, flush=True)

    rc, mins = run([
        sys.executable, "-m", "src.train.train",
        "--arm", arm,
        "--config", "configs/kaggle.yaml",
        "--cache-root", CACHE,
        "--run-id", run_id,
        "--runs-root", RUNS,
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--num-workers", "2",
        "--log-every", "200",
    ], timeout_min=ARM_TIMEOUT_MIN)

    m = read_metrics(run_id)
    entry = {"run_id": run_id, "exit_code": rc, "minutes": round(mins, 1),
             "ok": rc == 0 and m is not None}

    if m is not None:
        entry.update({
            "qwk": m.get("qwk"), "qwk_lo": m.get("qwk_ci", {}).get("lo"),
            "qwk_hi": m.get("qwk_ci", {}).get("hi"),
            "accuracy": m.get("accuracy"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "referable_sens": m.get("referable", {}).get("sensitivity"),
            "epochs_run": m.get("epochs_run"), "best_epoch": m.get("best_epoch"),
            "collapse_level": m.get("collapse", {}).get("level"),
        })
        print(f"\n  arm {arm}: qwk {m['qwk']:.4f} "
              f"[{m['qwk_ci']['lo']:.4f}, {m['qwk_ci']['hi']:.4f}]  "
              f"bal {m['balanced_accuracy']:.4f}  "
              f"ref_sens {m['referable']['sensitivity']:.4f}  "
              f"({m['epochs_run']} epochs, best {m['best_epoch']}, {mins:.1f} min)")
    else:
        # No metrics.json means train.py did not reach the end. Whatever epochs it did
        # finish are still in train_log.csv, and that is worth fetching.
        log = RUNS / run_id / "train_log.csv"
        entry["partial_epochs"] = sum(1 for _ in log.open()) - 1 if log.exists() else 0
        print(f"\n  *** ARM {arm} FAILED (exit {rc}) - continuing to the next arm. "
              f"{entry['partial_epochs']} epoch(s) survive in train_log.csv ***")

    results[arm] = entry
    save_status()

failed = [a for a, e in results.items() if not e["ok"]]
print("\n" + "=" * 78)
print(f"ABLATION FINISHED - {len(ARMS) - len(failed)}/{len(ARMS)} arms produced metrics")
if failed:
    print(f"FAILED: {failed}. Their partial logs are still in runs/; diagnose before "
          "re-running only those arms.")
print("=" * 78)
'''

CELL_3 = r'''
# -- Cell 3 -- the comparison table -------------------------------------------
# Selection is on VALIDATION QWK (R3). The test set is untouched and stays that way
# until one model is chosen.
import pandas as pd

rows = []
for arm in ARMS:
    e = results.get(arm, {})
    if not e.get("ok"):
        rows.append({"arm": arm, "status": f"FAILED (exit {e.get('exit_code')})"})
        continue
    rows.append({
        "arm": arm,
        "qwk": round(e["qwk"], 4),
        "ci": f"[{e['qwk_lo']:.4f}, {e['qwk_hi']:.4f}]",
        "bal_acc": round(e["balanced_accuracy"], 4),
        "ref_sens": round(e["referable_sens"], 4),
        "acc": round(e["accuracy"], 4),
        "epochs": e["epochs_run"],
        "best": e["best_epoch"],
        "min": e["minutes"],
        "collapse": e["collapse_level"],
        "status": "ok",
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))

ok = [r for r in rows if r.get("status") == "ok"]
if ok:
    ranked = sorted(ok, key=lambda r: -r["qwk"])
    best = ranked[0]
    print(f"\nhighest val QWK: arm {best['arm']} at {best['qwk']:.4f} {best['ci']}")

    # A ranking inside the noise is not a ranking. The baseline's interval is +-0.028.
    close = [r for r in ranked[1:] if r["qwk"] >= best["qwk"] - 0.03]
    if close:
        print(f"within 0.03 of it: {[r['arm'] for r in close]} - NOT separable on one "
              "seed. Stage 2 repeats these at seeds 43 and 44 before any arm is called "
              "the winner.")
    print(f"\nstage 2 candidates: {[r['arm'] for r in ranked[:3]]}")

    # Arm F is selected on its EYEPACS-ONLY val QWK: its pooled val is a different
    # population and QWK is distribution-sensitive (DECISION-007).
    f = read_metrics(RUN_ID.get("F", ""))
    if f and "by_dataset" in f:
        eye = f["by_dataset"].get("eyepacs", {})
        print(f"\narm F pooled val qwk {f['qwk']:.4f}; EyePACS-only "
              f"{eye.get('qwk', float('nan')):.4f} <- use this one for cross-arm "
              "comparison (DECISION-007)")
        for ds, block in f["by_dataset"].items():
            print(f"    {ds:<8} n={block['n']:<5} qwk {block['qwk']:.4f}")

for r in rows:
    if r.get("status") != "ok":
        print(f"\narm {r['arm']}: {r['status']} - see runs/{RUN_ID[r['arm']]}/")
'''

CELL_4 = r'''
# -- Cell 4 -- MEASURE EfficientNet-B0's cost before committing to stage 3 -----
# The 2x-ResNet18 figure in the plan is an ESTIMATE. Three epochs on the real data turns
# it into a measurement, for about 7 minutes, and stage 3's budget stops being a guess.
#
# This is a TIMING PROBE, not a result: 3 epochs of a 30-epoch schedule is a different
# trajectory and its config says epochs: 3, so it can never be read as an arm.
rc, mins = run([
    sys.executable, "-m", "src.train.train",
    "--arm", "A",
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", "phase4_timing_efficientnet_b0",
    "--runs-root", RUNS,
    "--arch", "efficientnet_b0",
    "--epochs", "3",
    "--seed", str(SEED),
    "--num-workers", "2",
    "--log-every", "200",
], timeout_min=30)

if rc == 0:
    import pandas as pd
    log = pd.read_csv(RUNS / "phase4_timing_efficientnet_b0" / "train_log.csv")
    steady = log["train_seconds"][1:].mean() if len(log) > 1 else log["train_seconds"][0]
    r18_steady = 70.6          # MEASURED, phase3_baseline_resnet18
    full = (log["train_seconds"][0] + 29 * steady) / 60
    print(f"\nMEASURED efficientnet_b0: {steady:.0f} s/epoch steady "
          f"(epoch 0 {log['train_seconds'][0]:.0f} s)")
    print(f"  vs resnet18 {r18_steady:.0f} s/epoch  ->  {steady / r18_steady:.2f}x")
    print(f"  a full 30-epoch arm would cost {full:.0f} min; "
          f"stage 3 (2 arms) = {2 * full / 60:.1f} h")
    print("  ^ record this in EXPERIMENTS.md; the plan's 2x was an estimate")
else:
    print(f"\ntiming probe failed (exit {rc}) - stage 3 stays an estimate")
'''

CELL_5 = r'''
# -- Cell 5 -- keep the evidence ----------------------------------------------
# /kaggle/working is wiped between sessions. runs/ is the record (R4/R6).
import shutil

total = 0
for p in sorted(RUNS.rglob("*")):
    if p.is_file():
        kb = p.stat().st_size / 1024
        total += kb
        if p.suffix != ".pth":
            print(f"  {p.relative_to(RUNS)}  {kb:.0f} KB")
print(f"\n{total / 1024:.1f} MB in runs/ (checkpoints included; they are not fetched)")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)

print("\nCommit this notebook (Save Version -> Save & Run All), then locally:")
print()
ids = " ".join(f"--run-id {RUN_ID[a]}" for a in ARMS)
print(f"  python -m src.data.fetch_run --kernel rah098/<this-notebook-slug> \\")
print(f"      {ids}")
print()
print("fetch_run verifies each run's three artefacts and their agreement with each other")
print("before installing, and never copies *.pth. Then: python -m notebooks.gen_experiments")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(CELLS[int(w) - 1])

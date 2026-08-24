"""Phase 4 stage 2 - seed stability for arms E and A on EfficientNet-B0. Four cells, ~2.4 h.

    python -m notebooks.phase4_stage2 1     # or 2, 3, 4

WHAT THIS CAN SETTLE, AND WHAT IT CANNOT. Read this before the numbers land, because the
tempting claim is the one it does not support.

IT CAN settle:
  * the SEED VARIANCE of the headline number, so the thesis reports arm E's QWK as a
    mean over seeds with a spread, rather than as one draw presented as a fact
  * whether the operating point is stable - sens@spec>=0.95 matters more than QWK for
    deployment and it is the number closest to a floor
  * whether arm E's unusually small generalisation gap (0.034) is a property of the ARM
    or of that one seed
  * whether the sign of the A-vs-E difference is even consistent across seeds, reported
    as a descriptive fact and never as a significance claim

IT CANNOT settle the A-vs-E tie. Not at three seeds and not at thirty.

    Every seed is evaluated on THE SAME 5,268 validation images. Averaging over seeds
    reduces the training-stochasticity component of the uncertainty and does NOTHING to
    the validation-set sampling component, which is the same images every time and is a
    fixed floor. The A-vs-E effect on B0 is +0.0224 QWK and the paired interval's
    half-width is about 0.027 - the effect is SMALLER than a floor that seed-averaging
    cannot lower. More seeds move the point estimate around; they do not move that
    interval off zero.

    Properly accounting for seed variance makes the honest interval WIDER than the
    single-seed paired CI, which understates uncertainty by ignoring training noise
    altogether. So stage 2's effect on the tie, if any, is to make it more of a tie.

    A and E are reported as a tie (DECISION-035). Arm E is carried forward on the
    secondary criteria that were stated before this run: a better operating point, a
    smaller gap, and an output already calibrated in grade units.

THE REPORTED NUMBER IS THE MEAN ACROSS SEEDS, NEVER THE BEST SEED. Picking the best of
three would be a fourth architecture-or-configuration choice made on validation, against
a soft budget of four with a positive-argument gate (DECISION-036). Stage 2 is a
measurement, not a selection step, and it spends none of that budget.

SEEDS 42, 43, 44 - sequential and obviously arbitrary. Seed 42 already exists for both
arms from stage 3, so only FOUR new runs are needed rather than six.

    arm E, seed 42  ->  runs/phase4_stage3_arm_e_efficientnet_b0   (have it)
    arm A, seed 42  ->  runs/phase4_stage3_arm_a_efficientnet_b0   (have it)
    arms E and A, seeds 43 and 44                                  (this notebook)

SESSION SETTINGS
    Accelerator : GPU T4 x2      Internet : ON
    Inputs      : fyp-dr-eyepacs-224, fyp-dr-code, AND the stage 3 notebook's output
                  (needed in cell 3 to compare against the seed-42 runs; if you skip it,
                  cell 3 says so and you run the comparison locally instead)

HOW TO RUN
    Cell 1     by hand  (~4 min)
    Cells 2-4  by COMMIT
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, cache, pre-flight ---------------------------------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

EPOCHS = 30
ARCH = "efficientnet_b0"          # FIXED by DECISION-039. Not a variable here.
ARMS = ["E", "A"]
NEW_SEEDS = [43, 44]              # 42 already exists for both arms, from stage 3
ALL_SEEDS = [42] + NEW_SEEDS

# MEASURED, stage 3, seed 42, committed artefacts. Cell 3 tests spread against these.
SEED42 = {
    "E": {"qwk": 0.7615, "held_out": 0.7560, "gap": 0.034, "sens_at_spec": 0.7464},
    "A": {"qwk": 0.6975, "held_out": 0.7327, "gap": 0.157, "sens_at_spec": 0.6822},
}
B0_SECONDS_PER_EPOCH = 76.0

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

try:
    timm.create_model(ARCH, pretrained=True, num_classes=1)
    print(f"pretrained weights {ARCH}: OK")
except Exception as e:
    raise SystemExit(f"could not fetch weights for {ARCH}: {e}\nInternet -> On.")


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
for _arm in ARMS:
    assert run([sys.executable, "-m", "src.train.smoke",
                "--arm", _arm, "--arch", ARCH])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

n = len(ARMS) * len(NEW_SEEDS)
budget = n * (148 + 29 * B0_SECONDS_PER_EPOCH) / 3600
print(f"\nplan: arms {ARMS} on {ARCH}, seeds {NEW_SEEDS} ({n} runs; seed 42 already exists)")
print(f"      {EPOCHS} epochs each; worst case ~{budget:.1f} h (early stopping will cut it)")
print("""
WHAT THIS CAN AND CANNOT SETTLE - on record before the numbers land:

  CAN     seed variance of the headline number; stability of the operating point;
          whether arm E's 0.034 gap is the arm or the seed
  CANNOT  the A-vs-E tie. Every seed sees the SAME 5,268 validation images, so
          seed-averaging cannot lower the image-sampling floor. The effect (+0.0224)
          is smaller than that floor's half-width (~0.027). Accounting for seed
          variance makes the honest interval WIDER, not narrower.

  The reported number is the MEAN ACROSS SEEDS, never the best seed. Best-of-three
  would be a selection on validation and this is a measurement (DECISION-036).
""")
'''

CELL_2 = r'''
# -- Cell 2 -- four runs: arms E and A, seeds 43 and 44 -----------------------
# COMMIT THIS. ~2.4 h worst case.
#
# Per-run isolation as in stage 1: a failure is recorded and the sweep continues, so one
# bad run does not cost the session.
RUNS.mkdir(parents=True, exist_ok=True)
STATUS = RUNS / "_stage2_status.json"
results = {}


def save_status():
    STATUS.write_text(json.dumps(results, indent=2), encoding="utf-8")


for arm in ARMS:
    for seed in NEW_SEEDS:
        run_id = f"phase4_stage2_arm_{arm.lower()}_{ARCH}_s{seed}"
        print("\n" + "#" * 78)
        print(f"# {ARCH}  arm {arm}  seed {seed}  ->  {run_id}")
        print("#" * 78, flush=True)

        rc, mins = run([
            sys.executable, "-m", "src.train.train",
            "--arm", arm,
            "--config", "configs/kaggle.yaml",
            "--cache-root", CACHE,
            "--run-id", run_id,
            "--runs-root", RUNS,
            "--arch", ARCH,
            "--epochs", str(EPOCHS),
            "--seed", str(seed),
            "--num-workers", "2",
            "--log-every", "200",
        ], timeout_min=90)

        entry = {"arm": arm, "arch": ARCH, "seed": seed, "run_id": run_id,
                 "exit_code": rc, "minutes": round(mins, 1), "ok": rc == 0}
        if rc == 0:
            m = json.loads((RUNS / run_id / "metrics.json").read_text())
            entry.update({"qwk": m["qwk"], "epochs_run": m["epochs_run"],
                          "best_epoch": m["best_epoch"],
                          "referable_sens": m["referable"]["sensitivity"]})
            print(f"\n  arm {arm} seed {seed}: qwk {m['qwk']:.4f}  "
                  f"({m['epochs_run']} epochs, best {m['best_epoch']}, {mins:.1f} min)")
        else:
            print(f"\n  *** arm {arm} seed {seed} FAILED (exit {rc}) - continuing ***")
        results[f"{arm}/s{seed}"] = entry
        save_status()

ok = sum(1 for e in results.values() if e["ok"])
print("\n" + "=" * 78)
print(f"stage 2 finished: {ok}/{len(results)} runs")
print("=" * 78)
'''

CELL_3 = r'''
# -- Cell 3 -- seed spread, and the claim it does NOT support -----------------
# COMMIT THIS. Seconds.
import statistics

print("SEED STABILITY - arms E and A on EfficientNet-B0\n")
print("As-run validation QWK per seed. Seed 42 is the stage 3 run.\n")

table = {}
for arm in ARMS:
    vals = {42: SEED42[arm]["qwk"]}
    for seed in NEW_SEEDS:
        e = results.get(f"{arm}/s{seed}", {})
        if e.get("ok"):
            vals[seed] = e["qwk"]
    table[arm] = vals
    got = [vals[s] for s in sorted(vals)]
    line = "  ".join(f"s{s}={vals[s]:.4f}" for s in sorted(vals))
    if len(got) > 1:
        mean, sd = statistics.mean(got), statistics.pstdev(got)
        print(f"  arm {arm}:  {line}   mean {mean:.4f}  sd {sd:.4f}  "
              f"range {max(got)-min(got):.4f}")
    else:
        print(f"  arm {arm}:  {line}   (only one seed succeeded - no spread to report)")

print("""
HOW TO READ THIS, and how not to.

  The mean is the headline number. The spread is what makes it a claim rather than a
  draw. Report both; report the mean, never the best seed.

  If the two arms' seed ranges OVERLAP, that is consistent with the tie already found
  and adds nothing to it. If they do NOT overlap, that is still not a significance
  test: three points give a spread, not an interval, and every seed was scored on the
  same 5,268 images, so the image-sampling uncertainty is common to all of them and
  does not average away.

  THE A-VS-E TIE STANDS EITHER WAY (DECISION-035). Arm E is carried forward on the
  secondary criteria stated before this run - operating point, generalisation gap, and
  an output already calibrated in grade units - not because it won a comparison.
""")

# The matched comparison over every seed of both arms, which is where the labels earn
# their keep: before DECISION-040 these six runs collapsed to two rows.
seed42 = [RUNS / f"phase4_stage3_arm_{a.lower()}_{ARCH}" for a in ARMS]
if not all((d / "val_outputs.npz").exists() for d in seed42):
    print("\nThe seed-42 runs are not in this session (/kaggle/working is wiped, and the")
    print("stage 3 output was not attached as an input). Fetch these four runs and run")
    print("the comparison locally - the command is in cell 4.")
else:
    from src.data.notebook_check import validate_argv

    cmd = [sys.executable, "-m", "src.eval.compare_arms",
           "--runs-root", str(RUNS), "--pattern", f"phase4_*arm_[ae]_{ARCH}*",
           "--repeats", "200", "--operating-point"]
    why = validate_argv(cmd)
    if why:
        raise SystemExit(f"compare_arms command is malformed: {why}")
    run(cmd)
'''

CELL_4 = r'''
# -- Cell 4 -- keep the evidence ----------------------------------------------
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
print("Then the comparison over all six runs (three seeds x two arms):")
print(f"  python -m src.eval.compare_arms --pattern 'phase4_*arm_[ae]_{ARCH}*' \\")
print("      --repeats 200 --operating-point")
print("  python -m notebooks.gen_experiments")
print()
print("Next after this: the 384px decision on arm E alone, then Phase 5.")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4"]):
        print(CELLS[int(w) - 1])

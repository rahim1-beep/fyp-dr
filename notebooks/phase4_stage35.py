"""Phase 4 stage 3.5 - one backbone step, arm E on EfficientNet-B2 @224. Three cells, ~1 h.

    python -m notebooks.phase4_stage35 1     # or 2, 3

THIS IS THE LAST BACKBONE STEP. The stopping rule is in cell 1 and in
`docs/EXPERIMENTS.md`, written before the run. There is no B3 cell and there will not be
one: a ladder that only stops when it stops climbing is not an experiment, it is a search
whose optimism nobody is counting.

WHAT VARIABLE THIS MANIPULATES - measured locally with timm and
`torch.utils.flop_counter`, not quoted from a paper:

    backbone            params    GMAC @224   native input
    resnet18            11.18M      1.814        224
    efficientnet_b0      4.01M      0.385        224
    efficientnet_b2      7.70M      0.658        256

The step is +3.69M parameters (+92%) and +0.273 GMAC (+71%) over B0, and it stays BELOW
ResNet18 on both. That matters for what may be claimed: the whole ladder sits under the
baseline in parameters AND in FLOPs, so no gain anywhere on it can be attributed to
capacity in either sense. What varies is architecture family and pretrained-feature
quality.

THE CONFOUND, STATED IN ADVANCE. B2's native input is 256; we run it at 224 to hold
resolution fixed against every other run in the project. B0's native input IS 224. So B2
is handicapped and B0 is not, which means:

    A NULL RESULT HERE DOES NOT SHOW THE BACKBONE LEVER IS EXHAUSTED.
    It shows this step, at this resolution, did not pay. Those are different claims and
    only the second one goes in the write-up.

PREDICTION (on record before the run): B2 will NOT beat B0 by a separable margin.
Expected |dQWK| < 0.02 with a paired interval spanning zero. The reasoning: the
ResNet18 -> B0 gain of +0.0408 [+0.0181, +0.0654] came from crossing architecture
FAMILIES - a different inductive bias and a stronger ImageNet initialisation. B0 -> B2 is
within-family compound scaling at a below-native input size, which is the weakest form of
that same lever.

THE STOPPING RULE - all four outcomes stop.

    paired held-out dQWK (B2 - B0)      decision
    ---------------------------------   --------------------------------------------
    not separable (spans zero)          KEEP B0. Simpler, cheaper, native at 224.
    separable and positive              KEEP B2. Do NOT run B3.
    separable and negative              KEEP B0, and note the off-native confound.
    any of the above, but B2 clears     report it as deployment-relevant, and still
    sens@spec>=0.95 of 0.80             do NOT run B3.

Nothing on this table escalates. Reopening the ladder needs a positive argument written
down first and logged as a decision (DECISION-036), never momentum from a good result.

SESSION SETTINGS
    Accelerator : GPU T4 x2      Internet : ON      Inputs : fyp-dr-eyepacs-224, fyp-dr-code

HOW TO RUN
    Cell 1    by hand  (~4 min)
    Cells 2-3 by COMMIT
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, cache, pre-flight, and the cost check -------------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

EPOCHS = 30
SEED = 42
ARM = "E"                       # the ordinal head: the only arm whose gap NARROWED on B0
ARCH = "efficientnet_b2"
BASELINE_ARCH = "efficientnet_b0"

# MEASURED, stage 3, from the committed artefacts. The verdict in cell 3 is tested
# against these, so they are constants here: runs/ is not in the code bundle.
B0_QWK_AS_RUN = 0.7615
B0_QWK_HELD_OUT = 0.7564        # matched decision rule, DECISION-035
B0_GAP = 0.034
B0_SENS_AT_SPEC95 = 0.7464
SENS_FLOOR = 0.80               # DECISION-032, PROVISIONAL pending supervisor
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
    _probe = timm.create_model(ARCH, pretrained=True, num_classes=1)
except Exception as e:
    raise SystemExit(f"could not fetch weights for {ARCH}: {e}\nInternet -> On.")

# The manipulated variable, restated from the model that is actually about to train
# rather than from the docstring. A stale docstring is harmless; a stale number in a
# write-up is not.
n_params = sum(p.numel() for p in _probe.parameters())
_cfg = timm.get_pretrained_cfg(ARCH)
print(f"\n{ARCH}: {n_params/1e6:.2f}M params, weights '{_cfg.tag}', "
      f"native input {_cfg.input_size} - WE RUN AT 224 (below native; see the docstring)")


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
# --arch matters here: without it the smoke runs arm E on the DEFAULT backbone and
# the B2 code path's first ever execution would be the hour-long one.
assert run([sys.executable, "-m", "src.train.smoke",
            "--arm", ARM, "--arch", ARCH])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

# Ten seconds of measurement instead of an estimate. Depthwise convolutions are
# memory-bandwidth bound rather than FLOP bound, so B2's 1.71x GMAC over B0 does NOT
# imply 1.71x wall clock, in either direction. Every cost surprise in this project so far
# has come from not checking a number that took seconds to check.
_m = _probe.cuda().train()
_opt = torch.optim.AdamW(_m.parameters(), lr=1e-4)
_x = torch.randn(64, 3, 224, 224, device="cuda")
_y = torch.randn(64, 1, device="cuda")
for _i in range(6):
    if _i == 2:
        torch.cuda.synchronize()
        _t0 = time.time()
    _opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        _loss = torch.nn.functional.smooth_l1_loss(_m(_x), _y)
    _loss.backward()
    _opt.step()
torch.cuda.synchronize()
_per_step = (time.time() - _t0) / 4
_est = _per_step * (24586 // 64)
print(f"\nmeasured {_per_step*1000:.0f} ms/step at batch 64 -> ~{_est:.0f} s/epoch "
      f"({_est/B0_SECONDS_PER_EPOCH:.2f}x B0's {B0_SECONDS_PER_EPOCH:.0f} s)")
print(f"worst case {EPOCHS} epochs: ~{(_est*EPOCHS + 80)/3600:.1f} h "
      "(early stopping will cut it)")
if _est * EPOCHS > 5400:
    print("  ** over 1.5 h: the cell-2 timeout is 100 min. Reduce EPOCHS or raise it. **")
del _m, _opt, _x, _y, _probe
torch.cuda.empty_cache()

print(f"""
PREDICTION ON RECORD (docs/EXPERIMENTS.md, DECISION-038):
  B2 will NOT beat B0 by a separable margin. |dQWK| < 0.02, interval spanning zero.
  B0 held-out {B0_QWK_HELD_OUT:.4f}, sens@spec>=.95 {B0_SENS_AT_SPEC95:.4f} vs floor {SENS_FLOOR}.

THIS IS THE LAST BACKBONE STEP. Every outcome in the stopping table stops. No B3 cell.
""")
'''

CELL_2 = r'''
# -- Cell 2 -- arm E on EfficientNet-B2 @224 ---------------------------------
# COMMIT THIS. ~1 h.
#
# Same budget, seed, splits, cache and image size as every other Phase 4 run. The ONLY
# change from the B0 run is --arch, which is what makes the difference attributable.
RUNS.mkdir(parents=True, exist_ok=True)
RUN_ID = f"phase4_stage35_arm_{ARM.lower()}_{ARCH}"
STATUS = RUNS / "_stage35_status.json"

print("#" * 78)
print(f"# {ARCH}  arm {ARM}  ->  {RUN_ID}")
print("#" * 78, flush=True)

rc, mins = run([
    sys.executable, "-m", "src.train.train",
    "--arm", ARM,
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", RUN_ID,
    "--runs-root", RUNS,
    "--arch", ARCH,
    "--epochs", str(EPOCHS),
    "--seed", str(SEED),
    "--num-workers", "2",
    "--log-every", "200",
], timeout_min=100)

result = {"arm": ARM, "arch": ARCH, "run_id": RUN_ID, "exit_code": rc,
          "minutes": round(mins, 1), "ok": rc == 0}
if rc == 0:
    m = json.loads((RUNS / RUN_ID / "metrics.json").read_text())
    result.update({"qwk": m["qwk"], "epochs_run": m["epochs_run"],
                   "best_epoch": m["best_epoch"],
                   "referable_sens": m["referable"]["sensitivity"]})
    print(f"\n  arm {ARM} on {ARCH}: qwk {m['qwk']:.4f} (as run)  "
          f"ref_sens {m['referable']['sensitivity']:.4f}  "
          f"({m['epochs_run']} epochs, best {m['best_epoch']}, {mins:.1f} min)")
    print(f"  B0 was {B0_QWK_AS_RUN:.4f} as run. THIS IS NOT THE COMPARISON - cell 3 is.")
else:
    print(f"\n  *** FAILED (exit {rc}) - nothing to compare, stop here ***")
STATUS.write_text(json.dumps(result, indent=2), encoding="utf-8")
'''

CELL_3 = r'''
# -- Cell 3 -- the stopping rule, applied ------------------------------------
# COMMIT THIS. Seconds.
#
# The decision was written down before the run. This cell APPLIES it rather than
# interpreting it, which is the whole point of having written it down.
import numpy as np, pandas as pd

if not result.get("ok"):
    raise SystemExit("the run failed; no verdict")

from src.eval.compare_arms import load_run, continuous_score, paired_difference
from src.eval.thresholds import (choose_operating_point, referable_scores,
                                 sensitivity_specificity_curve)

b2 = load_run(RUNS / RUN_ID)
b0_dir = RUNS / f"phase4_stage3_arm_e_{BASELINE_ARCH}"
if not (b0_dir / "val_outputs.npz").exists():
    print(f"{b0_dir} is not in this session (/kaggle/working is wiped between sessions).")
    print("Fetch this run locally and compare there:")
    print(f"  python -m src.data.fetch_run --kernel rah098/<slug> --run-id {RUN_ID}")
    print("  python -m src.eval.compare_arms --pattern 'phase4_*'")
    raise SystemExit(0)

b0 = load_run(b0_dir)
assert np.array_equal(b2["y"], b0["y"]), "different validation sets - not pairable"

d = paired_difference(continuous_score(b2["outputs"]), continuous_score(b0["outputs"]),
                      b2["y"], repeats=200, seed=0)

log = pd.read_csv(RUNS / RUN_ID / "train_log.csv")
m = json.loads((RUNS / RUN_ID / "metrics.json").read_text())
row = log[log.epoch == m["best_epoch"]].iloc[0]
gap = float(row.train_qwk - row.val_qwk)

curve = sensitivity_specificity_curve(
    referable_scores(b2["outputs"], "ordinal_regression"), b2["y"])
op = choose_operating_point(curve)
sens = (op.get("max_sens_at_spec") or {}).get("sensitivity", float("nan"))

print("BACKBONE STEP - the stopping rule, applied\n")
print(f"  held-out QWK (B2 - B0)   {d['mean']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]")
print(f"  separable                {d['separable']}     (B2 better in "
      f"{d['a_better_fraction']*100:.0f}% of resamples)")
print(f"  train-val gap            {gap:.3f}   (B0 was {B0_GAP:.3f})")
print(f"  sens @ spec >= 0.95      {sens:.4f}   (B0 {B0_SENS_AT_SPEC95:.4f}, "
      f"floor {SENS_FLOOR} [PROVISIONAL - DECISION-032])")

predicted = (not d["separable"]) and abs(d["mean"]) < 0.02
print("\n  PREDICTION was: not separable, |dQWK| < 0.02")
print(f"  -> {'AS PREDICTED' if predicted else 'PREDICTION WRONG'}")

if not d["separable"]:
    keep = BASELINE_ARCH
    why = "not separable from B0; B0 is smaller, cheaper, and native at 224"
elif d["mean"] > 0:
    keep = ARCH
    why = "separably better than B0"
else:
    keep = BASELINE_ARCH
    why = ("separably WORSE than B0 - but B2 ran below its native 256, so this does NOT "
           "show the backbone lever is exhausted, only that this step at this resolution "
           "did not pay")

print(f"\n  DECISION: keep {keep}  ({why})")
if sens >= SENS_FLOOR:
    print(f"  NOTE: sens {sens:.4f} clears the provisional floor. Deployment-relevant, "
          "and still not a reason to run B3.")
print("\n  THE BACKBONE QUESTION IS NOW CLOSED. Next: stage 2 seeds on "
      f"{keep} (arms E and A, 3 seeds), then the 384px decision on arm E alone.")
print("  Reopening it needs a positive argument logged first (DECISION-036), never a "
      "good result on its own.")

print("\n\nCommit this notebook, then locally:")
print(f"  python -m src.data.fetch_run --kernel rah098/<this-notebook-slug> "
      f"--run-id {RUN_ID}")
print("  python -m src.eval.compare_arms --pattern 'phase4_*'")
print("  python -m notebooks.gen_experiments")
'''

CELLS = [CELL_1, CELL_2, CELL_3]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3"]):
        print(CELLS[int(w) - 1])

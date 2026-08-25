"""Phase 4, final item - the 384px ablation (DECISION-044). TWO Kaggle sessions, ~11 h.

    python -m notebooks.phase4_res384 1     # cells 1-4 = SESSION A (cache, ~8-9 h)
    python -m notebooks.phase4_res384 5     # cells 5-7 = SESSION B (train, ~2 h)

RUN THIS AFTER PHASE 5 AND PHASE 6, not before. Phase 5's rim gate can invalidate the
preprocessing this cache would be built with, and rebuilding at 384 twice is 16-18 hours.
See DECISION-045.

=============================================================================
 THIS RESULT CANNOT BECOME THE HEADLINE NUMBER. HARD GATE, NOT A NOTE.
=============================================================================

The proposal fixes **224 for all headline results** (CLAUDE.md section 5). Adopting 384
is a DEVIATION and needs the supervisor's approval IN WRITING before anything downstream
changes. Concretely, even if this run clears 0.78:

    * the headline model stays arm E on B0 at 224
    * no seeds are run at 384
    * docs/EXPERIMENTS.md's headline is not touched
    * the EyePACS test set is NOT opened on a 384 model
    * Phase 6's APTOS numbers are not re-run

until `docs/DECISIONS.md` carries a decision recording written supervisor approval, with
the date. Cell 7 refuses to print an escalation path and prints the gate instead. A good
result is not permission.

WHAT THIS MANIPULATES

Input resolution ONLY: 224 -> 384, which is 2.94x the pixels ((384/224)^2). Arm E,
EfficientNet-B0, 30 epochs, seed 42, same splits, same patients, same preprocessing
parameters. It is a DATA change - the cache is rebuilt so the model sees genuine detail
rather than an upsampled 224 image.

THE PREDICTION, on record before the run

    sens@spec>=0.95 improves but does NOT reach 0.80, landing around 0.76-0.80, and
    QWK improves by LESS than sensitivity does.

Microaneurysms are ~50 um against a ~3000 px fundus, so at 224 they are sub-pixel. They
define grades 1-2, which is exactly where the referable boundary sits - so resolution
should move the operating point more than it moves overall ordinal agreement, which is
already dominated by the easy grade-0 mass.

THE STOPPING RULE - against the 224 three-seed RANGE, never a point estimate

The 224 baseline is NOT 0.7464. It is 0.7211-0.7464, mean 0.7360, over seeds 42/43/44.
A single 384 seed must clear the TOP of that range to mean anything. This is
DECISION-042's lesson made structural: the 0.7464 that would have been the comparator
was the most extreme of three seeds.

    384 sens@spec>=0.95     reading                      action
    --------------------    -------------------------    ----------------------------
    >= 0.78                 clearly above the range      STOP. Write it up. Ask the
                                                         supervisor for the deviation.
                                                         Nothing else moves until the
                                                         approval is logged.
    0.7464 - 0.78           one seed above a 3-seed      suggestive, NOT established.
                            range                        Report as the ablation. Stop.
    < 0.7464                inside or below the range    clean negative. Report. Stop.

Every row stops. There is no 512 cell, no other-arms-at-384 cell, and no
adopt-as-headline cell.

SESSION SETTINGS
    Both sessions : GPU T4 x2, Internet ON
    Session A     : inputs dreamer07/eyepacs, mariaherrerot/aptos2019, fyp-dr-code
    Session B     : inputs fyp-dr-eyepacs-384 (published by session A), fyp-dr-code

HOW TO RUN
    SESSION A   cell 1 by hand (~4 min), cells 2-4 by COMMIT (~8-9 h)
    SESSION B   cell 5 by hand (~4 min), cells 6-7 by COMMIT (~2 h)
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- SESSION A: setup and pre-flight for the 384 cache build --------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
OUT = WORK / "processed384"
IMAGE_SIZE = 384
EXPECT_IMAGES = 38788

print("mounted under /kaggle/input:")
for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
    if root.is_dir():
        print(f"  {root}: {sorted(p.name for p in root.iterdir())[:20]}")

CODE = None
for cand in (Path("/kaggle/input/datasets/rah098/fyp-dr-code"),
             Path("/kaggle/input/fyp-dr-code")):
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

from src.data.kaggle_paths import resolve_input

EYEPACS = resolve_input("eyepacs", owner="dreamer07")
APTOS = resolve_input("aptos2019", owner="mariaherrerot")
print("eyepacs:", EYEPACS)
print("aptos  :", APTOS)


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
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

# The --image-size flag is the whole experiment. If it did not reach PreprocessConfig the
# build would produce a 224 cache over 8 hours and every check downstream would pass,
# because a 224 cache IS a valid cache -- just not this one. Twenty images prove the
# plumbing before the other 38,768.
PROBE = WORK / "probe384"
shutil.rmtree(PROBE, ignore_errors=True)
rc, _ = run([sys.executable, "-m", "src.data.preprocess",
             "--split", "val", "--src-root", EYEPACS, "--out-root", PROBE,
             "--image-size", str(IMAGE_SIZE), "--limit", "20", "--no-scan",
             "--workers", "2"])
assert rc == 0, "probe build failed"

import cv2, numpy as np
probe_files = [p for p in PROBE.rglob("*.jpeg") if p.is_file()]
assert probe_files, "probe produced no images"
shapes = {cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR).shape
          for p in probe_files[:20]}
print(f"\nprobe wrote {len(probe_files)} images, shapes {shapes}")
assert shapes == {(IMAGE_SIZE, IMAGE_SIZE, 3)}, (
    f"--image-size did NOT take: got {shapes}, wanted {(IMAGE_SIZE, IMAGE_SIZE, 3)}. "
    "STOP - an 8-hour build would produce the wrong cache.")

prov = json.loads((PROBE / "_cache_provenance.json").read_text())
assert prov["runs"][-1]["preprocess"]["image_size"] == IMAGE_SIZE, (
    "the provenance sidecar does not record 384; reconcile_cache would then certify a "
    "cache whose pipeline cannot be verified")
print("provenance records image_size:", prov["runs"][-1]["preprocess"]["image_size"])
shutil.rmtree(PROBE, ignore_errors=True)

print(f"""
PLUMBING VERIFIED at {IMAGE_SIZE}px. Cells 2-4 are ~8-9 h; commit them.

PREDICTION ON RECORD (DECISION-044): sens@spec>=0.95 improves to roughly 0.76-0.80
and does NOT reach the 0.80 floor; QWK improves less than sensitivity does.

THIS RESULT CANNOT BECOME THE HEADLINE without written supervisor approval logged as
a decision. The proposal fixes 224. A good number is not permission.
""")
'''

CELL_2 = r'''
# -- Cell 2 -- SESSION A: build the 384 cache --------------------------------
# COMMIT THIS. ~8-9 h. EyePACS then APTOS, same pipeline, only image_size differs.
OUT.mkdir(parents=True, exist_ok=True)

rc_e, mins_e = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "train", "--split", "val", "--split", "test",
    "--src-root", EYEPACS, "--out-root", OUT,
    "--stats", str(WORK / "processed384_stats.csv"),
    "--image-size", str(IMAGE_SIZE),
    "--no-scan", "--workers", "4",
], timeout_min=600)
print(f"\nEyePACS at {IMAGE_SIZE}px: exit {rc_e} in {mins_e/60:.1f} h")
assert rc_e == 0, "EyePACS build failed - do not continue to APTOS"

rc_a, mins_a = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "aptos_train", "--split", "aptos_val", "--split", "aptos_test",
    "--src-root", APTOS, "--out-root", OUT,
    "--stats", str(WORK / "aptos384_stats.csv"),
    "--image-size", str(IMAGE_SIZE),
    "--no-scan", "--workers", "4",
], timeout_min=180)
print(f"APTOS at {IMAGE_SIZE}px: exit {rc_a} in {mins_a/60:.1f} h")
assert rc_a == 0, "APTOS build failed"

n = sum(1 for p in OUT.rglob("*.jpeg") if p.is_file())
gb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1024**3
print(f"\n{n} images, {gb:.3f} GB  (224 cache was 38,788 images / 0.836 GB)")
assert n == EXPECT_IMAGES, f"expected {EXPECT_IMAGES} images, got {n}"
'''

CELL_3 = r'''
# -- Cell 3 -- SESSION A: reconcile, then leakage gate ------------------------
# COMMIT THIS. ~15 min. Nothing is published before both of these pass.
rc, _ = run([
    sys.executable, "-m", "src.data.reconcile_cache",
    "--cache-root", OUT,
    "--stats", str(WORK / "processed384_stats.csv"),
    "--stats", str(WORK / "aptos384_stats.csv"),
])
assert rc == 0, "reconciliation FAILED - do not publish this cache"

assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0
print("\nreconciled and leakage-gated at 384.")
'''

CELL_4 = r'''
# -- Cell 4 -- SESSION A: pack, verify, publish -------------------------------
# COMMIT THIS. ~20 min.
#
# Pack into ONE archive and VERIFY its central directory before deleting anything
# (DECISION-021: 38,289 images and 2.5 h of compute went to Kaggle's 500-file output cap).
from src.data.notebook_check import validate_argv

ARCHIVE = WORK / "fyp-dr-eyepacs-384.zip"
PACK_CMD = [sys.executable, "-m", "src.data.archive_cache", "pack",
            "--cache-root", str(OUT), "--out", str(ARCHIVE)]
why = validate_argv(PACK_CMD)
if why:
    raise SystemExit(f"pack command is malformed: {why}")
assert run(PACK_CMD, timeout_min=60)[0] == 0

VERIFY_CMD = [sys.executable, "-m", "src.data.archive_cache", "verify",
              "--archive", str(ARCHIVE), "--expect-images", str(EXPECT_IMAGES)]
why = validate_argv(VERIFY_CMD)
if why:
    raise SystemExit(f"verify command is malformed: {why}")
assert run(VERIFY_CMD)[0] == 0, "archive did NOT verify - keep the source directory"

print(f"\narchive verified: {ARCHIVE.stat().st_size / 1024**3:.3f} GB")
shutil.rmtree(OUT, ignore_errors=True)
os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)

print("""
Now publish it as a private Kaggle Dataset named  fyp-dr-eyepacs-384

  Output tab -> the .zip -> New Dataset -> title "fyp-dr-eyepacs-384", Private.

Kaggle AUTO-EXTRACTS published archives (DECISION-023), which is why session B uses
resolve_cache() to find the cache by SHAPE rather than by path.

Then start SESSION B: a fresh notebook, cells 5-7.
""")
'''

CELL_5 = r'''
# -- Cell 5 -- SESSION B: setup against the 384 cache -------------------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"

ARM = "E"
ARCH = "efficientnet_b0"        # FIXED (DECISION-039). Resolution is the only variable.
IMAGE_SIZE = 384
EPOCHS = 30
SEED = 42

# The 224 baseline, THREE SEEDS, from committed artefacts. The stopping rule is stated
# against this RANGE, not against its best member (DECISION-042).
BASE224_SENS = [0.7464, 0.7211, 0.7405]     # mean 0.7360
BASE224_QWK_HELD_OUT = [0.7578, 0.7548, 0.7564]
CLEAR_BAR = 0.78                # must exceed max(BASE224_SENS) meaningfully
SENS_FLOOR = 0.80               # DECISION-032, PROVISIONAL pending supervisor
B0_224_SECONDS_PER_EPOCH = 76.0

CODE = None
for cand in (Path("/kaggle/input/datasets/rah098/fyp-dr-code"),
             Path("/kaggle/input/fyp-dr-code")):
    if (cand / "src").is_dir():
        CODE = cand
        break
if CODE is None:
    hits = list(Path("/kaggle/input").rglob("src/data/preprocess.py"))
    if not hits:
        raise SystemExit("fyp-dr-code not found. + Add Input -> Your Datasets.")
    CODE = hits[0].parents[2]

if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-384", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

import cv2, numpy as np, timm, torch
if not torch.cuda.is_available():
    raise SystemExit("no GPU. Session sidebar -> Accelerator -> GPU T4 x2.")
print(f"cuda: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")

# Confirm this really is the 384 cache. A 224 cache would train fine and produce a
# perfectly plausible number for the wrong experiment.
sample = [p for p in Path(CACHE).rglob("*.jpeg")][:20]
shapes = {cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR).shape
          for p in sample}
print("cache image shapes:", shapes)
assert shapes == {(IMAGE_SIZE, IMAGE_SIZE, 3)}, (
    f"this is NOT a {IMAGE_SIZE}px cache: {shapes}. Wrong input attached.")

prov = Path(CACHE) / "_cache_provenance.json"
if prov.exists():
    got = json.loads(prov.read_text())["runs"][-1]["preprocess"]["image_size"]
    assert got == IMAGE_SIZE, f"provenance says image_size={got}"
    print("provenance confirms image_size:", got)


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
assert run([sys.executable, "-m", "src.train.smoke", "--arm", ARM, "--arch", ARCH])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0

print(f"""
plan: arm {ARM} on {ARCH} at {IMAGE_SIZE}px, {EPOCHS} epochs, seed {SEED}
      ~{2.94 * B0_224_SECONDS_PER_EPOCH:.0f} s/epoch expected -> ~{2.94 * B0_224_SECONDS_PER_EPOCH * EPOCHS / 3600:.1f} h

224 baseline, three seeds: sens {BASE224_SENS} (mean {sum(BASE224_SENS)/3:.4f})
The bar to clear is {CLEAR_BAR}, above the TOP of that range - not above its mean.
""")
'''

CELL_6 = r'''
# -- Cell 6 -- SESSION B: arm E at 384 ---------------------------------------
# COMMIT THIS. ~2 h.
RUNS.mkdir(parents=True, exist_ok=True)
RUN_ID = f"phase4_res{IMAGE_SIZE}_arm_{ARM.lower()}_{ARCH}"

rc, mins = run([
    sys.executable, "-m", "src.train.train",
    "--arm", ARM,
    "--config", "configs/kaggle.yaml",
    "--cache-root", CACHE,
    "--run-id", RUN_ID,
    "--runs-root", RUNS,
    "--arch", ARCH,
    "--image-size", str(IMAGE_SIZE),
    "--epochs", str(EPOCHS),
    "--seed", str(SEED),
    "--num-workers", "2",
    "--log-every", "200",
], timeout_min=200)

result = {"arm": ARM, "arch": ARCH, "image_size": IMAGE_SIZE, "run_id": RUN_ID,
          "exit_code": rc, "minutes": round(mins, 1), "ok": rc == 0}
if rc == 0:
    m = json.loads((RUNS / RUN_ID / "metrics.json").read_text())
    result.update({"qwk": m["qwk"], "epochs_run": m["epochs_run"],
                   "best_epoch": m["best_epoch"]})
    print(f"\n  arm {ARM} at {IMAGE_SIZE}px: qwk {m['qwk']:.4f} (as run)  "
          f"({m['epochs_run']} epochs, best {m['best_epoch']}, {mins:.1f} min)")
    print("  AS-RUN QWK IS NOT THE COMPARISON (DECISION-035). Cell 7 is.")
else:
    print(f"\n  *** FAILED (exit {rc}) ***")
(RUNS / "_res384_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
'''

CELL_7 = r'''
# -- Cell 7 -- SESSION B: the stopping rule, and the gate --------------------
# COMMIT THIS. Seconds.
import numpy as np

if not result.get("ok"):
    raise SystemExit("the run failed; no verdict")

from src.eval.compare_arms import load_run
from src.eval.thresholds import (choose_operating_point, referable_scores,
                                 sensitivity_specificity_curve)

r = load_run(RUNS / RUN_ID)
curve = sensitivity_specificity_curve(
    referable_scores(r["outputs"], "ordinal_regression"), r["y"])
op = choose_operating_point(curve)
sens = (op.get("max_sens_at_spec") or {}).get("sensitivity", float("nan"))

lo, hi = min(BASE224_SENS), max(BASE224_SENS)
print(f"384 sens@spec>=0.95 : {sens:.4f}")
print(f"224 three-seed range: {lo:.4f} - {hi:.4f}  (mean {sum(BASE224_SENS)/3:.4f})")
print(f"bar to clear        : {CLEAR_BAR}")
print(f"screening floor     : {SENS_FLOOR}  [PROVISIONAL - DECISION-032]")

predicted = 0.76 <= sens < SENS_FLOOR
print(f"\nPREDICTION was 0.76-0.80, not reaching the floor "
      f"-> {'AS PREDICTED' if predicted else 'PREDICTION WRONG'}")

if sens >= CLEAR_BAR:
    verdict = "CLEARS THE BAR"
    what = ("Resolution is a real lever. This is a RESULT, not a decision.")
elif sens >= hi:
    verdict = "SUGGESTIVE, NOT ESTABLISHED"
    what = ("One seed above a three-seed range. Report as the ablation. Do not escalate.")
else:
    verdict = "CLEAN NEGATIVE"
    what = ("Inside or below the 224 range. Resolution was not the binding constraint "
            "at this data scale - a genuine finding, and the write-up says so plainly.")

print(f"\n  VERDICT: {verdict}\n  {what}")

print(f"""
================================================================================
 GATE - nothing below happens without WRITTEN supervisor approval
================================================================================
The proposal fixes 224 for all headline results (CLAUDE.md section 5). Whatever
this number is, until a decision in docs/DECISIONS.md records written supervisor
approval of the deviation, WITH THE DATE:

  * the headline model stays arm E on EfficientNet-B0 at 224
  * NO seeds are run at 384
  * docs/EXPERIMENTS.md's headline is NOT changed
  * the EyePACS test set is NOT opened on a 384 model
  * Phase 6's APTOS numbers are NOT re-run at 384

A good result is not permission. Take the number to the supervisor; come back with
the approval; log it; only then re-plan.

Phase 4 is COMPLETE either way. Next is Phase 7 (web app) and Phase 8 (write-up).
================================================================================

Fetch locally:
  python -m src.data.fetch_run --kernel rah098/<slug> --run-id {RUN_ID}
  python -m src.eval.compare_arms --pattern 'phase4_*' --operating-point
  python -m notebooks.gen_experiments
""")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7]

if __name__ == "__main__":
    for w in (sys.argv[1:] or [str(i) for i in range(1, 8)]):
        print(CELLS[int(w) - 1])

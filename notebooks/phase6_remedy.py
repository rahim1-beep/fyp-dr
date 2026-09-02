"""Phase 6b — the surround-randomisation remedy. Four cells, ~2.5 h (one training cell).

    python -m notebooks.phase6_remedy 1     # or 2, 3, 4

Arm E on EfficientNet-B0, three seeds, retrained with ONE factor changed: the masked-out
surround is replaced by a per-image random constant colour on 75% of training draws
(`configs/remedy_surround.yaml`). Nothing else moves — same head, loss, schedule, epochs,
seeds, splits, and cache. No cache rebuild: the augmentation is applied at load time, and
the extra mask costs 1.19 ms/image, ~29 s per epoch on one worker against a ~76 s epoch.

=============================================================================
 WHAT IS BEING MANIPULATED, AND WHY THIS ONE THING
=============================================================================

G4 fired in Phase 6: within-grade correlation between predicted score and retina
coverage was larger on APTOS (mean max |r| 0.2360) than on EyePACS validation (0.1367).
The pre-registered reading is a FIELD-OF-VIEW SHORTCUT — the model partly reads how much
frame the retina fills, which is a property of the camera and the site, not of the eye.

Two candidate manipulations were measured locally before choosing (DECISION-063):

  * EXTENT — randomise how much surround there is. REJECTED as the primary factor
    because training ALREADY does it: the existing affine (scale 0.9-1.1, shift, rotate)
    moves synthetic-disc coverage over 0.611-0.864, sd 0.075 — a WIDER range than the
    whole Phase 5 erosion sweep (0.694 -> 0.619). Adding more of what is already there is
    a weak manipulation, and a null result from it would mean nothing.

  * APPEARANCE — randomise what the surround LOOKS like. CHOSEN. Measured on the same
    fixture, the surround stays essentially black in 96% of training draws (mean surround
    value 0.011 after brightness/contrast jitter). This cue is genuinely absent from
    training, and it is what DECISION-051 pre-registered.

The distinction matters, and the first measurement of it was WRONG in a way worth
recording: an earlier probe indexed the surround with the UNROTATED disc mask, so
rotated-in retina counted as "lifted" surround and appeared to show the surround was
already randomised in 89% of draws. It is not. The corrected probe warps an indicator
channel through the same geometry, which is also how the augmentation itself works.

WHAT THIS DOES NOT DO. It does not remove the boundary. A random constant still marks
exactly where the retina ends, and DECISION-051 says so outright. What it removes is the
surround's appearance as a STABLE, noise-free cue — "fraction of the frame that is black"
is a feature a global pool computes for free, and after this it is not.

=============================================================================
 PRE-REGISTERED. WRITTEN BEFORE THE RUN. DECISION-063.
=============================================================================

PRIMARY ENDPOINT — G4 recomputed on the remedied model, same rule, same aggregation:
the mean across seeds of the max within-grade |r|, on APTOS and on EyePACS validation.

  REMEDY WORKS      APTOS mean < 0.20 AND APTOS mean <= EyePACS mean (i.e. G4 no longer
                    fires), AND the per-seed direction weakens: APTOS > EyePACS in at
                    most 1 of 3 seeds, down from 3 of 3.

  REMEDY FAILS,     APTOS mean still >= 0.20 and still > EyePACS mean. Removing the
  AND THE SHORTCUT  surround's appearance did not touch the correlation, so the
  READING WEAKENS   correlation is most likely a CONFOUND — coverage tracking
                    acquisition quality, which tracks severity — rather than a learned
                    dependence on framing. Phase 6's headline stands as a measured
                    generalisation defect, but the word "shortcut" comes out of it.

  TRADE-OFF         APTOS |r| falls AND external QWK or sens@spec falls with it. The
                    framing signal carried real information. Report both, claim neither.

  AMBIGUOUS         |r| moves but stays on the wrong side of a threshold. See POWER.

SECONDARY (DECISION-051's own paired test): the remedied model should LOSE LESS on APTOS
than the baseline does, even if it is equal or slightly worse on EyePACS. Worse
in-domain, better out-of-domain is the signature of removing a shortcut.

IN-DOMAIN COST, and the noise scale it is read against: baseline arm E validation QWK is
0.7615 / 0.7534 / 0.7560 across seeds 42/43/44 — mean 0.7570, range 0.0081. A remedied
mean inside +/- 0.0081 of 0.7570 is NOT a detectable cost. Below 0.7489 it is a real one.

=============================================================================
 POWER. THE HONEST PART, AND IT IS NOT REASSURING.
=============================================================================

The effect being remedied is seed-dependent: APTOS max |r| was 0.3285 / 0.1345 / 0.2451,
a range of 0.194. A three-seed test of an effect whose seed range is 0.194 CANNOT
distinguish "the remedy removed a modest dependence" from "the seeds landed differently".

    A drop from 0.2360 to, say, 0.15 is INSIDE the baseline's own seed range and
    proves nothing on its own.

So the decisive readings are the ones that do NOT rest on the magnitude:
  * the DIRECTION count (APTOS > EyePACS in 3 of 3 baseline seeds -> at most 1 of 3)
  * whether the criterion fires under ALL FOUR aggregation rules, not only the mean

Both are reported. If the result lands in the ambiguous band it will be REPORTED as
ambiguous, and the pre-registered conclusion is that three seeds were not enough to
settle it — not a re-analysis until something crosses a line.

THE AGGREGATION RULE IS FIXED HERE, IN ADVANCE, THIS TIME: the seed MEAN, per
DECISION-042. All four rules are printed alongside, as in Phase 6. That the rule was
settled post hoc for the baseline is recorded in DECISION-059 and is not repeated.

NEEDS: fyp-dr-eyepacs-224, fyp-dr-code, AND the phase6_aptos output (for the baseline
APTOS scores and the baseline G4 numbers). Cell 1 refuses to run without them rather than
falling back to hardcoded constants.

SESSION SETTINGS
    Accelerator : GPU T4 x2      Internet : ON

HOW TO RUN
    Cell 1     by hand   (~4 min)
    Cells 2-4  by COMMIT (cell 2 is ~2.1 h)
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, pre-flight, and the pre-registration ------------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
RUNS = WORK / "runs"
OUT = WORK / "phase6_remedy"

ARCH = "efficientnet_b0"
ARM = "E"
SEEDS = [42, 43, 44]
EPOCHS = 30
OVERLAY = "configs/remedy_surround.yaml"
RUN_ID = {s: f"phase6_remedy_arm_e_{ARCH}_s{s}" for s in SEEDS}

# MEASURED baselines, arm E, EfficientNet-B0, committed artefacts (R4).
BASE_VAL_QWK = {42: 0.7615, 43: 0.7534, 44: 0.7560}      # mean 0.7570, range 0.0081

# CODE resolution goes through resolve_input, NOT a hand-rolled candidate list with an
# rglob fallback over all of /kaggle/input. Attached NOTEBOOK OUTPUTS carry their own copy
# of the repo under /kaggle/working/fyp-dr, so a first-hit rglob can silently select a
# STALE repo. resolve_input knows the doubled-directory mount layout and refuses to guess.
#
# Bootstrapping is the awkward part: resolve_input lives in the code being located. So
# find kaggle_paths.py directly, EXCLUDING /kaggle/input/notebooks for exactly the reason
# above, hand over to resolve_input, then UNDO the bootstrap import so every later
# `import src.*` comes from the copy under REPO rather than from the read-only mount.
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

# THE SPLIT CSVs DISAPPEAR MID-RUN. MEASURED, NOT GUESSED — AND SURVIVED.
#
# 2026-08-29, Version #2: the leakage gate in THIS cell reported "36 passed, 5 skipped".
# Reproduced locally with the splits INTACT and only the gitignored trainLabels.csv
# hidden: 36 passed, 5 skipped, the same five source-label skips. So every split test RAN
# and PASSED here — data/splits was complete at cell 1. Cell 4 then died on
# `data/splits/val.csv does not exist`. Present at cell 1, absent at cell 4, with nothing
# in between that writes to that directory.
#
# The cause is NOT established. Rather than guess at it, this does three things:
#   1. records the exact state (names + sizes + hashes) at cell 1,
#   2. keeps a copy OUTSIDE the repo tree, in WORK, and
#   3. lets cell 4 re-check, and restore from that copy if files have vanished — loudly,
#      verifying the hashes so a restored file is provably the same bytes the gate saw.
# A run should not lose two hours to this while the cause is still unknown, and a silent
# restore would be worse than the bug, so it prints everything it does.
import hashlib

FYP_SPLITS = ["train", "val", "test", "aptos_train", "aptos_val", "aptos_test"]
SPLIT_BACKUP = WORK / "splits_backup"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def snapshot_splits():
    """Copy the six split CSVs somewhere outside REPO, and record their hashes."""
    d = REPO / "data" / "splits"
    absent = [n for n in FYP_SPLITS if not (d / f"{n}.csv").exists()]
    if absent:
        have = sorted(q.name for q in d.glob("*.csv")) if d.is_dir() else []
        raise SystemExit(
            f"[splits @ cell 1] MISSING: {absent}\n"
            f"  present    : {have}\n"
            f"  copied from: {CODE}\n"
            "The upload or the mount is incomplete. Re-upload fyp-dr-code as a New "
            "Version and confirm the notebook's input is pinned to it.")
    SPLIT_BACKUP.mkdir(parents=True, exist_ok=True)
    sums = {}
    print("[splits @ cell 1] all 6 present; snapshotting to", SPLIT_BACKUP)
    for n in FYP_SPLITS:
        src = d / f"{n}.csv"
        shutil.copy2(src, SPLIT_BACKUP / f"{n}.csv")
        sums[n] = _sha(src)
        print(f"    {n:<14} {src.stat().st_size:>9,} B  sha {sums[n]}")
    return sums


def check_splits(where):
    """Re-check, and restore from the cell-1 snapshot if anything has vanished."""
    d = REPO / "data" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    gone = [n for n in FYP_SPLITS if not (d / f"{n}.csv").exists()]
    if not gone:
        print(f"[splits @ {where}] all 6 still present")
        return
    print(f"[splits @ {where}] !! {len(gone)} split CSV(s) VANISHED since cell 1: {gone}")
    print(f"    still present: {sorted(q.name for q in d.glob('*.csv'))}")
    # WHICH files vanish is the diagnostic. Version #3 lost ALL SIX at once, i.e. the
    # whole directory, not one file. REPO is a copy of the input dataset, so if the code
    # files are gone too then something is pruning the copy wholesale (already-imported
    # modules keep working from memory, which is why cell 4 got as far as it did). If
    # ONLY data/splits is gone, it is specific to that directory. One cheap print settles
    # it, so the next run does not have to guess either.
    probe = {
        "src/data/manifest.py": (REPO / "src/data/manifest.py").exists(),
        "src/train/train.py": (REPO / "src/train/train.py").exists(),
        "configs/base.yaml": (REPO / "configs/base.yaml").exists(),
        "configs/arm_e.yaml": (REPO / "configs/arm_e.yaml").exists(),
        "notebooks/phase6_aptos.py": (REPO / "notebooks/phase6_aptos.py").exists(),
    }
    print("    repo integrity probe (is the whole copy being pruned, or just splits?):")
    for k, ok in probe.items():
        print(f"      {'present' if ok else 'GONE   '}  {k}")
    print(f"      REPO itself exists: {REPO.exists()}   "
          f"top-level entries: {sorted(q.name for q in REPO.iterdir())[:12] if REPO.exists() else '-'}")
    print("    Restoring the splits from the cell-1 snapshot and verifying hashes.")
    for n in gone:
        shutil.copy2(SPLIT_BACKUP / f"{n}.csv", d / f"{n}.csv")
        got = _sha(d / f"{n}.csv")
        if got != SPLIT_SUMS[n]:
            raise SystemExit(
                f"restored {n}.csv but its hash {got} != the cell-1 hash "
                f"{SPLIT_SUMS[n]}. Refusing to continue against a partition that is not "
                "the one the leakage gate checked.")
        print(f"    restored {n}.csv  sha {got}  (matches cell 1)")
    print("    REPORT THIS in the run notes: the restore is a workaround, not a fix.")


SPLIT_SUMS = snapshot_splits()

# Real run: the leakage gate must not fail open by skipping absent splits.
os.environ["FYP_REQUIRE_SPLITS"] = "1"
OUT.mkdir(parents=True, exist_ok=True)

from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

# The Phase 6 baseline artefacts. REQUIRED — the paired comparison is the secondary
# endpoint, and the baseline G4 numbers are what this run is measured against. Reading
# them from the attached artefact rather than retyping them is the R4 rule.
BASE = None
for cand in Path("/kaggle/input").rglob("phase6_aptos/aptos_scores.npz"):
    BASE = cand.parent
    break
if BASE is None:
    raise SystemExit(
        "phase6_aptos output not attached. + Add Input -> Your Work -> the Phase 6 "
        "notebook. Without it there is no baseline to compare against and the paired "
        "endpoint cannot be computed.")
print("phase6 baseline:", BASE)

import numpy as np

base_cov_a = json.loads((BASE / "coverage_correlation.json").read_text(encoding="utf-8"))
_p = BASE / "coverage_correlation_eyepacs.json"
if not _p.exists():
    raise SystemExit(
        "coverage_correlation_eyepacs.json is missing from the Phase 6 output — that is "
        "the EyePACS half of G4. Re-run Phase 6 cell 4 and commit its artefacts first "
        "(R4); this notebook will not retype numbers from a console log.")
base_cov_e = json.loads(_p.read_text(encoding="utf-8"))
BASE_G4 = {s: {"eyepacs": base_cov_e[str(s)]["max_abs_r"],
               "aptos": base_cov_a[str(s)]["max_abs_r"]} for s in SEEDS}
print("\nBASELINE G4, from the committed artefacts:")
for s in SEEDS:
    b = BASE_G4[s]
    print(f"  seed {s}: EyePACS {b['eyepacs']:.4f}  APTOS {b['aptos']:.4f}  "
          f"APTOS larger: {b['aptos'] > b['eyepacs']}")
BASE_MEAN_E = float(np.mean([BASE_G4[s]["eyepacs"] for s in SEEDS]))
BASE_MEAN_A = float(np.mean([BASE_G4[s]["aptos"] for s in SEEDS]))
BASE_DIR = sum(BASE_G4[s]["aptos"] > BASE_G4[s]["eyepacs"] for s in SEEDS)
print(f"  seed means: EyePACS {BASE_MEAN_E:.4f}  APTOS {BASE_MEAN_A:.4f}  "
      f"| direction {BASE_DIR}/{len(SEEDS)}")

import cv2, pandas as pd, timm, torch, torchvision
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\npython {sys.version.split()[0]}  |  torch {torch.__version__}  "
      f"|  timm {timm.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("no GPU. Session sidebar -> Accelerator -> GPU T4 x2.")
print(f"cuda: {torch.cuda.get_device_name(0)}")


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


# THE MANIPULATION IS EXACTLY ONE KNOB. Verified here rather than assumed: the overlay
# must change surround_randomisation and NOTHING else, and arm E's own config must still
# win on everything it sets (merge order is base <- overlay <- arm file).
from src.data.dataset import AugmentConfig
from src.train.train import load_yaml, merge

_base = merge(load_yaml(REPO / "configs/base.yaml"),
              load_yaml(REPO / "configs/arm_e.yaml"))
_rem = merge(load_yaml(REPO / "configs/base.yaml"), load_yaml(REPO / OVERLAY),
             load_yaml(REPO / "configs/arm_e.yaml"))
a0, a1 = AugmentConfig.from_yaml(_base), AugmentConfig.from_yaml(_rem)
diff = {k: (getattr(a0, k), getattr(a1, k)) for k in a0.__dataclass_fields__
        if getattr(a0, k) != getattr(a1, k)}
assert diff == {"surround_randomisation": (0.0, 0.75)}, f"more than one knob moved: {diff}"
assert _rem["model"] == _base["model"] and _rem["imbalance"] == _base["imbalance"], \
    "the overlay changed the arm, not just the augmentation"
print(f"\nmanipulation: {diff}  (everything else identical)")

assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_dataset.py", "-q"])[0] == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])[0] == 0
assert run([sys.executable, "-m", "src.train.smoke", "--arm", ARM, "--arch", ARCH])[0] == 0

print(f"""
PRE-REGISTERED (DECISION-063). Cell 4 APPLIES this; it does not interpret it.

  PRIMARY   G4 recomputed, seed MEAN of max within-grade |r| (DECISION-042, fixed
            HERE in advance this time):
              WORKS  APTOS mean < 0.20 AND <= EyePACS mean AND direction <= 1/3
              FAILS  APTOS mean >= 0.20 AND > EyePACS mean -> the correlation is most
                     likely a CONFOUND, and "shortcut" comes out of the write-up
  SECONDARY the remedied model loses LESS on APTOS than the baseline does
  COST      baseline val QWK mean 0.7570, seed range 0.0081. Inside +/- 0.0081 is not
            a detectable cost; below 0.7489 it is.

  POWER     baseline APTOS |r| ranged 0.194 ACROSS SEEDS. A drop to ~0.15 is inside
            that range and proves nothing. The direction count and the four-rule
            agreement carry the result; the magnitude does not. An ambiguous landing
            is REPORTED as ambiguous.

  baseline: EyePACS {BASE_MEAN_E:.4f}  APTOS {BASE_MEAN_A:.4f}  direction {BASE_DIR}/3
""")
'''

CELL_2 = r'''
# -- Cell 2 -- retrain arm E with surround randomisation, three seeds --------
# COMMIT THIS. ~2.1 h.
#
# Per-run isolation: a failure is recorded and the sweep continues, so one bad seed does
# not cost the whole session.
RUNS.mkdir(parents=True, exist_ok=True)
results = {}
for seed in SEEDS:
    # Before EVERY seed, not just once. Cell 2 is ~2.1 h across three subprocesses, and
    # each `src.train.train` re-reads the split CSVs at startup. Version #3 showed the
    # whole splits directory can empty mid-run, so a disappearance between seed 42 and
    # seed 43 would kill the remaining seeds two hours in. Checked and restored here for
    # a few milliseconds per seed.
    check_splits(f"cell 2, before seed {seed}")
    rc, mins = run([sys.executable, "-m", "src.train.train",
                    "--arm", ARM, "--arch", ARCH,
                    "--config", OVERLAY,
                    "--cache-root", CACHE, "--runs-root", RUNS,
                    "--run-id", RUN_ID[seed], "--epochs", EPOCHS, "--seed", seed,
                    "--num-workers", 2], timeout_min=60)
    results[seed] = {"rc": rc, "minutes": round(mins, 1)}
    if rc != 0:
        print(f"!! seed {seed} FAILED (exit {rc}) — continuing")

print("\n" + "=" * 70)
for seed, r in results.items():
    print(f"  seed {seed}: exit {r['rc']}  {r['minutes']} min")

ok = [s for s in SEEDS if results[s]["rc"] == 0]
if len(ok) < 3:
    print(f"\n!! only {len(ok)}/3 seeds trained. The remedy is a three-seed comparison "
          f"against a three-seed baseline; a partial result is reported as partial and "
          f"the pre-registered rule is NOT applied to it.")

# The run's own config must record what it did (R6). Check, do not trust.
import yaml as _yaml
for seed in ok:
    c = _yaml.safe_load((RUNS / RUN_ID[seed] / "config.yaml").read_text(encoding="utf-8"))
    got = c.get("augment", {}).get("surround_randomisation")
    assert got == 0.75, f"seed {seed} recorded surround_randomisation={got}, not 0.75"
print("\nevery run config records surround_randomisation=0.75")
'''

CELL_3 = r'''
# -- Cell 3 -- the in-domain cost, against the baseline's own seed range -----
# COMMIT THIS. Seconds — reads each run's metrics.json.
rows = []
for seed in ok:
    m = json.loads((RUNS / RUN_ID[seed] / "metrics.json").read_text(encoding="utf-8"))
    r = m["referable"]          # at the run's own cut point, NOT sens@spec>=0.95
    rows.append({"seed": seed, "qwk": m["qwk"],
                 "balanced_accuracy": m["balanced_accuracy"],
                 "sensitivity": r["sensitivity"], "specificity": r["specificity"]})

# `sens` here is referable sensitivity AT THE RUN'S OWN CUT POINT (baseline seed 42:
# 0.6676 at specificity 0.9717) — not the 0.7360 sens@spec>=0.95 headline, which comes
# from the threshold sweep in src/eval/thresholds.py. Comparing like with like matters
# more than quoting the more familiar number.
print(f"{'seed':>5}{'QWK':>10}{'baseline':>10}{'delta':>9}{'bal acc':>10}{'sens':>9}")
for r in rows:
    b = BASE_VAL_QWK[r["seed"]]
    print(f"{r['seed']:>5}{r['qwk']:>10.4f}{b:>10.4f}{r['qwk'] - b:>+9.4f}"
          f"{r['balanced_accuracy']:>10.4f}{r['sensitivity']:>9.4f}")

mean_q = float(np.mean([r["qwk"] for r in rows]))
BASE_MEAN_Q, BASE_RANGE_Q = 0.7570, 0.0081
print(f"\n  remedied mean QWK {mean_q:.4f}   baseline mean {BASE_MEAN_Q:.4f}   "
      f"delta {mean_q - BASE_MEAN_Q:+.4f}")
if abs(mean_q - BASE_MEAN_Q) <= BASE_RANGE_Q:
    print(f"  -> INSIDE the baseline's own seed range ({BASE_RANGE_Q:.4f}). No "
          f"detectable in-domain cost.")
elif mean_q < BASE_MEAN_Q:
    print("  -> a REAL in-domain cost, beyond seed noise. Pre-registered as the price "
          "of the remedy, not as a failure of it — the endpoint is external.")
else:
    print("  -> better in-domain than the baseline, beyond seed noise. Unexpected; "
          "report as measured and do not explain it after the fact.")

json.dump({"in_domain": rows, "mean_qwk": mean_q, "baseline_mean_qwk": BASE_MEAN_Q},
          open(OUT / "in_domain.json", "w"), indent=1)
'''

CELL_4 = r'''
# -- Cell 4 -- G4 recomputed on the remedied model, both sides, and the rule --
# COMMIT THIS. ~15 min.
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.data.manifest import load_split
from src.eval.external import coverage_correlation
from src.models.factory import ModelConfig, build_model, load_checkpoint, normalisation
from src.xai.border_check import NoRetinaError, region_masks
from src.xai.gradcam import scalar_target

APTOS = pd.concat([load_split(n) for n in
                   ("aptos_train", "aptos_val", "aptos_test")], ignore_index=True)
assert len(APTOS) == 3662, f"expected 3,662 APTOS images, got {len(APTOS)}"
check_splits("cell 4")
VAL = load_split("val")

# Coverage is a property of the IMAGE, not of the model, so the baseline's own APTOS
# coverage array is reused verbatim — recomputing it could only introduce a difference
# that has nothing to do with the remedy.
base_npz = np.load(BASE / "aptos_scores.npz")
cov_aptos = base_npz["coverage"]
print(f"APTOS coverage reused from the baseline artefact: median "
      f"{np.nanmedian(cov_aptos):.4f}")


def coverage_of(paths):
    out, undef = [], 0
    for path in paths:
        bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        try:
            out.append(float((~region_masks(bgr)["outside"]).mean()))
        except NoRetinaError:
            out.append(float("nan"))
            undef += 1
    return np.asarray(out), undef


_vds = DRDataset(VAL, CACHE, train=False, image_size=224,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
cov_val, n_undef = coverage_of(_vds.cache_paths)
print(f"EyePACS validation coverage: median {np.nanmedian(cov_val):.4f}  "
      f"({n_undef} no-retina, excluded and counted — DECISION-052)")


def score(model, df, mean, std):
    ds = DRDataset(df, CACHE, train=False, image_size=224, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    sc, ys = [], []
    with torch.no_grad():
        for x, y, idx in loader:
            sc.extend(scalar_target(model(x.to(DEV))).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    return np.asarray(sc), np.asarray(ys)


rem_e, rem_a, rem_scores = {}, {}, {}
for seed in ok:
    d = RUNS / RUN_ID[seed]
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(ModelConfig(
        arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
        head=cfg["model"]["head"], pretrained=False))
    load_checkpoint(d / "best.pth", model)
    mean, std = normalisation(model)
    model.eval().to(DEV)

    sa, ya = score(model, APTOS, mean, std)
    sv, yv = score(model, VAL, mean, std)
    rem_scores[seed] = sa
    rem_a[seed] = coverage_correlation(sa, ya, cov_aptos)
    rem_e[seed] = coverage_correlation(sv, yv, cov_val)
    print(f"  seed {seed}: EyePACS max|r| {rem_e[seed]['max_abs_r']:.4f}   "
          f"APTOS max|r| {rem_a[seed]['max_abs_r']:.4f}")

# ---- the pre-registered rule, applied ------------------------------------------
E = [rem_e[s]["max_abs_r"] for s in ok]
A = [rem_a[s]["max_abs_r"] for s in ok]
mean_e, mean_a = float(np.mean(E)), float(np.mean(A))
med_e, med_a = float(np.median(E)), float(np.median(A))
direction = sum(a > e for a, e in zip(A, E))

print("\nG4 on the REMEDIED model:")
print(f"{'seed':>5}{'EyePACS':>10}{'APTOS':>10}{'APTOS larger?':>16}")
for s, e, a in zip(ok, E, A):
    print(f"{s:>5}{e:>10.4f}{a:>10.4f}{str(a >= 0.2 and a > e):>16}")

fires_per_seed = sum(a >= 0.2 and a > e for a, e in zip(A, E))
rules = {
    "mean":      (mean_a, mean_e, mean_a >= 0.2 and mean_a > mean_e),
    "median":    (med_a, med_e, med_a >= 0.2 and med_a > med_e),
    "majority":  (mean_a, mean_e, fires_per_seed * 2 >= len(ok)),
    "unanimity": (mean_a, mean_e, fires_per_seed == len(ok)),
}
print("\n  aggregation rule    APTOS   EyePACS   G4 fires?")
for name, (a, e, f) in rules.items():
    print(f"  {name:<18}{a:>7.4f}{e:>10.4f}   {f}")

print(f"\n  baseline  APTOS {BASE_MEAN_A:.4f}  EyePACS {BASE_MEAN_E:.4f}  "
      f"direction {BASE_DIR}/{len(SEEDS)}")
print(f"  remedied  APTOS {mean_a:.4f}  EyePACS {mean_e:.4f}  "
      f"direction {direction}/{len(ok)}")

works = mean_a < 0.20 and mean_a <= mean_e and direction <= 1
fails = mean_a >= 0.20 and mean_a > mean_e
if works:
    outcome = ("REMEDY WORKS — G4 no longer fires and the per-seed direction collapsed. "
               "The shortcut reading is supported causally, not only correlationally.")
elif fails:
    outcome = ("REMEDY FAILS — removing the surround's appearance did not move the "
               "correlation. The SHORTCUT reading weakens: this looks like a confound "
               "(coverage tracking acquisition quality, which tracks severity). Phase "
               "6's generalisation finding stands; the word 'shortcut' comes out of it.")
else:
    outcome = ("AMBIGUOUS — the criterion moved without settling. Pre-registered "
               "conclusion: three seeds were not enough, given the baseline's own "
               "0.194 seed range. Report as ambiguous; do not re-analyse until it "
               "crosses a line.")
print("\n  OUTCOME: " + outcome)

# ---- the secondary, paired endpoint ---------------------------------------------
print("\nSecondary (DECISION-051): does the remedied model LOSE LESS on APTOS?")
print("  Mean predicted score, baseline vs remedied. The full carried-over evaluation "
      "belongs to Phase 6 and is re-run there if the primary endpoint warrants it.")
for s in ok:
    print(f"  seed {s}: baseline {base_npz[f'seed_{s}'].mean():+.4f}   "
          f"remedied {rem_scores[s].mean():+.4f}")

json.dump({"remedied": {"eyepacs": rem_e, "aptos": rem_a,
                        "mean_eyepacs": mean_e, "mean_aptos": mean_a,
                        "direction": f"{direction}/{len(ok)}"},
           "baseline": {"mean_eyepacs": BASE_MEAN_E, "mean_aptos": BASE_MEAN_A,
                        "direction": f"{BASE_DIR}/{len(SEEDS)}"},
           "rules": {k: {"aptos": v[0], "eyepacs": v[1], "fires": bool(v[2])}
                     for k, v in rules.items()},
           "outcome": outcome, "seeds": ok},
          open(OUT / "remedy_verdict.json", "w"), indent=1, default=float)
np.savez(OUT / "remedy_aptos_scores.npz", y_true=base_npz["y_true"],
         **{f"seed_{s}": rem_scores[s] for s in ok})
print(f"\nwrote {OUT / 'remedy_verdict.json'}")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4"]):
        print(CELLS[int(w) - 1])

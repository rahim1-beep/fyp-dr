"""Phase 6 — external validation on APTOS. Four cells, ~18 min. Inference only.

    python -m notebooks.phase6_aptos 1     # or 2, 3, 4

Arm E on EfficientNet-B0, **all three seeds**, over **all 3,662 APTOS images** pooled
(DECISION-004: the author-provided split is ignored). Arm F is excluded — APTOS is its
training data and cannot also be its external validation set (DECISION-007).

**The EyePACS test set is NOT opened.** It opens once, after the 384 decision is final
(DECISION-045).

=============================================================================
 THE ONE RULE THAT MAKES THIS EXTERNAL VALIDATION RATHER THAN A SECOND FIT
=============================================================================

    Cut points and the operating-point threshold are CARRIED OVER from EyePACS
    validation, UNCHANGED. Re-fitting either on APTOS is fitting on the external set.

`evaluate_external` takes them as REQUIRED arguments and raises if they are absent —
there is no "fit if missing" branch, because that is the single easiest way to inflate
this number. A re-fitted figure is also reported, always labelled secondary, because the
gap between the two separates CALIBRATION shift from DISCRIMINATION shift.

THE VERDICT IS PRE-REGISTERED (DECISION-057) and cell 3 APPLIES it rather than
interpreting it. Any one of these means the model does not generalise:

    G1  re-fitted APTOS QWK < 0.60          (below the Phase 3 baseline's 0.6138)
    G2  re-fitted sens@spec>=0.95 < 0.60
    G3  grade 3-4 recall < 0.30 while grade 0 recall > 0.90   (the dangerous SHAPE:
                                                               reliable where it
                                                               matters least)
    G4  coverage correlation |r| >= 0.2 within grades AND larger on APTOS than EyePACS

"Degraded but usable" requires re-fitted QWK >= 0.65, re-fitting recovering >= half the
drop, and none of G1-G4. Between 0.60 and 0.65 the verdict is "generalises weakly; not
usable without further work" — a band named in advance precisely because it is the one a
motivated reader rounds towards the better neighbour.

WHAT "DEPLOYABLE" MEANS HERE. It is not a claim this thesis can make either way: the
model already fails the screening floor IN-DOMAIN (0.7360 against 0.80). Phase 6 answers
the narrower question of whether it generalises beyond its training population.

WHAT TO EXPECT. A drop, and the drop is the finding. **Balanced accuracy may RISE while
QWK falls** — APTOS is 49.3% grade 0 against EyePACS's 73.5% — and that is not evidence
of generalisation.

NEEDS THE CHECKPOINTS. `best.pth` is gitignored and `fetch_run` does not pull it, so the
stage 3 AND stage 2 notebook outputs must both be attached (seed 42 is in stage 3; seeds
43 and 44 are in stage 2).

SESSION SETTINGS
    Accelerator : GPU T4 x2 (CPU works; still only minutes)
    Internet    : ON
    Inputs      : fyp-dr-eyepacs-224, fyp-dr-code, stage 3 output, stage 2 output

HOW TO RUN
    Cell 1     by hand (~3 min)
    Cells 2-4  by COMMIT

CELL 4 COMPLETES G4. The shortcut criterion is a comparison BETWEEN datasets, so the
APTOS half alone decides nothing; until cell 4 runs, `verdict()` reports G4 as UNDECIDED
and marks the verdict PROVISIONAL.
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, checkpoints, and the CARRIED-OVER decision rules --------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
OUT = WORK / "phase6_aptos"

ARCH = "efficientnet_b0"
ARM = "E"
SEEDS = [42, 43, 44]
# seed 42 lives in the stage 3 run; 43 and 44 in the stage 2 runs.
RUN_IDS = {42: f"phase4_stage3_arm_e_{ARCH}",
           43: f"phase4_stage2_arm_e_{ARCH}_s43",
           44: f"phase4_stage2_arm_e_{ARCH}_s44"}
EYEPACS_QWK_HELD_OUT = 0.7563        # 3-seed mean, matched decision rule

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

# ALL SIX SPLIT CSVs MUST BE HERE. On 2026-08-28 the three EyePACS splits were absent from
# the copied repo while the three APTOS ones were present; the leakage gate SKIPPED the
# tests needing them, reported "33 passed, 5 skipped", the notebook's `assert rc == 0` was
# satisfied, and the run continued for ten more minutes before dying in cell 4 on the same
# missing file. Checked here, at the copy, where the error names the actual cause.
FYP_SPLITS = ["train", "val", "test", "aptos_train", "aptos_val", "aptos_test"]
_absent = [n for n in FYP_SPLITS if not (REPO / "data" / "splits" / f"{n}.csv").exists()]
if _absent:
    _have = sorted(q.name for q in (REPO / "data" / "splits").glob("*.csv"))
    raise SystemExit(
        f"the copied repo is missing split CSV(s): {_absent}\n"
        f"  present    : {_have}\n"
        f"  copied from: {CODE}\n"
        "A PARTIAL partition is never legitimate. Re-upload fyp-dr-code as a New "
        "Version, confirm the notebook's input is pinned to that version, and re-run.")
print(f"splits: all {len(FYP_SPLITS)} present")

# Real run: the leakage gate must not fail open by skipping absent splits.
os.environ["FYP_REQUIRE_SPLITS"] = "1"
OUT.mkdir(parents=True, exist_ok=True)

from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

CKPTS = {}
for seed, rid in RUN_IDS.items():
    for cand in Path("/kaggle/input").rglob(f"{rid}/best.pth"):
        CKPTS[seed] = cand
        break
missing = [s for s in SEEDS if s not in CKPTS]
if missing:
    raise SystemExit(
        f"no checkpoint for seed(s) {missing}. Attach BOTH the stage 3 output (seed 42) "
        f"and the stage 2 output (seeds 43, 44): + Add Input -> Your Work.")
for s, c in sorted(CKPTS.items()):
    print(f"  seed {s}: {c}")

import cv2, numpy as np, pandas as pd, torch, yaml
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\ntorch {torch.__version__}  device {DEV}")

from src.data.manifest import load_split
from src.models.factory import (ModelConfig, build_model, load_checkpoint,
                                normalisation)

# ALL of APTOS, pooled. The author split is ignored (DECISION-004) and arm F is excluded
# (DECISION-007), so every APTOS image is unseen by these models.
APTOS = pd.concat([load_split(n) for n in
                   ("aptos_train", "aptos_val", "aptos_test")], ignore_index=True)
print(f"\nAPTOS pooled: {len(APTOS)} images, per grade "
      f"{APTOS.label.value_counts().sort_index().tolist()}")
assert len(APTOS) == 3662, f"expected 3,662 APTOS images, got {len(APTOS)}"
print(f"  grade-0 share {100 * (APTOS.label == 0).mean():.1f}%  "
      f"(EyePACS validation is 73.7%)")

# THE CARRIED-OVER DECISION RULES. Read from the EyePACS validation outputs of each seed,
# never re-derived on APTOS.
from src.eval.compare_arms import continuous_score, load_run
from src.eval.thresholds import (choose_operating_point, optimise_qwk_cuts,
                                 referable_scores, sensitivity_specificity_curve)

RULES = {}
for seed, rid in RUN_IDS.items():
    rd = None
    for cand in Path("/kaggle/input").rglob(f"{rid}/val_outputs.npz"):
        rd = cand.parent
        break
    if rd is None:
        raise SystemExit(f"no val_outputs.npz for {rid}; cannot carry over its cuts")
    r = load_run(rd)
    sc = continuous_score(r["outputs"])
    cuts, _ = optimise_qwk_cuts(sc, r["y"])
    op = choose_operating_point(sensitivity_specificity_curve(
        referable_scores(r["outputs"], "ordinal_regression"), r["y"]))
    thr = (op.get("max_sens_at_spec") or {}).get("threshold")
    if thr is None:
        raise SystemExit(f"seed {seed}: no threshold reaches specificity 0.95 on EyePACS")
    RULES[seed] = {"cuts": [float(c) for c in cuts], "threshold": float(thr)}
    print(f"  seed {seed}: cuts {[round(c, 3) for c in cuts]}  threshold {thr:.4f}")

(OUT / "carried_over_rules.json").write_text(json.dumps(RULES, indent=1), encoding="utf-8")


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_external.py", "-q"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0

print(f"""
CARRIED OVER from EyePACS validation, unchanged. Cell 2 does NOT re-fit them.

PRE-REGISTERED VERDICT (DECISION-057) — cell 3 applies it:
  G1 re-fit QWK < 0.60 | G2 re-fit sens < 0.60 | G3 severe recall < 0.30 with
  grade-0 > 0.90 | G4 coverage |r| >= 0.2 and larger on APTOS
  usable needs re-fit QWK >= 0.65 AND >= half the drop recovered AND none of G1-G4.

Expect a drop; it is the finding. Balanced accuracy may RISE while QWK falls.
""")
'''

CELL_2 = r'''
# -- Cell 2 -- inference over all 3,662 APTOS images, three seeds ------------
# COMMIT THIS. ~10 min.
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.xai.border_check import NoRetinaError, region_masks
from src.xai.gradcam import scalar_target

ds = DRDataset(APTOS, CACHE, train=False, image_size=224,
               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
print(f"dataset: {len(ds)} images")

# Retina coverage per image, for the field-of-view diagnostic (DECISION-054). Computed
# ONCE and shared across seeds — it is a property of the image, not of the model.
# No-retina images get NaN and are excluded downstream, counted (DECISION-052).
coverage, n_undef = [], 0
for path in ds.cache_paths:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    try:
        regions = region_masks(bgr)
    except NoRetinaError:
        coverage.append(float("nan"))
        n_undef += 1
        continue
    coverage.append(float((~regions["outside"]).mean()))
coverage = np.asarray(coverage)
print(f"retina coverage: median {np.nanmedian(coverage):.4f}  "
      f"range {np.nanmin(coverage):.4f}-{np.nanmax(coverage):.4f}")
print(f"  {n_undef} image(s) with no detectable retina, excluded from the diagnostic "
      f"and counted (DECISION-052)")

SCORES = {}
for seed in SEEDS:
    cfg = yaml.safe_load((CKPTS[seed].parent / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(ModelConfig(
        arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
        head=cfg["model"]["head"], pretrained=False))
    load_checkpoint(CKPTS[seed], model)
    mean, std = normalisation(model)
    model.eval().to(DEV)

    ds_seed = DRDataset(APTOS, CACHE, train=False, image_size=224, mean=mean, std=std)
    loader = DataLoader(ds_seed, batch_size=64, shuffle=False, num_workers=2)
    sc, ys = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y, idx in loader:
            sc.extend(scalar_target(model(x.to(DEV))).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    SCORES[seed] = np.asarray(sc)
    Y = np.asarray(ys)
    print(f"  seed {seed}: {len(sc)} predictions in {time.time() - t0:.0f}s  "
          f"mean score {np.mean(sc):.4f}")

np.savez(OUT / "aptos_scores.npz", y_true=Y, coverage=coverage,
         **{f"seed_{s}": SCORES[s] for s in SEEDS})
print(f"\nwrote {OUT / 'aptos_scores.npz'}")
'''

CELL_3 = r'''
# -- Cell 3 -- the carried-over result, the re-fitted secondary, the verdict --
# COMMIT THIS. Seconds.
import numpy as np

from src.data.fetch_run import write_manifest
from src.eval.external import (coverage_correlation, evaluate_external, refit_reference,
                               verdict)

carried_all, refit_all, cov_all = {}, {}, {}
for seed in SEEDS:
    s = SCORES[seed]
    carried_all[seed] = evaluate_external(
        s, Y, cuts=RULES[seed]["cuts"], threshold=RULES[seed]["threshold"])
    refit_all[seed] = refit_reference(s, Y)
    cov_all[seed] = coverage_correlation(s, Y, coverage)

print("APTOS EXTERNAL VALIDATION — arm E, EfficientNet-B0, three seeds\n")
print("HEADLINE: EyePACS decision rules carried over unchanged (DECISION-054)\n")
print(f"{'seed':>5}{'QWK':>9}{'acc':>8}{'bal acc':>9}{'ref sens':>10}{'ref spec':>10}")
for seed in SEEDS:
    c = carried_all[seed]
    print(f"{seed:>5}{c['qwk']:>9.4f}{c['accuracy']:>8.4f}{c['balanced_accuracy']:>9.4f}"
          f"{c['referable']['sensitivity']:>10.4f}{c['referable']['specificity']:>10.4f}")

cq = [carried_all[s]["qwk"] for s in SEEDS]
rq = [refit_all[s]["qwk"] for s in SEEDS]
print(f"\ncarried-over QWK  mean {np.mean(cq):.4f}  range {min(cq):.4f}-{max(cq):.4f}")
print(f"EyePACS held-out  {EYEPACS_QWK_HELD_OUT:.4f}   "
      f"DROP {EYEPACS_QWK_HELD_OUT - np.mean(cq):+.4f}")
print(f"\nSECONDARY, re-fitted on APTOS (never the headline):")
print(f"  QWK mean {np.mean(rq):.4f}  range {min(rq):.4f}-{max(rq):.4f}")
print(f"  sens@spec>=0.95 mean "
      f"{np.mean([refit_all[s]['sens_at_spec95'] for s in SEEDS]):.4f}")

print("\nfield-of-view diagnostic (DECISION-054), within grade:")
for seed in SEEDS:
    cc = cov_all[seed]
    print(f"  seed {seed}: max |r| {cc['max_abs_r']:.4f}  median |r| "
          f"{cc['median_abs_r']:.4f}  ({cc['n_used']} used, {cc['n_excluded']} excluded)")
print("  EyePACS comparison is computed locally; see cell 3's note.")

# The verdict, on the SEED MEAN (DECISION-042: never a single draw).
mean_carried = {"qwk": float(np.mean(cq)),
                "per_grade_recall": {g: float(np.mean(
                    [carried_all[s]["per_grade_recall"][g] for s in SEEDS]))
                    for g in range(5)}}
mean_refit = {"qwk": float(np.mean(rq)),
              "sens_at_spec95": float(np.mean(
                  [refit_all[s]["sens_at_spec95"] for s in SEEDS])),
              "per_grade_recall": {g: float(np.mean(
                  [refit_all[s]["per_grade_recall"][g] for s in SEEDS]))
                  for g in range(5)}}
v = verdict(mean_carried, mean_refit, None,
            {"max_abs_r": float(np.mean([cov_all[s]["max_abs_r"] for s in SEEDS]))},
            eyepacs_qwk=EYEPACS_QWK_HELD_OUT)

print("\n" + "=" * 78)
print("PRE-REGISTERED VERDICT (DECISION-057)")
print("=" * 78)
for k, failed in v["criteria"].items():
    print(f"  {'FAIL' if failed else 'ok  '}  {k}")
print(f"\n  fraction of the drop recovered by re-fitting: "
      f"{v['fraction_of_drop_recovered_by_refitting']:.3f}")
print(f"\n  {v['verdict']}")
print(f"\n  {v['note']}")
print("=" * 78)
print("""
G4 is only half-decided here: the APTOS side is computed, the EyePACS side is not,
because the EyePACS coverage figures need the validation cache and belong locally.
Run this after fetching, which completes G4:

  python -m src.eval.external --help    # (module is library-only; see the runbook)
""")

for name, obj in (("carried_over.json", carried_all), ("refit_secondary.json", refit_all),
                  ("coverage_correlation.json", cov_all), ("verdict.json", v)):
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")

write_manifest(OUT)
print(f"\nartefacts + MANIFEST.json in {OUT}")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)
print("""
Commit this notebook, then locally:
  python -m src.data.fetch_run --kernel rah098/<slug> --artefacts phase6_aptos
  python -m notebooks.gen_experiments
""")
'''


CELL_4 = r'''
# -- Cell 4 -- G4's EyePACS side, which completes the shortcut criterion ------
# COMMIT THIS. ~3 min.
#
# G4 is a comparison BETWEEN datasets: |r| >= 0.2 on APTOS AND larger than on EyePACS.
# The APTOS half alone decides nothing — a coverage-score correlation present equally on
# both datasets is a confound (coverage tracking image quality tracking severity), not a
# shortcut. Until this cell runs, `verdict()` reports G4 as UNDECIDED rather than as
# passing, because an uncomputed criterion reported as "ok" is the same class of error as
# the NaN recovery hatch (DECISION-058).
from src.data.manifest import load_split
from src.eval.external import coverage_correlation
from src.xai.border_check import NoRetinaError, region_masks

VAL = load_split("val")
vds = DRDataset(VAL, CACHE, train=False, image_size=224,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
print(f"EyePACS validation: {len(vds)} images")

vcov, v_undef = [], 0
for path in vds.cache_paths:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    try:
        regions = region_masks(bgr)
    except NoRetinaError:
        vcov.append(float("nan"))
        v_undef += 1
        continue
    vcov.append(float((~regions["outside"]).mean()))
vcov = np.asarray(vcov)
print(f"coverage: median {np.nanmedian(vcov):.4f}  "
      f"({v_undef} excluded, no retina — DECISION-052)")

cov_eyepacs = {}
for seed in SEEDS:
    cfg = yaml.safe_load((CKPTS[seed].parent / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(ModelConfig(
        arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
        head=cfg["model"]["head"], pretrained=False))
    load_checkpoint(CKPTS[seed], model)
    mean, std = normalisation(model)
    model.eval().to(DEV)
    ds_seed = DRDataset(VAL, CACHE, train=False, image_size=224, mean=mean, std=std)
    loader = DataLoader(ds_seed, batch_size=64, shuffle=False, num_workers=2)
    sc, ys = [], []
    with torch.no_grad():
        for x, y, idx in loader:
            sc.extend(scalar_target(model(x.to(DEV))).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    cov_eyepacs[seed] = coverage_correlation(np.asarray(sc), np.asarray(ys), vcov)
    pg = cov_eyepacs[seed]["per_grade_r"]
    print(f"  seed {seed}: " + "  ".join(f"g{g}={pg[g]:+.4f}" for g in range(5))
          + f"   max {cov_eyepacs[seed]['max_abs_r']:.4f}")

print()
print("G4 — the comparison, on max |r| across grades (DECISION-058):")
print(f"{'seed':>5}{'EyePACS':>10}{'APTOS':>10}{'APTOS larger?':>16}")
for seed in SEEDS:
    e = cov_eyepacs[seed]["max_abs_r"]
    a = cov_all[seed]["max_abs_r"]
    print(f"{seed:>5}{e:>10.4f}{a:>10.4f}{str(a >= 0.2 and a > e):>16}")

mean_e = float(np.mean([cov_eyepacs[s]["max_abs_r"] for s in SEEDS]))
mean_a = float(np.mean([cov_all[s]["max_abs_r"] for s in SEEDS]))
print()
print(f"  seed means: EyePACS {mean_e:.4f}  APTOS {mean_a:.4f}")

v2 = verdict(mean_carried, mean_refit, {"max_abs_r": mean_e}, {"max_abs_r": mean_a},
             eyepacs_qwk=EYEPACS_QWK_HELD_OUT)
print()
print(f"  G4 confirmed: {v2['criteria']['G4_shortcut_confirmed']}")
print(f"  G4 undecided: {v2['G4_undecided']}")
print()
print(f"  FINAL VERDICT: {v2['verdict']}")

(OUT / "coverage_correlation_eyepacs.json").write_text(
    json.dumps(cov_eyepacs, indent=1, default=float), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(v2, indent=1, default=float),
                                  encoding="utf-8")
write_manifest(OUT)
print()
print("verdict.json REPLACED with the G4-complete version; MANIFEST refreshed.")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4"]):
        print(CELLS[int(w) - 1])

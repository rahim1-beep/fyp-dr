"""Phase 5 — Grad-CAM and the border-artefact gate. Four cells, ~15 min.

    python -m notebooks.phase5_gradcam 1     # or 2, 3, 4

Arm E, seed 42, EfficientNet-B0 at 224 — the selected model. **Validation images only.**
The EyePACS test set is not opened here and does not open until after the 384 decision is
final (DECISION-045).

THIS IS A GATE, NOT A FIGURE. Thresholds were fixed in DECISION-046 before any heatmap
existed, and the failure path was fixed in DECISION-047 before any number landed. Cell 3
applies them and exits non-zero on failure. The `outside` threshold was later recalibrated
from 0.5 to 1.0 against measured nulls (DECISION-050) — an UNTRAINED network scores 0.615,
so the original could not be passed by any model; the Phase 5 measurement of 2.342 fails
both, so the change did not alter that verdict.

    statistic                                        PASS      FAIL
    ----------------------------------------------   -------   -------
    median RIM mass ratio (outer 10% of retina)      < 1.5     >= 1.5
    median OUTSIDE mass ratio (beyond the disc)      < 1.0     >= 1.0
    median RIM mass ratio, grades 3-4 only           < 1.5     >= 1.5
    median |corr(cam, cam_randomised)|               < 0.5     >= 0.5

THE RANDOMISATION GATE COMES FIRST because it can invalidate the other three. A saliency
map that barely changes when the target layer's weights are randomised is a property of
the image rather than of the model, and if that is true the other numbers are describing
edge contrast. The check stands on its own logic regardless of attribution.

THE SAMPLE IS PRE-SPECIFIED. 200 validation images, 40 per grade, drawn with SAMPLE_SEED,
plus a 20-image qualitative panel drawn by the same seed. Curating attractive heatmaps
after the fact is how an XAI chapter becomes decorative, and it is not detectable in the
output.

WHAT GRAD-CAM CANNOT DO HERE. EfficientNet-B0 at 224 ends on a 7x7 grid, so one CAM cell
covers ~32x32 input pixels. It **cannot localise microaneurysms**, which are sub-pixel at
this resolution. It resolves gross attention only — disc, macula, arcades, rim. That is
the honest claim for the write-up, and it is exactly the resolution this gate needs.

CELL 1 RUNS THE GRAD-CAM TESTS ON THE GPU, deliberately. They are device-parameterised,
so the cuda cases only actually execute here — a CPU-only machine can do nothing but
skip them. That is the countermeasure for cell 2's first failure (DECISION-048).

NEEDS THE CHECKPOINT. `best.pth` is gitignored and not fetched by `src.data.fetch_run`,
so the stage 3 notebook's OUTPUT must be attached as an input.

SESSION SETTINGS
    Accelerator : GPU T4 x2 (CPU works; ~4x slower and still only minutes)
    Internet    : ON
    Inputs      : fyp-dr-eyepacs-224, fyp-dr-code, AND the stage 3 notebook's output

HOW TO RUN
    Cell 1     by hand (~3 min)
    Cells 2-4  by COMMIT
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, checkpoint, and the pre-specified sample ----------------
# RUN INTERACTIVELY.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
OUT = WORK / "phase5"

RUN_ID = "phase4_stage3_arm_e_efficientnet_b0"   # arm E, seed 42 — the selected model
SAMPLE_SEED = 20260825       # pre-registered; do not change after seeing a result
N_PER_GRADE = 40             # 200 images total
N_PANEL = 20                 # qualitative panel, same seed

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
OUT.mkdir(parents=True, exist_ok=True)

from src.data.archive_cache import resolve_cache
from src.data.kaggle_paths import resolve_input

MOUNT = resolve_input("fyp-dr-eyepacs-224", owner="rah098")
CACHE = resolve_cache(MOUNT, WORK / "cache", expect_images=38788)
print("cache:", CACHE)

# The checkpoint. It is gitignored and fetch_run does not pull it, so it can only come
# from the stage 3 notebook's output.
CKPT = None
for cand in Path("/kaggle/input").rglob(f"{RUN_ID}/best.pth"):
    CKPT = cand
    break
if CKPT is None:
    raise SystemExit(
        f"{RUN_ID}/best.pth not found. + Add Input -> Your Work -> the stage 3 notebook's "
        "output. Grad-CAM needs the weights, not just the metrics.")
RUN_DIR = CKPT.parent
print("checkpoint:", CKPT)

import numpy as np, pandas as pd, torch, yaml
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from src.data.manifest import load_split
from src.models.factory import (ModelConfig, build_model, load_checkpoint,
                                normalisation)

cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
print(f"run config: arm {cfg['arm']}  arch {cfg['model']['arch']}  "
      f"head {cfg['model']['head']}  seed {cfg['seed']}")

# Rebuilt from the RUN's config, never from current defaults: a model analysed under a
# different head than it was trained with is a picture of a model that never existed.
model = build_model(ModelConfig(
    arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
    head=cfg["model"]["head"], pretrained=False))
# Via the shared loader (src/models/factory), which owns the ONE checkpoint convention.
# This line previously guessed `state["model"]` — a key that has never existed — and the
# fallback then handed load_state_dict the whole outer dict, producing a RuntimeError
# that reads like an architecture mismatch. DECISION-049.
state = load_checkpoint(CKPT, model)
print(f"checkpoint: epoch {state.get('epoch')}  val_qwk {state.get('val_qwk')}")
model.eval().to(DEV)
MEAN, STD = normalisation(model)
print("weights loaded; normalisation", MEAN, STD)

# The sample, fixed by seed BEFORE any heatmap exists.
val = load_split("val")
rng = np.random.default_rng(SAMPLE_SEED)
picks = []
for g in range(5):
    rows = val[val.label == g]
    k = min(N_PER_GRADE, len(rows))
    picks.append(rows.iloc[rng.choice(len(rows), k, replace=False)])
SAMPLE = pd.concat(picks, ignore_index=True)
PANEL = SAMPLE.iloc[rng.choice(len(SAMPLE), N_PANEL, replace=False)].reset_index(drop=True)
print(f"\nsample: {len(SAMPLE)} images, per grade "
      f"{SAMPLE.label.value_counts().sort_index().tolist()}")
print(f"panel : {len(PANEL)} images (seed {SAMPLE_SEED})")

assert (SAMPLE.label.isin([3, 4])).sum() > 0, (
    "the grade 3-4 gate cannot be cleared by a sample with no grade 3-4 images")


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"]) == 0
# NOT a formality. tests/test_gradcam.py is device-parameterised, so running it HERE
# exercises the cuda paths that a CPU-only dev box can only skip. Phase 5 cell 2
# died on exactly such a path -- a CPU torch.Generator seeding a CUDA tensor --
# after 24 CPU tests passed. This runs before cell 2 spends ten minutes.
assert run([sys.executable, "-m", "pytest", "tests/test_gradcam.py", "-v",
           "-k", "device or randomis"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_gradcam.py", "-q"]) == 0
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0

print("""
GATE THRESHOLDS (DECISION-046, fixed before any heatmap existed):
  median rim ratio < 1.5   |  median outside ratio < 1.0
  median rim ratio on grades 3-4 < 1.5   |  median |corr vs randomised| < 0.5

FAILURE PATH (DECISION-047, fixed before any number landed):
  randomisation fails -> TOOLING bug. Fix gradcam.py. No rebuild, no heatmap reported.
  outside fails       -> retina_mask is under-segmenting. A preprocessing BUG.
  rim fails           -> F1 occlusion test FIRST. A high CAM ratio is not yet proof
                         the DECISION depends on the rim. Only if |dsens| >= 0.02 does
                         the erosion value change and the cache get rebuilt.
""")
'''

CELL_2 = r'''
# -- Cell 2 -- CAMs, region ratios, and the randomisation control -------------
# COMMIT THIS. ~10 min.
import cv2
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset
from src.xai.gradcam import GradCAM, cam_correlation, randomise_last_block
from src.xai.border_check import NoRetinaError, image_ratios

ds = DRDataset(SAMPLE, CACHE, train=False, mean=MEAN, std=STD, image_size=224)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

# The randomised twin is built ONCE, outside the loop: a fresh randomisation per batch
# would compare each CAM against a different model and the median correlation would mean
# nothing.
model_rand = randomise_last_block(model, seed=SAMPLE_SEED).eval().to(DEV)

rows, corrs, n_border_undefined = [], [], 0
t0 = time.time()
with GradCAM(model) as cam_real, GradCAM(model_rand) as cam_rand:
    for bi, (x, y, idx) in enumerate(loader):
        x = x.to(DEV)
        real = cam_real(x)
        rand = cam_rand(x)
        for j in range(len(idx)):
            row = ds.df.iloc[int(idx[j])]
            bgr = cv2.imdecode(
                np.fromfile(str(ds.cache_paths[int(idx[j])]), np.uint8),
                cv2.IMREAD_COLOR)
            try:
                r = image_ratios(real.cam[j], bgr)
            except NoRetinaError:
                # DECISION-052: excluded from the statistic, counted, reported by the
                # gate. Cell 2's 200-image sample happened never to draw one; cell 3B on
                # all 5,268 did, which is how this surfaced.
                n_border_undefined += 1
                continue
            r["label"] = int(row["label"])
            r["image_path"] = str(row["image_path"])
            r["score"] = float(real.score[j])
            rows.append(r)
            corrs.append(cam_correlation(real.cam[j], rand.cam[j]))
        print(f"  batch {bi + 1}/{len(loader)}", flush=True)

print(f"\n{len(rows)} images in {time.time() - t0:.0f}s")
(OUT / "ratios.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
(OUT / "randomisation.json").write_text(json.dumps(corrs, indent=1), encoding="utf-8")
(OUT / "n_border_undefined.json").write_text(json.dumps(n_border_undefined),
                                             encoding="utf-8")
print("wrote ratios.json and randomisation.json")
'''

CELL_3 = r'''
# -- Cell 3 -- THE GATE ------------------------------------------------------
# COMMIT THIS. Seconds. Exits non-zero on failure — that is the point.
from src.data.notebook_check import validate_argv

GATE_CMD = [sys.executable, "-m", "src.xai.border_check",
            "--ratios", str(OUT / "ratios.json"),
            "--randomisation", str(OUT / "randomisation.json"),
            "--out", str(OUT / "gate.json"),
            "--n-border-undefined", str(n_border_undefined),
            "--gate"]
why = validate_argv(GATE_CMD)
if why:
    raise SystemExit(f"gate command is malformed: {why}")

rc = run(GATE_CMD)
gate = json.loads((OUT / "gate.json").read_text())

if rc == 0:
    print("""
GATE PASSED. DECISION-016's erosion choice is supported by evidence, which is what it
was deferred to Phase 5 for. Record the numbers in the write-up alongside the
7x7-resolution limitation, and proceed to Phase 6.
""")
else:
    failed = [k for k, v in gate["gates"].items() if not v]
    print(f"""
GATE FAILED: {failed}

Follow DECISION-047. Do NOT rebuild anything yet.

  randomisation -> TOOLING. Fix src/xai/gradcam.py. Nothing about the model is
                   implicated and no heatmap goes in the write-up until it passes.
  outside       -> NOT presumed to be the mask. Measured (DECISION-050): retina_mask
                   is slightly GENEROUS on cached images, not under-segmenting. Run
                   CELL 3B: O1 for dependence, O2 for whether it is a field-of-view
                   shortcut. Thresholds in DECISION-051.
  rim           -> RUN CELL 3B FIRST. A high CAM ratio is correlational. Only
                   |dsens| >= 0.03 justifies changing erosion and rebuilding, which
                   costs ~12 h including stage 2's seeds.
""")
'''

CELL_3B = r'''
# -- Cell 3B -- O1 and O2. ONLY IF THE RIM OR OUTSIDE GATE FAILED ------------
# COMMIT THIS but it is a no-op unless one of those gates failed. ~5 min.
#
# ON THE FULL VALIDATION SPLIT, not the 200-image gate sample. sens@spec>=0.95 needs the
# NATURAL distribution: a stratified 200 leaves ~80 non-referable images to estimate a
# 0.95-specificity threshold from, which is noise dressed as a measurement. Forward-only,
# so it costs seconds of GPU plus about a minute of mask computation (DECISION-051).
from src.data.manifest import load_split
from src.eval.thresholds import (choose_operating_point, referable_scores,
                                 sensitivity_specificity_curve)
from src.xai.border_check import (occlusion_delta, partition_borders,
                                  shrink_field_of_view)
from src.xai.gradcam import scalar_target

FAILED = [k for k in ("rim", "outside") if not gate["gates"].get(k, True)]
if not FAILED:
    print("rim and outside both passed — no occlusion test required, skipping.")

VAL = load_split("val")
vds = DRDataset(VAL, CACHE, train=False, mean=MEAN, std=STD, image_size=224)
vloader = DataLoader(vds, batch_size=48, shuffle=False, num_workers=2)
print(f"full validation split: {len(vds)} images")


def sens_at_spec(scores, y, fixed_threshold=None):
    """sens@spec>=0.95. The operating point is FIXED on the unoccluded data and reused —
    re-optimising after occluding would absorb the effect being measured."""
    if fixed_threshold is None:
        op = choose_operating_point(sensitivity_specificity_curve(scores, y))
        m = op.get("max_sens_at_spec") or {}
        return m.get("sensitivity", float("nan")), m.get("threshold")
    pos, pred = y >= 2, scores >= fixed_threshold
    return float((pred & pos).sum() / max(1, pos.sum())), fixed_threshold


base_scores, ys = [], []
with torch.no_grad():
    for x, y, idx in vloader:
        base_scores.extend(scalar_target(model(x.to(DEV))).cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
base_scores, ys = np.asarray(base_scores), np.asarray(ys)
base_sens, THR = sens_at_spec(base_scores, ys)
print(f"baseline sens@spec>=0.95 {base_sens:.4f} at threshold {THR:.4f}\n")

# ---- O1: replace the region, measure the change at the FIXED operating point ----
for REGION in FAILED:
    occ = []
    with torch.no_grad():
        for x, y, idx in vloader:
            x = x.to(DEV)
            bgrs = [cv2.imdecode(np.fromfile(str(vds.cache_paths[int(i)]), np.uint8),
                                 cv2.IMREAD_COLOR) for i in idx]
            d = occlusion_delta(model, x, bgrs, region=REGION)
            occ.extend((scalar_target(model(x)).cpu().numpy() + d).tolist())
    occ = np.asarray(occ)

    # DECISION-052: images with no detectable retina come back NaN. They are EXCLUDED and
    # COUNTED, never silently dropped — and the exclusion is applied to BOTH ARMS. A
    # baseline over 5,268 against an occluded arm over 5,267 is not a paired comparison,
    # and the difference would fold the exclusion into the effect being measured.
    keep = np.isfinite(occ)
    n_undef = int((~keep).sum())
    base_sub, _ = sens_at_spec(base_scores[keep], ys[keep], THR)
    s_occ, _ = sens_at_spec(occ[keep], ys[keep], THR)
    dsens = s_occ - base_sub
    print(f"[O1 {REGION}] excluded {n_undef} image(s) with no detectable retina "
          f"(DECISION-018/052); {int(keep.sum())} of {len(occ)} analysed")
    if n_undef:
        print(f"     full-split baseline {base_sens:.4f} -> subset baseline "
              f"{base_sub:.4f}  (both arms use the subset)")
    print(f"[O1 {REGION}] sens {base_sub:.4f} -> {s_occ:.4f}   dsens {dsens:+.4f}")
    if abs(dsens) < 0.01:
        print("     < 0.01  -> CORRELATE. Documented limitation, no remedy.")
    elif abs(dsens) < 0.03:
        print("     0.01-0.03 -> EQUIVOCAL. The fill is a large out-of-distribution")
        print("     change, so O2 decides.")
    else:
        print("     >= 0.03 -> REAL DEPENDENCE (subject to O2 for the mechanism).")
    (OUT / f"o1_{REGION}.json").write_text(
        json.dumps({"base_sens_full_split": base_sens, "base_sens_subset": base_sub,
                    "occluded_sens": s_occ, "dsens": dsens, "threshold": float(THR),
                    "n_analysed": int(keep.sum()), "n_border_undefined": n_undef},
                   indent=1), encoding="utf-8")

# ---- O2: the actual field-of-view test, only when `outside` failed ----
if "outside" in FAILED:
    print("\n[O2] field-of-view sweep — varies HOW MUCH surround there is, using the")
    print("     pipeline's own erosion, so every pixel stays in distribution.")
    means, n_undef_o2 = [], None
    for extra in (0.0, 0.03, 0.06, 0.09):
        sc = []
        with torch.no_grad():
            for x, y, idx in vloader:
                raw = [cv2.imdecode(np.fromfile(str(vds.cache_paths[int(i)]), np.uint8),
                                    cv2.IMREAD_COLOR) for i in idx]
                # DECISION-052: same exclusion, and it must be the SAME IMAGES at every
                # rung. A sweep whose membership changes between rungs would show a
                # "trend" that is really a change of denominator.
                usable, undef = partition_borders(raw)
                bgrs = [shrink_field_of_view(raw[j], extra) for j in usable]
                if not bgrs:
                    continue
                xb = torch.stack([
                    (torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
                     .permute(2, 0, 1).float() / 255 - torch.tensor(MEAN).view(3, 1, 1))
                    / torch.tensor(STD).view(3, 1, 1) for b in bgrs]).to(DEV)
                sc.extend(scalar_target(model(xb)).cpu().numpy().tolist())
        if n_undef_o2 is None:
            n_undef_o2 = len(vds) - len(sc)
            print(f"     excluded {n_undef_o2} image(s) with no detectable retina; "
                  f"{len(sc)} analysed at every rung")
        means.append(float(np.mean(sc)))
        print(f"     +{extra:.0%} erosion -> mean predicted grade {means[-1]:.4f}")

    total = means[-1] - means[0]
    monotone = all(b >= a for a, b in zip(means, means[1:])) or \
               all(b <= a for a, b in zip(means, means[1:]))
    print(f"\n     total change {total:+.4f}   monotone {monotone}")
    if monotone and abs(total) >= 0.10:
        print("     -> FIELD-OF-VIEW SHORTCUT CONFIRMED (DECISION-051).")
        print("        Remedy: randomise the surround as a TRAIN-ONLY augmentation.")
        print("        No cache rebuild — ~40 min retrain + ~2 h for stage 2's seeds.")
        print("        NOT constant-fill at preprocess: a constant still marks the")
        print("        boundary, so it moves the shortcut rather than removing it.")
    else:
        print("     -> NOT a field-of-view shortcut. O1's effect was the fill being")
        print("        out of distribution. Record as a limitation.")
    (OUT / "o2_fov_sweep.json").write_text(
        json.dumps({"extra_erosion": [0.0, 0.03, 0.06, 0.09], "mean_score": means,
                    "total": total, "monotone": monotone,
                    "n_border_undefined": n_undef_o2}, indent=1), encoding="utf-8")

print("""
Either way, Phase 6 now tests this hypothesis rather than merely reporting a drop
(DECISION-051): a drop on APTOS confirms nothing by itself, but a within-grade
correlation of |r| >= 0.2 between predicted score and retina-coverage fraction —
larger on APTOS than on EyePACS validation — does.
""")
'''

CELL_4 = r'''
# -- Cell 4 -- the qualitative panel, and keep the evidence ------------------
# COMMIT THIS. ~1 min.
#
# The panel is the PRE-SPECIFIED 20 from cell 1. It is rendered whatever the heatmaps
# look like; that is the difference between a figure and an exhibit.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pds = DRDataset(PANEL, CACHE, train=False, mean=MEAN, std=STD, image_size=224)
ploader = DataLoader(pds, batch_size=len(PANEL), shuffle=False, num_workers=0)
x, y, idx = next(iter(ploader))
with GradCAM(model) as g:
    res = g(x.to(DEV))

n = len(PANEL)
fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6))
for j in range(n):
    bgr = cv2.imdecode(np.fromfile(str(pds.cache_paths[int(idx[j])]), np.uint8),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    true = int(pds.df.iloc[int(idx[j])]["label"])
    pred = float(res.score[j])
    axes[0, j].imshow(rgb); axes[0, j].axis("off")
    axes[0, j].set_title(f"true {true}\npred {pred:.2f}", fontsize=8)
    axes[1, j].imshow(rgb); axes[1, j].imshow(res.cam[j], cmap="jet", alpha=0.45)
    axes[1, j].axis("off")
fig.suptitle(f"Grad-CAM — arm E, EfficientNet-B0 @224, seed 42 "
             f"(7x7 CAM: gross attention only, NOT lesion localisation)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "panel.png", dpi=140, bbox_inches="tight")
print("wrote panel.png")

for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1024:.0f} KB")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)
print("""
Commit this notebook, then locally:
  python -m src.data.fetch_run --kernel rah098/<slug> --run-id phase5

Phase 5 output is the gate verdict + the panel. Next: Phase 6 (APTOS, inference only),
then the 384 ablation, then Phase 7.
""")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_3B, CELL_4]

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(CELLS[int(w) - 1])

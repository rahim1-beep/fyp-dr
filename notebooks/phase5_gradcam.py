"""Phase 5 — Grad-CAM and the border-artefact gate. Four cells, ~15 min.

    python -m notebooks.phase5_gradcam 1     # or 2, 3, 4

Arm E, seed 42, EfficientNet-B0 at 224 — the selected model. **Validation images only.**
The EyePACS test set is not opened here and does not open until after the 384 decision is
final (DECISION-045).

THIS IS A GATE, NOT A FIGURE. Thresholds were fixed in DECISION-046 before any heatmap
existed, and the failure path was fixed in DECISION-047 before any number landed. Cell 3
applies them and exits non-zero on failure.

    statistic                                        PASS      FAIL
    ----------------------------------------------   -------   -------
    median RIM mass ratio (outer 10% of retina)      < 1.5     >= 1.5
    median OUTSIDE mass ratio (beyond the disc)      < 0.5     >= 0.5
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
from src.models.factory import ModelConfig, build_model, normalisation

cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
print(f"run config: arm {cfg['arm']}  arch {cfg['model']['arch']}  "
      f"head {cfg['model']['head']}  seed {cfg['seed']}")

# Rebuilt from the RUN's config, never from current defaults: a model analysed under a
# different head than it was trained with is a picture of a model that never existed.
model = build_model(ModelConfig(
    arch=cfg["model"]["arch"], num_outputs=cfg["model"]["num_outputs"],
    head=cfg["model"]["head"], pretrained=False))
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state["model"] if "model" in state else state)
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
  median rim ratio < 1.5   |  median outside ratio < 0.5
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
from src.xai.border_check import image_ratios

ds = DRDataset(SAMPLE, CACHE, train=False, mean=MEAN, std=STD, image_size=224)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

# The randomised twin is built ONCE, outside the loop: a fresh randomisation per batch
# would compare each CAM against a different model and the median correlation would mean
# nothing.
model_rand = randomise_last_block(model, seed=SAMPLE_SEED).eval().to(DEV)

rows, corrs = [], []
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
            r = image_ratios(real.cam[j], bgr)
            r["label"] = int(row["label"])
            r["image_path"] = str(row["image_path"])
            r["score"] = float(real.score[j])
            rows.append(r)
            corrs.append(cam_correlation(real.cam[j], rand.cam[j]))
        print(f"  batch {bi + 1}/{len(loader)}", flush=True)

print(f"\n{len(rows)} images in {time.time() - t0:.0f}s")
(OUT / "ratios.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
(OUT / "randomisation.json").write_text(json.dumps(corrs, indent=1), encoding="utf-8")
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
  outside       -> retina_mask is probably under-segmenting. Inspect the masks on the
                   worst images; a wrong mask is a preprocessing BUG, not an erosion
                   parameter.
  rim           -> RUN CELL 3B (the occlusion test) FIRST. A high CAM ratio is
                   correlational. Only |dsens| >= 0.02 justifies changing erosion and
                   rebuilding, which costs ~12 h including stage 2's seeds.
""")
'''

CELL_3B = r'''
# -- Cell 3B -- occlusion test. ONLY IF THE RIM GATE FAILED ------------------
# COMMIT THIS but it is a no-op unless the rim gate failed. ~2 min.
#
# DECISION-047 step F1: does the DECISION depend on the rim, or does the CAM merely
# light up there? Grad-CAM cannot tell you; replacing the region and re-running can.
from src.xai.border_check import occlusion_delta

if gate["gates"].get("rim", True):
    print("rim gate passed — occlusion test not required, skipping.")
else:
    deltas = []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(DEV)
            bgrs = [cv2.imdecode(np.fromfile(str(ds.cache_paths[int(i)]), np.uint8),
                                 cv2.IMREAD_COLOR) for i in idx]
            deltas.extend(occlusion_delta(model, x, bgrs).tolist())

    d = np.asarray(deltas)
    print(f"occlusion delta on the scalar score: mean {d.mean():+.4f}  "
          f"median {np.median(d):+.4f}  |mean| {abs(d.mean()):.4f}")
    (OUT / "occlusion.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
    print("""
Now recompute sens@spec>=0.95 with the occluded scores and compare:

  |dsens| <  0.02  -> the CAM shows a CORRELATE. Record as a documented limitation.
                      DO NOT rebuild the cache.
  |dsens| >= 0.02  -> real dependence. Go to DECISION-047 F2: measure the residual
                      annulus ratio at 5% and 10% erosion on a 2,000-image subsample
                      and take the SMALLEST value that brings it into line. Then F3:
                      rebuild (~9 h) + retrain arm E (~40 min) + re-run stage 2's
                      three seeds (~2 h).

NOT an option either way: full 0.9r masking. DECISION-016 rejected it for costing 19%
of retinal area including the periphery where proliferative disease appears, and that
reasoning does not change with this result.
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

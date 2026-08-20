"""Phase 2 full cache build  -  the four Kaggle notebook cells, kept in the repo.

Runs on Kaggle only. The local machine has no CUDA and does not have the 35.3 GB of
source images (CLAUDE.md S3). Expect **~2.5 h** with `--workers 4`.

The step-by-step UI walkthrough lives alongside this file; these are the cells themselves,
kept here so the repo and the runbook cannot drift apart. Print any cell with:

    python -m notebooks.phase2_build_cache 1      # or 2, 3, 4; no argument prints all

Notebook settings: Accelerator **None** (this is CPU work  -  do not spend GPU quota),
Internet **off**, three inputs attached: `dreamer07/eyepacs`, `mariaherrerot/aptos2019`,
and your own `fyp-dr-code`.
"""

import sys

CELL_1 = r'''
# -- Cell 1 -- setup, version report, pre-flight, leakage gate ----------------
import os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"
OUT  = WORK / "processed"

print("inputs mounted:", sorted(p.name for p in Path("/kaggle/input").iterdir()))

# Kaggle unzips an uploaded archive into the dataset, so the code arrives as a tree.
# If it landed one level deeper, find it rather than failing forty minutes from now.
CODE = Path("/kaggle/input/fyp-dr-code")
if not (CODE / "src").is_dir():
    found = [p for p in CODE.rglob("src/data/preprocess.py")]
    if not found:
        raise SystemExit(f"no src/ under {CODE}  -  check the fyp-dr-code dataset upload")
    CODE = found[0].parents[2]
print("code:", CODE)

# /kaggle/input is read-only; pytest and __pycache__ need to write. Copy it out.
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import cv2, numpy, pandas
print(f"\npython {sys.version.split()[0]}  |  {os.cpu_count()} cpus")
print(f"cv2 {cv2.__version__}  |  numpy {numpy.__version__}  |  pandas {pandas.__version__}")
try:
    import timm, torch, torchvision
    print(f"torch {torch.__version__}  |  torchvision {torchvision.__version__}  "
          f"|  timm {timm.__version__}")
    print("^ if these differ from requirements.txt, update the pins to match THIS image")
except ImportError as e:
    print("torch/timm not importable here:", e)

# -- Pre-flight: do the paths in the split CSVs actually exist? ---------------
# Two hours into a build is a bad time to learn that APTOS is nested differently than
# the manifest says. Three rows per split answers it in a second.
from src.data.manifest import load_split

SRC = {"eyepacs": Path("/kaggle/input/eyepacs"),
       "aptos":   Path("/kaggle/input/aptos2019")}

bad = []
for name in ("train", "val", "test", "aptos_train", "aptos_val", "aptos_test"):
    df = load_split(name)
    for row in df.head(3).itertuples(index=False):
        p = SRC[row.dataset] / row.image_path
        if not p.exists():
            bad.append(str(p))
    print(f"{name:<12} {len(df):>6} rows   e.g. {SRC[df.dataset.iloc[0]] / df.image_path.iloc[0]}")
if bad:
    raise SystemExit("SOURCE PATHS NOT FOUND:\n  " + "\n  ".join(bad))
print("\npre-flight OK  -  every sampled source path exists")


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    return rc


# CLAUDE.md S7  -  the leakage tests run before anything is written, every time.
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0, \
    "leakage tests failed  -  nothing gets built until they are green"
'''

CELL_2 = r'''
# -- Cell 2 -- EyePACS: 35,126 images. THE LONG ONE, roughly 2 hours ----------
rc_eyepacs = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "train", "--split", "val", "--split", "test",
    "--config", "configs/kaggle.yaml",
    "--src-root", "/kaggle/input/eyepacs",
    "--out-root", OUT,
    "--stats", WORK / "processed_stats.csv",
    "--workers", "4",
])
print("eyepacs exit:", rc_eyepacs)
'''

CELL_3 = r'''
# -- Cell 3 -- APTOS: 3,662 images, roughly 10 minutes ------------------------
rc_aptos = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "aptos_train", "--split", "aptos_val", "--split", "aptos_test",
    "--config", "configs/kaggle.yaml",
    "--src-root", "/kaggle/input/aptos2019",
    "--out-root", OUT,
    "--stats", WORK / "aptos_stats.csv",
    "--workers", "4",
])
print("aptos exit:", rc_aptos)

# A non-zero exit means SOME image has status != 'ok'. That is not a reason to stop: it
# is a reason to read the reconciliation below, log those images in docs/DECISIONS.md,
# and LEAVE THEM IN THEIR SPLIT. Dropping a val or test row silently rebalances that
# split  -  an R2 violation by omission.
'''

CELL_4 = r'''
# -- Cell 4 -- reconcile, re-run the leakage tests, measure, prepare to publish
rc_recon = run([
    sys.executable, "-m", "src.data.reconcile_cache",
    "--cache-root", OUT,
    "--config", "configs/kaggle.yaml",
    "--stats", WORK / "processed_stats.csv",
    "--stats", WORK / "aptos_stats.csv",
    "--decode-sample", "800",
])

rc_tests = run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])

tot = sum(p.stat().st_size for p in OUT.rglob("*.jpg"))
n = sum(1 for _ in OUT.rglob("*.jpg"))
print(f"\nMEASURED: {n} files, {tot / 1024**3:.3f} GB, mean {tot / n / 1024:.1f} KB/image")
print(f"free in /kaggle/working: {shutil.disk_usage(WORK).free / 1024**3:.1f} GB")

ok = (rc_recon == 0 and rc_tests == 0)
print("\n" + "=" * 72)
print("READY TO PUBLISH" if ok else "DO NOT PUBLISH  -  fix the failures above")
print("=" * 72)

if ok:
    # Make the published Dataset self-describing: whoever mounts it can tell which
    # splits it was reconciled against and which images were flagged.
    shutil.copytree(REPO / "data/splits", OUT / "splits", dirs_exist_ok=True)
    for f in ("processed_stats.csv", "aptos_stats.csv"):
        shutil.copy(WORK / f, OUT / f)
    os.chdir(WORK)               # never rmtree the directory you are standing in
    shutil.rmtree(REPO)          # keep the output Dataset to the cache alone
    print("\nNow: Save Version -> Save & Run All. When it finishes, open the notebook's")
    print("Output tab and create a New Dataset named  fyp-dr-eyepacs-224  (PRIVATE).")
    print("configs/kaggle.yaml already expects it at /kaggle/input/fyp-dr-eyepacs-224.")
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4]

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        print(CELLS[int(w) - 1])

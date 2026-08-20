"""Phase 2 full cache build — paste this into ONE Kaggle notebook cell and run it.

Runs on Kaggle only. The local machine has no CUDA and, more to the point, does not have
the 35.3 GB of source images (CLAUDE.md §3). Expect **~2.5 h** with --workers 4.

GETTING THE CODE THERE
----------------------
No git remote is configured, so the code travels as a Kaggle utility Dataset:

    # locally
    .venv/Scripts/python -m notebooks.make_bundle          # -> dist/fyp-dr-code.zip
    # upload dist/fyp-dr-code.zip as a PRIVATE Kaggle Dataset named `fyp-dr-code`
    # then in the notebook: + Add Input -> Datasets -> fyp-dr-code

The bundle carries `src/`, `configs/`, `tests/`, and `data/splits/` — the split CSVs
travel WITH the code, because reconciliation compares the cache against them and a
mismatched pair of versions is exactly the failure this step exists to catch.

NOTEBOOK SETTINGS
-----------------
    Accelerator : None (this is CPU work — do not burn GPU quota on it)
    Internet    : off
    Inputs      : dreamer07/eyepacs, mariaherrerot/aptos2019, fyp-dr-code

AFTER IT FINISHES
-----------------
The last three steps are the leakage-auditor's conditions and they are not optional:
reconciliation, the leakage tests, and the measured size. If any of them fails, the cache
does not get published — read the output, do not re-run and hope.

Then: Save Version -> "Save & Run All", and when it completes, create a new Dataset from
the notebook output so `/kaggle/working/processed/` survives the session wipe.
"""

CELL = r'''
# ============================================================================
# Phase 2 — full preprocessed cache build (EyePACS + APTOS)
# ============================================================================
import os, shutil, subprocess, sys, time
from pathlib import Path

CODE = "/kaggle/input/fyp-dr-code"          # the utility dataset
WORK = Path("/kaggle/working")
REPO = WORK / "fyp-dr"

# The code must be WRITABLE (pytest writes .pytest_cache, and __pycache__ appears
# everywhere); /kaggle/input is read-only, so copy it out first.
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.chdir(REPO)
sys.path.insert(0, str(REPO))

print("python :", sys.version.split()[0])
print("cpus   :", os.cpu_count())
import cv2, numpy, pandas
print("cv2    :", cv2.__version__, "| numpy:", numpy.__version__,
      "| pandas:", pandas.__version__)
print("splits :", sorted(p.name for p in (REPO / "data/splits").glob("*.csv")))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[exit {rc} in {(time.time() - t0) / 60:.1f} min]", flush=True)
    return rc


# --- 0. Leakage gate BEFORE anything is written (CLAUDE.md §7) -------------
assert run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"]) == 0, \
    "leakage tests failed — nothing gets built until they are green"

OUT = WORK / "processed"

# --- 1. EyePACS: 35,126 images --------------------------------------------
rc_e = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "train", "--split", "val", "--split", "test",
    "--config", "configs/kaggle.yaml",
    "--src-root", "/kaggle/input/eyepacs",
    "--out-root", str(OUT),
    "--stats", str(WORK / "processed_stats.csv"),
    "--workers", "4",
])

# --- 2. APTOS: 3,662 images ------------------------------------------------
rc_a = run([
    sys.executable, "-m", "src.data.preprocess",
    "--split", "aptos_train", "--split", "aptos_val", "--split", "aptos_test",
    "--config", "configs/kaggle.yaml",
    "--src-root", "/kaggle/input/aptos2019",
    "--out-root", str(OUT),
    "--stats", str(WORK / "aptos_stats.csv"),
    "--workers", "4",
])

# A non-zero exit here means SOME IMAGE has status != 'ok'. That is not a reason to stop
# — it is a reason to read the reconciliation report below, log those images in
# docs/DECISIONS.md, and LEAVE THEM IN THEIR SPLIT (dropping a val/test row silently
# rebalances that split: an R2 violation by omission).
print(f"\npreprocess exit codes: eyepacs={rc_e} aptos={rc_a}")

# --- 3. Reconcile the cache against the split CSVs, row for row -----------
rc_r = run([
    sys.executable, "-m", "src.data.reconcile_cache",
    "--cache-root", str(OUT),
    "--config", "configs/kaggle.yaml",
    "--stats", str(WORK / "processed_stats.csv"),
    "--stats", str(WORK / "aptos_stats.csv"),
    "--decode-sample", "800",
])

# --- 4. Leakage tests AGAIN, against the built artefact --------------------
rc_t = run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q"])

# --- 5. Measured size ------------------------------------------------------
tot = sum(p.stat().st_size for p in OUT.rglob("*.jpg"))
n = sum(1 for _ in OUT.rglob("*.jpg"))
print(f"\nMEASURED: {n} files, {tot / 1024**3:.3f} GB, mean {tot / n / 1024:.1f} KB")
print(f"free in /kaggle/working: "
      f"{shutil.disk_usage(WORK).free / 1024**3:.1f} GB")

ok = (rc_r == 0 and rc_t == 0)
print("\n" + ("=" * 70))
print("READY TO PUBLISH" if ok else "DO NOT PUBLISH — fix the failures above")
print("=" * 70)

# Copy the stats CSVs and the split CSVs into the output so the published Dataset is
# self-describing: whoever mounts it can tell which splits it was reconciled against.
if ok:
    shutil.copytree(REPO / "data/splits", OUT / "splits", dirs_exist_ok=True)
    for f in ("processed_stats.csv", "aptos_stats.csv"):
        shutil.copy(WORK / f, OUT / f)
    shutil.rmtree(REPO)      # keep the output Dataset to the cache alone
    print("\nNow: Save Version -> Save & Run All, then New Dataset from the output.")
    print("Name it `fyp-dr-eyepacs-224`, PRIVATE — configs/kaggle.yaml already expects")
    print("it at /kaggle/input/fyp-dr-eyepacs-224.")
'''

if __name__ == "__main__":
    print(CELL)

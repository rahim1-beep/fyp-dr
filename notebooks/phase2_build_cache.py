"""Phase 2 full cache build  -  the four Kaggle notebook cells, kept in the repo.

Runs on Kaggle only. The local machine has no CUDA and does not have the 35.3 GB of
source images (CLAUDE.md S3). Expect **~2.5 h** with `--workers 4`.

The step-by-step UI walkthrough lives alongside this file; these are the cells themselves,
kept here so the repo and the runbook cannot drift apart. Print any cell with:

    python -m notebooks.phase2_build_cache 1      # 1-5; 6 is the optional API cell

CELL 5 IS NOT OPTIONAL (DECISION-021). Kaggle caps notebook OUTPUT at 500 FILES.
The first build of this cache wrote all 38,788 images correctly and the saved output
kept 499 of them, silently. The cache is now packed into ONE archive and that archive
is verified while the session is still alive, before the loose tree is deleted.

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


# -- The commands the later cells will run, DEFINED AND VALIDATED HERE -------
# DECISION-022. Three sessions were lost at the packaging step, the last after 8.1 hours,
# to a command line nobody had ever executed: a missing `pack` subcommand that argparse
# rejected in 0.0 seconds once every piece of real work was already finished.
#
# So the archive command is built HERE, checked HERE against the module's real parser,
# and cell 5 only runs the variable. The command that executes is the command that was
# validated, and it is validated in minute one.
OUT = WORK / "processed"
ARCHIVE = WORK / "fyp-dr-eyepacs-224.zip"

PACK_CMD = [
    sys.executable, "-m", "src.data.archive_cache", "pack",
    "--cache-root", OUT,
    "--out", ARCHIVE,
    "--arcname", "processed",
    "--expect-images", "38788",
    "--expect-gb", "0.836",
]

from src.data.notebook_check import validate_argv

why = validate_argv(PACK_CMD)
if why:
    raise SystemExit(f"PACK_CMD is not a valid invocation: {why}")
print("\nPACK_CMD validated against src.data.archive_cache's own parser")

# And the static + executed check over every cell in the repo's copy of this notebook,
# including a real pack -> verify -> unpack on a three-file directory. Seconds.
assert run([sys.executable, "-m", "src.data.notebook_check", "--all", "--self-test"]) == 0

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
# -- Cell 4 -- reconcile, re-run the leakage tests, measure -------------------
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
print("READY TO ARCHIVE" if ok else "DO NOT ARCHIVE - fix the failures above")
print("=" * 72)

if ok:
    # The sidecars travel INSIDE the archive, so the published dataset says which splits
    # it was reconciled against, which images were flagged, and what config produced the
    # pixels. Copy them into the cache tree before packing.
    shutil.copytree(REPO / "data/splits", OUT / "splits", dirs_exist_ok=True)
    for f in ("processed_stats.csv", "aptos_stats.csv"):
        shutil.copy(WORK / f, OUT / f)
    print("\nsidecars copied into the cache. Run cell 5 to pack it.")
'''

CELL_5 = r'''
# -- Cell 5 -- pack into ONE file and PROVE it is complete --------------------
# DECISION-021. Kaggle caps notebook OUTPUT at 500 FILES. The first build wrote all
# 38,788 images correctly and the saved output kept 499 of them - silently, after
# reconciliation had passed, at the boundary between the session and the output. Then
# /kaggle/working is wiped. One archive is one file, and one file is under any
# file-count cap.
# PACK_CMD was built AND validated in cell 1 (DECISION-022). Do not retype it here:
# retyping is exactly how the subcommand went missing and cost 8.1 hours.
rc_pack = run(PACK_CMD)

if rc_pack != 0:
    raise SystemExit(
        "ARCHIVE VERIFICATION FAILED. The loose tree is still on disk - do not end this "
        "session and do not publish. Read the problems printed above."
    )

# Only now is deleting the tree safe, and only because the archive was read back and
# checked while the source still existed to be compared against.
shutil.rmtree(OUT)
os.chdir(WORK)                 # never rmtree the directory you are standing in
shutil.rmtree(REPO, ignore_errors=True)

left = sorted(p for p in WORK.rglob("*") if p.is_file())
print(f"\n/kaggle/working now holds {len(left)} file(s):")
for p in left:
    print(f"  {p.relative_to(WORK)}  {p.stat().st_size / 1024**3:.3f} GB")
if len(left) > 400:
    print("\nWARNING: more than 400 files - the 500-file output cap is close.")

print("\nNow: Save Version -> Save & Run All. When it finishes, open the Output tab,")
print("confirm you see ONE .zip of about 0.836 GB, and create a New Dataset from it")
print("named  fyp-dr-eyepacs-224  (PRIVATE).")
'''

CELL_5B = r'''
# -- Cell 5b (OPTIONAL) -- publish straight from the notebook via the Kaggle API ------
# Only worth it if you would rather the notebook confirm the dataset landed than click
# through the Output tab. It needs BOTH of these, neither of which cell 5 needs:
#   * Internet ON for this session
#   * Add-ons -> Secrets -> KAGGLE_KEY (and KAGGLE_USERNAME), from kaggle.com/settings
# Run it INSTEAD of the manual publish step, after cell 5 has verified the archive.
import json, os

from kaggle_secrets import UserSecretsClient

sec = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = sec.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = sec.get_secret("KAGGLE_KEY")

STAGE = WORK / "publish"
STAGE.mkdir(exist_ok=True)
shutil.move(str(ARCHIVE), STAGE / ARCHIVE.name)
(STAGE / "dataset-metadata.json").write_text(json.dumps({
    "title": "fyp-dr-eyepacs-224",
    "id": f"{os.environ['KAGGLE_USERNAME']}/fyp-dr-eyepacs-224",
    "licenses": [{"name": "other"}],
}, indent=2))

# --dir-mode zip would re-zip; the archive is already one file, so upload it as-is.
rc_pub = run([sys.executable, "-m", "kaggle", "datasets", "create",
              "-p", STAGE, "--private"])
print("publish exit:", rc_pub)

# The point of doing it this way: confirm it landed WHILE THE SESSION IS STILL ALIVE.
if rc_pub == 0:
    run([sys.executable, "-m", "kaggle", "datasets", "files",
         f"{os.environ['KAGGLE_USERNAME']}/fyp-dr-eyepacs-224"])
'''

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_5B]

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]   # 6 prints the optional API cell
    for w in which:
        print(CELLS[int(w) - 1])

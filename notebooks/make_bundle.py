"""Zip the code + split CSVs into a Kaggle utility Dataset.

There is no git remote, so this is how the project reaches Kaggle. The bundle is small
(< 1 MB) and deliberately carries no images and no credentials.

    .venv/Scripts/python -m notebooks.make_bundle

Upload `dist/fyp-dr-code.zip` as a PRIVATE Kaggle Dataset named `fyp-dr-code`. On every
later re-upload use "New Version" on the same dataset, so the notebook's input reference
keeps working.

The split CSVs travel WITH the code on purpose. `src/data/reconcile_cache.py` compares the
built cache against them; a cache built from one generation of splits and reconciled
against another would pass every count and still be wrong.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "dist" / "fyp-dr-code.zip"

INCLUDE_DIRS = ["src", "configs", "tests", "data/splits", "notebooks"]
INCLUDE_FILES = ["requirements.txt", "CLAUDE.md", "PROGRESS.md", "docs/DECISIONS.md"]

# Credentials must never end up in a dataset, private or not.
FORBIDDEN = {"kaggle.json", ".env"}


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, check=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def files() -> list[Path]:
    out: list[Path] = []
    for d in INCLUDE_DIRS:
        for p in sorted((REPO / d).rglob("*")):
            if not p.is_file():
                continue
            if any(part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
                   for part in p.parts):
                continue
            if p.suffix in {".pyc", ".pyo"}:
                continue
            out.append(p)
    for f in INCLUDE_FILES:
        p = REPO / f
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    picked = files()
    bad = [p for p in picked if p.name in FORBIDDEN]
    if bad:
        print(f"REFUSING: bundle would contain credentials: {bad}", file=sys.stderr)
        return 2

    DEST.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED) as z:
        for p in picked:
            z.write(p, p.relative_to(REPO).as_posix())
        z.writestr("BUNDLE.txt", f"fyp-dr code bundle\ngit: {sha}\nfiles: {len(picked)}\n")

    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()[:16]
    print(f"{DEST}  ({DEST.stat().st_size / 1024:.0f} KB, {len(picked)} files)")
    print(f"git    : {sha}")
    print(f"sha256 : {digest}")
    if sha.endswith("-dirty"):
        print("\nWARNING: working tree is dirty — the bundle does not match any commit.")
    print("\nUpload as a PRIVATE Kaggle Dataset named `fyp-dr-code`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

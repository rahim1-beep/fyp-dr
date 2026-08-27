"""Pull a committed Kaggle notebook's run artefacts into `runs/<run_id>/`.

Six arms in Phase 4, plus re-runs, means doing this a dozen times through the Output tab
otherwise — and the Output tab is where two of this project's three lost sessions ended
up (DECISION-021).

TWO KINDS OF THING. `--run-id` fetches a TRAINING run and requires config.yaml,
metrics.json and train_log.csv — those checks are what make a committed run the record of
a real execution (R4/R6). `--artefacts` fetches an ANALYSIS set (Phase 5/6/7): a directory
of JSON and figures with none of those files, because nothing was trained. The download
and auth machinery is shared; only the locating, verification and destination differ.
See DECISION-055.

WHAT COMES BACK. `runs/<run_id>/config.yaml`, `metrics.json`, `train_log.csv`, plus
`val_outputs.npz` and `thresholds.json` when the run has them. **Never `*.pth`** — checkpoints are gitignored, they are hundreds of MB, and they are not the
record of a run. The three text files are (R4/R6).

VERIFY BEFORE WRITING, NOT AFTER. The download lands in a temp directory and every check
runs there. Only if all of them pass does anything move into `runs/`. A partial or
mismatched download therefore never half-lands in the repo, where it would look like a
committed result. That ordering is DECISION-021's lesson applied to the other direction
of transfer.

THE CHECKS. Not "the files exist" — that is what the Output tab already tells you:

  * `metrics.json` parses, carries the keys the eval layer writes, and its `run_id`
    matches the one asked for
  * `config.yaml` parses and its `run_id` matches too
  * `train_log.csv` has the columns the training loop writes, and at least one row
  * the three agree with each other: `best_epoch` falls inside the log's epoch range, and
    `epochs_run` matches the number of rows

The last group is the point. Three files from three different code paths that disagree
mean something was mixed up between sessions, and no per-file check would see it.

    python -m src.data.fetch_run --kernel rah098/fyp-dr-phase3-baseline \\
        --run-id phase3_baseline_resnet18

Auth, and the version that works, are in `AUTH_HELP` below and DECISION-027.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

WANTED = ("config.yaml", "metrics.json", "train_log.csv")

# Copied when present, not required. `val_outputs.npz` is ~150 KB of raw validation
# outputs and is what makes threshold and operating-point analysis a local, GPU-free step
# (DECISION-030); `thresholds.json` is its analysis. Runs from before those existed are
# still perfectly valid and must not be rejected for lacking them.
OPTIONAL = ("val_outputs.npz", "thresholds.json")
NEVER_COPY_SUFFIXES = (".pth", ".pt", ".ckpt", ".onnx", ".safetensors")

# Keys `src/eval/metrics.compute_all` writes plus what train.py adds. Checked because a
# file that parses is not the same as a file that says anything.
METRICS_KEYS = ("run_id", "arm", "qwk", "qwk_ci", "accuracy", "balanced_accuracy",
                "confusion_matrix", "referable", "collapse", "support")

# The STABLE core of what `fit` writes per epoch. Deliberately not the full current
# schema: `val_collapse_level` and `val_collapse_warnings` were added by DECISION-026 and
# runs committed before that do not have them. Demanding the current schema would reject
# the very run this module was written to fetch.
TRAIN_LOG_COLUMNS = ("epoch", "train_loss", "val_qwk", "val_acc", "val_balanced_acc")


class FetchError(RuntimeError):
    """Raised with everything known about why the fetch is not trustworthy."""


AUTH_HELP = """
Kaggle auth, in the order that actually worked here (DECISION-027):

  1. `kaggle auth login`  — the OAuth flow. This is the current mechanism.
  2. Move any legacy `~/.kaggle/kaggle.json` OUT OF THE WAY. A stale API-token file
     SHADOWS the newer OAuth credentials, and the symptom is a flat
     `401 Unauthenticated` with no hint that a second credential source exists.
  3. The `kaggle` package must be recent: 2.2.4 works, 1.7.4.5 does not expose the
     import path this module used to rely on.
"""


def kaggle_cli() -> list[str] | None:
    """The kaggle console script, if one is on PATH or beside this interpreter."""
    import shutil

    found = shutil.which("kaggle")
    if found:
        return [found]

    beside = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32"
                                            else "kaggle")
    return [str(beside)] if beside.exists() else None


def download_kernel_output(kernel: str, dest: Path, prefer: str = "cli") -> Path:
    """`kaggle kernels output <kernel> -p <dest>`.

    THE CLI FIRST, the library second (DECISION-027). The import path proved to be the
    fragile part: `kaggle` 1.7.4.5 and 2.2.4 do not agree about where `KaggleApi` lives,
    and the console script is also what knows how to use `kaggle auth login`'s OAuth
    credentials. The library remains as a fallback for an environment with no console
    script, and both failures are reported together rather than the first one hiding the
    second.
    """
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    attempts: list[str] = []

    order = ["cli", "lib"] if prefer == "cli" else ["lib", "cli"]
    for how in order:
        if how == "cli":
            argv = kaggle_cli()
            if argv is None:
                attempts.append("cli: no `kaggle` executable on PATH or beside python")
                continue
            cmd = argv + ["kernels", "output", kernel, "-p", str(dest)]
            print("  $ " + " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return dest
            attempts.append(
                f"cli: exit {proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip().splitlines()[-1:] or ['no output']}"
            )
        else:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                api.kernels_output(kernel, path=str(dest))
                return dest
            except Exception as exc:                # noqa: BLE001
                attempts.append(f"library: {type(exc).__name__}: {exc}")

    raise FetchError(
        f"could not fetch output for kernel {kernel!r}. Tried:\n  - "
        + "\n  - ".join(attempts)
        + "\n\nThe slug is '<username>/<kernel-slug>' from the notebook URL, and the "
        "notebook must have been COMMITTED (Save & Run All) — an interactive session has "
        "no saved output.\n" + AUTH_HELP
    )


def find_run_dir(root: Path, run_id: str) -> Path:
    """Locate `runs/<run_id>/` in a download, whatever the output nesting is.

    By shape rather than by path, for the same reason as DECISION-023: the layout of
    someone else's archive is not a stable interface. A directory named `run_id` that
    contains `metrics.json` is.
    """
    root = Path(root)
    candidates = [d for d in root.rglob(run_id) if d.is_dir()]
    with_metrics = [d for d in candidates if (d / "metrics.json").exists()]

    if len(with_metrics) == 1:
        return with_metrics[0]
    if len(with_metrics) > 1:
        shallow = sorted(with_metrics, key=lambda d: len(d.relative_to(root).parts))
        if len(shallow[0].relative_to(root).parts) < len(shallow[1].relative_to(root).parts):
            return shallow[0]
        raise FetchError(
            f"{len(with_metrics)} directories named {run_id!r} contain metrics.json and "
            f"none is shallower: {[str(d.relative_to(root)) for d in with_metrics]}"
        )

    listing = []
    for d in sorted(root.rglob("*")):
        if d.is_dir() and len(d.relative_to(root).parts) <= 3:
            listing.append(f"  {d.relative_to(root)}/")
    if candidates:
        listing.append(f"  (found {run_id!r} but without metrics.json: "
                       f"{[str(d.relative_to(root)) for d in candidates]})")
    raise FetchError(
        f"no run directory {run_id!r} with a metrics.json in the kernel output."
        + ("\nwhat the output contains:\n" + "\n".join(listing[:25]) if listing else "")
    )


def verify(run_dir: Path, run_id: str) -> dict:
    """Every check, on the DOWNLOADED copy. Raises FetchError listing all problems.

    Returns a small summary for the caller to print, so the operator sees what landed
    rather than just "ok".
    """
    import pandas as pd
    import yaml

    problems: list[str] = []

    missing = [w for w in WANTED if not (run_dir / w).exists()]
    if missing:
        present = sorted(p.name for p in run_dir.iterdir())
        raise FetchError(
            f"{run_dir} is missing {missing}. It holds: {present}. A run without these "
            "is not a record of anything (R4/R6)."
        )

    # --- metrics.json ---
    metrics = {}
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"metrics.json does not parse: {exc}")

    if metrics:
        absent = [k for k in METRICS_KEYS if k not in metrics]
        if absent:
            problems.append(f"metrics.json is missing key(s): {absent}")
        if metrics.get("run_id") not in (None, run_id):
            problems.append(
                f"metrics.json is for run_id {metrics.get('run_id')!r}, not {run_id!r}"
            )
        cm = metrics.get("confusion_matrix")
        if cm is not None and len(cm) != 5:
            problems.append(f"confusion_matrix is {len(cm)}x?, expected 5x5")
        if metrics.get("is_smoke_test"):
            problems.append(
                "metrics.json says is_smoke_test: true. A smoke run is not a result and "
                "does not belong in runs/ as one; re-fetch the real run, or pass --force "
                "deliberately."
            )

    # --- config.yaml ---
    cfg = {}
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        problems.append(f"config.yaml does not parse: {exc}")

    if cfg and cfg.get("run_id") not in (None, run_id):
        problems.append(f"config.yaml is for run_id {cfg.get('run_id')!r}, not {run_id!r}")

    # --- train_log.csv ---
    log = None
    try:
        log = pd.read_csv(run_dir / "train_log.csv")
    except Exception as exc:                        # noqa: BLE001
        problems.append(f"train_log.csv does not parse: {exc}")

    if log is not None:
        absent = [c for c in TRAIN_LOG_COLUMNS if c not in log.columns]
        if absent:
            problems.append(f"train_log.csv is missing column(s): {absent}")
        if len(log) == 0:
            problems.append("train_log.csv has no rows")

    # --- the three against each other ---
    if metrics and log is not None and len(log) and "epoch" in log.columns:
        best = metrics.get("best_epoch")
        epochs = set(log["epoch"].tolist())
        if best is not None and best >= 0 and best not in epochs:
            problems.append(
                f"metrics.json says best_epoch {best}, which is not in train_log.csv's "
                f"epochs {sorted(epochs)[:12]} — these files are from different runs"
            )
        ran = metrics.get("epochs_run")
        if ran is not None and ran != len(log):
            problems.append(
                f"metrics.json says epochs_run {ran} but train_log.csv has {len(log)} "
                "rows — these files are from different runs"
            )

    if problems:
        raise FetchError(
            f"{run_dir} failed verification ({len(problems)} problem(s)):\n  - "
            + "\n  - ".join(problems)
            + "\n\nNothing was copied into the repo."
        )

    return {
        "run_id": run_id,
        "arm": metrics.get("arm"),
        "qwk": metrics.get("qwk"),
        "epochs": len(log) if log is not None else 0,
        "best_epoch": metrics.get("best_epoch"),
        "git": cfg.get("git"),
        "is_smoke": bool(metrics.get("is_smoke_test")),
    }


def install(run_dir: Path, target: Path, force: bool = False) -> list[str]:
    """Copy the wanted files into the repo. Returns what was copied."""
    target = Path(target)
    if target.exists() and any(target.iterdir()) and not force:
        existing = sorted(p.name for p in target.iterdir())
        raise FetchError(
            f"{target} already exists and holds {existing}. A committed run is the "
            "record of that run — overwriting it silently would break R6. Pass --force "
            "if replacing it is what you mean."
        )

    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in WANTED:
        shutil.copy2(run_dir / name, target / name)
        copied.append(name)
    for name in OPTIONAL:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, target / name)
            copied.append(name)
    return copied


def skipped_artefacts(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir()
                  if p.suffix.lower() in NEVER_COPY_SUFFIXES)


# ----------------------------------------------------------------------------------
# ARTEFACT SETS — analysis output, not training runs
#
# Phase 5 (and Phase 6, and Phase 7) produce a directory of JSON and figures with NO
# config.yaml, metrics.json or train_log.csv, because nothing was trained. `find_run_dir`
# and `verify` both hard-require metrics.json, correctly: those checks are what make a
# committed run a record of a real execution (R4/R6), and loosening them to admit
# analysis output would weaken the guarantee for the artefacts that actually need it.
#
# So this is a second, parallel path that reuses the download and auth machinery — which
# is the part that has actually cost sessions (kaggle 2.2.4, the shadowed OAuth
# credentials) — and applies verification appropriate to what an analysis set IS.
# ----------------------------------------------------------------------------------

ARTEFACT_ROOT = REPO / "analysis"
ARTEFACT_MANIFEST = "MANIFEST.json"


def find_artefact_dir(root: Path, name: str) -> Path:
    """Locate an artefact directory by SHAPE, as `find_run_dir` does (DECISION-023).

    A directory named `name` holding at least one `.json`. Requiring JSON rather than
    "any file" is what stops an empty or half-written output directory being installed
    and committed as though it were a result.
    """
    root = Path(root)
    candidates = [d for d in root.rglob(name) if d.is_dir()]
    with_json = [d for d in candidates if any(d.glob("*.json"))]

    if len(with_json) == 1:
        return with_json[0]
    if len(with_json) > 1:
        shallow = sorted(with_json, key=lambda d: len(d.relative_to(root).parts))
        if len(shallow[0].relative_to(root).parts) < len(shallow[1].relative_to(root).parts):
            return shallow[0]
        raise FetchError(
            f"{len(with_json)} directories named {name!r} contain JSON and none is "
            f"shallower: {[str(d.relative_to(root)) for d in with_json]}"
        )
    if candidates:
        raise FetchError(
            f"found {name!r} but it holds no .json: "
            f"{[sorted(q.name for q in d.iterdir()) for d in candidates]}. An artefact "
            "set with no JSON is not a result."
        )
    raise FetchError(f"no directory named {name!r} in the kernel output.")


def verify_artefacts(art_dir: Path) -> dict:
    """Every JSON parses; checksums match when a MANIFEST is present.

    A figure that cannot be traced to the numbers beside it is decoration, so the JSON
    is the part that must be intact. `MANIFEST.json` is written by newer notebooks and
    is verified when present; the Phase 5 set predates it, which is reported rather than
    treated as a failure.
    """
    import hashlib

    art_dir = Path(art_dir)
    files = sorted(q for q in art_dir.rglob("*") if q.is_file())
    if not files:
        raise FetchError(f"{art_dir} is empty")

    problems, parsed = [], {}
    for q in files:
        if q.suffix.lower() != ".json" or q.name == ARTEFACT_MANIFEST:
            continue
        try:
            parsed[q.name] = json.loads(q.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{q.name} is not valid JSON: {exc}")

    manifest_path = art_dir / ARTEFACT_MANIFEST
    checked = 0
    if manifest_path.exists():
        try:
            want = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
        except json.JSONDecodeError as exc:
            problems.append(f"{ARTEFACT_MANIFEST} is not valid JSON: {exc}")
            want = {}
        for rel, digest in want.items():
            q = art_dir / rel
            if not q.exists():
                problems.append(f"{rel} is in the manifest but missing")
                continue
            got = hashlib.sha256(q.read_bytes()).hexdigest()
            if got != digest:
                problems.append(f"{rel} sha256 {got[:12]} != manifest {digest[:12]}")
            checked += 1

    if problems:
        raise FetchError(f"{art_dir} failed verification:\n  - " + "\n  - ".join(problems))

    return {"n_files": len(files), "n_json": len(parsed), "n_checksummed": checked,
            "has_manifest": manifest_path.exists(),
            "bytes": sum(q.stat().st_size for q in files),
            "names": [q.relative_to(art_dir).as_posix() for q in files]}


def install_artefacts(art_dir: Path, target: Path, force: bool = False) -> list[str]:
    """Copy the WHOLE artefact set, minus anything in NEVER_COPY_SUFFIXES.

    Unlike a training run there is no fixed WANTED list — the set is whatever the
    analysis produced, and an allowlist here would silently drop a new artefact the day
    it is added.
    """
    target = Path(target)
    if target.exists() and any(target.iterdir()) and not force:
        raise FetchError(
            f"{target} already exists and holds "
            f"{sorted(q.name for q in target.iterdir())}. Pass --force if replacing it "
            "is what you mean."
        )
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for q in sorted(art_dir.rglob("*")):
        if not q.is_file() or q.suffix.lower() in NEVER_COPY_SUFFIXES:
            continue
        dest = target / q.relative_to(art_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(q, dest)
        copied.append(q.relative_to(art_dir).as_posix())
    return copied


def write_manifest(art_dir: Path) -> Path:
    """Write `MANIFEST.json` (sha256 per file) so a later fetch can verify the set.

    Called at the END of an analysis notebook. Without it, "verified" means only that
    the JSON parses.
    """
    import hashlib

    art_dir = Path(art_dir)
    files = {q.relative_to(art_dir).as_posix():
             hashlib.sha256(q.read_bytes()).hexdigest()
             for q in sorted(art_dir.rglob("*"))
             if q.is_file() and q.name != ARTEFACT_MANIFEST}
    out = art_dir / ARTEFACT_MANIFEST
    out.write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kernel", required=True,
                    help="kernel slug, '<username>/<kernel-slug>' from the notebook URL")
    ap.add_argument("--run-id", action="append", default=[],
                    help="run id to pull; repeatable, for a multi-arm notebook")
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--artefacts", action="append", default=[], metavar="NAME",
                    help="an ANALYSIS artefact set (Phase 5/6/7): a directory of JSON "
                         "and figures with no config.yaml or metrics.json, because "
                         "nothing was trained. Installed under --analysis-root. "
                         "Repeatable, and combinable with --run-id.")
    ap.add_argument("--analysis-root", type=Path, default=ARTEFACT_ROOT,
                    help="where artefact sets land (default: analysis/)")
    ap.add_argument("--from-dir", type=Path,
                    help="skip the download and read an already-downloaded output "
                         "directory; for re-verifying, and for testing this module")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing runs/<run_id>/")
    ap.add_argument("--keep-download", type=Path,
                    help="also leave the raw download here")
    ap.add_argument("--prefer", choices=("cli", "lib"), default="cli",
                    help="transport to try first; the CLI is the default because the "
                         "library import path is the part that has broken (DECISION-027)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.run_id and not args.artefacts:
        build_parser().error("give at least one --run-id or --artefacts")

    tmp = None
    try:
        if args.from_dir:
            root = Path(args.from_dir)
            print(f"reading kernel output from {root} (no download)")
        else:
            tmp = tempfile.TemporaryDirectory()
            root = Path(tmp.name)
            print(f"fetching output of kernel {args.kernel} ...", flush=True)
            download_kernel_output(args.kernel, root, prefer=args.prefer)
            n = sum(1 for _ in root.rglob("*") if _.is_file())
            print(f"  downloaded {n} file(s)")

        failures = []
        for run_id in args.run_id:
            print(f"\n--- {run_id} ---")
            try:
                run_dir = find_run_dir(root, run_id)
                print(f"  found   : {run_dir}")

                skipped = skipped_artefacts(run_dir)
                if skipped:
                    print(f"  skipping: {skipped} (checkpoints are not the record)")

                summary = verify(run_dir, run_id)
                print(f"  verified: arm {summary['arm']}, {summary['epochs']} epochs, "
                      f"best {summary['best_epoch']}, qwk "
                      f"{summary['qwk']:.4f}" if summary["qwk"] is not None
                      else "  verified")
                if summary.get("git"):
                    print(f"  git     : {summary['git']}")

                target = Path(args.runs_root) / run_id
                copied = install(run_dir, target, force=args.force)
                print(f"  copied  : {copied} -> {target}")
            except FetchError as exc:
                print(f"  FAILED: {exc}")
                failures.append(run_id)

        for name in args.artefacts:
            print(f"\n--- {name} (artefact set) ---")
            try:
                art_dir = find_artefact_dir(root, name)
                print(f"  found   : {art_dir}")
                summary = verify_artefacts(art_dir)
                print(f"  verified: {summary['n_files']} file(s), "
                      f"{summary['n_json']} JSON, "
                      f"{summary['bytes'] / 1024:.0f} KB")
                if summary["has_manifest"]:
                    print(f"  checksums: {summary['n_checksummed']} file(s) match "
                          f"{ARTEFACT_MANIFEST}")
                else:
                    print(f"  checksums: no {ARTEFACT_MANIFEST} in this set — JSON was "
                          "parsed but bytes were not verified")
                target = Path(args.analysis_root) / name
                copied = install_artefacts(art_dir, target, force=args.force)
                print(f"  copied  : {len(copied)} file(s) -> {target}")
                for c in copied:
                    print(f"              {c}")
            except FetchError as exc:
                print(f"  FAILED: {exc}")
                failures.append(name)

        if args.keep_download and tmp is not None:
            shutil.copytree(root, args.keep_download, dirs_exist_ok=True)
            print(f"\nraw download kept at {args.keep_download}")
    finally:
        if tmp is not None:
            tmp.cleanup()

    print()
    if failures:
        print(f"FETCH FAILED for {failures}. Nothing partial was written.")
        return 1
    if args.run_id:
        print(f"FETCH OK - {len(args.run_id)} run(s) installed under {args.runs_root}")
        print("Commit them: runs/<id>/{config.yaml,metrics.json,train_log.csv} are "
              "tracked, *.pth is not.")
    if args.artefacts:
        print(f"FETCH OK - {len(args.artefacts)} artefact set(s) installed under "
              f"{args.analysis_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

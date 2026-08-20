"""Resolve a Kaggle input dataset to a real directory, whatever layout is in force.

Kaggle moved where attached datasets appear (DECISION-020). Measured on the Phase 2 build:

    was:  /kaggle/input/{slug}/
    now:  /kaggle/input/datasets/{owner}/{slug}/

and `/kaggle/input` is read-only, so nothing can be symlinked *into* it — a working session
symlinks into `/kaggle/working/inputs/` instead, which is a third place to look.

Hardcoding any one of those is how the next notebook dies at line 3 after the queue wait.
This tries the known layouts in order and, when none matches, raises with the ACTUAL
directory listing, so the fix is one line here rather than a diagnostic session.

    from src.data.kaggle_paths import resolve_input
    eyepacs = resolve_input("eyepacs", owner="dreamer07")

Off Kaggle, `resolve_input` also accepts an explicit override so the same code runs
locally against `configs/local.yaml` paths.
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORKING_INPUTS = Path("/kaggle/working/inputs")


def on_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def candidate_paths(slug: str, owner: str | None = None) -> list[Path]:
    """Every place this dataset could be, newest layout first."""
    out: list[Path] = []
    if owner:
        out.append(KAGGLE_INPUT / "datasets" / owner / slug)   # current layout
    out.append(KAGGLE_INPUT / slug)                            # historical layout
    out.append(WORKING_INPUTS / slug)                          # session symlink
    if owner:
        out.append(WORKING_INPUTS / owner / slug)
    return out


def resolve_input(
    slug: str,
    owner: str | None = None,
    *,
    override: str | Path | None = None,
    must_contain: str | None = None,
) -> Path:
    """The directory holding this dataset's files.

    `must_contain` is a relative path that has to exist inside the candidate — pass it
    when a layout could plausibly resolve to an empty or wrapper directory. For EyePACS
    that is `data/data`; for APTOS, `train_images`.
    """
    if override is not None:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(f"override path {p} is not a directory")
        return p

    tried = candidate_paths(slug, owner)
    for p in tried:
        if not p.is_dir():
            continue
        if must_contain and not (p / must_contain).exists():
            continue
        return p

    # Refuse to guess, and show what is actually there. A notebook that fails here has
    # lost a minute; one that silently resolves to the wrong directory loses the session
    # and produces a cache nobody can trust.
    listing = []
    for root in (KAGGLE_INPUT, KAGGLE_INPUT / "datasets", WORKING_INPUTS):
        if root.is_dir():
            names = sorted(q.name for q in root.iterdir())[:20]
            listing.append(f"  {root}: {names}")
    raise FileNotFoundError(
        f"could not locate dataset {slug!r}"
        + (f" (owner {owner!r})" if owner else "")
        + (f" containing {must_contain!r}" if must_contain else "")
        + "\ntried:\n  " + "\n  ".join(str(p) for p in tried)
        + ("\nwhat is actually mounted:\n" + "\n".join(listing) if listing else "")
    )


def resolve_all(spec: dict[str, dict]) -> dict[str, Path]:
    """Resolve several at once and report them together, for a notebook's first cell.

        resolve_all({
            "eyepacs": {"owner": "dreamer07", "must_contain": "data/data"},
            "aptos2019": {"owner": "mariaherrerot", "must_contain": "train_images"},
        })
    """
    out = {}
    for slug, kw in spec.items():
        out[slug] = resolve_input(slug, **kw)
    return out

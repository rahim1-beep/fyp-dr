"""Render the Phase 2 before/after contact sheet — the visual QA gate.

This exists to answer one question that no metric can answer: does circle-crop plus Ben
Graham at 224x224 PRESERVE the small lesions that separate grade 1 from grade 2, or does
it destroy them? If it destroys them, no amount of training fixes the result, and the
whole cache is wrong before a single epoch runs.

Requirements set by the user (PROGRESS.md, Phase 2):
  1. Original and processed side by side AT THE SAME DISPLAY SIZE, patient ID and grade
     labelled on every pair.
  2. Weighted toward grades 1 and 2 — at least 6 of ~20. The sample has 9.
  3. 2-3 deliberately bad inputs, because the web app will receive exactly those.

TWO THINGS THIS GETS RIGHT THAT AN OBVIOUS IMPLEMENTATION GETS WRONG. Both were caught
in code review, and both made the gate answer PASS regardless of the truth:

  A. THE TWO PANELS MUST SHOW THE RETINA AT THE SAME SCALE. Letterboxing the raw original
     into the panel renders its retina SMALLER than the processed panel's, because the
     processed image has already been cropped to the retina. Measured across the 20 QA
     images, the original's retina spanned 139-203 px against the processed 224 — every
     pair biased 0.62x-0.91x in preprocessing's favour. A lesion the pipeline destroyed
     would then be invisible in the reference too, and the comparison would look clean.
     So the original is square-cropped the same way, and drawn from its NATIVE pixels.

  B. THE PROCESSED PANEL MUST BE THE CACHED JPEG, not an in-memory recomputation.
     Training reads the file on disk. The q95 round-trip is not free — measured mean
     3.60/255 and max 64 inside the retina, and it lifts 39% of the masked surround off
     zero by ringing. Signing off on a recomputation approves an artefact nobody uses.

The third panel is a 3x zoom on the macula region of both, which is where the
microaneurysm question is actually decided; at full-frame scale a lesion is a couple of
pixels and neither panel can settle it.

Usage:
    python -m src.data.contact_sheet \
        --sample docs/phase2_qa_sample.csv \
        --src-root data/raw/qa \
        --cache-root data/processed/qa \
        --out docs/phase2_contact_sheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display on this machine and none on Kaggle
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocess import (  # noqa: E402
    cache_relpath,
    imread_unicode,
    load_preprocess_config,
    retina_mask,
    scan_quality,
    square_crop,
)

REPO = Path(__file__).resolve().parents[2]

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

# Fraction of the frame width used for the zoom panel. 1/3 of the retina at 3x fills it.
DETAIL_FRAC = 1.0 / 3.0


def fit(bgr: np.ndarray, size: int, interp: int | None = None) -> np.ndarray:
    """Resize a square image to `size`, choosing a sane interpolation by direction."""
    if interp is None:
        interp = cv2.INTER_AREA if bgr.shape[0] > size else cv2.INTER_NEAREST
    return cv2.resize(bgr, (size, size), interpolation=interp)


def detail_panel(orig_sq: np.ndarray, proc: np.ndarray, size: int) -> np.ndarray:
    """A 3x zoom of the same central region from both, original above processed.

    Centred on the retina's centre, which after `square_crop` is the frame centre. The
    macula sits near there and is where clinically significant lesions concentrate. A
    fixed region keeps the panel reproducible; picking the "most interesting" region per
    image would make the sheet a different comparison for every row.

    Each half is cropped 2:1 so the two stack into a square without distorting aspect.
    """
    half = size // 2
    strips = []
    for img in (orig_sq, proc):
        n = img.shape[0]
        w = max(2, int(n * DETAIL_FRAC))
        h = max(1, w // 2)
        y0, x0 = (n - h) // 2, (n - w) // 2
        crop = img[y0:y0 + h, x0:x0 + w]
        interp = cv2.INTER_AREA if crop.shape[1] > size else cv2.INTER_NEAREST
        strips.append(cv2.resize(crop, (size, half), interpolation=interp))

    panel = np.vstack(strips)
    cv2.line(panel, (0, half), (size, half), (60, 60, 60), 1)
    return panel


def render(
    sample: pd.DataFrame,
    src_root: Path,
    cache_root: Path,
    out_path: Path,
    cfg,
    *,
    panel_px: int = 224,
    cols: int = 3,
    dpi: int = 130,
) -> pd.DataFrame:
    """Render the sheet. Returns per-image stats gathered while rendering.

    `cols` counts TRIPLES per row, so a row holds `3 * cols` panels.
    """
    n = len(sample)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols * 3,
        figsize=(cols * 3 * 1.95, rows * 2.72),
        dpi=dpi,
    )
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("white")

    records = []
    for i, row in enumerate(sample.itertuples(index=False)):
        r, c = divmod(i, cols)
        ax_o, ax_p, ax_d = axes[r, c * 3], axes[r, c * 3 + 1], axes[r, c * 3 + 2]
        for ax in (ax_o, ax_p, ax_d):
            ax.set_xticks([]); ax.set_yticks([])

        dataset = getattr(row, "dataset", "eyepacs")
        src = src_root / row.image_path
        cached = cache_root / cache_relpath(row.image_path, dataset)

        orig = imread_unicode(src) if src.exists() else None
        proc = imread_unicode(cached) if cached.exists() else None

        if orig is None or proc is None:
            missing = "SOURCE MISSING" if orig is None else "NOT IN CACHE"
            for ax in (ax_o, ax_p, ax_d):
                ax.text(0.5, 0.5, missing, ha="center", va="center",
                        color="crimson", fontsize=9, transform=ax.transAxes)
            records.append({"image": row.image, "status": missing.lower().replace(" ", "-")})
            continue

        q = scan_quality(orig)

        # Same crop the pipeline applies, so both panels show the retina at one scale.
        mask = retina_mask(orig)
        orig_sq = square_crop(orig, mask)[0] if mask.any() else orig

        ax_o.imshow(cv2.cvtColor(fit(orig_sq, panel_px), cv2.COLOR_BGR2RGB))
        ax_p.imshow(cv2.cvtColor(fit(proc, panel_px), cv2.COLOR_BGR2RGB))
        ax_d.imshow(cv2.cvtColor(detail_panel(orig_sq, proc, panel_px), cv2.COLOR_BGR2RGB))

        flags = q.flags()
        is_poor = getattr(row, "category", "clean") == "poor_quality"
        edge = "crimson" if (flags or is_poor) else "0.75"
        lw = 2.4 if (flags or is_poor) else 0.8
        for ax in (ax_o, ax_p, ax_d):
            for s in ax.spines.values():
                s.set_edgecolor(edge); s.set_linewidth(lw)

        grade = int(row.label)
        gname = GRADE_NAMES.get(grade, f"UNKNOWN GRADE {grade}")
        ax_o.set_title(f"{row.image}\ngrade {grade} — {gname}", fontsize=7.5, pad=3)
        ax_p.set_title(f"processed — from cache\n{orig.shape[1]}x{orig.shape[0]} source",
                       fontsize=7.5, color="0.35", pad=3)
        ax_d.set_title("3× detail: orig / processed", fontsize=7.5, color="0.35", pad=3)

        caption = (f"bright {q.mean_brightness:.0f} · sat {q.saturation:.0f} · "
                   f"focus {q.laplacian_var:.1f}")
        if flags:
            caption += "\n⚑ " + ", ".join(flags)
        elif is_poor:
            caption += "\n(selected as poor by file size)"
        ax_o.set_xlabel(caption, fontsize=6.6,
                        color=("crimson" if flags else "0.4"), labelpad=2)

        rec = {"image": row.image, "label": grade, "status": "ok",
               "category": getattr(row, "category", ""),
               "src_w": orig.shape[1], "src_h": orig.shape[0],
               "cache_path": cache_relpath(row.image_path, dataset),
               "cache_bytes": cached.stat().st_size,
               "quality_flags": ";".join(flags)}
        rec.update(q.__dict__)
        records.append(rec)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        for k in range(3):
            axes[r, c * 3 + k].axis("off")

    g12 = int(sample["label"].isin([1, 2]).sum())
    poor = int((sample.get("category", pd.Series(dtype=str)) == "poor_quality").sum())
    fig.suptitle(
        "Phase 2 preprocessing QA — original (left) vs cached 224×224 JPEG (middle) vs 3× detail (right)\n"
        f"retina scale-matched across panels · middle and right panels read the ACTUAL CACHE FILE · "
        f"{cfg.enhancement} · q{cfg.jpeg_quality}\n"
        f"{len(sample)} images, all TRAIN split   ·   grades 1–2: {g12}   ·   "
        f"deliberately poor inputs: {poor}   ·   red border = quality-flagged",
        fontsize=10, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.958))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame.from_records(records)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", type=Path, default=REPO / "docs/phase2_qa_sample.csv")
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True,
                    help="the preprocessed cache; the sheet renders the REAL files")
    ap.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "docs/phase2_contact_sheet.png")
    ap.add_argument("--stats", type=Path, help="write per-image quality stats here")
    ap.add_argument("--cols", type=int, default=3, help="triples per row")
    args = ap.parse_args()

    cfg = load_preprocess_config(args.config)
    sample = pd.read_csv(args.sample, comment="#")

    # Requirement 2 is a property of the sample, not of the renderer. Check it here so a
    # regenerated sample that quietly loses its grade-1/2 weighting cannot slip through.
    g12 = int(sample["label"].isin([1, 2]).sum())
    if g12 < 6:
        print(f"ERROR: sample has {g12} grade-1/2 images; the user's requirement is >= 6.",
              file=sys.stderr)
        return 2

    unknown = sorted(set(sample["label"]) - set(GRADE_NAMES))
    if unknown:
        print(f"ERROR: sample has labels outside 0-4: {unknown}", file=sys.stderr)
        return 2

    # Order by grade so the 1-vs-2 comparison sits together, with the deliberately-bad
    # inputs last where they are easy to find.
    sample = sample.sort_values(
        ["category", "label", "image"], ascending=[False, True, True]
    ).reset_index(drop=True)

    print(f"sample  : {args.sample}  ({len(sample)} images, grades 1-2: {g12})")
    print(f"source  : {args.src_root}")
    print(f"cache   : {args.cache_root}")
    print(f"pipeline: circle_crop={cfg.circle_crop} enhancement={cfg.enhancement} "
          f"size={cfg.image_size}\n")

    stats = render(sample, args.src_root, args.cache_root, args.out, cfg, cols=args.cols)

    bad = stats[stats["status"] != "ok"]
    if len(bad):
        print(f"WARNING: {len(bad)} image(s) could not be rendered:")
        for r in bad.itertuples(index=False):
            print(f"    {r.image}: {r.status}")

    ok = stats[stats["status"] == "ok"]
    if len(ok):
        flagged = ok[ok["quality_flags"].astype(str) != ""]
        print(f"quality-flagged: {len(flagged)}/{len(ok)}")
        for r in flagged.itertuples(index=False):
            print(f"    {r.image} (grade {r.label}, {r.category or 'clean'}) "
                  f"-> {r.quality_flags}")

    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats, index=False, lineterminator="\n")
        print(f"\nstats  -> {args.stats}")

    print(f"sheet  -> {args.out}")
    return 0 if not len(bad) else 1


if __name__ == "__main__":
    sys.exit(main())

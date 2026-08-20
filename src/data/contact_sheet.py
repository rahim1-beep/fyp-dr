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

The originals are shown letterboxed into a square of the same pixel size as the processed
image rather than stretched. Stretching changes the aspect ratio between the two panels,
which makes the comparison dishonest — a lesion that looks displaced would be an artefact
of the figure, not of the preprocessing.

Usage:
    python -m src.data.contact_sheet \
        --sample docs/phase2_qa_sample.csv \
        --src-root data/raw/qa \
        --stats docs/phase2_qa_stats.csv \
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
    imread_unicode,
    load_preprocess_config,
    preprocess_image,
    scan_quality,
)

REPO = Path(__file__).resolve().parents[2]

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
    """Fit an image into a square of `size` on black, preserving aspect ratio.

    Used for the ORIGINAL panel only. The processed panel is already square.
    """
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)

    canvas = np.zeros((size, size, 3), dtype=bgr.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def render(
    sample: pd.DataFrame,
    src_root: Path,
    out_path: Path,
    cfg,
    *,
    panel_px: int = 224,
    cols: int = 4,
    dpi: int = 130,
) -> pd.DataFrame:
    """Render the sheet. Returns the per-image stats gathered while rendering.

    `cols` counts PAIRS per row, so a row holds `2 * cols` panels.
    """
    n = len(sample)
    rows = int(np.ceil(n / cols))

    # 2 axes per pair; the extra width per column keeps the caption readable.
    fig, axes = plt.subplots(
        rows, cols * 2,
        figsize=(cols * 2 * 2.05, rows * 2.62),
        dpi=dpi,
    )
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("white")

    records = []
    for i, row in enumerate(sample.itertuples(index=False)):
        r, c = divmod(i, cols)
        ax_o, ax_p = axes[r, c * 2], axes[r, c * 2 + 1]

        src = src_root / row.image_path
        bgr = imread_unicode(src) if src.exists() else None

        if bgr is None:
            for ax, t in ((ax_o, "MISSING"), (ax_p, "MISSING")):
                ax.text(0.5, 0.5, t, ha="center", va="center", color="crimson",
                        fontsize=11, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            records.append({"image": row.image, "status": "missing"})
            continue

        q = scan_quality(bgr)
        proc = preprocess_image(bgr, cfg)

        # Same display size for both panels — requirement 1.
        orig_panel = letterbox(bgr, panel_px)

        ax_o.imshow(cv2.cvtColor(orig_panel, cv2.COLOR_BGR2RGB))
        ax_p.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))

        flags = q.flags()
        is_poor = getattr(row, "category", "clean") == "poor_quality"

        # Colour the pair's border: red for a flagged/deliberately-bad input, grey
        # otherwise. Makes the 3 bad inputs findable at a glance on a 20-pair sheet.
        edge = "crimson" if (flags or is_poor) else "0.75"
        lw = 2.4 if (flags or is_poor) else 0.8
        for ax in (ax_o, ax_p):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(edge); s.set_linewidth(lw)

        grade = int(row.label)
        ax_o.set_title(
            f"{row.image}\ngrade {grade} — {GRADE_NAMES[grade]}",
            fontsize=7.5, color="black", pad=3,
        )
        ax_p.set_title(
            f"processed {cfg.image_size}x{cfg.image_size}\n"
            f"{bgr.shape[1]}x{bgr.shape[0]} source",
            fontsize=7.5, color="0.35", pad=3,
        )

        caption = f"bright {q.mean_brightness:.0f} · sat {q.saturation:.0f} · focus {q.laplacian_var:.0f}"
        if flags:
            caption += "\n⚑ " + ", ".join(flags)
        elif is_poor:
            caption += "\n(selected as poor by file size)"
        ax_o.set_xlabel(caption, fontsize=6.6, color=("crimson" if flags else "0.4"),
                        labelpad=2)

        rec = {"image": row.image, "label": grade, "status": "ok",
               "category": getattr(row, "category", ""),
               "src_w": bgr.shape[1], "src_h": bgr.shape[0],
               "quality_flags": ";".join(flags)}
        rec.update(q.__dict__)
        records.append(rec)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    g12 = int(sample["label"].isin([1, 2]).sum())
    poor = int((sample.get("category", pd.Series(dtype=str)) == "poor_quality").sum())
    fig.suptitle(
        f"Phase 2 preprocessing QA — original (left) vs processed (right), equal display size\n"
        f"circle-crop → {cfg.enhancement} → {cfg.image_size}×{cfg.image_size} JPEG q{cfg.jpeg_quality}   ·   "
        f"{len(sample)} images, all TRAIN split   ·   grades 1–2: {g12}   ·   "
        f"deliberately poor inputs: {poor}   ·   red border = quality-flagged",
        fontsize=10.5, y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
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
    ap.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "docs/phase2_contact_sheet.png")
    ap.add_argument("--stats", type=Path, help="write per-image quality stats here")
    ap.add_argument("--cols", type=int, default=4, help="pairs per row")
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

    # Order the sheet by grade so the 1-vs-2 comparison sits together, with the
    # deliberately-bad inputs last where they are easy to find.
    sample = sample.sort_values(
        ["category", "label", "image"], ascending=[False, True, True]
    ).reset_index(drop=True)

    print(f"sample  : {args.sample}  ({len(sample)} images, grades 1-2: {g12})")
    print(f"source  : {args.src_root}")
    print(f"pipeline: circle_crop={cfg.circle_crop} enhancement={cfg.enhancement} "
          f"size={cfg.image_size}\n")

    stats = render(sample, args.src_root, args.out, cfg, cols=args.cols)

    missing = stats[stats["status"] != "ok"]
    if len(missing):
        print(f"WARNING: {len(missing)} image(s) not found under {args.src_root}:")
        for r in missing.itertuples(index=False):
            print(f"    {r.image}")

    ok = stats[stats["status"] == "ok"]
    if len(ok):
        flagged = ok[ok["quality_flags"].astype(str) != ""]
        print(f"\nquality-flagged: {len(flagged)}/{len(ok)}")
        for r in flagged.itertuples(index=False):
            print(f"    {r.image} (grade {r.label}, {r.category or 'clean'}) -> {r.quality_flags}")

    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats, index=False, lineterminator="\n")
        print(f"\nstats  -> {args.stats}")

    print(f"sheet  -> {args.out}")
    return 0 if not len(missing) else 1


if __name__ == "__main__":
    sys.exit(main())

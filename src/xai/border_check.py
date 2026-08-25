"""The Phase 5 border-artefact gate. Pre-registered thresholds — DECISION-046.

DECISION-016 chose 2.5% mask erosion over full 0.9r circular masking, and deferred the
question to here **with evidence**. So this is a gate with numbers fixed before any
heatmap was generated, not a figure someone looks at and pronounces fine.

THREE GATES, ALL OF WHICH MUST PASS.

    statistic                                          PASS      FAIL
    ------------------------------------------------   -------   -------
    median RIM mass ratio (outer 10% of retina)        < 1.5     >= 1.5
    median OUTSIDE mass ratio (beyond the disc)        < 0.5     >= 0.5
    median RIM mass ratio on grade 3-4 images only     < 1.5     >= 1.5
    median |corr(cam, cam_randomised)|                 < 0.5     >= 0.5

`mass ratio` = (share of CAM mass in a region) / (share of image area in that region).
A model ignoring a region scores about 1.0 there; one keying on it scores well above.

Why OUTSIDE is stricter: beyond the retinal disc there is literally nothing to see, so
mass there is unambiguous artefact rather than a judgement call.

Why grades 3-4 are separate: peripheral disease appears there, and the periphery is
precisely what erosion was chosen to preserve. A model keying on the rim *only* for
severe grades is a different and worse failure than a uniform rim bias, and a median over
all grades — 73.5% of which are grade 0 — would hide it.

Why RANDOMISATION is a gate and not an extra: a saliency map that barely changes when the
target layer's weights are randomised is measuring the image, not the model. If that is
true, every other number in this file is describing edge contrast and the chapter is
void. The check stands on its own logic regardless of attribution — it needs no citation
to be correct, and the one commonly given for it is unverified and deliberately not
repeated here.

A HIGH CAM RATIO IS NOT PROOF OF DEPENDENCE. Grad-CAM is correlational and high-contrast
boundaries attract activation for reasons that have nothing to do with the decision. That
is what `occlusion_delta` is for, and it is the first step of the documented failure path
(DECISION-047) rather than an afterthought.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import retina_mask

RIM_FRACTION = 0.10          # outer 10% of the retinal radius
RIM_MAX = 1.5
OUTSIDE_MAX = 0.5
RANDOMISATION_MAX = 0.5
SEVERE_GRADES = (3, 4)


def region_masks(bgr: np.ndarray, rim_fraction: float = RIM_FRACTION) -> dict:
    """`outside` / `rim` / `interior` boolean masks for one cached image.

    The rim is defined by the DISTANCE TRANSFORM of the retina mask rather than by a
    circle of radius 0.9r. Cached images are circle-cropped but not perfectly centred or
    perfectly circular, and a geometric circle would put chunks of genuine retina in the
    "rim" bucket for exactly the off-centre images most likely to have real vignetting —
    biasing the statistic in the direction it is meant to detect.
    """
    m = retina_mask(bgr) > 0
    if not m.any():
        raise ValueError("retina mask is empty; the image is black or the mask is broken")
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    rim = m & (dist <= dist.max() * rim_fraction)
    return {"outside": ~m, "rim": rim, "interior": m & ~rim}


def mass_ratio(cam: np.ndarray, region: np.ndarray) -> float:
    """(share of CAM mass in the region) / (share of image area in the region)."""
    cam = np.asarray(cam, float)
    total = cam.sum()
    area = region.mean()
    if total <= 0 or area <= 0:
        return 0.0
    return float((cam[region].sum() / total) / area)


def image_ratios(cam: np.ndarray, bgr: np.ndarray,
                 rim_fraction: float = RIM_FRACTION) -> dict:
    regions = region_masks(bgr, rim_fraction)
    return {k: mass_ratio(cam, v) for k, v in regions.items()}


def occlusion_delta(model, x, bgr_batch, *, region: str = "rim",
                   rim_fraction: float = RIM_FRACTION):
    """Does the DECISION actually depend on the rim? The causal half of the question.

    Replaces the rim annulus with the interior's mean colour — NOT with black, which is
    itself an out-of-distribution artefact and would confound the very thing being
    measured — then re-runs the model and returns the per-image change in the scalar
    score.

    Step F1 of the DECISION-047 failure path: if the CAM lights up on a region but the
    score barely moves when that region is replaced, the map is showing a correlate and
    there is nothing to rebuild.

    `region="outside"` asks the same question of the black surround. That one carries a
    specific worry: the EXTENT of the surround encodes the camera's field of view, which
    is site-specific, which can correlate with disease prevalence. A model reading it
    would be taking a shortcut that external validation punishes (DECISION-050).
    """
    import torch

    from src.xai.gradcam import scalar_target

    if region not in ("rim", "outside"):
        raise ValueError(f"region={region!r}; expected 'rim' or 'outside'")
    occluded = x.clone()
    for i, bgr in enumerate(bgr_batch):
        regions = region_masks(bgr, rim_fraction)
        rim = torch.as_tensor(regions[region], device=x.device)
        interior = torch.as_tensor(regions["interior"], device=x.device)
        if not interior.any():
            continue
        for c in range(occluded.shape[1]):
            plane = occluded[i, c]
            plane[rim] = plane[interior].mean()

    model.eval()
    with torch.no_grad():
        before = scalar_target(model(x)).cpu().numpy()
        after = scalar_target(model(occluded)).cpu().numpy()
    return after - before


def summarise(rows: list[dict], randomisation: list[float] | None = None) -> dict:
    """Medians and the pass/fail verdict for every gate. Never partially reports."""
    def med(key, keep=lambda r: True):
        vals = [r[key] for r in rows if keep(r)]
        return float(np.median(vals)) if vals else float("nan")

    severe = lambda r: int(r.get("label", -1)) in SEVERE_GRADES  # noqa: E731
    out = {
        "n_images": len(rows),
        "rim_median": med("rim"),
        "outside_median": med("outside"),
        "interior_median": med("interior"),
        "rim_median_severe": med("rim", severe),
        "n_severe": sum(1 for r in rows if severe(r)),
        "thresholds": {"rim_max": RIM_MAX, "outside_max": OUTSIDE_MAX,
                       "randomisation_max": RANDOMISATION_MAX},
    }
    if randomisation:
        out["randomisation_median_abs_corr"] = float(np.median(np.abs(randomisation)))

    gates = {
        "rim": out["rim_median"] < RIM_MAX,
        "outside": out["outside_median"] < OUTSIDE_MAX,
        # A sample with no severe images cannot clear a gate about severe images. NaN
        # would compare False here by accident; it is made explicit so an under-stratified
        # sample fails loudly instead of passing by omission.
        "rim_severe": (out["n_severe"] > 0
                       and out["rim_median_severe"] < RIM_MAX),
    }
    if randomisation:
        gates["randomisation"] = out["randomisation_median_abs_corr"] < RANDOMISATION_MAX
    out["gates"] = gates
    out["passed"] = all(gates.values())
    return out


def format_report(s: dict) -> str:
    t = s["thresholds"]
    L = ["BORDER-ARTEFACT GATE — thresholds pre-registered in DECISION-046", ""]
    L.append(f"  images analysed            {s['n_images']}  "
             f"(grade 3-4: {s['n_severe']})")
    L.append("")
    L.append(f"  {'statistic':<34}{'value':>9}{'threshold':>12}{'':>8}")
    def row(name, val, thr, ok, cmp="<"):
        L.append(f"  {name:<34}{val:>9.3f}{cmp + ' ' + format(thr, '.2f'):>12}"
                 f"{'  PASS' if ok else '  FAIL':>8}")
    g = s["gates"]
    row("median rim mass ratio", s["rim_median"], t["rim_max"], g["rim"])
    row("median outside mass ratio", s["outside_median"], t["outside_max"], g["outside"])
    row("median rim ratio, grades 3-4", s["rim_median_severe"], t["rim_max"],
        g["rim_severe"])
    if "randomisation" in g:
        row("median |corr| vs randomised", s["randomisation_median_abs_corr"],
            t["randomisation_max"], g["randomisation"])
    L.append("")
    L.append(f"  interior mass ratio (context, no gate)  {s['interior_median']:.3f}")
    L.append("")
    if s["passed"]:
        L.append("  GATE PASSED — the model is not keying on the border, and the")
        L.append("  saliency maps are a property of the model rather than the image.")
    else:
        failed = [k for k, v in g.items() if not v]
        L.append(f"  *** GATE FAILED: {', '.join(failed)} ***")
        if "randomisation" in failed:
            L.append("  The randomisation gate failing invalidates the OTHER numbers:")
            L.append("  fix the Grad-CAM implementation before believing any of them.")
            L.append("  This is a TOOLING failure, not a model failure. No rebuild.")
        else:
            L.append("  Follow the DECISION-047 failure path, starting with the")
            L.append("  occlusion test (F1). A high CAM ratio is not yet proof that the")
            L.append("  DECISION depends on the border — do not rebuild the cache until")
            L.append("  occlusion shows it does.")
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", type=Path, required=True,
                    help="JSON list of per-image ratio dicts from the Phase 5 notebook")
    ap.add_argument("--randomisation", type=Path,
                    help="JSON list of per-image cam correlations against the "
                         "randomised model")
    ap.add_argument("--out", type=Path, help="write the summary as JSON")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero if any gate fails")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    rows = json.loads(Path(args.ratios).read_text(encoding="utf-8"))
    rand = (json.loads(Path(args.randomisation).read_text(encoding="utf-8"))
            if args.randomisation else None)
    s = summarise(rows, rand)
    print(format_report(s))
    if args.out:
        Path(args.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    if args.gate and not s["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

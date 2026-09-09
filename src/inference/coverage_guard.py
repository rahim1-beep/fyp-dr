"""The input check that turns Phase 6's finding into a control (DECISION-060).

Phase 6 established that the model's score depends partly on **retina coverage** — how
much of the frame the retinal disc fills — and that the dependence is stronger on external
data than in-domain (G4, DECISION-059/065). That is a property of the camera and the
clinic, not of the eye.

A disclaimer paragraph warns about that in the abstract. This measures it on the actual
upload and says whether *this* image sits inside the framing the model was trained on. A
known, measurable failure mode should be controlled, not merely mentioned.

WHAT THIS IS NOT. It is not a quality score, not a diagnosis, and not a claim that an
in-range image is safe. It answers one narrow question: is the framing of this photograph
inside the range the training images occupied? Outside that range the model is
extrapolating along the exact axis Phase 6 showed it is sensitive to.

THE THRESHOLDS ARE MEASURED, NEVER CHOSEN. They come from a committed calibration
artefact produced by `python -m src.inference.calibrate_coverage` over the TRAINING split
(R4). There is no default range baked into this module and no fallback: without the
artefact `assess` raises. A guard with invented bounds would be worse than no guard,
because it would look like evidence.

WHY THE TRAINING SPLIT AND NOT VALIDATION. The question is what the model *saw while
learning*, so the reference is the images it was fitted on. Using validation would also
quietly involve an evaluation split in a deployment decision.

A NOTE ON CONSERVATISM. Training augmentation (rotate/scale/shift) moves coverage over
roughly 0.81x-1.15x of an image's own value — measured on a synthetic disc, DECISION-063 —
so the model has some tolerance beyond the raw range of the cached images. This guard
flags against the RAW training distribution, which therefore flags slightly sooner than
strictly necessary. That is the right direction to be wrong in for a screening prototype,
and it is stated rather than left implicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.xai.border_check import NoRetinaError, region_masks

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = REPO / "analysis" / "coverage_guard" / "calibration.json"

# Status values. Kept as plain strings so the API contract does not depend on importing
# this module, and so a frontend can switch on them without a shared enum.
OK = "ok"
LOW = "framing_below_training_range"
HIGH = "framing_above_training_range"
NO_RETINA = "no_retina_detected"


@dataclass(frozen=True)
class Calibration:
    """The measured coverage distribution of the training split."""

    split: str
    n_images: int
    n_no_retina: int
    low: float
    high: float
    low_pct: float
    high_pct: float
    median: float
    cache_fingerprint: str

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Calibration":
        path = Path(path) if path is not None else DEFAULT_CALIBRATION
        if not path.exists():
            raise FileNotFoundError(
                f"no coverage calibration at {path}. The guard refuses to run on invented "
                "bounds (R4). Produce it with:\n"
                "    python -m src.inference.calibrate_coverage "
                "--cache-root <cache> --split train\n"
                "on a machine that has the preprocessed cache — i.e. Kaggle."
            )
        d = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"calibration has unexpected key(s): {sorted(unknown)}")
        missing = known - set(d)
        if missing:
            raise ValueError(f"calibration is missing key(s): {sorted(missing)}")
        return cls(**d)


def retina_coverage(bgr: np.ndarray) -> float:
    """Fraction of the frame occupied by retina, for one BGR image.

    The SAME quantity Phase 6's diagnostic correlated against the predicted score
    (`1 - outside.mean()`), computed by the SAME function. If these two ever diverge the
    guard stops guarding the thing that was measured, so it deliberately does not
    reimplement the mask.

    Raises `NoRetinaError` when no illuminated disc is found — the caller decides what
    that means, because in Phase 5/6 it meant "exclude from a statistic" and in an upload
    path it means "this is probably not a fundus photograph".
    """
    return float((~region_masks(bgr)["outside"]).mean())


def assess(bgr: np.ndarray, calibration: Calibration | None = None) -> dict:
    """Judge one uploaded image against the training framing range.

    Returns a dict with `status`, `coverage`, the bounds it was judged against, and a
    `message` written for a person rather than a log. `status` is OK / LOW / HIGH /
    NO_RETINA.
    """
    cal = calibration if calibration is not None else Calibration.load()

    try:
        cov = retina_coverage(bgr)
    except NoRetinaError:
        return {
            "status": NO_RETINA,
            "coverage": None,
            "low": cal.low,
            "high": cal.high,
            "message": (
                "No retina was detected in this image. Either it is not a fundus "
                "photograph, or it is too dark or too damaged to locate the retinal "
                "disc. No grade is produced."
            ),
        }

    out = {"status": OK, "coverage": cov, "low": cal.low, "high": cal.high}
    if cov < cal.low:
        out["status"] = LOW
        out["message"] = (
            f"The retina fills {cov:.1%} of this image, below the {cal.low:.1%} that "
            f"{100 - cal.low_pct:.0f}% of training images exceeded. The model was shown "
            "very few images framed this widely, and its score is known to depend partly "
            "on framing (Phase 6), so treat any grade from this image with extra caution."
        )
    elif cov > cal.high:
        out["status"] = HIGH
        out["message"] = (
            f"The retina fills {cov:.1%} of this image, above the {cal.high:.1%} that "
            f"{cal.high_pct:.0f}% of training images stayed below. The image may be "
            "cropped more tightly than the model was trained on, and its score is known "
            "to depend partly on framing (Phase 6), so treat any grade with extra "
            "caution."
        )
    else:
        out["message"] = (
            f"Framing is within the range the model was trained on (retina fills "
            f"{cov:.1%}; training range {cal.low:.1%}-{cal.high:.1%})."
        )
    return out

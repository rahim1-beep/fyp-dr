"""The Phase 7 input check (DECISION-060), and the ways it could quietly stop guarding.

The guard exists because Phase 6 MEASURED a dependence on framing. So the things worth
testing are not "does it return a dict" but: does it refuse to run on invented bounds,
does it measure the SAME quantity Phase 6 measured, and does it reject a non-fundus image
rather than grading it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.coverage_guard import (HIGH, LOW, NO_RETINA, OK, Calibration,
                                          assess, retina_coverage)
from src.xai.border_check import NoRetinaError, region_masks

CAL = dict(split="train", n_images=24586, n_no_retina=3, low=0.60, high=0.85,
           low_pct=1.0, high_pct=99.0, median=0.74, cache_fingerprint="deadbeefdeadbeef")


def _disc(radius: int, size: int = 224) -> np.ndarray:
    """A BGR frame holding a retina-like disc of the given radius on black."""
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (60, 90, 140), -1)
    return img


# ----------------------------------------------------------------------------------
# It must not invent bounds
# ----------------------------------------------------------------------------------
def test_a_missing_calibration_raises_rather_than_defaulting(tmp_path):
    """A guard with made-up bounds is worse than no guard: it looks like evidence."""
    with pytest.raises(FileNotFoundError, match="invented bounds"):
        Calibration.load(tmp_path / "nope.json")


def test_a_calibration_missing_a_field_is_refused(tmp_path):
    p = tmp_path / "cal.json"
    partial = {k: v for k, v in CAL.items() if k != "high"}
    p.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        Calibration.load(p)


def test_a_calibration_with_a_stray_field_is_refused(tmp_path):
    """A renamed or extra key means the artefact and the reader disagree about what the
    numbers mean. Fail rather than silently ignore it."""
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({**CAL, "threshold": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected key"):
        Calibration.load(p)


def test_a_round_trip_through_json_preserves_the_bounds(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(CAL), encoding="utf-8")
    c = Calibration.load(p)
    assert (c.low, c.high, c.split) == (0.60, 0.85, "train")


# ----------------------------------------------------------------------------------
# It must measure what Phase 6 measured
# ----------------------------------------------------------------------------------
def test_coverage_is_the_same_quantity_phase6_correlated():
    """Phase 6 used `1 - region_masks(bgr)['outside'].mean()`. If the guard ever computes
    its own version, it stops guarding the thing that was actually measured."""
    img = _disc(90)
    assert retina_coverage(img) == pytest.approx(
        float((~region_masks(img)["outside"]).mean())
    )


def test_coverage_rises_with_disc_size():
    assert retina_coverage(_disc(60)) < retina_coverage(_disc(90)) < retina_coverage(_disc(110))


def test_a_frame_with_no_retina_raises_from_the_low_level_helper():
    with pytest.raises(NoRetinaError):
        retina_coverage(np.zeros((224, 224, 3), np.uint8))


# ----------------------------------------------------------------------------------
# The verdicts
# ----------------------------------------------------------------------------------
def _cal(low: float, high: float) -> Calibration:
    return Calibration(**{**CAL, "low": low, "high": high})


def test_an_in_range_image_passes():
    img = _disc(90)
    cov = retina_coverage(img)
    r = assess(img, _cal(cov - 0.05, cov + 0.05))
    assert r["status"] == OK
    assert r["coverage"] == pytest.approx(cov)


def test_a_widely_framed_image_is_flagged_low():
    img = _disc(90)
    cov = retina_coverage(img)
    r = assess(img, _cal(cov + 0.02, cov + 0.10))
    assert r["status"] == LOW
    assert "caution" in r["message"]


def test_a_tightly_cropped_image_is_flagged_high():
    img = _disc(90)
    cov = retina_coverage(img)
    r = assess(img, _cal(cov - 0.10, cov - 0.02))
    assert r["status"] == HIGH
    assert "caution" in r["message"]


def test_a_non_fundus_upload_is_rejected_rather_than_graded():
    """Somebody will upload a selfie. The guard doubles as an 'is this even a fundus
    photograph' check, and must say so instead of returning a coverage number."""
    r = assess(np.zeros((224, 224, 3), np.uint8), _cal(0.60, 0.85))
    assert r["status"] == NO_RETINA
    assert r["coverage"] is None
    assert "No grade is produced" in r["message"]


def test_every_verdict_carries_the_bounds_it_was_judged_against():
    """A flag with no stated threshold is unauditable — the reader cannot tell whether the
    image was unusual or the bounds were."""
    for img in (_disc(90), np.zeros((224, 224, 3), np.uint8)):
        r = assess(img, _cal(0.60, 0.85))
        assert r["low"] == 0.60 and r["high"] == 0.85
        assert r["message"]


def test_the_boundary_is_inclusive_so_an_exactly_typical_image_passes():
    img = _disc(90)
    cov = retina_coverage(img)
    assert assess(img, _cal(cov, cov))["status"] == OK


# ----------------------------------------------------------------------------------
# The calibration CLI, end to end
#
# It runs on Kaggle where the cache lives, so a mistake in it is normally discovered
# after a session has already been spent. Exercised here against a synthetic cache
# through the REAL main().
# ----------------------------------------------------------------------------------
def _synthetic_cache(root: Path, n: int = 120):
    """n cached fundus-like images with a spread of disc sizes, plus the manifest."""
    import pandas as pd

    (root / "eyepacs").mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        radius = int(70 + 40 * rng.random())          # a real spread of coverage
        cv2.imwrite(str(root / "eyepacs" / f"{i}_left.jpg"), _disc(radius))
        rows.append({"image_path": f"data/data/{i}_left.jpeg", "patient_id": str(i),
                     "eye": "left", "label": i % 5, "dataset": "eyepacs",
                     "split": "train"})
    return pd.DataFrame(rows)


def test_the_calibration_cli_produces_an_artefact_the_guard_accepts(tmp_path, monkeypatch):
    import src.inference.calibrate_coverage as cal

    root = tmp_path / "cache"
    df = _synthetic_cache(root)
    monkeypatch.setattr(cal, "load_split", lambda split: df)

    out = tmp_path / "calibration.json"
    monkeypatch.setattr("sys.argv", ["calibrate_coverage", "--cache-root", str(root),
                                     "--split", "train", "--out", str(out)])
    assert cal.main() == 0

    # the whole point: what it writes, the guard can read
    c = Calibration.load(out)
    assert c.split == "train"
    assert c.n_images == len(df)
    assert 0.0 < c.low < c.median < c.high < 1.0
    assert (c.low_pct, c.high_pct) == (1.0, 99.0)

    # and it actually separates framings, rather than emitting a degenerate range
    assert c.high - c.low > 0.05, "the measured range is too narrow to guard anything"
    assert assess(_disc(90), c)["status"] == OK


def test_the_cli_refuses_to_write_a_calibration_from_too_few_images(tmp_path, monkeypatch):
    """A guard would trust whatever this writes, so a run that measured almost nothing
    must fail loudly rather than emit tight bounds from a handful of images."""
    import src.inference.calibrate_coverage as cal

    root = tmp_path / "cache"
    df = _synthetic_cache(root, n=20)
    monkeypatch.setattr(cal, "load_split", lambda split: df)
    monkeypatch.setattr("sys.argv", ["calibrate_coverage", "--cache-root", str(root),
                                     "--out", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit, match="refusing to write a calibration"):
        cal.main()


def test_no_retina_images_are_counted_not_silently_dropped(tmp_path, monkeypatch):
    """DECISION-052: they have no coverage so they cannot enter a percentile, but the
    denominator has to stay honest."""
    import pandas as pd

    import src.inference.calibrate_coverage as cal

    root = tmp_path / "cache"
    df = _synthetic_cache(root)
    (root / "eyepacs" / "blank_left.jpg").parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "eyepacs" / "blank_left.jpg"), np.zeros((224, 224, 3), np.uint8))
    df = pd.concat([df, pd.DataFrame([{
        "image_path": "data/data/blank_left.jpeg", "patient_id": "blank", "eye": "left",
        "label": 0, "dataset": "eyepacs", "split": "train"}])], ignore_index=True)
    monkeypatch.setattr(cal, "load_split", lambda split: df)

    out = tmp_path / "c.json"
    monkeypatch.setattr("sys.argv", ["calibrate_coverage", "--cache-root", str(root),
                                     "--out", str(out)])
    assert cal.main() == 0
    c = Calibration.load(out)
    assert c.n_no_retina == 1
    assert c.n_images == len(df), "n_images must count every image, not just usable ones"

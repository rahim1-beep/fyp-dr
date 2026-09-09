"""The inference path, tested without a server (repo map, §6).

The tests that matter here are not "does it return a grade". They are the four ways a
deployed DR model silently stops matching the model that was evaluated:

  * the upload is preprocessed differently from the training cache
  * the channels get swapped
  * the normalisation constants drift from the model's own
  * the decision thresholds are re-derived at inference time instead of carried over

Each has its own test below, and each would leave a system that runs, returns plausible
grades, and is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.preprocess import load_preprocess_config, preprocess_image
from src.eval.thresholds import grades_from_scores
from src.inference.coverage_guard import NO_RETINA, OK, Calibration
from src.inference.predictor import (GRADE_NAMES, Deployment, Predictor, disclaimer,
                                     overlay_cam)
from src.models.factory import ModelConfig, build_model

REPO = Path(__file__).resolve().parents[1]

DEPLOY = dict(run_id="phase4_stage3_arm_e_efficientnet_b0", seed=42,
              arch="efficientnet_b0", head="ordinal_regression", num_outputs=1,
              image_size=224, cuts=[0.5, 1.62164, 2.17199, 3.272691],
              referral_threshold=0.9487713575363159,
              seed_val_qwk=0.7615, seed_val_sens_at_spec95=0.7464,
              reported_val_qwk=0.7570, reported_val_sens_at_spec95=0.7360,
              source="analysis/phase6_aptos/carried_over_rules.json")

CAL = dict(split="train", n_images=24586, n_no_retina=3, low=0.55, high=0.90,
           low_pct=1.0, high_pct=99.0, median=0.74, cache_fingerprint="0123456789abcdef")


def _fundus(size: int = 600, radius_frac: float = 0.45) -> np.ndarray:
    """A plausible raw upload: a bright disc on black, larger than 224 so the real
    pipeline has to crop and downscale it."""
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * radius_frac), (40, 70, 120), -1)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 18, img.shape, dtype=np.uint8)
    return cv2.add(img, cv2.bitwise_and(noise, noise, mask=(img.max(2) > 0).astype(np.uint8)))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """An untrained arm-E model saved in the project's checkpoint format. Its GRADES are
    meaningless; every test here is about the plumbing around them."""
    p = tmp_path_factory.mktemp("ckpt") / "best.pth"
    m = build_model(ModelConfig(arch="efficientnet_b0", num_outputs=1,
                                head="ordinal_regression", pretrained=False))
    torch.save({"epoch": 0, "val_qwk": 0.0, "state_dict": m.state_dict()}, p)
    return p


@pytest.fixture
def predictor(checkpoint) -> Predictor:
    return Predictor(deployment=Deployment(**DEPLOY), checkpoint=checkpoint,
                     calibration=Calibration(**CAL))


# ----------------------------------------------------------------------------------
# The manifest must be explicit
# ----------------------------------------------------------------------------------
def test_a_missing_deployment_manifest_raises(tmp_path):
    """Without it the predictor would have to invent thresholds — which is fitting on the
    user's own data."""
    with pytest.raises(FileNotFoundError, match="which checkpoint ships|WHICH checkpoint"):
        Deployment.load(tmp_path / "nope.json")


def test_unsorted_cut_points_are_refused(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({**DEPLOY, "cuts": [0.5, 2.2, 1.6, 3.3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not ascending"):
        Deployment.load(p)


def test_wrong_number_of_cut_points_is_refused(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({**DEPLOY, "cuts": [0.5, 1.6, 2.2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 4 cut points"):
        Deployment.load(p)


def test_a_manifest_round_trips(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(DEPLOY), encoding="utf-8")
    d = Deployment.load(p)
    assert d.seed == 42 and d.cuts == DEPLOY["cuts"]


# ----------------------------------------------------------------------------------
# The four silent failures
# ----------------------------------------------------------------------------------
def test_the_upload_goes_through_the_cache_s_own_preprocessing(predictor):
    """Not a lookalike resize — the actual function that built the training cache. If
    these ever differ the model is shown a different distribution than it was fitted on,
    and every reported number stops applying."""
    raw = _fundus()
    got = predictor.preprocess_only(raw)
    want = preprocess_image(raw, load_preprocess_config(REPO / "configs" / "base.yaml"))
    assert np.array_equal(got, want)
    assert got.shape == (224, 224, 3)


def test_bgr_becomes_rgb_exactly_once(predictor):
    """A model fed BGR trains, converges, and quietly underperforms. Asserted by feeding
    an image whose channels are unambiguous and checking the tensor's channel order."""
    bgr = np.zeros((224, 224, 3), np.uint8)
    bgr[:, :, 2] = 200                      # pure RED in BGR
    x = predictor._to_tensor(bgr)[0]
    denorm = x * torch.tensor(predictor.std).view(3, 1, 1) + \
        torch.tensor(predictor.mean).view(3, 1, 1)
    assert denorm[0].mean() > 0.7, "channel 0 should be red after BGR->RGB"
    assert denorm[2].mean() < 0.1, "channel 2 should be blue after BGR->RGB"


def test_normalisation_comes_from_the_model_not_a_constant(predictor):
    """`normalisation(model)` reads timm's default_cfg. Hardcoding ImageNet numbers works
    until the backbone changes, then silently shifts the input distribution."""
    from src.models.factory import normalisation
    assert (predictor.mean, predictor.std) == normalisation(predictor.model)


def test_the_grade_uses_the_carried_over_cuts_and_nothing_else(predictor):
    """The mapping from score to grade must be reproducible from the manifest alone."""
    r = predictor.predict(_fundus(), explain=False)
    expected = int(grades_from_scores(np.array([r["score"]]), DEPLOY["cuts"])[0])
    assert r["grade"] == expected
    assert r["cuts"] == DEPLOY["cuts"]


# ----------------------------------------------------------------------------------
# The guard is wired in, on the right image
# ----------------------------------------------------------------------------------
def test_the_guard_judges_the_preprocessed_image_not_the_raw_upload(predictor):
    """The calibration was measured over the PREPROCESSED cache. Judging a 600px original
    against bounds derived from 224px crops would produce a plausible number that means
    nothing."""
    raw = _fundus()
    from src.inference.coverage_guard import retina_coverage
    r = predictor.predict(raw, explain=False)
    assert r["guard"]["coverage"] == pytest.approx(
        retina_coverage(predictor.preprocess_only(raw))
    )


def test_a_non_fundus_upload_is_declined_without_a_grade(predictor):
    r = predictor.predict(np.zeros((600, 600, 3), np.uint8), explain=False)
    assert r["graded"] is False
    assert r["guard"]["status"] == NO_RETINA
    assert "grade" not in r


def test_a_flagged_but_gradeable_image_still_returns_a_grade(checkpoint):
    """A framing flag qualifies the result; it does not withhold it. Withholding would
    push the user to an interface that simply doesn't mention the problem."""
    tight = Calibration(**{**CAL, "low": 0.95, "high": 0.99})
    p = Predictor(deployment=Deployment(**DEPLOY), checkpoint=checkpoint,
                  calibration=tight)
    r = p.predict(_fundus(), explain=False)
    assert r["graded"] is True and r["guard"]["status"] != OK
    assert isinstance(r["grade"], int)


# ----------------------------------------------------------------------------------
# What the interface is required to carry
# ----------------------------------------------------------------------------------
def test_every_result_carries_the_disclaimer_and_the_run_id(predictor):
    for img in (_fundus(), np.zeros((600, 600, 3), np.uint8)):
        r = predictor.predict(img, explain=False)
        assert len(r["disclaimer"]) == 4
        assert r["run_id"] == DEPLOY["run_id"]


def test_the_disclaimer_states_all_four_required_points():
    """DECISION-060 names them specifically, because a generic 'for research only' line
    does not tell a reader that the thresholds will not transfer."""
    text = " ".join(disclaimer()).lower()
    assert "not a medical device" in text
    assert "0.7360" in text and "0.80" in text          # the in-domain shortfall, named
    assert "framing" in text                            # the measured defect
    assert "recalibration" in text                      # thresholds do not transfer


# ----------------------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------------------
def test_gradcam_is_produced_at_the_image_size_and_in_range(predictor):
    cam = predictor.gradcam(predictor.preprocess_only(_fundus()))
    assert cam.shape == (224, 224)
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0


def test_predict_can_skip_the_explanation(predictor):
    assert "cam" not in predictor.predict(_fundus(), explain=False)
    assert "cam" in predictor.predict(_fundus(), explain=True)


def test_the_overlay_is_displayable_rgb(predictor):
    proc = predictor.preprocess_only(_fundus())
    out = overlay_cam(proc, predictor.gradcam(proc))
    assert out.shape == proc.shape and out.dtype == np.uint8


def test_encoded_bytes_and_arrays_take_the_same_path(predictor):
    raw = _fundus()
    ok, buf = cv2.imencode(".png", raw)
    assert ok
    a = predictor.predict(raw, explain=False)
    b = predictor.predict(buf.tobytes(), explain=False)
    assert a["grade"] == b["grade"]
    assert a["score"] == pytest.approx(b["score"], abs=1e-5)


def test_undecodable_bytes_raise_rather_than_returning_a_grade(predictor):
    with pytest.raises(ValueError, match="did not decode"):
        predictor.predict(b"this is not an image", explain=False)


def test_grade_names_line_up_with_the_five_grades(predictor):
    r = predictor.predict(_fundus(), explain=False)
    assert r["grade_name"] == GRADE_NAMES[r["grade"]]
    assert len(GRADE_NAMES) == 5


# ----------------------------------------------------------------------------------
# The shipped seed is the best-scoring of the three, so this is structural
# ----------------------------------------------------------------------------------
def test_the_interface_quotes_the_three_seed_mean_not_the_shipped_seed_s_own_score():
    """DECISION-042: report the mean across seeds, never the best seed. Seed 42 IS the
    best seed, so the guard against quoting 0.7464 has to be a test, not an intention."""
    text = " ".join(disclaimer())
    assert "0.7360" in text, "the 3-seed mean sensitivity must be the quoted figure"
    assert "0.7464" not in text, "the shipped seed's own sensitivity must not be quoted"


def test_provenance_states_both_numbers_side_by_side(tmp_path):
    """Anyone reading the interface should be able to see the shipped seed's own score
    and the reported mean together, rather than having to trust that they differ."""
    p = tmp_path / "d.json"
    p.write_text(json.dumps(DEPLOY), encoding="utf-8")
    line = Deployment.load(p).provenance()
    assert "0.7570" in line and "0.7615" in line and "seed 42" in line

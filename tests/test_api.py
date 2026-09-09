"""The HTTP layer. It must add nothing and hide nothing.

Two things are being checked. First that the transport is honest: an upload reaches the
predictor unchanged and the answer comes back whole, including the guard verdict and the
disclaimer. Second that the app REFUSES TO START when a required artefact is missing —
a server that boots and grades without a calibrated guard is the failure this whole
phase exists to prevent.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import build_predictor, create_app
from src.inference.coverage_guard import Calibration
from src.inference.predictor import Deployment, Predictor
from src.models.factory import ModelConfig, build_model

DEPLOY = dict(run_id="phase4_stage3_arm_e_efficientnet_b0", seed=42,
              arch="efficientnet_b0", head="ordinal_regression", num_outputs=1,
              image_size=224, cuts=[0.5, 1.62164, 2.17199, 3.272691],
              referral_threshold=0.9487713575363159,
              seed_val_qwk=0.7614671328146116, seed_val_sens_at_spec95=0.7464,
              reported_val_qwk=0.7569412145586636, reported_val_sens_at_spec95=0.7360,
              source="test")

CAL = dict(split="train", n_images=24586, n_no_retina=3, low=0.55, high=0.90,
           low_pct=1.0, high_pct=99.0, median=0.74, cache_fingerprint="0123456789abcdef")


def _fundus_png(size: int = 600) -> bytes:
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.45), (40, 70, 120), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    p = tmp_path_factory.mktemp("ckpt") / "best.pth"
    m = build_model(ModelConfig(arch="efficientnet_b0", num_outputs=1,
                                head="ordinal_regression", pretrained=False))
    torch.save({"epoch": 0, "val_qwk": 0.0, "state_dict": m.state_dict()}, p)
    predictor = Predictor(deployment=Deployment(**DEPLOY), checkpoint=p,
                          calibration=Calibration(**CAL))
    return TestClient(create_app(predictor))


# ----------------------------------------------------------------------------------
# It must refuse to start rather than serve without its artefacts
# ----------------------------------------------------------------------------------
def test_no_checkpoint_env_var_refuses_to_start(monkeypatch):
    """A default path would eventually load some other run's weights and report them
    under this run's provenance."""
    monkeypatch.delenv("FYP_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="FYP_CHECKPOINT is not set"):
        build_predictor()


def test_a_checkpoint_path_that_does_not_exist_refuses_to_start(monkeypatch, tmp_path):
    monkeypatch.setenv("FYP_CHECKPOINT", str(tmp_path / "absent.pth"))
    with pytest.raises(RuntimeError, match="does not exist"):
        build_predictor()


# ----------------------------------------------------------------------------------
# The transport
# ----------------------------------------------------------------------------------
def test_health_names_the_running_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["run_id"] == DEPLOY["run_id"]


def test_meta_reports_the_three_seed_mean_and_not_the_shipped_seed(client):
    """DECISION-069: the interface quotes the mean, never seed 42's own numbers. The
    frontend renders /meta, so this endpoint is where that promise is kept or broken."""
    m = client.get("/meta").json()
    assert m["reported_val_qwk"] == pytest.approx(0.7569, abs=5e-5)
    assert m["reported_val_sens_at_spec95"] == pytest.approx(0.7360, abs=5e-5)
    text = " ".join(m["disclaimer"])
    assert "0.7360" in text and "0.7464" not in text
    assert len(m["disclaimer"]) == 4


def test_meta_exposes_the_coverage_bounds_being_enforced(client):
    """A guard whose thresholds are not visible cannot be audited by whoever reads a
    flagged result."""
    c = client.get("/meta").json()["coverage_calibration"]
    assert c["low"] == CAL["low"] and c["high"] == CAL["high"]
    assert c["split"] == "train"


def test_predict_returns_a_grade_the_guard_verdict_and_the_disclaimer(client):
    r = client.post("/predict", files={"file": ("eye.png", _fundus_png(), "image/png")},
                    params={"explain": False})
    assert r.status_code == 200
    body = r.json()
    assert body["graded"] is True
    assert body["grade"] in range(5)
    assert body["guard"]["status"]
    assert len(body["disclaimer"]) == 4
    assert body["run_id"] == DEPLOY["run_id"]


def test_explain_returns_the_overlay_and_the_image_that_was_actually_graded(client):
    """The user should see the preprocessed 224px crop, not the original they uploaded —
    the original is not what the model saw."""
    r = client.post("/predict", files={"file": ("eye.png", _fundus_png(), "image/png")})
    body = r.json()
    for key in ("overlay_png", "preprocessed_png"):
        assert body[key].startswith("data:image/png;base64,")
        raw = base64.b64decode(body[key].split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (224, 224, 3)
    assert "cam" not in body, "the raw float array should not be shipped over HTTP"


def test_a_non_fundus_upload_comes_back_declined_not_graded(client):
    blank = np.zeros((600, 600, 3), np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    r = client.post("/predict", files={"file": ("x.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200, "declining is a normal outcome, not an HTTP error"
    assert r.json()["graded"] is False
    assert "grade" not in r.json()


def test_a_file_that_is_not_an_image_is_a_400_not_a_500(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "did not decode" in r.json()["detail"]


def test_an_empty_upload_is_rejected(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_an_oversized_upload_is_rejected_before_decoding(client):
    from backend.app import MAX_UPLOAD_BYTES
    r = client.post("/predict",
                    files={"file": ("big.png", b"\x00" * (MAX_UPLOAD_BYTES + 1),
                                    "image/png")})
    assert r.status_code == 413


def test_the_api_adds_no_grading_logic_of_its_own(client):
    """The HTTP answer must match what the predictor returns directly. If they ever
    diverge, a rule has escaped the layer that is actually tested."""
    png = _fundus_png()
    direct = client.app.state.predictor.predict(png, explain=False)
    over_http = client.post("/predict", files={"file": ("e.png", png, "image/png")},
                            params={"explain": False}).json()
    assert over_http["grade"] == direct["grade"]
    assert over_http["score"] == pytest.approx(direct["score"], abs=1e-6)
    assert over_http["referable"] == direct["referable"]

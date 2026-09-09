"""FastAPI over `src.inference.predictor`. Transport only — no inference logic here.

Everything that decides a grade lives in the predictor and is tested without a server
(`tests/test_predictor.py`). This module's job is to accept an upload, hand it over,
and shape the answer as JSON. If a rule about grading ever appears in this file, it has
escaped the layer that is actually tested.

STARTUP IS DELIBERATELY FRAGILE. The app loads the deployment manifest, the coverage
calibration and the checkpoint at import time and **refuses to start** if any is missing.
A server that boots and then returns 500 for every upload is harder to diagnose than one
that will not boot and says which file is absent — and a server that boots with a
*missing calibration* would be worse still, because the guard would be the thing silently
absent while grades kept coming.

THE CHECKPOINT IS NOT IN THE REPOSITORY. `*.pth` is gitignored (CLAUDE.md §7), so its
location comes from `FYP_CHECKPOINT`. There is no default path to fall back to, because
a fallback would eventually load some other run's weights and report them under this
run's provenance.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference.coverage_guard import Calibration
from src.inference.predictor import Deployment, Predictor, disclaimer, overlay_cam

REPO = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def build_predictor() -> Predictor:
    """Resolve everything the predictor needs, or fail with the reason."""
    ckpt = os.environ.get("FYP_CHECKPOINT")
    if not ckpt:
        raise RuntimeError(
            "FYP_CHECKPOINT is not set. Checkpoints are gitignored, so the app cannot "
            "guess where best.pth lives, and a default path would eventually load some "
            "other run's weights under this run's provenance. Set it to the best.pth of "
            "the run named in analysis/deployment/deployment.json."
        )
    if not Path(ckpt).exists():
        raise RuntimeError(f"FYP_CHECKPOINT={ckpt} does not exist")

    deployment = Deployment.load()          # raises if the manifest is absent
    calibration = Calibration.load()        # raises if the guard is uncalibrated
    return Predictor(deployment=deployment, checkpoint=Path(ckpt),
                     calibration=calibration)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """`predictor` is injectable so the tests exercise the real routes without a
    checkpoint on disk."""
    app = FastAPI(
        title="DR grading — research prototype",
        description="NOT a medical device. See /meta for the full disclaimer.",
        version="0.1.0",
    )
    # The Next.js dev server is a different origin. Wide open is acceptable for a local
    # demo and would not be for anything reachable.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    app.state.predictor = predictor if predictor is not None else build_predictor()

    @app.get("/health")
    def health() -> dict:
        p: Predictor = app.state.predictor
        return {"status": "ok", "run_id": p.deployment.run_id}

    @app.get("/meta")
    def meta() -> dict:
        """What is running and what is claimed. The frontend renders this rather than
        holding its own copy of the numbers."""
        p: Predictor = app.state.predictor
        d = p.deployment
        return {
            "run_id": d.run_id,
            "seed": d.seed,
            "arch": d.arch,
            "head": d.head,
            "image_size": d.image_size,
            "provenance": d.provenance(),
            "reported_val_qwk": d.reported_val_qwk,
            "reported_val_sens_at_spec95": d.reported_val_sens_at_spec95,
            "disclaimer": disclaimer(),
            "coverage_calibration": {
                "split": p.calibration.split,
                "n_images": p.calibration.n_images,
                "low": p.calibration.low,
                "high": p.calibration.high,
            },
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), explain: bool = True) -> dict:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload is {len(raw)} bytes; the limit is {MAX_UPLOAD_BYTES}",
            )
        p: Predictor = app.state.predictor
        try:
            result = p.predict(raw, explain=explain)
        except ValueError as e:
            # A file that is not an image is the user's mistake, not a server fault.
            raise HTTPException(status_code=400, detail=str(e)) from e

        # The CAM is a float array; ship it as a PNG data URI over the preprocessed
        # image, which is what the user should be shown — the original is not what was
        # graded.
        if "cam" in result:
            proc = p.preprocess_only(raw)
            result["overlay_png"] = _png_data_uri(overlay_cam(proc, result.pop("cam")))
            result["preprocessed_png"] = _png_data_uri(
                proc[:, :, ::-1]                       # BGR -> RGB for display
            )
        return result

    return app


def _png_data_uri(rgb: np.ndarray) -> str:
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])    # cv2 writes BGR
    if not ok:
        raise RuntimeError("failed to encode the overlay as PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# `uvicorn backend.app:app` — built at import, so a misconfigured deployment fails at
# startup rather than on the first upload.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)

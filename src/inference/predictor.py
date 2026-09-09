"""One uploaded image -> one graded, guarded, explained result. No HTTP anywhere.

The repo map puts this module under the FastAPI layer on purpose: the backend should be a
thin transport over something that can be tested, scripted and demonstrated without a
server. Everything here takes bytes or arrays and returns plain Python.

=============================================================================
 THE THING THAT SILENTLY BREAKS A DEPLOYED DR MODEL
=============================================================================

An uploaded photograph must travel through the **same pipeline that built the training
cache**, or the model is being shown a different distribution than it was fitted on and
nothing about its reported performance transfers. So this module calls
`src.data.preprocess.preprocess_image` — the actual function — rather than reimplementing
resize-and-crop. There is no second copy of that logic to drift out of step.

Three specific ways it can go wrong, each guarded here:

1. **Channel order.** The cache is BGR on disk and `DRDataset` converts to RGB exactly
   once before normalising. This does the same, in one place, for the same reason: a
   model fed BGR runs happily and quietly underperforms.
2. **Normalisation.** Constants come from the model's own `default_cfg` via
   `normalisation(model)`, never hardcoded and never computed from data.
3. **Where the coverage guard runs.** The calibration is measured over the PREPROCESSED
   cache, so the guard must judge the PREPROCESSED image. Running it on the raw upload
   would compare a 3000px original against bounds derived from 224px crops — the numbers
   would look plausible and mean nothing.

=============================================================================
 THE DECISION RULES ARE CARRIED OVER, NEVER RE-DERIVED
=============================================================================

Arm E emits one continuous score. Turning that into a grade needs four cut points, and
the referral flag needs an operating point. Both were fitted on the EyePACS VALIDATION
split (R3) and are carried into deployment unchanged, exactly as Phase 6 carried them to
APTOS. They are loaded from a committed deployment manifest; nothing here re-optimises
anything, because a threshold fitted at inference time is a threshold fitted on the user's
own data.

`DECISION-060` requires the interface to state that these thresholds are NOT transferable:
a new population needs local recalibration. `disclaimer()` returns that text so the app
and this module cannot drift apart on the wording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.preprocess import PreprocessConfig, load_preprocess_config, preprocess_image
from src.eval.thresholds import grades_from_scores
from src.inference.coverage_guard import NO_RETINA, Calibration, assess
from src.models.factory import ModelConfig, build_model, load_checkpoint, normalisation
from src.xai.gradcam import GradCAM, scalar_target

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT = REPO / "analysis" / "deployment" / "deployment.json"

GRADE_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
REFERABLE_FROM = 2


@dataclass(frozen=True)
class Deployment:
    """Exactly which model ships, and the decision rules it ships with.

    Committed as JSON so the answer to "which checkpoint produced this grade" is a file
    in the repository rather than a variable in someone's notebook.
    """

    run_id: str
    seed: int
    arch: str
    head: str
    num_outputs: int
    image_size: int
    cuts: list[float]
    referral_threshold: float
    # This seed's OWN validation figures, for provenance.
    seed_val_qwk: float
    seed_val_sens_at_spec95: float
    # The 3-SEED MEANS, which are what the interface reports. Both are recorded because
    # the shipped seed is the highest-scoring of the three (DECISION-069): keeping its
    # own number next to the number actually quoted makes cherry-picking structurally
    # visible rather than a matter of trust. DECISION-042 -- report the mean, never the
    # best seed -- applies to what is claimed, not to which file is loaded.
    reported_val_qwk: float
    reported_val_sens_at_spec95: float
    source: str

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Deployment":
        path = Path(path) if path is not None else DEFAULT_DEPLOYMENT
        if not path.exists():
            raise FileNotFoundError(
                f"no deployment manifest at {path}. It records WHICH checkpoint ships "
                "and the validation-fitted cut points it ships with; without it the "
                "predictor would have to invent thresholds, which is fitting on the "
                "user's own data (R3/R4)."
            )
        d = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        if set(d) != known:
            raise ValueError(
                f"deployment manifest keys {sorted(set(d) ^ known)} do not match the "
                "expected schema"
            )
        if len(d["cuts"]) != 4:
            raise ValueError(f"expected 4 cut points, got {len(d['cuts'])}")
        if list(d["cuts"]) != sorted(d["cuts"]):
            raise ValueError(f"cut points are not ascending: {d['cuts']}")
        return cls(**d)

    def provenance(self) -> str:
        """One line for the interface: what is running, and what is being claimed."""
        return (f"{self.run_id} (seed {self.seed}). Reported performance is the 3-seed "
                f"mean: QWK {self.reported_val_qwk:.4f}, referable sensitivity "
                f"{self.reported_val_sens_at_spec95:.4f} at 95% specificity. This seed's "
                f"own validation QWK is {self.seed_val_qwk:.4f}.")


def disclaimer() -> list[str]:
    """The four points DECISION-060 requires the interface to carry.

    Returned as data, not prose, so the app renders one source of truth and a reviewer
    can diff it. The numbers are the committed 3-seed validation means.
    """
    return [
        "Research prototype. NOT a medical device, and not for clinical use.",
        "Below the screening reference in-domain: referable sensitivity 0.7360 at 95% "
        "specificity, against a 0.80 reference that is itself provisional.",
        "Demonstrated dependence on image framing, stronger on external data than on the "
        "training distribution. Present in two of three training runs.",
        "The decision thresholds are NOT transferable. They were fitted on EyePACS; any "
        "use on a new population requires local recalibration.",
    ]


@dataclass
class Predictor:
    """Load once, call `predict` many times. CPU by default — the app is a demo."""

    deployment: Deployment
    checkpoint: Path
    calibration: Calibration | None = None
    preprocess: PreprocessConfig = field(default_factory=lambda: load_preprocess_config(
        REPO / "configs" / "base.yaml"))
    device: str = "cpu"

    def __post_init__(self) -> None:
        d = self.deployment
        self.model = build_model(ModelConfig(arch=d.arch, num_outputs=d.num_outputs,
                                             head=d.head, pretrained=False))
        load_checkpoint(self.checkpoint, self.model)
        self.mean, self.std = normalisation(self.model)
        self.model.eval().to(self.device)
        if self.calibration is None:
            self.calibration = Calibration.load()

    # -- the pipeline, in the order it must happen ---------------------------------
    def _decode(self, image: bytes | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("the uploaded bytes did not decode as an image")
        return bgr

    def _to_tensor(self, bgr224: np.ndarray) -> torch.Tensor:
        """Preprocessed BGR -> normalised NCHW, converting to RGB exactly once."""
        rgb = cv2.cvtColor(bgr224, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        return ((x - mean) / std).unsqueeze(0)

    def predict(self, image: bytes | np.ndarray, *, explain: bool = True) -> dict:
        """Grade one image, or decline to.

        Returns `graded: False` with the guard's reason when no retina is found. Every
        other result carries the guard verdict alongside the grade rather than instead of
        it, so a flagged-but-gradeable image still produces a number the user can see is
        qualified.
        """
        d = self.deployment
        bgr = self._decode(image)
        proc = preprocess_image(bgr, self.preprocess)     # the cache's own pipeline

        # On the PREPROCESSED image: the calibration was measured over the cache.
        guard = assess(proc, self.calibration)
        if guard["status"] == NO_RETINA:
            return {"graded": False, "guard": guard, "run_id": d.run_id,
                    "disclaimer": disclaimer()}

        x = self._to_tensor(proc).to(self.device)
        with torch.no_grad():
            score = float(scalar_target(self.model(x)).cpu().item())

        grade = int(grades_from_scores(np.array([score]), d.cuts)[0])
        out = {
            "graded": True,
            "grade": grade,
            "grade_name": GRADE_NAMES[grade],
            "score": score,
            "referable": bool(score >= d.referral_threshold),
            "referral_threshold": d.referral_threshold,
            "cuts": list(d.cuts),
            "guard": guard,
            "run_id": d.run_id,
            "disclaimer": disclaimer(),
        }
        if explain:
            out["cam"] = self.gradcam(proc)
        return out

    def gradcam(self, bgr224: np.ndarray) -> np.ndarray:
        """(H, W) attention map in [0, 1] for an already-preprocessed image."""
        x = self._to_tensor(bgr224).to(self.device)
        with GradCAM(self.model) as cam:
            return cam(x).cam[0]

    def preprocess_only(self, image: bytes | np.ndarray) -> np.ndarray:
        """The cache-identical 224px BGR the model actually sees. Exposed so the app can
        show the user what was graded, rather than the original they uploaded."""
        return preprocess_image(self._decode(image), self.preprocess)


def overlay_cam(bgr224: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Heatmap over the preprocessed image, returned as RGB uint8 for direct display."""
    if cam.shape != bgr224.shape[:2]:
        cam = cv2.resize(cam, (bgr224.shape[1], bgr224.shape[0]))
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(bgr224, 1.0 - alpha, heat, alpha, 0.0)
    return cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)

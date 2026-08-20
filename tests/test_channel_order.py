"""Settles the BGR/RGB question end to end: source file -> cache -> tensor -> model.

WHY THIS FILE EXISTS SEPARATELY. A channel swap is the highest-cost silent bug in this
project. Nothing raises, nothing looks wrong, the loss falls, the model converges — and
it underperforms for a reason that appears in no log. It is a documented route to a DR
project sitting at 20-45% accuracy. Individual unit tests in `test_preprocess.py` and
`test_dataset.py` each check one hop; this file checks the whole chain against an
independent reference decoder, because a swap applied consistently in two places would
pass every single-hop test.

THE ANSWER, stated once so nobody has to re-derive it:

  * `cv2` works in **BGR** arrays. `cv2.imwrite` takes a BGR array and writes a
    CORRECT jpeg — it converts on the way out. So the files in the cache are ordinary
    images, not "BGR files". There is no such thing as a BGR JPEG.
  * `src/data/preprocess.py` reads BGR, works in BGR, writes with `cv2.imwrite`.
    Consistent, and the file on disk is right.
  * `src/data/dataset.py::load_cached_image` reads with cv2 (BGR) and converts to
    **RGB exactly once**, at line 1 of the loader. Everything downstream is RGB.
  * `timm`'s `default_cfg` mean/std are **RGB** and channel-asymmetric
    (0.485/0.456/0.406), so the order is load-bearing: a swap shifts every input by a
    different amount per channel.

The independent reference is **PIL**, which decodes to RGB natively and shares no code
with cv2. If our loader and PIL agree pixel for pixel, the chain is right.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import DRDataset, load_cached_image
from src.data.preprocess import (
    imwrite_unicode,
    load_preprocess_config,
    preprocess_image,
    process_manifest,
)
from src.models.factory import ModelConfig, build_model, normalisation

REPO = Path(__file__).resolve().parents[1]
CFG = load_preprocess_config(REPO / "configs" / "base.yaml")
GEOM_ONLY = dataclasses.replace(CFG, enhancement="none")

QA_DIR = REPO / "data" / "raw" / "qa" / "data" / "data"
needs_qa = pytest.mark.skipif(
    not QA_DIR.is_dir() or not any(QA_DIR.glob("*.jpeg")),
    reason="QA images absent; run src.data.fetch_sample",
)


def _red_dominant_fundus(w: int = 600, h: int = 600) -> np.ndarray:
    """A BGR array of a fundus-like disc: red channel highest, blue lowest.

    Real fundus images are strongly red-dominant — that is what the retina looks like —
    so this is the property to track through the pipeline. If it inverts, the chain
    swapped somewhere.
    """
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(min(w, h) * 0.45), (40, 90, 210), -1)  # BGR
    return img


# ----------------------------------------------------------------------------------
# 1. The loader against an independent decoder
# ----------------------------------------------------------------------------------

def test_our_loader_agrees_with_pil_pixel_for_pixel(tmp_path):
    """PIL decodes to RGB natively and shares no code with cv2. Exact agreement settles
    the question without relying on either library's convention being remembered right."""
    PIL = pytest.importorskip("PIL.Image")

    f = tmp_path / "x.jpg"
    cv2.imwrite(str(f), _red_dominant_fundus(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    ours = load_cached_image(f)
    theirs = np.array(PIL.open(f).convert("RGB"))

    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs), (
        "our loader disagrees with PIL — one of them is not RGB"
    )


def test_a_deliberately_swapped_loader_would_fail_that_test(tmp_path):
    """Proof the test above can actually fail. A test that passes for both orders would
    be worthless, and that is exactly the trap with symmetric fixtures."""
    PIL = pytest.importorskip("PIL.Image")

    f = tmp_path / "x.jpg"
    cv2.imwrite(str(f), _red_dominant_fundus(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    swapped = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)  # BGR, unconverted
    theirs = np.array(PIL.open(f).convert("RGB"))
    assert not np.array_equal(swapped, theirs)


# ----------------------------------------------------------------------------------
# 2. The whole chain: source file -> preprocess -> cache file -> Dataset tensor
# ----------------------------------------------------------------------------------

def test_red_dominance_survives_source_to_tensor(tmp_path):
    """The end-to-end statement. Runs the real batch driver, writes a real cache file,
    reads it back through the real Dataset, and checks the retina is still red-dominant
    in the tensor the model would receive.

    Uses `enhancement="none"`: Ben Graham subtracts the LOCAL MEAN, so a flat disc
    correctly cancels to grey 128 in every channel and carries no colour at all. Testing
    hue through it would assert nothing. The geometry-only path is where a cvtColor slip
    would actually live.
    """
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "data" / "data").mkdir(parents=True)
    imwrite_unicode(src / "data" / "data" / "1_left.jpeg", _red_dominant_fundus(), 100)

    df = pd.DataFrame({"image_path": ["data/data/1_left.jpeg"], "label": [0],
                       "dataset": ["eyepacs"], "patient_id": ["1"], "split": ["train"]})
    stats = process_manifest(df, src, out, GEOM_ONLY, scan=False, progress=False)
    assert stats.loc[0, "status"] == "ok"

    ds = DRDataset(df, out, train=False, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                   image_size=CFG.image_size)
    img, _, _ = ds[0]                       # [3, 224, 224], RGB, unnormalised here

    centre = img[:, 96:128, 96:128].reshape(3, -1).mean(dim=1)
    r, g, b = (float(v) for v in centre)
    assert r > g > b, f"channel order broke somewhere in the chain: R={r:.3f} G={g:.3f} B={b:.3f}"
    assert r - b > 0.4, f"red dominance collapsed: R={r:.3f} B={b:.3f}"


def test_the_cache_file_itself_is_a_correct_image(tmp_path):
    """`cv2.imwrite` takes a BGR array and writes a correct JPEG — there is no such thing
    as a 'BGR file'. Confirmed by decoding the written file with PIL."""
    PIL = pytest.importorskip("PIL.Image")

    bgr = _red_dominant_fundus()
    out = preprocess_image(bgr, GEOM_ONLY)
    f = tmp_path / "cached.jpg"
    imwrite_unicode(f, out, 100)

    rgb = np.array(PIL.open(f).convert("RGB"))
    centre = rgb[96:128, 96:128].reshape(-1, 3).mean(axis=0)
    assert centre[0] > centre[1] > centre[2], (
        f"the cache file is not a correct RGB image: {centre.round(1)}"
    )


# ----------------------------------------------------------------------------------
# 3. The tensor matches what timm expects
# ----------------------------------------------------------------------------------

def test_the_dataset_normalises_with_timms_rgb_constants(tmp_path):
    """The last hop. `normalisation(model)` is RGB and channel-asymmetric, so feeding it
    BGR would shift each channel by a different, wrong amount."""
    model = build_model(ModelConfig(arch="resnet18", pretrained=False))
    mean, std = normalisation(model)

    src, out = tmp_path / "src", tmp_path / "out"
    (src / "data" / "data").mkdir(parents=True)
    imwrite_unicode(src / "data" / "data" / "1_left.jpeg", _red_dominant_fundus(), 100)
    df = pd.DataFrame({"image_path": ["data/data/1_left.jpeg"], "label": [0],
                       "dataset": ["eyepacs"], "patient_id": ["1"], "split": ["train"]})
    process_manifest(df, src, out, GEOM_ONLY, scan=False, progress=False)

    ds = DRDataset(df, out, train=False, mean=mean, std=std)
    img, _, _ = ds[0]

    # Undo the normalisation and the red disc must reappear in channel 0.
    m = np.array(mean).reshape(3, 1, 1)
    s = np.array(std).reshape(3, 1, 1)
    back = img.numpy() * s + m
    centre = back[:, 96:128, 96:128].reshape(3, -1).mean(axis=1)
    assert centre[0] > centre[1] > centre[2], f"not RGB at the model boundary: {centre.round(3)}"


def test_a_swap_at_the_model_boundary_would_be_visible(tmp_path):
    """Quantifies why this matters rather than asserting it. With ImageNet constants a
    swapped input is off by (0.485-0.406)/0.229 ~= 0.35 sigma on red and the reverse on
    blue — a systematic, per-channel shift on every image the model ever sees."""
    mean, std = normalisation(build_model(ModelConfig(arch="resnet18", pretrained=False)))
    shift_r = abs(mean[0] - mean[2]) / std[0]
    assert shift_r > 0.2, (
        "if this ever becomes ~0 the normalisation went symmetric and a swap would be "
        "undetectable; revisit every assumption in this file"
    )


# ----------------------------------------------------------------------------------
# 4. On a real fundus image
# ----------------------------------------------------------------------------------

@needs_qa
def test_a_real_eyepacs_image_is_red_dominant_end_to_end(tmp_path):
    """Synthetic fixtures can encode the same mistake twice. A real retina is
    unambiguously red-dominant, and no amount of consistent-but-wrong plumbing makes it
    look blue."""
    path = sorted(QA_DIR.glob("*.jpeg"))[0]

    from src.data.preprocess import imread_unicode
    bgr = imread_unicode(path)
    assert bgr is not None

    out = preprocess_image(bgr, GEOM_ONLY)
    f = tmp_path / "c.jpg"
    imwrite_unicode(f, out, 95)

    rgb = load_cached_image(f)
    lit = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 10
    means = [float(rgb[..., c][lit].mean()) for c in range(3)]
    assert means[0] > means[1] and means[0] > means[2], (
        f"{path.name}: a real fundus must be red-dominant, got R/G/B {np.round(means, 1)}"
    )

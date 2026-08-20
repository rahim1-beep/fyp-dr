"""Tests for src/data/preprocess.py.

These lock in the properties that are expensive to rediscover, each of which cost real
debugging time during Phase 2:

  - the fast large-sigma blur really does approximate the exact one (it is a 145x
    speed-up, and without it the full cache takes ~340 hours)
  - the focus measure is resolution-independent (the naive version flagged 19 of 20
    images as blurred)
  - preprocessing never drops an image, so the cache can never quietly disagree with
    the split CSVs
  - the cache layout is flat and split-agnostic (leakage-auditor condition 3)
  - nothing computes a statistic across images (leakage-auditor condition 1)

Synthetic fixtures where possible so the suite runs without the 35 GB dataset. Tests
that need real fundus images are skipped unless data/raw/qa/ is populated.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preprocess import (
    PreprocessConfig,
    QualityStats,
    cache_relpath,
    large_sigma_blur,
    load_preprocess_config,
    preprocess_image,
    process_manifest,
    retina_mask,
    scan_quality,
    square_crop,
)

REPO = Path(__file__).resolve().parents[1]
QA_DIR = REPO / "data" / "raw" / "qa"

needs_qa = pytest.mark.skipif(
    not (QA_DIR / "data" / "data").is_dir()
    or not any((QA_DIR / "data" / "data").glob("*.jpeg")),
    reason="QA images absent; run `python -m src.data.fetch_sample "
           "--manifest docs/phase2_qa_sample.csv --out data/raw/qa`",
)


def synthetic_fundus(w=1200, h=900, r=None, cx=None, cy=None, seed=0) -> np.ndarray:
    """A fake fundus image: a bright textured disc on black, optionally truncated."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    r = r or min(h, w) // 2
    cx = cx if cx is not None else w // 2
    cy = cy if cy is not None else h // 2

    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (cx, cy), r, 255, -1)
    texture = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    img[disc > 0] = texture[disc > 0]
    return img


# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------

def test_base_config_loads_and_matches_the_documented_pipeline():
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    assert cfg.image_size == 224, "DECISION-005 fixes headline results at 224x224"
    assert cfg.cache_format == "jpeg", "R5 forbids a monolithic array cache"
    assert cfg.enhancement in {"ben_graham", "clahe_green", "none"}


def test_unknown_config_key_is_rejected(tmp_path):
    """A typo'd key must fail loudly rather than silently doing nothing."""
    p = tmp_path / "c.yaml"
    p.write_text("preprocess:\n  image_size: 224\n  ben_graham_alpha_typo: 4\n")
    with pytest.raises(KeyError, match="unknown key"):
        load_preprocess_config(p)


def test_npy_cache_format_is_rejected():
    """R5 is enforced in code, not just in prose."""
    with pytest.raises(ValueError, match="R5"):
        PreprocessConfig(cache_format="npy")


# ----------------------------------------------------------------------------------
# The fast blur — the single biggest performance decision in Phase 2
# ----------------------------------------------------------------------------------

def test_large_sigma_blur_approximates_the_exact_blur():
    """Re-measures the accuracy claim rather than trusting the docstring's table.

    Tolerance is on the FINAL Ben Graham output, where alpha=4 amplifies any blur error
    fourfold — that is the number that actually matters.
    """
    img = synthetic_fundus(w=1200, h=900, seed=1)
    sigma = img.shape[1] / 10.0

    exact = cv2.GaussianBlur(img, (0, 0), sigma)
    fast = large_sigma_blur(img, sigma)

    a, b, g = 4.0, -4.0, 128.0
    fe = cv2.addWeighted(img, a, exact, b, g)
    ff = cv2.addWeighted(img, a, fast, b, g)
    diff = np.abs(fe.astype(np.int16) - ff.astype(np.int16))

    assert diff.max() <= 8, f"max final-image error {diff.max()} exceeds JPEG-noise scale"
    assert diff.mean() < 1.0


def test_large_sigma_blur_is_exact_for_small_sigma():
    """Below the working scale it must defer to cv2 rather than resample pointlessly."""
    img = synthetic_fundus(w=300, h=300, seed=2)
    got = large_sigma_blur(img, 4.0, work_sigma=48.0)
    want = cv2.GaussianBlur(img, (0, 0), 4.0)
    assert np.array_equal(got, want)


def test_large_sigma_blur_is_much_faster_than_exact():
    """The whole reason this function exists. Exact is ~30 s/image at full resolution."""
    import time

    img = synthetic_fundus(w=2000, h=1500, seed=3)
    sigma = img.shape[1] / 10.0

    t0 = time.perf_counter(); large_sigma_blur(img, sigma); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); cv2.GaussianBlur(img, (0, 0), sigma); t_exact = time.perf_counter() - t0

    assert t_exact / max(t_fast, 1e-6) > 10, (
        f"speed-up only {t_exact / max(t_fast, 1e-6):.1f}x; the full cache build "
        "depends on this staying large"
    )


# ----------------------------------------------------------------------------------
# Cropping and masking
# ----------------------------------------------------------------------------------

def test_square_crop_produces_a_square():
    img = synthetic_fundus(w=1200, h=900)
    out, msk = square_crop(img, retina_mask(img))
    assert out.shape[0] == out.shape[1]
    assert out.shape[:2] == msk.shape[:2], "image and mask must stay in registration"


def test_square_crop_handles_a_vertically_truncated_retina():
    """The common EyePACS case: the disc is wider than the frame is tall.

    The bbox-then-pad approach left black bars inside the frame, so the retina's scale
    in the output depended on the camera's aspect ratio. The retina must fill a
    consistent fraction of the output instead.
    """
    img = synthetic_fundus(w=1200, h=700, r=600)  # disc taller than the frame
    out, msk = square_crop(img, retina_mask(img))

    assert out.shape[0] == out.shape[1]
    frac = (msk > 0).mean()
    assert frac > 0.55, f"retina fills only {frac:.2f} of the square crop"


def test_preprocess_output_is_the_configured_size_and_masked():
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    out = preprocess_image(synthetic_fundus(), cfg)

    assert out.shape == (cfg.image_size, cfg.image_size, 3)
    assert out.dtype == np.uint8
    # The corners sit outside any retina, so they must be masked to black — otherwise
    # Ben Graham leaves them at ~gamma (128) as a constant frame around every image.
    for cy, cx in ((2, 2), (2, -3), (-3, 2), (-3, -3)):
        assert out[cy, cx].max() == 0, "corner not masked; expect a constant bright frame"


def test_ben_graham_does_not_ring_the_retina_edge():
    """The halo regression test.

    Without a normalised convolution the blur near the boundary is dragged toward the
    black surround, and `alpha*img - alpha*blur` blows the rim out to brighter than any
    lesion. Measured at 1.44x inner brightness before the fix.
    """
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    out = preprocess_image(synthetic_fundus(w=1400, h=1400), cfg)

    n = cfg.image_size
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(float)

    edge = grey[(rr > 0.85) & (rr < 0.97)].mean()
    inner = grey[rr < 0.5].mean()
    assert edge / inner < 1.15, f"edge/inner brightness {edge / inner:.2f}: halo is back"


def test_preprocessing_is_deterministic():
    """R6. No RNG anywhere in the pipeline; the same bytes in give the same bytes out."""
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    img = synthetic_fundus(seed=7)
    assert np.array_equal(preprocess_image(img, cfg), preprocess_image(img, cfg))


def test_a_black_frame_still_produces_an_image():
    """Degenerate inputs must survive to be SEEN and flagged, not raise.

    21 of 35,126 EyePACS files are under 50 KB and some are nearly black. They stay in
    the splits, so preprocessing has to emit something for every one of them.
    """
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    out = preprocess_image(np.zeros((400, 500, 3), np.uint8), cfg)
    assert out.shape == (cfg.image_size, cfg.image_size, 3)


# ----------------------------------------------------------------------------------
# scan_quality
# ----------------------------------------------------------------------------------

def test_focus_measure_is_resolution_independent():
    """The bug that flagged 19 of 20 images as blurred.

    A Laplacian is a per-pixel operator, so on the same scene a higher-resolution capture
    scores LOWER — adjacent pixels are more correlated. Measured natively, one 4.9 MP QA
    image scored 152.0 while a 16.1 MP one scored 1.9. The same scene at two resolutions
    must score comparably or no fixed threshold can mean anything.
    """
    small = synthetic_fundus(w=800, h=800, seed=11)
    big = cv2.resize(small, (3200, 3200), interpolation=cv2.INTER_CUBIC)

    f_small = scan_quality(small).laplacian_var
    f_big = scan_quality(big).laplacian_var

    ratio = max(f_small, f_big) / max(min(f_small, f_big), 1e-6)
    assert ratio < 4.0, (
        f"focus {f_small:.1f} vs {f_big:.1f} for the same scene at 4x scale "
        f"(ratio {ratio:.1f}) — the measure is still resolution-dependent"
    )


def test_flag_thresholds_separate_good_from_degraded():
    """Threshold behaviour only.

    NOT a leakage test, despite what an earlier version of this name claimed: two
    synthetic points on opposite sides of every threshold would pass even if `flags()`
    consulted a dataset-wide percentile table. The real leakage property is checked by
    test_flags_reads_no_module_state below.
    """
    good = QualityStats(120.0, 0.75, 0.0, 0.01, 0.01, 120.0, 40.0, 30.0, 0.0)
    bad = QualityStats(15.0, 0.05, 0.0, 0.80, 0.35, 10.0, 0.5, 5.0, 0.5)

    assert good.flags() == []
    for expected in ("tiny-retina", "dark", "mostly-black", "off-centre",
                     "washed-out", "blurred", "low-contrast", "bright-artefact"):
        assert expected in bad.flags()

    # clipped_fraction is 0.0 in both fixtures above, so exercise it explicitly rather
    # than leaving the over-exposed branch untested.
    blown = QualityStats(200.0, 0.75, 0.40, 0.0, 0.01, 120.0, 40.0, 30.0, 0.0)
    assert "over-exposed" in blown.flags()


def test_flags_reads_no_module_state():
    """leakage-auditor condition 1, checked at the source level.

    `flags()` must be a pure function of `self`. If it ever reads a module-level table —
    a dataset percentile, a cached calibration — then val and test begin influencing
    which train images are flagged. Names loaded inside the function body are inspected
    directly, because a behavioural test cannot distinguish a constant from a lookup that
    happens to agree.
    """
    import ast
    import inspect

    src = textwrap.dedent(inspect.getsource(QualityStats.flags))
    tree = ast.parse(src)

    import builtins

    # Builtins appear via the `-> list[str]` annotation and are not state.
    allowed = {"self", "f"} | set(dir(builtins))
    loaded = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert loaded <= allowed, (
        f"flags() reads non-local names {sorted(loaded - allowed)} — a dataset-derived "
        "threshold would be a leakage vector"
    )

    # Every comparison must be against a literal.
    for cmp_node in (n for n in ast.walk(tree) if isinstance(n, ast.Compare)):
        for comparator in cmp_node.comparators:
            assert isinstance(comparator, ast.Constant), (
                f"flags() compares against {ast.dump(comparator)}, not a literal constant"
            )


def test_black_frame_quality_is_reported_not_crashed():
    q = scan_quality(np.zeros((300, 400, 3), np.uint8))
    assert q.retina_fraction == 0.0
    assert "mostly-black" in q.flags()


# ----------------------------------------------------------------------------------
# Cache layout — leakage-auditor condition 3
# ----------------------------------------------------------------------------------

def test_cache_layout_is_flat_and_carries_no_split_name():
    """A split-shaped cache silently mismatches the CSVs after any re-split."""
    p = cache_relpath("data/data/10_left.jpeg", "eyepacs")
    assert p == "eyepacs/10_left.jpg"
    for part in ("train", "val", "test"):
        assert part not in p.split("/")


def test_aptos_cache_path_discards_the_author_split_and_is_namespaced():
    """DECISION-004 discards the author-provided APTOS split; its directory structure IS
    that split, so it must not survive into the cache. The prefix also stops hash-named
    APTOS files colliding with EyePACS `{patientID}_{eye}` names."""
    p = cache_relpath("train_images/train_images/1ae8c165fd53.png", "aptos")
    assert p == "aptos/aptos_1ae8c165fd53.jpg"
    assert "train_images" not in p


def test_cache_relpath_is_idempotent_on_an_already_prefixed_id():
    assert cache_relpath("aptos_abc.png", "aptos") == "aptos/aptos_abc.jpg"


# ----------------------------------------------------------------------------------
# Batch driver — the no-silent-drop guarantee
# ----------------------------------------------------------------------------------

def test_missing_images_are_recorded_never_skipped(tmp_path):
    """The cache must never quietly disagree with the split CSVs.

    A dropped row would desynchronise the cache from the manifests and, for val or test,
    would silently rebalance them — an R2 violation by omission. Failures are recorded
    with a status and reported, and `main` exits non-zero.
    """
    import pandas as pd

    src, out = tmp_path / "src", tmp_path / "out"
    (src / "data" / "data").mkdir(parents=True)
    real = src / "data" / "data" / "1_left.jpeg"
    cv2.imwrite(str(real), synthetic_fundus(w=400, h=400))

    df = pd.DataFrame({
        "image_path": ["data/data/1_left.jpeg", "data/data/999_gone.jpeg"],
        "label": [0, 3],
        "dataset": ["eyepacs", "eyepacs"],
    })
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    stats = process_manifest(df, src, out, cfg, progress=False)

    assert len(stats) == len(df), "every input row must appear in the output"
    assert set(stats["status"]) == {"ok", "missing"}
    assert stats.loc[stats["status"] == "missing", "image_path"].tolist() == [
        "data/data/999_gone.jpeg"
    ]


@needs_qa
def test_real_qa_images_round_trip():
    """End-to-end on genuine fundus photographs, when they are present locally."""
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    paths = sorted((QA_DIR / "data" / "data").glob("*.jpeg"))[:4]
    assert paths

    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        out = preprocess_image(img, cfg)
        assert out.shape == (cfg.image_size, cfg.image_size, 3)
        assert out.max() > 0, f"{p.name} preprocessed to an entirely black image"


@needs_qa
def test_the_three_known_bad_qa_images_are_flagged():
    """The pixel statistics must independently agree with the file-size selection.

    These three were picked as the smallest files in the dataset, before any pixel
    statistic existed. scan_quality shares no inputs with file size, so agreement is a
    real cross-check on both.
    """
    cfg_dir = QA_DIR / "data" / "data"
    for name in ("3829_left", "39106_left", "15942_right"):
        p = cfg_dir / f"{name}.jpeg"
        if not p.exists():
            pytest.skip(f"{name} not downloaded")
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        assert scan_quality(img).flags(), f"{name} is known-degraded but was not flagged"

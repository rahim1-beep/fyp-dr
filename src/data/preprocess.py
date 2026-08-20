"""EyePACS / APTOS image preprocessing: circle-crop -> enhancement -> 224x224 JPEG.

The pipeline, in order:

  1. `retina_mask`   — the REAL illuminated region, not an idealised circle. EyePACS
                       retinas are routinely truncated top and bottom by the sensor.
  2. `square_crop`   — a square of side 2r centred on the retina, so the retina fills the
                       same fraction of every output whatever the camera's aspect ratio.
  3. `enhance`       — `ben_graham` (default) or `clahe_green` (ablation arm), selected by
                       `preprocess.enhancement` in the config. Never both. Ben Graham's
                       local average is a NORMALISED CONVOLUTION over the mask; without
                       that it rings the retina in a halo brighter than any lesion.
  4. resize to `preprocess.image_size` (224, DECISION-005), re-mask, and write ONE JPEG.

R5: individual JPEGs, never a monolithic `.npy`.

EVERY OPERATION HERE IS PER-IMAGE. Nothing in this module computes a statistic across
images, so preprocessing cannot move information between splits. That property is
load-bearing. In particular, do NOT add a "compute dataset mean/std" step here — that
would fold test pixels into training. Normalisation belongs in the Dataset transform,
taken from timm's `model.default_cfg` (configs/base.yaml says so explicitly).

CACHE LAYOUT — flat and split-agnostic, on the leakage-auditor's instruction:

    {out_root}/eyepacs/{patientID}_{eye}.jpg
    {out_root}/aptos/aptos_{id_code}.jpg

Deliberately NOT `train/`, `val/`, `test/` subdirectories. Split membership is resolved
only from `data/splits/*.csv` at `Dataset` construction time; a split-shaped cache would
silently mismatch the CSVs after any re-split, and no current test would catch it. The
APTOS source path (`train_images/train_images/...`) is flattened for the same reason — it
encodes the author-provided split that DECISION-004 discards. The `aptos_` prefix keeps
hash-named APTOS files from ever colliding with EyePACS `{patientID}_{eye}` names.

Usage:
    # QA sample only (the Phase 2 gate)
    python -m src.data.preprocess --manifest docs/phase2_qa_sample.csv \
        --src-root data/raw/qa --out-root data/processed/qa \
        --stats docs/phase2_qa_stats.csv

    # Full cache (Kaggle, ONLY after the contact sheet is signed off)
    python -m src.data.preprocess --split train --split val --split test \
        --config configs/kaggle.yaml --src-root /kaggle/input/eyepacs \
        --out-root /kaggle/working/processed
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """Exactly the `preprocess:` block of configs/base.yaml. Built from the config file
    rather than from defaults, so a run's settings are always traceable (R6)."""

    circle_crop: bool = True
    enhancement: str = "ben_graham"
    ben_graham_sigma_scale: float = 10.0
    ben_graham_alpha: float = 4.0
    ben_graham_beta: float = -4.0
    ben_graham_gamma: float = 128.0
    image_size: int = 224
    jpeg_quality: int = 95
    cache_format: str = "jpeg"

    def __post_init__(self) -> None:
        if self.enhancement not in {"ben_graham", "clahe_green", "none"}:
            raise ValueError(
                f"preprocess.enhancement={self.enhancement!r} is not one of "
                "'ben_graham', 'clahe_green', 'none'"
            )
        if self.cache_format != "jpeg":
            raise ValueError(
                f"preprocess.cache_format={self.cache_format!r}; R5 requires individual "
                "JPEGs. A monolithic array cache is forbidden."
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality={self.jpeg_quality} out of range")


def load_preprocess_config(path: Path) -> PreprocessConfig:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    block = cfg.get("preprocess")
    if block is None:
        raise KeyError(f"{path} has no `preprocess:` block")
    unknown = set(block) - set(PreprocessConfig.__dataclass_fields__)
    if unknown:
        # A typo'd key that silently does nothing is how a config drifts away from the
        # behaviour it claims to describe.
        raise KeyError(f"{path} preprocess block has unknown key(s): {sorted(unknown)}")
    return PreprocessConfig(**block)


# ----------------------------------------------------------------------------------
# Step 1 - circle crop
# ----------------------------------------------------------------------------------

def retina_mask(bgr: np.ndarray, blur_ksize: int = 15, thresh: int = 10) -> np.ndarray:
    """uint8 0/255 mask of the illuminated retinal disc.

    Blur before thresholding: raw fundus images carry sensor noise in the black surround
    that a bare `gray > 0` test happily reports as retina, which makes the bounding box
    the whole frame and the crop a no-op. The blur is the difference between this working
    and this only appearing to work.

    The morphological close/open then fills specular speckle inside the retina and drops
    isolated hot pixels outside it, so the mask is one solid region. That matters because
    the mask is used as the domain of a normalised convolution below — a pinhole in the
    mask becomes a visible artefact in the output, not merely a cosmetic flaw.

    NOTE the mask is the REAL illuminated region, not an idealised circle. EyePACS
    retinas are routinely truncated top and bottom by the sensor, so an inscribed circle
    would include black regions that were never imaged.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    mask = ((gray > thresh).astype(np.uint8)) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def square_crop(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square of side 2r centred on the retina, where r is its larger half-extent.

    Returns (image, mask) cropped together so they stay in registration.

    Why not the plain bounding box padded to square: EyePACS retinas are usually cut off
    top and bottom, so the bbox is wider than it is tall. Padding that to square leaves
    BLACK BARS inside the frame, the retina no longer fills a consistent fraction of the
    output, and the scale of a lesion in the 224x224 image then depends on the camera's
    aspect ratio rather than on the eye. Centring a 2r square on the retina makes the
    retina occupy the same fraction of every output image.

    Regions outside the source frame are filled with black — honest, since nothing was
    imaged there — and the mask records them as not-retina so nothing downstream
    mistakes them for data.
    """
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = max((x1 - x0) // 2, (y1 - y0) // 2)
    side = max(2 * r, 1)

    sx0, sy0 = cx - r, cy - r
    ax0, ay0 = max(0, sx0), max(0, sy0)
    ax1 = min(bgr.shape[1], sx0 + side)
    ay1 = min(bgr.shape[0], sy0 + side)

    out_img = np.zeros((side, side, 3), dtype=bgr.dtype)
    out_msk = np.zeros((side, side), dtype=mask.dtype)
    out_img[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = bgr[ay0:ay1, ax0:ax1]
    out_msk[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = mask[ay0:ay1, ax0:ax1]
    return out_img, out_msk


# ----------------------------------------------------------------------------------
# Step 2 - enhancement
# ----------------------------------------------------------------------------------

def large_sigma_blur(bgr: np.ndarray, sigma: float, work_sigma: float = 48.0) -> np.ndarray:
    """Gaussian blur for very large sigma, computed on a downscaled copy.

    PERFORMANCE, NOT COSMETICS. Ben Graham's sigma here is width/10, so a 2560px-wide
    EyePACS image needs sigma 256 and `cv2.GaussianBlur` builds a ~6*sigma = 1537-tap
    separable kernel. MEASURED: 30.8 s for ONE image, and 35 s/image over the QA sample.
    That projects to roughly 340 hours for 35,126 images — the full cache would never
    finish, on Kaggle or anywhere else.

    A Gaussian this wide is pure low frequency, with essentially no energy above the
    Nyquist limit of an aggressive decimation. So downscale until sigma is `work_sigma`
    pixels, blur there, and scale back.

    Because sigma is itself width/10, `scale = work_sigma/sigma` means the working image
    is ALWAYS ~10*work_sigma px wide (480 px at the default) whatever the source size —
    so the blur costs the same for a 400px image as for a 4928px one, and only the
    resamples scale with input size.

    MEASURED accuracy against the exact full-resolution blur, on a 1920x2560 image,
    after the full `alpha*img + beta*blur + gamma` (alpha=4 amplifies any blur error 4x):

        work_sigma   time      max diff   mean diff
             8       0.008 s      24        1.77
            32       0.068 s       8        0.45
            48       0.212 s       4        0.29     <- default
            64       0.507 s       4        0.20

    4/255 is below JPEG q95 quantisation noise, and the error is smooth and low-frequency
    by construction — it cannot add or remove lesion-scale detail. Locked in by
    tests/test_preprocess.py, which re-measures this rather than trusting the table.
    """
    if sigma <= work_sigma:
        return cv2.GaussianBlur(bgr, (0, 0), sigma)

    h, w = bgr.shape[:2]
    scale = work_sigma / sigma
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    # A feature of width sigma in original coords is sigma*scale wide here, and
    # scale was chosen precisely so that equals work_sigma.
    small = cv2.GaussianBlur(small, (0, 0), sigma * scale)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def ben_graham(bgr: np.ndarray, cfg: PreprocessConfig,
               mask: np.ndarray | None = None) -> np.ndarray:
    """Subtract the local average colour (Ben Graham, winner of the 2015 Kaggle DR comp).

    out = alpha*img + beta*blur(img) + gamma. Sigma is tied to image width so the effect
    is scale-invariant — a fixed sigma would barely touch a 3000px image and turn a 500px
    image to mush, and EyePACS spans both.

    Note the fine detail lives in the `alpha*img` term at FULL resolution; only the
    subtracted local average is approximated by `large_sigma_blur`. Microaneurysm-scale
    structure is untouched by that approximation, which is the thing the contact sheet
    has to confirm.

    WHEN `mask` IS GIVEN, the local average is a NORMALISED CONVOLUTION — blur(img*mask)
    / blur(mask) — so it averages only over retina pixels.

    Without it, the blur window near the retina's edge straddles the black surround and
    is dragged toward zero. `alpha*img - alpha*blur` then blows up, ringing the retina in
    a thick white halo that is BRIGHTER THAN ANY LESION. Measured on the QA sample, the
    0.85-0.99r annulus ran 1.44x the inner-disc brightness. That halo is a constant
    artefact at a constant location, so a CNN can key on it and Grad-CAM will light it
    up — and worse, on the grade-4 examples it washed out lesions that are visible once
    it is removed. The mask is not cosmetic.
    """
    sigma = bgr.shape[1] / cfg.ben_graham_sigma_scale

    if mask is None:
        blurred = large_sigma_blur(bgr, sigma).astype(np.float32)
    else:
        m = (mask > 0).astype(np.float32)
        num = large_sigma_blur(bgr.astype(np.float32) * m[..., None], sigma)
        # One channel, then broadcast: the mask is identical across channels, so blurring
        # a 3-channel copy of it triples this term's cost for an identical result.
        den = large_sigma_blur(m, sigma)[..., None]
        # den -> 0 only outside the retina, which the final mask discards anyway.
        blurred = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)

    out = (cfg.ben_graham_alpha * bgr.astype(np.float32)
           + cfg.ben_graham_beta * blurred
           + cfg.ben_graham_gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_green(bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the green channel — the ablation alternative to Ben Graham.

    Green carries the most vessel and haemorrhage contrast in a fundus photo. The
    equalised green is written back into a 3-channel image so downstream shapes never
    change between arms.
    """
    b, g, r = cv2.split(bgr)
    g = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(g)
    return cv2.merge([b, g, r])


def enhance(bgr: np.ndarray, cfg: PreprocessConfig,
            mask: np.ndarray | None = None) -> np.ndarray:
    if cfg.enhancement == "ben_graham":
        return ben_graham(bgr, cfg, mask)
    if cfg.enhancement == "clahe_green":
        # CLAHE is tile-local and never spans the retina boundary widely enough to ring,
        # so it needs no mask.
        return clahe_green(bgr)
    return bgr


# ----------------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------------

def preprocess_image(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Full per-image pipeline. Input and output are both BGR uint8.

    The retina mask is computed ONCE and threaded through crop, enhancement, and the
    final re-mask, so all three agree about where the retina is. Deriving it more than
    once invites them to disagree by a pixel or two at the boundary, which is exactly
    where the halo artefact lives.
    """
    mask = retina_mask(bgr)

    # A frame with no detectable retina (the 8 KB QA image is close) must still produce
    # an output: it is real data, it stays in the split, and it has to be visible on the
    # contact sheet rather than vanishing behind an exception. scan_quality flags it.
    if not mask.any():
        n = cfg.image_size
        interp = cv2.INTER_AREA if max(bgr.shape[:2]) > n else cv2.INTER_CUBIC
        return cv2.resize(bgr, (n, n), interpolation=interp)

    if cfg.circle_crop:
        bgr, mask = square_crop(bgr, mask)

    out = enhance(bgr, cfg, mask)

    # INTER_AREA for downscaling: every fundus image is far larger than 224, and
    # INTER_LINEAR aliases fine lesion detail on a >5x reduction. Microaneurysms are
    # a few pixels across at source resolution; this is where they survive or die.
    n = cfg.image_size
    interp = cv2.INTER_AREA if max(out.shape[:2]) > n else cv2.INTER_CUBIC
    out = cv2.resize(out, (n, n), interpolation=interp)

    # Re-mask after resize. Ben Graham lifts everything outside the retina to ~gamma
    # (128), so without this every image carries a bright grey surround — a constant
    # artefact at a constant location for the network to key on. INTER_NEAREST keeps the
    # mask binary; a smooth interpolation would leave a feathered rim of partial values.
    if cfg.circle_crop:
        small = cv2.resize(mask, (n, n), interpolation=cv2.INTER_NEAREST)
        out = cv2.bitwise_and(out, out, mask=small)
    return out


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread cannot open a non-ASCII path on Windows — it returns None, no error.

    The repo currently lives under 'C:/Users/NEXT GEN/...'; a user path with an accent
    would turn every read into a silent None and produce an empty cache that looks like
    a logic bug. Decode from bytes instead.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, bgr: np.ndarray, quality: int) -> int:
    """Write a JPEG, returning its byte count (used to measure the real cache size)."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return int(buf.size)


# ----------------------------------------------------------------------------------
# scan_quality - pixel statistics, because file size cannot see any of this
# ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityStats:
    """Per-image quality statistics, computed on the ORIGINAL image before processing.

    File size is a proxy for detail and does find blank frames. It cannot tell a dark
    capture from a well-exposed one, cannot see clipping, and cannot see an off-centre
    disc. These can.
    """

    mean_brightness: float      # 0-255 over retina pixels. Low => underexposed.
    retina_fraction: float      # illuminated area / frame. Tiny => failed capture.
    clipped_fraction: float     # retina pixels >= 250. High => blown out.
    dark_fraction: float        # retina pixels <= 15. High => underexposed.
    centroid_offset: float      # |centroid - frame centre| / (diagonal/2). 0 = centred.
    saturation: float           # mean HSV S over retina. Low => washed out / greyscale.
    laplacian_var: float        # focus, measured at a NORMALISED 512px scale (see below).
    contrast: float             # std of grey over retina.

    def flags(self) -> list[str]:
        """Reasons this image is suspect. An empty list means it looks fine.

        THRESHOLDS ARE FIXED ABSOLUTE CONSTANTS, never percentiles of the dataset.
        A percentile cutoff computed over all 35,126 images would let val and test
        influence which train images get flagged (leakage-auditor condition 1).

        They were calibrated by measuring the 20-image QA sample, which is drawn from
        the TRAIN split only, then set to separate its three known-degraded images from
        the seventeen good ones. `laplacian_var` had the clearest gap: 0.5 / 1.6 / 1.9
        for the three bad images against 4.3 for the lowest good one, so 3.0 sits in the
        gap. That the pixel statistics independently rank the same three images last as
        file size did is a genuine cross-check, not a tautology — the two criteria share
        no inputs.

        This labels images for a human to look at. It does not drop anything, and
        nothing is excluded from the dataset without a docs/DECISIONS.md entry.
        """
        f = []
        if self.retina_fraction < 0.10:
            f.append("tiny-retina")
        if self.mean_brightness < 40:
            f.append("dark")
        if self.dark_fraction > 0.50:
            f.append("mostly-black")
        if self.clipped_fraction > 0.10:
            f.append("over-exposed")
        if self.centroid_offset > 0.20:
            f.append("off-centre")
        if self.saturation < 25:
            f.append("washed-out")
        if self.laplacian_var < 3.0:
            f.append("blurred")
        if self.contrast < 12:
            f.append("low-contrast")
        return f


FOCUS_WIDTH = 512


def _focus(core: np.ndarray) -> float:
    """Laplacian variance at a fixed spatial scale — a resolution-independent focus score.

    Measured on the raw crop this is useless: EyePACS spans 0.1 MP to 16 MP, and a
    Laplacian is a per-pixel operator, so a big sharp image scores LOWER than a small
    one simply because adjacent pixels are more correlated. Measured directly, the QA
    sample gave 152.0 for a 4.9 MP image and 1.9 for a 16.1 MP one, and a single absolute
    threshold flagged 19 of 20 images as blurred — including obviously sharp ones.

    Resampling every crop to a common 512px width first makes the number mean "detail at
    a fixed spatial scale", which is comparable across cameras and is what a fixed
    threshold needs in order to be meaningful.
    """
    if core.size == 0:
        return 0.0
    h, w = core.shape[:2]
    s = FOCUS_WIDTH / w
    nh = max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    core = cv2.resize(core, (FOCUS_WIDTH, nh), interpolation=interp)
    return float(cv2.Laplacian(core, cv2.CV_64F).var())


def scan_quality(bgr: np.ndarray) -> QualityStats:
    """Quality statistics for one ORIGINAL (un-preprocessed) image.

    Everything is measured over the retina mask, not the whole frame. Averaging in the
    black surround makes an image look dark in proportion to how much padding its camera
    produced, which measures the camera rather than the capture.
    """
    h, w = bgr.shape[:2]
    # retina_mask returns uint8 0/255; boolean-index with `> 0`, because indexing an
    # array with a uint8 array is integer fancy-indexing, not masking, and would quietly
    # return the wrong pixels rather than raising.
    mask = retina_mask(bgr) > 0
    n_ret = int(mask.sum())

    if n_ret == 0:
        # A genuinely black frame. Report it honestly rather than dividing by zero.
        return QualityStats(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ret_gray = gray[mask].astype(np.float64)

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    diag_half = float(np.hypot(h, w)) / 2.0
    offset = float(np.hypot(cy - h / 2.0, cx - w / 2.0) / diag_half)

    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1][mask].astype(np.float64)

    # A Laplacian over the masked image would score the mask's own hard edge as detail,
    # so measure focus on the central half of the retina's bounding box instead.
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ph, pw = (y1 - y0) // 4, (x1 - x0) // 4
    core = gray[y0 + ph:y1 - ph, x0 + pw:x1 - pw]
    lap = _focus(core)

    return QualityStats(
        mean_brightness=float(ret_gray.mean()),
        retina_fraction=n_ret / float(h * w),
        clipped_fraction=float((ret_gray >= 250).mean()),
        dark_fraction=float((ret_gray <= 15).mean()),
        centroid_offset=offset,
        saturation=float(sat.mean()),
        laplacian_var=lap,
        contrast=float(ret_gray.std()),
    )


# ----------------------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------------------

def cache_relpath(image_path: str, dataset: str = "eyepacs") -> str:
    """Map a manifest `image_path` to its cache-relative location.

    Flat and split-agnostic by design — see the module docstring. The source directory
    structure is discarded on purpose, because for APTOS that structure IS the
    author-provided split that DECISION-004 says to ignore.

        data/data/10_left.jpeg                  -> eyepacs/10_left.jpg
        train_images/train_images/1ae8c1.png    -> aptos/aptos_1ae8c1.jpg
    """
    stem = Path(image_path).stem
    if dataset == "aptos" and not stem.startswith("aptos_"):
        stem = f"aptos_{stem}"
    return f"{dataset}/{stem}.jpg"


def _process_one(
    task: tuple[str, str, object, Path, Path, PreprocessConfig, bool],
) -> dict:
    """Preprocess a single image. Module-level and picklable, for multiprocessing.

    Never raises on a bad input: the status field carries the failure so the caller can
    report it. One corrupt file must not abort a 35k-image build.
    """
    rel, dataset, label, src_root, out_root, cfg, scan = task
    out_rel = cache_relpath(rel, dataset)
    src = Path(src_root) / rel

    rec = {
        "image_path": rel,
        "dataset": dataset,
        "label": label,
        "cache_path": out_rel,
        "status": "ok",
        "out_bytes": 0,
    }

    if not src.exists():
        rec["status"] = "missing"
        return rec

    try:
        bgr = imread_unicode(src)
    except Exception as exc:                       # noqa: BLE001 - recorded, not hidden
        rec["status"] = f"error:{type(exc).__name__}"
        return rec

    if bgr is None:
        rec["status"] = "unreadable"
        return rec

    rec["src_h"], rec["src_w"] = bgr.shape[0], bgr.shape[1]
    rec["src_bytes"] = src.stat().st_size

    try:
        if scan:
            q = scan_quality(bgr)
            rec.update(asdict(q))
            rec["quality_flags"] = ";".join(q.flags())
        rec["out_bytes"] = imwrite_unicode(
            Path(out_root) / out_rel, preprocess_image(bgr, cfg), cfg.jpeg_quality
        )
    except Exception as exc:                       # noqa: BLE001 - recorded, not hidden
        rec["status"] = f"error:{type(exc).__name__}"

    return rec


def process_manifest(
    df: pd.DataFrame,
    src_root: Path,
    out_root: Path,
    cfg: PreprocessConfig,
    *,
    scan: bool = True,
    progress: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """Preprocess every row of a manifest. Returns one stats row per image, IN INPUT ORDER.

    Missing or unreadable images are RECORDED with status != 'ok' and the run continues,
    so one corrupt file cannot abort a 35k-image cache build. Nothing is silently
    skipped: every input row produces exactly one output row, the caller checks the
    status column, and `main` exits non-zero if any row failed. Per the leakage-auditor,
    a failed image is never dropped from the splits — that needs a docs/DECISIONS.md
    entry and an audited re-split.

    `workers > 1` uses a process pool. Preprocessing is pure per-image CPU work with no
    shared state, so this is safe and near-linear. It also has to exist: measured at
    ~0.8 s/image single-threaded, the full 38,788-image cache takes ~8.6 hours, which
    does not fit comfortably in one Kaggle session. `imap` with a chunksize preserves
    input order, so the stats CSV lines up with the manifest row for row.

    On Windows a pool requires the caller to be under `if __name__ == "__main__":`
    (CLAUDE.md §7); `main()` satisfies that.
    """
    has_dataset = "dataset" in df.columns
    tasks = [
        (
            row.image_path,
            row.dataset if has_dataset else "eyepacs",
            getattr(row, "label", None),
            src_root,
            out_root,
            cfg,
            scan,
        )
        for row in df.itertuples(index=False)
    ]

    total = len(tasks)
    records = []

    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            records.append(_process_one(task))
            if progress and (i % 250 == 0 or i == total):
                print(f"  {i}/{total}", flush=True)
    else:
        with mp.Pool(processes=workers) as pool:
            for i, rec in enumerate(pool.imap(_process_one, tasks, chunksize=16), start=1):
                records.append(rec)
                if progress and (i % 250 == 0 or i == total):
                    print(f"  {i}/{total}", flush=True)

    out = pd.DataFrame.from_records(records)
    assert len(out) == len(df), (
        f"{len(df)} images in, {len(out)} rows out — the cache must never silently "
        "disagree with the manifest"
    )
    return out


def summarise(stats: pd.DataFrame) -> str:
    """Human-readable run summary, including the MEASURED cache size (replaces the
    [ESTIMATE] in state/session_handoff.md once this has run on real images)."""
    ok = stats[stats["status"] == "ok"]
    bad = stats[stats["status"] != "ok"]

    lines = [f"processed : {len(ok)}/{len(stats)}"]
    if len(bad):
        counts = bad["status"].value_counts().to_dict()
        lines.append(f"FAILED    : {len(bad)}  {counts}")
        for r in bad.itertuples(index=False):
            lines.append(f"    {r.status}: {r.image_path}")

    if len(ok):
        mean_kb = ok["out_bytes"].mean() / 1024
        lines += [
            f"cache size: {ok['out_bytes'].sum() / 1024**2:.2f} MB "
            f"(mean {mean_kb:.1f} KB/image)   [MEASURED]",
            f"projected : {mean_kb * 35126 / 1024**2:.2f} GB for 35,126 EyePACS + "
            f"{mean_kb * 3662 / 1024**2:.2f} GB for 3,662 APTOS",
        ]
        if "quality_flags" in ok.columns:
            flagged = ok[ok["quality_flags"].astype(str) != ""]
            lines.append(f"flagged   : {len(flagged)}/{len(ok)} with quality flags")
            for r in flagged.itertuples(index=False):
                lines.append(f"    {r.image_path}  ->  {r.quality_flags}")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    ap.add_argument("--manifest", type=Path,
                    help="CSV with an image_path column (read with comment='#')")
    ap.add_argument("--split", action="append", default=[],
                    help="split name from data/splits/; repeatable")
    ap.add_argument("--src-root", type=Path, required=True,
                    help="root that the manifest's image_path values are relative to")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, help="write the per-image stats CSV here")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip scan_quality (faster for the full cache build)")
    ap.add_argument("--limit", type=int, help="process only the first N rows")
    ap.add_argument("--workers", type=int, default=1,
                    help="process pool size; use os.cpu_count() on Kaggle (4)")
    args = ap.parse_args()

    if bool(args.manifest) == bool(args.split):
        ap.error("give exactly one of --manifest or --split")

    cfg = load_preprocess_config(args.config)

    if args.manifest:
        df = pd.read_csv(args.manifest, comment="#")
        if "image_path" not in df.columns:
            ap.error(f"{args.manifest} has no image_path column")
    else:
        from src.data.manifest import load_split
        df = pd.concat([load_split(s) for s in args.split], ignore_index=True)

    if args.limit:
        df = df.head(args.limit)

    print(f"config    : {args.config}")
    print(f"pipeline  : circle_crop={cfg.circle_crop} enhancement={cfg.enhancement} "
          f"size={cfg.image_size} quality={cfg.jpeg_quality}")
    print(f"source    : {args.src_root}")
    print(f"cache     : {args.out_root}   (flat, split-agnostic)")
    print(f"images    : {len(df)}\n")

    stats = process_manifest(df, args.src_root, args.out_root, cfg,
                             scan=not args.no_scan, workers=args.workers)

    print()
    print(summarise(stats))

    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats, index=False, lineterminator="\n")
        print(f"\nstats written to {args.stats}")

    return 1 if (stats["status"] != "ok").any() else 0


if __name__ == "__main__":
    sys.exit(main())

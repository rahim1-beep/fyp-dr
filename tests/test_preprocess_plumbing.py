"""Plumbing tests for src/data/preprocess.py — the gaps found in code review.

`tests/test_preprocess.py` covers the image maths (the fast blur, the focus measure, the
halo). This module covers everything around it: the batch driver, the parallel path, the
files actually written to disk, and channel order. Each of these was a real hole — the
suite passed while an output file was never opened, `workers > 1` was never exercised,
and a BGR/RGB swap would not have been noticed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    imread_unicode,
    imwrite_unicode,
    load_preprocess_config,
    preprocess_image,
    process_manifest,
    retina_mask,
    square_crop,
)
from tests.test_preprocess import synthetic_fundus

REPO = Path(__file__).resolve().parents[1]
CFG = load_preprocess_config(REPO / "configs" / "base.yaml")


def _manifest(tmp_path: Path, names: list[str]) -> tuple[pd.DataFrame, Path, Path]:
    """Write synthetic source images; return (df, src_root, out_root)."""
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "data" / "data").mkdir(parents=True)
    for i, nm in enumerate(names):
        cv2.imwrite(str(src / "data" / "data" / f"{nm}.jpeg"),
                    synthetic_fundus(w=400, h=400, seed=i))
    df = pd.DataFrame({
        "image_path": [f"data/data/{nm}.jpeg" for nm in names],
        "label": [i % 5 for i in range(len(names))],
        "dataset": ["eyepacs"] * len(names),
    })
    return df, src, out


def test_written_files_exist_and_are_readable_at_the_configured_size(tmp_path):
    """No earlier test ever opened an output file, so 'written to the wrong path' and
    'wrote a corrupt JPEG' were both untested."""
    df, src, out = _manifest(tmp_path, ["10_left", "10_right", "11_left"])
    stats = process_manifest(df, src, out, CFG, progress=False)

    assert (stats["status"] == "ok").all()
    for rel in stats["cache_path"]:
        f = out / rel
        assert f.exists(), f"{rel} recorded ok but no file on disk"
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None, f"{rel} is not a decodable image"
        assert img.shape == (CFG.image_size, CFG.image_size, 3)
    assert stats["out_bytes"].min() > 0


def test_parallel_and_serial_agree_exactly(tmp_path):
    """The imap order-preservation claim was entirely untested.

    Nothing else stops someone swapping in `imap_unordered`, which would silently
    misalign every stats row with its manifest row — and the stats CSV is what the
    cache-vs-CSV reconciliation depends on.
    """
    names = [f"{i}_left" for i in range(12)]

    df, src1, out1 = _manifest(tmp_path / "a", names)
    serial = process_manifest(df, src1, out1, CFG, progress=False, workers=1)

    df2, src2, out2 = _manifest(tmp_path / "b", names)
    parallel = process_manifest(df2, src2, out2, CFG, progress=False, workers=3)

    assert serial["image_path"].tolist() == df["image_path"].tolist()
    assert parallel["image_path"].tolist() == df["image_path"].tolist()
    assert serial["out_bytes"].tolist() == parallel["out_bytes"].tolist()

    for rel in serial["cache_path"]:
        assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), (
            f"{rel} differs between serial and parallel builds"
        )


def test_colliding_cache_paths_raise_rather_than_overwrite(tmp_path):
    """Two rows sharing a filename stem map to one cache file: the second overwrites the
    first and BOTH report status 'ok'. A lost image wearing a success label."""
    src, out = tmp_path / "src", tmp_path / "out"
    for sub in ("data/data", "other"):
        (src / sub).mkdir(parents=True)
        cv2.imwrite(str(src / sub / "1_left.jpeg"), synthetic_fundus(w=400, h=400))

    df = pd.DataFrame({
        "image_path": ["data/data/1_left.jpeg", "other/1_left.jpeg"],
        "label": [0, 1],
        "dataset": ["eyepacs", "eyepacs"],
    })
    with pytest.raises(RuntimeError, match="overwrite"):
        process_manifest(df, src, out, CFG, progress=False)


def test_a_row_that_explodes_does_not_discard_every_other_record(tmp_path):
    """A NaN image_path used to raise out of the pool and destroy the entire run — on the
    ~2.5 h Kaggle build that is the whole session, with no stats CSV written."""
    df, src, out = _manifest(tmp_path, ["10_left", "10_right"])
    df = pd.concat(
        [df, pd.DataFrame({"image_path": [np.nan], "label": [0], "dataset": ["eyepacs"]})],
        ignore_index=True,
    )

    stats = process_manifest(df, src, out, CFG, progress=False)
    assert len(stats) == 3, "the bad row must not take the good ones with it"
    assert (stats["status"] == "ok").sum() == 2
    assert stats["status"].astype(str).str.startswith("error").sum() == 1


def test_no_retina_images_are_marked_distinctly_even_without_scan(tmp_path):
    """That branch emits a stretched, un-cropped, un-enhanced, un-masked image — a
    different domain from every other cache entry. Under --no-scan there was previously
    no record of which images took it."""
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "data" / "data").mkdir(parents=True)
    cv2.imwrite(str(src / "data" / "data" / "0_left.jpeg"),
                np.zeros((300, 400, 3), np.uint8))
    cv2.imwrite(str(src / "data" / "data" / "1_left.jpeg"),
                synthetic_fundus(w=400, h=400))

    df = pd.DataFrame({
        "image_path": ["data/data/0_left.jpeg", "data/data/1_left.jpeg"],
        "label": [0, 1],
        "dataset": ["eyepacs", "eyepacs"],
    })
    stats = process_manifest(df, src, out, CFG, scan=False, progress=False)

    assert stats.loc[0, "status"] == "ok:no-retina"
    assert stats.loc[1, "status"] == "ok"


def test_channel_order_survives_the_pipeline():
    """A BGR/RGB swap passes every other test in the suite.

    Downstream, timm's `default_cfg` normalisation assumes RGB. A swap trains a model
    that runs, converges, and quietly underperforms, with no error anywhere — the exact
    failure mode that lands a DR project at 20-45% accuracy.
    """
    import dataclasses

    img = np.zeros((600, 600, 3), np.uint8)
    cv2.circle(img, (300, 300), 290, (220, 40, 40), -1)   # BGR: blue-dominant disc

    # Ben Graham subtracts the LOCAL MEAN, so a flat disc correctly cancels to gamma
    # (128) in every channel and carries no colour information at all. Channel order is
    # therefore checked through the geometry-only path, which is where a cvtColor slip
    # would actually live.
    geom_only = dataclasses.replace(CFG, enhancement="none")
    out = preprocess_image(img, geom_only)

    centre = out[100:124, 100:124].reshape(-1, 3).mean(axis=0)
    assert centre[0] > centre[2] + 100, (
        f"channel order flipped: channel means {centre.round(1)}; "
        "blue must stay dominant for a blue disc"
    )

    # And the JPEG round-trip must not swap them either — that is what training reads.
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), CFG.jpeg_quality])
    assert ok
    back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rt = back[100:124, 100:124].reshape(-1, 3).mean(axis=0)
    assert rt[0] > rt[2] + 100, f"channel order flipped by the JPEG round-trip: {rt.round(1)}"


def test_square_crop_keeps_image_and_mask_registered():
    """Shape equality alone would pass for a mask shifted by rows against its image."""
    img = synthetic_fundus(w=1000, h=800, cx=380, cy=300, r=250)
    out, msk = square_crop(img, retina_mask(img))

    lit = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) > 10
    disagree = np.logical_xor(lit, msk > 0).sum() / msk.size
    assert disagree < 0.03, f"{disagree:.1%} of pixels disagree — mask is misregistered"


def test_bright_artefact_does_not_shrink_the_retina():
    """A bright corner blob used to inflate the bounding box, so square_crop centred on
    the union and rendered the retina ~0.67x smaller — a third of every lesion's pixels
    — while reporting status 'ok' with a plausible-looking output."""
    clean = synthetic_fundus(w=1000, h=1000, r=380)
    dirty = clean.copy()
    dirty[40:130, 40:100] = 240          # specular blob in the corner

    a = int((retina_mask(clean) > 0).sum())
    b = int((retina_mask(dirty) > 0).sum())
    assert abs(a - b) / a < 0.05, "largest_component should have dropped the artefact"

    side_clean = square_crop(clean, retina_mask(clean))[0].shape[0]
    side_dirty = square_crop(dirty, retina_mask(dirty))[0].shape[0]
    assert abs(side_clean - side_dirty) <= 4, (
        f"crop side {side_clean} -> {side_dirty}: the retina would render at a "
        "different scale, changing effective lesion size"
    )


def test_non_ascii_paths_round_trip(tmp_path):
    """The stated reason imread_unicode / imwrite_unicode exist: cv2.imread returns None
    on a non-ASCII Windows path and reports no error."""
    d = tmp_path / "reti_na_éè"
    d.mkdir()
    img = synthetic_fundus(w=300, h=300)

    n = imwrite_unicode(d / "imagé.jpg", img, 95)
    assert n > 0

    back = imread_unicode(d / "imagé.jpg")
    assert back is not None and back.shape == img.shape


def test_empty_manifest_fails_with_a_clear_message(tmp_path):
    df = pd.DataFrame({"image_path": [], "label": [], "dataset": []})
    with pytest.raises(ValueError, match="no rows"):
        process_manifest(df, tmp_path, tmp_path / "o", CFG, progress=False)

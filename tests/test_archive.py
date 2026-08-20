"""Tests for src/data/archive_cache.py.

The failure these exist for: Kaggle kept 499 of 38,788 images when the notebook output
was saved, and nothing anywhere said so. The archive path has to be able to detect a
short payload — including the case where the SOURCE is already short, which is what
happens if a build is interrupted.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.data.archive_cache import (
    SIDECARS,
    collect,
    pack,
    summarise_archive,
    unpack,
    verify_archive,
)


def _cache(tmp_path: Path, n_images: int = 20, sidecars=SIDECARS) -> Path:
    root = tmp_path / "processed"
    (root / "eyepacs").mkdir(parents=True)
    (root / "aptos").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)

    for i in range(n_images):
        sub = "eyepacs" if i % 5 else "aptos"
        (root / sub / f"{i}_left.jpg").write_bytes(b"\xff\xd8\xff" + bytes(1000))
    for s in sidecars:
        (root / s).write_text("x,y\n1,2\n", encoding="utf-8")
    (root / "splits" / "train.csv").write_text("# header\na,b\n", encoding="utf-8")
    return root


def test_pack_then_verify_round_trips(tmp_path):
    root = _cache(tmp_path, 20)
    out = tmp_path / "cache.zip"

    info = pack(root, out, expect_images=20, progress_every=0)
    assert info["images"] == 20
    assert out.exists()

    assert verify_archive(out, expect_images=20, expect_gb=None) == []


def test_pack_refuses_a_short_source_tree(tmp_path):
    """Checked on the SOURCE before anything is written. Packing a short tree and then
    verifying the archive against the same short tree would agree with itself."""
    root = _cache(tmp_path, 19)
    out = tmp_path / "cache.zip"

    with pytest.raises(RuntimeError, match="expected 20"):
        pack(root, out, expect_images=20, progress_every=0)
    assert not out.exists(), "nothing should be written when the source is short"


def test_verify_catches_a_truncated_archive(tmp_path):
    """THE regression. 499 of 38,788 got through once; it does not get through again."""
    root = _cache(tmp_path, 20)
    full = tmp_path / "full.zip"
    pack(root, full, expect_images=20, progress_every=0)

    # rebuild it holding only 5 images, exactly like a capped output
    short = tmp_path / "short.zip"
    with zipfile.ZipFile(full) as src, zipfile.ZipFile(short, "w") as dst:
        kept = 0
        for i in src.infolist():
            if i.filename.endswith(".jpg"):
                if kept >= 5:
                    continue
                kept += 1
            dst.writestr(i, src.read(i.filename))

    problems = verify_archive(short, expect_images=20, expect_gb=None)
    assert problems and "5 images, expected 20" in problems[0]
    assert "15 missing" in problems[0]


def test_verify_catches_a_missing_sidecar(tmp_path):
    """Without the provenance file the published dataset cannot say which pipeline
    produced its pixels, and reconcile_cache will refuse it later anyway."""
    root = _cache(tmp_path, 20, sidecars=("processed_stats.csv", "aptos_stats.csv"))
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=20, progress_every=0)

    problems = verify_archive(out, expect_images=20, expect_gb=None)
    assert any("_cache_provenance.json" in p for p in problems)


def test_verify_catches_a_wrong_total_size(tmp_path):
    root = _cache(tmp_path, 20)
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=20, progress_every=0)

    problems = verify_archive(out, expect_images=20, expect_gb=5.0, gb_tolerance=0.1)
    assert any("expected about 5.000 GB" in p for p in problems)


def test_verify_catches_a_zero_byte_entry(tmp_path):
    root = _cache(tmp_path, 20)
    (root / "eyepacs" / "1_left.jpg").write_bytes(b"")
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=20, progress_every=0)

    assert any("zero-byte" in p for p in verify_archive(out, expect_images=20,
                                                        expect_gb=None))


def test_verify_reports_a_missing_archive_rather_than_raising(tmp_path):
    assert verify_archive(tmp_path / "nope.zip", expect_images=1) != []


def test_deep_verification_passes_on_a_good_archive(tmp_path):
    root = _cache(tmp_path, 10)
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=10, progress_every=0)
    assert verify_archive(out, expect_images=10, expect_gb=None, deep=True) == []


# ----------------------------------------------------------------------------------
# unpack — what every training notebook does
# ----------------------------------------------------------------------------------

def test_unpack_restores_the_tree_and_returns_its_root(tmp_path):
    root = _cache(tmp_path, 20)
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=20, progress_every=0)

    dest = tmp_path / "extracted"
    got = unpack(out, dest, expect_images=20)

    assert got == dest / "processed"
    assert (got / "eyepacs").is_dir() and (got / "splits" / "train.csv").exists()
    assert sum(1 for _ in got.rglob("*.jpg")) == 20


def test_unpack_refuses_a_truncated_archive(tmp_path):
    """'The archive was complete' and 'the extraction completed' are different claims,
    and training depends on the second."""
    root = _cache(tmp_path, 20)
    full = tmp_path / "full.zip"
    pack(root, full, expect_images=20, progress_every=0)

    short = tmp_path / "short.zip"
    with zipfile.ZipFile(full) as src, zipfile.ZipFile(short, "w") as dst:
        for i in src.infolist():
            if i.filename.endswith(".jpg") and "1_left" in i.filename:
                continue
            dst.writestr(i, src.read(i.filename))

    with pytest.raises(RuntimeError, match="do not train on it"):
        unpack(short, tmp_path / "x", expect_images=20)


def test_the_archive_is_stored_not_deflated(tmp_path):
    """JPEGs are already entropy-coded; deflate costs minutes to save low single digits.
    STORED writes at disk speed."""
    root = _cache(tmp_path, 10)
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=10, progress_every=0)

    with zipfile.ZipFile(out) as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_summary_counts_images_separately_from_entries(tmp_path):
    root = _cache(tmp_path, 20)
    out = tmp_path / "cache.zip"
    pack(root, out, expect_images=20, progress_every=0)

    s = summarise_archive(out)
    assert s["images"] == 20
    assert s["entries"] == 20 + len(SIDECARS) + 1        # + splits/train.csv


def test_collect_separates_images_from_sidecars(tmp_path):
    root = _cache(tmp_path, 7)
    images, others = collect(root)
    assert len(images) == 7
    assert {p.name for p in others} >= set(SIDECARS)

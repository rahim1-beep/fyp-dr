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


# ----------------------------------------------------------------------------------
# Finding the cache root — DECISION-023
# ----------------------------------------------------------------------------------
#
# The same artefact has now appeared in three shapes, because the hosting platform
# reshaped it: as a loose `processed/` tree, as a single `.zip`, and — after Kaggle
# auto-extracted the zip on publish — as a folder named after the dataset with the stats
# CSVs beside it. Each of these tests is one of those shapes.

from src.data.archive_cache import find_cache_root, looks_like_cache_root, resolve_cache


def _tree(root: Path, n: int = 6) -> Path:
    """A minimal cache tree: eyepacs/ and aptos/ with images in them."""
    (root / "eyepacs").mkdir(parents=True, exist_ok=True)
    (root / "aptos").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sub = "eyepacs" if i % 3 else "aptos"
        (root / sub / f"{i}_left.jpg").write_bytes(b"\xff\xd8\xff" + bytes(100))
    return root


def test_finds_a_cache_root_that_is_the_mount_itself(tmp_path):
    _tree(tmp_path)
    assert find_cache_root(tmp_path, expect_images=6) == tmp_path


def test_finds_processed_one_level_down(tmp_path):
    """The layout the notebook used to assume."""
    _tree(tmp_path / "processed")
    assert find_cache_root(tmp_path, expect_images=6) == tmp_path / "processed"


def test_finds_the_layout_kaggle_actually_published(tmp_path):
    """Kaggle auto-extracted the archive and named the folder after the dataset, with the
    stats CSVs left beside it — neither MOUNT/*.zip nor MOUNT/processed."""
    _tree(tmp_path / "fyp-dr-eyepacs-224" / "processed")
    (tmp_path / "aptos_stats.csv").write_text("a,b\n", encoding="utf-8")
    (tmp_path / "processed_stats.csv").write_text("a,b\n", encoding="utf-8")

    got = find_cache_root(tmp_path, expect_images=6)
    assert got == tmp_path / "fyp-dr-eyepacs-224" / "processed"


def test_finds_a_root_nested_without_a_processed_level(tmp_path):
    """And the variant where the extract dropped the arcname entirely."""
    _tree(tmp_path / "fyp-dr-eyepacs-224")
    assert find_cache_root(tmp_path, expect_images=6) == tmp_path / "fyp-dr-eyepacs-224"


def test_the_shallowest_match_wins(tmp_path):
    """If a nested copy exists, the published outer one is the one to use."""
    _tree(tmp_path / "processed")
    _tree(tmp_path / "processed" / "backup" / "processed")
    assert find_cache_root(tmp_path) == tmp_path / "processed"


def test_an_empty_eyepacs_directory_is_not_a_cache_root(tmp_path):
    """A half-extracted tree has the right names and none of the data."""
    (tmp_path / "eyepacs").mkdir()
    (tmp_path / "aptos").mkdir()
    with pytest.raises(FileNotFoundError, match="no cache root"):
        find_cache_root(tmp_path)


def test_a_short_cache_is_refused_rather_than_returned(tmp_path):
    """Finding it and it being complete are different claims."""
    _tree(tmp_path / "processed", n=5)
    with pytest.raises(RuntimeError, match="expected 38788|holds 5 images"):
        find_cache_root(tmp_path, expect_images=38788)


def test_the_error_names_what_is_actually_mounted(tmp_path):
    (tmp_path / "something_else").mkdir()
    with pytest.raises(FileNotFoundError, match="what is actually there"):
        find_cache_root(tmp_path)


def test_looks_like_cache_root_needs_images_not_just_names(tmp_path):
    (tmp_path / "eyepacs").mkdir()
    assert not looks_like_cache_root(tmp_path)
    (tmp_path / "eyepacs" / "1_left.jpg").write_bytes(b"\xff\xd8\xff")
    assert looks_like_cache_root(tmp_path)


def test_resolve_cache_handles_the_extracted_layout(tmp_path):
    mount = tmp_path / "mount"
    _tree(mount / "fyp-dr-eyepacs-224" / "processed")
    got = resolve_cache(mount, tmp_path / "work", expect_images=6)
    assert got == mount / "fyp-dr-eyepacs-224" / "processed"
    assert not (tmp_path / "work").exists(), "nothing needed extracting"


def test_resolve_cache_still_handles_a_zip(tmp_path):
    """If a future publish leaves the archive intact, the same call must work."""
    src = _tree(tmp_path / "processed")
    for s in SIDECARS:
        (src / s).write_text("{}", encoding="utf-8")
    mount = tmp_path / "mount"
    mount.mkdir()
    pack(src, mount / "cache.zip", expect_images=6, progress_every=0)

    got = resolve_cache(mount, tmp_path / "work", expect_images=6)
    assert sum(1 for _ in got.rglob("*.jpg")) == 6
    assert (tmp_path / "work").exists(), "the zip should have been extracted"

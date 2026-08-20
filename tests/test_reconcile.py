"""Tests for src/data/reconcile_cache.py — the pre-training gate the auditor mandated.

The reason this file exists: an earlier version of `reconcile_cache` printed

    RECONCILIATION PASSED ...
    EXIT CODE: 0

on a partition with `10321_left` in train and `10321_right` in test. Every count it
checked was correct. Counting files cannot see a patient straddling the boundary, because
two images of one patient have two different names and two different cache files.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import PROVENANCE_FILE, load_preprocess_config
from src.data.reconcile_cache import (
    check_provenance,
    decode_check,
    patient_overlaps,
    ROLES,
)

REPO = Path(__file__).resolve().parents[1]
CFG = load_preprocess_config(REPO / "configs" / "base.yaml")


def _rows(pairs: list[tuple[str, str, int]]) -> pd.DataFrame:
    """(patient, split_file, label) -> the frame reconcile_cache builds internally."""
    return pd.DataFrame([
        {"image_path": f"data/data/{pid}.jpeg", "patient_id": pid.split("_")[0],
         "label": lab, "dataset": "eyepacs", "split_file": sf,
         "cache_path": f"eyepacs/{pid}.jpg"}
        for pid, sf, lab in pairs
    ])


# ----------------------------------------------------------------------------------
# R1 — the check that was missing
# ----------------------------------------------------------------------------------

def test_one_patient_split_across_train_and_test_is_caught():
    """The exact partition that used to reconcile clean: same patient, two eyes, two
    cache files, zero collisions, zero orphans."""
    exp = _rows([("10321_left", "train", 4), ("10321_right", "test", 0),
                 ("55_left", "train", 0), ("77_left", "val", 1)])
    problems = patient_overlaps(exp)
    assert problems and "R1 VIOLATION" in problems[0]
    assert "10321" in problems[0]


def test_a_clean_partition_reports_nothing():
    exp = _rows([("1_left", "train", 0), ("1_right", "train", 0),
                 ("2_left", "val", 1), ("3_left", "test", 2)])
    assert patient_overlaps(exp) == []


def test_eyepacs_and_aptos_train_are_the_same_side_of_the_boundary():
    """Arm F pools them. Two splits sharing the `train` role is legitimate; the check
    must compare roles, not split file names."""
    exp = _rows([("1_left", "train", 0), ("9_left", "aptos_train", 4),
                 ("2_left", "val", 1)])
    assert patient_overlaps(exp) == []
    assert ROLES["aptos_train"] == "train" and ROLES["aptos_val"] == "val"


def test_a_patient_in_both_eyepacs_val_and_aptos_test_is_caught():
    exp = _rows([("1_left", "aptos_val", 0), ("1_right", "test", 0),
                 ("2_left", "train", 1)])
    assert any("R1 VIOLATION" in p for p in patient_overlaps(exp))


def test_an_unknown_split_name_is_not_guessed_at():
    exp = _rows([("1_left", "train", 0), ("2_left", "holdout_2026", 1)])
    problems = patient_overlaps(exp)
    assert any("no known role" in p for p in problems)


def test_missing_patient_id_column_is_a_problem_not_a_pass():
    exp = _rows([("1_left", "train", 0)]).drop(columns=["patient_id"])
    assert patient_overlaps(exp) != []


# ----------------------------------------------------------------------------------
# Provenance — a cache built by a different pipeline
# ----------------------------------------------------------------------------------

def _provenance(root: Path, **overrides) -> None:
    from dataclasses import asdict, replace
    cfg = replace(CFG, **overrides) if overrides else CFG
    root.mkdir(parents=True, exist_ok=True)
    (root / PROVENANCE_FILE).write_text(json.dumps({"runs": [
        {"started": "2026-08-20T00:00:00+00:00", "src_root": "/kaggle/input/eyepacs",
         "git": "abc1234", "n_images": 10, "preprocess": asdict(cfg)}
    ]}), encoding="utf-8")


def test_a_cache_built_with_a_different_erosion_is_caught(tmp_path):
    """DECISION-016 landed mid-Phase-2. A cache half-built at 0.0 and half at 0.025 is
    two image domains sharing a directory, and every count still matches."""
    _provenance(tmp_path, mask_erode_frac=0.0)
    problems = check_provenance(tmp_path, CFG)
    assert problems and "mask_erode_frac" in problems[0]


def test_matching_provenance_passes(tmp_path):
    _provenance(tmp_path)
    assert check_provenance(tmp_path, CFG) == []


def test_a_cache_with_no_provenance_is_a_problem(tmp_path):
    assert check_provenance(tmp_path, CFG) != []


def test_provenance_appends_rather_than_overwrites(tmp_path):
    """The cache is built by two invocations into one root — EyePACS then APTOS. The
    second must not erase the first's record."""
    from src.data.preprocess import record_provenance
    record_provenance(tmp_path, CFG, Path("/kaggle/input/eyepacs"), 35126, ["train"])
    record_provenance(tmp_path, CFG, Path("/kaggle/input/aptos2019"), 3662,
                      ["aptos_train"])
    doc = json.loads((tmp_path / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert len(doc["runs"]) == 2
    assert {r["n_images"] for r in doc["runs"]} == {35126, 3662}
    assert check_provenance(tmp_path, CFG) == []


# ----------------------------------------------------------------------------------
# Decode sampling
# ----------------------------------------------------------------------------------

def test_decode_sample_is_stratified_by_dataset(tmp_path):
    """Unstratified, APTOS gets ~9% of the checks purely because it is ~9% of the cache —
    and APTOS is the half whose source format differs."""
    paths = []
    for ds, n in (("eyepacs", 200), ("aptos", 20)):
        (tmp_path / ds).mkdir()
        for i in range(n):
            f = tmp_path / ds / f"{i}.jpg"
            cv2.imwrite(str(f), np.full((224, 224, 3), 128, np.uint8))
            paths.append(f)

    bad = decode_check(paths, 224, sample=20)
    assert bad == []

    # a corrupt APTOS file must be reachable by a small sample
    (tmp_path / "aptos" / "3.jpg").write_bytes(b"not an image")
    assert decode_check(paths, 224, sample=20)


def test_decode_check_catches_a_black_image(tmp_path):
    (tmp_path / "eyepacs").mkdir()
    f = tmp_path / "eyepacs" / "x.jpg"
    cv2.imwrite(str(f), np.zeros((224, 224, 3), np.uint8))
    assert any("black" in b for b in decode_check([f], 224, sample=0))


def test_decode_check_catches_the_wrong_size(tmp_path):
    (tmp_path / "eyepacs").mkdir()
    f = tmp_path / "eyepacs" / "x.jpg"
    cv2.imwrite(str(f), np.full((160, 160, 3), 128, np.uint8))
    assert any("shape" in b for b in decode_check([f], 224, sample=0))


# ----------------------------------------------------------------------------------
# The real splits
# ----------------------------------------------------------------------------------

def test_the_committed_splits_are_patient_disjoint():
    """Runs against the real six CSVs, not a fixture."""
    from src.data.reconcile_cache import expected_rows, APTOS_SPLITS, EYEPACS_SPLITS
    exp = expected_rows(EYEPACS_SPLITS + APTOS_SPLITS)
    assert patient_overlaps(exp) == []
    assert exp["cache_path"].nunique() == len(exp)

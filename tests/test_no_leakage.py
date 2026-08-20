"""The gate. This must pass before ANY training run, and again at the top of every
training script.

It enforces R1 (patient-level splitting) and R2 (no balancing baked into the splits).
These are the project's headline contribution; if this file goes red, nothing downstream
means anything.

Run:
    .venv\\Scripts\\python -m pytest tests/test_no_leakage.py -v
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "data" / "splits"
CONFIG = REPO / "configs" / "base.yaml"

EXPECTED_COLUMNS = ["image_path", "patient_id", "eye", "label", "dataset", "split"]
SPLITS = ("train", "val", "test")

# Ground truth, established by the Phase 1 reconciliation (docs/data_report_eyepacs.json).
EYEPACS_TOTAL_IMAGES = 35_126
EYEPACS_TOTAL_PATIENTS = 17_563
APTOS_TOTAL_IMAGES = 3_662


def _read(name: str) -> pd.DataFrame:
    """Split CSVs carry a '#' provenance header — readers MUST pass comment='#'."""
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return pd.read_csv(path, comment="#", dtype={"patient_id": "string"})


@pytest.fixture(scope="module")
def eyepacs() -> dict[str, pd.DataFrame]:
    return {s: _read(s) for s in SPLITS}


@pytest.fixture(scope="module")
def aptos() -> dict[str, pd.DataFrame]:
    return {s: _read(f"aptos_{s}") for s in SPLITS}


# ---------------------------------------------------------------------------------
# R1 — the rule that matters
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", list(combinations(SPLITS, 2)))
def test_zero_patient_overlap_eyepacs(eyepacs, a, b):
    """No patient may appear in two splits. All three pairs are checked."""
    overlap = set(eyepacs[a]["patient_id"]) & set(eyepacs[b]["patient_id"])
    assert not overlap, (
        f"LEAKAGE: {len(overlap)} patients in both {a} and {b}: "
        f"{sorted(overlap)[:10]}"
    )


@pytest.mark.parametrize("a,b", list(combinations(SPLITS, 2)))
def test_zero_patient_overlap_aptos(aptos, a, b):
    overlap = set(aptos[a]["patient_id"]) & set(aptos[b]["patient_id"])
    assert not overlap, f"LEAKAGE: {len(overlap)} APTOS patients in both {a} and {b}"


def test_no_image_appears_twice(eyepacs):
    """Every image assigned exactly once — no duplicates, no orphans."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    dupes = allrows[allrows["image_path"].duplicated(keep=False)]
    assert dupes.empty, f"{dupes['image_path'].nunique()} images assigned more than once"


def test_both_eyes_of_a_patient_share_a_split(eyepacs):
    """The specific failure R1 exists to prevent: left and right eye on opposite sides."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    per_patient = allrows.groupby("patient_id")["split"].nunique()
    offenders = per_patient[per_patient > 1]
    assert offenders.empty, (
        f"LEAKAGE: {len(offenders)} patients span multiple splits: "
        f"{offenders.index[:10].tolist()}"
    )


# ---------------------------------------------------------------------------------
# Completeness — nothing silently dropped
# ---------------------------------------------------------------------------------

def test_total_image_count_matches_source(eyepacs):
    total = sum(len(df) for df in eyepacs.values())
    assert total == EYEPACS_TOTAL_IMAGES, (
        f"Expected {EYEPACS_TOTAL_IMAGES} images, splits contain {total}. "
        "Rows were dropped — every exclusion must be logged in docs/DECISIONS.md."
    )


def test_total_patient_count_matches_source(eyepacs):
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    assert allrows["patient_id"].nunique() == EYEPACS_TOTAL_PATIENTS


def test_aptos_total_count(aptos):
    total = sum(len(df) for df in aptos.values())
    assert total == APTOS_TOTAL_IMAGES


def test_every_patient_has_both_eyes(eyepacs):
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    eyes = allrows.groupby("patient_id")["eye"].apply(lambda s: set(s))
    bad = eyes[eyes != {"left", "right"}]
    assert bad.empty, f"{len(bad)} patients do not have exactly one left and one right eye"


# ---------------------------------------------------------------------------------
# Schema and label integrity
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLITS)
def test_columns(eyepacs, name):
    assert list(eyepacs[name].columns) == EXPECTED_COLUMNS


@pytest.mark.parametrize("name", SPLITS)
def test_exactly_five_classes_present(eyepacs, name):
    """Failure mode 1: a label vector of the wrong width or order."""
    labels = sorted(eyepacs[name]["label"].unique().tolist())
    assert labels == [0, 1, 2, 3, 4], f"{name} has labels {labels}, expected [0,1,2,3,4]"


@pytest.mark.parametrize("name", SPLITS)
def test_split_column_is_self_consistent(eyepacs, name):
    assert set(eyepacs[name]["split"].unique()) == {name}


def test_no_null_values(eyepacs):
    for name, df in eyepacs.items():
        assert not df.isna().any().any(), f"{name}.csv contains nulls"


# ---------------------------------------------------------------------------------
# R2 — balancing must NOT be baked into the splits
# ---------------------------------------------------------------------------------

def test_splits_are_not_balanced(eyepacs):
    """R2: balancing is train-only and happens at batch-draw time via
    WeightedRandomSampler. If a split CSV looks balanced, someone physically over- or
    under-sampled, and duplicated images are now on both sides of the partition.
    """
    for name, df in eyepacs.items():
        share = df["label"].value_counts(normalize=True)
        assert share.max() > 0.5, (
            f"{name} looks balanced (max class share {share.max():.3f}). "
            "Splits must retain the natural distribution."
        )


def test_val_and_test_match_natural_distribution(eyepacs):
    """Val and test must look like real screening, within sampling noise."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    natural = allrows["label"].value_counts(normalize=True).sort_index()

    for name in ("val", "test"):
        share = eyepacs[name]["label"].value_counts(normalize=True).sort_index()
        drift = (share - natural).abs().max()
        assert drift < 0.01, (
            f"{name} class distribution drifts {drift:.4f} from natural — expected <0.01"
        )


def test_no_duplicate_image_paths_within_a_split(eyepacs):
    """Physical oversampling would show up here first."""
    for name, df in eyepacs.items():
        assert not df["image_path"].duplicated().any(), f"{name} contains duplicate rows"


# ---------------------------------------------------------------------------------
# Reproducibility (R6)
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLITS + tuple(f"aptos_{s}" for s in SPLITS))
def test_seed_recorded_in_csv_header(name):
    """The seed must be traceable from the artefact itself, not just from config."""
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    header = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("#")]
    assert any("RNG seed" in l for l in header), f"{name}.csv header records no seed"

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = cfg["split"]["seed"]
    assert any(f"RNG seed: {seed}" in l for l in header), (
        f"{name}.csv header seed does not match configs/base.yaml split.seed={seed}"
    )


def test_split_proportions_close_to_config(eyepacs):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    n_patients = allrows["patient_id"].nunique()

    for name in SPLITS:
        want = cfg["split"][name]
        got = eyepacs[name]["patient_id"].nunique() / n_patients
        assert abs(got - want) < 0.01, f"{name}: {got:.4f} patients vs configured {want}"


# ---------------------------------------------------------------------------------
# Reconciliation against the SOURCE, not against a hardcoded integer.
#
# Added after the leakage-auditor review: the count tests above would pass on a
# manifest that is complete but systematically wrong (mangled labels, mis-parsed
# patient_id). These re-derive the truth from trainLabels.csv.
# ---------------------------------------------------------------------------------

SOURCE_CSV = REPO / "data" / "raw" / "eyepacs" / "trainLabels.csv"


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        pytest.skip(f"{SOURCE_CSV} not present (gitignored; refetch from Kaggle)")
    df = pd.read_csv(SOURCE_CSV, dtype={"image": "string"})
    df["level"] = df["level"].astype(int)
    return df


def test_image_set_matches_source_exactly(eyepacs, source):
    """Set equality both directions — not just a matching count."""
    got = set(pd.concat(eyepacs.values(), ignore_index=True)["image_path"])
    want = {f"data/data/{img}.jpeg" for img in source["image"]}
    assert got == want, (
        f"{len(want - got)} source images missing from splits, "
        f"{len(got - want)} images in splits not in source"
    )


def test_labels_match_source(eyepacs, source):
    """A complete-but-mislabelled manifest must fail here."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    allrows["image"] = (
        allrows["image_path"].str.rsplit("/", n=1).str[-1].str.replace(".jpeg", "", regex=False)
    )
    merged = allrows.merge(source, on="image", how="inner", validate="one_to_one")
    assert len(merged) == len(source)
    mismatched = merged[merged["label"] != merged["level"]]
    assert mismatched.empty, f"{len(mismatched)} rows carry a label differing from source"


def test_patient_id_and_eye_parse_matches_source(eyepacs, source):
    """Independent re-parse of {patientID}_{eye} from the source filenames."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    allrows["image"] = (
        allrows["image_path"].str.rsplit("/", n=1).str[-1].str.replace(".jpeg", "", regex=False)
    )
    parts = source["image"].str.partition("_")
    expected = pd.DataFrame({
        "image": source["image"],
        "want_patient": parts[0],
        "want_eye": parts[2],
    })
    merged = allrows.merge(expected, on="image", validate="one_to_one")
    assert (merged["patient_id"] == merged["want_patient"]).all()
    assert (merged["eye"] == merged["want_eye"]).all()


def test_stratified_on_max_grade_not_min_or_first(eyepacs, source):
    """Rule out the plausible wrong stratifications, rather than merely asserting the
    right one. Only max-grade reproduces the observed per-stratum patient counts."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    f_train, f_val = cfg["split"]["train"], cfg["split"]["val"]

    per_patient = allrows.groupby("patient_id").agg(
        max_grade=("label", "max"),
        min_grade=("label", "min"),
        split=("split", "first"),
    )

    for key, should_match in (("max_grade", True), ("min_grade", False)):
        matches = True
        for grade, grp in per_patient.groupby(key):
            n = len(grp)
            want_train = int(round(f_train * n))
            want_val = int(round(f_val * n))
            got = grp["split"].value_counts()
            if (got.get("train", 0) != want_train) or (got.get("val", 0) != want_val):
                matches = False
                break
        if should_match:
            assert matches, "per-stratum counts do not match a max-grade stratification"
        else:
            assert not matches, (
                "counts are ALSO consistent with min-grade stratification — the test "
                "cannot distinguish them, so it proves nothing"
            )


def test_asymmetric_patients_are_co_located(eyepacs):
    """The 2,240 patients whose eyes disagree are exactly the case R1 protects."""
    allrows = pd.concat(eyepacs.values(), ignore_index=True)
    g = allrows.groupby("patient_id").agg(
        spread=("label", lambda s: s.max() - s.min()),
        n_splits=("split", "nunique"),
    )
    asym = g[g["spread"] > 0]
    assert len(asym) == 2_240, f"expected 2,240 asymmetric patients, found {len(asym)}"
    assert (asym["n_splits"] == 1).all(), "an asymmetric patient spans multiple splits"


def test_split_is_reproducible(tmp_path):
    """Re-run the splitter into a temp dir and require identical assignments.

    Guards R6: a change to the RNG, the stratum ordering, or a library upgrade that
    perturbs the permutation stream would silently invalidate every committed result.
    """
    import subprocess
    import sys as _sys

    if not SOURCE_CSV.exists():
        pytest.skip("source labels CSV not present")

    r = subprocess.run(
        [_sys.executable, "-m", "src.data.split",
         "--config", str(CONFIG), "--out-dir", str(tmp_path), "--skip-aptos"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"splitter failed:\n{r.stdout}\n{r.stderr}"

    for name in SPLITS:
        committed = pd.read_csv(SPLIT_DIR / f"{name}.csv", comment="#",
                                dtype={"patient_id": "string"})
        fresh = pd.read_csv(tmp_path / f"{name}.csv", comment="#",
                            dtype={"patient_id": "string"})
        pd.testing.assert_frame_equal(committed, fresh, check_like=False)

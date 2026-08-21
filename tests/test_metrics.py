"""Tests for src/eval/metrics.py.

Every number in the thesis comes through this module (R4), so the tests are written
against hand-computable cases and against sklearn where sklearn has an equivalent —
never against this module's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    accuracy,
    balanced_accuracy,
    bootstrap_ci,
    compute_all,
    confusion_matrix,
    detect_collapse,
    per_class_recall,
    quadratic_weighted_kappa,
    recall_of_grade,
    referable_metrics,
)


# ----------------------------------------------------------------------------------
# QWK
# ----------------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    y = [0, 1, 2, 3, 4, 0, 2, 4]
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_the_majority_class_predictor_scores_zero():
    """The whole reason QWK is the primary metric. On a 73.5% grade-0 set, always-0
    scores 73.5% accuracy — and 0.0 QWK, which is the honest number."""
    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1, 2, 3, 4], size=2000, p=[.735, .07, .15, .025, .02])
    y_pred = np.zeros_like(y_true)

    assert accuracy(y_true, y_pred) > 0.70
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)


def test_qwk_punishes_distant_errors_far_more_than_near_ones():
    """A 4 called a 0 costs 16x a 4 called a 3. Accuracy scores them identically."""
    y_true = [4, 4, 4, 0, 0, 0, 2, 2]
    near = [3, 3, 3, 0, 0, 0, 2, 2]
    far = [0, 0, 0, 0, 0, 0, 2, 2]

    assert accuracy(y_true, near) == accuracy(y_true, far)
    assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


def test_qwk_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for _ in range(20):
        t = rng.integers(0, 5, 300)
        p = np.clip(t + rng.integers(-2, 3, 300), 0, 4)
        assert quadratic_weighted_kappa(t, p) == pytest.approx(
            sklearn.cohen_kappa_score(t, p, weights="quadratic", labels=range(5))
        )


def test_qwk_can_go_negative():
    """Systematically inverted predictions are worse than chance, and the metric says so
    rather than clamping at zero."""
    t = [0, 0, 0, 4, 4, 4]
    p = [4, 4, 4, 0, 0, 0]
    assert quadratic_weighted_kappa(t, p) < 0


def test_qwk_of_two_constant_sequences_is_zero_not_one():
    """Undefined, not perfect. Returning 1.0 would score an all-grade-0 predictor on an
    all-grade-0 subset as flawless — exactly the collapse this project screens for."""
    assert quadratic_weighted_kappa([0] * 10, [0] * 10) == 0.0


# ----------------------------------------------------------------------------------
# Confusion matrix — the fixed shape is the point
# ----------------------------------------------------------------------------------

def test_confusion_matrix_is_always_5x5_even_when_a_grade_is_absent():
    """A grade in neither y_true nor y_pred still gets its row and column. sklearn's
    default drops it, shrinking the matrix and shifting every index after it."""
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
    assert cm.shape == (5, 5)
    assert cm.sum() == 4
    assert cm[2:].sum() == 0


def test_confusion_matrix_orientation_is_truth_by_prediction():
    cm = confusion_matrix([4], [0])
    assert cm[4, 0] == 1 and cm[0, 4] == 0


def test_confusion_matrix_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    t, p = rng.integers(0, 5, 400), rng.integers(0, 5, 400)
    assert np.array_equal(
        confusion_matrix(t, p),
        sklearn.confusion_matrix(t, p, labels=range(5)),
    )


# ----------------------------------------------------------------------------------
# Accuracy / balanced accuracy / recall
# ----------------------------------------------------------------------------------

def test_balanced_accuracy_is_not_fooled_by_the_majority_class():
    y_true = np.array([0] * 90 + [4] * 10)
    y_pred = np.zeros(100, dtype=int)
    assert accuracy(y_true, y_pred) == pytest.approx(0.90)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.50)


def test_balanced_accuracy_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    t = rng.integers(0, 5, 500)
    p = np.clip(t + rng.integers(-1, 2, 500), 0, 4)
    assert balanced_accuracy(t, p) == pytest.approx(sklearn.balanced_accuracy_score(t, p))


def test_recall_for_an_absent_grade_is_nan_not_zero():
    """'Got 0% of the grade-4 cases' and 'there were no grade-4 cases' are different
    facts. Averaging the second as zero would drag balanced accuracy down for a reason
    that has nothing to do with the model."""
    rec = per_class_recall([0, 1, 2], [0, 1, 2])
    assert np.isnan(rec[3]) and np.isnan(rec[4])
    assert balanced_accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------
# Referable DR
# ----------------------------------------------------------------------------------

def test_referable_split_is_at_grade_2():
    t = [0, 1, 2, 3, 4]
    p = [0, 1, 2, 3, 4]
    m = referable_metrics(t, p, threshold=2)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["tp"] == 3 and m["tn"] == 2


def test_referable_sensitivity_counts_the_missed_referrals():
    t = [2, 3, 4, 0, 0]
    p = [1, 3, 4, 0, 0]          # one referable case called non-referable
    m = referable_metrics(t, p)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["fn"] == 1


# ----------------------------------------------------------------------------------
# The detection rule (CLAUDE.md §2)
# ----------------------------------------------------------------------------------

def test_an_all_zero_confusion_column_is_caught():
    """The fatal one: a grade the system cannot flag even in principle."""
    t = [0, 1, 2, 3, 4] * 20
    p = [0, 1, 2, 3, 3] * 20      # grade 4 never predicted
    report = detect_collapse(t, p)
    assert report and "never predicted" in str(report) and "4" in str(report)


def test_accuracy_near_the_majority_rate_is_caught():
    rng = np.random.default_rng(0)
    t = rng.choice([0, 1, 2, 3, 4], size=1000, p=[.735, .07, .15, .025, .02])
    report = detect_collapse(t, np.zeros_like(t))
    assert report and "majority-class rate" in str(report)


def test_a_healthy_prediction_is_not_flagged():
    rng = np.random.default_rng(1)
    t = rng.choice([0, 1, 2, 3, 4], size=2000, p=[.4, .15, .2, .15, .1])
    p = t.copy()
    flip = rng.random(2000) < 0.15
    p[flip] = np.clip(t[flip] + rng.choice([-1, 1], flip.sum()), 0, 4)

    report = detect_collapse(t, p)
    assert not report, str(report)


def test_an_absent_grade_is_a_row_of_zeros_and_is_not_a_collapse():
    """An all-zero ROW is a property of the evaluation set, not of the model."""
    assert not detect_collapse([0, 1, 0, 1], [0, 1, 1, 1])


# ----------------------------------------------------------------------------------
# Bootstrap (DECISION-006)
# ----------------------------------------------------------------------------------

def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    t = rng.integers(0, 5, 400)
    p = np.clip(t + rng.integers(-1, 2, 400), 0, 4)

    out = bootstrap_ci(t, p, quadratic_weighted_kappa, n_boot=300, seed=1)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["lo"] < out["hi"]


def test_a_thin_rare_class_gets_a_wide_interval():
    """98 grade-4 test images. The interval has to show that a bare point estimate would
    imply a precision the data does not have (DECISION-006)."""
    rng = np.random.default_rng(2)
    t = np.array([0] * 3000 + [4] * 98)
    p = t.copy()
    miss = rng.choice(np.flatnonzero(t == 4), 30, replace=False)
    p[miss] = 0

    wide = bootstrap_ci(t, p, recall_of_grade(4), n_boot=400, seed=3)
    common = bootstrap_ci(t, p, recall_of_grade(0), n_boot=400, seed=3)
    assert (wide["hi"] - wide["lo"]) > 10 * (common["hi"] - common["lo"])


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(9)
    t = rng.integers(0, 5, 200)
    p = rng.integers(0, 5, 200)
    a = bootstrap_ci(t, p, quadratic_weighted_kappa, n_boot=200, seed=42)
    b = bootstrap_ci(t, p, quadratic_weighted_kappa, n_boot=200, seed=42)
    c = bootstrap_ci(t, p, quadratic_weighted_kappa, n_boot=200, seed=7)
    assert a == b and a["lo"] != c["lo"]


def test_undefined_resamples_are_counted_not_hidden():
    """A draw that happens to contain no grade-4 image makes that recall undefined. The
    count is reported rather than the NaN being silently dropped."""
    t = np.array([0] * 200 + [4] * 2)
    p = t.copy()
    out = bootstrap_ci(t, p, recall_of_grade(4), n_boot=200, seed=0)
    assert out["n_undefined"] > 0
    assert np.isfinite(out["lo"])


# ----------------------------------------------------------------------------------
# Input validation — a bad label must not become a plausible number
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[0, 1, 5], [-1, 0, 1], [0, 1, np.nan], [0.5, 1, 2]])
def test_invalid_labels_raise(bad):
    with pytest.raises(ValueError):
        quadratic_weighted_kappa(bad, [0, 1, 2])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        quadratic_weighted_kappa([0, 1, 2], [0, 1])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


# ----------------------------------------------------------------------------------
# The metrics.json payload (R4)
# ----------------------------------------------------------------------------------

def test_compute_all_is_json_serialisable_and_complete():
    import json

    rng = np.random.default_rng(4)
    t = rng.choice([0, 1, 2, 3, 4], size=800, p=[.735, .07, .15, .025, .02])
    p = np.clip(t + rng.integers(-1, 2, 800), 0, 4)

    out = compute_all(t, p, split="val", bootstrap_n=100)
    json.dumps(out)                        # must not raise

    for key in ("qwk", "accuracy", "balanced_accuracy", "confusion_matrix",
                "referable", "collapse", "qwk_ci", "rare_class_ci", "support"):
        assert key in out
    assert len(out["confusion_matrix"]) == 5
    assert set(out["rare_class_ci"]) == {"3", "4"}
    assert out["n"] == 800


def test_compute_all_records_a_collapse_rather_than_only_printing_it():
    """A collapsed run must be recorded as collapsed in the file the write-up quotes."""
    t = np.array([0] * 900 + [4] * 100)
    out = compute_all(t, np.zeros_like(t), split="test", bootstrap_n=50)
    assert out["collapse"]["collapsed"] is True
    assert out["predicted_counts"][4] == 0


# ----------------------------------------------------------------------------------
# Severity-aware collapse — DECISION-026
# ----------------------------------------------------------------------------------
#
# The Phase 3 baseline (arm A, ResNet18, 8 epochs) posted val QWK 0.6138 with a CI far
# from zero, accuracy 6pp above the majority rate, and four of five grades predicted --
# and never predicted grade 1. The old rule called that a collapse and blocked the phase.
#
# It is not a collapse. Grade 1 is BELOW the referable threshold, so folding it into
# grade 0 changes no referral decision, and QWK weights a 1-called-0 error at 1/16 of a
# 4-called-0 one. A never-predicted grade 4 is a completely different matter.

VAL_SUPPORT = [3882, 357, 788, 133, 108]      # the real committed val split


def _from_recall(support, recall, spill=0):
    """A y_true/y_pred pair with the given per-class recall, errors pushed to `spill`."""
    y_true, y_pred = [], []
    for g, (s, r) in enumerate(zip(support, recall)):
        hit = int(round(r * s))
        y_true += [g] * s
        y_pred += [g] * hit + [spill] * (s - hit)
    return np.array(y_true), np.array(y_pred)


def test_a_never_predicted_mild_grade_is_a_warning_not_a_collapse():
    """The Phase 3 baseline's actual shape."""
    y_true, y_pred = _from_recall(VAL_SUPPORT, [0.982, 0.0, 0.376, 0.316, 0.407])
    rep = detect_collapse(y_true, y_pred, referable_threshold=2)

    assert rep.collapsed is False
    assert rep.level == "warning"
    assert rep.warnings and "1 (Mild)" in rep.warnings[0]
    assert "below the referable threshold" in rep.warnings[0].lower()


def test_a_never_predicted_referable_grade_is_still_fatal():
    """Grade 4 is what the system exists to catch. Same shape, different grade."""
    y_true, y_pred = _from_recall(VAL_SUPPORT, [0.982, 0.30, 0.376, 0.316, 0.0])
    rep = detect_collapse(y_true, y_pred, referable_threshold=2)

    assert rep.collapsed is True
    assert any("4 (Proliferative)" in r for r in rep.reasons)
    assert any("cannot flag these cases" in r for r in rep.reasons)


@pytest.mark.parametrize("grade,fatal", [(0, True), (1, False), (2, True), (3, True),
                                         (4, True)])
def test_fatality_follows_the_referable_threshold(grade, fatal):
    recall = [0.9] * 5
    recall[grade] = 0.0
    y_true, y_pred = _from_recall(VAL_SUPPORT, recall, spill=1 if grade == 0 else 0)
    rep = detect_collapse(y_true, y_pred, referable_threshold=2)
    assert rep.collapsed is fatal, f"grade {grade}: {rep}"


def test_the_threshold_is_configurable_not_hardcoded():
    """`eval.referable_threshold` is config; the rule must follow it."""
    recall = [0.9, 0.9, 0.0, 0.9, 0.9]
    y_true, y_pred = _from_recall(VAL_SUPPORT, recall)
    assert detect_collapse(y_true, y_pred, referable_threshold=2).collapsed is True
    assert detect_collapse(y_true, y_pred, referable_threshold=3).collapsed is False


def test_predicting_only_one_class_is_still_the_degenerate_case():
    y_true = np.array([0] * 900 + [1] * 50 + [4] * 50)
    rep = detect_collapse(y_true, np.zeros_like(y_true))
    assert rep.collapsed
    assert any("every prediction is the same class" in r for r in rep.reasons)


def test_a_healthy_run_has_no_warnings_either():
    rng = np.random.default_rng(1)
    t = rng.choice([0, 1, 2, 3, 4], size=2000, p=[.4, .15, .2, .15, .1])
    p = t.copy()
    flip = rng.random(2000) < 0.15
    p[flip] = np.clip(t[flip] + rng.choice([-1, 1], flip.sum()), 0, 4)

    rep = detect_collapse(t, p)
    assert rep.level == "none" and not rep.warnings


def test_warnings_reach_metrics_json():
    """A warning nothing records is the same as no rule at all."""
    y_true, y_pred = _from_recall(VAL_SUPPORT, [0.982, 0.0, 0.376, 0.316, 0.407])
    out = compute_all(y_true, y_pred, split="val", bootstrap_n=50)

    assert out["collapse"]["collapsed"] is False
    assert out["collapse"]["level"] == "warning"
    assert out["collapse"]["warnings"]


def test_compute_all_passes_the_configured_threshold_through():
    recall = [0.9, 0.9, 0.0, 0.9, 0.9]
    y_true, y_pred = _from_recall(VAL_SUPPORT, recall)
    strict = compute_all(y_true, y_pred, split="val", referable_threshold=2,
                         bootstrap_n=20)
    loose = compute_all(y_true, y_pred, split="val", referable_threshold=3,
                        bootstrap_n=20)
    assert strict["collapse"]["collapsed"] is True
    assert loose["collapse"]["collapsed"] is False

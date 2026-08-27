"""External validation — the carried-over rule, and the DECISION-057 verdict.

The tests that matter here are the ones that check the API cannot be used the wrong way:
the easiest way to inflate an external number is to let the thresholds be re-fitted on the
external set, so that must be impossible rather than merely discouraged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.external import (NOT_GENERALISING_QWK, coverage_correlation,
                               evaluate_external, per_grade_recall, refit_reference,
                               sens_at_fixed_threshold, verdict)

CUTS = [0.5, 1.5, 2.5, 3.5]


def synthetic(n=600, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    return y + rng.normal(0, noise, n), y


# ------------------------------------------------------------- the carried-over rule

def test_evaluate_external_refuses_to_fit_its_own_cut_points():
    """THE TEST THAT PROTECTS THE CLAIM. Fitting on APTOS would make this an internal
    evaluation of APTOS wearing the name of external validation."""
    s, y = synthetic()
    with pytest.raises(ValueError, match="must come from EyePACS"):
        evaluate_external(s, y, cuts=None, threshold=1.5)
    with pytest.raises(ValueError, match="must come from EyePACS"):
        evaluate_external(s, y, cuts=CUTS, threshold=None)


def test_carried_over_result_uses_exactly_the_cuts_it_was_given():
    s, y = synthetic()
    out = evaluate_external(s, y, cuts=CUTS, threshold=1.5)
    assert out["cuts"] == CUTS
    assert "carried over" in out["decision_rule"]


def test_the_refit_reference_is_labelled_as_secondary():
    s, y = synthetic()
    assert "never the headline" in refit_reference(s, y)["decision_rule"]


def test_refitting_cannot_score_worse_than_the_carried_over_cuts():
    """Sanity on the instrument: re-fitting optimises QWK on this data, so it is an upper
    bound. If it ever came out lower the optimiser would be broken."""
    s, y = synthetic()
    carried = evaluate_external(s, y, cuts=CUTS, threshold=1.5)
    assert refit_reference(s, y)["qwk"] >= carried["qwk"] - 1e-9


def test_sensitivity_at_a_fixed_threshold_does_not_search():
    """A different threshold must give a different answer — proof nothing is optimised."""
    s, y = synthetic()
    a = sens_at_fixed_threshold(s, y, 1.0)
    b = sens_at_fixed_threshold(s, y, 2.5)
    assert a["sensitivity"] > b["sensitivity"]
    assert a["threshold"] == 1.0 and b["threshold"] == 2.5


def test_referable_counts_are_internally_consistent():
    s, y = synthetic()
    r = sens_at_fixed_threshold(s, y, 1.5)
    assert r["tp"] + r["fn"] == r["n_referable"]
    assert r["tp"] + r["fn"] + r["tn"] + r["fp"] == r["n"]


# ------------------------------------------------------------------ the verdict

def _mk(qwk, sens, rec):
    return ({"qwk": qwk, "per_grade_recall": rec},
            {"qwk": qwk, "sens_at_spec95": sens, "per_grade_recall": rec})


GOOD_REC = {0: 0.9, 1: 0.3, 2: 0.5, 3: 0.5, 4: 0.5}


def test_g1_a_collapsed_refit_qwk_does_not_generalise():
    c, r = _mk(0.55, 0.70, GOOD_REC)
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert v["criteria"]["G1_refit_qwk_below_0.60"]
    assert "DOES NOT GENERALISE" in v["verdict"]


def test_g2_a_collapsed_sensitivity_does_not_generalise():
    c, r = _mk(0.70, 0.55, GOOD_REC)
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert v["criteria"]["G2_refit_sens_below_0.60"]
    assert "DOES NOT GENERALISE" in v["verdict"]


def test_g3_catches_the_dangerous_shape_that_g1_and_g2_both_miss():
    """Severe grades collapse while grade 0 stays intact — the model is most reliable
    exactly where it matters least. Aggregate QWK and sensitivity can both look fine."""
    rec = {0: 0.95, 1: 0.4, 2: 0.5, 3: 0.10, 4: 0.05}
    c, r = _mk(0.70, 0.70, rec)
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert not v["criteria"]["G1_refit_qwk_below_0.60"]
    assert not v["criteria"]["G2_refit_sens_below_0.60"]
    assert v["criteria"]["G3_severe_collapse_with_intact_grade0"]
    assert "DOES NOT GENERALISE" in v["verdict"]


def test_g4_a_confirmed_shortcut_overrides_a_healthy_headline():
    """DECISION-057: the shortcut invalidates the claim REGARDLESS of the numbers."""
    c, r = _mk(0.72, 0.70, GOOD_REC)
    v = verdict(c, r, {"max_abs_r": 0.05}, {"max_abs_r": 0.35}, eyepacs_qwk=0.7563)
    assert v["criteria"]["G4_shortcut_confirmed"]
    assert "DOES NOT GENERALISE" in v["verdict"]


def test_a_shortcut_equally_present_on_both_datasets_is_a_confound_not_a_shortcut():
    """The third row of the DECISION-054 table: coverage may correlate with grade for a
    reason that is not a shortcut, and that shows on BOTH datasets."""
    c, r = _mk(0.72, 0.70, GOOD_REC)
    v = verdict(c, r, {"max_abs_r": 0.30}, {"max_abs_r": 0.30}, eyepacs_qwk=0.7563)
    assert not v["criteria"]["G4_shortcut_confirmed"]


def test_a_mostly_calibration_drop_generalises_with_a_stated_condition():
    c = {"qwk": 0.62, "per_grade_recall": GOOD_REC}
    r = {"qwk": 0.71, "sens_at_spec95": 0.68, "per_grade_recall": GOOD_REC}
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert "GENERALISES, with" in v["verdict"]
    assert "local recalibration" in v["verdict"]
    assert v["fraction_of_drop_recovered_by_refitting"] > 0.5


def test_the_middle_band_is_named_and_not_rounded_up():
    """0.60-0.65 is the band a motivated reader rounds toward the better neighbour."""
    c = {"qwk": 0.60, "per_grade_recall": GOOD_REC}
    r = {"qwk": 0.63, "sens_at_spec95": 0.65, "per_grade_recall": GOOD_REC}
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert v["verdict"] == "GENERALISES WEAKLY; not usable without further work"


def test_recovering_less_than_half_the_drop_is_not_usable():
    c = {"qwk": 0.60, "per_grade_recall": GOOD_REC}
    r = {"qwk": 0.66, "sens_at_spec95": 0.70, "per_grade_recall": GOOD_REC}
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert v["fraction_of_drop_recovered_by_refitting"] < 0.5
    assert "WEAKLY" in v["verdict"]


def test_the_verdict_always_carries_the_in_domain_caveat():
    c, r = _mk(0.72, 0.70, GOOD_REC)
    v = verdict(c, r, None, None, eyepacs_qwk=0.7563)
    assert "IN-DOMAIN" in v["note"]


# ------------------------------------------------------ the coverage diagnostic

def test_coverage_correlation_is_computed_within_grade_not_pooled():
    """Pooled, a coverage-grade relationship looks like a shortcut even when the model
    ignores coverage entirely. Within grade it does not."""
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(5), 100)
    coverage = 0.7 - 0.02 * y + rng.normal(0, 0.001, len(y))   # coverage tracks grade
    scores = y + rng.normal(0, 0.05, len(y))                   # model reads GRADE only
    out = coverage_correlation(scores, y, coverage)
    assert abs(out["max_abs_r"]) < 0.2, out["per_grade_r"]
    assert abs(np.corrcoef(scores, coverage)[0, 1]) > 0.9      # pooled would mislead


def test_coverage_correlation_detects_a_real_within_grade_dependence():
    rng = np.random.default_rng(2)
    y = np.repeat(np.arange(5), 100)
    coverage = rng.uniform(0.55, 0.75, len(y))
    scores = y + 8.0 * (coverage - 0.65)                       # model reads COVERAGE
    assert coverage_correlation(scores, y, coverage)["max_abs_r"] > 0.2


def test_no_retina_images_are_excluded_and_counted():
    """DECISION-052 again: NaN coverage is excluded from the statistic and reported."""
    y = np.repeat(np.arange(5), 20)
    coverage = np.full(len(y), 0.65)
    coverage[:3] = np.nan
    out = coverage_correlation(np.arange(len(y), dtype=float), y, coverage)
    assert out["n_excluded"] == 3
    assert out["n_used"] == len(y) - 3


def test_a_grade_with_too_few_usable_images_yields_nan_not_a_spurious_correlation():
    y = np.array([0, 0, 4])
    out = coverage_correlation(np.array([0.1, 0.2, 4.0]), y, np.array([0.6, 0.7, 0.65]))
    assert np.isnan(out["per_grade_r"][4])


def test_per_grade_recall_reports_every_grade():
    y = np.repeat(np.arange(5), 10)
    assert set(per_grade_recall(y, y)) == {0, 1, 2, 3, 4}
    assert all(v == 1.0 for v in per_grade_recall(y, y).values())

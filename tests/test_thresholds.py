"""Tests for src/eval/thresholds.py — DECISION-030.

Arm E posted the best QWK and the worst referable sensitivity while using the untuned
default cut points. These tests pin the two things that claim fixes: that optimising cut
points is a real operation with a measurable effect, and that the operating point is
chosen against a stated standard rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import quadratic_weighted_kappa
from src.eval.thresholds import (
    DEFAULT_CUTS,
    NICE_SENSITIVITY,
    NICE_SPECIFICITY,
    analyse,
    choose_operating_point,
    grades_from_scores,
    optimise_qwk_cuts,
    referable_scores,
    sensitivity_specificity_curve,
)

VAL_SUPPORT = [3882, 357, 788, 133, 108]


def _ordinal_scores(rng, support=VAL_SUPPORT, noise=0.8, shift=0.0):
    """Continuous outputs that correlate with the true grade, like a trained ordinal head."""
    y = np.concatenate([np.full(n, g) for g, n in enumerate(support)])
    s = y + rng.normal(0, noise, len(y)) + shift
    return s, y


# ----------------------------------------------------------------------------------
# Cut points
# ----------------------------------------------------------------------------------

def test_default_cuts_are_round_half_up():
    s = np.array([-0.4, 0.6, 1.7, 2.5, 3.9, 99.0])
    assert grades_from_scores(s, DEFAULT_CUTS).tolist() == [0, 1, 2, 3, 4, 4]


def test_optimising_cuts_never_makes_qwk_worse():
    rng = np.random.default_rng(0)
    s, y = _ordinal_scores(rng)
    base = quadratic_weighted_kappa(y, grades_from_scores(s, DEFAULT_CUTS))
    cuts, qwk = optimise_qwk_cuts(s, y)
    assert qwk >= base - 1e-12


def test_optimising_cuts_recovers_a_shifted_scale():
    """The case that matters: a head whose outputs are systematically offset. The default
    cuts are then simply in the wrong place, and QWK recovers when they move."""
    rng = np.random.default_rng(1)
    s, y = _ordinal_scores(rng, shift=1.2)

    base = quadratic_weighted_kappa(y, grades_from_scores(s, DEFAULT_CUTS))
    cuts, qwk = optimise_qwk_cuts(s, y)

    assert qwk > base + 0.05, f"default {base:.4f} -> optimised {qwk:.4f}"
    assert cuts == sorted(cuts), "cut points must stay ordered"


def test_cuts_stay_sorted():
    """Unsorted cuts still produce grades, but they scramble the ordinal structure QWK
    assumes — the optimiser must not be able to buy score that way."""
    rng = np.random.default_rng(2)
    s, y = _ordinal_scores(rng)
    cuts, _ = optimise_qwk_cuts(s, y)
    assert cuts == sorted(cuts)
    assert len(cuts) == 4


def test_optimisation_is_deterministic():
    rng = np.random.default_rng(3)
    s, y = _ordinal_scores(rng)
    assert optimise_qwk_cuts(s, y) == optimise_qwk_cuts(s, y)


def test_degenerate_scores_do_not_crash():
    y = np.array([0, 1, 2, 3, 4] * 4)
    cuts, qwk = optimise_qwk_cuts(np.zeros(len(y)), y)
    assert len(cuts) == 4 and np.isfinite(qwk)


# ----------------------------------------------------------------------------------
# Referable scores — both heads
# ----------------------------------------------------------------------------------

def test_softmax_referable_score_is_probability_mass_above_the_line():
    """Not argmax-then-threshold: that throws away what the decision is about."""
    logits = np.array([[10.0, 0, 0, 0, 0], [0, 0, 0, 0, 10.0], [0, 0, 5.0, 5.0, 0]])
    s = referable_scores(logits, "softmax")
    assert s[0] < 0.01
    assert s[1] > 0.99
    assert s[2] > 0.99          # mass on grades 2 and 3


def test_ordinal_referable_score_is_the_scalar_itself():
    s = referable_scores(np.array([[0.3], [3.7]]), "ordinal_regression")
    assert s.tolist() == [0.3, 3.7]


# ----------------------------------------------------------------------------------
# The operating point
# ----------------------------------------------------------------------------------

def test_the_curve_covers_every_achievable_split():
    y = np.array([0, 0, 2, 4])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    curve = sensitivity_specificity_curve(score, y)
    assert curve["n_referable"] == 2 and curve["n_not_referable"] == 2
    assert max(curve["sensitivity"]) == 1.0
    assert max(curve["specificity"]) == 1.0


def test_a_perfect_separator_meets_both_floors():
    y = np.concatenate([np.zeros(100, int), np.full(100, 3)])
    score = np.concatenate([np.zeros(100), np.ones(100)])
    op = choose_operating_point(sensitivity_specificity_curve(score, y))
    assert op["meets_both"] is not None
    assert op["meets_both"]["sensitivity"] == 1.0
    assert op["meets_both"]["specificity"] == 1.0


def test_an_unreachable_standard_is_reported_as_unreachable():
    """A model that cannot meet the standard must say so, not return its best guess."""
    rng = np.random.default_rng(4)
    y = np.concatenate([np.zeros(500, int), np.full(50, 3)])
    score = rng.normal(0, 1, 550)          # pure noise: no useful operating point
    op = choose_operating_point(sensitivity_specificity_curve(score, y))
    assert op["meets_both"] is None


def test_the_floors_are_the_nice_standard_and_are_overridable():
    assert (NICE_SENSITIVITY, NICE_SPECIFICITY) == (0.80, 0.95)

    y = np.concatenate([np.zeros(100, int), np.full(100, 3)])
    score = np.concatenate([np.zeros(100), np.ones(100)])
    strict = choose_operating_point(sensitivity_specificity_curve(score, y),
                                    min_sensitivity=0.999, min_specificity=0.999)
    assert strict["min_sensitivity"] == 0.999
    assert strict["meets_both"] is not None      # a perfect separator still clears it


def test_max_sens_at_spec_respects_its_constraint():
    rng = np.random.default_rng(5)
    y = np.concatenate([np.zeros(800, int), np.full(200, 3)])
    score = np.concatenate([rng.normal(0, 1, 800), rng.normal(1.5, 1, 200)])
    op = choose_operating_point(sensitivity_specificity_curve(score, y))
    if op["max_sens_at_spec"]:
        assert op["max_sens_at_spec"]["specificity"] >= NICE_SPECIFICITY - 1e-9
    if op["max_spec_at_sens"]:
        assert op["max_spec_at_sens"]["sensitivity"] >= NICE_SENSITIVITY - 1e-9


def test_trading_surplus_specificity_buys_sensitivity():
    """The Phase 4 observation: arm A sits at 0.98 specificity against a 0.95 floor and
    0.64 sensitivity against a 0.80 floor. Surplus specificity is spendable."""
    rng = np.random.default_rng(6)
    y = np.concatenate([np.zeros(900, int), np.full(100, 3)])
    score = np.concatenate([rng.normal(0, 1, 900), rng.normal(2.2, 1, 100)])
    curve = sensitivity_specificity_curve(score, y)

    se = np.asarray(curve["sensitivity"])
    sp = np.asarray(curve["specificity"])
    conservative = se[sp >= 0.99].max()
    relaxed = se[sp >= 0.95].max()
    assert relaxed > conservative, "loosening specificity must buy sensitivity"


# ----------------------------------------------------------------------------------
# analyse()
# ----------------------------------------------------------------------------------

def test_analyse_reports_both_answers_for_an_ordinal_head():
    rng = np.random.default_rng(7)
    s, y = _ordinal_scores(rng, shift=0.8)
    res = analyse(s.reshape(-1, 1), y, "ordinal_regression")

    assert res["default_cuts"] == list(DEFAULT_CUTS)
    assert res["qwk_optimal_cuts"] != res["default_cuts"]
    assert res["qwk_optimal_qwk"] >= res["default_qwk"]
    assert "operating_point" in res and "curve" in res
    assert res["default_referable"]["sensitivity"] >= 0.0


def test_analyse_gives_a_softmax_head_an_operating_point_too():
    """Every arm gets a sensitivity/specificity curve, not just arm E."""
    rng = np.random.default_rng(8)
    y = np.concatenate([np.zeros(400, int), np.full(100, 3)])
    logits = rng.normal(0, 1, (500, 5))
    logits[400:, 2:] += 2.0
    res = analyse(logits, y, "softmax")

    assert res["default_cuts"] is None
    assert "qwk_optimal_cuts" not in res      # cut points are meaningless for softmax
    assert res["operating_point"]["max_sens_at_spec"] is not None


def test_analyse_is_json_serialisable():
    import json

    rng = np.random.default_rng(9)
    s, y = _ordinal_scores(rng)
    json.dumps(analyse(s.reshape(-1, 1), y, "ordinal_regression"))


def test_missing_outputs_explain_how_to_produce_them(tmp_path):
    from src.eval.thresholds import load_outputs

    with pytest.raises(FileNotFoundError, match="src.eval.predict"):
        load_outputs(tmp_path)


def test_grades_from_scores_agrees_with_the_training_loop():
    """THE cross-check. `predictions_from` produced the grades in every run's
    metrics.json; if this module bucketises differently, its analysis is about a
    different model than the one that was trained."""
    import torch

    from src.train.losses import predictions_from

    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(2, 1.5, 500), np.array(DEFAULT_CUTS),
                             np.array([-5.0, 9.0])])

    mine = grades_from_scores(scores, DEFAULT_CUTS)
    theirs = predictions_from(torch.tensor(scores).reshape(-1, 1),
                              "ordinal_regression").numpy()
    assert np.array_equal(mine, theirs), (
        f"disagree at {np.flatnonzero(mine != theirs)[:5]}: "
        f"{scores[mine != theirs][:5]}"
    )


def test_agreement_holds_for_custom_cuts_too():
    import torch

    from src.train.losses import predictions_from

    cuts = [0.9, 1.8, 2.4, 3.9]
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(2, 1.5, 300), np.array(cuts)])

    mine = grades_from_scores(scores, cuts)
    theirs = predictions_from(torch.tensor(scores).reshape(-1, 1),
                              "ordinal_regression", cuts).numpy()
    assert np.array_equal(mine, theirs)


# ----------------------------------------------------------------------------------
# Matched-decision-rule comparison — DECISION-033
# ----------------------------------------------------------------------------------

def test_expected_grade_is_the_softmax_continuous_score():
    """argmax collapses "spread over 2 and 3" and "confident 2" to the same answer.
    The expected grade keeps them apart, which is what the cut points need."""
    from src.eval.compare_arms import continuous_score

    confident_2 = np.array([[0.0, 0.0, 20.0, 0.0, 0.0]])
    spread_23 = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]])
    assert confident_2.argmax(1)[0] == spread_23.argmax(1)[0] == 2
    assert continuous_score(spread_23)[0] > continuous_score(confident_2)[0] + 0.4


def test_continuous_score_passes_an_ordinal_output_through():
    from src.eval.compare_arms import continuous_score

    assert continuous_score(np.array([[1.5], [3.2]])).tolist() == [1.5, 3.2]


def test_split_half_reports_optimism_not_just_a_fitted_number():
    """Fitting cut points on the same data used to select an arm is only honest if the
    optimism is measured. A fitted number with no held-out check is a claim."""
    from src.eval.compare_arms import split_half_qwk

    rng = np.random.default_rng(0)
    s, y = _ordinal_scores(rng, shift=0.7)
    r = split_half_qwk(s, y, repeats=6)

    assert set(r) == {"in_sample", "held_out", "optimism"}
    assert r["optimism"] == pytest.approx(r["in_sample"] - r["held_out"])
    assert r["in_sample"] >= r["held_out"] - 0.05      # fitting should not help held-out


def test_paired_difference_calls_a_tie_a_tie():
    """Two arms with the same underlying signal must come back NOT separable, or the
    comparison would manufacture winners."""
    from src.eval.compare_arms import paired_difference

    rng = np.random.default_rng(1)
    s, y = _ordinal_scores(rng)
    other = s + rng.normal(0, 0.01, len(s))           # same model, trivial jitter

    d = paired_difference(s, other, y, repeats=30)
    assert d["separable"] is False
    assert d["lo"] < 0 < d["hi"]


def test_paired_difference_detects_a_real_gap():
    """And it must still separate arms that genuinely differ."""
    from src.eval.compare_arms import paired_difference

    rng = np.random.default_rng(2)
    good, y = _ordinal_scores(rng, noise=0.6)
    bad = y + rng.normal(0, 2.5, len(y))              # much noisier ordering

    d = paired_difference(good, bad, y, repeats=30)
    assert d["separable"] is True
    assert d["mean"] > 0

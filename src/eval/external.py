"""External validation on APTOS. Pre-registered in DECISION-054, judged by DECISION-057.

THE ONE RULE THAT MAKES THIS EXTERNAL VALIDATION RATHER THAN A SECOND FIT.

    The cut points and the operating-point threshold are CARRIED OVER from EyePACS
    validation, unchanged. Re-fitting either on APTOS is fitting on the external set,
    and it destroys the claim the phase exists to make.

`evaluate_external` therefore takes the EyePACS cut points as a REQUIRED argument. There
is no default and no "fit if absent" branch — the easiest way to inflate this number is to
let it quietly re-fit, so the API makes that impossible rather than merely discouraged.

A re-fitted figure IS also reported, by `refit_reference`, clearly labelled as secondary.
It is not decoration: the gap between carried-over and re-fitted separates **calibration**
shift from **discrimination** shift, and DECISION-057 draws the generalisation line on the
re-fitted number precisely because calibration is fixable by local recalibration and
discrimination collapse is not.

WHAT IS EXPECTED. A drop, and the drop is the finding. Balanced accuracy may RISE while
QWK falls — APTOS is 49.3% grade 0 against EyePACS's 73.5% — and that is not evidence of
generalisation. Recorded so the outcome is not misread in either direction.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import accuracy, balanced_accuracy, quadratic_weighted_kappa
from src.eval.thresholds import (grades_from_scores, optimise_qwk_cuts,
                                 sensitivity_specificity_curve)

# DECISION-057. Fixed before the number exists.
NOT_GENERALISING_QWK = 0.60          # G1 — below the Phase 3 baseline's 0.6138
NOT_GENERALISING_SENS = 0.60         # G2
G3_SEVERE_RECALL = 0.30              # G3 — dangerous shape, not uniform degradation
G3_GRADE0_RECALL = 0.90
USABLE_QWK = 0.65
USABLE_RECOVERY = 0.5                # re-fitting must recover half the drop
SHORTCUT_R = 0.20                    # G4
G4_MIN_N = 100                       # a flagged grade must carry this many images


def sens_at_fixed_threshold(scores: np.ndarray, y_true: np.ndarray,
                            threshold: float) -> dict:
    """Sensitivity and specificity for referable disease at a CARRIED-OVER threshold.

    Not `choose_operating_point`: that one searches for a threshold, which is exactly what
    must not happen on the external set.
    """
    scores, y_true = np.asarray(scores, float), np.asarray(y_true, int)
    pos = y_true >= 2
    pred = scores >= threshold
    tp, fn = int((pred & pos).sum()), int((~pred & pos).sum())
    tn, fp = int((~pred & ~pos).sum()), int((pred & ~pos).sum())
    return {"threshold": float(threshold),
            "sensitivity": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "n_referable": int(pos.sum()), "n": int(len(y_true))}


def per_grade_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    out = {}
    for g in range(5):
        m = y_true == g
        out[g] = float((y_pred[m] == g).mean()) if m.any() else float("nan")
    return out


def evaluate_external(scores: np.ndarray, y_true: np.ndarray, *,
                      cuts, threshold: float) -> dict:
    """The HEADLINE external result, using EyePACS decision rules unchanged.

    `cuts` and `threshold` are required. See the module docstring.
    """
    if cuts is None or threshold is None:
        raise ValueError(
            "cuts and threshold are required and must come from EyePACS validation. "
            "Fitting them on APTOS would make this an internal evaluation of APTOS "
            "wearing the name of external validation (DECISION-054)."
        )
    scores, y_true = np.asarray(scores, float), np.asarray(y_true, int)
    pred = grades_from_scores(scores, np.asarray(cuts, float))
    ref = sens_at_fixed_threshold(scores, y_true, threshold)
    return {
        "n": int(len(y_true)),
        "qwk": quadratic_weighted_kappa(y_true, pred),
        "accuracy": accuracy(y_true, pred),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "referable": ref,
        "per_grade_recall": per_grade_recall(y_true, pred),
        "cuts": [float(c) for c in cuts],
        "decision_rule": "carried over from EyePACS validation (DECISION-054)",
    }


def refit_reference(scores: np.ndarray, y_true: np.ndarray) -> dict:
    """SECONDARY: what the model could do on APTOS with cut points re-fitted there.

    Always labelled. This is not the external-validation number — it is the instrument
    that separates calibration shift from discrimination shift (DECISION-057).
    """
    scores, y_true = np.asarray(scores, float), np.asarray(y_true, int)
    cuts, _ = optimise_qwk_cuts(scores, y_true)
    cuts = np.asarray(cuts, float)
    pred = grades_from_scores(scores, cuts)
    curve = sensitivity_specificity_curve(scores, y_true)
    spec = np.asarray(curve["specificity"], float)
    sens = np.asarray(curve["sensitivity"], float)
    ok = spec >= 0.95
    best = float(np.nanmax(sens[ok])) if ok.any() else float("nan")
    return {
        "qwk": quadratic_weighted_kappa(y_true, pred),
        "accuracy": accuracy(y_true, pred),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "sens_at_spec95": best,
        "per_grade_recall": per_grade_recall(y_true, pred),
        "cuts": [float(c) for c in cuts],
        "decision_rule": "RE-FITTED ON APTOS — secondary figure, never the headline",
    }


def coverage_correlation(scores: np.ndarray, y_true: np.ndarray,
                         coverage: np.ndarray) -> dict:
    """Within-grade correlation between predicted score and retina-coverage fraction.

    The field-of-view shortcut diagnostic (DECISION-054). WITHIN grade, because coverage
    and grade could both relate to acquisition quality — a pooled correlation would show
    that as a shortcut when it is a confound.

    Images with no detectable retina carry NaN coverage and are excluded by the caller
    (DECISION-052).
    """
    scores = np.asarray(scores, float)
    y_true = np.asarray(y_true, int)
    coverage = np.asarray(coverage, float)
    ok = np.isfinite(coverage) & np.isfinite(scores)

    per_grade, n_used = {}, 0
    for g in range(5):
        m = ok & (y_true == g)
        n_used += int(m.sum())
        if m.sum() < 3 or scores[m].std() == 0 or coverage[m].std() == 0:
            per_grade[g] = float("nan")
            continue
        per_grade[g] = float(np.corrcoef(scores[m], coverage[m])[0, 1])

    finite = [abs(v) for v in per_grade.values() if np.isfinite(v)]
    return {"per_grade_r": per_grade,
            "max_abs_r": float(max(finite)) if finite else float("nan"),
            "median_abs_r": float(np.median(finite)) if finite else float("nan"),
            "n_used": n_used,
            "n_excluded": int((~ok).sum())}


def verdict(carried: dict, refit: dict, shortcut_eyepacs: dict | None,
            shortcut_aptos: dict | None, eyepacs_qwk: float) -> dict:
    """Apply DECISION-057. Returns the verdict and every criterion's state.

    Applies the rule rather than interpreting it — the whole point of having fixed it
    before the number existed.
    """
    # THREE cases, not two. The original code had `recovered = ... if drop > 0 else nan`
    # and the usable branch then read `not isfinite(recovered) or recovered >= 0.5`,
    # so an unknown recovery PASSED the criterion. That is the wrong default for a
    # criterion: "we could not compute this" must never satisfy a requirement.
    #
    # Phase 6 exposed it. APTOS QWK came in ABOVE EyePACS (0.7902 vs 0.7563), so the
    # drop was NEGATIVE and `recovered` was NaN — and the verdict was reached through
    # the escape hatch rather than on its merits (DECISION-058).
    drop = eyepacs_qwk - carried["qwk"]
    if drop > 0:
        recovered = (refit["qwk"] - carried["qwk"]) / drop
        recovery_state = "measured"
    elif drop <= 0:
        # VACUOUS, not unknown: there was no drop, so there is nothing to recover. The
        # criterion is satisfied because it does not apply, and it says so.
        recovered = float("nan")
        recovery_state = "not applicable — QWK did not drop"
    recovery_ok = (recovery_state.startswith("not applicable")
                   or (np.isfinite(recovered) and recovered >= USABLE_RECOVERY))

    rec = refit["per_grade_recall"]
    severe = [rec[g] for g in (3, 4) if np.isfinite(rec.get(g, float("nan")))]

    # G4 uses MAX |r| across grades, compared against the SAME statistic on EyePACS.
    # DECISION-054 wrote only "|r| >= 0.2 within grades", which is ambiguous between max
    # and median; the ambiguity is resolved to max in DECISION-058, on two grounds. Max
    # is the conservative choice for a safety check — it flags more readily. And the
    # selection-bias worry (the maximum of five noisy correlations exceeds 0.2 by chance)
    # is answered by requiring the flagged grade to carry `G4_MIN_N` images and by making
    # the trigger a BETWEEN-DATASET comparison on the same statistic, so a selection
    # effect present on both sides cancels.
    #
    # `g4_undecided` is NOT False. Until the EyePACS side exists this criterion has no
    # value, and reporting it as "ok" would be the same class of error as the NaN
    # recovery hatch.
    g4, g4_undecided = False, False
    if shortcut_aptos is None or shortcut_eyepacs is None:
        g4_undecided = True
    else:
        a, e = shortcut_aptos["max_abs_r"], shortcut_eyepacs["max_abs_r"]
        if not np.isfinite(a) or not np.isfinite(e):
            g4_undecided = True
        else:
            g4 = bool(a >= SHORTCUT_R and a > e)

    criteria = {
        "G1_refit_qwk_below_0.60": bool(refit["qwk"] < NOT_GENERALISING_QWK),
        "G2_refit_sens_below_0.60": bool(
            np.isfinite(refit["sens_at_spec95"])
            and refit["sens_at_spec95"] < NOT_GENERALISING_SENS),
        "G3_severe_collapse_with_intact_grade0": bool(
            severe and max(severe) < G3_SEVERE_RECALL
            and np.isfinite(rec.get(0, float("nan")))
            and rec[0] > G3_GRADE0_RECALL),
        "G4_shortcut_confirmed": g4,
    }
    fails = [k for k, v in criteria.items() if v]

    if fails:
        v = ("DOES NOT GENERALISE beyond its training population; the in-domain "
             "results should be read as an upper bound")
    elif refit["qwk"] >= USABLE_QWK and recovery_ok:
        v = ("GENERALISES, with a domain-shift penalty that is predominantly "
             "CALIBRATION. Deployment in a new population would require local "
             "recalibration on site data; the EyePACS thresholds are not transferable")
    else:
        v = "GENERALISES WEAKLY; not usable without further work"

    if g4_undecided:
        v += (" [PROVISIONAL: G4 is UNDECIDED — the EyePACS side of the coverage "
              "comparison has not been computed, so the shortcut criterion carries no "
              "value yet]")

    return {"verdict": v, "criteria": criteria, "failed": fails,
            "G4_undecided": bool(g4_undecided),
            "eyepacs_qwk": float(eyepacs_qwk), "carried_qwk": carried["qwk"],
            "refit_qwk": refit["qwk"], "drop": float(drop),
            "fraction_of_drop_recovered_by_refitting": float(recovered),
            "recovery_state": recovery_state,
            "recovery_criterion_satisfied": bool(recovery_ok),
            "note": ("'Deployable' is not available to this thesis in either direction: "
                     "the model fails the screening floor IN-DOMAIN at 0.7360 vs 0.80 "
                     "(DECISION-057).")}

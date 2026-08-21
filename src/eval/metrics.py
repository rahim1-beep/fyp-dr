"""Metrics. Quadratic Weighted Kappa is the primary one (CLAUDE.md §5).

Every number that reaches a document comes from here, through
`runs/<run_id>/metrics.json` (R4). Nothing in this module has a default that quietly
hides a broken model — the opposite: `detect_collapse` exists to make a collapsed model
impossible to mistake for a working one.

WHY QWK AND NOT ACCURACY. DR grades are ORDERED. Calling a grade-4 eye grade-3 is a
small error; calling it grade-0 is a catastrophic one, and accuracy scores them
identically. QWK weights each error by the SQUARE of the grade distance, so it is the
only headline metric that reflects what the mistake costs a patient. And on a set that
is 73.5% grade 0, a model that predicts 0 for everything scores 73.5% accuracy and
**0.0 QWK** — which is the honest number.

BALANCED ACCURACY IS REPORTED ALONGSIDE ACCURACY, ALWAYS (R2). Val and test keep their
natural imbalanced distribution, so plain accuracy on them is dominated by grade 0.

For grades 3 and 4, report bootstrap CIs, never bare point estimates: the test split
holds 133 grade-3 and 98 grade-4 images (DECISION-006).

`detect_collapse` is SEVERITY-AWARE (DECISION-026): a never-predicted grade at or above
the referable threshold is fatal, one below it is a warning. A model that cannot emit
grade 4 cannot flag proliferative disease; a model that folds grade 1 into grade 0
changes no referral decision and is what an unbalanced baseline looks like. A recall that moves by 0.01 per
image is not a point estimate.

Implemented on numpy rather than deferred to sklearn: QWK's weighting is three lines, and
the confusion matrix must be reindexed 0-4 in a way `sklearn.metrics.confusion_matrix`
does not do by default — a class missing from BOTH y_true and y_pred silently vanishes
from its output, shrinking the matrix and shifting every index after it.
`test_metrics.py` checks this module against sklearn where sklearn has an equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_CLASSES = 5
CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")


def _as_int_array(y, name: str) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if np.isnan(np.asarray(a, dtype=float)).any():
        raise ValueError(f"{name} contains NaN")
    out = a.astype(np.int64)
    if not np.array_equal(out, np.asarray(a, dtype=float)):
        raise ValueError(f"{name} contains non-integer values")
    if out.min() < 0 or out.max() >= N_CLASSES:
        raise ValueError(
            f"{name} has values outside 0-{N_CLASSES - 1}: "
            f"min {out.min()}, max {out.max()}"
        )
    return out


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Rows are truth, columns are prediction. ALWAYS n_classes x n_classes.

    Fixed shape is the point. A grade that appears in neither y_true nor y_pred still
    gets its row and column, because "the model never once predicted grade 4" is the
    single most important thing a confusion matrix can tell you here, and a matrix that
    silently shrinks to 4x4 hides it while shifting every index after it.
    """
    t = _as_int_array(y_true, "y_true")
    p = _as_int_array(y_pred, "y_pred")
    if len(t) != len(p):
        raise ValueError(f"length mismatch: y_true {len(t)}, y_pred {len(p)}")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    """Cohen's kappa with quadratic weights: 1.0 perfect, 0.0 chance, negative worse.

    w[i,j] = (i-j)^2 / (n-1)^2, so a 4-vs-0 error costs 16x a 4-vs-3 error.

    Returns 0.0 when the expected-disagreement denominator is zero, which happens when
    both sequences are a single constant class. That case is genuinely undefined rather
    than perfect, and returning 1.0 for it would score an all-grade-0 predictor on an
    all-grade-0 subset as flawless.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    n = cm.sum()

    i, j = np.indices((n_classes, n_classes))
    w = (i - j) ** 2 / (n_classes - 1) ** 2

    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / n

    denom = (w * expected).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (w * cm).sum() / denom)


def accuracy(y_true, y_pred) -> float:
    t = _as_int_array(y_true, "y_true")
    p = _as_int_array(y_pred, "y_pred")
    return float((t == p).mean())


def per_class_recall(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Recall per grade; NaN for a grade with no true examples.

    NaN rather than 0.0 on purpose: "the model got 0% of the grade-4 cases" and "there
    were no grade-4 cases" are different facts, and averaging the second as a zero drags
    balanced accuracy down for a reason that has nothing to do with the model.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diag(cm) / support
    return np.where(support > 0, rec, np.nan)


def balanced_accuracy(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    """Mean per-class recall over the classes that are actually present.

    Reported next to accuracy every time (R2). On a 73.5% grade-0 test set, accuracy
    rewards ignoring the rare classes and balanced accuracy does not.
    """
    return float(np.nanmean(per_class_recall(y_true, y_pred, n_classes)))


def referable_metrics(y_true, y_pred, threshold: int = 2) -> dict[str, float]:
    """Referable DR = grade >= threshold (2 by default, per configs/base.yaml).

    The screening question the deployed system actually answers: does this patient need
    to see an ophthalmologist? Sensitivity is the number that matters clinically — a
    missed referable case is the expensive error.
    """
    t = _as_int_array(y_true, "y_true") >= threshold
    p = _as_int_array(y_pred, "y_pred") >= threshold

    tp = int((t & p).sum())
    tn = int((~t & ~p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())

    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


# ----------------------------------------------------------------------------------
# The detection rule (CLAUDE.md §2) — stop and diagnose, do not train on
# ----------------------------------------------------------------------------------

@dataclass
class CollapseReport:
    """The verdict of the detection rule, at one of three levels.

    `collapsed` means FATAL — stop and diagnose. `warnings` are real findings that are
    not fatal, and they are reported rather than swallowed: a warning that nothing prints
    is the same as no rule at all.
    """

    collapsed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.collapsed:
            return "collapsed"
        return "warning" if self.warnings else "none"

    def __bool__(self) -> bool:          # `if detect_collapse(...):` means FATAL
        return self.collapsed

    def __str__(self) -> str:
        parts = list(self.reasons) + [f"[warning] {w}" for w in self.warnings]
        return "; ".join(parts) if parts else "no collapse detected"


def detect_collapse(y_true, y_pred, near_majority: float = 0.02,
                    referable_threshold: int = 2) -> CollapseReport:
    """CLAUDE.md §2, made severity-aware. DECISION-026.

    The rule exists to catch a model that has learned the PRIOR and nothing else — the
    one that predicts the majority class everywhere and still posts a respectable
    accuracy. Three things are fatal:

      1. Predictions concentrated in a single class. That is the degenerate case.
      2. Accuracy within `near_majority` of the majority-class rate.
      3. A grade AT OR ABOVE the referable threshold that is never predicted while it
         has support.

    (3) is the part that is not simply "an all-zero column". A never-predicted grade 4 is
    catastrophic: proliferative disease is the thing this system exists to catch, and a
    model that cannot emit that grade cannot flag it even in principle. A never-predicted
    grade 1 is a different animal — mild DR is BELOW the referable threshold, so
    absorbing it into grade 0 changes no referral decision, and QWK weights a 1-called-0
    error at 1/16 of a 4-called-0 error. It is the expected behaviour of an unbalanced
    baseline and it is the specific thing arms B–E exist to fix.

    So a never-predicted grade below the referable threshold is a WARNING. It is still
    reported, in the run log and in metrics.json, because it is the number that has to
    improve — it is just not a reason to halt a pipeline that is demonstrably working.

    An all-zero ROW means that grade was absent from this evaluation set, which is a
    property of the data, so a column that is zero only because the row is zero is not
    reported at all.
    """
    cm = confusion_matrix(y_true, y_pred)
    reasons: list[str] = []
    warnings: list[str] = []

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    # (1) the degenerate case
    used = [int(c) for c in range(N_CLASSES) if predicted[c] > 0]
    if len(used) <= 1:
        name = CLASS_NAMES[used[0]] if used else "nothing"
        reasons.append(
            f"every prediction is the same class: {used[0] if used else '-'} ({name}). "
            "The model has learned the prior and nothing else."
        )

    # (3) never predicted, split by whether it changes a referral
    never = [int(c) for c in range(N_CLASSES) if predicted[c] == 0 and support[c] > 0]
    fatal = [c for c in never if c >= referable_threshold]
    minor = [c for c in never if c < referable_threshold]

    if fatal:
        names = ", ".join(f"{c} ({CLASS_NAMES[c]})" for c in fatal)
        reasons.append(
            f"referable grade(s) never predicted: {names}. At or above the referable "
            f"threshold ({referable_threshold}), so the system cannot flag these cases "
            "even in principle."
        )
    if minor:
        names = ", ".join(f"{c} ({CLASS_NAMES[c]})" for c in minor)
        warnings.append(
            f"grade(s) never predicted: {names}. Below the referable threshold "
            f"({referable_threshold}), so no referral decision changes; this is the "
            "characteristic weakness of an unbalanced baseline and what the imbalance "
            "arms are measured against."
        )

    # (2) accuracy indistinguishable from predicting the majority class
    if support.sum():
        majority_rate = float(support.max() / support.sum())
        acc = accuracy(y_true, y_pred)
        if acc <= majority_rate + near_majority:
            reasons.append(
                f"accuracy {acc:.4f} is within {near_majority} of the majority-class "
                f"rate {majority_rate:.4f} — predicting the majority class for "
                "everything would score about the same"
            )

    return CollapseReport(collapsed=bool(reasons), reasons=reasons, warnings=warnings)


# ----------------------------------------------------------------------------------
# Bootstrap CIs — mandatory for grades 3 and 4 (DECISION-006)
# ----------------------------------------------------------------------------------

def bootstrap_ci(
    y_true,
    y_pred,
    metric,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap over IMAGES, resampled with replacement.

    The test split holds 133 grade-3 and 98 grade-4 images. At that support a single
    image moves recall by ~0.01, so a bare point estimate implies a precision the data
    does not have (DECISION-006).

    Resampling is over images, not patients. That is the right unit here because the
    metric is computed per image and the split is already patient-disjoint (R1) — no
    patient's two eyes can land on opposite sides of anything. It does slightly
    understate the interval when both eyes of a patient are in the set and correlated;
    that is recorded rather than hidden.

    Returns NaN bounds rather than raising if the metric is undefined on some resample
    (a grade absent from a draw), and reports how many draws that was.
    """
    t = _as_int_array(y_true, "y_true")
    p = _as_int_array(y_pred, "y_pred")
    if len(t) != len(p):
        raise ValueError(f"length mismatch: y_true {len(t)}, y_pred {len(p)}")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = np.random.default_rng(seed)
    n = len(t)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric(t[idx], p[idx])

    finite = vals[np.isfinite(vals)]
    alpha = (1.0 - ci) / 2.0
    if finite.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": float(n_boot), "n_undefined": float(n_boot), "ci": ci}

    return {
        "point": float(metric(t, p)),
        "lo": float(np.quantile(finite, alpha)),
        "hi": float(np.quantile(finite, 1.0 - alpha)),
        "n_boot": float(n_boot),
        "n_undefined": float(n_boot - finite.size),
        "ci": ci,
    }


def recall_of_grade(grade: int):
    """A metric callable for `bootstrap_ci`, for one grade's recall."""
    def _m(t, p):
        return float(per_class_recall(t, p)[grade])
    _m.__name__ = f"recall_grade_{grade}"
    return _m


# ----------------------------------------------------------------------------------
# The bundle that becomes metrics.json (R4)
# ----------------------------------------------------------------------------------

def compute_all(
    y_true,
    y_pred,
    *,
    split: str,
    referable_threshold: int = 2,
    bootstrap_n: int = 1000,
    bootstrap_ci_level: float = 0.95,
    bootstrap_grades: tuple[int, ...] = (3, 4),
    seed: int = 42,
) -> dict:
    """Everything for one evaluation, JSON-serialisable, ready for metrics.json.

    `qwk` first because it is the primary metric. `collapse` is in the payload rather
    than printed and forgotten, so a collapsed run is recorded as collapsed in the file
    that the write-up quotes from.
    """
    t = _as_int_array(y_true, "y_true")
    p = _as_int_array(y_pred, "y_pred")
    cm = confusion_matrix(t, p)
    collapse = detect_collapse(t, p, referable_threshold=referable_threshold)

    out = {
        "split": split,
        "n": int(len(t)),
        "qwk": quadratic_weighted_kappa(t, p),
        "accuracy": accuracy(t, p),
        "balanced_accuracy": balanced_accuracy(t, p),
        "per_class_recall": [None if np.isnan(v) else float(v)
                             for v in per_class_recall(t, p)],
        "support": [int(v) for v in cm.sum(axis=1)],
        "predicted_counts": [int(v) for v in cm.sum(axis=0)],
        "confusion_matrix": cm.tolist(),
        "referable": referable_metrics(t, p, referable_threshold),
        "collapse": {"collapsed": collapse.collapsed, "level": collapse.level,
                     "reasons": collapse.reasons, "warnings": collapse.warnings},
        "class_names": list(CLASS_NAMES),
    }

    out["qwk_ci"] = bootstrap_ci(t, p, quadratic_weighted_kappa,
                                 n_boot=bootstrap_n, ci=bootstrap_ci_level, seed=seed)
    out["rare_class_ci"] = {
        str(g): bootstrap_ci(t, p, recall_of_grade(g), n_boot=bootstrap_n,
                             ci=bootstrap_ci_level, seed=seed)
        for g in bootstrap_grades
    }
    return out

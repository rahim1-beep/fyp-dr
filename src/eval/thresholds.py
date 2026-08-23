"""Turn a model's continuous output into grades, and choose the operating point.

VALIDATION ONLY (R3). Every threshold here is fitted on validation scores and never on
test. A cut point chosen on test is test-set tuning wearing a different hat.

TWO ANSWERS, NOT ONE (DECISION-030). A single set of cut points cannot serve both
purposes, and collapsing them hides the trade-off the ablation exists to expose:

  * `qwk_optimal_cuts`  — the four cut points that maximise QWK. This is the grading
    answer: how well does the model agree with the grader across the whole ordinal scale.
  * `operating_point`   — the single referable-vs-not decision threshold, chosen against
    a screening standard. This is the clinical answer: does this patient get referred.

Arm E is the reason this exists. It posted the ablation's best QWK (0.7081) and its worst
referable sensitivity (0.5598) while using the **untuned default cut points**
`[0.5, 1.5, 2.5, 3.5]`. That sensitivity is a property of an arbitrary threshold, not of
the arm — and for an ordinal head the operating point is a free dial that nobody had
turned.

BOTH HEADS ARE SUPPORTED, because the operating-point question is not special to arm E:

  * `ordinal`  — one scalar per image; four cut points map it to 0-4, and the referable
    decision is the third cut.
  * `softmax`  — five probabilities per image; grades stay `argmax`, and the referable
    decision is a threshold on P(grade >= 2). Every arm therefore gets a
    sensitivity/specificity curve, not just arm E.

    python -m src.eval.thresholds --run-dir runs/phase4_arm_e_resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval.metrics import quadratic_weighted_kappa, referable_metrics

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CUTS = (0.5, 1.5, 2.5, 3.5)

# The UK screening standard, as a pair. Both halves are required: sensitivity alone is
# trivially satisfiable by referring everyone. See DECISION-032 for the citations.
NICE_SENSITIVITY = 0.80
NICE_SPECIFICITY = 0.95


def grades_from_scores(scores: np.ndarray, cuts) -> np.ndarray:
    """Continuous ordinal output -> integer grades.

    `side="right"` so a score sitting exactly ON a cut rounds UP, which with the default
    midpoints is round-half-up. This MUST match
    `src.train.losses.predictions_from`, which uses `torch.bucketize(..., right=True)`:
    if the two disagree, the grades analysed here are not the grades the training loop
    produced, and every number downstream is about a different model.
    `test_thresholds.py` asserts the two agree on random input.
    """
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(scores, dtype=float),
                           side="right").clip(0, 4)


def optimise_qwk_cuts(scores: np.ndarray, y_true: np.ndarray,
                      init=DEFAULT_CUTS, rounds: int = 12,
                      grid: int = 101) -> tuple[list[float], float]:
    """Coordinate ascent on QWK over the four cut points. Validation scores only.

    Coordinate ascent rather than a general optimiser because QWK as a function of a cut
    point is piecewise constant — it only changes when a cut crosses a sample — so
    gradient methods have nothing to work with and a grid over each coordinate in turn is
    both exact enough and trivially reproducible.

    The cuts are kept sorted after every move. Unsorted cuts still produce grades via
    `searchsorted`, but they are uninterpretable and would let the optimiser find a
    "better" QWK by scrambling the ordinal structure the metric assumes.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)
    cuts = list(map(float, init))
    best = quadratic_weighted_kappa(y_true, grades_from_scores(scores, cuts))

    lo, hi = float(scores.min()), float(scores.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return cuts, best

    for _ in range(rounds):
        improved = False
        for i in range(4):
            for candidate in np.linspace(lo, hi, grid):
                trial = list(cuts)
                trial[i] = float(candidate)
                trial.sort()
                score = quadratic_weighted_kappa(y_true, grades_from_scores(scores, trial))
                if score > best + 1e-12:
                    best, cuts, improved = score, trial, True
        if not improved:
            break
    return [round(c, 6) for c in cuts], float(best)


def referable_scores(outputs: np.ndarray, head: str) -> np.ndarray:
    """One scalar per image: higher means more likely to need referral.

    For a softmax head that is P(grade >= 2) — the probability mass above the referral
    line, which is the quantity the decision is actually about. Taking `argmax` and then
    thresholding would throw away exactly the information an operating point needs.
    """
    outputs = np.asarray(outputs, dtype=float)
    if head == "ordinal_regression":
        return outputs.reshape(-1)

    x = outputs - outputs.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, 2:].sum(axis=1)


def sensitivity_specificity_curve(score: np.ndarray, y_true: np.ndarray,
                                  threshold: int = 2) -> dict:
    """Sensitivity and specificity at every distinct threshold of `score`.

    Returned in full so the trade-off is visible rather than buried in one number
    (DECISION-030). The candidate thresholds are the midpoints between adjacent observed
    scores: any threshold between two samples gives an identical split, so this is the
    complete set of achievable operating points and not a grid approximation.
    """
    score = np.asarray(score, dtype=float).reshape(-1)
    referable = np.asarray(y_true) >= threshold

    order = np.unique(score)
    if len(order) > 1:
        cand = np.concatenate(([order[0] - 1e-6],
                               (order[:-1] + order[1:]) / 2.0,
                               [order[-1] + 1e-6]))
    else:
        cand = np.array([order[0] - 1e-6, order[0] + 1e-6])

    n_pos, n_neg = int(referable.sum()), int((~referable).sum())
    sens, spec = [], []
    for t in cand:
        pred = score >= t
        tp = int((pred & referable).sum())
        tn = int((~pred & ~referable).sum())
        sens.append(tp / n_pos if n_pos else float("nan"))
        spec.append(tn / n_neg if n_neg else float("nan"))

    return {"threshold": cand.tolist(), "sensitivity": sens, "specificity": spec,
            "n_referable": n_pos, "n_not_referable": n_neg}


def choose_operating_point(curve: dict,
                           min_sensitivity: float = NICE_SENSITIVITY,
                           min_specificity: float = NICE_SPECIFICITY) -> dict:
    """The operating point against a screening standard, and what it costs.

    Reports three things, because the honest answer depends on which constraint binds and
    on whether either is reachable at all:

      * `meets_both`      — the point satisfying BOTH floors, maximising sensitivity
                            among those. Empty if the model cannot reach the standard.
      * `max_sens_at_spec`— highest sensitivity subject to specificity >= min_specificity.
                            This is the one to quote when the standard is out of reach:
                            it says how close the model gets while staying deployable on
                            the specificity side.
      * `max_spec_at_sens`— highest specificity subject to sensitivity >= min_sensitivity,
                            i.e. what specificity costs to meet the sensitivity floor.
    """
    t = np.asarray(curve["threshold"])
    se = np.asarray(curve["sensitivity"])
    sp = np.asarray(curve["specificity"])

    def pick(mask, maximise):
        if not mask.any():
            return None
        idx = np.flatnonzero(mask)
        best = idx[np.argmax(maximise[idx])]
        return {"threshold": float(t[best]), "sensitivity": float(se[best]),
                "specificity": float(sp[best])}

    return {
        "min_sensitivity": min_sensitivity,
        "min_specificity": min_specificity,
        "meets_both": pick((se >= min_sensitivity) & (sp >= min_specificity), se),
        "max_sens_at_spec": pick(sp >= min_specificity, se),
        "max_spec_at_sens": pick(se >= min_sensitivity, sp),
    }


def analyse(outputs: np.ndarray, y_true: np.ndarray, head: str,
            referable_threshold: int = 2,
            min_sensitivity: float = NICE_SENSITIVITY,
            min_specificity: float = NICE_SPECIFICITY) -> dict:
    """Both answers for one run's validation outputs."""
    y_true = np.asarray(y_true)
    out = {"head": head, "n": int(len(y_true)),
           "referable_threshold": referable_threshold}

    if head == "ordinal_regression":
        scores = np.asarray(outputs, dtype=float).reshape(-1)
        default_grades = grades_from_scores(scores, DEFAULT_CUTS)
        cuts, qwk = optimise_qwk_cuts(scores, y_true)
        out["default_cuts"] = list(DEFAULT_CUTS)
        out["default_qwk"] = quadratic_weighted_kappa(y_true, default_grades)
        out["default_referable"] = referable_metrics(y_true, default_grades,
                                                     referable_threshold)
        out["qwk_optimal_cuts"] = cuts
        out["qwk_optimal_qwk"] = qwk
        out["qwk_optimal_referable"] = referable_metrics(
            y_true, grades_from_scores(scores, cuts), referable_threshold)
    else:
        grades = np.asarray(outputs, dtype=float).argmax(axis=1)
        out["default_cuts"] = None
        out["default_qwk"] = quadratic_weighted_kappa(y_true, grades)
        out["default_referable"] = referable_metrics(y_true, grades, referable_threshold)

    curve = sensitivity_specificity_curve(referable_scores(outputs, head), y_true,
                                          referable_threshold)
    out["operating_point"] = choose_operating_point(curve, min_sensitivity,
                                                    min_specificity)
    # The full curve is large; keep a decimated copy for plotting and the write-up.
    step = max(1, len(curve["threshold"]) // 200)
    out["curve"] = {k: (v[::step] if isinstance(v, list) else v)
                    for k, v in curve.items()}
    return out


def load_outputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """(outputs, y_true, head) from a run's saved validation outputs."""
    run_dir = Path(run_dir)
    npz = run_dir / "val_outputs.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} does not exist. Threshold optimisation needs the model's CONTINUOUS "
            "validation outputs, and metrics.json only stores integer predictions. "
            "Produce them with `python -m src.eval.predict --run-dir <dir>`, which "
            "reloads the checkpoint and re-runs validation."
        )
    d = np.load(npz)
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return d["outputs"], d["y_true"], cfg.get("model", {}).get("head", "softmax")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", type=Path, required=True, action="append",
                    help="runs/<run_id>/, holding val_outputs.npz and config.yaml; "
                         "repeatable, and with more than one a comparison table is "
                         "printed")
    ap.add_argument("--referable-threshold", type=int, default=2)
    ap.add_argument("--min-sensitivity", type=float, default=NICE_SENSITIVITY)
    ap.add_argument("--min-specificity", type=float, default=NICE_SPECIFICITY)
    ap.add_argument("--out", type=Path,
                    help="write the analysis here (default: <run-dir>/thresholds.json)")
    return ap


def comparison_table(results: dict[str, dict]) -> str:
    """One row per run: the grading answer, the clinical answer, and which floor binds.

    This is the table that decides whether an arm's ranking survives contact with the
    operating point, so it names the binding constraint explicitly rather than leaving it
    to be inferred from two numbers.
    """
    cols = (f"{'run':<32}{'QWK':>8}{'QWK*':>8}{'sens@spec':>11}{'spec@sens':>11}"
            f"  {'binding floor':<18}{'meets both':>11}")
    lines = [cols, "-" * len(cols)]

    for name, r in results.items():
        op = r["operating_point"]
        at_spec = op["max_sens_at_spec"]        # highest sens with spec >= floor
        at_sens = op["max_spec_at_sens"]        # highest spec with sens >= floor

        if op["meets_both"] is not None:
            binds, meets = "neither", "YES"
        elif at_spec is None and at_sens is None:
            binds, meets = "both unreachable", "no"
        elif at_spec is None:
            binds, meets = "specificity", "no"
        elif at_sens is None:
            binds, meets = "sensitivity", "no"
        else:
            # whichever floor the model falls further short of is the one doing the work
            short_sens = op["min_sensitivity"] - at_spec["sensitivity"]
            short_spec = op["min_specificity"] - at_sens["specificity"]
            binds = "sensitivity" if short_sens > short_spec else "specificity"
            meets = "no"

        qwk_opt = r.get("qwk_optimal_qwk")
        cell_opt = f"{qwk_opt:.4f}" if qwk_opt is not None else "-"
        cell_sens = f"{at_spec['sensitivity']:.4f}" if at_spec else "-"
        cell_spec = f"{at_sens['specificity']:.4f}" if at_sens else "-"

        lines.append(
            f"{name:<32}{r['default_qwk']:>8.4f}{cell_opt:>8}"
            f"{cell_sens:>11}{cell_spec:>11}  {binds:<18}{meets:>11}"
        )

    lines += [
        "",
        "QWK       as run: default cuts for an ordinal head, argmax for softmax",
        "QWK*      after optimising cut points on VALIDATION (ordinal heads only)",
        "sens@spec highest sensitivity while specificity stays above the floor",
        "spec@sens highest specificity while sensitivity stays above the floor",
    ]
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    if len(args.run_dir) > 1:
        results = {}
        for run_dir in args.run_dir:
            outputs, y_true, head = load_outputs(run_dir)
            results[run_dir.name] = analyse(outputs, y_true, head,
                                            args.referable_threshold,
                                            args.min_sensitivity, args.min_specificity)
            out = run_dir / "thresholds.json"
            out.write_text(json.dumps(results[run_dir.name], indent=2), encoding="utf-8")

        print(f"floor: sensitivity >= {args.min_sensitivity:.2f} AND specificity >= "
              f"{args.min_specificity:.2f}")
        print("  [PROVISIONAL - DECISION-032. The applicable standard is not yet")
        print("   confirmed; do not quote it in the write-up until it is.]\n")
        print(comparison_table(results))
        return 0

    run_dir = args.run_dir[0]
    outputs, y_true, head = load_outputs(run_dir)

    res = analyse(outputs, y_true, head, args.referable_threshold,
                  args.min_sensitivity, args.min_specificity)

    print(f"run   : {run_dir.name}")
    print(f"head  : {head}   n = {res['n']}   VALIDATION ONLY (R3)")
    print(f"\ngrading — as-run QWK {res['default_qwk']:.4f}")
    if res.get("qwk_optimal_cuts"):
        print(f"  default cuts {res['default_cuts']} -> QWK {res['default_qwk']:.4f}, "
              f"sens {res['default_referable']['sensitivity']:.4f}")
        print(f"  QWK-optimal  {res['qwk_optimal_cuts']} -> QWK "
              f"{res['qwk_optimal_qwk']:.4f}, sens "
              f"{res['qwk_optimal_referable']['sensitivity']:.4f}")
        print(f"  optimising cuts moved QWK by "
              f"{res['qwk_optimal_qwk'] - res['default_qwk']:+.4f} and referable "
              f"sensitivity by "
              f"{res['qwk_optimal_referable']['sensitivity'] - res['default_referable']['sensitivity']:+.4f}")

    op = res["operating_point"]
    print(f"\noperating point — floor: sensitivity >= {op['min_sensitivity']:.2f} "
          f"AND specificity >= {op['min_specificity']:.2f}")
    print("  [PROVISIONAL - DECISION-032; the applicable standard is unconfirmed]")
    for key, label in (("meets_both", "meets BOTH floors"),
                       ("max_sens_at_spec", f"max sens at spec >= {op['min_specificity']:.2f}"),
                       ("max_spec_at_sens", f"max spec at sens >= {op['min_sensitivity']:.2f}")):
        v = op[key]
        if v is None:
            print(f"  {label:<34} UNREACHABLE")
        else:
            print(f"  {label:<34} sens {v['sensitivity']:.4f}  spec "
                  f"{v['specificity']:.4f}  (t = {v['threshold']:.4f})")

    if op["meets_both"] is None:
        print("\n  This model does not reach the screening standard at any operating")
        print("  point. That is a finding to report, not a number to soften.")

    out = args.out or (run_dir / "thresholds.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

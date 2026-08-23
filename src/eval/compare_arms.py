"""Compare arms under a MATCHED decision rule, and say what is separable from what.

WHY THIS EXISTS (DECISION-033). The Phase 4 stage 1 table compared arm E's QWK *after*
fitting its four cut points on validation against every other arm's QWK from a bare
`argmax`. Those are not the same quantity. E was being scored with four fitted parameters
and the softmax arms with none, and the ranking that came out of it was partly an artefact
of that asymmetry rather than of the imbalance mechanism being tested.

WHAT A MATCHED COMPARISON MEANS HERE. Every arm gets the same four degrees of freedom:

  * ordinal head — the cut points on its scalar output, as before
  * softmax head — the cut points on its EXPECTED GRADE, sum(p_i * i)

The expected grade is the natural continuous analogue of an ordinal output, and it is
what `argmax` throws away: a prediction spread over grades 2 and 3 and one confident at
grade 2 both become "2" under argmax, and they are not the same evidence.

FITTING ON VALIDATION IS ONLY HONEST IF THE OPTIMISM IS MEASURED. Four parameters on
5,268 images should barely overfit, but "should" is not a measurement. `--split-half`
fits the cuts on one half and scores the other, repeatedly, and reports the difference.
On this ablation the optimism is 0.001-0.012 QWK, so the fitted numbers are usable — but
that is a result, not an assumption, and it is re-measured whenever this is re-run.

    python -m src.eval.compare_arms --runs-root runs --pattern 'phase4_arm_*'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval.metrics import quadratic_weighted_kappa
from src.eval.thresholds import DEFAULT_CUTS, grades_from_scores, optimise_qwk_cuts

REPO = Path(__file__).resolve().parents[2]


def continuous_score(outputs: np.ndarray) -> np.ndarray:
    """One ordinal score per image, whatever the head.

    Softmax -> expected grade. Ordinal -> the scalar itself.
    """
    outputs = np.asarray(outputs, dtype=float)
    if outputs.ndim == 1 or outputs.shape[1] == 1:
        return outputs.reshape(-1)
    x = outputs - outputs.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return (p * np.arange(outputs.shape[1])).sum(axis=1)


def as_run_qwk(outputs: np.ndarray, y_true: np.ndarray) -> float:
    """The number the run itself reported: argmax for softmax, default cuts for ordinal."""
    outputs = np.asarray(outputs)
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        return quadratic_weighted_kappa(y_true, outputs.argmax(axis=1))
    return quadratic_weighted_kappa(
        y_true, grades_from_scores(outputs.reshape(-1), DEFAULT_CUTS))


def split_half_qwk(score: np.ndarray, y_true: np.ndarray, repeats: int = 40,
                   seed: int = 0, grid: int = 61, rounds: int = 6) -> dict:
    """Fit cuts on half, score the other half. Returns in-sample, held-out, optimism."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ins, held = [], []
    for _ in range(repeats):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        cuts, fit = optimise_qwk_cuts(score[a], y_true[a], grid=grid, rounds=rounds)
        ins.append(fit)
        held.append(quadratic_weighted_kappa(y_true[b], grades_from_scores(score[b], cuts)))
    return {"in_sample": float(np.mean(ins)), "held_out": float(np.mean(held)),
            "optimism": float(np.mean(ins) - np.mean(held))}


def paired_difference(score_a, score_b, y_true, repeats: int = 200, seed: int = 0,
                      ci: float = 0.95) -> dict:
    """Held-out QWK difference between two arms, with a paired bootstrap interval.

    PAIRED on the same images, because the arms are evaluated on identical validation
    data and the shared difficulty of those images is not what is being measured. An
    unpaired interval would be wider for a reason that has nothing to do with the arms.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(repeats):
        idx = rng.permutation(n)
        fit_on, score_on = idx[: n // 2], idx[n // 2:]
        ca, _ = optimise_qwk_cuts(score_a[fit_on], y_true[fit_on], grid=61, rounds=6)
        cb, _ = optimise_qwk_cuts(score_b[fit_on], y_true[fit_on], grid=61, rounds=6)
        j = score_on[rng.integers(0, len(score_on), len(score_on))]
        diffs.append(
            quadratic_weighted_kappa(y_true[j], grades_from_scores(score_a[j], ca))
            - quadratic_weighted_kappa(y_true[j], grades_from_scores(score_b[j], cb))
        )
    d = np.array(diffs)
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(d, alpha)), float(np.quantile(d, 1 - alpha))
    return {"mean": float(d.mean()), "lo": lo, "hi": hi,
            "separable": bool(lo > 0 or hi < 0),
            "a_better_fraction": float((d > 0).mean())}


def load_run(run_dir: Path) -> dict:
    d = np.load(Path(run_dir) / "val_outputs.npz")
    outputs, y = d["outputs"], d["y_true"]
    m = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    log_path = Path(run_dir) / "train_log.csv"
    fit_gap = None
    if log_path.exists():
        import pandas as pd

        log = pd.read_csv(log_path)
        row = log[log.epoch == m.get("best_epoch")]
        if len(row) and "train_qwk" in log.columns:
            fit_gap = float(row.train_qwk.iloc[0] - row.val_qwk.iloc[0])
    return {"name": Path(run_dir).name, "arm": m.get("arm"), "outputs": outputs,
            "y": y, "metrics": m, "generalisation_gap": fit_gap}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--pattern", default="phase4_arm_*")
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("ARM_A", "ARM_B"),
                    help="two arm letters for a paired bootstrap, e.g. --compare A E")
    ap.add_argument("--out", type=Path, help="write the comparison as JSON")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    dirs = sorted(d for d in Path(args.runs_root).glob(args.pattern)
                  if (d / "val_outputs.npz").exists())
    if not dirs:
        print(f"no runs with val_outputs.npz under {args.runs_root}/{args.pattern}",
              file=sys.stderr)
        return 1

    runs = {}
    for d in dirs:
        r = load_run(d)
        r["score"] = continuous_score(r["outputs"])
        r["as_run"] = as_run_qwk(r["outputs"], r["y"])
        r.update(split_half_qwk(r["score"], r["y"], repeats=args.repeats, seed=args.seed))
        runs[r["arm"] or r["name"]] = r

    print("MATCHED-DECISION-RULE COMPARISON (validation only, R3)")
    print("Every arm gets the same four degrees of freedom; softmax arms are scored on")
    print("their expected grade rather than argmax.\n")
    hdr = (f"{'arm':<5}{'as run':>9}{'fitted':>9}{'held-out':>10}{'optimism':>10}"
           f"{'gain':>8}{'gen gap':>9}")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(runs):
        r = runs[arm]
        gap = f"{r['generalisation_gap']:+.3f}" if r["generalisation_gap"] is not None else "-"
        print(f"{arm:<5}{r['as_run']:>9.4f}{r['in_sample']:>9.4f}{r['held_out']:>10.4f}"
              f"{r['optimism']:>10.4f}{r['held_out'] - r['as_run']:>+8.4f}{gap:>9}")

    order = sorted(runs, key=lambda a: -runs[a]["held_out"])
    print(f"\nranking on held-out fitted QWK: {' > '.join(order)}")
    as_run_order = sorted(runs, key=lambda a: -runs[a]["as_run"])
    if order != as_run_order:
        print(f"as-run ranking was          : {' > '.join(as_run_order)}")
        print("^ the decision rule, not the imbalance mechanism, moved these")

    result = {a: {k: v for k, v in r.items()
                  if k not in ("outputs", "y", "score", "metrics")}
              for a, r in runs.items()}

    if args.compare:
        a, b = (x.upper() for x in args.compare)
        if a in runs and b in runs:
            d = paired_difference(runs[a]["score"], runs[b]["score"], runs[a]["y"],
                                  seed=args.seed)
            print(f"\npaired: QWK({a}) - QWK({b}) = {d['mean']:+.4f} "
                  f"95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]")
            print(f"  {a} better in {100 * d['a_better_fraction']:.0f}% of resamples")
            print("  SEPARABLE" if d["separable"]
                  else "  NOT SEPARABLE - the interval spans zero, so this pair is a tie")
            result["_compare"] = {"a": a, "b": b, **d}

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

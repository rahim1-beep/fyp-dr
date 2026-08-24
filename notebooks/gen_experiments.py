"""Regenerate docs/EXPERIMENTS.md from runs/*/ — R4: numbers come from artefacts.

    python -m notebooks.gen_experiments

NOTHING HERE RUNS AT IMPORT TIME (DECISION-028). `src/data/notebook_check.py` imports
every module under `notebooks/` to collect the command lines its cells will run, and on
Kaggle the code bundle carries no `runs/` directory. An earlier version of this file read
`runs/phase3_baseline_resnet18/metrics.json` at module level, so importing it raised
FileNotFoundError and the whole pre-flight gate died before checking anything — on the
GPU session it was meant to protect.

The root is derived from `__file__`, not from the working directory, for the same reason:
a module that only works when invoked from the repo root is a module that fails somewhere
else eventually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import CLASS_NAMES, detect_collapse

DEFAULT_RUN = "phase3_baseline_resnet18"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run id to write up (the comparison table grows in Phase 4)")
    ap.add_argument("--runs-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "runs")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "docs/EXPERIMENTS.md")
    return ap


def render(run: str, runs_root: Path) -> tuple[str, str]:
    """Return (markdown, current-rule verdict) for one run's artefacts."""
    REPO = Path(__file__).resolve().parents[1]
    d = Path(runs_root) / run

    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    log = pd.read_csv(d / "train_log.csv")

    # Re-derive today's verdict from the committed confusion matrix. The artefact's own
    # `collapse` block predates DECISION-026 and is left untouched: it is the record.
    yt, yp = [], []
    for i, row in enumerate(m["confusion_matrix"]):
        for j, n in enumerate(row):
            yt += [i] * n
            yp += [j] * n
    verdict = detect_collapse(np.array(yt), np.array(yp), referable_threshold=2)

    ci = m["qwk_ci"]
    ref = m["referable"]
    r3, r4 = m["rare_class_ci"]["3"], m["rare_class_ci"]["4"]
    steady = log["train_seconds"][1:].mean()

    rows = []
    for g in range(5):
        rec = m["per_class_recall"][g]
        extra = ""
        if str(g) in m["rare_class_ci"]:
            c = m["rare_class_ci"][str(g)]
            extra = f" [{c['lo']:.3f}, {c['hi']:.3f}]"
        rows.append(f"| {g} | {CLASS_NAMES[g]} | {m['support'][g]:,} | "
                    f"{rec:.3f}{extra} | {m['predicted_counts'][g]:,} |")

    curve = " | ".join(f"{v:.3f}" for v in log["val_qwk"])
    head = " | ".join(str(e) for e in log["epoch"])
    lr = " | ".join(f"{v:.1e}" for v in log["lr"])

    text = f"""# EXPERIMENTS.md

    Every number here is read from `runs/<run_id>/metrics.json`, `config.yaml` and
    `train_log.csv` — the committed artefacts of a real execution (R4). Regenerate with
    `python -m notebooks.gen_experiments` rather than editing by hand.

    **The test set has not been touched** (R3). Every figure below is **validation**.

    ---

    ## PRE-REGISTERED - the stage 3 "capacity" hypothesis - **RESOLVED: HALF WRONG**

    > **Outcome (DECISION-036).** Arm B's gap held (0.290 -> 0.329). Arm A held (+0.015).
    > **Arm E broke it** (+0.053), and the umbrella claim was **wrong**: E's
    > sens@spec>=0.95 went 0.6589 -> 0.7464, closing 62% of the distance to the floor.
    > **And the test manipulated the wrong variable** - B0 is 4.01M parameters and
    > 0.385 GMAC against ResNet18's 11.18M and 1.814, i.e. smaller on both. Capacity was
    > never tested and no claim about it appears in this document. What was tested is
    > backbone quality at matched compute. The prediction is left below exactly as
    > written; it is the record.


    **Written 2026-08-22, before the stage 3 runs.** Recorded here so the prediction is on
    record whichever way it goes.

    ### What stage 1 showed

    The `train - val` QWK gap at each arm's best epoch, from the committed `train_log.csv`
    files:

    | arm | train QWK | val QWK | gap |
    |---|---:|---:|---:|
    | A | 0.7748 | 0.6831 | +0.092 |
    | E | 0.7808 | 0.7081 | +0.073 |
    | B | 0.8744 | 0.5846 | **+0.290** |
    | D | 0.8547 | 0.5550 | **+0.300** |
    | F | 0.9198 | 0.6190 | **+0.301** |
    | C | 0.2604 | 0.4049 | -0.145 |

    The three arms with the weighted sampler reach train QWK 0.87-0.92 and lose about 0.30
    on validation. That is memorisation, not underfitting. Every arm early-stopped, so the
    30-epoch budget was not binding either.

    ### The prediction

    **Capacity is not the binding constraint. EfficientNet-B0 will not close the 0.14
    referable-sensitivity gap.**

    1. **Arm B's train-val gap will stay at or above 0.25** on B0. A higher-capacity
       backbone fits the duplicated rare images at least as well, so the gap does not close.
    2. **Arms A and E will move by less than 0.03 QWK** in either direction.

    ### What each outcome means

    - **Prediction holds** - capacity is ruled out by evidence rather than argument, and
      the route to the sensitivity gap is regularisation and input resolution.
    - **Prediction fails**, B0 closes the gap - capacity mattered, the reasoning above is
      wrong, and the resolution plan is re-costed before anything is spent on it.

    Either way it is a result. `notebooks/phase4_stage3.py` cell 4 evaluates it directly
    and prints AS PREDICTED or PREDICTION WRONG.

    ---

    ## PRE-REGISTERED - stage 3.5, one backbone step - **RESOLVED: AS PREDICTED**

    > **Outcome (DECISION-039).** `QWK(E/b2) - QWK(E/b0) = +0.0052 [-0.0195, +0.0285]`,
    > **not separable**, |dQWK| < 0.02 as predicted. Stopping rule row 1: **keep B0**.
    > B2 is also behind on the metric that gates deployment - sens@spec>=0.95 of 0.7211
    > against B0's 0.7464 - with nearly double the generalisation gap. **The backbone
    > question is closed.** The off-native confound below still stands: this does not
    > show backbone scaling is exhausted in general, only that this step at 224 did not
    > pay.


    **Written 2026-08-24, before the run.** Arm E on EfficientNet-B2 @224, one arm.

    ### The manipulated variable, in parameters and FLOPs

    Measured locally with `timm` and `torch.utils.flop_counter` - not a loose word like
    "capacity", which is what went wrong last time.

    | backbone | params | GMAC @224 | native input |
    |---|---:|---:|---:|
    | resnet18 | 11.18M | 1.814 | 224 |
    | efficientnet_b0 | 4.01M | 0.385 | 224 |
    | **efficientnet_b2** | **7.70M** | **0.658** | **256** |

    The step is **+92% parameters and +71% GMAC** over B0. **The whole ladder sits below
    ResNet18 on both**, so nothing measured on it can be attributed to capacity.

    ### The confound, stated in advance

    B2's native input is 256 and it is run at 224 to hold resolution fixed. B0's native
    input *is* 224. **A null result therefore does not show the backbone lever is
    exhausted** - only that this step, at this resolution, did not pay.

    ### The prediction

    **B2 will not beat B0 by a separable margin: |dQWK| < 0.02, paired interval spanning
    zero.** ResNet18 -> B0 crossed architecture families; B0 -> B2 is within-family
    scaling at a below-native input, the weakest form of the same lever.

    ### The stopping rule - every outcome stops

    | paired held-out dQWK (B2 - B0) | decision |
    |---|---|
    | not separable | keep **B0** - smaller, cheaper, native at 224 |
    | separable and positive | keep **B2**. Do not run B3. |
    | separable and negative | keep **B0**, noting the off-native confound |
    | any of the above, but B2 clears sens@spec>=0.95 of 0.80 | report as deployment-relevant, still do not run B3 |

    **This is the last backbone step.** Reopening the ladder needs a positive argument
    logged before the run, never a good result on its own.

    ---

    ## Selection optimism — the running total

    **Architecture choices made by looking at validation: 3 of a soft budget of 4.**

    | # | choice | decided on |
    |---|---|---|
    | 1 | ResNet18 as the Phase 3 baseline | proposal, not validation — **not counted** |
    | 2 | ResNet18 -> EfficientNet-B0 | validation QWK, stage 3 |
    | 3 | arm E (ordinal) over arm A (softmax) | validation, at a tie — decided on the
          operating point and the gap |
    | 4 | B0 -> B2, or keep B0 | validation, stage 3.5 — **pending** |

    **What it costs.** DECISION-035 measures the optimism from fitting four cut points on
    validation (0.006-0.010 QWK, re-measured every run). **Nothing measures the optimism
    from choosing an architecture on the same 5,268 images.** It is real, it is not
    separately estimable without a second held-out split we do not have, and it inflates
    every validation figure in this document by an unknown amount. The **test set is
    untouched** (R3), so the single number reported at the end of Phase 5 remains honest —
    that is exactly what the test set is being saved for.

    **The gate.** A fourth entry needs a **positive argument written down before the run**,
    logged as a decision. Momentum from a good result is not an argument (DECISION-036).
    `notebooks/phase4_stage35.py` carries a stopping rule with no escalating outcome.

    ---

    ## Comparison table

    | Run | Arm | Model | Epochs | val QWK [95% CI] | Acc | Bal acc | Ref. sens | Ref. spec |
    |---|---|---|---|---|---|---|---|---|
    | `{run}` | {m['arm']} | {cfg['model']['arch']} | {m['epochs_run']} | **{m['qwk']:.4f}** [{ci['lo']:.4f}, {ci['hi']:.4f}] | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {ref['sensitivity']:.4f} | {ref['specificity']:.4f} |

    Arms B–F: not yet run. **The comparison table is not yet a comparison** — see the
    schedule note below before adding rows to it.

    ---

    ## `{run}` — arm A baseline

    **Arm A**: natural class distribution, plain cross-entropy, label smoothing
    {cfg['train']['label_smoothing']}, **no sampler, no class weights**.
    {cfg['model']['arch']}, pretrained, fully fine-tuned · batch {cfg['train']['batch_size']} ·
    AdamW lr {cfg['train']['lr']}, wd {cfg['train']['weight_decay']} · cosine with
    {cfg['train']['warmup_epochs']} warmup epochs · seed {cfg['seed']} ·
    {cfg['n_train']:,} train / {cfg['n_val']:,} val.

    **Environment:** python {cfg['versions']['python']}, torch {cfg['versions']['torch']},
    timm {cfg['versions']['timm']}, cuda {cfg['versions']['cuda']}.
    **Wall clock {m['wall_clock_minutes']:.1f} min** for {m['epochs_run']} epochs
    ({steady:.0f} s/epoch steady state; epoch 0 cost {log['train_seconds'][0]:.0f} s).

    > **Provenance gap.** `config.yaml` records `git: {cfg['git']}` — on Kaggle the code
    > arrives as an unzipped Dataset rather than a git checkout, so `git rev-parse` failed.
    > Fixed for Phase 4: `_git_sha()` now falls back to the SHA that `make_bundle` writes
    > into `BUNDLE.txt`. This run cannot name its own commit, and that is a real if minor
    > R6 weakness in the baseline.

    ### Headline

    | Metric | Value |
    |---|---|
    | **val QWK** | **{m['qwk']:.4f}** [{ci['lo']:.4f}, {ci['hi']:.4f}] |
    | Accuracy | {m['accuracy']:.4f} |
    | Balanced accuracy | {m['balanced_accuracy']:.4f} |
    | Majority-class rate | 0.7369 |
    | Referable sensitivity | **{ref['sensitivity']:.4f}** ({int(ref['tp'])} of {int(ref['tp'] + ref['fn'])}) |
    | Referable specificity | {ref['specificity']:.4f} |
    | Referable PPV / NPV | {ref['ppv']:.4f} / {ref['npv']:.4f} |
    | Best epoch | {m['best_epoch']} of {m['epochs_run']} |
    | Early stopped | {m['stopped_early']} |

    ### Per-class

    | Grade | | Support | Recall | Predicted |
    |---|---|---:|---:|---:|
    {chr(10).join(rows)}

    ### Confusion matrix — rows truth, columns predicted

    | | 0 | 1 | 2 | 3 | 4 |
    |---|---:|---:|---:|---:|---:|
    {chr(10).join('| **' + str(i) + '** | ' + ' | '.join(f'{v:,}' for v in row) + ' |' for i, row in enumerate(m['confusion_matrix']))}

    ### Learning curve

    | epoch | {head} |
    |---|{'---|' * len(log)}
    | **val QWK** | {curve} |
    | train QWK | {' | '.join(f"{v:.3f}" for v in log['train_qwk'])} |
    | lr | {lr} |

    ### Reading

    **Phase 3 acceptance: MET.** QWK interval far from zero, no fatal collapse.

    > The artefact's own `collapse` block says `collapsed: true` — it was written before
    > DECISION-026 made the rule severity-aware. Re-deriving the verdict from the committed
    > confusion matrix under the **current** rule gives **{verdict.level}**, not collapsed.
    > The artefact is left exactly as the run produced it; it is the record.

    **What is right.** Monotone learning, {log['val_qwk'][0]:.3f} → {m['qwk']:.4f}. Accuracy
    {100 * (m['accuracy'] - 0.7369):.1f} pp above the majority rate — the degenerate
    always-grade-0 predictor scores 0.7369 accuracy and **0.0 QWK**, so this model has
    genuinely learned the ordering. Specificity {ref['specificity']:.3f}: very few false
    referrals.

    **What is weak, and is the point of the ablation.**

    - **Referable sensitivity {ref['sensitivity']:.4f}** — {int(ref['fn'])} of
      {int(ref['tp'] + ref['fn'])} referable cases missed. This is the clinically meaningful
      number and the one arms B–F must move.
    - **Grade 1 is never predicted**, absorbed into grade 0 (342 of its 357 val images).
      A warning, not a collapse: grade 1 is below the referable threshold, so no referral
      changes, and QWK weights a 1-called-0 error at 1/16 of a 4-called-0 one.
    - **Balanced accuracy {m['balanced_accuracy']:.4f} against accuracy {m['accuracy']:.4f}**
      is that imbalance in one line.
    - Grade 3 recall {r3['point']:.3f} [{r3['lo']:.3f}, {r3['hi']:.3f}] and grade 4
      {r4['point']:.3f} [{r4['lo']:.3f}, {r4['hi']:.3f}] — wide intervals on thin support,
      as DECISION-006 anticipated. **Do not rank arms on these point estimates.**

    ### The schedule was the binding constraint, not convergence

    This is the most important thing the artefact says, and it was not visible in the
    summary numbers.

    The learning rate ran **{log['lr'][1]:.1e} at epoch 1 down to {log['lr'][7]:.1e} at
    epoch 7** — the cosine had annealed essentially to zero. Validation QWK flattening over
    the last two epochs ({log['val_qwk'][6]:.4f}, {log['val_qwk'][7]:.4f}) is therefore the
    **schedule finishing**, not the model saturating. Train QWK was still climbing
    ({log['train_qwk'][6]:.3f} → {log['train_qwk'][7]:.3f}) while val did not follow, which
    is the beginning of overfitting under a nearly-zero LR rather than evidence of a ceiling.

    `early_stopping.patience` is 7, so with 8 epochs it could never have fired — this run was
    stopped by its epoch count and nothing else.

    **Consequence for Phase 4:** a longer schedule is not "more of the same", it stretches
    the cosine and produces a different trajectory. Arms B–F must therefore share one epoch
    budget with each other **and with a re-run of arm A**; the 8-epoch baseline above is the
    Phase 3 record and is *not* a valid comparator for a 30-epoch arm.

    ### Comparison discipline for arms B–F

    - Same splits, same cache, same seed, **same epoch budget**. An arm that trains longer is
      not a better arm.
    - Selection on **validation QWK** only (R3). The test set is opened once, at the end.
    - Arm F additionally reports metrics by source dataset and is selected on its
      **EyePACS-only** validation QWK, because its pooled val is a different population
      (DECISION-007).
    - Report grades 3 and 4 with intervals, never bare point estimates (DECISION-006).
    - Differences smaller than the QWK confidence interval are not results. This run's
      interval is ±{(ci['hi'] - ci['lo']) / 2:.4f}, so an arm beating it by less than about
      0.03 QWK needs a seed repeat before anyone calls it a win.
    """
    # The template above is indented with this function; Markdown would read that as a
    # code block, so the emitted document is dedented. `textwrap.dedent` needs a common
    # prefix, and blank lines in the template have none - hence the explicit pass over
    # lines rather than a bare dedent().
    text = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in text.splitlines()) + "\n"
    return text, verdict.level


def main() -> int:
    args = build_parser().parse_args()

    d = Path(args.runs_root) / args.run
    if not (d / "metrics.json").exists():
        print(f"no artefacts at {d}. Fetch them first:\n"
              f"  python -m src.data.fetch_run --kernel rah098/<slug> "
              f"--run-id {args.run}", file=sys.stderr)
        return 1

    text, level = render(args.run, Path(args.runs_root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"current-rule collapse verdict: {level}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

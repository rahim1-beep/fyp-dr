# EXPERIMENTS.md

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

## Seed stability - stage 2 (DECISION-041, -042, -043)

Arms E and A on EfficientNet-B0, seeds 42/43/44. Everything else held.

| | E | mean | sd | A | mean | sd |
|---|---|---:|---:|---|---:|---:|
| held-out QWK | 0.7578 / 0.7548 / 0.7564 | **0.7563** | 0.0012 | 0.7346 / 0.7370 / 0.7426 | 0.7381 | 0.0034 |
| sens@spec>=0.95 | 0.7464 / 0.7211 / 0.7405 | **0.7360** | 0.0108 | 0.6822 / 0.7085 / 0.7172 | 0.7026 | 0.0149 |
| train-val gap | 0.034 / 0.098 / 0.097 | **0.0765** | 0.0298 | 0.157 / 0.137 / 0.096 | 0.1300 | 0.0254 |

### What this corrected

**Arm E's train-val gap is 0.077, not 0.034.** The 0.034 was seed 42 alone, the most
extreme of the three, and it had been used as an argument twice before stage 2
existed - once for the mechanism claim in the stage 3 analysis, once as a reason to
prefer B0 over B2. Both are annotated and the affected claims struck (DECISION-042).
**The seed range, 0.064, is larger than most of the between-arm gap differences this
project had been reasoning about.**

Every single-seed gap in this document is marked `[1 seed]`. Arms B, C, C2, D and F
have never been seed-resolved, so **no gap carries an argument unless it was measured
across seeds.**

### What it licenses

Arm E is ahead of arm A on **all three seeds on both metrics**, with roughly a third
of the seed variance on QWK, and the ranges do not overlap. That is a factual
description of six runs and a defensible **selection rationale**.

It is **not** a significance claim. All six runs share the same 5,268 validation
images, so the validation-sampling component of the uncertainty is identical across
them and does not average away; the paired bootstrap puts it at +/-0.027 against a
seed-mean difference of +0.018. Seed consistency addresses training stochasticity and
is silent on which patients are in the split. **The A-vs-E tie stands** - what changed
is the confidence of the selection, not the status of the comparison (DECISION-043).

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
| `phase3_baseline_resnet18` | A | resnet18 | 8 | **0.6138** [0.5859, 0.6421] | 0.7965 | 0.4163 | 0.5131 | 0.9804 |

Arms B–F: not yet run. **The comparison table is not yet a comparison** — see the
schedule note below before adding rows to it.

---

## `phase3_baseline_resnet18` — arm A baseline

**Arm A**: natural class distribution, plain cross-entropy, label smoothing
0.1, **no sampler, no class weights**.
resnet18, pretrained, fully fine-tuned · batch 64 ·
AdamW lr 0.0003, wd 0.0001 · cosine with
2 warmup epochs · seed 42 ·
24,586 train / 5,268 val.

**Environment:** python 3.12.13, torch 2.10.0+cu128,
timm 1.0.26, cuda 12.8.
**Wall clock 12.3 min** for 8 epochs
(71 s/epoch steady state; epoch 0 cost 137 s).

> **Provenance gap.** `config.yaml` records `git: unknown` — on Kaggle the code
> arrives as an unzipped Dataset rather than a git checkout, so `git rev-parse` failed.
> Fixed for Phase 4: `_git_sha()` now falls back to the SHA that `make_bundle` writes
> into `BUNDLE.txt`. This run cannot name its own commit, and that is a real if minor
> R6 weakness in the baseline.

### Headline

| Metric | Value |
|---|---|
| **val QWK** | **0.6138** [0.5859, 0.6421] |
| Accuracy | 0.7965 |
| Balanced accuracy | 0.4163 |
| Majority-class rate | 0.7369 |
| Referable sensitivity | **0.5131** (528 of 1029) |
| Referable specificity | 0.9804 |
| Referable PPV / NPV | 0.8642 / 0.8924 |
| Best epoch | 6 of 8 |
| Early stopped | False |

### Per-class

| Grade | | Support | Recall | Predicted |
|---|---|---:|---:|---:|
| 0 | No DR | 3,882 | 0.982 | 4,657 |
| 1 | Mild | 357 | 0.000 | 0 |
| 2 | Moderate | 788 | 0.376 | 478 |
| 3 | Severe | 133 | 0.316 [0.241, 0.396] | 68 |
| 4 | Proliferative | 108 | 0.407 [0.324, 0.504] | 65 |

### Confusion matrix — rows truth, columns predicted

| | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 3,814 | 0 | 65 | 0 | 3 |
| **1** | 342 | 0 | 15 | 0 | 0 |
| **2** | 463 | 0 | 296 | 16 | 13 |
| **3** | 18 | 0 | 68 | 42 | 5 |
| **4** | 20 | 0 | 34 | 10 | 44 |

### Learning curve

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **val QWK** | 0.025 | 0.511 | 0.549 | 0.546 | 0.597 | 0.595 | 0.614 | 0.613 |
| train QWK | -0.002 | 0.337 | 0.502 | 0.550 | 0.588 | 0.619 | 0.632 | 0.646 |
| lr | 1.5e-04 | 3.0e-04 | 2.8e-04 | 2.3e-04 | 1.5e-04 | 7.7e-05 | 2.3e-05 | 3.0e-06 |

### Reading

**Phase 3 acceptance: MET.** QWK interval far from zero, no fatal collapse.

> The artefact's own `collapse` block says `collapsed: true` — it was written before
> DECISION-026 made the rule severity-aware. Re-deriving the verdict from the committed
> confusion matrix under the **current** rule gives **warning**, not collapsed.
> The artefact is left exactly as the run produced it; it is the record.

**What is right.** Monotone learning, 0.025 → 0.6138. Accuracy
6.0 pp above the majority rate — the degenerate
always-grade-0 predictor scores 0.7369 accuracy and **0.0 QWK**, so this model has
genuinely learned the ordering. Specificity 0.980: very few false
referrals.

**What is weak, and is the point of the ablation.**

- **Referable sensitivity 0.5131** — 501 of
  1029 referable cases missed. This is the clinically meaningful
  number and the one arms B–F must move.
- **Grade 1 is never predicted**, absorbed into grade 0 (342 of its 357 val images).
  A warning, not a collapse: grade 1 is below the referable threshold, so no referral
  changes, and QWK weights a 1-called-0 error at 1/16 of a 4-called-0 one.
- **Balanced accuracy 0.4163 against accuracy 0.7965**
  is that imbalance in one line.
- Grade 3 recall 0.316 [0.241, 0.396] and grade 4
  0.407 [0.324, 0.504] — wide intervals on thin support,
  as DECISION-006 anticipated. **Do not rank arms on these point estimates.**

### The schedule was the binding constraint, not convergence

This is the most important thing the artefact says, and it was not visible in the
summary numbers.

The learning rate ran **3.0e-04 at epoch 1 down to 3.0e-06 at
epoch 7** — the cosine had annealed essentially to zero. Validation QWK flattening over
the last two epochs (0.6138, 0.6135) is therefore the
**schedule finishing**, not the model saturating. Train QWK was still climbing
(0.632 → 0.646) while val did not follow, which
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
  interval is ±0.0281, so an arm beating it by less than about
  0.03 QWK needs a seed repeat before anyone calls it a win.


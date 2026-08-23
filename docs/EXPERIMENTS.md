# EXPERIMENTS.md

Every number here is read from `runs/<run_id>/metrics.json`, `config.yaml` and
`train_log.csv` — the committed artefacts of a real execution (R4). Regenerate with
`python -m notebooks.gen_experiments` rather than editing by hand.

**The test set has not been touched** (R3). Every figure below is **validation**.

---

## PRE-REGISTERED - the stage 3 capacity hypothesis

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


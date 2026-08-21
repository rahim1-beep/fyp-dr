# EXPERIMENTS.md

Every number here traces to `runs/<run_id>/metrics.json` from a real execution (R4).
Nothing is estimated, rounded from memory, or carried over from a previous run.

**The test set has not been touched** (R3). Every figure below is **validation**.

---

## Comparison table

| Run | Arm | Model | Epochs | val QWK [95% CI] | Acc | Bal acc | Ref. sens | Ref. spec | Status |
|---|---|---|---|---|---|---|---|---|---|
| `phase3_baseline_resnet18` | A | ResNet18 | 8 | **0.6138** [0.5859, 0.6421] | 0.7965 | 0.4163 | 0.5131 | 0.9804 | baseline |

Arms B–F: not yet run.

---

## `phase3_baseline_resnet18` — arm A baseline

**Date:** 2026-08-22 · **Arm A**: natural class distribution, plain cross-entropy with
label smoothing 0.1, no sampler, no class weights · ResNet18, ImageNet-pretrained, fully
fine-tuned · 224×224 · 8 epochs, best at **epoch 6** · Kaggle GPU T4.

> **Provenance note.** These figures were reported from the Kaggle session's cell-4
> output. `runs/phase3_baseline_resnet18/{config.yaml,metrics.json,train_log.csv}` still
> needs to be downloaded from the notebook output and committed, at which point this
> section should be regenerated from the file rather than transcribed. Until then, treat
> this entry as **pending artefact commit**.

### Headline

| Metric | Value |
|---|---|
| **val QWK** | **0.6138** [0.5859, 0.6421] |
| Accuracy | 0.7965 |
| Balanced accuracy | 0.4163 |
| Majority-class rate | 0.7369 |
| Referable sensitivity | **0.5131** |
| Referable specificity | 0.9804 |

### Per-class

| Grade | | Support | Recall | Predicted |
|---|---|---:|---:|---:|
| 0 | No DR | 3,882 | 0.982 | 4,657 |
| 1 | Mild | 357 | **0.000** | **0** |
| 2 | Moderate | 788 | 0.376 | 478 |
| 3 | Severe | 133 | 0.316 [0.241, 0.396] | 68 |
| 4 | Proliferative | 108 | 0.407 [0.324, 0.504] | 65 |

### Learning curve — val QWK by epoch

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 0.025 | 0.511 | 0.549 | 0.546 | 0.597 | 0.595 | **0.614** | 0.614 |

### Reading

**Phase 3 acceptance: MET.** The QWK interval is far from zero and there is no fatal
collapse (DECISION-026). The pipeline is demonstrably correct end to end: preprocessing,
patient-level splits, cache, loaders, model, loss, schedule, metrics, artefacts.

**What is right.** Monotone learning from 0.025 to 0.614. Accuracy 6.0 pp above the
majority rate — the degenerate always-grade-0 predictor scores 0.7369 accuracy and
**0.0 QWK**, so this model has genuinely learned the ordering. Specificity 0.98 means very
few false referrals.

**What is weak, and is the point of the ablation.**

- **Referable sensitivity 0.5131.** Roughly half of referable cases are missed. This is
  the clinically meaningful number and the one arms B–F must move.
- **Grade 1 is never predicted**, absorbed into grade 0. A warning, not a collapse: grade
  1 is below the referable threshold, so no referral decision changes, and QWK weights a
  1-called-0 error at 1/16 of a 4-called-0 one. It is the characteristic failure of an
  unbalanced baseline and precisely what arms B–E exist to fix.
- **Balanced accuracy 0.4163 against accuracy 0.7965** is that imbalance in one line.
- Grade 3 and 4 recalls carry wide intervals on thin support (133 and 108 val images),
  as DECISION-006 anticipated. Do not compare arms on their point estimates alone.

**Not converged.** QWK was still at its maximum on the last two epochs and the schedule
was only 8 epochs. Phase 4 should not treat 0.6138 as ResNet18's ceiling; a longer
schedule is a confound to control before attributing any arm's gain to its mechanism.

### Comparison discipline for arms B–F

- Same splits, same cache, same seed, same epoch budget. An arm that trains longer is not
  a better arm.
- Selection on **validation QWK** only (R3).
- Arm F additionally reports metrics by source dataset, and is selected on its
  **EyePACS-only** validation QWK, because its pooled val is a different population
  (DECISION-007).
- Report grade 3 and 4 with intervals, never bare point estimates (DECISION-006).

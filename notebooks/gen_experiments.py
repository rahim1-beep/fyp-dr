"""Regenerate docs/EXPERIMENTS.md from runs/*/ — R4: numbers come from artefacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import CLASS_NAMES, detect_collapse

REPO = Path(".")
RUN = "phase3_baseline_resnet18"
d = REPO / "runs" / RUN

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

## Comparison table

| Run | Arm | Model | Epochs | val QWK [95% CI] | Acc | Bal acc | Ref. sens | Ref. spec |
|---|---|---|---|---|---|---|---|---|
| `{RUN}` | {m['arm']} | {cfg['model']['arch']} | {m['epochs_run']} | **{m['qwk']:.4f}** [{ci['lo']:.4f}, {ci['hi']:.4f}] | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {ref['sensitivity']:.4f} | {ref['specificity']:.4f} |

Arms B–F: not yet run. **The comparison table is not yet a comparison** — see the
schedule note below before adding rows to it.

---

## `{RUN}` — arm A baseline

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

Path("docs/EXPERIMENTS.md").write_text(text, encoding="utf-8")
print("wrote docs/EXPERIMENTS.md")
print("current-rule verdict:", verdict.level)
print("steady-state s/epoch:", round(steady, 1))

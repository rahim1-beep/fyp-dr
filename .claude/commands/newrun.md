---
description: Scaffold a run directory and config from the arm template
argument-hint: <arm A-E> [model] [--img 224|384]
---

Scaffold a new run for arm **$1**.

1. Validate the arm is one of `A B C D E`. If not, stop and list the valid arms.
2. Generate `run_id` as `{YYYYMMDD}_{model}_{arm}_{img}` — e.g. `20260820_effb0_B_224`.
   If that directory already exists, append `_2`, `_3`, … Never overwrite an existing run.
3. Create `runs/<run_id>/` and write `config.yaml` as the merge of, in order:
   `configs/base.yaml` → `configs/arm_$1.yaml` → the environment overlay
   (`configs/kaggle.yaml`) → any CLI overrides passed here.
4. The written config must record: the resolved seed, the git commit SHA, the image size,
   the split CSV paths, and the exact `timm` model name. This is what makes the run
   reproducible (R6).
5. **Verify `tests/test_no_leakage.py` passes** before declaring the run ready. If it
   fails, stop — do not scaffold a run against a broken split.
6. Print the exact command the user should run on Kaggle. Do not start training here; the
   local machine has no CUDA device.

Do not write `metrics.json` — that is `eval-analyst`'s output, and only from a real
execution.

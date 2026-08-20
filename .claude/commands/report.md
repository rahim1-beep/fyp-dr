---
description: Regenerate the experiment comparison table from runs/
---

Delegate to `eval-analyst`. Regenerate `docs/EXPERIMENTS.md` from the filesystem.

1. Glob `runs/*/metrics.json`. **These files are the only source of numbers** (R4).
2. Build the comparison table with columns: `run_id`, `model`, `arm`, `img_size`,
   `epochs`, `val_qwk`, `test_qwk`, `balanced_acc`, `referable_sensitivity`,
   `checkpoint_path`, `notes`.
3. Sort by `val_qwk` descending. Model selection is on **validation** QWK (R3) — mark the
   selected model on that basis, never on test.
4. For any run missing a field, write `—`. **Never interpolate, estimate, or carry a value
   over from a similar run.**
5. Flag in the notes column any run whose confusion matrix has an all-zero column, or
   whose accuracy is within 2 points of the 73.48% majority-class rate. These are
   suspected collapses, not results.
6. If `runs/` is empty, write a table with a header and a single row saying no runs have
   been executed. Do not fabricate an example row.

Print a summary of what changed since the last regeneration.

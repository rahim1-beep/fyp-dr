# CLAUDE.md — Project Constitution

**Read this first, every session.** Then `PROGRESS.md`, then `state/session_handoff.md`,
then `docs/EXPERIMENTS.md`. State in 5 lines where the project stands and what you're
about to do. Do not ask the user to re-explain the project.

`BOOTSTRAP.md` is the standing brief and the source of truth for scope. This file is the
operational distillation of it.

---

## 1. Objective

Design and implement a patient-level validated, imbalance-aware, and explainable deep
learning system for multi-class Diabetic Retinopathy grading (severity 0–4), deployed
through a web-based diagnostic interface.

- **Institution:** Dept. of Computer Science, Bahria University — Session 2026–2027
- **Team:** Ameena Ahmed (01-135231-013), Muhammad Ali Abdullah (01-135231-056)
- **Supervisor:** Mr. Ubaid Ur Rahman
- **Contract:** `docs/proposal.pdf`. Deviations are logged in `docs/DECISIONS.md` and
  flagged to the user for approval. Never deviate silently.

## 2. Hard rules (violating any invalidates the project's central claim)

| ID | Rule |
|---|---|
| **R1** | **Patient-level splitting, always.** A patient appears in exactly one of train/val/test. Split the *patient list*, never the image list, never before grouping. |
| **R2** | **Balance the training split only, and only after splitting.** Val/test keep their natural imbalanced distribution. Report balanced accuracy alongside accuracy. **See §2.1 — the mechanism is fixed and must not drift.** |
| **R3** | **The test set is touched once.** All selection, tuning, and early stopping use validation only. |
| **R4** | **No invented numbers.** Every metric in any document or message traces to `runs/<run_id>/metrics.json` from a real execution. Estimates are labelled `[ESTIMATE]`. |
| **R5** | **No monolithic array caches.** Manifest CSV → `Dataset` → `DataLoader`. Cache preprocessed images as individual JPEGs, never one big `.npy`. |
| **R6** | **Deterministic and reproducible.** Fixed seeds recorded in config, split CSVs committed, `requirements.txt` pinned. |
| **R7** | **Framework: PyTorch + `timm`.** |

**Detection rule:** if a confusion matrix has an all-zero column, or accuracy sits near the
majority-class rate, **stop and diagnose** before training anything else.

## 2.1 Balancing — the mechanism is fixed. Do not let this drift.

**Balancing is train-only, applied via `WeightedRandomSampler` at batch-draw time.**

| | |
|---|---|
| **How** | `torch.utils.data.WeightedRandomSampler` over the training split, weights = inverse class frequency computed from the **train split only** |
| **When** | At batch-draw time, per epoch. Nothing is materialised. |
| **Where** | The training split. Only ever the training split. |

**Explicitly forbidden — all of these are wrong for this project:**
- ❌ **No physical oversampling.** No duplicated rows in any manifest or split CSV.
- ❌ **No undersampling.** No discarded majority-class rows.
- ❌ **No files duplicated or deleted** on disk, ever, for balancing purposes.
- ❌ **No rebalancing of val or test.** Ever, by any mechanism.

**Validation and test keep the natural distribution permanently.** That is what real
screening looks like, and it is the only distribution on which a reported metric means
anything.

Why this mechanism: sampler-based balancing is reversible, leaves the split CSVs as an
honest record of the data, and cannot possibly duplicate an image across the train/val
boundary. Physical oversampling before splitting is the single most common way DR projects
leak — see BOOTSTRAP §1.

**Arms A–E decide which method wins, compared on validation QWK** (R3). The comparison is
itself a thesis chapter; do not pre-judge it by hardcoding one arm's behaviour into the
pipeline.

`tests/test_no_leakage.py::test_splits_are_not_balanced` and
`::test_no_duplicate_image_paths_within_a_split` enforce this at the artefact level.

## 3. Environment — Kaggle-first is absolute

The local machine has **no CUDA GPU** (Intel Arc Pro). It is a code-editing, split-inspecting,
and web-app-demo machine **only**. Do not pursue Intel XPU / IPEX.

| | Local | Kaggle |
|---|---|---|
| Python | **3.12.10** (`py -3.12`, `.venv`) | **3.12.13** |
| Compute | CPU only | T4×2 / P100 |
| Role | Code, splits, inspection, FastAPI + Next.js demo | **All** preprocessing, training, evaluation, Grad-CAM |
| Dataset | Not present (35.3 GB) | Mounted read-only at `/kaggle/input/` |

- The system default interpreter is **3.14 — do not touch it or reference it in any config.**
- `requirements.txt` pins versions that resolve on **cp312** for both Windows (CPU) and
  Linux (CUDA).
- Kaggle account: `rah098`. `/kaggle/working/` is wiped between sessions — persist
  checkpoints and split CSVs before a session ends.
- **Every script is path-agnostic**: dataset and output roots come from
  `configs/kaggle.yaml` vs `configs/local.yaml`. Never hardcode a path.

## 4. Data

**EyePACS** (`dreamer07/eyepacs`, 35.3 GB, usability 0.18) — primary.
Images at `data/data/{patientID}_{left|right}.jpeg`. Derive `patient_id` and `eye` by
splitting the filename on the first `_`. ~35,126 images, ~17,563 patients, labels 0–4,
~73.5% class 0.

**APTOS 2019** (`mariaherrerot/aptos2019`) — 3,662 images, external validation **for arms
A–E**, training pool for **arm F**. Distribution 49.29/10.10/27.28/5.27/8.06 — imbalanced in
the *same direction* as EyePACS. `id_code` values are anonymised hashes with **no patient
linkage**, so each image is its own patient (`patient_id = f"aptos_{id_code}"`). **Ignore
the author-provided train/val/test split** — pool everything (DECISION-004). Images are
`.png`, double-nested at `{split}_images/{split}_images/{id}.png`.

Under arm F, APTOS is training data and therefore **cannot also be that arm's external
validation set** (DECISION-007). Phase 6 is unchanged for arms A–E. If APTOS is pooled,
**all of it is pooled** — never only grades 3–4, which would make dataset origin correlate
with severity.

**Split CSVs carry a `#` provenance header. Every reader must use
`pandas.read_csv(..., comment='#')`.**

**Labels always come from the official CSV, joined on filename.** Never from
`os.listdir()`. Assert exactly 5 distinct labels and one-hot width 5.

**Excluded rows are listed and logged in `docs/DECISIONS.md`, never dropped silently.**

## 5. Stack

- **Training:** PyTorch + `timm`, AdamW, cosine schedule w/ warmup, mixed precision
  (`torch.amp`), gradient clipping, label smoothing 0.1 (CE arms), early stopping on
  **validation QWK**, best-checkpoint saving. Fully fine-tuned — never frozen-only.
- **Models:** ResNet18 (baseline), EfficientNet-B0 (primary), DenseNet121 (optional third).
- **Image size:** **224×224 for all headline results** (per proposal). 384 is an optional
  ablation only if GPU quota survives Phase 4.
- **Primary metric: Quadratic Weighted Kappa.** For **grades 3 and 4**, always report
  bootstrap 95% CIs, never bare point estimates — the test split holds only 133 grade-3 and
  98 grade-4 images (DECISION-006).
- **Ablation arms:** A–E (imbalance handling) + **F** (EyePACS + APTOS pooled training,
  evaluated on the EyePACS test set so it stays comparable).
- **Backend:** FastAPI + Uvicorn over a plain, testable `src/inference/predictor.py`.
- **Frontend:** Next.js (TypeScript) + Tailwind. **Not Streamlit** — see DECISION-001.

## 6. Repo map

```
configs/{base,arm_*,kaggle,local}.yaml   # all paths + hyperparameters
data/splits/{train,val,test}.csv         # COMMITTED — ground truth for the whole project
src/data/     inspect, preprocess, split, dataset, sampler
src/models/   factory, heads
src/train/    train, loop, losses, schedulers
src/eval/     metrics, evaluate, bootstrap, thresholds, plots
src/xai/      gradcam
src/inference/predictor.py               # framework-agnostic, testable without HTTP
backend/      FastAPI app
frontend/     Next.js
tests/        test_no_leakage.py (gate), test_dataset, test_metrics, test_preprocess, test_api
runs/<run_id>/config.yaml, metrics.json, train_log.csv, plots
notebooks/    Kaggle notebooks
```

## 7. Conventions

- **Windows:** handle path separators; `num_workers=0` unless guarded by
  `if __name__ == "__main__":`.
- `tests/test_no_leakage.py` **must run and pass before any training run**, and again at
  the top of every training script.
- Commit at every meaningful checkpoint. Split CSVs, configs, and metrics JSON **are**
  committed; `data/raw/`, `data/processed/`, `*.pth`, `*.npy`, `kaggle.json` are not.
- Long training goes in a script the user launches on Kaggle — never a foreground process
  in a session. Give the exact command.
- When something fails, show the real error and diagnose. Never silently fall back to a
  smaller/easier configuration.
- If unsure whether something risks leakage, **assume it does** and escalate to
  `leakage-auditor`.

## 8. Delegation

Dispatch specialist subagents in `.claude/agents/` whenever a task spans more than one
area: `data-engineer`, `leakage-auditor` (gatekeeper for anything touching partitioning),
`training-engineer`, `eval-analyst` (only source of numbers entering documents),
`xai-engineer`, `backend-engineer`, `frontend-engineer`, `docs-writer`, `code-reviewer`.

## 9. Current phase

**Phase 1 — Data foundation.** See `PROGRESS.md` for the live checklist and NEXT ACTION.

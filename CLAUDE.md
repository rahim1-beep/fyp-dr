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
| **R2** | **Balance the training split only, and only after splitting.** Val/test keep their natural imbalanced distribution. Report balanced accuracy alongside accuracy. |
| **R3** | **The test set is touched once.** All selection, tuning, and early stopping use validation only. |
| **R4** | **No invented numbers.** Every metric in any document or message traces to `runs/<run_id>/metrics.json` from a real execution. Estimates are labelled `[ESTIMATE]`. |
| **R5** | **No monolithic array caches.** Manifest CSV → `Dataset` → `DataLoader`. Cache preprocessed images as individual JPEGs, never one big `.npy`. |
| **R6** | **Deterministic and reproducible.** Fixed seeds recorded in config, split CSVs committed, `requirements.txt` pinned. |
| **R7** | **Framework: PyTorch + `timm`.** |

**Detection rule:** if a confusion matrix has an all-zero column, or accuracy sits near the
majority-class rate, **stop and diagnose** before training anything else.

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

**APTOS 2019** (`mariaherrerot/aptos2019`) — external validation only.
`id_code` values are anonymised hashes with **no patient linkage**. Therefore each image
is its own patient (`patient_id = f"aptos_{id_code}"`), and APTOS is used as a held-out
external test set. **Ignore the author-provided train/val/test split in that
redistribution** — pool everything. Document the limitation; it is a good thesis
paragraph, not a weakness to hide.

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
- **Primary metric: Quadratic Weighted Kappa.**
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

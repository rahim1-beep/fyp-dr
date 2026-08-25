# BOOTSTRAP.md — AI-Powered Diabetic Retinopathy Detection & Grading System

You are the lead engineer on a university Final Year Project. This file is your standing brief. Read it fully before acting, and re-read it at the start of every new session.

## 0. Project identity

- **Title:** AI-Powered Diabetic Retinopathy Detection and Grading System
- **Institution:** Department of Computer Science, Bahria University — Session 2026–2027
- **Team:** Ameena Ahmed (01-135231-013), Muhammad Ali Abdullah (01-135231-056)
- **Supervisor:** Mr. Ubaid Ur Rahman
- **Status:** Clean-slate build. Earlier exploratory work exists but is being discarded entirely — do not look for it, import from it, or reference it. Everything below is built from scratch against the raw datasets.
- **Objective (verbatim from the approved proposal):** design and implement a patient-level validated, imbalance-aware, and explainable deep learning system for multi-class Diabetic Retinopathy grading (severity 0–4), deployed through a web-based diagnostic interface.

The proposal is `docs/proposal.pdf`. Read it. The proposal is a contract with the supervisor — do not silently deviate from it. If you believe a deviation is technically necessary, log it in `docs/DECISIONS.md` and flag it to the user for approval.

## 1. Known failure modes in DR pipelines — engineer against these from day one

These are the specific ways automated DR grading projects end up at 20–45% accuracy on a 5-class problem (chance = 20%). Every one of them is silent: the code runs, loss decreases, and the result is garbage. Build defences into the pipeline rather than discovering them later.

1. **Labels built from directory listings.** Deriving labels from `os.listdir()` on class folders picks up stray directories, caches, and unsorted orderings, producing a label vector of the wrong width or the wrong order. Always build labels from the official CSV and join on filename. Assert the number of distinct labels is exactly 5 and that the one-hot width is 5 — fail loudly otherwise.
2. **Frozen backbones.** Fundus images look nothing like ImageNet. Training only a classification head on a frozen ImageNet trunk cannot separate DR grades, and the model typically collapses to predicting one class for every image. Fully fine-tune.
3. **Preprocessing that doesn't match the architecture.** Each pretrained model expects specific input normalisation. Pull mean/std from the model's own config rather than hardcoding 0–1 scaling.
4. **No fundus-specific preprocessing.** A raw resize to 224 destroys microaneurysms and small haemorrhages, which carry essentially all the signal separating grades 1 and 2.
5. **Monolithic array caches.** A single `X.npy` of all images cannot scale, cannot be augmented per-epoch, and cannot be split by patient.
6. **Random image-level splits**, which put both eyes of a patient across the train/test boundary.
7. **Accuracy as the headline metric**, which on a 73% class-0 dataset rewards a model that never predicts anything else.
8. **Under-training** — too few epochs, no LR schedule, early stopping on the wrong metric.

Detection rule: if a confusion matrix has an all-zero column, or accuracy sits near the majority-class rate, stop and diagnose before training anything else.

## 2. Non-negotiable rules

These are hard constraints. Violating any of them invalidates the project's central claim.

**R1 — Patient-level splitting, always.** EyePACS filenames are `{patientID}_{left|right}` (e.g. `10_left.jpeg`, `10_right.jpeg`). Both eyes of one patient carry correlated pathology. A patient appears in **exactly one** of train / val / test. Never split on images. Never split before grouping. This is the project's headline contribution.

**R2 — Balance the training set only, and only after splitting.** Rebalancing (over/under-sampling, weighted sampling) happens *inside* the training split, after the patient-level partition is frozen. If you oversample first and split second, duplicated images land on both sides and the leakage is total. Validation and test sets keep their natural, imbalanced distribution, because that is what real screening looks like. Report balanced accuracy alongside accuracy so imbalance can't hide.

**R3 — The test set is touched once.** Model selection, threshold tuning, early stopping, and every hyperparameter decision use validation only. **Test metrics are computed exactly once, on the single selected model, after every decision that could change that model is final** — amended 2026-08-25, DECISION-045. This clause previously read "computed at the end of each phase and recorded", which describes many touches and contradicted the rule its own heading states; `CLAUDE.md` always carried the once-only reading and the two now agree. Interim phases report **validation** numbers. If you find yourself tuning against test numbers, stop and say so.

**R4 — No invented numbers, ever.** Every metric that appears in any document, table, or chat message must be traceable to a file under `runs/<run_id>/metrics.json` produced by an actual execution. If a number is an estimate or a placeholder, label it `[ESTIMATE]` in red. This is an academic deliverable; fabricated results are the one unrecoverable failure.

**R5 — No monolithic array caches.** Data flows as a manifest CSV of file paths → PyTorch `Dataset` → `DataLoader`. Preprocessed images may be cached as individual JPEG/PNG files on disk, never as one giant `.npy`.

**R6 — Deterministic and reproducible.** Fixed seeds, seeds recorded in config, split CSVs committed to the repo, `requirements.txt` pinned. Anyone must be able to reproduce a run from `runs/<run_id>/config.yaml`.

**R7 — Framework: PyTorch + `timm`,** as specified in the proposal. It gives finer control over freezing schedules, weighted samplers, and mixed precision than the alternatives.

## 3. Datasets

### 3.1 EyePACS — primary
- Kaggle: `https://www.kaggle.com/datasets/dreamer07/eyepacs`
- ~35,126 fundus images, ~17,563 patients, left+right per patient, labels 0–4.
- Severe imbalance: ~73.5% class 0; classes 3+4 together under 5%.
- Labels come from the official CSV (`image`, `level`). **Derive `patient_id` and `eye` by splitting the `image` field on `_`.**

### 3.2 APTOS 2019 — extension / external validation
- Kaggle: `https://www.kaggle.com/datasets/mariaherrerot/aptos2019?select=test_images`
- ~3,662 labelled images, columns `id_code`, `diagnosis`, same 0–4 scale.
- **Critical:** APTOS `id_code` values are anonymised hashes with **no patient linkage** — you cannot recover which images share a patient. Therefore:
  - Treat each APTOS image as its own patient (`patient_id = f"aptos_{id_code}"`), and
  - Use APTOS primarily as a **held-out external test set** (train on EyePACS, evaluate on all of APTOS) — this is a much stronger result for the thesis than mixing them, and it's the honest way to handle the missing patient IDs.
  - A second arm may fine-tune on APTOS train and test on APTOS test for a domain-adaptation comparison.
  - Document this limitation explicitly. It's a *good* thesis paragraph, not a weakness to hide.

### 3.3 Acquisition — Kaggle-first, do not download EyePACS locally

**Confirmed:** `dreamer07/eyepacs` is **35.3 GB**, and its Kaggle usability rating is 0.18, meaning it is essentially undocumented. Assume nothing about its internal folder structure or CSV naming until you have listed it.

The local machine is a Windows laptop. Downloading and extracting 35 GB, then writing a preprocessed cache on top, is not viable there. Therefore:

- **All heavy work runs on Kaggle Notebooks**, where the dataset mounts read-only at `/kaggle/input/<slug>/` with no download step. Preprocessing, splitting, and training happen there.
- **The local machine holds only lightweight artefacts**: source code, split CSVs, `metrics.json`, plots, model checkpoints, and a small sample of images (~50, spanning all five grades) for developing and demoing the web app.
- **Every script must be path-agnostic**, reading dataset and output roots from config (`configs/kaggle.yaml` vs `configs/local.yaml`) so identical code runs in both places. Never hardcode a path.
- `/kaggle/input/` is read-only; outputs go to `/kaggle/working/`, which has its own size cap and is wiped between sessions. Persist checkpoints and split CSVs before a session ends — download them, or write them out as a Kaggle Dataset.
- The `kaggle` CLI is configured and working locally. Use it for **inspection**, not bulk download: `kaggle datasets files <slug>` lists contents without transferring the images.
- APTOS is far smaller and may be pulled locally if useful.

The first data task is a **listing and reconciliation report**, not a download: what files exist, the labels CSV columns, image counts, whether CSV rows match files on disk, and the class distribution. Report mismatches rather than silently dropping rows.

## 4. Technical plan

### 4.1 Preprocessing pipeline
1. Circle-crop the retinal disc, removing black borders (find the non-black bounding box on a blurred grayscale threshold).
2. Resize to 224×224 (primary, per proposal) with a config switch for 384 if GPU allows — record image size in every run config.
3. **Ben Graham-style enhancement** (`addWeighted(img, 4, GaussianBlur(img, (0,0), sigma), -4, 128)`), which is what makes microaneurysms visible; keep a CLAHE-on-green-channel variant as an ablation arm.
4. Normalise with ImageNet mean/std (matching `timm` model defaults, pulled from the model's own config — do not hardcode).
5. Cache preprocessed images to `data/processed/{dataset}/` as individual JPEGs.
6. Augmentation **train split only**: horizontal + vertical flip, rotation ±20°, brightness/contrast jitter, slight scale/shift. No augmentation on val/test. (Optional TTA at inference is separate and must be labelled as such.)

### 4.2 Splitting
- Group images by `patient_id`, compute each patient's max grade, stratify the **patient list** by that max grade, then split 70/15/15.
- Emit `data/splits/train.csv`, `val.csv`, `test.csv` with columns: `image_path, patient_id, eye, label, dataset, split`.
- Write `tests/test_no_leakage.py` asserting: zero patient overlap across all three pairs; every image assigned exactly once; class distributions logged; total count matches the source CSV. **This test must run and pass before any training run starts**, and again in CI-style fashion at the top of every training script.
- Commit the split CSVs. They are the ground truth for the whole project.

### 4.3 Imbalance handling (ablation arms)
Run these as named, comparable experiments — the comparison itself is a thesis chapter:
- `A` natural distribution + plain cross-entropy (honest baseline)
- `B` `WeightedRandomSampler` balancing classes per epoch + cross-entropy
- `C` natural distribution + class-weighted cross-entropy
- `D` balanced sampler + focal loss
- `E` ordinal/regression head (single output, MSE/SmoothL1) with thresholds optimised on validation for QWK

Arm E frequently wins on QWK for DR grading because the classes are ordered — grading a 4 as a 0 is far worse than a 4 as a 3, and softmax cross-entropy is blind to that. Report it.

### 4.4 Models
- **Baseline:** ResNet18 (as specified in the approved proposal).
- **Primary:** EfficientNet-B0 (as specified in the approved proposal).
- **Optional third:** DenseNet121, only if time allows after Phase 4 — dense connectivity tends to do well on fine-grained retinal texture, and a third architecture strengthens the comparison chapter. Do not let it delay the core two.
- All via `timm`, ImageNet pretrained, **fully fine-tuned**. Optional 1–2 epoch frozen warmup for the head, then unfreeze everything. Never train frozen-only.
- Training: AdamW, cosine schedule with warmup, mixed precision (`torch.amp`), gradient clipping, label smoothing 0.1 for CE arms, early stopping on **validation QWK** (not loss, not accuracy), best-checkpoint saving.
- Batch size auto-scaled to available VRAM via config.

### 4.5 Evaluation
Primary metric: **Quadratic Weighted Kappa**. Also compute and store:
- Accuracy, balanced accuracy, macro & per-class precision/recall/F1
- Confusion matrix (counts + row-normalised), saved as PNG and JSON
- **Referable DR** binary view (grade ≥ 2): sensitivity, specificity, ROC-AUC, PR-AUC — this is the clinically meaningful screening framing and it belongs in the thesis
- 95% confidence intervals by bootstrap resampling of the test set (n=1000)
- Inference time per image (CPU and GPU), model size on disk, parameter count
- Per-arm comparison table auto-generated into `docs/EXPERIMENTS.md`

### 4.6 Explainability
- Grad-CAM and Grad-CAM++ on the final convolutional block.
- Produce a qualitative panel: for each grade 0–4, several correctly classified and several misclassified examples with heatmap overlays.
- Sanity check: heatmaps should concentrate on lesions/vasculature, not on image borders or the black background. If they light up borders, the preprocessing is leaking artefacts — investigate, don't ship it.
- Save overlays under `runs/<run_id>/gradcam/`.

### 4.7 Deployment — real web stack (deliberate deviation from the proposal)

The proposal names Streamlit. **We are not using Streamlit.** The system will be built as a proper client/server web application. Log this in `docs/DECISIONS.md` on day one with the rationale below, and remind the user to clear it with the supervisor, since it changes §9 (Tools and Technologies) of the approved proposal. The *objective* is unchanged — a web-based diagnostic interface — only the implementation is upgraded.

Rationale to record: separation of concerns between inference and presentation, a documented REST API that could be consumed by a hospital system, real request handling and error states, proper file-upload validation, and a UI that can be assessed as software engineering work rather than a script.

**Stack:**
- **Backend:** FastAPI + Uvicorn, on the same Python version as the training environment. Keeps the model in the same language as training, so no export/conversion risk. Pin the exact version in `requirements.txt` and the Dockerfile base image.
  - `POST /api/v1/predict` — multipart image upload → JSON: predicted grade, per-class probabilities, model version, inference time, and a URL for the Grad-CAM overlay.
  - `GET /api/v1/health` — model loaded, version, device.
  - `GET /api/v1/model-info` — architecture, training run ID, test-set metrics.
  - Pydantic response schemas; auto-generated OpenAPI docs at `/docs` (put a screenshot of this in the thesis).
  - Upload validation: MIME type, magic-byte check, max size, image dimensions, rejection of non-fundus-looking input where feasible. Return proper HTTP status codes, never a stack trace.
  - Model loaded once at startup, not per request. Grad-CAM generated on demand and written to a temp store with a short TTL.
- **Frontend:** Next.js (TypeScript) + Tailwind CSS.
  - Upload area with drag-and-drop and preview.
  - Results view: predicted grade 0–4 with the clinical label (No DR / Mild / Moderate / Severe / Proliferative), a confidence bar chart across all five classes, the Grad-CAM overlay side-by-side with the original (with an opacity slider), inference time, and a referral recommendation.
  - Loading and error states that actually handle a backend that is down or slow.
  - A persistent, prominent banner: research prototype, **not a medical device**, not for clinical use.
  - Responsive; must be demonstrable on a projector.
- **Packaging:** `docker-compose.yml` bringing up both services, plus a documented non-Docker path (two terminals) since the demo machine may not have Docker.
- **Constraint:** the whole thing must run CPU-only on a 16 GB RAM laptop with acceptable latency. Benchmark and record single-image CPU inference time.

Keep the backend importable as a plain Python module (`src/inference/predictor.py`) so the API is a thin wrapper over it and the same code path can be unit-tested without HTTP.

## 5. Use subagents

Set up specialist subagents under `.claude/agents/` (one markdown file each with YAML frontmatter: `name`, `description`, optional `tools` and `model`). If the subagent format has changed since your training data, run `/agents` and follow whatever the current interface says rather than guessing. Delegate proactively; each agent keeps its own context so the main thread stays clean.

| Agent | Owns | Notes |
|---|---|---|
| `data-engineer` | Download, inspect, preprocess, cache, build manifests | Must never touch splits directly |
| `leakage-auditor` | Splitting logic, leakage tests, any change to split CSVs | **Gatekeeper.** Any PR touching data partitioning is reviewed by this agent before it lands. Its verdict is recorded in `docs/DECISIONS.md`. |
| `training-engineer` | Model definitions, training loops, schedulers, checkpoints, ablation arms | Reads split CSVs read-only |
| `eval-analyst` | Metrics, confusion matrices, bootstrap CIs, comparison tables, plots | Only source of numbers that enter documents |
| `xai-engineer` | Grad-CAM, overlays, qualitative panels, sanity checks | |
| `backend-engineer` | FastAPI service, inference wrapper, upload validation, API schemas, CPU performance | Owns `src/inference/` and `backend/` |
| `frontend-engineer` | Next.js UI, upload flow, results view, Grad-CAM display, error states | Owns `frontend/`; consumes the API contract only |
| `docs-writer` | Thesis chapters, figures, README, supervisor-facing summaries | Pulls numbers only from `runs/*/metrics.json` |
| `code-reviewer` | Reviews every substantial change for correctness, leakage risk, silent failures | Invoked before each phase is marked complete |

Working rule: whenever a task spans more than one of those columns, dispatch the relevant agents rather than doing it inline.

Also create slash commands in `.claude/commands/`:
- `/status` — print current phase, last run, next action from `PROGRESS.md`
- `/newrun <arm>` — scaffold a run directory + config from a template
- `/report` — regenerate the experiment comparison table from `runs/`
- `/handoff` — update all tracking files and write a session summary

## 6. Progress tracking — so a new model never needs re-briefing

This is a hard requirement. The user will switch models and machines. Maintain these files and keep them current:

| File | Contents | Updated |
|---|---|---|
| `CLAUDE.md` | The living project constitution: objective, hard rules (§2), stack, repo map, conventions, current phase, "read these files in this order to get oriented". Auto-loaded by Claude Code every session. | Whenever a rule or convention changes |
| `PROGRESS.md` | Phase checklist with `[ ] / [~] / [x]`, per-phase acceptance criteria, "NEXT ACTION" line at the top, blockers | **End of every session, no exceptions** |
| `docs/DECISIONS.md` | ADR-style entries: date, decision, alternatives considered, rationale, who approved | Every non-obvious choice |
| `docs/EXPERIMENTS.md` | Auto-generated table: run_id, model, arm, img size, epochs, val QWK, test QWK, balanced acc, referable sensitivity, notes, checkpoint path | After every run |
| `runs/<run_id>/` | `config.yaml`, `metrics.json`, `train_log.csv`, plots, checkpoint pointer | Per run |
| `state/session_handoff.md` | Last session summary, exact next command to run, open questions for the user, known-broken things | End of every session |

**Session protocol.** First action of every session: read `CLAUDE.md`, `PROGRESS.md`, `state/session_handoff.md`, and the latest `docs/EXPERIMENTS.md`, then state in 5 lines where the project stands and what you're about to do. Last action of every session: update those files and commit. If the user says "continue", that protocol is the whole answer to "what were we doing?" — do not ask them to re-explain the project.

Git: commit at every meaningful checkpoint with descriptive messages. `.gitignore` must exclude `data/raw/`, `data/processed/`, `*.pth`, `*.npy`, and `kaggle.json`. Split CSVs, configs, and metrics JSON **are** committed.

## 7. Repo layout

```
fyp-dr/
├── CLAUDE.md
├── BOOTSTRAP.md
├── PROGRESS.md
├── README.md
├── requirements.txt
├── .claude/{agents,commands}/
├── configs/{base.yaml, arm_a.yaml, ... , kaggle.yaml, local.yaml}
├── data/
│   ├── raw/{eyepacs,aptos}/        # gitignored
│   ├── processed/                  # gitignored
│   └── splits/{train,val,test}.csv # committed
├── src/
│   ├── data/{download.py,inspect.py,preprocess.py,split.py,dataset.py,sampler.py}
│   ├── models/{factory.py,heads.py}
│   ├── train/{train.py,loop.py,losses.py,schedulers.py}
│   ├── eval/{metrics.py,evaluate.py,bootstrap.py,thresholds.py,plots.py}
│   ├── xai/gradcam.py
│   ├── inference/predictor.py      # framework-agnostic, testable
│   └── utils/{config.py,seed.py,logging.py,paths.py}
├── backend/
│   ├── main.py                     # FastAPI app
│   ├── routers/predict.py
│   ├── schemas.py
│   └── Dockerfile
├── frontend/                       # Next.js + TypeScript + Tailwind
│   ├── app/
│   ├── components/
│   ├── lib/api.ts
│   └── Dockerfile
├── docker-compose.yml
├── tests/{test_no_leakage.py,test_dataset.py,test_metrics.py,test_preprocess.py,test_api.py}
├── runs/<run_id>/
├── state/session_handoff.md
├── docs/{proposal.pdf,DECISIONS.md,EXPERIMENTS.md,thesis/}
└── notebooks/kaggle_train.ipynb
```

## 8. Phases (aligned to the proposal's incremental model)

**Phase 0 — Orientation (no code).** Read the proposal. Inspect the environment (OS, Python version, GPU availability, disk space, whether the Kaggle CLI is configured). Produce: the proposed plan mapped to the phases below, a list of every assumption you're making, the defences you'll put in place against each failure mode in §1, and any questions for the user. Then **stop and wait for approval.**

**Phase 1 — Data foundation.** Scaffold repo, tracking files, agents, config system. Download/locate data. Inspect and reconcile CSVs against files on disk. Build patient-level splits. Write and pass leakage tests. Emit a data report: per-class counts per split, patient counts, image-size distribution, corrupt-file list. Acceptance: `tests/test_no_leakage.py` green, distributions documented.

**Phase 2 — Preprocessing.** Circle crop, enhancement, resize, cache. Visual QA: save a contact sheet of before/after for ~20 images across all grades so a human can eyeball it. Acceptance: user approves the visual QA sheet.

**Phase 3 — Baseline.** ResNet18, arm A, short schedule. Purpose is a *correct* pipeline end to end, not a good score. Acceptance: val QWK clearly above 0 and confusion matrix not collapsed to one class.

**Phase 4 — Primary models + ablations.** EfficientNet-B0 and DenseNet121 across arms A–E. Full comparison table with CIs. Acceptance: a defensible best model selected on validation QWK.

**Phase 5 — Explainability.** Grad-CAM panels, sanity checks, qualitative analysis written up.

**Phase 6 — External validation.** Best model evaluated on APTOS unseen. Report the domain-shift drop honestly — a drop is expected and is itself a finding.

**Phase 7 — Deployment.** Inference wrapper, FastAPI backend with tested endpoints, Next.js frontend, Docker Compose, CPU inference benchmarks, model size comparison, and a written demo script for the viva. Acceptance: a clean run from `docker compose up` to a prediction with Grad-CAM in the browser, plus `tests/test_api.py` green.

**Phase 8 — Documentation.** Thesis chapters mirroring the proposal's structure, all figures regenerated from `runs/`, limitations section (APTOS patient IDs, no clinical validation, single-source training data), future work.

## 9. Interaction style

- Ask before large or irreversible operations (multi-GB downloads, long training runs, deleting caches).
- Long training goes in a script the user launches on Kaggle/Colab, not a foreground process in this session. Give the exact command.
- When something fails, show the real error and diagnose it — do not silently fall back to a smaller/easier configuration.
- Environment note: the team has been working on Windows with a `.venv`. Handle Windows path separators, and set `num_workers=0` on Windows unless the script is guarded by `if __name__ == "__main__":`.
- Keep explanations short. The user is technical.
- If you are ever unsure whether something risks leakage, assume it does and escalate to `leakage-auditor`.

**Begin with Phase 0. Write no code until the plan is approved.**

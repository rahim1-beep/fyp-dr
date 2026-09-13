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

**Phase 7 — the web application.** Phases 4, 5 and 6 are COMPLETE, including the
surround-randomisation remedy. No Kaggle work is outstanding.

Backbone fixed at EfficientNet-B0 (DECISION-039); arm E (ordinal-regression head) ships,
seed 42 (DECISION-069). **Report the 3-seed means: validation QWK 0.7569, referable
sensitivity 0.7360 at 95% specificity** — below the 0.80 screening reference, which is a
stated limitation rather than an open problem now.

Two findings the write-up turns on:

* **The model does not generalise** beyond its training population (DECISION-060). G1–G3
  passed; G4 fired — a demonstrated dependence on retinal framing, stronger out-of-domain.
  This does NOT mean performance collapsed.
* **The remedy outcome is AMBIGUOUS and stays that way** (DECISION-067). Do not upgrade
  it to "worked".

Built and live: the coverage guard (calibrated, DECISION-070), `src/inference/predictor.py`,
the FastAPI backend, and the Next.js frontend (DECISION-072). Remaining: Docker Compose and a CPU benchmark if wanted, then Phase 8.

*(This section previously described Phase 4 as in progress and quoted seed 42's own
0.7464. Both were five phases stale.)*

See `PROGRESS.md` for the live checklist and NEXT ACTION, and
`state/session_handoff.md` for the standing rules that are easiest to break.


---

# DR Grading Web App — frontend (Phase 7)

Appended 2026-09-09. **Everything above still holds** — the backend rules, R1–R7, the
balancing mechanism, the Kaggle-first environment. This section adds the frontend and
contradicts none of it.

## What this project is actually about

The contribution of the thesis is **honest measurement**, not the score. The model is
below the clinical reference bar and the write-up says so. An interface that oversells it
destroys the thing being defended. Every design decision resolves toward "state the
limitation plainly", never toward "make the number look good".

## Stack (fixed — do not propose alternatives)

Next.js (App Router) + TypeScript + Tailwind. **Not Streamlit** — DECISION-001.
Backend: FastAPI at `http://localhost:8000`, CORS fully open, local demo only.

## The contract lives in a file, not in memory

**`docs/api-contract.md`** is authoritative and was read off the source. Recorded payloads
for every state are in **`fixtures/expected/`**. Do not reconstruct response shapes from
recollection; read those.

## Hard rules

1. **All displayed metrics come from `GET /meta`.** Never hardcode a number the API
   returns. Call it once on load.
2. **Never render `0.7615` or `0.7464`** — seed 42's own figures. The UI reports the
   3-seed mean (`0.7569`, `0.7360`) only. Two backend tests enforce this server-side; do
   not reintroduce the numbers client-side.
3. **`disclaimer[]` renders verbatim, all four strings, in the page flow** — on the
   Result view **and** the Upload view. Not a modal, tooltip, accordion, "learn more", or
   truncation. See the note below on why both screens get the full four.
4. **Show `preprocessed_png` as the graded image**, never the user's original. The
   original is not what the model saw. You may show the original alongside, clearly
   labelled as the input file.
5. **Always render `guard.message`.** It is written for a human and explains itself. Do
   not paraphrase it or substitute your own copy.
6. **`grade` and `referable` are independent and may disagree.** Correct behaviour, not a
   bug. See below.
7. **The provenance line appears on the page** (footer is fine).
8. **No storage, no accounts, no patient data entry.** Upload → result → done. Nothing
   persists.

### Why the full disclaimer on both screens

The brief left this ambiguous, so it is decided here: **all four strings on Upload as well
as Result.** The Upload screen is where someone decides whether to use the thing at all,
and shortening the caveats at exactly that moment is the "make it look better" pressure
this project exists to resist. Four short strings is not a usability burden for an
audience of a thesis panel and clinicians. The alternative considered and rejected was a
one-line summary on Upload with the full set on Result.

## The disagreement (do not "fix" this)

`grade` comes from four cut points tuned for agreement with human graders. `referable`
comes from a separate threshold (**0.9488**) tuned for 95% specificity. That threshold
sits **below** the grade-1→2 cut point (**1.6216**), so any score in that 0.673-wide band
yields `grade: 1` "Mild" **and** `referable: true`.

Present them as two distinct answers of equal weight:

```
Estimated grade      Mild
Referral             Refer
```

Never derive one from the other. Never suppress or soften the disagreement. Never add a
reconciliation that picks a winner. Making this read as two correct answers rather than a
contradiction is the hardest UX problem in the app — solve it in the layout, not a
footnote.

## Language rules

Banned: "diagnosis", "diagnose", "detected", "confirmed", "AI-powered", "accurate",
"confident", "clear", "normal", "healthy", "all clear".

- This **estimates a grade**. It does not diagnose.
- **No confidence percentage, ever.** The model does not produce one; deriving one from
  the score would be fabrication.
- **No green tick or success state for grade 0.** Sensitivity is 0.736 — roughly one
  referable case in four is missed. Grade 0 is a reading, not reassurance, and gets the
  same visual weight as every other grade.
- Errors state what happened and what to do. They do not apologise.

## Colour rule

Do not build a red-amber-green severity ramp. Traffic-light encoding says "green = fine",
which is exactly the claim the model cannot support. Use a single-hue progression so
severity reads as position on a scale rather than as verdict. Colour is never the only
carrier of meaning — the grade is always stated in words.

## The four guard states

| `guard.status` | `graded` | UI |
|---|---|---|
| `ok` | true | show grade normally |
| `framing_below_training_range` | true | grade **+ prominent warning** |
| `framing_above_training_range` | true | grade **+ prominent warning** |
| `no_retina_detected` | false | **no grade** — show `guard.message`, invite re-upload |

The `/predict` response is a **discriminated union on `graded`**. The declined variant has
no `grade` key at all. Type it so the compiler rejects any path reading `grade` without
narrowing.

Grade names by index: `["No DR", "Mild", "Moderate", "Severe", "Proliferative"]`.
`grade_name` is already in the response — use it, do not re-derive it.

## Test fixtures

`fixtures/` — six verified images covering ok, both framing warnings, no-retina, 413 and
400. **They are training images**; they exercise UI states and must never be screenshotted
as a demo of performance. See `fixtures/README.md`.

## Running it

```
set FYP_CHECKPOINT=<path to seed 42 best.pth>
.venv\Scripts\python -m uvicorn backend.app:app --reload   # :8000
cd frontend && npm run dev                                   # :3000
```

The backend refuses to start without checkpoint, deployment manifest, and coverage
calibration. Deliberate — a misconfigured deploy fails loudly rather than grading with a
silently absent safety check. **Do not add fallbacks or defaults that would let it start
anyway.**

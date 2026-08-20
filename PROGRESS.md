# PROGRESS.md

> **NEXT ACTION:** Phase 2 — write `src/data/preprocess.py`, then render the contact sheet
> from the already-selected `docs/phase2_qa_sample.csv` (20 images, 17.1 MB).
> **HARD GATE: do not cache all 35,126 images until the user signs off on the sheet.**
>
> **First action of the session:** confirm the 9 subagents in `.claude/agents/` now
> dispatch by name (they were created mid-session last time and needed a restart).

**Current phase:** Phase 1 — Data foundation
**Last updated:** 2026-08-20

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Orientation `[x]`

- [x] Read `BOOTSTRAP.md` and `docs/proposal.pdf`
- [x] Environment audit (OS, Python, GPU, disk, Kaggle CLI)
- [x] Dataset reconnaissance without downloading (EyePACS path pattern confirmed)
- [x] Plan, assumptions, failure-mode defences, questions F1–F6
- [x] **User approval received 2026-08-20**

**Answers recorded:** Python 3.12 toolchain (DECISION-002) · Kaggle account `rah098`, quota
intact · preprocessed cache → private Kaggle Dataset, report size first · private GitHub
repo, **ask before first push** · DECISION-001 logged, supervisor confirmation before
Phase 7 · 224×224 headline (DECISION-005) · no Intel XPU/IPEX (DECISION-003)

---

## Phase 1 — Data foundation `[~]`

**Acceptance:** `tests/test_no_leakage.py` green; distributions documented.

- [x] Repo skeleton, `.gitignore`, tracking files
- [x] `.venv` on Python 3.12.10
- [x] `docs/DECISIONS.md` seeded with DECISION-001 … 005
- [~] `.claude/agents/` (9) and `.claude/commands/` (4)
- [ ] `requirements.txt` pinned for cp312 (Windows CPU + Linux CUDA)
- [ ] Config system: `configs/{base,local,kaggle}.yaml`
- [x] `requirements.txt` pinned for cp312 (Windows CPU + Linux CUDA)
- [x] Config system: `configs/{base,local,kaggle}.yaml` + `arm_a..arm_f.yaml`
- [x] **Locate labels CSV; report real column names, dtypes, row count, distinct labels**
      — found at `trainLabels.csv/trainLabels.csv`; schema `image,level`
- [x] **Data reconciliation report** — zero mismatches in either direction
- [x] **USER SIGN-OFF RECEIVED 2026-08-20**
- [x] Patient-level 70/15/15 split, stratified on each patient's max grade, seed 42
- [x] Emit `data/splits/{train,val,test}.csv` + `aptos_{train,val,test}.csv`
- [x] `tests/test_no_leakage.py` — **38 tests, all green**
- [x] APTOS manifest (pooled, `patient_id = aptos_{id_code}`)
- [x] APTOS real class distribution measured (DECISION-007)
- [x] `leakage-auditor` independent review — **PASS** (DECISION-008)
- [x] Auditor's 4 actionable observations fixed (small-strata guard, venv alignment,
      version provenance, test blind spots)
- [x] Commit split CSVs

**Phase 1 acceptance met:** leakage tests green, distributions documented, audit passed.

**Blockers:** none.

---

## Phase 2 — Preprocessing `[~]`

**Acceptance:** user approves the visual QA contact sheet.
**HARD GATE: do not cache all 35,126 images until the user signs off on the sheet.**

- [x] QA sample selected — `docs/phase2_qa_sample.csv` (20 images, seed 42, train split only)
- [ ] `src/data/preprocess.py` — circle-crop (non-black bbox on blurred grayscale
      threshold) → Ben Graham → 224×224 → individual JPEGs (R5)
- [ ] `scan_quality()` — pixel-stat pass (mean brightness, saturation, disc centroid
      offset) to find genuinely dark / over-exposed / off-centre images. File size alone
      cannot identify those.
- [ ] Render the contact sheet from `docs/phase2_qa_sample.csv`
- [ ] **← USER SIGN-OFF GATE**
- [ ] Cache all 35,126 EyePACS + 3,662 APTOS images
- [ ] **Measure and report actual cache size before uploading**
- [ ] Persist as private Kaggle Dataset under `rah098`; log in DECISIONS.md
- [ ] CLAHE-on-green variant kept as an ablation arm

### Contact sheet requirements (user, 2026-08-20)

1. **Original and processed side by side at the same display size**, with **patient ID and
   grade labelled on each pair**.
2. **Weighted toward grades 1 and 2 — at least 6 of ~20.** That is where circle-crop and
   Ben Graham either preserve microaneurysms and small haemorrhages or destroy them;
   grade 0/4 examples cannot answer the question. *Current sample has 9.*
3. **2–3 deliberately bad inputs** — the 8 KB minimum file, very dark, over-exposed, or
   off-centre. The web app will receive exactly these. *Current sample has 3 selected by
   file size; the pixel-stat scan should confirm or improve them.*

### Selected QA sample

20 images, all TRAIN split, 17.1 MB total to download.
Grades: 0→4, 1→4, 2→5, 3→4, 4→3. Poor-quality picks:

| image | grade | size | why |
|---|---|---|---|
| `3829_left` | 2 | **8.1 KB** | smallest file in the entire dataset |
| `39106_left` | 3 | 16.6 KB | rare-class image that may be degraded |
| `15942_right` | 0 | 15.9 KB | very small file |

Only **21 of 35,126** files fall under 50 KB, against a median of ~1,100 KB — these are
almost certainly blank, very dark, or failed captures. Four of the 21 are in `test` and two
in `val`; if preprocessing cannot rescue them, that is worth a note in the write-up, but
**they are not to be dropped** without a DECISIONS.md entry.

---

## Phase 3 — Baseline `[ ]`

**Acceptance:** val QWK clearly above 0; confusion matrix not collapsed to one class.

- [ ] ResNet18, arm A, short schedule — purpose is a *correct* pipeline, not a good score

---

## Phase 4 — Primary models + ablations `[ ]`

**Acceptance:** a defensible best model selected on validation QWK.

- [ ] EfficientNet-B0 × arms A–E
- [ ] ResNet18 × arms A–E
- [ ] **Arm F** — EyePACS + APTOS pooled training, evaluated on the EyePACS test set
      (DECISION-007). Settles the supervisor's imbalance suggestion empirically.
- [ ] DenseNet121 (optional, only if time allows)
- [ ] Comparison table with bootstrap CIs → `docs/EXPERIMENTS.md`
- [ ] 384×384 ablation (only if quota survives)

**Reporting constraint:** grades 3 and 4 are reported with bootstrap 95% CIs, never bare
point estimates (DECISION-006 — only 133 grade-3 and 98 grade-4 test images).

---

## Phase 5 — Explainability `[ ]`

- [ ] Grad-CAM + Grad-CAM++ on final conv block
- [ ] Qualitative panel: correct + misclassified per grade 0–4
- [ ] **Sanity gate:** heatmaps on lesions/vasculature, not borders or background

---

## Phase 6 — External validation `[ ]`

- [ ] Best model → all of APTOS, cold. Report the domain-shift drop honestly.

**Applies to arms A–E only.** Under Arm F, APTOS is training data and cannot also be the
external validation set (DECISION-007). If Arm F becomes the headline model, the write-up
must state that it trades a generalisation claim for in-domain performance.

---

## Phase 7 — Deployment `[ ]`

**Blocked on:** written supervisor confirmation of DECISION-001.
**Acceptance:** clean `docker compose up` → prediction with Grad-CAM in browser;
`tests/test_api.py` green.

- [ ] `src/inference/predictor.py` (plain module, testable without HTTP)
- [ ] FastAPI: `/api/v1/predict`, `/health`, `/model-info`; upload validation
- [ ] Next.js UI: upload, results, Grad-CAM w/ opacity slider, error states,
      "not a medical device" banner
- [ ] Docker Compose + documented two-terminal path
- [ ] CPU inference benchmark on a 16 GB target
- [ ] Viva demo script

---

## Phase 8 — Documentation `[ ]`

- [ ] Thesis chapters mirroring the proposal structure
- [ ] All figures regenerated from `runs/`
- [ ] Limitations: APTOS patient IDs, no clinical validation, single-source training data
- [ ] Future work

---

## Open items for the user

- **GitHub:** private repo + collaborator access for Ameena Ahmed and Muhammad Ali
  Abdullah. **Ask before the first push.**
- **Supervisor:** written confirmation of DECISION-001 before Phase 7.

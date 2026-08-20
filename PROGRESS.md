# PROGRESS.md

> **NEXT ACTION:** Phase 2 — preprocessing. Write `src/data/preprocess.py` (circle-crop →
> Ben Graham → 224×224 → individual JPEGs) and produce the visual QA contact sheet of ~20
> before/after pairs spanning all five grades. **Runs on Kaggle, not locally.**
> **Wait for user approval of the contact sheet before caching all 35,126 images.**

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

## Phase 2 — Preprocessing `[ ]`

**Acceptance:** user approves the visual QA contact sheet.

- [ ] Circle-crop retinal disc (non-black bbox on blurred grayscale threshold)
- [ ] Ben Graham enhancement; CLAHE-on-green kept as an ablation arm
- [ ] Resize 224×224, cache as individual JPEGs (R5)
- [ ] Report cache size **before** uploading
- [ ] Persist as private Kaggle Dataset under `rah098`; log in DECISIONS.md
- [ ] Contact sheet: ~20 before/after pairs spanning all five grades

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

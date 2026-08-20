# PROGRESS.md

> **NEXT ACTION:** **The user must launch the full cache build on Kaggle** — it is a
> ~2.5 h job and CLAUDE.md §7 forbids running it as a foreground session process.
> Everything it needs is prepared: `notebooks/make_bundle.py` → upload → paste
> `notebooks/phase2_build_cache.py`. That cell also runs the reconciliation, the leakage
> tests, and the measured-size report before it will say READY TO PUBLISH.
>
> **Gate status: contact sheet APPROVED by the user 2026-08-20.** The sheet was re-rendered
> once more after the approval, with DECISION-016's 2.5% erosion applied, and is visually
> unchanged.

**Current phase:** Phase 2 — Preprocessing (code complete; cache build is the user's to run)
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
- [x] `leakage-auditor` Phase 2 entry gate — **PASS**, 3 binding conditions, all satisfied
      (DECISION-012, DECISION-013)
- [x] `src/data/fetch_sample.py` — pulls named images from Kaggle without the 35.3 GB
- [x] 20 QA images downloaded, every byte count matching the remote listing
- [x] `src/data/preprocess.py` — retina mask → square crop → Ben Graham (normalised
      convolution) → 224×224 → individual JPEGs (R5)
- [x] `scan_quality()` — 8 pixel statistics + advisory flags, fixed absolute thresholds
- [x] `src/data/contact_sheet.py` — before/after renderer
- [x] `tests/test_preprocess.py` (22) + `tests/test_preprocess_plumbing.py` (10)
      — full suite **70 green**
- [x] `code-reviewer` review — **CHANGES REQUIRED, 7 defects, all fixed**
      (DECISION-014, DECISION-015)
- [x] QA cache rebuilt with the fixed pipeline → `data/processed/qa/`
- [x] Contact sheet **re-rendered from the actual cache files**, scale-matched,
      3 panels per image → `docs/phase2_contact_sheet.png`
- [x] **USER SIGN-OFF RECEIVED 2026-08-20** — sheet approved
- [x] DECISION-016: 2.5% boundary erosion, with the residual measured and reported
- [x] `src/data/reconcile_cache.py` — the leakage-auditor's 4 conditions, one command
- [x] `notebooks/make_bundle.py` + `notebooks/phase2_build_cache.py` — the Kaggle build
- [x] Cache-path uniqueness proven **before** the build: 38,788 split rows → 38,788
      distinct cache paths, **0 claimed by more than one split**
- [ ] **← USER RUNS THE KAGGLE BUILD** (~2.5 h, CPU, no GPU quota)
- [ ] Cache all 35,126 EyePACS + 3,662 APTOS images
- [ ] Persist as private Kaggle Dataset under `rah098`; log in DECISIONS.md
- [ ] CLAHE-on-green variant kept as an ablation arm (implemented, not yet run)

### Measured on the 20 QA images — replaces the earlier [ESTIMATE]

| | Value |
|---|---|
| Mean output size | **21.4 KB/image** [MEASURED] |
| Projected EyePACS cache | **0.72 GB** (35,126 images) |
| Projected APTOS cache | **0.08 GB** (3,662 images) |
| **Total** | **~0.80 GB** — a 44× reduction from 35.34 GB |
| Throughput | 0.94 s/image single-threaded [MEASURED] |
| **Full cache build** | **10.1 h @1 worker → 2.5 h @4 workers** |

Comfortably inside `warn_working_dir_gb: 15`. The earlier 0.82 GB estimate was accurate;
it is now a measurement. **Use `--workers 4` on Kaggle** — a single-threaded build would
not fit in a session. Parallel output verified byte-identical to serial.

### The code review found seven more — two invalidated the gate artefact itself

`code-reviewer` returned **CHANGES REQUIRED**. All seven are fixed and test-locked.

| # | Defect | Fix |
|---|---|---|
| 1 | Two rows sharing a filename stem silently overwrote one cache file, both reporting `ok` | `RuntimeError` on duplicate `cache_path` |
| 2 | A bright corner blob joined the retina in the crop extent: side 606→902 px, retina rendered **0.67×**, status still `ok` | largest connected component only (DECISION-014) |
| 3 | The no-retina fallback emits a different image domain and left no record under `--no-scan` | `status = "ok:no-retina"`, set independently of the scan |
| 4 | A NaN `image_path` raised out of the pool and destroyed the run — on the 2.5 h Kaggle build, the whole session, no stats CSV | whole worker body wrapped; one bad row is one bad row |
| 5 | **The sheet's "before" panel was downsampled below the "after" panel** — retina 139–203 px vs 224, every pair biased **0.62×–0.91×** in preprocessing's favour | original square-cropped from native pixels (DECISION-015) |
| 6 | **The sheet rendered a recomputation, not the cache.** The q95 round-trip costs mean 3.60/255, max 64, and lifts 39.3% of the masked surround off zero | `--cache-root` required; the panel is the decoded cache file |
| 7 | A size-mismatched download stayed on disk and counted as `cached` forever | `dest.unlink(missing_ok=True)` |

Defects 5 and 6 mean the sheet in commit `c948c1a` was not a valid gate artefact and was
**never approved**. The current sheet is rendered from a cache rebuilt after fix 2.

The reviewer independently verified as correct: the normalised convolution (within 1/255
of the exact masked form), `square_crop`'s index arithmetic across 5 edge cases, the
multiprocessing count/order/bit-identity, BGR handling end to end, and that **nothing in
the pipeline computes a statistic across images** — the leakage surface is clean.

### Three defects found and fixed by building the sheet

The contact sheet earned its place as a gate — it exposed three problems that metrics
could not have:

1. **White halo around every retina** (DECISION-009). Ben Graham's blur straddled the
   retina/black boundary, ringing each image at 1.44× inner brightness — brighter than
   any lesion, and on grade-4 images it washed lesions out entirely. Fixed with a
   normalised convolution over the retina mask.
2. **Black bars and camera-dependent lesion scale** (DECISION-010). Bounding-box-then-pad
   left the retina filling a fraction of the frame that depended on the camera's aspect
   ratio. Fixed with a square crop centred on the retina.
3. **340-hour build time** (DECISION-011). The naive large-sigma Gaussian took ~35 s/image.
   Fixed with a downscaled-domain blur: 145× faster, max error 4/255.

A fourth was caught by the statistics rather than the image: the focus measure was
resolution-dependent and flagged **19 of 20 images as blurred** (DECISION-013).

### The rim, measured properly — DECISION-016

The user chose 2.5% mask erosion over Ben Graham's 0.9 r (19% of retinal area, including
the periphery where proliferative disease appears, against a grade-4 test support of 98).

**The residual is not the flattering number.** A fixed 0.90–1.00 r annulus barely moves,
**1.610 → 1.516**, because trimming the boundary also moves which pixels fall in a fixed
radial band. Measured against depth from the *actual* edge, the excess is steep and
shallow — 2.61× at the outermost 1%, 1.86× at 1–2.5%, 1.59× at 2.5–5%, and back to the
interior level by 10%.

So 2.5% erosion removes the two worst bands (**≈63% of the peak excess**, 2.61× → ≈1.59×)
and costs 3.2% of the area, but leaves a ≈1.6× edge. It suppresses the peak; it does not
eliminate the rim. **Reported to the user before the build, as they asked.** Revisit only
on Phase 5 Grad-CAM evidence that the model keys on the rim — not on aesthetics.

### Quality-flag result on the QA sample

Exactly the **3 deliberately-bad images** flagged, 17 clean — and the pixel statistics
independently rank the same three last as file size did, despite sharing no inputs.

| image | grade | flags |
|---|---|---|
| `3829_left` | 2 | tiny-retina, dark, mostly-black, off-centre, blurred, low-contrast, bright-artefact |
| `39106_left` | 3 | dark, blurred |
| `15942_right` | 0 | blurred |

**None of these are dropped.** Flags are advisory (DECISION-013); the "Excluded data rows"
section of `docs/DECISIONS.md` remains **None**.

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

Started **in parallel with the cache build**, not behind it — the data layer is testable
against a synthetic cache, so none of this waits on Kaggle.

- [x] torch 2.9.1+cpu / torchvision 0.24.1+cpu / timm 1.0.28 installed locally, so the
      data layer is testable on this machine (the Kaggle pins still need confirming
      against the image on the first notebook run)
- [x] `src/data/dataset.py` — `DRDataset`, reads the flat cache, **BGR→RGB exactly once**,
      returns `(image, label, index)` so predictions join back to manifest rows
- [x] `src/data/sampler.py` — `WeightedRandomSampler` per §2.1, and `build_loaders`
      **refuses a sampler on val or test** rather than trusting the caller
- [x] `tests/test_dataset.py` — 24 tests. **Full suite: 99 green.**
- [~] `leakage-auditor` review of the data layer — dispatched
- [ ] `src/models/factory.py` — timm, full fine-tune, normalisation from `default_cfg`
- [ ] `src/train/` — loop, AdamW, cosine+warmup, AMP, grad clip, early stopping on val QWK
- [ ] `src/eval/metrics.py` — QWK first
- [ ] ResNet18, arm A, short schedule — purpose is a *correct* pipeline, not a good score

**Design notes worth not relitigating:**

- **Augmentation is refused on val/test**, not silently ignored. A flip on the validation
  set changes what early stopping measures and nothing in the log would show it.
- **`num_samples = len(train)`** whether or not balancing is on, so two arms see the same
  number of gradient steps per epoch. Arms that don't are not comparable, and the
  comparison is the thesis chapter.
- **`describe_balance` draws from a clone of the sampler.** Iterating the real one would
  advance its generator, so merely logging the balance table would change the first
  epoch — an R6 bug that would only appear when someone deleted the logging.

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

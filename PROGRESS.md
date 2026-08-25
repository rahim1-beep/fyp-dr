# PROGRESS.md

> **NEXT ACTION:** **The 384px ablation** (DECISION-044). Two Kaggle sessions: rebuild
> the cache at 384 (**~8-9 h**, ~2.4 GB) then train arm E at 384 (~2 h). The notebook is
> not written yet.
>
> **A 384 result cannot become the headline** - the proposal fixes 224. This answers
> "was resolution the binding constraint?" and nothing more without a supervisor
> deviation. Stopping rule is against the 224 three-seed range 0.7211-0.7464.
>
> After it: Phase 5 Grad-CAM, Phase 6 APTOS, Phase 7 web app, Phase 8 write-up - none of
> which needs meaningful GPU.

**Current phase:** Phase 4 — Ablation arms A–F. **Phases 1–3 complete.**
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
repo, **ask before first push** · DECISION-001 **approved by supervisor 2026-08-24**; was pending before
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
- [x] **Kaggle build run 2026-08-21** — `RECONCILIATION PASSED`, leakage tests green
- [x] Cached all 35,126 EyePACS + 3,662 APTOS images — the build was **correct**
- [x] `src/data/archive_cache.py` + `tests/test_archive.py` (24) — pack to one file,
      verify the central directory, refuse a short source, re-count on extract, and find
      the cache root by shape (DECISION-021, DECISION-023)
- [x] `src/data/notebook_check.py` + `tests/test_notebook_cells.py` (30) — every notebook
      command line validated against its module's real parser (DECISION-022)
- [x] **PUBLISHED 2026-08-21** — `rah098/fyp-dr-eyepacs-224` v1, private, 918.08 MB
- [x] `src/data/notebook_check.py` + `tests/test_notebook_cells.py` (30) — every
      notebook command line validated against its module's real parser, every direct call
      bound against the real signature, and the pack path executed on a toy tree
      (DECISION-022)
- [x] Third build: `ARCHIVE VERIFIED` — 38,797 entries (38,788 images + 9 sidecars),
      0.852 GB; reconciliation clean; same 4 `ok:no-retina` images, all still in place

### What it took: three attempts, and none of the losses were in the work

| Attempt | Outcome |
|---|---|
| 1 | Built correctly. Output truncated to **499 of 38,788** by Kaggle's 500-file cap, silently, after every check passed. `/kaggle/working` wiped (DECISION-021). |
| 2 | Built correctly, ran **8.1 h**, printed `READY TO ARCHIVE`, died in cell 5 on a missing `pack` subcommand — rejected by argparse in 0.0 s (DECISION-022). |
| 3 | **Built, verified, published.** 38,797 entries, 0.852 GB, `rah098/fyp-dr-eyepacs-224` v1. |

Then Kaggle **auto-extracted** the archive on publish, so the mounted dataset is a folder
tree rather than the zip the training notebook expected (DECISION-023).

Every one of these was in the glue around the work: output packaging, an argument list,
a directory layout. The preprocessing itself was right on the first attempt and produced
byte-identical results on all three.

### The publishing failure — worth reading before the rebuild

Everything in Phase 2 verified the artefact **inside the session**, and the thing that
broke was the artefact **that left it**. Reconciliation passed against a live tree of
38,788 files; the output cap then kept 499, after every check had run, and reported
nothing. A verification that does not run on the bytes that survive is a verification of
something else.

The fix is ordering, not cleverness: **pack → verify the archive → delete the source**,
all inside the session, with the pack step refusing a short source tree before it writes
anything. See DECISION-021, which also records why the Kaggle API route was considered
and not adopted.

### MEASURED — the real cache, 2026-08-21

| | |
|---|---|
| Files | **38,788** |
| Total | **0.836 GB** (a 42× reduction from 35.34 GB) |
| Mean | **22.6 KB/image** (min 2.7, max 33.6) |
| EyePACS | 35,126 files, 0.758 GB |
| APTOS | 3,662 files, 0.078 GB |
| Decode check | 800 sampled, **0 bad** |
| Published | **NO — output truncated to 499 files (DECISION-021)** |
| Patients | 21,225, **no overlap across splits** |
| `status != 'ok'` | **4**, all `ok:no-retina` (DECISION-018) |

The 21.4 KB/image projection from the QA sample was 5% low against the real 22.6 —
the QA sample happened to include three unusually small degraded files.

Two things the build surfaced, both now fixed rather than remembered:
**the Kaggle image's library versions did not match the pins** (DECISION-019), and
**Kaggle moved input mounts** to `/kaggle/input/datasets/{owner}/{slug}/` with
`/kaggle/input` read-only (DECISION-020, `src/data/kaggle_paths.py`).
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

## Phase 4 — Ablation arms A–F `[~]`

**Acceptance:** six arms compared on validation QWK under one identical budget, with a
seed repeat on the contenders. The test set stays closed (R3).

**Plan approved 2026-08-22.** 30 epochs, patience 7, seed 42, identical across arms;
order A→B→C→D→E→F by increasing implementation risk.

| Stage | Runs | Budget |
|---|---|---|
| 1 · six arms, ResNet18, seed 42 | 6 | **3.7 h** (measured: 70.6 s/epoch) |
| 2 · top 3 arms × seeds 43, 44 | 6 | **3.7 h** |
| 3 · winner + arm A on EfficientNet-B0 | 2 | measured by cell 4's timing probe |
| | **14** | **≈ 10 h of 30 h/week** |

- [x] `notebooks/phase4_ablation.py` — five cells, arms isolated per subprocess
- [x] **Stage 1 run 2026-08-22 — 5 of 6 arms.** Arm C died before epoch 1: `fit()` moved
      the model to the GPU but not the criterion, and arm C is the only arm whose loss
      carries class weights. Fixed and tested; provably inert for the five arms that ran
- [x] DECISION-029 — the collapse rule's accuracy test flagged 4 of 5 arms, including the
      one with the best QWK, because balancing puts accuracy below the majority rate **by
      design**. Now requires low QWK as well
- [x] DECISION-030 — QWK stays the ablation's selection metric; a referable-sensitivity
      floor gates the deployed model. Arm E's cut points are untuned, so its sensitivity
      is not yet a property of the arm
- [x] DECISION-031 — pooling APTOS is a **null result** against the correct comparator
      (F vs B, not F vs A): −0.022 QWK, inside B's interval
- [ ] Re-run arm C alone (~36 min), then regenerate `docs/EXPERIMENTS.md` from artefacts
- [x] `src/eval/thresholds.py` + `src/eval/predict.py` + 22 tests — QWK-optimal cut
      points AND an operating point against the screening standard, reported separately
      so the trade-off stays visible (DECISION-030, DECISION-032)
- [x] `train.py` now saves `val_outputs.npz` for every run, so threshold work is a local
      GPU-free step forever
- [x] `notebooks/phase4_armc_backfill.py` — arm C's re-run + the five-checkpoint
      backfill in one session, with a reproducibility check on every backfilled run
- [ ] **← RUN IT** (needs the ablation notebook's OUTPUT mounted, for the checkpoints)
- [ ] Operating-point table for all six arms — **this decides whether arm E's ranking
      survives**, and it comes before stage 2
- [x] Matched-decision-rule comparison (DECISION-033/035): ranking becomes
      A > E > F > D > B > C, and **A vs E is a tie** (+0.007 [-0.020, +0.029])
- [x] `configs/arm_c2.yaml` — effective-number weighting, 17.1:1 against inverse
      frequency's 36:1, so the arm C result stops resting on one scheme (DECISION-034)
- [x] `notebooks/phase4_stage3.py` — the capacity hypothesis as a falsifiable test,
      **pre-registered in EXPERIMENTS.md before the run**, plus arm C2
- [x] **Stage 3 run** — prediction half wrong; it tested backbone quality, not
      capacity, because B0 is 4.01M params against ResNet18's 11.18M (DECISION-036)
- [x] Arm C2 — the arm C conclusion survives both the weighting scheme and the matched
      decision rule; ~78% of C2's advantage is boundary displacement (DECISION-037)
- [x] **Stage 3.5 done** — B2 vs B0 = +0.0052 [-0.0195, +0.0285], NOT separable;
      **prediction held**, backbone fixed at **EfficientNet-B0** (DECISION-039). B2 is
      also worse on the deployment metric: sens@spec.95 0.7211 vs 0.7464.
- [x] `compare_arms` label fix — it was silently dropping 4 of 11 runs (DECISION-040)
- [x] **Stage 2 done** (0b13e9e) — six labels, no collapse. E 0.7563 +/- 0.0012 vs A
      0.7381 +/- 0.0034 held-out; E ahead on every seed on both metrics.
- [x] **DECISION-042** — arm E's 0.034 gap was a seed artefact; the real figure is 0.077
      over three seeds. It had been load-bearing in DECISION-036 and -039; both are
      annotated and their affected claims struck.
- [ ] **← NEXT: the 384px ablation** (DECISION-044) — cache rebuild then one run
- [ ] Phase 5 Grad-CAM
- [ ] Phase 6 APTOS external validation
- [ ] Phase 7 FastAPI + Next.js
- [ ] Phase 8 write-up

**Selection optimism: 3 architecture choices made on validation, soft budget 4.**
A fourth needs a positive argument written down BEFORE the run, logged as a decision —
not momentum from a good result (DECISION-036). The running total and what it costs are
in `docs/EXPERIMENTS.md`; cut-point optimism is measured, selection optimism is not.

**Why arm A is re-run:** the Phase 3 baseline was 8 epochs with the cosine annealed to
3.0e-06 by the end. It is the Phase 3 record and **not** a valid comparator for a
30-epoch arm.

**If an arm crashes** the loop records it and continues to the next; whatever epochs it
finished survive in its `train_log.csv`, and `_phase4_status.json` is rewritten after
every arm.

---

## Phase 3 — Baseline `[x]`

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
- [x] `leakage-auditor` review of the data layer — **FAIL, 4 blockers, all fixed**
      (DECISION-017, and a correction inside DECISION-007)
- [x] `src/data/reconcile_cache.py` gained the R1 check it was missing, plus a cache
      provenance sidecar, stats set-comparison, and a stratified decode sample
- [x] `tests/test_reconcile.py` — 14 tests. **Full suite: 123 green.**
- [x] `src/models/factory.py` — timm, full fine-tune, normalisation from `default_cfg`,
      all six arm configs build; `assert_fully_trainable` makes failure mode 2 loud
- [x] `src/eval/metrics.py` — QWK, balanced accuracy, referable sens/spec, bootstrap CIs,
      and `detect_collapse` implementing the §2 detection rule
- [x] **BGR/RGB settled end to end** — `tests/test_channel_order.py`, checked against PIL
      as an independent decoder and on a real EyePACS image
- [x] `tests/test_metrics.py` (30, cross-checked against sklearn) + `tests/test_models.py`
      (24) + `tests/test_channel_order.py` (7). **Full suite: 184 green.**
- [x] `src/train/losses.py` — CE / weighted CE / focal / ordinal; refuses class weights
      **and** sampler together, because both correct the same imbalance
- [x] `src/train/schedulers.py` — AdamW with no-decay on norms and biases, cosine with
      warmup **stepped per batch**, LR curve printed into the run log
- [x] `src/train/loop.py` — AMP, grad clip, early stopping on **val QWK**, best-checkpoint
      save and restore, per-epoch collapse check, log appended every epoch
- [x] `src/train/train.py` — the one entry point; runs the leakage gate, writes
      `config.yaml` **before** the first batch, and has no test argument at all (R3)
- [x] `tests/test_train.py` — 37 tests, the loop exercised end to end through the real
      `DRDataset`. **Full suite: 221 green.**
- [x] `notebooks/phase3_baseline.py` — smoke run then baseline, with an acceptance check
- [x] Arm A smoke run reached the training loop; failed in the balance PRINT, not the
      training path — `RandomSampler` has no `.weights` (DECISION-024). Fixed, plus
      end-to-end tests for the four config branches that no arm had ever run
- [x] GPU image versions recorded: same as CPU, `+cu128` builds (DECISION-019)
- [x] Second smoke failure: `yaml.safe_dump` refused `torch.__version__`, a `str`
      subclass. Whole run config is now coerced to primitives (DECISION-025)
- [x] `src/train/smoke.py` + `tests/test_train_smoke.py` (14) — the REAL `main()` end to
      end on a synthetic 90-image fixture, ~20 s/arm on CPU. It immediately found that
      arm F could not run at all for want of APTOS splits in the fixture
- [x] **Baseline run 2026-08-22** — arm A, ResNet18, 8 epochs, best at epoch 6.
      **val QWK 0.6138 [0.5859, 0.6421]**, acc 0.7965 (majority 0.7369), balanced acc
      0.4163, referable sens 0.5131 / spec 0.9804. Recorded in `docs/EXPERIMENTS.md`
- [x] **Phase 3 acceptance MET** — QWK CI far from zero, no fatal collapse. Grade 1 is
      never predicted; that is a WARNING, not a collapse, because grade 1 is below the
      referable threshold and changes no referral (DECISION-026)
- [x] `src/data/fetch_run.py` + `tests/test_fetch_run.py` (26) — pull a committed
      kernel's run artefacts, verify all three files and their agreement with each other,
      then install; never copies `*.pth`, never overwrites a committed run without
      `--force`
- [ ] Run it for `phase3_baseline_resnet18` (R4) — **blocked**: the stored Kaggle API key
      returns 401, so it needs a fresh token
- [ ] `src/eval/thresholds.py` — arm E's cut points, optimised on validation only (R3)

**What the audit found — worth reading before touching this layer:**

Three of the four blockers were demonstrated with a printed exit code, not argued.

1. **`reconcile_cache` printed `RECONCILIATION PASSED` and exited 0 on a patient with the
   left eye in train and the right eye in test.** Its R1 check compared *cache paths*, and
   two images of one patient have two different names, two different files, zero
   collisions and zero orphans. No count of files can see this. It now compares
   `patient_id` across split roles, first, before anything else.
2. **The sampler's val/test guard was a substring blocklist and failed open** on
   `holdout`, `development`, `tuning`, `screening_2026`, on a dropped `split` column, and
   on an all-NaN one. Now an allowlist: `{"train", "aptos_train"}`.
3. **`build_loaders` never checked R1 — and the suite's headline R2 fixture was a
   100-patient overlap, and green.** That was the proof the assertion was missing. The
   assertion is in, the fixtures are disjoint, and removing the assertion now fails a test.
4. **Arm F's origin/severity correlation is real** — P(aptos | grade) runs 0.065 at grade 0
   to 0.291 at grade 4 — **and DECISION-007 claimed it was absent.** Corrected there, with
   the measured table. The balanced sampler amplifies the payoff for using it: grade 4
   goes from 2.6% of the gradient to ~20%, 29% of it APTOS.

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

**Blocked on:** nothing — DECISION-001 was approved by the supervisor on 2026-08-24.
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
- [x] **Supervisor:** DECISION-001 (FastAPI + Next.js) **approved 2026-08-24**.

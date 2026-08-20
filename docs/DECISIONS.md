# Architecture Decision Record

ADR-style log of every non-obvious choice. Format: date, decision, alternatives
considered, rationale, who approved.

**Rule:** any deviation from `docs/proposal.pdf` is logged here *and* flagged to the user.
Any data rows excluded from the pipeline are listed here, never dropped silently.

---

## DECISION-001 — Streamlit replaced by FastAPI + Next.js

- **Date:** 2026-08-20
- **Status:** Accepted by user; **pending written supervisor confirmation before Phase 7**
- **Deviates from proposal:** Yes — §9 (Tools and Technologies) and §5.10/Month 8
  (Milestones) name Streamlit.

**Decision.** The web-based diagnostic interface is built as a proper client/server
application — FastAPI + Uvicorn backend, Next.js (TypeScript) + Tailwind frontend — rather
than a Streamlit script.

**Alternatives considered.**
1. *Streamlit as proposed.* Fastest to build; but produces a script, not assessable
   software engineering work, and offers no API surface.
2. *Gradio.* Same class of limitation as Streamlit.
3. *FastAPI + a server-rendered Jinja template.* Lighter than Next.js, but gives up the
   component model and the TypeScript type-safety story.

**Rationale.**
- Separation of concerns between inference and presentation.
- A documented REST API that a hospital system could in principle consume.
- Real request handling, error states, and file-upload validation (MIME type, magic bytes,
  size and dimension limits) rather than a notebook-grade upload widget.
- A UI that can be assessed as software engineering work.
- Auto-generated OpenAPI docs at `/docs` — a thesis figure in its own right.

**The objective is unchanged.** The proposal's stated objective is "a web-based diagnostic
interface"; only the implementation is upgraded. Scope §6.1 ("Web-based deployment") is
satisfied either way.

**Approval.** User approved 2026-08-20 and will obtain written confirmation from
Mr. Ubaid Ur Rahman before Phase 7 begins. Phases 1–6 do not depend on the frontend and
proceed regardless.

---

## DECISION-002 — Python 3.12 toolchain (local 3.12.10 / Kaggle 3.12.13)

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal does not pin a Python version)

**Decision.** The project targets **CPython 3.12**. The recorded version pair is:

| Environment | Version | ABI tag | Compute |
|---|---|---|---|
| Local (Windows 11) | **3.12.10** | `cp312` | CPU only |
| Kaggle Notebooks (Linux) | **3.12.13** | `cp312` | CUDA (T4×2 / P100) |

The virtualenv is built with `py -3.12 -m venv .venv`. `requirements.txt` pins versions
that resolve on `cp312` for **both** win_amd64 and manylinux.

**Rationale.**
- The machine's *system default* interpreter is 3.12's successor, 3.14, which Kaggle does
  not run and which parts of the CV/DL stack do not yet ship wheels for. Building against
  it would break R6 (reproducibility) on day one.
- Patch-level differences within 3.12 (10 vs 13) do not affect wheel compatibility — both
  build as `cp312` — so the local/Kaggle pair is genuinely interchangeable for
  dependency resolution.

**Constraint recorded.** The system default 3.14 interpreter is **not to be modified or
referenced in any config file.** All local execution goes through `.venv`.

---

## DECISION-003 — No local GPU acceleration; Kaggle-first is absolute

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal §7.2 already names Colab/Kaggle GPU)

**Decision.** The local machine is treated as a **code-editing, split-inspecting, and
web-app-demo machine only.** All preprocessing, training, evaluation, and Grad-CAM
generation run on Kaggle Notebooks.

**Measured environment.** Intel Arc Pro Graphics; no NVIDIA adapter, no CUDA runtime,
`nvidia-smi` absent. 31.4 GB system RAM, 797 GB free disk.

**Alternatives considered.**
1. *Intel XPU / IPEX backend for PyTorch.* **Explicitly rejected by the user.** Would add a
   second, divergent compute path to debug and maintain, with worse `timm` coverage, in
   exchange for acceleration that is still far below a T4.
2. *Download EyePACS locally and train on CPU.* Disk space is actually sufficient
   (797 GB free), so the original disk-based rationale does not apply — but CPU-only
   training on 35k images across 5 ablation arms is not viable on any timeline.

**Rationale.** With no CUDA device, the Kaggle-first mandate is compulsory rather than a
convenience. This is why every script must read its dataset and output roots from config
(`configs/kaggle.yaml` vs `configs/local.yaml`) and never hardcode a path.

---

## DECISION-004 — APTOS author-provided split discarded

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal does not specify an APTOS split)

**Decision.** The `mariaherrerot/aptos2019` redistribution ships its own `test.csv` (and
sibling split files). **These splits are ignored.** All ~3,662 labelled APTOS images are
pooled and used as a single held-out external test set.

**Rationale.** The partitioning rule behind that redistribution is undocumented. Adopting
it would silently import an unknown split into a project whose headline contribution is
partitioning rigour. Pooling is both simpler and honest.

**Related limitation (to document in the thesis, not hide).** APTOS `id_code` values are
anonymised hashes with no recoverable patient linkage. Each image is therefore treated as
its own patient (`patient_id = f"aptos_{id_code}"`). This means APTOS cannot support a
patient-level split at all — a further reason to use it only as an external test set.

---

## DECISION-005 — Image size fixed at 224×224 for headline results

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No — proposal §5.1/§5.3 specify 224×224.

**Decision.** All headline results use **224×224**. A config switch for 384×384 is built
but runs only as an optional ablation, and only if GPU quota survives Phase 4.

**Rationale.** Matching the approved proposal keeps the contract intact. 384 would roughly
triple training cost per run against a fixed weekly GPU quota that must first cover five
imbalance arms across two architectures.

---

## DECISION-006 — Rare-class test support is thin; report intervals, not point estimates

- **Date:** 2026-08-20
- **Status:** Accepted — **known limitation, to be stated in the thesis**
- **Deviates from proposal:** No

**The limitation.** Only 445 patients in all of EyePACS have max grade 4. A 15% patient-level
test split therefore yields, measured from the generated splits:

| Split | grade-3 images | grade-4 images | max-grade-3 patients | max-grade-4 patients |
|---|---:|---:|---:|---:|
| train | 607 | 502 | 349 | 312 |
| val | 133 | 108 | 75 | 67 |
| **test** | **133** | **98** | **75** | **66** |

**Consequence.** 98 grade-4 and 133 grade-3 test images are enough to populate a confusion
matrix and to compute a stable QWK (which is dominated by the bulk classes). They are
**not** enough for a tight interval on per-class recall. A bootstrap 95% CI on grade-4
recall over n=98 will be wide — on the order of ±10 percentage points — and a difference of
a few points between two arms on that metric is **not** evidence that one arm is better.

**Decision.** For grades 3 and 4, `eval-analyst` reports **bootstrap 95% confidence
intervals, never bare point estimates**, in every table, figure, and sentence. Any claim of
an improvement on rare-class recall must be justified by non-overlapping intervals, not by
a difference in point estimates. This is briefed in `.claude/agents/eval-analyst.md`.

**Alternatives considered and rejected.**
1. *A larger test fraction (e.g. 25%) to thicken the rare classes.* Rejected — it would
   deviate from the proposal's stated 70/15/15 and shrink training data for the classes
   that most need it.
2. *Stratified k-fold cross-validation.* Would give tighter intervals, but multiplies GPU
   cost by k across 6 arms × 2–3 architectures against a fixed weekly quota. Recorded as
   future work.

**This is a good thesis paragraph, not a weakness to hide.** Honest interval reporting on a
thin rare class is a stronger methodological result than a confident point estimate.

---

## DECISION-007 — APTOS pooled as Arm F; all-or-nothing, never severity-selective

- **Date:** 2026-08-20
- **Status:** Accepted
- **Origin:** Supervisor feedback suggesting APTOS could help with class imbalance
- **Deviates from proposal:** Extends it. Proposal §5.1 names EyePACS only; APTOS was
  already scoped in `BOOTSTRAP.md` §3.2 as external validation.

**The question.** Would pooling APTOS into training materially relieve the class imbalance?
Settled with measured numbers rather than argument.

**Measured APTOS distribution** (from the labels CSVs only — no images downloaded; all
three author split files pooled per DECISION-004, 3,662 rows, 0 duplicate `id_code`):

| Grade | Count | Percent | EyePACS percent |
|---|---:|---:|---:|
| 0 | 1,805 | **49.29%** | 73.48% |
| 1 | 370 | 10.10% | 6.96% |
| 2 | 999 | 27.28% | 15.07% |
| 3 | 193 | 5.27% | 2.48% |
| 4 | 295 | 8.06% | 2.02% |
| **Total** | **3,662** | | |

Imbalance ratio 9.35:1, versus EyePACS's 36.45:1.

**Finding.** The user's prior was correct almost exactly: APTOS is imbalanced **in the same
direction** as EyePACS — 49.29% grade 0, adding 193 grade-3 and 295 grade-4 images against
1,805 additional grade-0. It is *less* skewed than EyePACS, but it is not a rare-class
supplement.

**Effect on the Arm F training pool** (EyePACS train + APTOS train):

| Grade | EyePACS train | + APTOS train | = Arm F | share before → after |
|---|---:|---:|---:|---|
| 0 | 18,073 | 1,264 | 19,337 | 73.51% → 71.23% |
| 1 | 1,728 | 259 | 1,987 | 7.03% → 7.32% |
| 2 | 3,676 | 699 | 4,375 | 14.95% → 16.11% |
| 3 | 607 | 135 | 742 | 2.47% → **2.73%** |
| 4 | 502 | 206 | 708 | 2.04% → **2.61%** |
| **Total** | 24,586 | 2,563 | 27,149 | |

**So: pooling barely moves the proportions** — grade 4 goes 2.04% → 2.61% — but it does add
real absolute rare-class data: **+41% more grade-4 images and +22% more grade-3 images.**
Whether that extra absolute volume outweighs the added domain shift is an empirical
question, which is precisely why it becomes an arm rather than an argument.

**Arm F definition.**
- **Train:** `data/splits/train.csv` + `data/splits/aptos_train.csv`
- **Validate:** `data/splits/val.csv` + `data/splits/aptos_val.csv`
- **Test:** `data/splits/test.csv` (**EyePACS only**), so Arm F stays directly comparable
  to arms A–E on an identical test set.
- APTOS is split by the **same procedure and the same seed**, treated as one image per
  patient (no linkage exists — DECISION-004).

**Constraint: all-or-nothing pooling.** If APTOS is pooled, **all of it is pooled.** Adding
only grades 3 and 4 is explicitly forbidden.

*Reasoning:* APTOS and EyePACS were captured on different camera hardware under different
acquisition protocols, so dataset origin is visible in the image — colour balance, field of
view, illumination, compression. If only grades 3–4 were imported, **dataset origin would
correlate with severity**, and the model could reach a high score by learning "this looks
like an APTOS camera" as a proxy for "this is severe." That is a domain artefact
masquerading as a clinical finding: it would inflate rare-class recall on any test set
containing APTOS images and collapse on EyePACS-only data. Pooling everything keeps origin
uncorrelated with label, so any gain is attributable to the extra data rather than to a
detectable acquisition signature. `xai-engineer`'s Grad-CAM sanity check is the
second line of defence here.

**Consequence — the trade-off to defend in the write-up.** Under Arm F, APTOS is consumed
as **training** data, so it **cannot also serve as the Phase 6 external validation set**
for that arm. There is no version of this where APTOS is both.

- **Arms A–E:** Phase 6 is unchanged — trained on EyePACS only, externally validated on all
  of APTOS. This remains the stronger thesis result, because a held-out *external* test set
  measures generalisation across domains.
- **Arm F:** gains rare-class training volume, loses its external validation set. It can
  only be evaluated on the EyePACS test set.

The write-up must state this explicitly and defend whichever is chosen as the headline
model. If Arm F wins on validation QWK, the honest framing is that it trades a
generalisation claim for in-domain performance.

---

## DECISION-008 — Split generated and audited

- **Date:** 2026-08-20
- **Status:** Accepted
- **Seed:** 42 (`configs/base.yaml` → `split.seed`, and written into each split CSV header)

**Decision.** Patient-level 70/15/15 split, stratified on each patient's max grade,
generated by `src/data/split.py` and committed as `data/splits/*.csv`.

**Result.** 17,563 patients → 12,293 train / 2,634 val / 2,636 test. All 35,126 images
assigned exactly once. Val and test retain the natural distribution (max drift from natural
< 0.6 pp on any class).

**Verification.** `tests/test_no_leakage.py` — **38 tests, all passing.**

### `leakage-auditor` verdict: **PASS** — cleared to commit

Independently re-verified from scratch (100 assertions, 0 failures), *without* relying on
`tests/test_no_leakage.py`, which was run only as a cross-check.

- **0 patient overlaps** across all six split pairs (EyePACS ×3, APTOS ×3). 0 image-path
  overlaps. All 17,563 patients carry exactly 2 rows, both in the same split.
- **35,126/35,126** EyePACS images reconciled by set-equality against
  `trainLabels.csv` — 0 missing, 0 extra. **3,662/3,662** APTOS images likewise; the
  author-provided split is genuinely dissolved (source-file × our-split crosstab is fully
  mixed, not diagonal).
- **Stratification confirmed on patient max-grade.** The counter-hypotheses — min grade,
  left-eye grade, floor(mean grade) — were each tested and *ruled out* by exact per-stratum
  counts. All **2,240** asymmetric patients are co-located; maximum-spread cases
  spot-checked (e.g. patient `10321`, left=4/right=0, both in `val`).
- **Natural distribution retained.** Max drift from natural: train 0.00114, val 0.00212,
  test 0.00640 — all < 0.01. Majority share > 0.73 in every split.
- **Determinism.** Two fresh re-runs produced byte-identical bodies (SHA-256 of
  comment-stripped content); the only diff is the `# generated:` timestamp. Seed 43 changes
  46.8% of assignments, so the seed is load-bearing.
- **Code review of `split.py`.** Index coverage/overlap brute-forced for n = 0…3,000;
  rounding brute-forced for n = 0…200,000 across 11 fraction triples. No defect found.

### Actions taken on the auditor's non-blocking observations

Four were fixed before this commit rather than deferred:

1. **Small-strata hole (fixed).** `assign_splits` had no minimum-per-split guarantee — at
   70/15/15 any stratum with n < 4 patients silently produced an empty val *and* test,
   i.e. an all-zero confusion-matrix column. Harmless today (smallest real stratum is 445
   patients) but a filtered subset or debug sample would have hit it quietly. Now raises
   `SystemExit`; verified to fire on a synthetic 3-patient stratum.
2. **Venv was off-spec (fixed).** The splits were first generated under `numpy 2.5.2` /
   `pandas 3.0.5` while `requirements.txt` pins `2.2.6` / `2.3.3` — an R6 gap, and
   `pandas` 3.0 changes exactly the copy-on-write and string-dtype machinery the manifest
   builder relies on. The venv was corrected and **the splits regenerated**. Assignments
   are byte-identical across the change (auditor's spot-checked patients land in the same
   splits), so only the provenance header differs.
3. **Version provenance (fixed).** numpy's `Generator` stream is not contractually stable
   across versions (NEP 19). Each split CSV header now records
   `# python: … numpy: … pandas: …`.
4. **Test-suite blind spots (fixed).** The suite checked totals against a *hardcoded
   integer*, so a complete-but-mislabelled manifest would have passed. Added: set-equality
   and label reconciliation against `trainLabels.csv`; an independent re-parse of
   `patient_id`/`eye`; a stratification test that *rules out* min-grade rather than merely
   asserting max-grade; an asymmetric-patient co-location test; a subprocess re-run
   determinism test; and seed-header checks extended to the three `aptos_*.csv` files.
   29 tests → **38**.

Remaining observation, accepted without change: `patient_id` dtype inference. EyePACS ids
are bare digits and would infer as `int64` while APTOS ids are strings, so an Arm F concat
could produce a mixed-type column. Already handled — `src/data/manifest.py` forces
`dtype={"patient_id": "string"}` alongside `comment='#'`, and it is the only supported
reader.

**Note for all readers of the split CSVs.** They carry a `#` provenance header recording the
seed, generator, git SHA, fractions, and class distribution. **Every reader must use
`pandas.read_csv(..., comment='#')`** or the header will be parsed as data.

---

## DECISION-009 — Ben Graham's local average is a normalised convolution over the retina mask

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No — the proposal specifies Ben Graham enhancement; this
  fixes *how* it is computed at the retina boundary.

**The defect.** Ben Graham's method is `alpha*img + beta*blur(img) + gamma` with
`alpha=4, beta=-4, gamma=128`. Computed naively, the Gaussian window near the edge of the
retina straddles the black surround, so `blur` is dragged toward zero while `img` is not.
`4*img - 4*blur` then blows up, ringing every image in a **white halo brighter than any
lesion**. Measured on the QA sample: mean brightness in the 0.85–0.99r annulus was
**1.44×** the inner-disc mean, and on grade-4 examples the halo washed out lesions that
are plainly visible once it is removed.

**Why it matters beyond appearance.** The halo is a constant artefact at a constant
location. A CNN can key on it, and Grad-CAM will light it up — which would corrupt the
Phase 5 sanity gate ("heatmaps on lesions, not borders").

**The fix.** The local average is computed as a normalised convolution over the retina
mask, `blur(img*mask) / blur(mask)`, so it averages only over pixels that were actually
imaged. Variants compared visually on 5 images (ideal-circle mask + plain blur, real mask
+ plain blur, real mask + normalised convolution); the last was clearly best and is what
`src/data/preprocess.py:ben_graham` implements.

Locked in by `tests/test_preprocess.py::test_ben_graham_does_not_ring_the_retina_edge`,
which fails if the edge/inner brightness ratio returns above 1.15.

---

## DECISION-010 — Square crop centred on the retina, not bounding-box-then-pad

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No — implementation detail of "circle crop".

EyePACS retinas are routinely **truncated top and bottom** by the sensor, so the
non-black bounding box is wider than it is tall. Padding that box to a square leaves
black bars inside the frame, and the retina then fills a fraction of the output that
depends on the **camera's aspect ratio rather than on the eye** — so the apparent scale
of a lesion varies by camera model, which is a spurious cue correlated with acquisition
site.

`square_crop` instead takes a square of side 2r centred on the retina, where r is its
larger half-extent, so the retina occupies a consistent fraction of every 224×224 output.
The mask is carried through the crop so image and mask cannot drift out of registration.

---

## DECISION-011 — Large-sigma Gaussian computed on a downscaled copy

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No — a performance optimisation with a measured error bound.

**The problem.** Ben Graham's sigma is `width/10`, so a 2560px-wide image needs sigma 256
and `cv2.GaussianBlur` builds a ~1537-tap separable kernel. **Measured: 30.8 s for one
image**, ~35 s/image over the QA sample, which projects to roughly **340 hours** for
35,126 images. The cache would never have been built.

**The fix.** A Gaussian that wide is pure low frequency, so it is computed on a copy
downscaled until sigma is 48px and then scaled back. Because sigma is itself proportional
to width, the working image is always ~480px wide regardless of source size.

**Measured error** on a 1920×2560 image, after the full `4*img - 4*blur + 128` (which
amplifies any blur error fourfold): **max 4/255, mean 0.29** — below JPEG q95 quantisation
noise, and smooth by construction, so it cannot add or remove lesion-scale detail.
Speed-up 145×. Re-measured by `tests/test_preprocess.py::test_large_sigma_blur_approximates_the_exact_blur`
rather than trusted from this note.

---

## DECISION-012 — Preprocessed cache is flat and split-agnostic

- **Date:** 2026-08-20
- **Status:** Accepted — **required by `leakage-auditor` before the cache build**
- **Deviates from proposal:** No

The cache is written as:

```
{processed_root}/eyepacs/{patientID}_{eye}.jpg
{processed_root}/aptos/aptos_{id_code}.jpg
```

**Deliberately NOT** `train/`, `val/`, `test/` subdirectories. Split membership is
resolved only from `data/splits/*.csv` at `Dataset` construction time. A split-shaped
cache would silently mismatch the CSVs after any re-split, and no existing test would
catch it.

The APTOS source path `train_images/train_images/{id}.png` is flattened for the same
reason: that directory structure **is** the author-provided split which DECISION-004
discards, and preserving it would smuggle a discarded partition back into the project.
The `aptos_` prefix prevents hash-named APTOS files from colliding with EyePACS
`{patientID}_{eye}` names in a flat namespace.

Enforced by `tests/test_preprocess.py::test_cache_layout_is_flat_and_carries_no_split_name`
and `::test_aptos_cache_path_discards_the_author_split_and_is_namespaced`.

---

## DECISION-013 — Quality flags are advisory, with fixed absolute thresholds

- **Date:** 2026-08-20
- **Status:** Accepted — **required by `leakage-auditor` before the cache build**
- **Deviates from proposal:** No

`scan_quality()` computes per-image pixel statistics (brightness, retina fraction,
clipping, darkness, centroid offset, saturation, focus, contrast) and `flags()` labels
suspect images.

Two constraints, both binding:

1. **Thresholds are fixed absolute constants, never percentiles over the dataset.** A
   percentile cutoff computed across all 35,126 images would let val and test influence
   which train images are flagged. They were calibrated against the 20-image QA sample,
   which is **drawn from the train split only**.
2. **Nothing is dropped.** Flags are advisory and exist to direct human attention. No
   image is excluded from any split without an entry in this file — see
   "Excluded data rows" below, which remains **None**.

**Focus had to be made resolution-independent.** A Laplacian is a per-pixel operator, so
on the same scene a higher-resolution capture scores *lower*. Measured natively across
the QA sample's 0.1–16 MP range, a 4.9 MP image scored 152.0 and a 16.1 MP image 1.9, and
a single absolute threshold flagged **19 of 20 images as blurred**, including obviously
sharp ones. The measure now resamples the retina core to a fixed 512px width first.
After that fix, `laplacian_var` separated cleanly: **0.5 / 1.6 / 1.9** for the three
known-degraded images against **4.3** for the lowest good one, so the threshold is 3.0.

That the pixel statistics independently rank the same three images last as *file size*
did — two criteria sharing no inputs — is a genuine cross-check on both.

---

## Excluded data rows

**None.** The Phase 1 reconciliation found the EyePACS dataset completely clean: 35,126 CSV
rows, 35,126 image files, zero mismatches in either direction, zero duplicates, zero
unparseable filenames, zero zero-byte files. APTOS likewise: 3,662 rows across the three
author CSVs, 3,662 image stems on disk, zero duplicate `id_code`.

No row has been dropped from any manifest. If that ever changes, the excluded rows are
listed here with identifiers and reasons.

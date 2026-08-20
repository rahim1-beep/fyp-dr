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
containing APTOS images and collapse on EyePACS-only data.

> **CORRECTION, 2026-08-20 (leakage audit).** This decision originally said "pooling
> everything keeps origin uncorrelated with label." **That is wrong, and the table above
> refutes it.** APTOS is 8.06% grade 4 against EyePACS's 2.02%, which *guarantees* a
> correlation. Measured on the committed arm-F pooled training split:
>
> | grade | n | of which APTOS | P(aptos \| grade) |
> |---|---:|---:|---:|
> | 0 | 19,337 | 1,264 | 0.065 |
> | 1 | 1,987 | 259 | 0.130 |
> | 2 | 4,375 | 699 | 0.160 |
> | 3 | 742 | 135 | 0.182 |
> | 4 | 708 | 206 | **0.291** |
> | marginal | 27,149 | 2,563 | 0.094 |
>
> Monotonic, a 4.4x spread. All-or-nothing pooling **bounds** the correlation — importing
> only grades 3-4 would put it near 1.0 — but it does not remove it.
>
> **The balanced sampler amplifies the payoff.** One real balanced epoch (seed 7) lifts
> grade 4 from 2.6% of the gradient to about 20%, and 29% of those draws are APTOS. So
> "APTOS camera implies severe" is a shortcut worth several times more under arm F WITH
> the sampler than under arm F alone. The sampler is behaving exactly as §2.1 specifies;
> the confound is in the arm, not the mechanism.
>
> **Consequences, all binding:**
> 1. `describe_balance` emits a `p_aptos` column for pooled frames, so every arm-F run log
>    shows the correlation rather than leaving it to be rediscovered.
> 2. Arm F evaluation **must** report metrics broken down by source dataset. That is what
>    `DRDataset`'s returned index is for.
> 3. **Cross-arm selection uses arm F's val QWK restricted to the EyePACS val rows.**
>    Arm F's pooled val is a different population — n=5,818 with 9.5% APTOS, against
>    n=5,268 for arms A-E — and QWK is distribution-sensitive, so the pooled number is not
>    comparable to A-E's. Log both; select on the EyePACS-only one.
> 4. If arm F wins, the write-up must show the gain survives the per-origin breakdown.

`xai-engineer`'s Grad-CAM sanity check is the second line of defence here.

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

## DECISION-014 — The retina mask keeps only the largest connected component

- **Date:** 2026-08-20
- **Status:** Accepted — raised by `code-reviewer`
- **Deviates from proposal:** No

The mask started as a plain brightness threshold, so any bright non-retinal object joined
the retina in the region whose extent `square_crop` centres on. Measured on a synthetic
fundus with a small specular blob in one corner, the crop side went **606 → 902 px** and
the retina fraction **0.788 → 0.364**: the retina rendered at roughly **0.67×** its proper
scale, shrinking every lesion by a third of its pixels. `status` stayed `"ok"` and the
output looked plausible — a scale error is invisible when there is nothing beside it at
the correct scale.

Effective lesion size in pixels is the single feature this pipeline exists to preserve, and
an inconsistent one across images is a scale nuisance variable the model would have to
learn around.

`retina_mask` now keeps only the largest connected component. The discarded area fraction
is recorded per image as `off_disc_bright_fraction` and raises a `bright-artefact` flag
above 0.15, so a *genuinely* split retina (a lens shadow cutting the disc in two) is
reported rather than silently halved. On the 20-image QA sample exactly one image trips
it — `3829_left`, which is the near-black one that has almost no retina to find.

`tests/test_preprocess_plumbing.py::test_bright_artefact_does_not_shrink_the_retina` locks
the crop side to within 4 px of the clean image's.

---

## DECISION-015 — The contact sheet renders the real cache file, at a matched retina scale

- **Date:** 2026-08-20
- **Status:** Accepted — raised by `code-reviewer`
- **Deviates from proposal:** No

The sheet is the artefact the user approves the whole 35,126-image cache from, so it must
show what will actually be on disk. The first version did neither of those things:

1. **The "before" panel was downsampled below the "after" panel.** The original was fitted
   into the panel box while the processed image was a native 224 px; measured across the
   sample the retina occupied **139–203 px** against 224, so every pair was biased
   **0.62×–0.91×** in preprocessing's favour. Some of the apparent gain in sharpness was
   just the left panel being smaller. The original is now square-cropped from native pixels
   and rendered at the same retina scale.
2. **The "after" panel was an in-memory recomputation, not the cached JPEG.** Measured, the
   q95 round-trip costs mean **3.60/255** and max **64/255**, and lifts **39.3%** of the
   masked surround off exact zero. The sheet approved a file that was never written.
   `--cache-root` is now required and the panel is the decoded cache file.

A third panel was added: a 3× detail crop, original above processed, so the fine structure
the enhancement is meant to reveal can be judged rather than inferred from a 224 px
thumbnail.

---

## DECISION-016 — The retina mask is eroded 2.5% before the final re-mask

- **Date:** 2026-08-20
- **Status:** Accepted — user decision at the Phase 2 sign-off gate
- **Deviates from proposal:** No

### The artefact

Ben Graham's enhancement subtracts a local mean at sigma = width/10. The retina's own
optical vignetting falls off far faster than that, so the enhancement amplifies the
boundary falloff into a bright rim. Measured over the 20 cached QA images, mean
|pixel − 128| in the outer annulus (0.90–1.00 r) against the interior (< 0.85 r) was
**1.61× on average and up to 2.83×** (`1776_left`).

This is **not** the halo DECISION-009 fixed. `code-reviewer` confirmed the normalised
convolution matches the exact masked form to within 1/255. The rim is real optical
structure in the source, amplified by design.

### The three options

| Option | Rim | Area cost | Verdict |
|---|---|---|---|
| **Leave it** | 1.61× | 0 | A high-contrast ring at a fixed radius in all 38,788 images, uncorrelated with grade. A CNN can learn to ignore it, but it is the strongest edge in the frame and spends capacity. |
| **Ben Graham's 0.9 r mask** | 1.32× | **19% of retinal area** | Rejected. Grade 4 is already the thinnest class — 98 test images (DECISION-006) — and proliferative disease appears in the periphery, which is exactly what a fixed 0.9 r cut discards. |
| **Erode 2.5% of the equivalent radius** | see below | **3.2%** | **Chosen.** |

**Rationale (user, 2026-08-20):** full 0.9 r masking costs 19% of retinal area including
the periphery where proliferative disease appears, and grade 4 is already the thinnest
class at 98 test images. Erosion removes the vignetted pixels themselves rather than a
fixed fraction, so the cost tracks the artefact.

### Residual — the number the user asked to be recorded, and it is not the flattering one

**A fixed 0.90–1.00 r annulus barely moves: 1.610 → 1.516 at 2.5% erosion.** That metric
is degenerate under erosion, because trimming the boundary also moves which pixels fall
into a fixed radial band — the rim follows the boundary inward.

Measured properly, as a profile of mean |pixel − 128| against **depth from the actual
retina edge** in units of the equivalent radius, over the same 20 images:

| depth from edge | mean \|dev\| | vs interior |
|---|---|---|
| 0.000–0.010 | 76.10 | **2.61×** |
| 0.010–0.025 | 54.29 | 1.86× |
| 0.025–0.050 | 46.36 | 1.59× |
| 0.050–0.075 | 39.63 | 1.36× |
| 0.075–0.100 | 33.08 | 1.14× |
| 0.100–0.150 | 26.53 | 0.91× |
| > 0.150 | 22.9–29.1 | ≈ 1.00× |

So the excess is steep and shallow: it decays to the interior level by roughly **10% of
the radius**. Erosion at 2.5% removes the two worst bands — the outermost pixels drop from
**2.61× to ≈1.59×**, about **63% of the peak excess** — but leaves a ≈1.6× edge at the new
boundary. It suppresses the peak; it does not eliminate the rim. Eliminating it entirely
needs the ~10% cut that is Ben Graham's 0.9 r, at the area cost that was rejected.

**This was reported to the user before the cache build**, as they asked. 2.5% stands.
The knee of the profile is at 0.05 (1.36×, ~9.5% area) if the trade is ever reopened.
**It will not be reopened on aesthetics** — only on evidence, and the specified evidence
is the Phase 5 Grad-CAM sanity check showing the model keying on the rim.

### Implementation

`erode_mask(mask, frac)` uses a **distance transform**, not `cv2.erode` with a fixed
kernel: EyePACS retinas are routinely truncated top and bottom, and a distance transform
peels a constant depth off whatever shape is actually present. It also scales with the
image, so a 400px source and a 4928px source lose the same *fraction*.

Two properties are test-locked:

- **The erosion applies only to the final re-mask.** The normalised convolution still
  averages over the full mask, so every retained pixel is bit-identical to what it would
  have been with no erosion (`test_erosion_only_trims_and_never_alters_a_retained_pixel`).
  Erosion removes; it never alters.
- `scan_quality` still measures the **un-eroded** mask, so DECISION-013's fixed absolute
  thresholds keep the calibration they were set with.

One accepted cost: on a retina truncated by the sensor, the erosion also trims ~2.5% along
the straight cut edge, where the pixels are valid retina rather than vignetting. At 224px
that is a ~3px strip, and special-casing it would mean detecting which boundary segments
are optical and which are the sensor — complexity out of proportion to the loss.

`configs/base.yaml` → `preprocess.mask_erode_frac: 0.025`.

---

## DECISION-017 — Partitioning invariants are enforced at every layer, not documented at one

- **Date:** 2026-08-20
- **Status:** Accepted — `leakage-auditor` returned **FAIL / CHANGES REQUIRED** on the
  Phase 3 data layer
- **Deviates from proposal:** No

The audit found four blockers, three of them demonstrated by execution rather than
inferred. All are fixed and each has a regression test.

**1. `reconcile_cache` certified a leaking partition with exit code 0.** Its R1 check
compared *cache paths* across splits. A patient with the left eye in train and the right
eye in test produces two different filenames, two different cache files, zero collisions
and zero orphans — so the check passed, and the tool the auditor had mandated as the
pre-training gate blessed the textbook R1 violation. **No count of files can see this**;
patient identity has to be compared directly. There is now a check on `patient_id`, run
first, comparing split *roles* so that arm F's `train` + `aptos_train` are correctly
treated as one side of the boundary.

**2. The sampler's val/test guard was a substring blocklist, and it failed open.**
Measured: `holdout`, `development`, `tuning`, `screening_2026` were all accepted; a frame
with the `split` column dropped skipped the check entirely; an all-NaN column emptied the
set via `dropna()` and also skipped it. The last two were live, not hypothetical, because
`DRDataset` requires only `{image_path, label, dataset}` — a frame that was a valid
Dataset input was a frame with no guard. Replaced by an **allowlist**,
`TRAIN_SPLIT_NAMES = {"train", "aptos_train"}`, with a missing column and a NaN value both
counted as violations. A blocklist fails open on every name nobody thought of.

**3. `build_loaders` never checked R1, and the test suite's headline fixture violated it.**
`load_arm_splits` checks patient disjointness, but every path that calls `load_split`
directly bypasses it. `build_loaders` is the one function that sees all three frames at
once. It now asserts the three pairwise `patient_id` intersections. The audit also found
that `tests/test_dataset.py` **constructed a 100-patient overlap as its R2 fixture and was
green** — which is itself the proof the assertion was missing. The fixtures now use
disjoint patients, and `test_loaders_refuse_a_partition_that_shares_patients` locks it.

**4. Arm F's origin/severity correlation is real and was documented as absent.** See the
correction inside DECISION-007.

Also fixed, from the same audit:

- **R3:** `build_loaders(include_test=False)` by default. `load_arm_splits` returns three
  frames, so the natural call passed three and a live test loader existed for the whole of
  training. Touching the test set should be an act, not an inheritance.
- **The returned index was positional but sold as a join key.** Against a caller frame
  with a shuffled index, `.loc[idx]` returned a different row than the model saw
  (measured: `ds[5]` reported `5_left.jpeg`, the row actually loaded was `142_left.jpeg`).
  `DRDataset` now refuses a non-RangeIndex frame; the documented join is
  `loader.dataset.df.iloc[idx]`.
- **Physical oversampling was accepted in silence.** `pd.concat([train] + [rare] * 8)`
  produced 256 duplicated rows inside a Dataset with no complaint, while the module
  docstring claimed the forbidden operations were "not expressible." §2.1 forbids
  duplicated rows *in any manifest*; `tests/test_no_leakage.py` enforced that on the
  committed CSVs only. `DRDataset` now asserts `image_path` uniqueness, and the
  overclaiming docstring is gone — a safety claim that is not true tells the next reviewer
  not to check.
- **NaN labels** survived construction and failed at the first draw of the first epoch, on
  Kaggle. Refused at construction.
- **Half-specified normalisation** (`mean` without `std`) silently reverted *both* to
  ImageNet, discarding the caller's explicit value. Now raises.
- **Cache provenance.** `reconcile_cache` was content-blind: only `image_size` was
  verified, so a cache half-built before DECISION-016 and half after — two different
  `mask_erode_frac` values, two image domains in one directory — reconciled perfectly
  clean. `preprocess.py` now appends a `_cache_provenance.json` sidecar (full config, git
  SHA, source root, count) per invocation, and reconciliation compares every field.
- The stats CSVs are now **set-compared** against the split rows rather than read only for
  a status column, the split headers are checked to share a seed (DECISION-008), and the
  decode sample is **stratified by dataset** — unstratified, APTOS got ~9% of the checks
  purely for being ~9% of the cache, and APTOS is the half whose source format differs.

**The principle.** Every one of these passed review at the layer where it was written and
failed at the layer where it was used. Invariants get asserted at the point of use, by
code, with a test that fails when the assertion is removed.

---

## DECISION-018 — Four images have no detectable retina; all four stay in their splits

- **Date:** 2026-08-21
- **Status:** Accepted
- **Deviates from proposal:** No

The full cache build finished with **38,784 of 38,788 images at `status: ok`** and four at
`ok:no-retina`. `retina_mask` found no illuminated region in these frames, so
`preprocess_image` took its documented fallback: the image is resized to 224x224 **without
the square crop, without Ben Graham, and without the surround mask**.

| image | split | patient | eye | label |
|---|---|---|---|---:|
| `32253_right` | **train** | 32253 | right | 1 |
| `34689_left` | **train** | 34689 | left | 0 |
| `43457_left` | **train** | 43457 | left | 1 |
| `1986_left` | **val** | 1986 | left | 0 |

All four patients have both eyes in the same split, so R1 is untouched. The other eye of
each patient processed normally.

**They stay in their splits.** Dropping a row is never a neutral act here: removing
`1986_left` would silently rebalance the validation split, which is an R2 violation by
omission, and it would make the val set a slightly different population from the one every
other arm is measured on. The "Excluded data rows" section below therefore still reads
**None**.

**What this costs.** Those four cache entries are in a different image domain from the
other 38,784 — unenhanced, uncropped, aspect-distorted. Three are in train, where they are
four images out of 24,586 and will be treated as noise. The one that matters is
**`1986_left` in val**: it is a validation image the model will almost certainly get wrong,
and it is 1 of 5,268, so it can move validation accuracy by at most 0.019 percentage
points. That is far below the noise floor of the QWK bootstrap and changes no decision.

**Do not "fix" these by re-running with a lower mask threshold.** The threshold is a fixed
absolute constant by the leakage auditor's binding condition (DECISION-013); tuning it
until four specific images pass is fitting a preprocessing parameter to individual images,
and those images are in the data the model is selected on.

They are worth looking at again in Phase 5: if the Grad-CAM panel includes one, it will
show the model attending to something that is not a retina, which is a useful negative
example for the write-up rather than a defect.

---

## DECISION-019 — Pins follow the Kaggle image, and the image is the environment of record

- **Date:** 2026-08-21
- **Status:** Accepted
- **Deviates from proposal:** No

The first real Kaggle run measured the CPU image and it does not match what
`requirements.txt` pinned:

| package | was pinned | Kaggle CPU image (2026-08-21) | now pinned |
|---|---|---|---|
| numpy | 2.2.6 | **2.0.2** | 2.0.2 |
| opencv | 4.12.0.88 | **4.13.0** | 4.13.0.92 |
| torch | 2.9.1 | **2.10.0+cpu** | 2.10.0 |
| torchvision | 0.24.1 | **0.25.0** | 0.25.0 |
| timm | 1.0.28 | **1.0.26** | 1.0.26 |

**The Kaggle image wins.** Never `pip install` torch over the preinstalled build — it is
built against that image's CUDA and replacing it is the fastest way to lose a session to
a broken driver stack. So the pins move to match the image, and the local machine installs
the same versions, which is what makes a locally-inspected result and a Kaggle result the
same artefact (R6).

The cache was built under the image's versions, not under the old pins. That is fine and
does not require a rebuild: the preprocessing output is deterministic per image and the
provenance sidecar records the config, which is the thing that governs the pixels.

**The GPU image may differ from the CPU image**, and Phase 4 runs on GPU. Cell 1 of every
notebook prints the versions it actually has; if the GPU image differs, this table gains a
column rather than the pins being changed again. What must not happen is a pin being
edited from memory instead of from a printed version report.

`timm` 1.0.26 vs 1.0.28 is the one worth watching: `normalisation()` reads
`default_cfg` at runtime rather than hardcoding constants, so a change in timm's metadata
is picked up automatically instead of silently disagreeing with a literal in a config.

---

## DECISION-020 — Kaggle input mounts are resolved, not assumed

- **Date:** 2026-08-21
- **Status:** Accepted
- **Deviates from proposal:** No

Kaggle changed where attached datasets appear. Measured on the Phase 2 build:

    was:  /kaggle/input/{slug}/
    now:  /kaggle/input/datasets/{owner}/{slug}/

and `/kaggle/input` is read-only, so nothing can be symlinked *into* it — the workaround
during the build was to symlink into `/kaggle/working/inputs/`.

Hardcoding either layout is how the next notebook breaks. `src/data/kaggle_paths.py`
resolves a dataset by trying the known layouts in order and raising with the actual
directory listing when none matches, so a third layout costs one line here instead of a
lost session. `configs/kaggle.yaml` records the current canonical paths, and the resolver
is what code actually calls.

This is exactly the class of thing CLAUDE.md S3 means by "every script is path-agnostic":
the dataset root belongs in config and, where the platform can move it, behind a resolver.

---

## DECISION-021 — The cache is published as ONE verified archive, never a loose tree

- **Date:** 2026-08-21
- **Status:** Accepted — forced by a real failure
- **Deviates from proposal:** No

### What happened

The Phase 2 build ran correctly. All 38,788 images were written, reconciliation passed,
the leakage tests were green, and the measured size was 0.836 GB. **Kaggle caps a
notebook's saved OUTPUT at 500 files**, so the publish step kept **499 of 38,788 images**
— 1.3% — and said nothing. `/kaggle/working` is wiped between sessions, so the other
38,289 were gone, along with 2.5 hours of compute. `fyp-dr-eyepacs-224` never existed.

**The build was right and the packaging was wrong**, which is why every in-session check
passed. Reconciliation ran against the live tree, before the boundary where the loss
happened. Nothing in the pipeline was looking at the artefact that would actually survive.

### The rule

**Pack to one file, verify the archive, then delete the source — in that order, inside the
same session.**

1. `src/data/archive_cache.py pack` zips the cache with `ZIP_STORED` (the payload is
   38,788 JPEGs; deflate spends minutes to save low single digits) into
   `fyp-dr-eyepacs-224.zip`, with `splits/`, both stats CSVs and
   `_cache_provenance.json` inside it, so the published dataset describes itself.
2. It **refuses to pack a short source tree** — the expected image count is checked before
   a byte is written. Packing a short tree and verifying the archive against that same
   short tree would agree with itself perfectly.
3. `verify_archive` then reads the **central directory back** and checks the image count,
   the sidecars, zero-byte entries, and the total size against 0.836 GB. That is what a
   consumer reads, and it is exactly the check that would have caught 499.
4. Only after verification passes does the notebook delete the loose tree. If verification
   fails, the tree is still there and the session must not end.
5. One file is under any file-count cap, and the count is asserted again at the end of the
   cell: if `/kaggle/working` holds more than 400 files, it warns.

Training reverses it: `unpack` extracts to `/kaggle/working` and **re-counts the extracted
images**, because "the archive was complete" and "the extraction completed" are different
claims, and training depends on the second.

`tests/test_archive.py` includes the regression directly — an archive rebuilt holding 5 of
20 images, which `verify_archive` must reject.

### Kaggle API vs notebook output — considered, and the notebook output wins

Pushing straight to a dataset from the notebook via `kaggle datasets create` was the
obvious alternative. It needs, and cell 5 does not:

- **Internet ON** for the whole build session, which is otherwise unnecessary — the
  preprocessing pipeline has no network dependency at all.
- **Credentials in the notebook**, via Add-ons → Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
  Never `kaggle.json` in the repo, which `.gitignore` already forbids.

Its one real advantage is that the notebook can confirm the dataset landed **while the
session is still alive**, instead of trusting a UI step afterwards — which is precisely
what failed. That is a genuine argument, and it is why the API version is kept as an
optional cell 5b rather than dismissed.

**The manual publish is still the recommendation**, because the archive removes the
failure mode the API was going to protect against: once the output is a single verified
file, the cap cannot truncate it, and "create a dataset from one file" is a step with
nothing to silently drop. Adding a credential and a network dependency to a 2.5-hour
CPU-only job buys a confirmation that a verified single file no longer needs. The API
route also has upload behaviour this project has not tested — in particular whether Kaggle
re-extracts an uploaded archive into the dataset — and the middle of a rebuild is the
wrong time to find out.

**Revisit if** the manual publish ever fails again, or once arms A–F make dataset
publishing a routine step rather than a one-off.

### The general lesson, worth more than the fix

Every check in Phase 2 ran against the artefact **in the session**, and the thing that
broke was the artefact **that left the session**. A verification that does not run on the
bytes that survive is a verification of something else.

---

## Excluded data rows

**None.** Four images finished preprocessing as `ok:no-retina` (DECISION-018) and
**remain in their splits** — that is a processing note, not an exclusion.

The Phase 1 reconciliation found the EyePACS dataset completely clean: 35,126 CSV
rows, 35,126 image files, zero mismatches in either direction, zero duplicates, zero
unparseable filenames, zero zero-byte files. APTOS likewise: 3,662 rows across the three
author CSVs, 3,662 image stems on disk, zero duplicate `id_code`.

No row has been dropped from any manifest. If that ever changes, the excluded rows are
listed here with identifiers and reasons.

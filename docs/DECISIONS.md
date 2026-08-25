# Architecture Decision Record

ADR-style log of every non-obvious choice. Format: date, decision, alternatives
considered, rationale, who approved.

**Rule:** any deviation from `docs/proposal.pdf` is logged here *and* flagged to the user.
Any data rows excluded from the pipeline are listed here, never dropped silently.

---

## DECISION-001 — Streamlit replaced by FastAPI + Next.js

- **Date:** 2026-08-20
- **Status:** **APPROVED** — supervisor confirmed 2026-08-24. No longer a pending
  deviation; the proposal's §9 and §5.10/Month 8 Streamlit references are superseded
  and the thesis states the change and its reasons rather than flagging it as open.
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

| package | was pinned | Kaggle CPU image (2026-08-21) | Kaggle **GPU** image (2026-08-22) | now pinned |
|---|---|---|---|---|
| numpy | 2.2.6 | **2.0.2** | 2.0.2 | 2.0.2 |
| pandas | 2.3.3 | 2.3.3 | 2.3.3 | 2.3.3 |
| opencv | 4.12.0.88 | **4.13.0** | 4.13.0 | 4.13.0.92 |
| torch | 2.9.1 | **2.10.0+cpu** | **2.10.0+cu128** | 2.10.0 |
| torchvision | 0.24.1 | **0.25.0** | **0.25.0+cu128** | 0.25.0 |
| timm | 1.0.28 | **1.0.26** | 1.0.26 | 1.0.26 |

**The GPU image measured 2026-08-22 matches the CPU image version for version**, differing
only in the CUDA build tags (`+cu128`). That is the good case and it means one set of pins
describes both environments; it was not safe to assume in advance, which is why every
notebook prints its versions.

**The Kaggle image wins.** Never `pip install` torch over the preinstalled build — it is
built against that image's CUDA and replacing it is the fastest way to lose a session to
a broken driver stack. So the pins move to match the image, and the local machine installs
the same versions, which is what makes a locally-inspected result and a Kaggle result the
same artefact (R6).

The cache was built under the image's versions, not under the old pins. That is fine and
does not require a rebuild: the preprocessing output is deterministic per image and the
provenance sidecar records the config, which is the thing that governs the pixels.

**The GPU image was measured on 2026-08-22 and did not differ** — see the column above.
Cell 1 of every notebook still prints its versions, and if a future image differs this
table gains a column rather than the pins being edited from memory. What must not happen is a pin being
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

## DECISION-022 — No notebook command line runs before something has executed it

- **Date:** 2026-08-21
- **Status:** Accepted — forced by a third lost session
- **Deviates from proposal:** No

### What happened

The Phase 2 rebuild ran for **8.1 hours** and died in cell 5 with:

    archive_cache.py: error: argument cmd: invalid choice: '/kaggle/working/processed'

A missing `pack` subcommand, rejected by argparse in 0.0 seconds — after the build was
finished, after reconciliation had passed, after `READY TO ARCHIVE` had printed.
`/kaggle/working` is wiped, so all of it went.

**Three sessions, about 13 hours, none of it lost to the modelling.** DECISION-021 fixed
the truncation and this failed one layer further out: the fix itself was invoked wrongly.
Both copies in the repo — `notebooks/phase2_build_cache.py` and the runbook — contain
`pack`, so what ran had drifted from what was written, and nothing anywhere would have
noticed either way.

### The actual defect

**Notebook cells were the only executable code in this project that nothing imported,
nothing type-checked, and no test ran.** `archive_cache.py` had 13 unit tests. Its CLI had
never once been invoked the way the notebook invoked it. Every other layer here is
defended by tests that run on every change; the cells are strings in a docstring, and they
are simultaneously the code that runs **last**, **once**, **after hours of compute**, on a
machine that then erases itself. That is the worst possible combination, and it was the
least defended.

### The rule

**Every command line a notebook will run is validated against the real parser of the
module it calls, and the packaging path is executed for real, before any long work
starts.**

Four parts, and each covers a hole the others do not:

1. **`build_parser()` is split out of `main()`** in `preprocess`, `reconcile_cache`,
   `archive_cache` and `train`. A parser that only exists inside `main()` cannot be
   checked without running the program.
2. **`src/data/notebook_check.py`** reads `CELLS` out of every `notebooks/*.py`, extracts
   each `[sys.executable, "-m", ...]` list — including one assigned to a variable and run
   from another cell — substitutes a placeholder for values only known at runtime, and
   feeds the rest to the target module's parser. It also binds every direct call into
   `src.*` (`unpack(...)`, `resolve_input(...)`) against the real signature, because not
   every mistake is a subprocess and argparse never sees those.
3. **`--self-test` runs pack → verify → unpack for real** on a three-file temporary
   directory. This is the part that matters: an inspection proves the arguments are
   shaped right; only an execution proves the path works. It takes about a second.
4. **`tests/test_notebook_cells.py`** runs all of it on every test run, and includes the
   8.1-hour failure verbatim as a negative case — a `src.data.archive_cache` invocation
   with the subcommand removed, which `check()` must reject. A checker that cannot fail
   proves nothing.

### The paste is what runs, so the paste is what gets checked

The repo's copy of a cell is not what executes on Kaggle — a **paste** is, and the paste
is what drifted. Two things follow, and both are now in place:

- **Cell 1 of every notebook runs `notebook_check --all --self-test`.** It costs seconds
  and it happens before anything expensive.
- **The archive command is built and validated in cell 1, as `PACK_CMD`, and cell 5 runs
  only the variable.** The list that executes is the list that was checked, in the same
  pasted notebook, in minute one. Retyping the command in cell 5 is exactly how the
  subcommand went missing, so cell 5 no longer contains a command to retype.

### The general rule, which outlives this bug

Each of the last three failures was in the layer *around* the work rather than the work
itself: output packaging, then mount layout, then an argument list. The pattern is that
**the glue is the least-tested and most expensive part of the system**, because it runs
once, at the end, where a failure costs everything before it. Anything that runs last
gets tested first, and anything that cannot be tested cheaply gets executed cheaply on a
toy input before it is trusted with a real one.

---

## DECISION-023 — The cache root is found by shape, never by path

- **Date:** 2026-08-21
- **Status:** Accepted
- **Deviates from proposal:** No

### What happened

The Phase 2 rebuild worked. `ARCHIVE VERIFIED`, 38,797 entries, 0.852 GB, reconciliation
clean, published as `rah098/fyp-dr-eyepacs-224` v1 at 918.08 MB.

**Kaggle then auto-extracted the archive when it created the dataset.** The single
verified `.zip` that DECISION-021 exists to produce is not a `.zip` once it is mounted:
the dataset shows a `fyp-dr-eyepacs-224/` folder holding 38.8k files, with
`aptos_stats.csv` and `processed_stats.csv` beside it. Phase 3's cell 1 looked for
`MOUNT/*.zip` and then fell back to `MOUNT/processed`, and the real tree is neither.

That would have failed at the top of the training notebook rather than after hours of
work, so it was cheap this time. It is the same class of mistake as the last three.

### The rule

**Three layouts have now been observed for one artefact**, because the platform reshaped
it each time:

| | layout |
|---|---|
| The build writes | `/kaggle/working/processed/{eyepacs,aptos,splits}` |
| The archive holds | `processed/...` inside one `.zip` |
| The published dataset mounts | `fyp-dr-eyepacs-224/processed/...`, zip gone, CSVs beside it |

Predicting the fourth is a bet with a session on it, so the code stops predicting.
`find_cache_root` searches for the **shape** — a directory with `eyepacs/` and `aptos/`
under it that actually contain images — breadth-first from the mount, shallowest match
wins, and raises with the real directory listing when nothing matches. `resolve_cache`
wraps it: extract if there is a zip, otherwise locate in place, and **verify the image
count either way** before returning a path.

This is the same principle as DECISION-020's mount resolver, one level further in. Both
exist because a hosting platform's directory conventions are not a stable interface, and
`cache_relpath` is: it writes `eyepacs/<stem>.jpg` and `aptos/aptos_<stem>.jpg`, so the
cache root is definitionally the directory those sit under. That fact survives any
renaming or re-nesting done around it.

`python -m src.data.archive_cache locate --mount <path>` answers "where is it, and is it
complete" from a shell, without a notebook.

`tests/test_archive.py` covers every observed layout by name, including the one Kaggle
actually published, plus a nested copy (shallowest wins), a half-extracted tree with the
right directory names and no images, and a short cache that must be refused rather than
returned.

---

## DECISION-024 — Every arm's real path is exercised, not just its components

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No

The arm A smoke run failed twelve seconds in:

    AttributeError: 'RandomSampler' object has no attribute 'weights'
        src/data/sampler.py:168 in describe_balance, from src/train/train.py:187

`describe_balance` tested `if sampler is None` to decide whether to model a weighted draw.
That is the wrong question. `build_sampler` returns `None` for an unbalanced arm, but
`build_loaders` then passes `shuffle=True` and **`DataLoader` substitutes a
`RandomSampler`** — so what reaches `describe_balance` is a sampler object with no
weights, not `None`.

**Arm A is the only arm that never gets a weighted sampler.** Every existing test called
`build_sampler` directly and passed its return value straight to `describe_balance`, so
the `None` branch was well covered and the branch that actually runs under arm A had never
executed. The diagnostic print was the only thing broken; everything before it was correct
— 11.18M parameters 100% trainable, timm normalisation, sane class counts, leakage gate
green.

`describe_balance` now duck-types on `.weights`: anything that cannot reweight draws the
natural distribution by definition. The table also states which sampler it saw and notes
`drawn == natural` explicitly, because an arm whose sampler silently did nothing and an
arm that correctly draws naturally otherwise produce identical tables.

**The audit that followed.** Every branch on arm configuration in the train/eval path,
and whether it had ever run:

| Branch | Arms | Was it exercised? |
|---|---|---|
| `describe_balance` weighted vs natural | A/C/E vs B/D/F | **No** — the bug |
| `predictions_from` softmax vs ordinal | A-D,F vs **E** | Unit-tested, never inside `fit`/`evaluate` |
| `build_loss` weighted CE | **C** | Unit-tested, never inside `fit` |
| `metrics_by_dataset` pooled origins | **F** | Never — and arm F runs last, so it would have first executed at the end of the whole ablation |
| `build_loaders` `shuffle=(sampler is None)` | all | Yes |
| `build_loss` weights+sampler refusal | all | Yes |

Four of six branches were reachable only by an arm that had not been run. All four now
have end-to-end tests through the real call chain — `build_loaders` → `loader.sampler` →
`describe_balance`, and `fit`/`evaluate` with the ordinal head and with a weighted loss.
`metrics_by_dataset` was extracted out of `main()` to make it testable at all, and its
test includes a shuffled evaluation order, because that index is positional and joining it
with `.loc` would silently mix the two origins.

**The rule.** A component tested in isolation is not the same thing as an arm's path
tested end to end. Where behaviour branches on configuration, the test parametrises over
the real configs rather than over hand-built arguments — `tests/test_train.py` reads
`configs/arm_*.yaml` and asserts the sampler kind it finds, so a config change that
invalidates the test fails loudly instead of quietly bypassing it.

This is the same shape as DECISION-022: the least-run path is the most expensive one, and
it is cheapest to run it deliberately on synthetic data first.

---

## DECISION-025 — `train.main()` runs end to end locally before any session starts

- **Date:** 2026-08-22
- **Status:** Accepted — forced by the third consecutive failure inside `main()`
- **Deviates from proposal:** No

### What happened

Three Kaggle sessions in a row died inside `src/train/train.py`, each one step further in:

    1. describe_balance   AttributeError: 'RandomSampler' object has no attribute 'weights'
    2. yaml.safe_dump     RepresenterError: cannot represent an object '2.10.0+cu128'

`torch.__version__` is a `TorchVersion`, a `str` **subclass**, and `yaml.SafeDumper`
dispatches on **exact type**, not `isinstance` — so it refuses. Same for torchvision's.

The fix is not to special-case torch. **Nothing reaches `safe_dump` unless it is a
primitive**: `to_primitive` recursively coerces the whole run config, because the next
exotic type will be a numpy scalar from a YAML file, a `Path`, or an enum from a library
nobody has added yet.

### The actual defect, again

Every one of these was in `main()`, past where the unit tests reach. `fit`, `evaluate`,
`build_loaders`, `build_loss`, `describe_balance`, the metrics — all green, all along.
**`main()` itself had never once executed start to finish.** It was the only substantial
function in the project with no test, and it was the function every session ran first.

That is the same shape as DECISION-022, where notebook cells were the untested glue. The
common thread across five failures now: **the least-tested code is whatever runs last and
costs most, because it is the code that is hardest to run cheaply — which is exactly why
it has to be made cheap to run.**

### The rule

`python -m src.train.smoke` builds a throwaway 90-image cache and a patient-disjoint split
trio with real provenance headers, then calls **the real `main()` through its real argv**
and checks that `config.yaml` and `metrics.json` come out well-formed. About 20 seconds
per arm on CPU, no network, no GPU, no real data.

Three flags on `train.py` make it possible, and all three are recorded in the run config
so a smoke run can never be mistaken for a result:

| flag | why |
|---|---|
| `--splits-root` | point the real `main()` at a synthetic partition; `train.py` prints a loud warning and `config.yaml` records the path |
| `--no-pretrained` | no network in the test; it changes what a run measures, so it sets `is_smoke_run` |
| `--limit-train` | already existed; same treatment |

`config.yaml` gains `is_smoke_run` and `metrics.json` gains `is_smoke_test`, both true if
any of the three was used.

**It runs in three places**: `tests/test_train_smoke.py` (arms A and F, ~1 min, the
slowest tests in the suite and worth it), cell 1 of the Phase 3 notebook before anything
is spent, and `--all-arms` by hand before a real session.

**Arm A and arm F are the pair that matters.** A has no sampler and one origin; F has a
weighted sampler and two. Between them they touch every branch in `main()`. Arms B–E are
covered at unit level and by `--all-arms`.

### What building the fixture immediately caught

Arm F could not run at all: the fixture had no APTOS splits. Arm F is the last arm in the
ablation, so the real first execution of that path would have come after five other arms'
worth of GPU time. The fixture now writes `aptos_{train,val,test}` too, with hash-like ids
and one image per patient per DECISION-004, and the arm-F smoke shows the `p_aptos` column
DECISION-007 requires.

### What it deliberately does not do

It asserts **nothing about the score**. Twelve images per class cannot produce a meaningful
QWK, and asserting one would make this a flaky test instead of a wiring check. It asserts
the pipeline runs and its artefacts are readable. The collapse detector fires on every
smoke run, correctly, and that is expected rather than a failure.

---

## DECISION-026 — The collapse rule is severity-aware; Phase 3 acceptance follows it

- **Date:** 2026-08-22
- **Status:** Accepted — user challenge to a blocking verdict, upheld
- **Deviates from proposal:** No. CLAUDE.md §2 says "stop and diagnose", and this is the
  diagnosis. The rule's *trigger* is refined; its purpose is unchanged.

### What prompted it

The Phase 3 baseline (arm A, ResNet18, 8 epochs, best at epoch 6) produced:

| | |
|---|---|
| val QWK | **0.6138** [0.5859, 0.6421] |
| accuracy | 0.7965 (majority rate 0.7369, so **+6.0 pp**) |
| balanced accuracy | 0.4163 |
| referable sensitivity / specificity | 0.5131 / 0.9804 |
| per-class recall | 0.982 / **0.000** / 0.376 / 0.316 / 0.407 |
| predicted counts | 4657 / **0** / 478 / 68 / 65 |
| grade 3 recall | 0.316 [0.241, 0.396] |
| grade 4 recall | 0.407 [0.324, 0.504] |

QWK by epoch: 0.025, 0.511, 0.549, 0.546, 0.597, 0.595, 0.614, 0.614.

The rule reported a collapse and Phase 3 acceptance failed, because grade 1 was never
predicted.

### The assessment

**That is not a collapse, and the user was right to challenge it — but not quite for the
reason given.** "Four of five grades are predicted" is not the principle. Consider the
same shape with grade 4 unpredicted instead: also four of five, also a healthy QWK, and
**catastrophic**, because proliferative disease is the thing this system exists to catch
and a model that cannot emit that grade cannot flag it even in principle.

The principle is **whether the missing grade changes a referral**:

- The referable threshold is **grade ≥ 2** (`configs/base.yaml`). Grade 1 is Mild DR,
  **below** it. Folding grade 1 into grade 0 moves 357 val images (6.8%) between two
  labels that are *both non-referable*, so the clinical decision the deployed system
  makes is **unchanged**.
- QWK weights a 1-called-0 error at **1/16** of a 4-called-0 error, so the primary metric
  already prices this correctly.
- The degenerate failure the rule exists for — learning the prior and nothing else —
  scores 0.7369 accuracy and **0.0 QWK**. This run scores 0.6138 with a CI nowhere near
  zero, and its per-epoch curve rose monotonically from 0.025. It learned the ordering.

Absorbing a rare middle grade under plain cross-entropy at 6.8% prevalence is the
**expected behaviour of an unbalanced baseline**, and it is the specific deficiency arms
B–E exist to fix. Blocking the phase on it would block the pipeline for demonstrating the
problem the thesis is about.

### The rule, restated

`detect_collapse` now returns one of three levels. **Fatal**, meaning stop and diagnose:

1. Predictions concentrated in a **single class** — the degenerate case.
2. Accuracy within 0.02 of the **majority-class rate**.
3. A grade **at or above the referable threshold** never predicted while it has support.

**Warning**, reported everywhere but not blocking:

4. A grade **below** the referable threshold never predicted.

The threshold comes from `eval.referable_threshold` rather than being hardcoded, so a
project that moved the referral line would move this rule with it. An all-zero *row* is
still not reported at all: that is a property of the evaluation set, not the model.

Warnings reach `metrics.json` (`collapse.level`, `collapse.warnings`), `train_log.csv`
(`val_collapse_level`, `val_collapse_warnings`) and the per-epoch console line. A warning
that nothing records is the same as no rule at all.

### Phase 3 acceptance

Was: *val QWK CI excludes zero, and every grade predicted.*
Now: **val QWK CI excludes zero, and no FATAL collapse** — i.e. not single-class, not at
the majority rate, and every referable grade reachable.

### What this run does not excuse

The baseline is **correct**, not good, and the write-up must not blur those:

- **Referable sensitivity is 0.5131.** Roughly half of referable cases are missed. That is
  the number arms B–F have to move, and it is the clinically meaningful headline.
- **Balanced accuracy 0.4163** against accuracy 0.7965 is the imbalance, stated plainly.
- Grade 3 recall 0.316 [0.241, 0.396] and grade 4 recall 0.407 [0.324, 0.504] — wide
  intervals on thin support, exactly as DECISION-006 anticipated.
- QWK was still rising at epoch 7 (0.614 for two epochs, best at 6). Eight epochs is a
  short schedule; Phase 4 should not read this as converged.

---

## DECISION-027 — Kaggle transport: the CLI first, and the auth trap that cost an hour

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No

Fetching the Phase 3 artefacts needed three things fixed, none of them obvious from the
error:

1. **The `kaggle` package had to go 1.7.4.5 → 2.2.4.** The older pin does not expose
   `KaggleApi` where `fetch_run` imported it from. `requirements.txt` now pins 2.2.4.
2. **`kaggle auth login`** is the current mechanism — the OAuth flow, not an API token.
3. **A legacy `~/.kaggle/kaggle.json` SHADOWS the newer OAuth credentials.** This is the
   trap. The symptom is a flat `401 Unauthenticated` with no indication that a second
   credential source exists or that the one being used is the stale one. Renaming that
   file out of the way is what fixed it.

**`fetch_run` now shells out to the CLI first and falls back to the library**, on the
user's observation that the import path is the fragile part — and it is: the import
broke across a minor version bump, while `kaggle kernels output` did not change. The CLI
is also what knows how to use `kaggle auth login`'s credentials. Both transports are
tried, and **both failures are reported together** rather than the first hiding the
second, which is precisely what made this hour expensive.

The verification layer is unchanged and transport-independent — what makes a fetched run
trustworthy is that the three artefacts parse and agree with each other, not how the
bytes arrived.

**Kernel slug for the Phase 3 baseline: `rah098/fyp-dr-phase3-baseline`.**

### A provenance gap the fetched artefact revealed

`runs/phase3_baseline_resnet18/config.yaml` records `git: unknown`. On Kaggle the code
arrives as an unzipped Dataset, not a git checkout, so `git rev-parse` fails — and the
run cannot name the commit that produced it, which is an R6 weakness.

`make_bundle` already writes the SHA into `BUNDLE.txt` for exactly this reason.
`_git_sha()` now falls back to reading it, so every Phase 4 run records its code version.
The baseline's `unknown` stands: it is what that run recorded.

---

## DECISION-028 — Nothing under `notebooks/` runs at import time, and the gate survives one that does

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No

Phase 4 cell 1 died on Kaggle before any GPU time, in the gate itself:

    FileNotFoundError: 'runs/phase3_baseline_resnet18/metrics.json'
      gen_experiments.py line 15, during importlib.import_module("notebooks.gen_experiments")

`notebook_check` imports every module under `notebooks/` to collect the command lines its
cells will run. `gen_experiments.py` read the Phase 3 artefacts at **module level** — it
had been lifted out of a throwaway script without moving its body into `main()`. `runs/`
is not in the code bundle, so the import raised, and `notebook_check` exited 1 **before
checking anything**.

Two separate defects, and both are worth fixing.

**1. A module under `notebooks/` must not touch the filesystem at import time.** Those
modules exist to be imported by tooling — that is the whole mechanism `notebook_check`
relies on. `gen_experiments` now does all its work inside `main()`, exposes
`build_parser()`, and derives its root from `__file__` rather than `Path(".")`; the
cwd-dependence was a second latent bug in the same six lines.

**2. The gate must not be brought down by one module.** A checker that dies on the first
bad input tells you nothing about the other inputs — and it failed *closed* in the worst
possible way here, reporting a single import error while 21 command lines across three
notebooks went unexamined. `notebook_import_errors()` now reports a failing module as a
finding, and `collect_invocations` / `collect_api_calls` skip it and carry on. The failure
is reported **first and loudest**, because the consequence is specific and severe: that
notebook's cells are **UNCHECKED**, which is worse than any single bad command line.

`tests/test_notebook_cells.py` pins both halves. One test imports every module under
`notebooks/` fresh from a working directory with **no `runs/`, no `data/`, no `docs/`** —
the Kaggle condition, the code bundle and nothing else. Another plants a module that
raises on import and asserts the gate names it *and* still collects every other
notebook's invocations.

### The pattern, for the fifth time

Cell 1 exists because three sessions were lost to unvalidated glue. This failure was **in
the validator**, which is the same shape one level up: the thing that runs first, once,
protecting something expensive, was itself the least-exercised code. It is now exercised
by tests that reproduce the environment it actually runs in rather than the one it was
written in.

Everything downstream of it had already passed on Kaggle before it fell over — cache
38,788 verified, Tesla T4, bundle git `456444a`, pretrained weights OK.

---

## DECISION-029 — The collapse rule's accuracy test needs both halves

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No

Phase 4 stage 1 reported `collapse: collapsed` for **four of the five arms that ran** —
B, D, E and F — including arm E, which posted the ablation's **highest QWK, 0.7081**.
That is a false positive, and its cause is arithmetic:

| arm | accuracy | ≤ majority + 0.02 (0.7569)? | QWK |
|---|---:|---|---:|
| A | 0.8050 | no | 0.6823 |
| B | 0.6902 | **yes** | 0.5841 |
| D | 0.6327 | **yes** | 0.5538 |
| E | 0.7272 | **yes** | 0.7081 |
| F | 0.6695 | **yes** | 0.5617 |

**Rebalancing deliberately trades accuracy for rare-class recall**, so an arm with a
weighted sampler sits at or below the 0.7369 majority rate *by design*. The bare accuracy
test therefore fires on exactly the thing the ablation exists to measure.

The failure the test is for is "respectable-looking accuracy, no actual learning", and
its signature is accuracy near the majority rate **and QWK near zero**. A model with a
real QWK is by definition not predicting one class for everything — and that case is
already caught by rule (1). So the test now requires **both halves**, with `qwk_floor`
defaulting to 0.10. When accuracy is low but QWK is healthy it emits a *warning* saying
so, because the observation is still worth printing.

**A second, separate fix.** An existing test asserted that never predicting grade 0 is
fatal, and it passed only because the accuracy heuristic happened to catch it. That is
not a reason. Never predicting the **majority grade** is now fatal in its own right:
on a 73.7%-grade-0 population it means referring every patient, so specificity collapses
and the screen is useless — even though grade 0 sits below the referral line.

**No model changes.** This is a reporting fix; nothing about any trained arm is affected,
and the corrected verdicts are re-derived from the committed confusion matrices when
`docs/EXPERIMENTS.md` is regenerated.

---

## DECISION-030 — QWK selects the mechanism; a sensitivity floor gates deployment

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** Extends it. The proposal names QWK as the primary metric
  and that is unchanged; this adds a feasibility constraint on the *deployed* model.

### The tension, in the measured numbers

| arm | val QWK | referable sensitivity |
|---|---:|---:|
| E (ordinal) | **0.7081** (best) | **0.5598** (worst) |
| A (baseline) | 0.6823 | 0.6404 |
| B (sampler) | 0.5841 | 0.6725 |
| D (focal+sampler) | 0.5538 | 0.6307 |
| F (pooled) | 0.5617 (EyePACS-only) | **0.6917** (best) |

The ranking by QWK is close to the **inverse** of the ranking by referable sensitivity.
Spearman ρ = −0.40 across the five arms — with n=5 that is **not statistically
significant** (p = 0.51) and must not be reported as a correlation. What is not in doubt
is the concrete pair: the arm with the best QWK has the worst sensitivity, and the arm
with the best sensitivity is near the bottom on QWK.

That is not noise, it is a real trade-off. QWK rewards agreement across the whole ordinal
scale; referable sensitivity rewards catching disease at the 2+ boundary and is
indifferent to everything else. Optimising one does not optimise the other.

### The decision

**1. The ablation's selection metric stays validation QWK.** It is pre-registered in
CLAUDE.md §5 and the proposal. Changing it *after seeing the results* would be
criterion-shopping, and the comparison would no longer mean what it claims. This is not a
close call.

**2. QWK alone must not choose the deployed model.** A screening system is defined by its
operating characteristic. The headline model must additionally meet a **referable
sensitivity floor at an operating point chosen on validation** (R3), reported with its
specificity. Selecting a grading mechanism and choosing an operating point are two
different decisions and the project was conflating them.

**3. Arm E cannot be compared on sensitivity yet, and this is the important caveat.**
Arm E is an ordinal head whose grades come from cut points, and it is currently using the
**untuned defaults `[0.5, 1.5, 2.5, 3.5]`**. `configs/arm_e.yaml` says those are to be
optimised on validation; `src/eval/thresholds.py` does not exist yet. Its 0.5598 is a
property of an arbitrary cut point, not of the arm. Moving the grade-2 threshold trades
sensitivity for specificity directly — which is an **advantage** of the ordinal head, not
a defect: it is the only arm whose operating point is a free dial. Comparing it against
softmax arms on sensitivity before turning that dial is not a fair comparison in either
direction.

**4. None of these models is deployable as a screen.** Published DR screening standards
ask for roughly **80%+ sensitivity for referable disease at ≥95% specificity**
(figure from the literature, not measured here — the supervisor should confirm the
standard that applies). Every arm sits at 0.56–0.69. That gap is an honest finding for
the write-up, not something to be smoothed over, and it is what Phase 4's remaining work
and Phase 5 exist to address.

### What happens next, in order

1. Build `src/eval/thresholds.py` and optimise arm E's cut points on **validation only**.
2. Re-report every arm's sensitivity/specificity at a matched operating point.
3. Apply the floor to the headline model, not to the ablation ranking.

---

## DECISION-031 — Pooling APTOS is a null result on the target domain

- **Date:** 2026-08-22
- **Status:** Accepted — this is the measured answer to the supervisor's suggestion
- **Deviates from proposal:** No. DECISION-007 set this up as an empirical question.

### The comparison that was nearly made, and why it is wrong

Arm F's EyePACS-only QWK is **0.5617** against arm A's **0.6823** — an apparent loss of
0.12 from pooling. **That comparison is invalid.** Arm F uses the weighted sampler and
arm A does not, so F-vs-A confounds *pooling* with *sampling*.

The comparison that isolates pooling is **F vs B**, which share the sampler and differ
only in the training data:

| | sampler | training data | EyePACS-only val QWK |
|---|---|---|---:|
| B | weighted | EyePACS | 0.5841 [0.5574, 0.6108] |
| F | weighted | EyePACS + APTOS | **0.5617** |

**−0.0224, and F's value sits inside B's 95% interval.** Pooling APTOS has **no
detectable effect** on EyePACS grading. The large effect in the table belongs to the
sampler, not the data: B − A = −0.098.

### Is the domain artefact visible?

Partly, and less conclusively than the raw numbers suggest.

**Arm F scores 0.8609 on APTOS and 0.5617 on EyePACS.** That gap is *not* by itself
evidence of a shortcut: APTOS is intrinsically easier — 49.3% grade 0 against EyePACS's
73.5%, so QWK is mechanically higher on it — and its validation set is n=550 against
5,268. A large APTOS/EyePACS gap is the expected result even for a model using no domain
cue at all.

What **is** demonstrated is the inflation DECISION-007 predicted: F's **pooled** QWK is
0.6192 against 0.5617 EyePACS-only, so **+0.058 of arm F's headline number comes purely
from including easier APTOS images**. That is exactly why DECISION-007 mandated
EyePACS-only selection for this arm, and it is now measured rather than anticipated.

Whether the model is *using* a camera cue as a severity proxy remains open. The
`p_aptos` column shows the correlation is present in the training draw; establishing
that the model exploits it needs the Phase 5 Grad-CAM comparison by origin, or an arm F
run without the sampler. Neither has been done, and the claim should not be made until
one of them has.

### The answer to the supervisor

The suggestion was that APTOS could relieve the class imbalance. The measured answer:
**it does not improve grading on the target domain.** Against the correct comparator it
is a null result within the confidence interval, while it adds a domain confound and
inflates the arm's own headline metric.

That is a real contribution rather than a disappointment — a reasonable idea, tested
properly with the right comparator, and reported honestly. The write-up should state the
F-vs-B comparison explicitly, because F-vs-A is the comparison a reader will reach for
and it is the wrong one.

**Caveat: one seed.** F is in the stage 2 top three, so the null result will be repeated
at seeds 43 and 44 before it is claimed.

---

## DECISION-032 — The sensitivity floor is 80% at 95% specificity, from the UK screening standard

- **Date:** 2026-08-22
- **Status:** Accepted, **pending supervisor confirmation of the applicable standard**
- **Deviates from proposal:** Extends it. Implements the floor DECISION-030 called for.

### The proposed floor

**Referable-DR sensitivity ≥ 0.80 at specificity ≥ 0.95**, on validation, at an operating
point chosen on validation only (R3).

Both halves are required and the pair is the standard, not either number alone.
Sensitivity by itself is trivially satisfiable by referring every patient — the model that
scores 1.00 sensitivity and 0.00 specificity is the one that refers everybody, and it is
useless. `choose_operating_point` therefore reports the joint point, and separately what
each constraint costs when the other binds.

### The basis, with citations

**Primary — the UK standard.** Diabetic retinopathy screening programmes in the UK are
held to a minimum of **≥80% sensitivity, ≥95% specificity, and ≤5% technical failure
rate**. The figures originate with the British Diabetic Association (now Diabetes UK) and
are carried into NICE guidance and NHS Diabetic Eye Screening Programme standards.
Verified 2026-08-22 against:

- [Screening for diabetic retinopathy — GPnotebook](https://gpnotebook.com/pages/ophthalmology/screening-for-diabetic-retinopathy)
- [Diabetic eye screening programme — PrimaryCareNotebook](https://primarycarenotebook.com/pages/ophthalmology/diabetic-eye-screening-programme)

**Secondary — what a cleared autonomous AI device achieves.** The FDA-authorised IDx-DR
system reported **87.2% sensitivity and 90.7% specificity** (with 96.1% imageability) for
more-than-mild DR in its pivotal trial, exceeding all pre-specified superiority endpoints:

- Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. *Pivotal trial of an autonomous
  AI-based diagnostic system for detection of diabetic retinopathy in primary care
  offices.* **npj Digital Medicine** 2018;1:39.
  [nature.com/articles/s41746-018-0040-6](https://www.nature.com/articles/s41746-018-0040-6)
  · summary confirmed via [PMC8120059](https://pmc.ncbi.nlm.nih.gov/articles/PMC8120059/)

### What is NOT verified, and must be before it is written up

Three things were **not** confirmed and must not be stated as fact in the thesis:

1. **The exact numeric values of IDx-DR's pre-specified endpoints.** The trial "exceeded
   all pre-specified superiority endpoints" is confirmed; the specific thresholds
   (commonly quoted as 85% sensitivity / 82.5% specificity) were **not** verifiable — the
   FDA De Novo document DEN180001 returned 404 and the Nature article is behind an
   authentication redirect. Cite the achieved figures, not the endpoints, unless the
   primary source is obtained.
2. **The precise NICE guideline number and clause.** The 80/95 figures are corroborated
   by multiple secondary sources, but the primary NICE/NHS DESP document was not read
   directly. **Obtain the primary source before it goes in the thesis.**
3. **Whether mtmDR maps exactly onto this project's referable definition.** "More than
   mild DR" is ETDRS-based; this project's referable line is `grade >= 2` on the
   EyePACS/ICDR-like 0–4 scale. They are closely aligned in intent — grade 2 is Moderate
   NPDR — but they are not the same instrument, and UK DESP referable criteria also
   include maculopathy, which these labels do not encode at all.

**This is a benchmark, not a like-for-like requirement**, and the write-up should say so.
The value is that it anchors the number to a named external standard instead of leaving
"sensitivity 0.64" floating without a reference point.

### Where this project stands against it

| arm | referable sensitivity | specificity |
|---|---:|---:|
| standard | **≥ 0.80** | **≥ 0.95** |
| IDx-DR (cleared device) | 0.872 | 0.907 |
| A | 0.6404 | 0.9804 |
| B | 0.6725 | — |
| D | 0.6307 | — |
| E | 0.5598 | — |
| F | 0.6917 | — |

**No arm meets the standard.** That is the honest headline and it belongs in the write-up.

But note what arm A's pair says: **0.98 specificity against a 0.95 floor, and 0.64
sensitivity against a 0.80 floor.** It has roughly three points of specificity in surplus
and sixteen points of sensitivity missing. The operating point is sitting far too
conservative, and surplus specificity is spendable — which is exactly what
`src/eval/thresholds.py` now measures, per arm, without retraining anything. How much of
the gap that closes is an empirical question that the tool answers and this decision does
not prejudge.

---

## DECISION-033 — Arms are compared under a matched decision rule, and the stage 1 ranking changes

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No. QWK remains the metric; what changes is that every arm
  is scored with the same number of free parameters.

### The asymmetry

The stage 1 table compared **arm E's QWK after fitting four cut points on validation**
against every other arm's QWK from a bare `argmax`. Those are not the same quantity: E
was scored with four fitted parameters and the softmax arms with none.

Giving every arm the same four degrees of freedom — cut points on the **expected grade**
`sum(p_i * i)` for a softmax head, which is the natural continuous analogue that `argmax`
discards — and measuring the optimism by fitting on half the validation set and scoring
the other half:

| arm | as run | fitted | held-out | optimism | gain | train−val gap |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6823 | 0.7315 | **0.7194** | 0.0121 | +0.037 | +0.092 |
| E | 0.7081 | 0.7263 | **0.7132** | 0.0131 | +0.005 | +0.073 |
| F | 0.6192 | 0.7091 | 0.7004 | 0.0087 | +0.081 | +0.301 |
| D | 0.5538 | 0.6541 | 0.6400 | 0.0140 | +0.086 | +0.300 |
| B | 0.5841 | 0.6474 | 0.6295 | 0.0180 | +0.045 | +0.290 |
| C | 0.4048 | 0.6078 | 0.5967 | 0.0111 | **+0.192** | −0.145 |

**Ranking on held-out fitted QWK: A > E > F > D > B > C.**
**As run it was: E > A > F > B > D > C.**

Two positions moved, and neither move is about imbalance handling — the decision rule
moved them. **Threshold-fitting optimism is 0.009–0.018 QWK**, so the fitted numbers are
usable; that is measured here rather than assumed, and it is re-measured whenever
`src/eval/compare_arms.py` is re-run.

### A and E are a tie, not a lead

Paired bootstrap on held-out fitted QWK, on the same validation images:

    QWK(A) - QWK(E) = +0.0069   95% CI [-0.0203, +0.0291]   A better in 74% of resamples

**Not separable.** E's apparent 0.04 QWK lead was substantially the fitting asymmetry.
At a fixed operating point the two were already tied (sens@spec 0.6618 vs 0.6589), and
they are tied on QWK too once the comparison is fair. **No arm is the presumptive winner
on this evidence.**

### What the arms are actually doing

The `train − val` QWK gap at the best epoch separates them far more cleanly than the
headline metric does:

- **A (+0.092) and E (+0.073)** — mild overfitting, and the two best arms. Neither
  rebalances anything.
- **B (+0.290), D (+0.300), F (+0.301)** — the three arms with the **weighted sampler**.
  They reach train QWK 0.87–0.92 and lose ~0.30 on validation. Sampling with replacement
  shows the same few hundred grade-3 and grade-4 images many times per epoch, and the
  model memorises them. Their epochs confirm it: B stopped at 16 (best 8), D at 14
  (best 6), against A and E at 28 (best 20).
- **C (−0.145)** — the only arm that *underfits*. Train QWK 0.26 against val 0.40, train
  loss 1.71 where every other arm is 0.13–0.71.

**The headline result of the ablation is therefore negative: neither rebalancing mechanism
beats doing nothing on this data.** That is a real finding, not a disappointment, and it
is the honest answer to the question the ablation was built to ask.

---

## DECISION-034 — Arm C's failure is a displaced decision boundary, not a broken model

- **Date:** 2026-08-22
- **Status:** Accepted
- **Deviates from proposal:** No

Arm C looks catastrophic as run — QWK 0.4048 against arm B's 0.5841 — and the obvious
write-up line, "the sampler beats class weights", would be **overstated**.

**Re-thresholding recovers most of it: +0.192 QWK, the largest gain of any arm, to
0.5967 against B's 0.6295.** At matched decision rules the gap is **0.03, not 0.18**, and
that is inside the range where these arms are not separable on one seed.

### The mechanism, for the viva

`class_weights_from` uses inverse frequency, normalised to mean 1. On the committed train
split that is:

    grade  0      1      2      3      4
    weight 0.061  0.637  0.299  1.812  2.191      ratio 4:0 = 36x

**A grade-0 mistake costs 1/36 of a grade-4 mistake**, so the model rationally gives grade
0 away. Its confusion matrix shows exactly that: **1,385 of 3,882 grade-0 images (35.7%)
predicted as grade 1**, and grade-0 recall collapsing to 0.493. It is the only arm where
**specificity** is the binding constraint on the operating point — every other arm is
limited by sensitivity. That is the signature of a decision boundary pushed too far
toward the rare classes, and moving the boundary back is exactly what re-thresholding does.

The residual damage, after the boundary is corrected, is **optimisation instability**.
Weighting does not change which images are in a batch: a batch of 64 still holds about
1.3 grade-4 images on average, and often none — but when one appears it arrives with 36×
the gradient of a grade-0. The expected gradient matches the sampler's; its **variance is
far higher**. That shows up as the highest train loss in the ablation (1.71), a train QWK
*below* its own validation QWK, and an early stop at epoch 9.

So the two mechanisms fail in opposite directions, and neither is subtle:

| | what it changes | how it fails here |
|---|---|---|
| **Sampler** (B, D, F) | which images are in the batch | rare images repeat within an epoch and get memorised — train QWK 0.87–0.92, gap ~0.30 |
| **Class weights** (C) | how much each image counts | boundary displaced 36:1, and high-variance gradients destabilise optimisation |

### The claim the thesis can actually make

Not "the sampler beats class weights". The defensible claims are:

1. **Inverse-frequency weighting displaces the decision boundary severely, and most of
   the apparent damage is recoverable by re-thresholding on validation.** The comparison
   is only fair after that correction.
2. **Neither mechanism beat the unbalanced baseline** on validation QWK at matched
   decision rules.
3. The result is specific to **inverse-frequency** weighting. `class_weights_from` also
   implements `effective_number` (Cui et al. 2019), which exists precisely because
   inverse frequency over-corrects, and **it has not been run**. Claiming "class weighting
   does not work" from arm C alone would overreach; claiming it about inverse-frequency
   weighting at a 36:1 ratio is supported.

---

## DECISION-035 — Every ranking claim uses a matched decision rule. Permanently.

- **Date:** 2026-08-22
- **Status:** Accepted — **standing rule, applies to every comparison in the thesis**
- **Deviates from proposal:** No. QWK is still the metric.

**This is the single easiest way for the write-up to mislead**, and it already did once:
the stage 1 table ranked arm E first on a QWK computed with four fitted cut points, above
arms scored with a bare `argmax` and no fitted parameters at all. Two positions in the
ranking moved when that was corrected, and the apparent winner became a statistical tie.

### The rule

**No ranking claim — in a table, a figure, a sentence, or the viva — may compare arms
scored with different numbers of free parameters.**

1. **Every arm gets the same degrees of freedom.** Four cut points, fitted on validation.
2. **Softmax heads are scored on the expected grade** `sum(p_i * i)`, not `argmax`. The
   expected grade is the continuous analogue of an ordinal output; `argmax` collapses
   "spread across 2 and 3" and "confident 2" into the same answer, and those are not the
   same evidence.
3. **Fitting optimism is measured and reported**, by fitting on half the validation set
   and scoring the other half. On this ablation it is 0.009–0.018 QWK. That number is a
   result, not an assumption, and it is re-measured on every run of
   `src/eval/compare_arms.py`.
4. **Differences are reported with a paired bootstrap interval**, paired on the same
   validation images. An interval spanning zero is a **tie** and is written up as one.
5. **The as-run column stays** — it is what the run actually produced and dropping it
   would hide the effect of the decision rule. It is **never** the basis of a ranking
   claim.

### Why the as-run column must stay but must not rank

The gap between the two columns is itself a finding. Arm C gains **+0.192 QWK** from
re-thresholding and arm E gains **+0.005**: that difference says something real about how
badly inverse-frequency weighting displaces the decision boundary (DECISION-034). Deleting
the as-run numbers would erase the evidence for that; ranking on them would attribute a
decision-rule artefact to the imbalance mechanism.

`src/eval/compare_arms.py` prints both columns side by side and, when the two rankings
disagree, says so explicitly:

    ranking on held-out fitted QWK: A > E > F > D > B > C
    as-run ranking was          : E > A > F > B > D > C
    ^ the decision rule, not the imbalance mechanism, moved these

### Where it applies

Every arm comparison, stage 2's seed repeats, stage 3's capacity test, the arm F pooled
question (which additionally uses EyePACS-only scores, DECISION-007), and the final
model-selection decision. The one number that is **not** subject to this is the test-set
result, which is produced once, at a single operating point fixed on validation
beforehand, and is not a ranking.

---

## DECISION-036 — The stage 3 prediction is half wrong, and it tested the wrong variable

- **Date:** 2026-08-24
- **Status:** Accepted
- **Deviates from proposal:** No.

The pre-registration in `docs/EXPERIMENTS.md` called stage 3 a **capacity** test. It was
not one. **EfficientNet-B0 has 4.01M parameters against ResNet18's 11.18M** — measured,
not looked up. The run swapped in a backbone that is 64% *smaller*, so whatever it
measured, it did not measure the effect of more capacity. That is a design error in the
pre-registration, recorded here rather than quietly re-labelled.

### What the prediction said and what happened

| claim | outcome |
|---|---|
| Arm B's train−val gap stays ≥ 0.25 | **held** — 0.290 → 0.329 |
| Arms A and E move < 0.03 QWK | **A held** (+0.015); **E broke it** (+0.053) |
| B0 will not close the 0.14 referable-sensitivity gap | **wrong** — E's sens@spec≥0.95 went 0.6589 → 0.7464, closing 62% of the distance to the 0.80 floor |

### What it actually tested, and the result

Backbone quality at matched compute (B0 is 1.08× ResNet18 per epoch). All three arms
improved. The informative part is that **the two softmax arms turned better features into
more overfitting and the ordinal arm did not**:

> **CORRECTED BY DECISION-042 (2026-08-25).** The B0 figures below are single seeds.
> Seed-resolved, arm E's B0 gap is **0.077** (mean of 0.034/0.098/0.097) and arm A's is
> **0.130** (0.157/0.137/0.096). **Arm E's gap did not narrow — it was flat (+0.004).**
> The "opposite response" claim is withdrawn. What survives: arm A's gap widened and arm
> E's did not, and arm E's gap is consistently about 60% of arm A's.

| arm | head | gap on R18 `[1 seed]` | gap on B0 `[1 seed, superseded]` | Δgap |
|---|---|---:|---:|---:|
| A | softmax | 0.092 | 0.157 | +0.065 |
| B | softmax | 0.290 | 0.329 | +0.039 |
| E | ordinal regression | 0.073 | ~~0.034~~ → 0.077 | ~~−0.039~~ → +0.004 |

The gap response still differs by head, but the difference is that arm A's gap widened
while arm E's held steady — not that they moved in opposite directions.

The mechanism: cross-entropy over five logits can always spend a better representation on
driving the correct logit higher on individual training images, and confidence does not
transfer. Smooth-L1 on a single scalar cannot — the output is already in grade units and
the loss stops rewarding a residual once it is small — so a better representation has
nowhere to go except better ordering. The corroborating evidence is in the cut points:
**arm E on B0 has a fitted first cut of exactly 0.500, the default**, and re-thresholding
buys it +0.002 QWK against +0.041 for A and +0.046 for B.

### Consequences

1. **Capacity remains untested.** No claim about it may be made in the write-up.
2. **Backbone quality is a demonstrated, cheap lever**: +0.0408 held-out QWK
   [+0.0181, +0.0654], separable, for 1.08× compute.
3. **"Regularise" now applies only to arms we are not going to use.** Arm E's gap is
   **0.077 across three seeds** (DECISION-042 corrects the 0.034 first written here),
   against arm A's 0.130 and arm B's 0.329. Arm E still overfits far less than the arms
   it is being carried forward over, so the conclusion holds — the number was wrong by
   2.2×, the prescription was not.
4. **384px moves down the queue but not off it.** Its argument — microaneurysms are a few
   pixels across at 224 — is physical and untouched by this result, but it costs ~2.9×
   compute plus a ~2.5 h cache rebuild, against ~1 h to test one more backbone.

### The new methodological risk this creates

Architecture is now being selected on the same 5,268 validation images as everything else.
DECISION-035's optimism measurement covers **cut points only**; it does not cover
architecture choice. The test set is still untouched, so the final reported number stays
honest — but the *validation* figure is now optimistic by an amount nobody has measured.
Every architecture decision made on validation is counted in `PROGRESS.md` and the count
is reported in the write-up.

---

## DECISION-037 — Arm C's failure is mostly, but not only, a displaced boundary

- **Date:** 2026-08-24
- **Status:** Accepted
- **Deviates from proposal:** No.

DECISION-034 proposed that inverse-frequency class weighting fails by displacing the
decision boundary rather than by learning a worse model. Arm C2 (effective-number
weighting, 17.1:1 against inverse frequency's 36:1) tests it. **The story holds, and the
decomposition is not 100/0.**

The displacement is monotone in how aggressive the weighting is:

| arm | weight ratio | mean expected grade on **true grade-0** images | fitted first cut | as-run grade-0 recall | re-thresholding gain |
|---|---:|---:|---:|---:|---:|
| A | 1.0 | 0.414 | 0.642 | 0.964 | +0.045 |
| C2 | 17.1 | 1.456 | 1.581 | 0.879 | +0.045 |
| C | 36.0 | **1.877** | **1.970** | **0.493** | **+0.200** |

Arm C's average output on an image that is genuinely healthy is **1.877** — nearly two
grades of severity — and recovering from that needs the first cut moved +1.47. Once the
weights are gentler, **C2's re-thresholding gain is identical to plain arm A's**, so the
pathology is a property of the weight ratio, not of class weighting as such.

**The decomposition.** C2 beats C by **+0.198 as-run**, but at matched decision rules by
only **+0.0436 [+0.0144, +0.0746]** (paired bootstrap, separable). So **about 78% of the
apparent advantage is decision-boundary displacement and about 22% is a genuinely better
learned ordering.** Aggressive weighting hurts twice: mostly by moving the operating
point, which thresholding recovers, and partly by degrading the ranking, which it does not.

**What survives for the write-up.** At matched rules C2 is 0.6483 and C is 0.6050, both
still well below arm A's 0.7270. The conclusion that **loss-level class weighting
underperforms leaving the distribution alone** is now robust to the weighting scheme and
to the decision rule, which is exactly what one run of one scheme could not establish.

---

## DECISION-038 — Stage 3.5: one backbone step, pre-registered, with a stopping rule

- **Date:** 2026-08-24
- **Status:** Accepted — **pre-registered before the run**
- **Deviates from proposal:** No. EfficientNet is the proposal's primary family.

DECISION-036 found backbone quality to be the only lever with separable evidence
(+0.0408 QWK [+0.0181, +0.0654] for 1.08x compute). This tests one more step and then
**closes the question**, so that a ladder does not turn into an unbudgeted search.

### The manipulated variable, in parameters and FLOPs

Measured locally with `timm` and `torch.utils.flop_counter`:

| backbone | params | GMAC @224 | native input |
|---|---:|---:|---:|
| resnet18 | 11.18M | 1.814 | 224 |
| efficientnet_b0 | 4.01M | 0.385 | 224 |
| **efficientnet_b2** | **7.70M** | **0.658** | **256** |

The step is **+3.69M parameters (+92%) and +0.273 GMAC (+71%)** over B0. **The entire
ladder sits below ResNet18 on both measures**, which is why no result anywhere on it may
be attributed to capacity. What varies is architecture family and pretrained-feature
quality.

### The confound, stated before the run

B2's native input is 256 and we run at 224 to hold resolution fixed against every other
run in the project. B0's native input **is** 224. B2 is therefore handicapped and B0 is
not, so **a null result does not show the backbone lever is exhausted** — only that this
step, at this resolution, did not pay. Only the second claim may be written up.

### The prediction

**B2 will not beat B0 by a separable margin: |dQWK| < 0.02, paired interval spanning
zero.** The ResNet18 -> B0 gain came from crossing architecture *families* — a different
inductive bias and a stronger ImageNet initialisation. B0 -> B2 is within-family compound
scaling at a below-native input size, the weakest form of the same lever.

### The stopping rule — every outcome stops

| paired held-out dQWK (B2 − B0) | decision |
|---|---|
| not separable | keep **B0** — smaller, cheaper, native at 224 |
| separable and positive | keep **B2**. Do not run B3. |
| separable and negative | keep **B0**, noting the off-native confound |
| any of the above, but B2 clears sens@spec>=0.95 of 0.80 | report as deployment-relevant, still do not run B3 |

Nothing escalates. Reopening the ladder requires a positive argument logged first
(DECISION-036), never a good result on its own. `notebooks/phase4_stage35.py` cell 3
applies this table rather than interpreting it.

### After this

Backbone fixed -> stage 2 seeds (arms E and A, 3 seeds, on the chosen backbone) -> the
384px decision on arm E alone.

---

## DECISION-039 — Stage 3.5 resolved: keep B0. The backbone question is closed.

- **Date:** 2026-08-25
- **Status:** Accepted — **the pre-registered prediction held**
- **Deviates from proposal:** No.

DECISION-038 predicted B2 would not beat B0 separably, |dQWK| < 0.02, interval spanning
zero. Measured, arm E, matched decision rule, paired on the same 5,268 validation images:

```
paired: QWK(E/b2) - QWK(E/b0) = +0.0052  95% CI [-0.0195, +0.0285]
  E/b2 better in 68% of resamples
  NOT SEPARABLE - the interval spans zero, so this pair is a tie
```

**AS PREDICTED.** Stopping rule row 1: **keep EfficientNet-B0** — smaller (4.01M vs
7.70M), cheaper (0.385 vs 0.658 GMAC) and native at 224 where B2 is not.

The screening operating point points the same way, which the QWK tie alone would not have
shown:

> **THE SUPPORTING ARGUMENTS BELOW ARE STRUCK BY DECISION-042 (2026-08-25).** Both
> compared single seeds. Seed-resolved at 224, B0's sens@spec≥0.95 ranges 0.7211–0.7464
> (mean 0.7360) and its gap averages 0.077 — so **B2's 0.7211 is exactly B0's worst seed,
> not 2.5 points behind it**, and B2's gap of 0.061 is **lower** than B0's mean, not
> double it. **The decision to keep B0 stands** on the two reasons that survive: the QWK
> difference is not separable, and B0 is smaller, cheaper and native at 224.

| run | held-out QWK | sens@spec>=0.95 | spec@sens>=0.80 | gen gap |
|---|---:|---:|---:|---:|
| **E/b0** | 0.7560 | 0.7464 `[1 seed; 3-seed mean 0.7360]` | 0.9030 `[1 seed]` | ~~0.034~~ → 0.077 |
| E/b2 | 0.7600 | 0.7211 `[1 seed]` | 0.8738 `[1 seed]` | 0.061 `[1 seed]` |

B2 is nominally ahead on QWK by an amount that is not separable. ~~and behind on the
metric that gates deployment, with nearly double the generalisation gap~~ — struck; both
differences are inside seed noise.

**Consequence.** The backbone is fixed at EfficientNet-B0. Selection-optimism count stays
at **3 of 4**: this step was pre-registered and resolved to "keep what we had", so it
spent a run, not a degree of freedom. Reopening the ladder still needs a positive
argument logged before a run (DECISION-036), and the off-native confound (DECISION-038)
means this result does **not** license the claim that backbone scaling is exhausted in
general — only that this step, at 224, did not pay.

---

## DECISION-040 — `compare_arms` was silently dropping runs; comparisons are keyed on a label

- **Date:** 2026-08-25
- **Status:** Accepted — bug fixed, regression tests added
- **Deviates from proposal:** No.

`src/eval/compare_arms.py` built its table as `runs[metrics["arm"]] = r`. That is correct
for exactly as long as one arm means one run, which is how it was written and how it was
true until arm E was trained on a second backbone.

**Found by the user, from the symptom: the E row read 0.7615, which is the B0 run, after
B2 had been fetched.** Three runs of arm E collapsed to one, last write won, and
`sorted()` put B0 last. **Four of eleven runs were being dropped**, and the ranking was
then computed over the seven survivors. Nothing warned, because nothing counted.

This is the DECISION-024 shape again — a branch that was never wrong until the data grew
a dimension it did not model — and it would have hit stage 2 immediately, where three
seeds of arm E would have reported as one seed and the seed-stability claim would have
been built on a single run.

**The fix.** `label_runs()` keys each run on the arm **plus whichever of arch and seed
actually varies in the matched set**, so a homogeneous set still reads `A`, `B`, `E`,
while a mixed one reads `E/r18`, `E/b0`, `E/b2` and a seed sweep reads `E/s1`, `E/s2`,
`E/s3`. A label collision that still gets through falls back to the run directory name
and then raises — it never overwrites. `main()` asserts `len(runs) == len(dirs)` and
prints the label-to-directory mapping above every table, so a dropped run is impossible
to have and impossible to miss.

`--compare` now resolves labels rather than uppercasing arm letters, and lists the
available labels when a token is ambiguous.

**Four regression tests** cover the three-backbone case, the three-seed case, the
homogeneous case (the label must not get noisier when there is nothing to disambiguate),
and the indistinguishable-runs case.

**Everything reported before this fix used single-backbone sets**, where the arm letter
was unique and the label is identical, so no committed number changes. Verified by
re-running the stage 1 comparison after the fix.

---

## DECISION-041 — Stage 2 is a stability measurement, not a tie-break, and cannot become one

- **Date:** 2026-08-25
- **Status:** Accepted — **stated before the run**
- **Deviates from proposal:** No.

Written down in advance because the tempting claim after three seeds is exactly the one
the design does not support.

### What stage 2 can settle

- The **seed variance of the headline number**, so arm E's QWK is reported as a mean over
  seeds with a spread rather than one draw presented as a fact.
- Whether the **operating point** is stable — `sens@spec>=0.95` gates deployment and is
  the number closest to a floor.
- Whether arm E's unusually small **generalisation gap (0.034)** is a property of the arm
  or of one seed.
- Whether the **sign** of the A-vs-E difference is consistent across seeds — a
  descriptive fact, never a significance claim.

### What it cannot settle, at three seeds or thirty

**The A-vs-E tie.** Every seed is scored on **the same 5,268 validation images**.
Averaging over seeds reduces the training-stochasticity component of the uncertainty and
does **nothing** to the validation-set sampling component, which is common to all seeds
and is a fixed floor. The measured effect on B0 is **+0.0224 QWK** and the paired
interval's half-width is **about 0.027** — the effect is smaller than a floor that
seed-averaging cannot lower.

Worse for the tempting claim: properly accounting for seed variance makes the honest
interval **wider** than the single-seed paired CI, which understates uncertainty by
ignoring training noise altogether. **Stage 2's effect on the tie, if any, is to make it
more of a tie.**

Three seeds give a **spread, not an interval**. Non-overlapping ranges are not a
significance test and must not be written up as one.

### The rule this creates

**The reported number is the mean across seeds. Never the best seed.** Best-of-three
would be a configuration choice made on validation, against the soft budget of four with
a positive-argument gate (DECISION-036). Stage 2 is a measurement and spends none of that
budget — but only because this is fixed in advance.

Arm E is carried forward on the secondary criteria stated **before** this run — a better
operating point, a smaller gap, and an output already calibrated in grade units — and not
because it won a comparison. A and E are reported as a tie (DECISION-035).

### Design

Seeds **42, 43, 44**, sequential and obviously arbitrary. Seed 42 already exists for both
arms from stage 3, so stage 2 is **four new runs, not six** (~2.4 h rather than ~3.5 h).

---

## DECISION-042 — Arm E's 0.034 gap was a seed artefact, and it was load-bearing

- **Date:** 2026-08-25
- **Status:** Accepted — **corrects DECISION-036 and DECISION-039**
- **Deviates from proposal:** No.

Stage 2 measured arm E's train−val gap on EfficientNet-B0 across three seeds:

| arm | s42 | s43 | s44 | **mean** | range |
|---|---:|---:|---:|---:|---:|
| E | **0.034** | 0.098 | 0.097 | **0.0765** | 0.064 |
| A | 0.157 | 0.137 | 0.096 | **0.1300** | 0.061 |

**The honest figure to quote for arm E is a mean of 0.077 over three seeds, range
0.034–0.098** — never 0.034, which is the most extreme of the three and was the only one
we had. The seed-to-seed range (0.064) is larger than most of the between-arm differences
this project has been reasoning about.

### What this invalidates

**1. DECISION-036's central mechanism claim does not survive.** It said the two softmax
arms turned better features into more overfitting while the ordinal arm's gap *narrowed*,
and called that "same backbone change, opposite gap response". Recomputed:

| arm | gap on R18 (s42) | gap on B0 | change |
|---|---:|---:|---:|
| A | 0.092 | 0.130 (3-seed mean) | **+0.038, widened** |
| E | 0.073 | 0.077 (3-seed mean) | **+0.004, unchanged** |

Arm E's gap did **not** halve. It was flat. The surviving claim is weaker and must be
stated as such: **arm A's gap widened on B0 and arm E's did not, and arm E's gap is
consistently about 60% of arm A's.** The "opposite response" framing goes.

**2. DECISION-039's two supporting arguments both dissolve.** It preferred B0 over B2
partly because B2 was "2.5 points of sensitivity behind" and had "nearly double the
generalisation gap". Against the seed-resolved 224 baseline:

- B2's sens@spec≥0.95 of 0.7211 is **exactly B0's worst seed**, inside a 224 range of
  0.7211–0.7464. Not behind — indistinguishable.
- B2's gap of 0.061 is **lower** than B0's 3-seed mean of 0.077, not double it.

Both were single-seed noise on both sides of the comparison. **The decision to keep B0
still stands**, on the two reasons that do survive: the QWK difference was not separable
(+0.0052 [−0.0195, +0.0285]), and B0 is smaller, cheaper and native at 224. The
supporting arguments are struck.

**3. "Arm E's gap is 0.034, there is nothing to regularise" becomes weaker but survives.**
At 0.077 arm E still overfits far less than A (0.130) and B (0.329), so regularisation
remains the wrong prescription for the arm we are carrying forward. The number was wrong
by 2.2×; the conclusion was not.

### The rule this creates

**Every single-seed generalisation gap in this project is provisional and must be labelled
so.** The stage 1 gaps for arms B (0.329), C, C2, D and F are all one seed each and none
of them has been seed-resolved. **No gap may carry an argument unless it has been measured
across seeds.** Where a single-seed gap appears in a table it is marked `[1 seed]`.

This is the second time a single-seed number has been used as evidence (DECISION-036 was
the first). The cost both times was a claim that had to be withdrawn rather than a wrong
decision — but only by luck.

---

## DECISION-043 — What arm E beating arm A on every seed does and does not license

- **Date:** 2026-08-25
- **Status:** Accepted
- **Deviates from proposal:** No.

Stage 2, three seeds each, EfficientNet-B0:

| | E (s42/s43/s44) | mean | sd | A (s42/s43/s44) | mean | sd |
|---|---|---:|---:|---|---:|---:|
| held-out QWK | 0.7578 / 0.7548 / 0.7564 | **0.7563** | 0.0012 | 0.7346 / 0.7370 / 0.7426 | 0.7381 | 0.0034 |
| sens@spec≥0.95 | 0.7464 / 0.7211 / 0.7405 | **0.7360** | 0.0108 | 0.6822 / 0.7085 / 0.7172 | 0.7026 | 0.0149 |

Arm E is ahead on **all three seeds on both metrics**, the ranges do not overlap
(QWK by 0.0122, sensitivity by 0.0039), and E's seed variance is roughly a third of A's on
QWK.

### What may be written

> Arm E outperformed arm A on every seed on both metrics, with lower seed-to-seed
> variability (held-out QWK range 0.003 against 0.008). Arm E was selected on that basis
> together with the pre-stated secondary criteria.

That is a factual description of six runs and a **selection rationale**. It is defensible.

### What may NOT be written

**Not** "arm E significantly outperforms arm A", or any phrasing implying the ordering is
established for the population.

The reason is specific and worth stating in the viva. Uncertainty here has two components:

1. **Training stochasticity** — which seed. Stage 2 measures this, and it is small.
2. **Validation-set sampling** — which patients happen to be in the validation split.
   **All six runs share the same 5,268 images**, so this component is *identical* across
   them and does not average away. The paired bootstrap puts it at ±0.027 on the
   difference, against a seed-mean difference of +0.018.

Consistency across seeds addresses (1) and is **completely silent** on (2). Six
measurements of the same object with the same mis-calibrated ruler agree beautifully;
agreement is not accuracy. Non-overlapping ranges over three points are a **spread, not an
interval**, and cannot be converted into one.

**The A-vs-E tie therefore stands** (DECISION-035). What stage 2 changes is the *confidence
of the selection*, not the *status of the comparison*. Arm E is what we carry forward; we
have not shown it is the better arm.

The only thing that could settle it is the held-out test set, and it will be opened once,
for the selected model only (R3). Running both arms on test would be selection on test.

---

## DECISION-044 — 384px goes ahead as ONE pre-registered ablation, not as a floor fix

- **Date:** 2026-08-25
- **Status:** Accepted — **pre-registered before the run**
- **Deviates from proposal:** No. §5 already scopes 384 as an optional ablation "if GPU
  quota survives Phase 4". It has.

### The framing that matters most

**A 384 result cannot become the headline number.** The proposal fixes **224 for all
headline results** (CLAUDE.md §5). So this run cannot "fix" the sensitivity floor for the
reported system — adopting 384 as the headline would be a separate deviation needing
supervisor approval. What it can do is answer a real thesis question: **was input
resolution the binding constraint?**

The thesis should state plainly that **the system does not meet the screening floor at
224** — best sens@spec≥0.95 is 0.7360 (3-seed mean) against 0.80 — and report this
ablation as evidence about *why*, rather than presenting it as a rescue. That is a
stronger and more honest chapter than a floor chased and missed.

### What it manipulates

**Input resolution only: 224 → 384.** Everything else is held — arm E, EfficientNet-B0,
30 epochs, seed 42, same splits, same patients, same preprocessing pipeline and
parameters. The pixel count rises **2.94×** ((384/224)²), which is also the training-cost
multiplier.

This is a **data** change, not a model change: it requires the cache to be rebuilt at 384,
so the model sees genuinely more detail rather than an upsampled 224 image.

### The prediction

**Sensitivity improves but does not reach 0.80**, landing in roughly **0.76–0.80**, and
**QWK improves by less than sensitivity does.**

Reasoning: microaneurysms are on the order of 50 µm and a fundus image is ~3000 px across,
so at 224 they are sub-pixel to one pixel. They are the defining lesion of grades 1–2 —
which is exactly the referable boundary that `sens@spec≥0.95` measures. Resolution should
therefore move the *operating point* more than it moves overall ordinal agreement, which
is already dominated by the easy grade-0 mass.

### The stopping rule — stated against the 224 seed RANGE, not a point estimate

This is DECISION-042's lesson made structural. The 224 baseline is **not** 0.7464; it is a
three-seed range of **0.7211–0.7464, mean 0.7360**. A single 384 seed must clear the top
of that range to mean anything at all.

| 384 sens@spec≥0.95 (seed 42) | reading | action |
|---|---|---|
| **≥ 0.78** | clearly above the 224 range | resolution is a real lever. **Take it to the supervisor** as a headline-resolution deviation. Only *after* approval, run 3 seeds at 384. |
| **0.7464 – 0.78** | one seed above a three-seed range | **suggestive, not established.** Report as the ablation. Do not escalate, do not change the headline. |
| **< 0.7464** | inside or below the 224 range | clean negative result. Report and stop. |

**In no case** does this lead to 512px, to 384 on other arms, or to a 384 headline without
supervisor approval.

### Cost, and a correction to the figure previously quoted

**The ~2.5 h cache rebuild written in the handoff was wrong.** That number was the compute
*lost* to the first failed Phase 2 attempt (DECISION-021), not the build time. **The 224
cache build actually took 8.1 hours** at `--workers 4` (DECISION-022).

Preprocessing cost is dominated by decoding ~3000 px source JPEGs and the Ben Graham blur
at source scale; the output size barely affects it. So:

| item | cost |
|---|---|
| Cache rebuild at 384 + verify + publish | **~8–9 h**, one full Kaggle session |
| Storage | ~2.4 GB (224 cache is 0.836 GB × 2.94) |
| Reconcile + leakage gate | ~15 min |
| Train arm E at 384, 30 epochs | 76 s/epoch × 2.94 ≈ 224 s/epoch → **~2 h** |
| **Single-seed total** | **~11 h across two sessions** |
| If it wins and 3 seeds are approved | +4 h |

### Against what remains

Quota is 30 h/week and **the remaining phases barely touch the GPU**:

| phase | GPU |
|---|---|
| 5 — Grad-CAM | inference + overlays, **minutes** |
| 6 — external validation on APTOS | inference only, **minutes** |
| 7 — FastAPI + Next.js | local CPU, **none** |
| 8 — write-up | **none** |

**GPU quota is not the binding constraint; calendar time and attention are.** 11 h is
roughly one week's quota for a result that is informative either way, and nothing
downstream is blocked waiting on it.

### The positive argument required by DECISION-036

Selection-optimism count would reach **4 of 4** only if a 384 result were allowed to change
the headline model. Under this decision it cannot without a separate supervisor approval,
so the count **stays at 3**. The positive argument, written before the run as required:

1. It is the **only remaining untested lever with a physical mechanism** — lesion size
   against pixel size — rather than an architectural guess.
2. It is **already scoped in the proposal**, so it is not an invention.
3. The floor is a **deployment gate**, and a thesis that never tests the most-cited fix
   for it is weaker for the omission.
4. **A negative result is a genuine finding**: it would say the limit is data scale and
   label noise, not resolution.

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

# Session Handoff

**Session:** 2026-08-20 (session 2)
**Phase:** 2 — Preprocessing — **at the user sign-off gate**
**Next phase:** 2 (cache build) once the contact sheet is approved

---

## FIRST ACTION NEXT SESSION

**Run `notebooks/phase4_armc_backfill.py`** — arm C's re-run (~36 min) plus the
five-checkpoint backfill (~5 min), one session. It needs the Phase 4 ablation notebook's
**output** mounted as a third input, for the checkpoints.

After it: the operating-point table for all six arms, then stage 2.

## THE ROUTE TO THE SENSITIVITY GAP — REVISED after stage 3 (DECISION-036)

Best operating point is now **arm E on EfficientNet-B0: sens@spec>=0.95 = 0.7464**
against the 0.80 floor. The gap was 0.141 on ResNet18; it is **0.054**. Stage 3 closed
62% of it for 1.08x compute, which changes the priority order that was written here
before the run.

| lever | evidence | cost | priority |
|---|---|---|---|
| **Backbone quality** | +0.0408 held-out QWK [+0.0181, +0.0654], **separable**; sens +0.087 | 1.08x, already paid once | **1 — one more step (B2 @224, ~1 h)** |
| **Input resolution 224 -> 384** | none yet; the argument is physical, not measured | **~2.9x** compute + ~2.5 h cache rebuild + ~2.4 GB | 2 — after the backbone step |
| **Regularisation** | applies to the softmax arms, whose gaps WIDENED on B0 (A +0.065, B +0.039) | cheap | 3 — only if a softmax arm is kept; **arm E's gap is 0.034, there is nothing to regularise** |

Sequencing matters: **settle the backbone before spending seeds on it.** Seeding B0 and
then moving to B2 throws the seeds away.

## DO NOT WRITE UP — three unconfirmed items (DECISION-032)

The screening floor of **sensitivity >= 0.80 at specificity >= 0.95** is implemented and
printed, but it is marked PROVISIONAL everywhere it appears and **must not reach the
thesis** until the supervisor confirms:

1. **Whether UK DESP referable criteria map onto this project's `grade >= 2`.** DESP
   referable includes **maculopathy**, which these labels do not encode at all. *This is
   the one that matters most:* if they do not map, the benchmark is **indicative only**
   and the write-up has to say so plainly rather than implying compliance.
2. The precise NICE guideline number and clause — the 80/95 pair is corroborated by
   secondary sources; the primary document was not read.
3. IDx-DR's exact pre-specified endpoints. That it exceeded them is confirmed; the
   numbers commonly quoted (85% / 82.5%) are **not** — FDA DEN180001 returned 404. Cite
   the achieved 87.2 / 90.7 only.

`docs/EXPERIMENTS.md` states none of these, and `src/eval/thresholds.py` prints
`[PROVISIONAL]` beside the floor so it cannot be copied out as settled.

## What was done this session

**Phase 2 preprocessing, built and validated on real images, stopped at the gate.**

- `src/data/fetch_sample.py` — downloads named images from Kaggle without the 35.3 GB.
- `src/data/preprocess.py` — retina mask → square crop → Ben Graham (normalised
  convolution) → 224×224 → individual JPEGs. Plus `scan_quality()` and a parallel
  batch driver.
- `src/data/contact_sheet.py` — the QA renderer: original / cached / 3x detail.
- `tests/test_preprocess.py` (22) + `tests/test_preprocess_plumbing.py` (10).
  **Full suite: 70 green.**
- `docs/phase2_contact_sheet.png` — rendered from the 20 real QA images **and their
  actual cache files**. **Approved by the user 2026-08-20.**
- `src/data/reconcile_cache.py`, `notebooks/make_bundle.py`,
  `notebooks/phase2_build_cache.py` — the Kaggle build and its three gates.
- DECISION-009 … DECISION-016 logged.

**`code-reviewer` ran and returned CHANGES REQUIRED — 7 defects, all now fixed.** Two of
them (the "before" panel downsampled below the "after" panel; the "after" panel being an
in-memory recomputation rather than the cached JPEG) invalidated the first sheet, so the
version in commit `c948c1a` was never a valid gate artefact. The QA cache was rebuilt
after the connected-component fix and the sheet re-rendered from it. See PROGRESS.md for
the full table.

## MEASURED numbers — these replace the old [ESTIMATE]

| | Value |
|---|---|
| Mean output size | **21.4 KB/image** |
| EyePACS cache | **0.72 GB** (35,126) |
| APTOS cache | **0.08 GB** (3,662) |
| **Total** | **~0.80 GB**, a 44× reduction from 35.34 GB |
| Throughput | 0.94 s/image, 1 worker |
| Full build | **10.1 h @1 worker → 2.5 h @4 workers** |

The old 0.82 GB projection was accurate. It is now a measurement, not an estimate.

## Gotchas discovered this session (do not rediscover these)

1. **The kaggle CLI cannot download a single nested file.** It fails to URL-encode the
   path separators and gets 404 for every file in these datasets — including
   `trainLabels.csv/trainLabels.csv`, which it downloaded successfully in session 1, so
   this is a regression in kaggle 1.7.4.5, not a dataset quirk. Use
   `src/data/fetch_sample.py`, which calls the endpoint with `quote(path, safe="")`.
2. **Kaggle serves large files ZIP-WRAPPED.** Files above roughly 1 MB come back as a
   one-entry ZIP (`PK` magic) with `Content-Type: image/jpeg`; smaller files come back
   raw. Written straight to disk the large ones are ZIPs named `.jpeg`, `cv2.imdecode`
   returns None, and the byte count looks like a truncated download. `fetch_sample.py`
   sniffs the magic bytes.
3. **A large-sigma `cv2.GaussianBlur` is catastrophically slow.** sigma = width/10 means
   a ~1537-tap kernel and **30.8 s for one image**. See DECISION-011.
4. **Laplacian-variance focus is resolution-dependent.** It flagged 19 of 20 images as
   blurred. Must be measured at a normalised scale. See DECISION-013.
5. **`retina_mask` returns uint8 0/255, not bool.** Boolean-index with `mask > 0` —
   indexing a numpy array *with* a uint8 array is integer fancy-indexing and silently
   returns the wrong pixels rather than raising.
6. **The local venv had only numpy/pandas/pyyaml/pytest installed**, despite
   `requirements.txt` being "resolved" in session 1. The imaging subset (opencv, pillow,
   matplotlib, tqdm, kaggle) was installed at pinned versions this session; pytest was
   corrected from 9.1.1 to the pinned 8.4.2. **torch/timm are still NOT installed
   locally** — install them before any local smoke test or the Phase 7 demo.

## Regenerating the gate artefact

Both steps, in this order — the sheet reads the cache, so a stale cache means a stale sheet:

```bash
.venv/Scripts/python -m src.data.preprocess --manifest docs/phase2_qa_sample.csv     --src-root data/raw/qa --out-root data/processed/qa --stats docs/phase2_qa_stats.csv

.venv/Scripts/python -m src.data.contact_sheet --sample docs/phase2_qa_sample.csv     --src-root data/raw/qa --cache-root data/processed/qa     --out docs/phase2_contact_sheet.png --stats docs/phase2_qa_quality.csv
```

`--cache-root` is required and deliberately has no default (DECISION-015).

## The rim question — decided, DECISION-016

2.5% mask erosion, chosen by the user over Ben Graham's 0.9 r. **The residual was measured
and reported as asked, and it is not flattering:** a fixed annulus ratio moves only
1.610 → 1.516, and against depth from the actual edge the excess runs 2.61× at the
outermost 1% and reaches the interior level only by 10% of the radius. Erosion at 2.5%
removes ~63% of the peak excess and leaves a ≈1.6× edge.

**Do not reopen this on aesthetics.** The user's stated trigger for revisiting is Phase 5
Grad-CAM evidence that the model keys on the rim.

## The Kaggle build

`notebooks/phase2_build_cache.py` holds the exact cell. `--workers 4` is not optional:
single-threaded is 10.1 h and will not fit in a session. Set the accelerator to **None** —
this is CPU work and must not burn GPU quota.

The three auditor conditions are wired into that cell, not left to memory:
reconciliation (`src/data/reconcile_cache.py`), a re-run of `tests/test_no_leakage.py`
against the built artefact, and the measured size. It prints DO NOT PUBLISH if any fails.

**Already proven, before the build:** the 38,788 split rows map to 38,788 distinct cache
paths with **zero claimed by more than one split**, so the flat layout cannot alias an
image across the train/test boundary. That was the leakage risk in DECISION-012 and it is
now settled independently of whether the build succeeds.

## Phase 3 — the colour-order question is CLOSED

`tests/test_channel_order.py` settles it end to end and nobody needs to re-derive it:

- There is **no such thing as a BGR file.** `cv2.imwrite` takes a BGR array and writes a
  correct JPEG; the cache holds ordinary images.
- `src/data/dataset.py::load_cached_image` converts BGR→RGB **exactly once**, and its
  output is asserted **pixel-identical to PIL's**, an independent decoder sharing no code
  with cv2. One test deliberately shows the un-converted array failing that comparison, so
  the check is known to be capable of failing.
- Red dominance is tracked from a source file, through the real batch driver, into a real
  cache file, back through the real Dataset, to the tensor — on a synthetic fundus and on
  a **real EyePACS image**.
- timm's constants are RGB and channel-asymmetric, so a swap would shift red by ~0.35σ and
  blue the other way on every image forever. A test asserts the asymmetry itself, so if
  normalisation ever went symmetric the file's assumptions get revisited.

Ben Graham subtracts the local mean, so a flat disc cancels to grey 128 and carries no
hue. Colour assertions use `enhancement="none"`; testing hue through Ben Graham asserts
nothing.

## Watch out in Phase 4 — pretrained weights need internet

`model.pretrained: true` makes timm download weights from HuggingFace. **Kaggle notebooks
have Internet OFF by default**, and the cache build cell requires it off. The first
TRAINING notebook must either turn Internet on (Settings → Internet → On; needs a verified
phone number on the account) or mount the weights as a Kaggle Dataset. Discovering this at
the top of a training run costs the session.

## Open questions for the user

1. **Kaggle build** — the user launches it; see "The Kaggle build" above.
2. **GitHub** — private repo + collaborator access for Ameena Ahmed and Muhammad Ali
   Abdullah. **Do not push without asking.** No remote is configured. User previously
   mentioned distributing as an archive instead; confirm which.
3. ~~**Supervisor** — written confirmation of DECISION-001 before Phase 7.~~ **DONE 2026-08-24 — approved.**
4. **Arm F vs Phase 6** — unchanged from session 1; decide when the numbers exist.

## Known-broken / not yet built

- `src/models/`, `src/train/`, `src/eval/`, `src/xai/`, `src/inference/` — empty.
- No Kaggle notebook has been run; **GPU quota fully unconsumed.**
- `docs/EXPERIMENTS.md` does not exist (no runs).
- torch 2.9.1+cpu / torchvision 0.24.1+cpu / timm 1.0.28 are now installed locally, so the
  data layer is testable here. The pins are still unverified against the Kaggle image —
  the first notebook must print its versions and the pins be updated to match.
- The CLAHE-on-green ablation path is implemented but has never been run.

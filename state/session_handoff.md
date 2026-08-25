# Session Handoff

**Last updated:** 2026-08-25
**Phase:** 4 — ablation, near the end. Stages 1, 3 and 3.5 done; **stage 2 is next.**
**HEAD:** `3cc6982`. Working tree clean; every number below traces to a committed
`runs/<run_id>/metrics.json` (R4).

---

## FIRST ACTION NEXT SESSION

**Run `notebooks/phase4_stage2.py`** — arms E and A on EfficientNet-B0, seeds 43 and 44,
**four new runs, ~2.4 h**. Cell 1 by hand, cells 2–4 by commit.

Inputs: `fyp-dr-eyepacs-224`, `fyp-dr-code`, **and the stage 3 notebook's output** (cell 3
needs the seed-42 runs to compare against; without it cell 3 says so and prints the local
command instead of failing). GPU T4 ×2, Internet **ON**.

Rebuild the code bundle first — `compare_arms.py`, `smoke.py`, `gen_experiments.py` and
two notebooks have all changed since the last bundle.

**Stage 2 is a MEASUREMENT, not a tie-break (DECISION-041).** It cannot break the A-vs-E
tie at three seeds or thirty: every seed is scored on the same 5,268 validation images, so
seed-averaging lowers the training-noise component and does nothing to the
validation-sampling floor. The effect is +0.0224 QWK against a paired half-width of
~0.027. **Report the mean across seeds, never the best seed** — best-of-three would be a
selection on validation.

After stage 2: **the 384px decision on arm E alone**, then Phase 5 (Grad-CAM).

---

## WHERE THE PROJECT ACTUALLY STANDS

**Best model: arm E (ordinal regression head) on EfficientNet-B0, seed 42.**

| | value |
|---|---|
| held-out matched QWK | **0.7560** |
| sens @ spec ≥ 0.95 | **0.7464** (floor 0.80 — **not met**) |
| spec @ sens ≥ 0.80 | 0.9030 (floor 0.95 — not met) |
| train−val gap | 0.034 |

**No arm reaches the screening floor.** That gap — 0.054 of sensitivity — is the open
problem going into the rest of the project.

The full table is `python -m src.eval.compare_arms --pattern 'phase4_*' --operating-point`
(~2 min). Rankings come from the **held-out** column only, never the as-run column
(DECISION-035).

### The route to the sensitivity gap — priorities as they now stand

| lever | status |
|---|---|
| **Backbone quality** | **CLOSED.** B0 → B2 was +0.0052 [−0.0195, +0.0285], not separable, and B2 was *worse* on the operating point. Backbone fixed at B0 (DECISION-039). |
| **Input resolution 224 → 384** | **The remaining live lever.** No evidence yet; the argument is physical (microaneurysms are a few pixels at 224). Costs ~2.9× compute **plus** a ~2.5 h cache rebuild at ~2.4 GB. Scoped as an optional ablation in the proposal (DECISION-005), so it is a deviation to approve rather than invent. |
| **Regularisation** | Applies only to the softmax arms, whose gaps *widened* on B0. **Arm E's gap is 0.034 — there is nothing there to regularise.** |

---

## STANDING RULES THAT ARE EASY TO BREAK

1. **Matched decision rule, always (DECISION-035).** No ranking claim may compare arms
   scored with different numbers of free parameters. Softmax heads score on the expected
   grade `Σ pᵢ·i`, not argmax. Every arm gets four fitted cut points. Differences carry a
   paired bootstrap interval; one spanning zero is **a tie**. The as-run column stays but
   **never ranks**.
2. **Selection optimism is a budget: 3 of 4 spent (DECISION-036).** Cut-point optimism is
   measured (0.006–0.010 QWK, re-measured every run); **architecture-selection optimism is
   not measured and is not estimable** without a second held-out split we do not have. A
   fourth selection needs a positive argument written down *before* the run. The running
   total is a table in `docs/EXPERIMENTS.md`.
3. **Pre-register before you run.** Stage 3's prediction was half wrong *and* tested the
   wrong variable — it was labelled a capacity test when B0 is smaller than ResNet18 on
   both parameters (4.01M vs 11.18M) and FLOPs (0.385 vs 1.814 GMAC). State the
   manipulated variable in measured units, and state what result would make you stop.
4. **The test set has not been touched** (R3) and stays untouched until Phase 5.

---

## DO NOT WRITE UP — three unconfirmed items (DECISION-032)

The screening floor of **sensitivity ≥ 0.80 at specificity ≥ 0.95** is implemented and
printed, but is marked PROVISIONAL everywhere and **must not reach the thesis** until the
supervisor confirms:

1. **Whether UK DESP referable criteria map onto this project's `grade >= 2`.** DESP
   referable includes **maculopathy**, which these labels do not encode at all. *This is
   the one that matters most:* if they do not map, the benchmark is **indicative only**
   and the write-up must say so plainly rather than implying compliance.
2. The precise NICE guideline number and clause — the 80/95 pair is corroborated by
   secondary sources; the primary document was not read.
3. IDx-DR's exact pre-specified endpoints. That it exceeded them is confirmed; the
   commonly quoted 85% / 82.5% are **not** — FDA DEN180001 returned 404. Cite the
   achieved 87.2 / 90.7 only.

`docs/EXPERIMENTS.md` states none of these, and `src/eval/thresholds.py` prints
`[PROVISIONAL]` beside the floor so it cannot be copied out as settled.

---

## THE RECURRING FAILURE MODE — read this before writing a notebook

Six sessions have now lost time to the same shape: **glue code whose first execution is
the real one.** DECISION-021 (archive deleted before verification), -022 (missing
subcommand after an 8.1 h build), -023 (Kaggle auto-extracted the archive), -024 (a config
branch never exercised), -025 (a str subclass SafeDumper would not serialise), -028
(import-time file reads), -040 (`compare_arms` silently dropping runs once one arm meant
several runs).

The countermeasures exist — **use them**:

- `python -m src.data.notebook_check --all --self-test` validates every notebook command
  line against the real parser. Run it before pasting anything.
- `python -m src.train.smoke --arm X --arch Y` runs a whole arm end to end on a synthetic
  fixture, on CPU, in ~15 s. **Pass `--arch` when the run changes backbone** or the smoke
  exercises the wrong path.
- `tests/test_no_leakage.py` runs before every training run, and again inside it.

---

## Gotchas that are still live

1. **Kaggle notebooks have Internet OFF by default** and `model.pretrained: true` makes
   timm download from HuggingFace. Turn it on, or a training run dies at minute zero.
2. **Kaggle caps notebook output at 500 files** and **auto-extracts published archives.**
   `resolve_cache()` finds the cache by shape rather than by path for this reason.
3. **`/kaggle/working` is wiped between sessions.** Anything needed later must be fetched
   (`src/data/fetch_run.py`) or published.
4. **`retina_mask` returns uint8 0/255, not bool.** Index with `mask > 0` — indexing a
   numpy array *with* a uint8 array is integer fancy-indexing and silently returns the
   wrong pixels.
5. **Split CSVs carry a `#` provenance header.** Every reader needs
   `pd.read_csv(..., comment='#')`.
6. **The kaggle CLI needs 2.2.4+**, and a legacy `~/.kaggle/kaggle.json` shadows newer
   OAuth credentials — `kaggle auth login` plus renaming that file is what fixed it.
   Kernel slugs so far are `rah098/fyp-dr-phase3-baseline` and the Phase 4 ones.

---

## Open questions for the user

1. **GitHub** — private repo + collaborator access for Ameena Ahmed and Muhammad Ali
   Abdullah. **Do not push without asking.** No remote is configured. The user previously
   mentioned distributing as an archive instead; confirm which.
2. **384px** — needs approval before it is spent (~2.9× compute + ~2.5 h cache rebuild).
3. **Arm F vs Phase 6** — arm F's pooled-training result was a null on the target domain
   (DECISION-031). Decide what Phase 6 external validation looks like given that.

## Not yet built

- `src/xai/` (Grad-CAM) — Phase 5.
- `backend/`, `frontend/`, `src/inference/predictor.py` — Phase 7. DECISION-001
  (FastAPI + Next.js over Streamlit) is **approved** by the supervisor as of 2026-08-24.
- The CLAHE-on-green ablation path is implemented but has never been run.

# Session Handoff

**Last updated:** 2026-08-25
**Phase:** 4 — ablation. **Stages 1, 2, 3 and 3.5 are all done. The 384px ablation is
the last item in Phase 4.**
**HEAD:** `0b13e9e` + this session's corrections. Every number below traces to a committed
`runs/<run_id>/metrics.json` (R4), and every one that is a **3-seed mean says so**.

---

## FIRST ACTION NEXT SESSION

**Phase 7 — the web app.** Everything experimental is finished. Start with the two pieces
already decided: the **coverage guard** that flags an upload whose framing falls outside
the training range (DECISION-060 — it reuses `region_masks` from Phases 5/6), and the
**four-point disclaimer**. Then the upload -> predict -> Grad-CAM flow.

## THE REMEDY IS COMPLETE AND THE OUTCOME IS AMBIGUOUS (DECISION-067)

|  | APTOS | EyePACS | direction | G4 fires |
|---|---|---|---|---|
| baseline | 0.2360 | 0.1367 | 3/3 | 3 of 4 rules |
| remedied | 0.1853 | 0.1598 | 1/3 | 0 of 4 rules |

**DO NOT UPGRADE THIS TO "THE REMEDY WORKED."** Three reasons, all pre-registered:

1. The WORKS clause required APTOS <= EyePACS. It is 0.1853 vs 0.1598. Not met.
2. **The gap narrowed partly from the WRONG END** — APTOS fell 0.0507 but EyePACS ROSE
   0.0230, so about a third of the narrowing is the IN-DOMAIN dependence getting stronger.
   A clean remedy would not do that. This is the most important qualifier.
3. The 0.0507 drop is inside the baseline's own 0.194 across-seed range.

**What is true and worth saying:** every marker that does not rest on magnitude moved as
predicted (direction 3/3 -> 1/3; 0 of 4 rules fire), at **no in-domain cost** — mean val
QWK 0.7570 against 0.7570. That is WEAK CAUSAL SUPPORT for the shortcut reading, not none
and not proof.

**A defect in my own pre-registration is recorded in DECISION-067.** The notebook glossed
the WORKS clauses as "(i.e. G4 no longer fires)", which is wrong: G4 fires only if BOTH
|r|>=0.2 AND APTOS>EyePACS, so it stops firing when EITHER fails, while WORKS demanded
BOTH fail. Under the literal G4 definition the criterion does NOT fire. **Resolved as
AMBIGUOUS — the stricter reading**, because an internally ambiguous pre-registration must
be read against interest or it is worth nothing.

**The secondary endpoint was never tested.** "Loses less on APTOS" needs the carried-over
evaluation; what cell 4 prints is mean predicted score, a proxy that cannot distinguish
better calibration from worse sensitivity. Optional ~18 min re-run of Phase 6 with the
remedied checkpoints completes the registered plan. **It cannot change the primary.**

**Phase 6's verdict is unchanged.** The thesis reports the BASELINE model; G4 fired on it
and DECISION-060 stands. The remedy is a follow-up about why, not a new headline.

## THE SPLITS STILL VANISH MID-RUN (DECISION-064)

Version #3: **all six** CSVs gone between cell 3 and cell 4, `still present: []`. The
restore fired, every hash matched cell 1, and the run completed. Version #2's `val.csv`
error was just the first file cell 4 asked for.

Verified byte-identical local == bundle == Kaggle, so nothing is corrupted in transit —
they exist, then they are gone. Nothing in `src/` deletes anything. **Cause still open.**
The next run prints a repo-integrity probe (does `src/` vanish too?) which distinguishes
"the whole copied tree is pruned" from "specific to data/splits". Report what it prints.

Cell 2 of the remedy now checks before **every seed** — that cell is 2.1 h across three
subprocesses and each re-reads the splits at startup.

## THE REMEDY IS BUILT AND PRE-REGISTERED — DO NOT REDESIGN IT MID-RUN

One knob: `augment.surround_randomisation: 0.75` (`configs/remedy_surround.yaml`). Cell 1
asserts exactly one config key differs from the baseline.

**Why appearance and not extent** — both were measured first (DECISION-063). The existing
affine ALREADY moves coverage over 0.611-0.864 (sd 0.075), wider than the whole Phase 5
erosion sweep, so an extent remedy would add nothing. The surround stays black in 96% of
draws, so appearance is genuinely absent. **The first probe of this got it backwards** by
indexing the surround with the unrotated mask; the corrected one warps an indicator
channel, which is how the augmentation itself works.

**Every outcome is reported, whichever way it goes.** That was the condition for running
it. A FAILED remedy is the more interesting result — it falsifies the shortcut
interpretation, and nothing else in this project can. Do not quietly reframe a null.

**The test is underpowered and that is on the record.** Baseline APTOS |r| ranged 0.194
across seeds; a drop to ~0.15 is inside that range and proves nothing. The decisive
readings are the DIRECTION count (3/3 -> at most 1/3) and agreement across all four
aggregation rules — not the magnitude. Ambiguous is reported as ambiguous.

**384 is deliberately not run** (DECISION-062). Write it up as skipped, with the reason.
Never as a null result — it was never run, so nothing is known about it.

## THE HEADLINE CONCLUSION CHANGED

**The model does not generalise beyond its training population** (DECISION-060).

**Read that precisely — performance did not collapse.** G1, G2 and G3 all passed
comfortably. It fired on **G4 alone**: a demonstrated dependence on retinal framing,
stronger out-of-domain than in-domain. DECISION-057 fixed in advance that this invalidates
the generalisation claim regardless of the headline numbers.

**And the claim is narrower than "the model uses a shortcut"** (DECISION-059): present in
**2 of 3 seeds**, r² 6-11% where present and 1.8% in the third, and the
**seed-aggregation rule was never pre-registered** — the code used the mean. It fires under
mean, median and majority; not under unanimity. Best characterised as a property of the
**training procedure**, which sometimes produces a framing-dependent model.

**The methodological finding is the strongest part.** Phase 5 tested the same hypothesis
in-domain and found it unsupported (0.0075 grade units). Only the between-dataset
comparison exposed it. Single-dataset shortcut analysis is blind to exactly the
dependencies that matter for deployment.

**Phase 7 must carry** the four specific disclaimer points AND a coverage guard that flags
uploads whose framing falls outside the training range — a control, not just a warning.

---

## WHERE THE PROJECT ACTUALLY STANDS

**Best model: arm E (ordinal regression head) on EfficientNet-B0**, three seeds.

| | value |
|---|---|
| held-out matched QWK | **0.7563** (3 seeds: 0.7578/0.7548/0.7564) |
| sens @ spec ≥ 0.95 | **0.7360** (3 seeds: 0.7464/0.7211/0.7405) — floor 0.80 **not met** |
| spec @ sens ≥ 0.80 | 0.9030 `[1 seed]` (floor 0.95 — not met) |
| train−val gap | **0.077** (3 seeds: 0.034/0.098/0.097) |

**Quote the 3-seed means, never seed 42 alone.** The 0.034 gap and the 0.7464 sensitivity
are the most extreme of their three seeds and both were used as arguments before stage 2
existed (DECISION-042).

**No arm reaches the screening floor.** That gap — **0.064** of sensitivity on the
3-seed mean — is the open
problem going into the rest of the project.

The full table is `python -m src.eval.compare_arms --pattern 'phase4_*' --operating-point`
(~2 min). Rankings come from the **held-out** column only, never the as-run column
(DECISION-035).

### The route to the sensitivity gap — priorities as they now stand

| lever | status |
|---|---|
| **Backbone quality** | **CLOSED.** B0 → B2 was +0.0052 [−0.0195, +0.0285], not separable, and B2 was *worse* on the operating point. Backbone fixed at B0 (DECISION-039). |
| **Input resolution 224 → 384** | **The remaining live lever, and it is next.** Pre-registered with a stopping rule (DECISION-044). Costs 2.94× training (~2 h) **plus an ~8–9 h cache rebuild** — the "~2.5 h" written here before was wrong, it was the compute lost to the failed first Phase 2 attempt, not the build time. **A 384 result cannot become the headline** without a supervisor deviation; the proposal fixes 224. |
| **Regularisation** | Applies to the softmax arms: arm A's gap widened on B0 (0.092 → 0.130) while arm E's held flat (0.073 → 0.077). At 0.077 against A's 0.130 and B's 0.329, **arm E still overfits far less than the arms it is carried over**, so this stays the wrong prescription for it. |

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

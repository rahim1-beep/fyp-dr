# Session handoff

Last updated 2026-09-09. Overwritten each session — this is the state of play, not a log.

---

## FIRST ACTION NEXT SESSION

**Start a FRESH session for the frontend.** Do not continue in the session that produced
this file: it is 59% full, and the design and verification work is screenshot-heavy.

Everything the new session needs is committed and readable without any prior context:

| file | what it is |
|---|---|
| `CLAUDE.md` (frontend section) | the binding rules — read every session |
| `docs/api-contract.md` | the API, read off the source, authoritative |
| `fixtures/expected/*.json` | recorded responses for all 7 UI states |
| `fixtures/README.md` | what each fixture triggers, measured |

In the new session, ask for a **design plan and stop for approval before any code**. That
is where generic output gets caught, and it is cheap to redirect there.

To actually run the app you need seed 42's `best.pth` locally — gitignored, download once
from the `fyp-dr-phase4-stage3` notebook output:

    set FYP_CHECKPOINT=<path to best.pth>
    .venv\Scripts\python -m uvicorn backend.app:app --reload   # :8000

**No Kaggle work is outstanding.** Every experimental phase and the calibration are done.

---

## What happened this session

1. **Phase 6 closed.** The remedy ran to completion on all three seeds. Outcome
   **AMBIGUOUS** (DECISION-067) — see below, it is the thing most likely to be
   misreported.
2. **Phase 7 built:** coverage guard (calibrated on Kaggle, DECISION-068/070),
   `src/inference/predictor.py` (21 tests), FastAPI backend (12 tests).
3. **Frontend handoff pack** (DECISION-071): `CLAUDE.md` frontend section,
   `docs/api-contract.md`, 6 verified fixtures, recorded responses for every state.

**702 passed, 2 skipped.** Working tree clean.

---

## Things that are easy to get wrong

### The remedy outcome is AMBIGUOUS and stays that way (DECISION-067)

|  | APTOS | EyePACS | direction | G4 fires |
|---|---|---|---|---|
| baseline | 0.2360 | 0.1367 | 3/3 | 3 of 4 rules |
| remedied | 0.1853 | 0.1598 | 1/3 | 0 of 4 rules |

**Do not upgrade this to "the remedy worked."** Three pre-registered reasons: the WORKS
clause required APTOS ≤ EyePACS and it is 0.1853 vs 0.1598; **the gap narrowed partly from
the wrong end** — APTOS fell 0.0507 but EyePACS *rose* 0.0230, so about a third of the
narrowing is the in-domain dependence getting stronger; and the drop sits inside the
baseline's own 0.194 seed range.

What is true: every marker that does not rest on magnitude moved as predicted, at **no
in-domain cost** (QWK 0.7569 either way). That is weak causal support, not proof.

### The interface must never show 0.7615 or 0.7464

Seed 42 ships and it is the **best-scoring** of the three seeds (DECISION-069). The
interface reports the 3-seed means — **0.7569** and **0.7360** — only. Two tests enforce
it (`test_api.py::test_meta_reports_the_three_seed_mean_and_not_the_shipped_seed`,
`test_predictor.py::test_the_interface_quotes_the_three_seed_mean_not_the_shipped_seed_s_own_score`).
Both assert `"0.7360" in text and "0.7464" not in text`. Do not "simplify" the manifest by
deleting one of the two pairs — having them side by side is what makes the choice
auditable.

### grade and referable disagree by design

`referral_threshold` 0.9488 sits **below** the grade-1→2 cut point 1.6216, so any score in
that band is `grade: 1` "Mild" **and** `referable: true`. Correct behaviour. Do not
reconcile them.

### The coverage guard is narrower than it looks (DECISION-070)

It flags 4.8% of APTOS against 2% in-domain — right direction — but **95.2% of APTOS
passes**, even though APTOS is where G4 fired. A per-image check cannot see a shift in the
median. Lead with that limitation; it is future work, not a fix.

### Fixtures are TRAINING images

The only real fundus photographs on this machine are the 20 QA images, all from the
**train** split. Fine for exercising UI states, which is all they are for. **Never
screenshot one as a demo of performance** — that would be reporting a training-set result.

### The splits still vanish mid-run on Kaggle (DECISION-064)

Cause unknown. Cell 1 of the Phase 6/7 notebooks snapshots them with hashes and restores
if they disappear. It did not fire in the last two runs. If it fires, report it.

---

## Open questions for the user

1. **Docker Compose** is still on the Phase 7 checklist. Worth it for the viva demo, or is
   the two-terminal path enough? It was not attempted this session.
2. **The CPU inference benchmark** on a 16 GB target is unstarted. Needed for the
   write-up, or drop it?
3. **The optional ~18 min Phase 6 re-run** with the remedied checkpoints would complete
   DECISION-051's secondary endpoint ("loses less on APTOS"). It **cannot** change the
   ambiguous primary. Worth the quota or not?

## Known broken / incomplete

- **Two fixtures cannot be built here** — a clean grade 0 and the real disagreement case
  need `best.pth` plus validation images. `build_fixtures.py` makes them once
  `FYP_CHECKPOINT` is set. A synthetic stub covers the disagreement state meanwhile.
- **`frontend/` is empty.** Next.js is not scaffolded.
- **No Docker.** Nothing containerised yet.
- `context7` MCP server failed to connect this session (timeout) — unrelated to the repo.

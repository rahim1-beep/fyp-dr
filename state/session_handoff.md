# Session Handoff

**Session:** 2026-08-20 (session 1)
**Phase:** 1 — Data foundation — **COMPLETE**
**Next phase:** 2 — Preprocessing

---

## What was done this session

**Phase 0 — Orientation.** Read `BOOTSTRAP.md` and `docs/proposal.pdf`. Audited the
environment. Produced the plan, assumptions, failure-mode defences, and questions F1–F6.
**User approved**; answers recorded as DECISION-001…005.

**Phase 1 — Data foundation. Complete.**
- Repo skeleton, `.gitignore`, `CLAUDE.md`, `PROGRESS.md`, 9 subagents, 4 slash commands.
- `.venv` on Python 3.12.10. `requirements.txt` pinned for cp312, dry-run resolved.
- Configs: `base`, `local`, `kaggle`, `arm_a`…`arm_f`.
- Labels CSV located, schema discovered, full reconciliation — **zero mismatches**.
- APTOS distribution measured from labels CSVs only (DECISION-007).
- **Patient-level 70/15/15 split generated, seed 42.**
- `tests/test_no_leakage.py` — **38 tests green.**
- Reviewed by `leakage-auditor` — **PASS** (verdict + actions in DECISION-008).
- `src/data/manifest.py` — the only supported reader for split CSVs.

## Auditor findings that changed the code

The audit passed but surfaced four things worth fixing, all done before commit:

1. **`assign_splits` had no minimum-per-split guard** — a stratum with n < 4 patients
   silently produced an empty val *and* test (all-zero confusion-matrix column). Now
   raises. Verified to fire.
2. **The venv was off-spec** — splits were first generated under `numpy 2.5.2`/`pandas
   3.0.5` against pins of `2.2.6`/`2.3.3`. Venv corrected, splits regenerated; assignments
   byte-identical, only the header changed. **If you rebuild the venv, install from
   `requirements.txt`, not ad hoc.**
3. **Split headers now record `python/numpy/pandas` versions** — numpy's Generator stream
   is not stable across versions (NEP 19).
4. **The test suite checked totals against a hardcoded integer**, so a complete-but-wrong
   manifest would have passed. Now reconciles against `trainLabels.csv` and rules out
   min-grade stratification rather than merely asserting max-grade.

## Key technique: reconciliation without downloading

`kaggle datasets files` returns filenames and sizes without transferring image bytes.
Paging the full listing gives a complete remote manifest, so CSV↔disk reconciliation runs
on a laptop that will never hold the 35.3 GB dataset. Only the 465 KB labels CSV was
downloaded. Same trick used for APTOS.

Regenerate the listings (gitignored under `data/raw/`):

```
.venv\Scripts\python -m src.data.list_remote --slug dreamer07/eyepacs --out data/raw/eyepacs/remote_listing.csv
.venv\Scripts\python -m src.data.list_remote --slug mariaherrerot/aptos2019 --out data/raw/aptos/remote_listing.csv
kaggle datasets download dreamer07/eyepacs -f "trainLabels.csv/trainLabels.csv" -p data/raw/eyepacs --unzip
kaggle datasets download mariaherrerot/aptos2019 -f train_1.csv -p data/raw/aptos --unzip   # also valid.csv, test.csv
```

## Gotchas discovered (do not rediscover these)

1. **EyePACS labels live at `trainLabels.csv/trainLabels.csv`** — a *directory* named
   `trainLabels.csv` containing a file of the same name.
2. **EyePACS CSV `image` values carry no extension** (`10_left`) while disk files do
   (`10_left.jpeg`). Joining on the raw filename matches **zero rows**.
3. **APTOS images are `.png`**, double-nested: `train_images/train_images/{id}.png`.
4. **Split CSVs carry a `#` provenance header** — every reader must pass
   `pandas.read_csv(..., comment='#')` or the header parses as data.
5. **Two opencv distributions are unavoidable.** `albumentations` needs
   `opencv-python-headless`, `grad-cam` needs `opencv-python`; unpinned, pip resolves them
   to different major versions. Both pinned to 4.12.0.88.
6. **pandas pinned to 2.3.3, not 3.x** — 3.0 changed copy-on-write and string dtypes.
7. **Custom subagents in `.claude/agents/` are not available until Claude Code restarts.**
   They were created this session, so the auditor ran as a general-purpose agent with the
   brief inlined. Next session they should be selectable by name.

## FIRST ACTION NEXT SESSION

**Confirm the 9 subagents in `.claude/agents/` dispatch by name.** They were created
mid-session and Claude Code only loads agent definitions at startup — last session the
`leakage-auditor` dispatch failed with "Agent type not found" and had to run as a
general-purpose agent with the brief inlined. The user restarted specifically to fix this.
Verify before relying on delegation, and report it to the user.

## Exact next command

Phase 2 preprocessing. **Phase 2 is approved.** Nothing heavy runs locally — no CUDA
device, and the images are on Kaggle.

The QA sample is **already selected and committed**: `docs/phase2_qa_sample.csv`
(20 images, seed 42, all TRAIN split, 17.1 MB to download). Regenerate with
`python -m src.data.qa_sample` if needed.

Next artefacts to write:
1. `src/data/preprocess.py` — circle-crop (non-black bbox on blurred grayscale threshold)
   → Ben Graham → 224×224 → individual JPEGs.
2. `scan_quality()` — pixel statistics (mean brightness, saturation, disc centroid offset)
   to find genuinely dark / over-exposed / off-centre images. **File size alone cannot
   identify those**; the current poor-quality picks are size-based proxies and should be
   confirmed or improved by this scan.
3. The contact sheet renderer.

**Contact sheet requirements are in `PROGRESS.md` under Phase 2 — read them.** Summary:
side-by-side at equal display size, patient ID + grade labelled on each pair, ≥6 of ~20
from grades 1–2 (current sample has 9), 2–3 deliberately bad inputs (current sample has 3).

### HARD GATE

**Do not cache all 35,126 images until the user signs off on the contact sheet.**

### Cache persistence — confirmed with the user

`/kaggle/working/processed/` during the run → then published as a **private Kaggle Dataset**
under account `rah098` so it survives the session wipe → re-mounted read-only at
`/kaggle/input/fyp-dr-eyepacs-224` for training (already wired as
`paths.processed_dataset_mount` in `configs/kaggle.yaml`).

**Report the measured cache size before uploading.** Projection below is an estimate only —
measure the real per-image mean on the 20 QA images first, then project.

## Expected cache size — **[ESTIMATE]**, not measured

No image has been preprocessed yet, so this is a projection and **must be labelled
`[ESTIMATE]` wherever it appears** (R4) until replaced by a measurement.

A 224×224 RGB JPEG at quality 95 typically lands at 15–30 KB.

| | Images | @15 KB | @22 KB (central) | @30 KB |
|---|---:|---:|---:|---:|
| EyePACS | 35,126 | 0.50 GB | **0.74 GB** | 1.00 GB |
| APTOS | 3,662 | 0.05 GB | **0.08 GB** | 0.10 GB |
| **Total** | 38,788 | 0.55 GB | **~0.82 GB** | 1.11 GB |

Source is 35.34 GB, so the cache should be roughly a **40× reduction**. Comfortably inside
`/kaggle/working`'s cap (`warn_working_dir_gb: 15` in `configs/kaggle.yaml`) and inside
Kaggle Dataset limits.

**Replace this with a measured number before uploading**, projected from the actual mean
output size over the 20 QA images.

## Open questions for the user

1. **GitHub** — private repo + collaborator access for Ameena Ahmed and Muhammad Ali
   Abdullah. **Do not push without asking.** No remote is configured yet. *(User has since
   said they are distributing as an archive, so this may be moot — confirm.)*
2. **Supervisor** — written confirmation of DECISION-001 (Streamlit → FastAPI + Next.js)
   before Phase 7.
3. **Arm F vs Phase 6** — if Arm F wins, APTOS cannot be the external validation set for
   it. The trade-off is logged (DECISION-007); the choice is the user's to make when the
   numbers exist.

## Known-broken / not yet built

- `src/data/{preprocess,dataset,sampler}.py` — not written.
- `src/models/`, `src/train/`, `src/eval/`, `src/xai/`, `src/inference/` — empty.
- No Kaggle notebook has been run; **GPU quota fully unconsumed.**
- `docs/EXPERIMENTS.md` does not exist (no runs).
- Torch pins unverified against the Kaggle image — the first Kaggle notebook must print its
  versions and the pins updated to match (note in `requirements.txt`).
- Git identity is repo-local `syedrahim079 <syedrahim079@gmail.com>`; user confirmed single
  authorship is intended.

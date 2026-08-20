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

## Exact next command

Phase 2 preprocessing. Nothing runs locally — no CUDA device, and the images are on Kaggle.

The next artefact to write is `src/data/preprocess.py` (circle-crop → Ben Graham →
224×224 → individual JPEGs), then `notebooks/02_preprocess.ipynb` to run it on Kaggle.

**Before caching all 35,126 images, produce the contact sheet of ~20 before/after pairs
spanning all five grades and get user approval** (Phase 2 acceptance criterion).

Also required in Phase 2:
- Report the cache size **before** uploading (user requirement F3).
- Persist the cache as a **private Kaggle Dataset** under account `rah098`; log in
  `DECISIONS.md`.

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

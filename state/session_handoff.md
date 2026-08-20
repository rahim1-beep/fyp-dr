# Session Handoff

**Session:** 2026-08-20 (session 1)
**Phase:** 1 — Data foundation
**Status:** Paused at the user sign-off gate before split generation.

---

## What was done this session

**Phase 0 — Orientation.** Read `BOOTSTRAP.md` and `docs/proposal.pdf`. Audited the
environment. Produced the plan, assumptions, failure-mode defences, and questions F1–F6.
**User approved**, answers recorded in `docs/DECISIONS.md` as DECISION-001…005.

**Phase 1 — Data foundation (partial).**
- Repo skeleton per `BOOTSTRAP.md` §7; `.gitignore`; `CLAUDE.md`; `PROGRESS.md`.
- `.venv` built on Python 3.12.10 (`py -3.12`). System default 3.14 untouched.
- `requirements.txt` pinned for cp312, **dry-run resolved successfully** on Windows.
- 9 subagents in `.claude/agents/`, 4 slash commands in `.claude/commands/`.
- Config system: `configs/{base,local,kaggle,arm_a..arm_e}.yaml`.
- **Labels CSV located and the full reconciliation report produced — all assertions PASS.**

## Key finding: reconciliation ran locally without downloading EyePACS

`kaggle datasets files` returns every filename and size without transferring image bytes.
Enumerating all 176 pages (35,127 entries) gave a complete remote manifest, so the full
CSV↔disk reconciliation ran on a laptop that will never hold the 35.3 GB dataset. Only the
465 KB labels CSV was downloaded.

The listing is cached at `data/raw/eyepacs/remote_listing.csv` (2.1 MB). It is **gitignored**
(`data/raw/` rule), so a fresh clone will not have it. Regenerate with:

```
.venv\Scripts\python -m src.data.list_remote --slug dreamer07/eyepacs --out data/raw/eyepacs/remote_listing.csv
```

Takes a few minutes (176 pages). The conclusions drawn from it are committed in
`docs/data_report_eyepacs.json`, so it is only needed to re-verify.

Likewise `data/raw/eyepacs/trainLabels.csv` is gitignored; refetch with:

```
kaggle datasets download dreamer07/eyepacs -f "trainLabels.csv/trainLabels.csv" -p data/raw/eyepacs --unzip
```

## Gotchas discovered (do not rediscover these)

1. **The labels CSV path is `trainLabels.csv/trainLabels.csv`** — a *directory* named
   `trainLabels.csv` containing a file of the same name. Undocumented; usability 0.18.
2. **CSV `image` values carry no extension** (`10_left`) while disk files do
   (`10_left.jpeg`). Join on the stem, not the filename.
3. **Two opencv distributions are unavoidable.** `albumentations` needs
   `opencv-python-headless`, `grad-cam` needs `opencv-python`. Unpinned, pip resolves them
   to different major versions. Both are pinned to 4.12.0.88.
4. **pandas is pinned to 2.3.3, not 3.x.** pandas 3.0 changed copy-on-write and string
   dtype defaults; Kaggle is on 2.x.

## Exact next command

Awaiting user sign-off on the data report. **On sign-off**, the next step is the
patient-level split:

```
.venv\Scripts\python -m src.data.split --config configs/local.yaml --seed 42
```

(`src/data/split.py` is **not yet written** — writing it is the next action.)

Then `tests/test_no_leakage.py`, then `leakage-auditor` review before the splits are
committed.

## Open questions for the user

1. **Data report sign-off** — required before splits are generated (user's constraint 2).
2. **GitHub** — private repo + collaborator access for Ameena Ahmed and Muhammad Ali
   Abdullah. **Do not push without asking.** No remote is configured yet.
3. **Supervisor** — written confirmation of DECISION-001 before Phase 7.

## Known-broken / not yet built

- `src/data/{split,preprocess,dataset,sampler}.py` — not written.
- `tests/` — empty. `test_no_leakage.py` does not exist yet.
- No Kaggle notebook has been run; **GPU quota is fully unconsumed.**
- `docs/EXPERIMENTS.md` does not exist (no runs).
- Torch pins are unverified against the Kaggle image — the first Kaggle notebook must
  print its versions and the pins updated to match (see the note in `requirements.txt`).

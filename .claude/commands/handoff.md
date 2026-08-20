---
description: Update all tracking files and write a session summary
---

End-of-session protocol. Do all of it; skipping a step means the next session starts blind.

1. **`PROGRESS.md`** — update phase checkboxes (`[ ]`/`[~]`/`[x]`), rewrite the
   **NEXT ACTION** line at the top so it is a concrete next command, and update blockers.
2. **`docs/DECISIONS.md`** — add an entry for every non-obvious choice made this session.
   Add any newly excluded data rows, with identifiers and reasons.
3. **`docs/EXPERIMENTS.md`** — run `/report` if any run completed.
4. **`state/session_handoff.md`** — overwrite with:
   - what was done this session,
   - **the exact next command to run**,
   - open questions for the user,
   - known-broken things.
5. **`CLAUDE.md`** — update only if a rule, convention, or the current phase changed.
6. **Commit.** Descriptive message. Check `git status` first and review what is staged —
   split CSVs, configs, and metrics JSON are committed; `data/raw/`, `data/processed/`,
   `*.pth`, `*.npy`, and `kaggle.json` are not. If anything looks like it could contain a
   secret, open it before committing.
7. **Do not push** without asking the user first.

Then print a 5-line summary of where the project stands.

---
description: Print current phase, last run, and next action from PROGRESS.md
---

Read `PROGRESS.md`, `state/session_handoff.md`, and `docs/EXPERIMENTS.md`.

Print, and nothing else:

1. **Current phase** and its acceptance criteria.
2. **NEXT ACTION** — verbatim from the top of `PROGRESS.md`.
3. **Last run** — run_id, arm, model, val QWK, test QWK from `docs/EXPERIMENTS.md`. If
   there are no runs yet, say so; do not invent a placeholder row.
4. **Blockers** and any open items awaiting the user.
5. **Uncommitted changes** — `git status --short`.

Keep it under 20 lines. Every number must come from a file; if it is not in a file, say
"not yet recorded" (R4).

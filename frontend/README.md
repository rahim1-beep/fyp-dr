# Frontend — retinopathy grade estimate

Next.js 16 (App Router) + TypeScript + Tailwind v4. Rules for this app are in the repo-root
`CLAUDE.md` (frontend section); the API contract is `docs/api-contract.md`.

## Run

The backend must be up first — this page reads every figure and all four limits from it.

```
# repo root
set FYP_CHECKPOINT=<path to seed 42 best.pth>
.venv\Scripts\python -m uvicorn backend.app:app --port 8000

# frontend/
npm install
npm run dev          # http://localhost:3000
```

Point at a different backend with `NEXT_PUBLIC_API_BASE`.

## Layout

| path | what |
|---|---|
| `lib/types.ts` | the contract; `/predict` is a union on `graded` so `grade` cannot be read unnarrowed |
| `lib/api.ts` | one client; every call resolves to ok / http / network / malformed, validated at runtime |
| `components/ScoreRuler.tsx` | the score against the four cut points and the referral threshold, to scale |
| `components/Readout.tsx` | grade and referral as two answers of equal weight |
| `app/page.tsx` | upload, grading, declined, both framing warnings, graded, every error |
| `app/about/page.tsx` | what the model was measured to do, from `/meta` |

## Checks before committing

```
npx tsc --noEmit && npm run lint && npm run build
grep -rn "0\.7615\|0\.7464" app components lib .next   # must print nothing
```

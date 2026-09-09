# API contract — read from the source, not from memory

Generated 2026-09-09 by reading `backend/app.py`, `src/inference/predictor.py`,
`src/inference/coverage_guard.py` and `analysis/deployment/deployment.json`, and by
calling every route through `fastapi.testclient`. The recorded payloads are in
`fixtures/expected/`.

**Base URL:** `http://localhost:8000` · CORS `allow_origins=["*"]` (local demo only).

---

## `GET /health`

```json
{ "status": "ok", "run_id": "phase4_stage3_arm_e_efficientnet_b0" }
```

## `GET /meta`

Call once on load. **Every metric shown in the UI comes from here.** Full payload in
`fixtures/expected/meta.json`.

| field | type | value today |
|---|---|---|
| `run_id` | string | `phase4_stage3_arm_e_efficientnet_b0` |
| `seed` | int | `42` |
| `arch` | string | `efficientnet_b0` |
| `head` | string | `ordinal_regression` |
| `image_size` | int | `224` |
| `provenance` | string | one line naming the run and both figures |
| `reported_val_qwk` | float | `0.7569412145586636` |
| `reported_val_sens_at_spec95` | float | `0.736` |
| `disclaimer` | string[4] | render verbatim |
| `coverage_calibration` | object | `{ split, n_images, low, high }` |

`coverage_calibration` today: `split "train"`, `n_images 24586`, `low 0.6138193558673469`,
`high 0.8172433035714286`.

**`num_outputs`, `cuts` and `referral_threshold` are NOT in `/meta`.** `cuts` and
`referral_threshold` come back on each `/predict` response instead.

## `POST /predict`

`multipart/form-data`, file field name **`file`**. Query param `explain` (bool, default
`true`).

The response is a **discriminated union on `graded`**. Type it that way — the declined
variant has no `grade`, `grade_name`, `score`, `referable`, `referral_threshold` or `cuts`
key at all, so any code reading `grade` without narrowing is a runtime error waiting.

### `graded: true`

`fixtures/expected/predict_ok.json`

| field | type | notes |
|---|---|---|
| `graded` | `true` | discriminant |
| `grade` | 0–4 | from `cuts` |
| `grade_name` | string | already resolved — do not re-derive |
| `score` | float | the raw ordinal output; can be negative |
| `referable` | bool | `score >= referral_threshold` |
| `referral_threshold` | float | `0.9487713575363159` |
| `cuts` | float[4] | `[0.5, 1.62164, 2.17199, 3.272691]` |
| `guard` | object | see below — present in **both** variants |
| `run_id` | string | |
| `disclaimer` | string[4] | |
| `overlay_png` | string | `data:image/png;base64,…` — Grad-CAM, ~48 KB. Only when `explain=true` |
| `preprocessed_png` | string | `data:image/png;base64,…` — the 224px image actually graded, ~21 KB. Only when `explain=true` |

### `graded: false`

`fixtures/expected/predict_no_retina.json`

```json
{ "graded": false, "guard": {...}, "run_id": "...", "disclaimer": [...] }
```

Returned with **HTTP 200** — declining is a normal outcome, not an error.

### `guard`

| field | type | notes |
|---|---|---|
| `status` | enum | four values below |
| `coverage` | float or **null** | `null` exactly when `status == "no_retina_detected"` |
| `low`, `high` | float | the bounds this image was judged against |
| `message` | string | written for a human; render it, don't paraphrase |

| `status` | `graded` | meaning |
|---|---|---|
| `ok` | `true` | framing inside the training range |
| `framing_below_training_range` | `true` | retina fills too little of the frame |
| `framing_above_training_range` | `true` | cropped too tight |
| `no_retina_detected` | `false` | not a fundus photograph — no grade produced |

### Errors

| code | when | body |
|---|---|---|
| `400` | empty upload | `{"detail": "empty upload"}` |
| `400` | undecodable file | `{"detail": "the uploaded bytes did not decode as an image"}` |
| `413` | over 20,971,520 bytes | `{"detail": "upload is N bytes; the limit is 20971520"}` |

Recorded in `fixtures/expected/error_400_undecodable.json` and `error_413_too_large.json`.

---

## The grade/referral disagreement is real and correct

`grade` comes from `cuts`, fitted to maximise agreement with human graders.
`referable` comes from `referral_threshold`, fitted for 95% specificity on the
referable/not-referable decision. They are **independent fits**, and:

```
referral_threshold  0.9488
cuts[1] (1 -> 2)    1.6216
```

The threshold sits **below** the grade-1→2 boundary, so any score in
`(0.9488, 1.6216)` yields **`grade: 1` "Mild" with `referable: true`**. That band is
0.673 wide and is a normal, expected result — not a bug, not something to reconcile.

`fixtures/expected/predict_disagreement_SYNTHETIC.json` carries that state at score
1.2852.

---

## The forbidden numbers

The interface reports the **3-seed mean**, never the shipped seed's own figures
(DECISION-042 / DECISION-069). Seed 42 is the best-scoring of the three, so this is
enforced by test rather than by intent:

| must appear | must NOT appear |
|---|---|
| `0.7360` — 3-seed mean sensitivity | `0.7464` — seed 42's own sensitivity |
| `0.7569` — 3-seed mean QWK | `0.7615` — seed 42's own QWK |

Enforcing tests:

- `tests/test_api.py::test_meta_reports_the_three_seed_mean_and_not_the_shipped_seed`
- `tests/test_predictor.py::test_the_interface_quotes_the_three_seed_mean_not_the_shipped_seed_s_own_score`

Both assert `"0.7360" in text and "0.7464" not in text`.

---

## Startup

The server **refuses to start** without all three of `FYP_CHECKPOINT` (set to seed 42's
`best.pth`), `analysis/deployment/deployment.json`, and
`analysis/coverage_guard/calibration.json`. `FYP_CHECKPOINT` has no default, because a
fallback would eventually load some other run's weights and report them under this run's
provenance. Do not add one.

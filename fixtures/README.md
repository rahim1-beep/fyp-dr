# Frontend test fixtures

Rebuild with `python -m fixtures.build_fixtures`. Every image below was pushed through the
real `preprocess_image` and the real coverage guard, and the script **asserts** the
measured status before writing the file — so what this table records is what the code
produced, not what was intended.

## ⚠ These are TRAINING images. Do not screenshot them as a demo of performance.

The only real fundus photographs on this machine are the 20-image QA sample in
`data/raw/qa/`, and all 20 are from the **train** split. The model has seen them. They are
fine for exercising UI states, which is all they are for. A screenshot of one presented as
"the model grading an image" would be reporting a training-set result (R3).

## Built and verified

| file | triggers | measured coverage | note |
|---|---|---|---|
| `in_range_fundus.jpeg` | `guard.status == "ok"` | **0.6986** | real EyePACS image `13345_right`, unmodified |
| `framing_too_wide.jpeg` | `framing_below_training_range` | **0.3667** | top half of the same image — the retina is cut off, so it fills less of its own bounding box |
| `framing_too_tight.jpeg` | `framing_above_training_range` | **1.0000** | centre crop inside the disc — no surround left at all |
| `not_a_fundus.png` | `no_retina_detected`, `graded: false` | — | no retinal disc; the response has **no `grade` key** |
| `oversized_over_20mb.png` | HTTP **413** | — | incompressible noise, over the 20,971,520-byte limit |
| `not_an_image.jpg` | HTTP **400** | — | a `.txt` renamed; exercises the decode-failure path |

Machine-readable copy in `fixtures.json`.

Bounds these were judged against: **0.6138 – 0.8172** (`analysis/coverage_guard/calibration.json`).

## Score-selected fixtures — built with the real seed 42 checkpoint

Produced on 2026-09-13 by scoring the 20 QA images with
`runs/phase4_stage3_arm_e_efficientnet_b0/best.pth`, downloaded from Kaggle and verified
against the committed run (epoch 12, stored val QWK 0.7619686857570801 — identical to
`metrics.json`). Candidates must PASS the framing guard.

| file | source | label in train split | real result |
|---|---|---|---|
| `clean_grade0.jpeg` | `13613_left` | 0 | score 0.1845 → No DR, not flagged, framing ok |
| `disagreement_mild_but_refer.jpeg` | `3895_left` | 2 | score 1.2306 → **Mild + Refer**, framing ok |

**Why candidates must pass the guard.** The first real run picked `3829_left` for the
disagreement case. It is a 400×315, almost entirely black, underexposed photograph; the
preprocessing salvaged only a crescent-shaped artefact, coverage 0.2529. Its score did land
in the disagreement band — but that was the model grading a broken image, and the guard
correctly flagged it. A fixture for one state must not silently depend on a second, so the
builder now skips anything the guard does not pass.

These are still **training images**. `3895_left` being labelled Moderate and graded Mild is
one image the model was trained on; it is not a performance result.

## `expected/` — recorded responses for every state

Build and test the UI against these without needing a checkpoint or a running server.

| file | state |
|---|---|
| `meta.json` | the real `/meta` payload |
| `predict_ok.json` | graded, framing ok |
| `predict_framing_below.json` | graded, framed too wide |
| `predict_framing_above.json` | graded, cropped too tight |
| `predict_no_retina.json` | declined — note the absent `grade` key |
| `predict_disagreement_SYNTHETIC.json` | **Mild + refer**, score 1.2852 |
| `error_400_undecodable.json` | 400 body |
| `error_413_too_large.json` | 413 body |

**Two honesty labels are baked into those files, in a `_note` field. Read them.**

1. Every `predict_*.json` was produced with an **untrained** checkpoint. Key sets and
   types are exact; `score`, `grade`, `grade_name` and `referable` are **not real
   predictions**. Build against the shape, never quote the values.
2. `predict_disagreement_SYNTHETIC.json` is **hand-built**. Its score sits midway between
   the referral threshold and the grade-1→2 cut point. Every other field is a real
   response.

Strip `_note` before using a file as a mock; its presence in a diff is a useful signal
that a synthetic payload leaked somewhere it shouldn't.

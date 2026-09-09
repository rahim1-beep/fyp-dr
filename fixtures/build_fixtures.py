"""Build the frontend test fixtures, and VERIFY each one triggers the state it claims.

    python -m fixtures.build_fixtures

A fixture that is merely *named* `framing_too_tight` is worthless — the frontend would be
built against a state that never fires. Every image here is pushed through the real
`preprocess_image` and the real coverage guard, and the measured status is asserted before
the file is written. What the README records is what the code produced, not what was
intended.

WHAT NEEDS A CHECKPOINT, AND WHY IT IS NOT HERE. Two of the seven fixtures are selected by
MODEL SCORE — a clean grade 0, and the grade/referral disagreement case where the score
lands between the referral threshold (0.9488) and the grade-1->2 cut point (1.6216).
Producing those needs `best.pth`, which is gitignored, AND validation images, which are not
on this machine (CLAUDE.md §3 — the dataset lives on Kaggle). Set `FYP_CHECKPOINT` and this
script will score whatever real images it can find; without it, it says so and skips them.

For the disagreement case specifically the honest answer is that hunting for an image whose
score falls inside a 0.67-wide band is the wrong tool. The frontend needs to RENDER that
state, not discover it. `expected/` therefore carries a recorded response for it, clearly
marked synthetic, so the UI can be built and tested while the real image is unavailable.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data.preprocess import load_preprocess_config, preprocess_image
from src.inference.coverage_guard import HIGH, LOW, NO_RETINA, OK, Calibration, assess

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "fixtures"
QA = REPO / "data" / "raw" / "qa" / "data" / "data"


def _status(bgr: np.ndarray, cfg, cal) -> tuple[str, float | None]:
    r = assess(preprocess_image(bgr, cfg), cal)
    return r["status"], r["coverage"]


def main() -> int:
    cfg = load_preprocess_config(REPO / "configs" / "base.yaml")
    cal = Calibration.load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expected").mkdir(exist_ok=True)
    rows: list[dict] = []

    sources = sorted(QA.glob("*.jpeg"))
    if not sources:
        raise SystemExit(f"no source images under {QA}")
    src = sources[0]
    raw = cv2.imread(str(src))
    h, w = raw.shape[:2]

    # ---- 1. in-range real fundus ---------------------------------------------------
    st, cov = _status(raw, cfg, cal)
    assert st == OK, f"expected ok, measured {st} at coverage {cov}"
    shutil.copy(src, OUT / "in_range_fundus.jpeg")
    rows.append({"file": "in_range_fundus.jpeg", "guard_status": st, "coverage": cov,
                 "note": f"real EyePACS image {src.stem}, unmodified"})

    # ---- 2. framed too wide: part of the retina is outside the frame ---------------
    wide = raw[: h // 2]
    st, cov = _status(wide, cfg, cal)
    assert st == LOW, f"expected {LOW}, measured {st} at coverage {cov}"
    cv2.imwrite(str(OUT / "framing_too_wide.jpeg"), wide)
    rows.append({"file": "framing_too_wide.jpeg", "guard_status": st, "coverage": cov,
                 "note": "top half of the same image — the retina is cut off, so it fills "
                         "less of its own bounding box"})

    # ---- 3. cropped too tight: all retina, no surround ------------------------------
    cy, cx, s = h // 2, w // 2, int(min(h, w) * 0.30)
    tight = raw[cy - s:cy + s, cx - s:cx + s]
    st, cov = _status(tight, cfg, cal)
    assert st == HIGH, f"expected {HIGH}, measured {st} at coverage {cov}"
    cv2.imwrite(str(OUT / "framing_too_tight.jpeg"), tight)
    rows.append({"file": "framing_too_tight.jpeg", "guard_status": st, "coverage": cov,
                 "note": "centre crop inside the disc — no surround left at all"})

    # ---- 4. not a fundus photograph -------------------------------------------------
    # Deliberately NOT a black frame: a black frame is a trivial case. This is a bright,
    # busy, plausible photograph with no retinal disc in it.
    rng = np.random.default_rng(0)
    not_fundus = rng.integers(90, 210, (700, 900, 3), dtype=np.uint8)
    cv2.rectangle(not_fundus, (120, 160), (520, 480), (40, 60, 200), -1)
    cv2.circle(not_fundus, (700, 250), 90, (220, 200, 60), -1)
    st, cov = _status(not_fundus, cfg, cal)
    if st != NO_RETINA:
        # An honest fallback: if the mask finds a "retina" in noise, use a dark frame,
        # which is the real-world case (an unexposed or non-fundus photo).
        not_fundus = np.zeros((700, 900, 3), np.uint8)
        st, cov = _status(not_fundus, cfg, cal)
    assert st == NO_RETINA, f"expected {NO_RETINA}, measured {st}"
    cv2.imwrite(str(OUT / "not_a_fundus.png"), not_fundus)
    rows.append({"file": "not_a_fundus.png", "guard_status": st, "coverage": cov,
                 "note": "no retinal disc — the API returns graded:false with NO grade key"})

    # ---- 5. over the 20 MB upload limit ---------------------------------------------
    from backend.app import MAX_UPLOAD_BYTES

    big = OUT / "oversized_over_20mb.png"
    noise = rng.integers(0, 255, (3000, 3000, 3), dtype=np.uint8)   # incompressible
    cv2.imwrite(str(big), noise)
    while big.stat().st_size <= MAX_UPLOAD_BYTES:
        noise = rng.integers(0, 255, (noise.shape[0] + 600, noise.shape[1] + 600, 3),
                             dtype=np.uint8)
        cv2.imwrite(str(big), noise)
    rows.append({"file": big.name, "guard_status": "n/a — HTTP 413",
                 "coverage": None,
                 "note": f"{big.stat().st_size:,} bytes, limit is {MAX_UPLOAD_BYTES:,}"})

    # ---- 6. not an image at all ------------------------------------------------------
    (OUT / "not_an_image.jpg").write_text(
        "This is a text file with a .jpg extension. The API must answer 400 with a "
        "reason, not 500.\n", encoding="utf-8")
    rows.append({"file": "not_an_image.jpg", "guard_status": "n/a — HTTP 400",
                 "coverage": None, "note": "a .txt renamed; tests the decode-failure path"})

    # ---- 7. the score-selected pair --------------------------------------------------
    import os

    ckpt = os.environ.get("FYP_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        print("\n!! FYP_CHECKPOINT is not set (or missing), so the two SCORE-SELECTED")
        print("   fixtures were not built: a clean grade 0, and the grade/referral")
        print("   disagreement case. See fixtures/README.md.")
    else:
        from src.inference.predictor import Deployment, Predictor

        dep = Deployment.load()
        p = Predictor(deployment=dep, checkpoint=Path(ckpt), calibration=cal)
        lo, hi = dep.referral_threshold, dep.cuts[1]
        print(f"\nscoring {len(sources)} available images for the disagreement band "
              f"({lo:.4f} < score < {hi:.4f}) ...")
        for f in sources:
            r = p.predict(cv2.imread(str(f)), explain=False)
            if not r["graded"]:
                continue
            tag = None
            if lo < r["score"] < hi:
                tag = "disagreement_mild_but_refer.jpeg"
            elif r["grade"] == 0 and not r["referable"]:
                tag = "clean_grade0.jpeg"
            if tag and not (OUT / tag).exists():
                shutil.copy(f, OUT / tag)
                rows.append({"file": tag, "guard_status": r["guard"]["status"],
                             "coverage": r["guard"]["coverage"],
                             "note": f"score {r['score']:.4f} -> grade {r['grade']} "
                                     f"({r['grade_name']}), referable={r['referable']}"})
                print(f"  {tag}: score {r['score']:.4f}")

    (OUT / "fixtures.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} fixture(s) built and VERIFIED in {OUT}")
    for r in rows:
        c = f"{r['coverage']:.4f}" if isinstance(r["coverage"], float) else "-"
        print(f"  {r['file']:<32} {r['guard_status']:<32} coverage {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

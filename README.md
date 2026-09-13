# Diabetic retinopathy grading — patient-level validated, imbalance-aware, explainable

Final-year project, Department of Computer Science, Bahria University (2026–2027).
A deep-learning system that estimates diabetic retinopathy severity (grade 0–4) from a
retinal fundus photograph, served through a web interface.

**This is a research prototype, not a medical device.** The model is below the clinical
screening reference, and the interface says so on every screen.

| | |
|---|---|
| Model | EfficientNet-B0 with an ordinal-regression head, 224 × 224 |
| Training data | EyePACS, split by **patient** so no patient is in both training and testing |
| Validation QWK | **0.7569** (mean of three training runs) |
| Referable sensitivity | **0.7360** at 95% specificity — below the 0.80 screening reference |
| External validation | APTOS 2019. Under criteria fixed before the test, the model is **not** considered to generalise beyond its training population |

The full reasoning behind every choice is in `docs/DECISIONS.md`.

---

## Run the app

You need **Python 3.12** and **Node.js 20 or newer**. No GPU is needed. The trained model
is included at `runs/phase4_stage3_arm_e_efficientnet_b0/best.pth`.

### 1. Backend (Python)

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:FYP_CHECKPOINT = "runs\phase4_stage3_arm_e_efficientnet_b0\best.pth"
.venv\Scripts\python -m uvicorn backend.app:app --port 8000
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export FYP_CHECKPOINT=runs/phase4_stage3_arm_e_efficientnet_b0/best.pth
.venv/bin/python -m uvicorn backend.app:app --port 8000
```

On Linux, `pip` installs the CUDA build of PyTorch, which is a large download. It still
runs on CPU.

Leave it running. `http://localhost:8000/health` should answer `{"status":"ok",...}`.

The backend **refuses to start** if the model file or its calibration files are missing.
That is deliberate: it will not grade images with a safety check silently absent.

### 2. Frontend (in a second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

### 3. Try it

Upload any colour fundus photograph (JPEG or PNG). You get an estimated grade, a separate
referral decision, the image the model actually graded, and a Grad-CAM heatmap.

The test fixtures used during development are **not** included, because they are derived
from EyePACS images that cannot be redistributed. `fixtures/expected/` holds the recorded
API responses for every screen state instead.

---

## Run the tests

```bash
.venv/bin/python -m pytest tests -q        # Windows: .venv\Scripts\python -m pytest tests -q
```

A handful of tests skip without the original raw EyePACS label file. That is expected.

---

## What is not in this repository

- **The datasets.** EyePACS (35 GB) and APTOS 2019 come from Kaggle and are not
  redistributed here. All training, evaluation and Grad-CAM runs were done on Kaggle, from
  the notebooks in `notebooks/`.
- **Other checkpoints.** Only the deployed model (arm E, seed 42) is included.

## Where to look

| path | what |
|---|---|
| `docs/DECISIONS.md` | every non-obvious decision, with its reasoning and results |
| `docs/api-contract.md` | the backend API |
| `data/splits/` | the patient-level train / validation / test splits |
| `src/` | data pipeline, models, training, evaluation, Grad-CAM, inference |
| `backend/` | FastAPI service |
| `frontend/` | Next.js interface |
| `notebooks/` | the Kaggle notebooks that produced every result |
| `tests/` | including `test_no_leakage.py`, which enforces the patient-level split |

# Architecture Decision Record

ADR-style log of every non-obvious choice. Format: date, decision, alternatives
considered, rationale, who approved.

**Rule:** any deviation from `docs/proposal.pdf` is logged here *and* flagged to the user.
Any data rows excluded from the pipeline are listed here, never dropped silently.

---

## DECISION-001 — Streamlit replaced by FastAPI + Next.js

- **Date:** 2026-08-20
- **Status:** Accepted by user; **pending written supervisor confirmation before Phase 7**
- **Deviates from proposal:** Yes — §9 (Tools and Technologies) and §5.10/Month 8
  (Milestones) name Streamlit.

**Decision.** The web-based diagnostic interface is built as a proper client/server
application — FastAPI + Uvicorn backend, Next.js (TypeScript) + Tailwind frontend — rather
than a Streamlit script.

**Alternatives considered.**
1. *Streamlit as proposed.* Fastest to build; but produces a script, not assessable
   software engineering work, and offers no API surface.
2. *Gradio.* Same class of limitation as Streamlit.
3. *FastAPI + a server-rendered Jinja template.* Lighter than Next.js, but gives up the
   component model and the TypeScript type-safety story.

**Rationale.**
- Separation of concerns between inference and presentation.
- A documented REST API that a hospital system could in principle consume.
- Real request handling, error states, and file-upload validation (MIME type, magic bytes,
  size and dimension limits) rather than a notebook-grade upload widget.
- A UI that can be assessed as software engineering work.
- Auto-generated OpenAPI docs at `/docs` — a thesis figure in its own right.

**The objective is unchanged.** The proposal's stated objective is "a web-based diagnostic
interface"; only the implementation is upgraded. Scope §6.1 ("Web-based deployment") is
satisfied either way.

**Approval.** User approved 2026-08-20 and will obtain written confirmation from
Mr. Ubaid Ur Rahman before Phase 7 begins. Phases 1–6 do not depend on the frontend and
proceed regardless.

---

## DECISION-002 — Python 3.12 toolchain (local 3.12.10 / Kaggle 3.12.13)

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal does not pin a Python version)

**Decision.** The project targets **CPython 3.12**. The recorded version pair is:

| Environment | Version | ABI tag | Compute |
|---|---|---|---|
| Local (Windows 11) | **3.12.10** | `cp312` | CPU only |
| Kaggle Notebooks (Linux) | **3.12.13** | `cp312` | CUDA (T4×2 / P100) |

The virtualenv is built with `py -3.12 -m venv .venv`. `requirements.txt` pins versions
that resolve on `cp312` for **both** win_amd64 and manylinux.

**Rationale.**
- The machine's *system default* interpreter is 3.12's successor, 3.14, which Kaggle does
  not run and which parts of the CV/DL stack do not yet ship wheels for. Building against
  it would break R6 (reproducibility) on day one.
- Patch-level differences within 3.12 (10 vs 13) do not affect wheel compatibility — both
  build as `cp312` — so the local/Kaggle pair is genuinely interchangeable for
  dependency resolution.

**Constraint recorded.** The system default 3.14 interpreter is **not to be modified or
referenced in any config file.** All local execution goes through `.venv`.

---

## DECISION-003 — No local GPU acceleration; Kaggle-first is absolute

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal §7.2 already names Colab/Kaggle GPU)

**Decision.** The local machine is treated as a **code-editing, split-inspecting, and
web-app-demo machine only.** All preprocessing, training, evaluation, and Grad-CAM
generation run on Kaggle Notebooks.

**Measured environment.** Intel Arc Pro Graphics; no NVIDIA adapter, no CUDA runtime,
`nvidia-smi` absent. 31.4 GB system RAM, 797 GB free disk.

**Alternatives considered.**
1. *Intel XPU / IPEX backend for PyTorch.* **Explicitly rejected by the user.** Would add a
   second, divergent compute path to debug and maintain, with worse `timm` coverage, in
   exchange for acceleration that is still far below a T4.
2. *Download EyePACS locally and train on CPU.* Disk space is actually sufficient
   (797 GB free), so the original disk-based rationale does not apply — but CPU-only
   training on 35k images across 5 ablation arms is not viable on any timeline.

**Rationale.** With no CUDA device, the Kaggle-first mandate is compulsory rather than a
convenience. This is why every script must read its dataset and output roots from config
(`configs/kaggle.yaml` vs `configs/local.yaml`) and never hardcode a path.

---

## DECISION-004 — APTOS author-provided split discarded

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No (proposal does not specify an APTOS split)

**Decision.** The `mariaherrerot/aptos2019` redistribution ships its own `test.csv` (and
sibling split files). **These splits are ignored.** All ~3,662 labelled APTOS images are
pooled and used as a single held-out external test set.

**Rationale.** The partitioning rule behind that redistribution is undocumented. Adopting
it would silently import an unknown split into a project whose headline contribution is
partitioning rigour. Pooling is both simpler and honest.

**Related limitation (to document in the thesis, not hide).** APTOS `id_code` values are
anonymised hashes with no recoverable patient linkage. Each image is therefore treated as
its own patient (`patient_id = f"aptos_{id_code}"`). This means APTOS cannot support a
patient-level split at all — a further reason to use it only as an external test set.

---

## DECISION-005 — Image size fixed at 224×224 for headline results

- **Date:** 2026-08-20
- **Status:** Accepted
- **Deviates from proposal:** No — proposal §5.1/§5.3 specify 224×224.

**Decision.** All headline results use **224×224**. A config switch for 384×384 is built
but runs only as an optional ablation, and only if GPU quota survives Phase 4.

**Rationale.** Matching the approved proposal keeps the contract intact. 384 would roughly
triple training cost per run against a fixed weekly GPU quota that must first cover five
imbalance arms across two architectures.

---

## Excluded data rows

*None yet — Phase 1 reconciliation not complete. Any row excluded from the manifest is
listed here with its identifier and the reason.*

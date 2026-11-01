# SpectraCheck application usage guide

Status: **Supervisor-demo workflow verified. Authorized research activation remains pending.**

## Architecture

The frontend is React/Vite under `app/frontend`. The backend is a local FastAPI
service under `app/backend`. `InferenceService` selects the configured R2
research config, verifies the committed serving lock and metadata hashes, loads
the selected frequency model, and reuses the research preprocessing and feature
functions. Uploads are bounded raw image bytes processed in memory; they are not
saved by the application.

## Local setup

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
npm.cmd --prefix app/frontend ci
npm.cmd --prefix app/frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The service is local-only by default.

## Serving contract

The active serving configuration is `config_r2_fast_rf_t4.yaml`. A real model
response requires the reviewed `FREEZE.lock`, matching `feature_choice.json`,
`primary_model.json`, `versions.json`, the selected
`frequency_svm.joblib`, and every source/config hash named by the lock. Missing,
changed or incompatible inputs must remain fail-closed.

Without that reviewed bundle, the UI may show model-independent feature analysis
or an unavailable-model state. It must not display a fabricated classification.

For supervisor software demonstration before authorization, use the explicitly
separated command:

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py --host 127.0.0.1 --port 8000
```

This uses the verified Phase 6 `frequency_svm` artifact and exact R2 metadata,
but intentionally does not use `FREEZE.lock`. Every response is labelled as a
development demonstration and is not research evidence. Do not use its outputs
for metrics, test claims or Phase 9 conclusions.

## API behavior

- `GET /api/health`: reports model readiness, input limits and warnings.
- `POST /api/analyze`: accepts one JPEG, PNG or WebP and returns classification,
  AI probability, spectrum image and RAPSD coordinates only when the bundle is
  verified.
- Invalid, undersized or oversized inputs return an error; changed model inputs
  return an unavailable/503 response.

## Research-use limitations

The model was selected on validation accuracy for a single Stable Diffusion v1.5
R2 study. A score is not proof of authenticity. Do not generalize to other
generators, phone photos, screenshots or compressed images without a separately
approved study.

## Final activation record

- Development-demo verification: **passed 2026-11-01** using the existing
  non-sealed `reports/phase_8_desktop.png` image through the browser, Vite proxy
  and FastAPI demo service. The UI displayed the `frequency_svm` model name, an
  AI-probability meter and the development-only warning. This is software
  verification, not a research result.
- Development-demo health: `ready: true`, `mode: development_demo`,
  `research_authorized: false`.
- Production serving lock: **absent by design; production readiness remains
  false until the authorized Phase 7 freeze is completed.**
- Serving bundle/freeze commit: **TBD**
- Authorized model-readiness check: **TBD**
- Real-model smoke-check evidence: **TBD**
- Physical phone review: **TBD / not performed**
- Reviewer and date: **TBD**

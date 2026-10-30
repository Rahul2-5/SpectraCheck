# SpectraCheck local prototype (Phase 8)

Run from the project root with Python 3.11+ and Node 22.12+.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
npm.cmd --prefix app/frontend ci
npm.cmd --prefix app/frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. On macOS/Linux, replace the Windows Python path
with `.venv/bin/python` and `npm.cmd` with `npm`. The default service binds only
to the local computer. It does not need a paid service, API key, or GPU.

For frontend development, keep the API running and use a second terminal:

```powershell
npm.cmd --prefix app/frontend run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` to the local Python service.

## Supplying the research model

The UI is available before model artifacts exist; predictions remain disabled.
After the actual Phase 7 work is complete/reviewed, supply the trusted local
bundle from that run under the configured artifacts directory:

- `FREEZE.lock`, `feature_choice.json`, `primary_model.json`, `versions.json`.
- `models/frequency_<selected-model>.joblib` named by the primary-model record.
- The exact research `config.yaml` and package source matching the lock.

Restore the ignored model binary from the verified Phase 6 Drive backup. The
adapter verifies config, primary/feature selection, model, version-record, and
package-source hashes before deserialization/inference. It confirms the
frequency model was selected on validation accuracy and warns about runtime
library mismatches. Do not load unrelated downloaded pickle/joblib models.

This is serving-input verification, not another Phase 7 evaluation. The API
never opens sealed batches, split rows, test predictions, or results. App settings
live separately in `app/backend/settings.yaml`, leaving the research config hash
unchanged. Backend settings control upload bytes, maximum pixel count, supported
formats, and compiled frontend location.

## Upload and inference behavior

- One JPEG/PNG/WebP at a time, at most 10 MiB and 40 million pixels by default.
- The same `preprocess_image` and frequency extractor used by the research code.
- Binary label 1 is AI-generated; displayed probability is the model's AI column.
- The page shows the predicted class, probability, centered log spectrum, RAPSD,
  model-compatibility warnings, and the single-generator disclaimer.
- Uploads use a bounded raw binary request stream and in-memory processing.
  There are no saved uploads or prediction histories; previews use revocable
  object URLs. No multipart spooling or external image service is involved.
- Corrupt, unsupported, undersized, oversized, and unavailable-model cases return
  clear errors. Replacement/removal cancels pending browser requests and prevents
  stale results from appearing.

## API contract

| Route | Behavior |
| --- | --- |
| `GET /api/health` | Model readiness, supported target size/limit, warnings and disclaimer |
| `POST /api/analyze` | Raw encoded image bytes; prediction/probability, spectrum PNG data URI, RAPSD coordinates |

Analysis statuses: 200 result; 413 upload too large; 422 invalid/unsupported or
undersized image; 503 missing, incompatible, or changed model inputs. The API
derives format from encoded bytes, not a filename or MIME assertion.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_app_api.py
npm.cmd --prefix app/frontend test
npm.cmd --prefix app/frontend run build
npm.cmd --prefix app/frontend run test:browser
```

Browser tests use installed Chrome (`npx playwright install chrome` if needed),
mock only synthetic API responses, and include axe checks, desktop/mobile widths,
keyboard file selection, error handling, and result clearing. They do not supply
a research model. Touch/keyboard feel and safe areas on physical phones remain
a team check; desktop emulation is not evidence of real-device testing.
# Approved R2 experiment

For an R2 serving bundle, set `research_config: config_r2.yaml` in the local
backend settings. All referenced config/artifacts must be inside the serving
project and match the reviewed freeze. The default remains the historical
`config.yaml`; it does not silently select a different model or experiment.
R2 upload inference and extraction both consume original bytes through
`preprocess_image`, performing fixed-centre cropping and one JPEG generation.
Never supply the diagnostic's encoded patches as training or serving originals.
The UI scope note identifies fixed-centre-patch analysis. There is currently no
real R2 model bundle; analysis availability does not imply a trained detector.

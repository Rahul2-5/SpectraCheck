# Phase 8 report - React prototype and local inference API

Status: **Software implemented and verified locally. Real model/evaluation review remains pending. Phase 9 is not started.**

## What changed, in easy words

SpectraCheck now has a local web interface. A user can choose or drop an image,
preview it, request analysis when the research model is available, and inspect
the prediction, AI probability, frequency image, and radial power plot.

React handles the interface. A Python service uses the same preparation and
frequency calculations as the research pipeline. Images stay in memory and are
not saved. The current checkout has no real model bundle, so the page clearly
says the research model is not connected and leaves analysis disabled.

The user explicitly authorized Phase 8 code development on 28 October 2026 while
keeping real-data checks pending. No training output, freeze lock, research
prediction, or Phase 7 sign-off was created or represented as real.

## Done

- `app/frontend/`: React/Vite interface, responsive styles, SVG radial power plot, file preview, drag/drop, clear/reset, busy/error states, connection retry, study scope, version notices and the fixed disclaimer. Dependencies are exact-pinned with package-lock.json.
- `app/backend/inference.py`: shared preprocessing/extraction and evaluation probability-column handling; validation-selected primary model and feature/config/split provenance checks; serving input hashes; version warnings; in-memory spectrum encoding. It never invokes the held-out evaluator or opens sealed batches/results.
- `app/backend/main.py`: local FastAPI health and raw-binary analysis endpoints, bounded streaming, clear errors, local host restrictions, and compiled React serving. Raw streaming avoids multipart upload spooling to disk.
- `app/backend/settings.yaml`: app upload/pixel limits and frontend path, separate from the unchanged research config.
- `tests/test_app_api.py`: synthetic frozen serving bundle tests, evaluation-path feature/probability parity, no file writes, bad uploads, unavailable/changed artifacts and version notices.
- Frontend component tests: readiness, real response display, invalid/multiple files, API errors/retry and cancellation on removal.
- Browser tests: desktop/mobile widths, keyboard file choice, result/error/clear flows and axe WCAG checks with synthetic API mocks only.
- `requirements.txt`, `.gitignore`, and CI: pinned serving/test packages; ignored dependencies/build/test outputs; added Node component/build/browser job alongside Python jobs.
- `app/README.md`, root README, `docs/DECISIONS.md`, AGENTS.md and the notebook: setup, user override, serving contract, pending checks and Phase 8 STOP instructions.
- `reports/phase_8_desktop.png`, `reports/phase_8_mobile.png`: interface previews of the unavailable-model state, without any invented prediction.

## UI/UX skill application and review

Applied `emil-design-eng` and `mobile-native`. The interface has a clear upload /
analysis hierarchy, readable supporting text, semantic labels/figures, visible
keyboard focus, error/status announcements and unchanged browser zoom. File
names truncate without breaking layout; images fit without distortion. Hover
styles are capability-gated; pointer press feedback is short, keyboard actions
do not animate, and reduced-motion disables movement. Safe-area padding and a
single-column small-screen layout are included. There are no external fonts,
analytics, decorative loading delays, or fabricated model scores.

| Before | After | Why |
| --- | --- | --- |
| Several secondary labels below 4.5:1 contrast | Darkened supporting text; checked screens have zero axe violations | Keep small labels readable |
| Preview image briefly received an empty URL | Mount preview only when its object URL belongs to the selected file | Avoid stale/empty requests and misleading previews |
| Three narrow explanatory columns on mobile | Stacked explanatory rows | Improve reading at narrow widths |
| Generic press transform could apply to keyboard focus | Pointer-only press transform with reduced-motion override | Preserve immediate keyboard behavior |
| Unavailable model could leave users guessing | Explicit unavailable-model state; preview remains usable | Communicate what is possible without inventing results |

## Tests

Final Python regression command:

```powershell
$phase8TestRoot = Join-Path (Get-Location) '.venv/phase8-regression-20261028a'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.venv/matplotlib'
.\.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp $phase8TestRoot -o cache_dir=.venv/phase8-pytest-cache
.\.venv\Scripts\python.exe -m pip check
```

The explicit test directory was fresh and inside the workspace. For another run
use a fresh name; do not reuse a directory containing data you want to keep.

Actual summaries:

```text
137 passed in 31.91s
No broken requirements found.
```

Additional executed checks:

- Focused API tests: 7 passed in 4.27s.
- `npm test`: 5 React component tests passed.
- `npm run build`: passed; production JS approximately 73.69 kB gzip, CSS approximately 4 kB gzip.
- `npm run test:browser`: 2 isolated installed-Chrome tests passed in 5.1s. Axe reported zero WCAG A/AA violations on the tested desktop, 390px mobile, and result states. Keyboard Enter opens the file chooser; result removal and API errors work.
- Live production server smoke check: `/` HTTP 200; `/api/health` reports model unavailable as expected; corrupt binary upload HTTP 422.
- Dependency pin/import checks, notebook JSON/AST, app compile checks and `git diff --check` are recorded in phase_8_checks.json.
- Desktop and mobile screenshots were inspected visually. Physical phone testing was not available; browser emulation is not a claim of real-device behavior.

Test issues fixed: short pytest IDs avoid Windows filename-length failures; a
full run initially could not access the existing user temp/cache folders, so the
complete suite was rerun successfully in a fresh project-owned folder. Browser
testing exposed contrast issues and the preview URL timing problem; these were
corrected before final checks. Isolated browser process cleanup required the
appropriate Windows process permissions. No tests were skipped to pass.

## Outputs produced

The app source, API source, tests, lockfile, documentation, screenshots, and check
record exist. Compiled frontend files are built locally under ignored
`app/frontend/dist/`. The preview service binds to http://127.0.0.1:8000 only.

No real models, freeze lock, evaluation marker, results, or saved user uploads
were produced. The original research config and package/script source are
unchanged, so Phase 6's recorded config hash remains historical/active.

## Checks for the team

Use [app/README.md](../app/README.md) for laptop setup. Colab is still for the
research pipeline; the React demo runs locally.

1. Complete/review the actual Phase 6/7 work. Restore the selected model from the verified Drive backup along with matching config/source, lock, feature choice, primary model and version records.
2. Build the frontend and run `python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000`. Open localhost:8000.
3. Analyze five reviewed images and compare probabilities with the appropriate saved evaluation results without rerunning the held-out evaluator or altering frozen inputs.
4. Check a text file renamed .jpg, undersized image, unsupported file, too-large upload, offline API, long file name, replacement/removal during analysis and keyboard-only operation.
5. On a physical phone, verify touch feedback, zoom, orientation and safe areas. This remains unverified here; use an explicitly authorized local-device setup if sharing the laptop service on a LAN.
6. Note phone-photo/screenshot/compressed-image behavior as out-of-scope limitations. A model score is not authenticity proof.
7. Inspect the pushed commit's Python/frontend CI results. No remote Phase 8 CI success is claimed here.

## Assumptions and unverified items

- The user's explicit reply authorizes code development before real Phase 7 review, not fabricated research results or bypassing the evaluation protocol.
- Model artifacts are the trusted local Phase 6/7 bundle. Hash verification protects against drift; joblib is not used to deserialize uploads or arbitrary downloaded models.
- Serving checks only model/config/selection/version/package files. It deliberately does not call the complete evaluator verification routine, which also inspects sealed-batch metadata; app inference needs no test data.
- Exact pinned library versions do not guarantee cross-platform model compatibility. Mismatches are shown, and missing/failed verification blocks predictions.
- The UI supports jpg/jpeg/png/webp via actual encoded format validation. The configured default bounds are 10 MiB and 40 million pixels.
- Real-model parity, real prediction quality, supervisor sign-off, physical device feel and remote CI remain pending.

Technical references consulted: [FastAPI direct requests](https://fastapi.tiangolo.com/advanced/using-request-directly/), [Vite setup](https://vite.dev/guide/), and [React effect cleanup](https://react.dev/reference/react/useEffect). Installed API signatures were inspected. UI skills informed design choices; no skill sent messages or published external content.

## STOP

Phase 8 software is complete locally. Wait for review/authorization before Phase 9;
the report pack must use saved real outputs and must not retrain or rerun tests.

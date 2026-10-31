# SpectraCheck repository handover and Phase 6 audit

Audit date: 31 October 2026
Auditor: Senior ML Engineer / Software Architecture review

## Executive verdict

- Before this handover report was added, the checkout was clean on `main` at `d0f7278`, aligned with `origin/main`.
- R2 Phase 6 is complete under the approved `config_r2_fast_rf_t4.yaml` profile according to the committed completion and independent-verification records.
- The validation-selected primary is `frequency_svm` (RBF SVM): **605/750 = 80.67% validation accuracy**, ROC-AUC `0.883072`, Brier score `0.135568`.
- The six serialized model binaries are **not present locally**. They are ignored by Git and are evidenced only by the verified Drive backup hashes recorded in [`phase_6_fast_rf_t4_backup_hashes.json`](phase_6_fast_rf_t4_backup_hashes.json).
- No local `FREEZE.lock`, evaluation marker, test results, or test predictions exist. No evaluator or training command was run during this audit.
- Phase 7 is blocked by the missing team/supervisor practical-use decision, unverified supervisor sign-off, missing local model bundle, and explicit authorization requirements.
- Phase 8 software is already implemented. It has been tested with synthetic serving fixtures, but real-model serving and probability parity remain unverified.
- Phase 9 report-pack implementation has not started.

## 1. Repository status and structure

The repository contains 165 tracked files and these principal areas:

| Area | Responsibility |
|---|---|
| `src/aidetect/` | Acquisition/audit support, preprocessing, FFT features, pixel baseline, extraction, training, checkpoints, freeze, evaluation, statistics |
| `scripts/` | CLI entry points for acquisition, audit, extraction, training, freeze and evaluation |
| `notebooks/` | Colab phase runners, R2 extraction/training cells, fast-profile status and verification cells |
| `artifacts/` | Committed provenance and selection metadata; model binaries are intentionally ignored |
| `reports/` | Phase reports, checks, R2 results and figures |
| `app/backend/` | FastAPI API and hash-verified Python inference adapter |
| `app/frontend/` | React/Vite interface, component tests and browser tests |
| `tests/` | Python unit, protocol and synthetic integration coverage |
| `.github/workflows/` | Frontend and Python CI |

The active research profile is `config_r2_fast_rf_t4.yaml`, with semantic config hash:

`46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766f`

Its R2 scope is a treated-input fixed-centre 256x256 patch with one common JPEG Q96/4:4:4 encoding. The original raw shortcut audit remains blocked; the passing audit is only for the approved treated-input R2 domain. Supervisor acceptance remains unverified.

The older phase reports and notebook sections contain historical statements such as “no real Phase 6 artifacts.” The current source of truth is the latest R2 completion record and [`phase_6_fast_rf_t4_report.md`](phase_6_fast_rf_t4_report.md); those older statements should not be used to assess the current Phase 6 state.

## 2. Completed phases

| Phase | Current status |
|---|---|
| 0 | Repository, package layout, configuration, CI and local environment implemented and checked |
| 1 | R2 source acquisition/provenance implementation and reviewed cohort evidence recorded |
| 2 | R2 treated-input audit passed; immutable balanced split recorded. Original raw audit remains blocked |
| 3 | Shared version-2 preprocessing implemented and reviewed on train-only R2 inputs |
| 4 | FFT/RAPSD feature module implemented; R2 train-only spectrum review completed |
| 5 | R2 train/validation/test-sealed feature extraction completed; test values remained sealed |
| 6 | Real R2 training completed and independently verified under the approved fast RF/T4 amendment |
| 7 | Software implementation complete; real freeze and one-time test evaluation not run |
| 8 | React/FastAPI software implemented and locally synthetic-tested; real model review pending |
| 9 | Not started |

The fast profile is an explicit amendment: three shared CV folds and ten candidates per model, rather than the original five folds and thirty candidates. It must not be combined with the original or other fast-profile results.

## 3. Phase 6 artifact and result audit

### Provenance and protocol

The committed metadata consistently records:

- train rows: `3,500`; validation rows: `750`;
- seed: `42`; shared CV folds: `3`; search candidates per model: `10`;
- feature contract hash: `1a1538d2...01c1a31`;
- frozen split hash: `1e823b5d...ccfef93`;
- checkpoint contract hash: `eb276737...666e1`;
- `test_features_read: false`;
- selected feature-choice grid size: `75` candidates.

Feature selection chose `r0=120`, `k_mad=5` at validation accuracy `0.7866667`, using the recorded highest-accuracy/tie rule. The primary-model record independently selects `frequency_svm` by validation accuracy.

### Six-model comparison

| Artifact | Features | CV accuracy mean +/- SD | Validation accuracy | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| `frequency_svm` | 68 | 0.8366 +/- 0.0210 | **0.8067** | 0.883072 | 0.135568 |
| `frequency_rf` | 68 | 0.7749 +/- 0.0283 | 0.7680 | 0.856092 | 0.154932 |
| `frequency_xgb` | 68 | 0.7934 +/- 0.0200 | 0.7893 | 0.869468 | 0.155469 |
| `pixel_svm` | 1,024 | 0.6409 +/- 0.0138 | 0.6600 | 0.723371 | 0.212452 |
| `pixel_rf` | 1,024 | 0.6709 +/- 0.0078 | 0.6667 | 0.750471 | 0.208623 |
| `pixel_xgb` | 1,024 | 0.6686 +/- 0.0071 | 0.6733 | 0.762830 | 0.199505 |

The best-parameter records are present in [`training_results.json`](../artifacts/r2_fast_rf_t4/training_results.json). The RF amendment was applied to both RF searches: 100–300 trees and `max_features=sqrt`. The report records three reused pipelines (`frequency_svm`, `frequency_xgb`, `pixel_svm`) and two newly rerun RF searches plus the final pixel XGBoost fit; no old RF model or candidate was reused.

### Versions

The trained bundle records:

```text
Python 3.13.16
joblib 1.6.0
numpy 2.4.6
scikit-learn 1.9.1
xgboost 3.2.0
```

The repository environment used for this audit is Python 3.12.1 with the same pinned library versions. The app correctly treats the Python minor-version difference as a serving warning; an actual restored-model load has not been tested here.

### Binary availability

All six expected local paths under `artifacts/r2_fast_rf_t4/models/` are absent:

```text
frequency_svm.joblib  frequency_rf.joblib  frequency_xgb.joblib
pixel_svm.joblib      pixel_rf.joblib      pixel_xgb.joblib
```

The recorded Drive model hashes are:

| Model | Recorded SHA-256 |
|---|---|
| `frequency_svm` | `f5eb064260349873ebeddb16dc090a0aa842ce7b844bfd0c0e203d97d578bbda` |
| `frequency_rf` | `d2837f98d3e84630dfa660e1559fb347250c0b1a2524d3d7aa32872cfa8c799d` |
| `frequency_xgb` | `5e08c55a9e987bbb77c209997efabf96147bd30261ad705b509c03dba642b9da` |
| `pixel_svm` | `8cf1e768e24c85c6bfc373d576c176e6713bb9a3ad59dbc63f1c28c9864f51b5` |
| `pixel_rf` | `457ae5da86ec1fcf7caa26a2e98fde5f8aed2aa772a02a3a276b48c2caac5175` |
| `pixel_xgb` | `2cd68428cd0a4c29522a5396ea22fd62aa1efff6dff733f2933ef3885f8e6ab8` |

The completion records state that the six files were backed up and independently hashed in Drive under `aidetect/r2/review_backup/r2_fast_rf_t4_training_20261031T093525Z/artifacts/models`. This audit could verify the committed metadata and recorded hashes, but could not rehash the Drive binaries from this local workspace. Do not call the local checkout a runnable model bundle until those six files are restored and their hashes match the recorded backup manifest.

No `FREEZE.lock`, `EVALUATION_STARTED.json`, `reports/results.json`, `reports/results.md`, or `reports/test_predictions.csv` exists locally.

## 4. Current backend architecture

`app/backend/main.py` creates a local FastAPI service with:

- `GET /api/health` for readiness, scope, limits and version warnings;
- `POST /api/analyze` for one bounded raw-binary image request;
- localhost-only trusted hosts;
- optional serving of the compiled React `dist/` directory.

`app/backend/inference.py`:

- validates encoded format, upload size and pixel limits without disk-backed uploads;
- calls the shared `aidetect.preprocess.preprocess_image` and frequency extractor;
- uses the shared positive-probability handling from evaluation code;
- verifies the config, freeze lock, selected-model records, source hashes and model hash before loading joblib;
- returns model-independent spectrum/RAPSD analysis when the reviewed model bundle is unavailable;
- never opens sealed feature rows, test predictions or evaluation results.

The current `app/backend/settings.yaml` does not select the active R2 config, so the default service looks for historical `config.yaml`. That is intentional while the reviewed Phase 7 bundle is absent, but it must be changed to the active R2 config before real serving.

## 5. Current frontend architecture

`app/frontend/src/App.jsx` and `styles.css` provide:

- file picker and drag/drop for JPEG, PNG and WebP;
- object-URL preview with cleanup;
- health/retry, busy, clear, cancellation and error states;
- explicit unavailable-model/feature-analysis state;
- prediction label, AI probability, spectrum image and SVG RAPSD display when a model is available;
- study scope, single-generator disclaimer, keyboard focus behavior and responsive mobile layout.

Vite builds the production bundle. The frontend calls only the local `/api/health` and `/api/analyze` routes; inference remains Python-side.

## 6. Checks performed during this handover

These checks used synthetic fixtures or source/build assets only:

- `tests/test_app_api.py`: **9 passed**;
- React component tests: **6 passed**;
- production Vite build: passed;
- Playwright browser/accessibility tests: **2 passed**;
- `pip check`: passed;
- pre-report Git status: clean on `main`.

The tests do not prove real-model compatibility, real-image predictions, held-out accuracy, or physical-phone behavior. The full historical regression counts in the phase reports were not treated as a replacement for those checks.

## 7. Missing work and blockers

### Immediate Phase 7 blockers

1. `reports/practical_target.md` is still a placeholder with `Status: Pending`. The active R2 config sets `reports_root: reports/r2_fast_rf_t4`, so the freeze implementation will look for `reports/r2_fast_rf_t4/practical_target.md`, which does not currently exist.
2. Supervisor sign-off and explicit Phase 7 authorization are unverified.
3. The six model binaries must be restored from the verified Drive backup.
4. The Phase 7 notebook cells still load historical `config.yaml`, hardcode the historical root lock path in their instructions, and do not directly target the active `r2_fast_rf_t4` artifact/report roots.
5. `scripts/freeze.py` and related notebook text print generic `artifacts/FREEZE.lock` paths even when an alternate config places the lock under `artifacts/r2_fast_rf_t4/`. The underlying freeze code resolves configured roots, but the handoff instructions are misleading for R2.

### Reproducibility risk to fix before locking

The Windows checkout has `core.autocrlf=true`. JSON files have no explicit LF rule in `.gitattributes`, so their working-tree SHA-256 differs from the committed Git-blob/Colab LF hash. For example, the committed Phase 6 completion record is the reported `5eb86a...96025`, while the current Windows working-tree bytes hash to `939446...4173b`. This matters because the freeze and serving contracts hash JSON artifacts byte-for-byte. Normalize JSON and hash-record line endings, or run the complete freeze/serving workflow in a controlled LF checkout, before relying on a committed lock.

### Phase 8 gaps

- No real R2 `FREEZE.lock`/model bundle is available to the app.
- No actual primary-model load or five-image prediction-parity check has been performed.
- Physical-phone touch, safe-area, zoom and orientation checks remain pending.
- The app documentation still describes the real R2 bundle as unavailable; it should be updated after reviewed activation, not before.

### Phase 9 gaps

The following deliverables are absent: `scripts/make_figures.py`, `reports/model_card.md`, `reports/deviations.md`, final results tables/figures, and the Phase 9 report. They must be generated only from saved Phase 2–8 evidence and final Phase 7 results; no retraining or second evaluation is needed.

## 8. Phase 7 prerequisite and execution plan

After team/supervisor approval only:

1. Record the agreed practical-use threshold, rationale, team members, supervisor and date under the active R2 report root. Do not derive it from test results.
2. Update the Phase 7 notebook cells to use `config_r2_fast_rf_t4.yaml` explicitly and to display/commit the configured nested lock and report paths.
3. Restore all six Drive model binaries into `artifacts/r2_fast_rf_t4/models/` and verify all six recorded SHA-256 values before freeze.
4. Run the Phase 7 preflight and freeze command with the explicit R2 config. Freeze may inspect sealed Parquet metadata and raw file hashes, but must not parse test feature values.
5. Inspect the generated lock, commit it on clean `main`, and only then run the evaluator once with the same explicit R2 config.
6. If the evaluator creates its start marker and then fails, do not rerun it. Preserve the marker and escalate to the supervisor.
7. Review all six held-out metrics, confusion-matrix arithmetic, H1/H1a/H1b decisions, and the practical target before enabling real app predictions.

This audit did not execute any of these steps.

## 9. Exact Phase 8 implementation plan

The application architecture does not need to be redesigned. Reuse the current React/FastAPI path:

1. Add the verified R2 model bundle and lock under `artifacts/r2_fast_rf_t4/` after Phase 7 review.
2. Set `research_config: config_r2_fast_rf_t4.yaml` in `app/backend/settings.yaml`; do not alter the research YAML to suit the app.
3. Verify the selected `frequency_svm` binary, feature-choice record, primary-model record, versions record, config and all source hashes against the committed lock.
4. Run the existing API/component/browser checks, then run the real local service and verify health reports the selected model rather than the unavailable state.
5. Analyze five reviewed images and compare the returned probabilities with the appropriate saved evaluation predictions. Record any mismatch; do not tune the model or rerun evaluation.
6. Check corrupt/renamed files, undersized images, unsupported formats, upload limits, offline API, long names, replacement/removal during analysis and keyboard-only use.
7. Perform the physical-phone checks and document phone-photo, screenshot and compression behavior as out of scope rather than treating those inputs as validated.
8. Update the app handover documentation with the actual bundle/config and runtime-version warning observed.

## 10. Recommended execution order

1. Human team/supervisor decision: practical target, R2 amendment acceptance and Phase 7 authorization.
2. Small implementation/documentation fix: parameterize Phase 7 for `config_r2_fast_rf_t4.yaml` and establish LF-stable artifact hashing.
3. Colab Phase 7 preflight, model restore, freeze, commit, single evaluation and review.
4. Activate the reviewed R2 bundle in the existing Phase 8 app and complete real-image/device checks.
5. Implement Phase 9 from saved outputs only: figures, tables, model card, deviations and reproduction guide.

## Exact next implementation task

Implement the Phase 7 handoff wiring, not a new model or a new app architecture: make the Phase 7 notebook/CLI path explicit for `config_r2_fast_rf_t4.yaml`, resolve the configured nested artifact/report roots, and make the hashed JSON/record files LF-stable. Add or update synthetic path/provenance tests for that wiring. Then stop for the human practical-target and supervisor decision. Do not retrain, open sealed test values, create `FREEZE.lock`, or run the evaluator in that implementation task.

End of audit.

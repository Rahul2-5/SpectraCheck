# SpectraCheck reproducibility guide

Status: **Template. Final freeze, evaluation and result hashes are pending.**

## Environment

- Python: 3.11+ within the pinned project environment.
- Node: 22.12+ for the React frontend.
- Install Python dependencies from `requirements.txt` and the package in
  editable mode; install frontend dependencies with `npm ci`.
- Record `pip check`, Python version, Node version and package versions in the
  final evidence bundle.

## Source and configuration contracts

Use `main` and the reviewed commit. The active Phase 6 configuration is
`config_r2_fast_rf_t4.yaml`, SHA-256
`46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766`. Preserve
the recorded split, feature-contract, artifact and report-root hashes. Do not
recreate the split or alter frozen inputs.

## Research pipeline

1. Review Phase 1-6 evidence and the practical target.
2. Complete team/supervisor approvals and commit the target under the configured
   R2 report root.
3. Follow the Phase 7 notebook/CLI preflight and freeze instructions. **Do not
   run these steps until authorization is recorded.**
4. Commit and verify `artifacts/r2_fast_rf_t4/FREEZE.lock` before any test access.
5. Run the single evaluator once from the clean committed checkout.
6. Record the marker, results, predictions and their hashes.
7. Populate Phase 9 tables only from those committed outputs.

No command in this template has been executed by documenting it here.

## Phase 8 application

The React/Vite frontend calls the local FastAPI service. The service reads one
encoded image into memory, calls the shared `preprocess_image`, extracts the
same ordered frequency vector, and loads only the hash-verified validation-
selected frequency model. It never reads sealed batches or evaluation rows.
See [`PHASE_9_APPLICATION_USAGE.md`](PHASE_9_APPLICATION_USAGE.md).

## Evidence manifest

Complete after the authorized run:

| Evidence | Path | SHA-256 |
| --- | --- | --- |
| Freeze lock | `artifacts/r2_fast_rf_t4/FREEZE.lock` | **TBD** |
| Evaluation marker | `artifacts/r2_fast_rf_t4/EVALUATION_STARTED.json` | **TBD** |
| Results JSON | `reports/r2_fast_rf_t4/results.json` | **TBD** |
| Results report | `reports/r2_fast_rf_t4/results.md` | **TBD** |
| Predictions | `reports/r2_fast_rf_t4/test_predictions.csv` | **TBD** |
| Final report | `reports/phase_9_report.md` | **TBD** |

## Reproduction record

- Final source commit: **TBD**
- Environment record: **TBD**
- Phase 9 review date: **TBD**
- Reproduced by: **TBD**
- Differences from the recorded run: **TBD; do not infer.**

# Phase 6 report — training and model selection

Status: **Phase 6 implementation and local synthetic checks are complete. The real Colab training run, practical-use target agreement, and remote CI review remain pending.** No real-data score or trained research model was generated in this checkout.

## Done

- `config.yaml`: added the Phase 6 feature-selection grid, model order, randomized-search ranges, CPU job limits, accuracy scoring, sigmoid calibration setting, feature-importance settings, and explicit compatibility fingerprints for verified Phase 5 feature configurations. Active full-config SHA-256: `ce79d3ef5f012768bbc377db9cdd9226377a1449a13a7725d9d49189f3549dce`.
- `src/aidetect/train.py`: reconstructs selected frequency and pixel matrices from Phase 5 summaries; selects `r0` and `k_mad` using train-fit/default-RF validation accuracy with the specified tie rule; creates the shared shuffled stratified folds; runs all six scaled pipelines through equal-budget `RandomizedSearchCV`; sigmoid-calibrates SVM on training rows; records validation accuracy, ROC-AUC, and Brier score; writes provenance, models, versions, and descriptive frequency RF/XGB importance outputs.
- `scripts/train_models.py`: added the train/validation-only Phase 6 entry point. It verifies the immutable split-file digest, train/validation Parquet configuration and batch-ID metadata, balanced labels, and non-crossing IDs/groups. It refuses to run after `artifacts/FREEZE.lock` exists and never calls the sealed-test reader.
- `tests/test_train.py`: added nine synthetic checks for summary reconstruction, feature-choice ties, configured search bounds, deterministic shared folds, fold-local scaling, legacy Phase 5 contract validation, test-path exclusion, six-model separability/determinism, and output/provenance generation.
- `notebooks/colab_runner.ipynb`: replaced the Phase 6 placeholder with split/hash preflight, the training command, result checks, and the Phase 6 stop gate.
- `reports/practical_target.md`: added the required team/supervisor decision template. It remains unsigned and contains no invented numeric threshold.
- `README.md`, `AGENTS.md`, and `reports/phase_6_checks.json`: documented the implementation, safeguards, actual local checks, and pending team work.

## Tests

- Focused command: `.venv/Scripts/python.exe -m pytest -q tests/test_train.py`
- Focused result: **9 passed, 0 failed, 0 skipped** — `9 passed in 36.33s`.
- Full regression command (PowerShell):

  ```powershell
  $env:PYTHONPATH = 'src'
  .venv/Scripts/python.exe -m pytest -q
  ```

- Full regression result: **114 passed, 0 failed, 0 skipped** — `114 passed in 24.82s`.
- A first full-suite invocation without `PYTHONPATH` had **1 failed and 113 passed**. The existing Phase 5 subprocess test could not import `aidetect` because this virtual environment is not installed editable. The documented editable-install command could not fetch its isolated `setuptools>=68` build dependency: the sandbox blocked network access, and the escalated retry failed TLS certificate verification. The successful full-suite invocation set `PYTHONPATH=src`; no dependency or certificate bypass was used.
- `.venv/Scripts/python.exe -m pip check`: passed — `No broken requirements found.`
- `.venv/Scripts/python.exe -m py_compile src/aidetect/train.py scripts/train_models.py tests/test_train.py`: passed.
- `.venv/Scripts/python.exe scripts/train_models.py --help`: passed and printed the train/validation-only usage.
- Notebook JSON and unique IDs passed; all **9 Phase 3–6 code cells** parsed with Python AST.
- `git diff --check`: exit code 0, no whitespace errors. Git reported that the notebook working-copy CRLF will normalize to LF.
- GitHub Actions `Python tests` completed with conclusion **success** for commit `6f9a6a881f5ad19ca5153a41644a9a2d4dfaf287` on `main` ([run](https://github.com/Rahul2-5/SpectraCheck/actions/runs/40533619083)).
- Installed package inspection on Python 3.11.4 verified `CalibratedClassifierCV`, `RandomizedSearchCV`, `SVC`, and `RandomForestClassifier` signatures. Scikit-learn was 1.9.1; NumPy 2.4.6; SciPy 1.17.1; XGBoost 3.2.0; joblib 1.6.0. The XGBoost effective configured parameters and search parameter names are checked by the code/tests.
- The Phase 6 synthetic end-to-end test confirmed all six validation pipelines exceeded 0.90 accuracy on its generated separable data. This is a software test only and is not a research result.

## Outputs produced (paths)

- `src/aidetect/train.py`
- `scripts/train_models.py`
- `tests/test_train.py`
- `config.yaml`
- `notebooks/colab_runner.ipynb`
- `reports/phase_6_checks.json`
- `reports/phase_6_report.md`
- `reports/practical_target.md` (pending team/supervisor completion)
- Updated `README.md` and `AGENTS.md`

The real-data outputs `artifacts/feature_choice.json`, `artifacts/primary_model.json`, `artifacts/versions.json`, `artifacts/models/*.joblib`, `reports/phase_6_results.json`, `reports/phase_6_results.md`, and the feature-importance CSV/PNG files were **not** generated locally. The checkout has no real frozen split or train/validation feature batches.

## Checks for the team to run in Colab

1. Update the `main` clone and install the pinned dependencies. Confirm the reviewed Phase 2 audit, `data_root/splits.csv`, `artifacts/splits.sha256`, and Phase 5 train/validation batches exist. The split file must hash to the recorded digest. Confirm `artifacts/FREEZE.lock` is absent.
2. Review the Phase 5 batch metadata for config hash, split hash, image side, RAPSD ring count, pixel size, peak thresholds, batch index, and ordered ID hash. Phase 6 accepts the active config hash, or the recorded Phase 5 hash only while the feature-contract fingerprint remains unchanged. If batches have another hash or feature contract, stop and preserve them; make a fresh feature output directory and rerun the authorized Phase 5 extraction instead of deleting or mixing batches.
3. Run:

   ```text
   python scripts/train_models.py
   ```

   The command reads only `features_root/train` and `features_root/val`. It must refuse to run if the freeze lock exists or batch/split provenance differs. Check reported train/validation row counts, exact class balance, and that no image IDs or duplicate groups cross those splits.
4. Review `artifacts/feature_choice.json`, all six rows in `reports/phase_6_results.md`, `artifacts/primary_model.json`, `artifacts/versions.json`, six model files under `artifacts/models/`, and the descriptive frequency RF/XGB top-15 plots. Confirm that the primary model is selected by validation accuracy and that no test feature values were read.
5. Record actual Colab outputs, runtime versions, failures, row counts, and batch/model timing in the team record. Do not turn synthetic test success into a real-data claim.
6. Agree the practical-use threshold and rationale with the supervisor and complete `reports/practical_target.md` before Phase 7. Check GitHub Actions on `main`; remote CI has not been verified locally.

## Assumptions I made

- The 2026-10-27 Phase 5 configuration hash in the project record identifies batches produced with the unchanged Phase 3/4/5 feature settings. Phase 6 accepts that legacy hash only when the current feature-contract fingerprint matches; otherwise it fails closed. If the team used another config, use a new output directory and rerun Phase 5 rather than reusing or deleting unknown batches.
- The configured model tie order `svm`, `rf`, `xgb` is the deterministic tie-break when two frequency models have exactly equal validation accuracy. The PRD requires validation-accuracy selection but does not define this tie case; the order is recorded in config and in `primary_model.json`.
- The project is intended to run from its editable install in Colab. This local virtual environment lacked that install, so the successful regression run supplied `PYTHONPATH=src`; the normal install command could not complete due the recorded TLS/network issue.

## Open questions / anything I could not verify

- The team has not supplied real Phase 5 train/validation feature batches, their row counts/timings, or the real split hash. No actual model tuning, validation score, or selected model is available.
- The practical-use target and supervisor approvals remain pending.
- The Colab runtime and real-data checks have not been run.
- A real feature directory with an unrecognized config hash or changed Phase 3/4/5 contract will be rejected until the team provides compatible provenance or regenerates Phase 5 outputs in a separate directory.
- The editable install did not complete locally; full regression tests passed with `PYTHONPATH=src`. Install the project normally in the team's Colab runtime before running the research workflow.

## STOP

Phase 6 code and local checks are complete. Wait for the team to run and review the real Colab checks, agree and record the practical-use target, and for your explicit **`PROCEED TO PHASE 7`** approval. Do not start Phase 7 from this report alone.

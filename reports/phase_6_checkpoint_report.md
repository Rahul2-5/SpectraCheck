# Phase 6 checkpoints — 31 October 2026

Implemented the user's request to add checkpoints after each model. This is
software verification with synthetic data, not evidence of real R2 training
completion. The existing Colab process was not interrupted or updated; it was
launched without checkpoints and cannot acquire this feature retrospectively.

## Changes

- `scripts/train_models.py --checkpoint-dir PATH` enables persistent resume.
  The R2 Phase 6 notebook and helper cells use
  `/content/drive/MyDrive/aidetect/r2/phase6_checkpoints`.
- Feature selection and each complete frequency/pixel SVM, RF and XGBoost
  pipeline receive atomic receipts. Model checkpoints include the fitted
  pipeline, selected parameters, CV/validation metrics and feature names.
  SVM calibration finishes before its checkpoint is committed.
- `status.json` reports stage, current model, completed models and timestamp;
  log output identifies searches and reuse. A writer lock prevents concurrent
  use of the same checkpoint folder. Incomplete uncommitted model files are
  ignored and recomputed on resume.
- Resume verifies exact config, frozen split, train/validation batch bytes,
  folds, package/CLI/requirements source hashes, Python/library versions,
  receipt and model checksums before loading checkpoint models. Validation
  metrics and feature schemas are checked again when reusing each model.
- Final artifacts remain separate until all six models complete. Publication
  reserves/checksums each output and supports recovery after interruption or
  loss of local final artifacts. Changed or unrelated existing outputs are
  refused, and an intact completed run refuses rerunning.
- R2 preparation restores imports/paths and checks the reviewed Phase 5 Drive
  evidence without depending on old kernel globals. Dataset acquisition,
  audit, splits and Phase 5 extraction are not rerun.

The research protocol remains unchanged: seed 42, five shared folds and 30
search trials per model. `config_r2.yaml` still hashes to
`ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6`.
The implementation reads train/validation only. No real test values, freeze
or evaluation were accessed or started by this task.

## Verification

Full local Python regression: **185 passed in 76.63s** on Python 3.13.5:

```text
.venv\Scripts\python.exe -u -m pytest -q --basetemp .venv/pytest-training-checkpoints-final -p no:cacheprovider
```

The suite includes interruption after the first saved model, reuse of five
remaining searches without repeating selection/the first model, identical
metrics/parameters/predictions versus uninterrupted execution, interrupted
final publication, missing final artifact recovery, corruption/provenance
rejection before deserialization, orphan writes, concurrent/stale locks and
sealed-input directory rejection. Existing protocol/API regressions passed.

The initial sandbox run hung in the existing API test while Python created
its Windows asyncio socket pair (`socket.accept`), before inference. The same
test passed outside the sandbox (1 passed in 3.39s), followed by the successful
full suite above outside the sandbox. No application bypass was introduced.

`pip check` passed. Notebook JSON, applicable Python cells and helper scripts
parsed; the two Phase 6 notebook sources match their helpers. CLI and installed
RandomizedSearchCV/joblib signatures were inspected. `git diff --check` passed.

## Resume and limits

Use the same command, checkpoint folder, checkout, dependencies, config and
inputs on the restored runtime. The notebook already supplies the checkpoint
argument. Changed versions/code fail closed and require review, not deleting
the manifest to force reuse.

Only completed pipelines are saved; an unfinished search/calibration restarts
that pipeline. Individual trials/folds are not resumable. Drive storage must
remain accessible; checksum failures block reuse. After a hard runtime kill,
remove only `running.lock` once the previous process/runtime is confirmed
stopped; do not remove a live writer's lock. No live checkpoint-enabled Colab
run was launched by this task. Phase 7 remains unauthorized.

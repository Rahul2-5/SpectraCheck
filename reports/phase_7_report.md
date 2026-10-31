# Phase 7 report ? freeze and single test evaluation

Status: **Phase 7 software implementation and synthetic checks are complete. The real Colab freeze/evaluation remains pending.** No real test rows or feature values were opened, and this checkout contains the reviewed R2 Phase 6 metadata but no local model binaries, agreed practical target, freeze lock, or research results.

## Done

- `src/aidetect/freeze.py`: validates the immutable split digest and group/class assignments, Phase 6 config/split provenance, six model outputs, validation-selected primary model, completed team/supervisor practical target, and sealed batch footer/schema metadata. It requires a clean `main` tree and refuses an existing evaluation marker/results. It hashes config, split, tracked package and script code, Phase 6 records and models, target, and raw sealed Parquet bytes. Before freeze, it reads only Parquet footer/schema metadata and raw bytes; it does not parse test rows or feature values.
- `src/aidetect/evaluate.py`: verifies every lock hash and requires the committed lock on a clean `main` checkout. For the active R2 profile it creates `artifacts/r2_fast_rf_t4/EVALUATION_STARTED.json` exclusively before reading test rows, validates test IDs/labels/groups against the frozen assignments, evaluates all six fixed models, computes exact Clopper-Pearson 95% accuracy intervals and the requested metrics, then saves per-image predictions and reports. A failed attempt after marker creation cannot be rerun through the command.
- `src/aidetect/stats_tests.py`: implements one-sided exact binomial H1, exact paired McNemar H1a with Holm correction and directional claim guard, and Cochran's Q H1b with conditional pairwise exact McNemar/Holm tests. It handles identical frequency-model correctness without emitting Statsmodels' undefined Q result.
- `scripts/freeze.py` and `scripts/evaluate_test.py`: added explicit freeze and single-evaluation commands.
- `tests/test_freeze_evaluate.py` and `tests/test_stats_tests.py`: added temporary-repository and generated-feature tests for the clean-tree gate, target gate, metadata-only freeze, locked batch hashes, prior-marker refusal, one-shot evaluation outputs, metrics, test ID/split parity, exact tests, Holm decisions, and edge cases. The synthetic evaluation reads only its generated fixture.
- `notebooks/colab_runner.ipynb`: replaced the Phase 7 placeholder with explicit `config_r2_fast_rf_t4.yaml` and nested-root Phase 6/target preflight, exact reviewed Drive-backup verification, freeze, the commit-before-evaluate stop, one-shot evaluation, and result-review instructions. The Phase 7 cells can restore model binaries after a Colab restart only when no evaluation marker exists.
- `reports/practical_target.md`: replaced the prose template with fields the freeze command can validate. The threshold and sign-off remain blank; no value was invented.
- `README.md`, `AGENTS.md`, and this report/check record document the Phase 7 contract and pending real-data work.

## Tests and checks

- Focused command: `$env:PYTHONPATH = 'src'; .venv/Scripts/python.exe -m pytest -q tests/test_stats_tests.py tests/test_freeze_evaluate.py` ? **16 passed, 0 failed, 0 skipped** in 35.03s.
- Full regression command: `$env:PYTHONPATH = 'src'; .venv/Scripts/python.exe -m pytest -q` ? **130 passed, 0 failed, 0 skipped** in 134.15s.
- `.venv/Scripts/python.exe -m pip check` ? passed: `No broken requirements found.`
- `py_compile` for the three Phase 7 modules, two scripts, and two new test files ? passed.
- Both `scripts/freeze.py --help` and `scripts/evaluate_test.py --help` ? passed.
- Notebook JSON ? valid; 44 unique cell IDs; all 3 Phase 7 code cells parsed with Python AST.
- `git diff --check` ? passed. Git noted that the notebook working-copy CRLF will normalize to LF on the next Git operation.
- `help()` inspection confirmed installed APIs for SciPy `binomtest` and exact `proportion_ci`, Statsmodels `mcnemar`, `cochrans_q`, and `multipletests`. No deprecated `return_object` argument is passed to `cochrans_q`.
- Known-answer check: `binomtest(783, 1500, p=0.5, alternative="greater")` returned **0.046628450539476056**.
- Exact command outputs and installed versions are recorded in `reports/phase_7_checks.json`.

## Outputs

- `src/aidetect/freeze.py`
- `src/aidetect/evaluate.py`
- `src/aidetect/stats_tests.py`
- `scripts/freeze.py`
- `scripts/evaluate_test.py`
- `tests/test_freeze_evaluate.py`
- `tests/test_stats_tests.py`
- `notebooks/colab_runner.ipynb`
- `reports/practical_target.md`
- `reports/phase_7_checks.json`
- `reports/phase_7_report.md`
- Updated `README.md` and `AGENTS.md`

No `artifacts/r2_fast_rf_t4/FREEZE.lock`, `artifacts/r2_fast_rf_t4/EVALUATION_STARTED.json`, `reports/r2_fast_rf_t4/results.json`, `reports/r2_fast_rf_t4/results.md`, or `reports/r2_fast_rf_t4/test_predictions.csv` was created in this checkout. The integration tests created temporary locks/results only inside pytest temporary repositories.

## Checks for the team in Colab

1. Review the completed R2 Phase 6 outputs selected by `config_r2_fast_rf_t4.yaml` (hash `46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766`). Agree the practical-use target with the supervisor; complete `reports/r2_fast_rf_t4/practical_target.md`, set `Status: Agreed`, and commit the target and reviewed Phase 6 outputs to `main`.
2. Run `phase-7-preflight`. It requires a clean `main` checkout, validates/restores the six ignored model files only from `/content/drive/MyDrive/aidetect/r2/review_backup/r2_fast_rf_t4_training_20261031T093525Z/artifacts/models` using `reports/phase_6_fast_rf_t4_backup_hashes.json`, and reports the agreed threshold. It does not open test feature rows.
3. Run `phase-7-freeze`, inspect `artifacts/r2_fast_rf_t4/FREEZE.lock`, then commit and push it before test access:

   ```text
   git add artifacts/r2_fast_rf_t4/FREEZE.lock
   git commit -m "Freeze Phase 7 test evaluation inputs"
   git push origin main
   git status --short --branch
   ```

4. Only on the clean checkout containing the committed lock, run `phase-7-evaluate-once` or `python scripts/evaluate_test.py --config config_r2_fast_rf_t4.yaml`. Run it once. It writes `artifacts/r2_fast_rf_t4/EVALUATION_STARTED.json`, `reports/r2_fast_rf_t4/results.json`, `reports/r2_fast_rf_t4/results.md`, and `reports/r2_fast_rf_t4/test_predictions.csv`.
5. Review all six metrics and the H1/H1a/H1b decisions. Hand-check at least one accuracy from its confusion matrix using `(TP + TN) / n`. After a successful run, commit and push the marker/results on `main`. If the marker exists after an interrupted or failed attempt, do not rerun the evaluator; preserve the marker and contact the supervisor.

## Assumptions and consequences

- Phase 5 test batches use the active feature config or the recorded legacy config with the unchanged feature-contract hash. The freeze/evaluation checks fail closed if batch provenance or schema differs; the team must resolve the provenance without changing the frozen split or bypassing the lock.
- Colab Drive is mounted and has room for six model backups. If it is unavailable, the Phase 7 notebook cannot guarantee model recovery across runtime restarts; do not create the lock until the backup manifest is verified.
- `reports/r2_fast_rf_t4/practical_target.md` records an accuracy fraction from 0 to 1, rationale, agreeing team members, supervisor, and ISO decision date. The freeze refuses placeholders or an unsigned target.
- A process interruption after the evaluation marker is created may leave the evaluation incomplete. The guard intentionally blocks a rerun; any recovery requires a supervisor decision outside this implementation.

## Open questions and unverified items

- Real Phase 6 feature batches, split file, model files, and training results are not present in this checkout.
- The team/supervisor practical-use threshold and sign-off are pending.
- The real test batch provenance, model compatibility, freeze lock, and single held-out evaluation have not been run.
- No test metrics, hypothesis outcomes, or research conclusions are available.
- GitHub Actions for the Phase 7 commit has not yet been checked.

## STOP

Phase 7 code and local synthetic checks are complete. Wait for the team to complete the real Colab gates, run and review the single evaluation, and provide the actual outputs. Do not start Phase 8 until those checks are reviewed and the user explicitly authorizes `PROCEED TO PHASE 8`.

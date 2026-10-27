# Phase 5 report

Status: **Pixel baseline and resumable feature-extraction code are complete and locally verified. Real-data extraction, batch timings, split row counts, and remote CI remain pending team checks.**

## Done

- `src/aidetect/pixel_baseline.py`: added `extract_pixel_features(x, size=32)`. It accepts the shared `preprocess_image` array, downsizes with Pillow LANCZOS without decoding or re-preprocessing, clips the normalized result to `[0, 1]`, and returns a flattened 1,024-value float64 vector by default.
- `src/aidetect/feature_extraction.py`: added one-image summaries and resumable Parquet batch writing. Each summary preserves RAPSD, normalized entropy, cumulative non-DC energy indexed by integer radius cutoffs, peak count/strength for k in `[3, 4, 5, 6, 8]`, and pixel features. Existing batches are metadata-checked and skipped.
- `src/aidetect/feature_extraction.py`: added the official batch reader. `train` and `val` are readable for Phase 5 checks; `test` maps to `test_sealed` and raises `PermissionError` unless `artifacts/FREEZE.lock` exists. Test extraction itself remains deterministic write-only; resume checks inspect batch metadata and never load test feature values.
- `scripts/extract_features.py`: added the split command. It joins the active manifest to immutable `splits.csv`, verifies `artifacts/splits.sha256`, preprocesses each selected image once, and prints row/batch/timing results.
- `config.yaml`: recorded the Phase 5 peak-threshold list under `extraction.peak_k_mad`; the active configuration SHA-256 is `7ac0df38870f031ee724c7b9b1610d12b02890bd9fd2760adb1215facc498d0f`.
- `notebooks/colab_runner.ipynb`: replaced the Phase 5 placeholder with preparation, train/validation extraction, write-only test extraction, and train/validation/seal-gate checks. It does not inspect test feature values.
- `tests/test_pixel_baseline.py` and `tests/test_feature_extraction.py`: added synthetic checks for the 1,024-value baseline, finite summary components, direct-versus-lookup energy ratios, resumable batches, the sealed reader, and the command-line manifest/split-hash path.
- `README.md`, `AGENTS.md`, and `src/aidetect/__init__.py`: documented the Phase 5 architecture, the prescribed pixel-baseline resize exception, the test-set boundary, and the current unverified real-data state.
- `reports/phase_5_checks.json`: recorded the exact local environment, commands, actual summaries, signature inspections, synthetic gates, and pending real-data checks.

## Tests

- Exact focused command: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_pixel_baseline.py tests/test_feature_extraction.py`
- Focused passed / failed / skipped: **11 / 0 / 0**
- Focused actual summary: `11 passed in 3.52s`
- Exact regression command: `.\\.venv\\Scripts\\python.exe -m pytest -q`
- Regression passed / failed / skipped: **105 / 0 / 0**
- Regression actual summary: `105 passed in 28.59s`
- Exact dependency check: `.\\.venv\\Scripts\\python.exe -m pip check`
- Actual summary: `No broken requirements found.`
- `py_compile` passed for `pixel_baseline.py`, `feature_extraction.py`, and `scripts/extract_features.py`; the script help command also completed successfully.
- Notebook JSON, unique IDs, and AST parsing passed for the existing Phase 3/4 cells and the four Phase 5 code cells.
- Installed signatures were inspected for NumPy FFT functions, Pillow `Image.resize`, and `pyarrow.parquet.write_table`; the signatures are recorded in `reports/phase_5_checks.json`.
- During implementation, one initial command-path test failed because its synthetic frozen split accidentally included `rel_path`, causing a merge suffix. The fixture was corrected to match the real `splits.csv` schema; the final focused and full runs above pass.

## Outputs produced (paths)

- `src/aidetect/pixel_baseline.py`
- `src/aidetect/feature_extraction.py`
- `scripts/extract_features.py`
- `tests/test_pixel_baseline.py`
- `tests/test_feature_extraction.py`
- `config.yaml`
- `notebooks/colab_runner.ipynb`
- `reports/phase_5_checks.json`
- `reports/phase_5_report.md`
- Updated `README.md`, `AGENTS.md`, and `src/aidetect/__init__.py`

No real `features_root/{train,val,test_sealed}` batches, research split, model, evaluation output, or test lock was created locally. The Parquet outputs described by the command are planned Drive outputs until the team runs Colab.

## Checks for the team to run in Colab

1. Clone/update `main`, install the pinned requirements, run the setup/test cells, and confirm the Phase 2 audit and immutable split outputs already exist. Do not run Phase 5 with a changed audit or regenerated `splits.csv`. Confirm `artifacts/FREEZE.lock` is absent; Phase 7 creates it.
2. Run the notebook's Phase 5 preparation cell. Confirm the active manifest joins one-to-one with `data_root/splits.csv`, and record the exact frozen per-split class counts.
3. Run the train and validation cells, or the equivalent commands:

   ```text
   python scripts/extract_features.py --split train
   python scripts/extract_features.py --split val
   ```

   Confirm the printed rows match `splits.csv`, the configured batch size is 500, each batch reports a timing, and the output metadata contains the current config hash and `artifacts/splits.sha256` value.
4. Run the test write-only cell:

   ```text
   python scripts/extract_features.py --split test
   ```

   Confirm the rows/batches reported match the test assignments and files are under `features_root/test_sealed/`. Do not call `read_feature_batches(..., "test")`, `pandas.read_parquet`, or exploratory feature inspection on that folder before the Phase 7 freeze.
5. Run the Phase 5 checks cell. It must report zero NaN/Inf values for train and validation, exact row-count matches for those readable splits, and an expected `PermissionError` for the sealed test reader. Preserve the command output and batch timings in the team record.
6. Check GitHub Actions on `main`; local success does not verify remote CI. Do not train, tune, select a model, freeze, or evaluate the test set in Phase 5.

## Assumptions I made

- The Phase 3 output is a square, finite float64 array in `[0, 1]`. If the shared preprocessing contract changes, update its version/config first and treat all downstream hashes as invalid.
- The preprocessed side is at least the 32-pixel baseline size. Smaller inputs are rejected rather than upscaled, consistent with the main image policy.
- For integer `r0`, assigning each FFT bin to `ceil(radius)` makes the saved cumulative-energy lookup exactly match the direct `radius <= r0` and `radius > r0` masks, including corner bins. RAPSD continues to use the Phase 4 ring range and excludes corners as before.
- Peak count and maximum z-strength for the five configured thresholds are the persisted peak-map summaries required to reconstruct the Phase 4 peak outputs. If later analysis needs pixel-level peak locations, that is a new requirement and must not alter the sealed experiment silently.
- Test feature creation before freeze is allowed only as deterministic write-only generation. The official reader gate and the notebook check preserve the no-test-inspection rule; any direct bypass would violate the project protocol.
- Existing batch metadata is read during resume to reject changed config, split hash, split name, batch index, or ID contents. This is provenance validation, not test feature-value access.

## Open questions / anything I could not verify

- The team has not supplied real Colab outputs for feature row counts, batch timings, preprocessing failures, or train/validation finite-value counts.
- No real research split or split hash is present locally, so no real feature batch was generated and no test feature value was accessed.
- Dataset availability, generator counts, audit/probe decisions, duplicate groups, supervisor decisions, and remote GitHub Actions results remain unverified.
- Phase 6 must consume these summaries without opening `test_sealed`; Phase 7 must create and commit the freeze lock before the sealed reader is used.

## STOP

Phase 5 is complete locally. Waiting for team checks and your explicit **`PROCEED TO PHASE 6`** approval before starting Phase 6.

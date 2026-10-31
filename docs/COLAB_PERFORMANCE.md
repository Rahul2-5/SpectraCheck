# Faster Colab execution

Latest T4 profile: **`config_r2_fast_rf_t4.yaml`**. After measuring an eight-minute
pixel RF candidate in the first fast run, the user explicitly approved **100–300
RF trees and max_features='sqrt'** for both feature sets. Other RF ranges and all
SVM/XGBoost settings stay unchanged. This further narrows the original RF search;
report it as a protocol amendment. Independent supervisor sign-off is unverified.
The earlier fast CPU/T4 configs remain available as historical alternatives and
can be substantially slower on pixel RF. Do not switch profiles after test access.

For the active pre-test run, `scripts/migrate_fast_rf_checkpoints.py` verifies the
exact parent contract and permits reuse of completed SVM/XGBoost pipelines and
the unaffected feature choice only. Both RF searches start under the revised
ranges; their old candidates/models are excluded. Old evidence stays intact.
Migration requires a stopped original writer, separate output roots and a new
checkpoint folder. Do not rerun migration or a completed training run.

The user approved the fast training amendment on 31 October 2026. Use
`notebooks/colab_fast_training.ipynb` after the normal Drive mount and pinned
environment setup. The canonical runner also contains the opt-in fast section.
Run only the authorized section, never **Run all**.

## Phase 6

Original fast default: `config_r2_fast.yaml` on a **CPU** runtime. Optional:
`config_r2_fast_t4.yaml` on a T4 runtime, after a timing comparison. The T4
accelerates XGBoost only. RF and RBF SVM still use CPU. GPU allocation is not
needed for any other implemented phase.

| Setting | Historical R2 | Approved fast profile |
| --- | --- | --- |
| Shared stratified CV folds | 5 | 3 |
| Search candidates per model | 30 | 10 |
| Total search fold fits | 900 | 180 |
| SVM sigmoid calibration folds | 5 | 3 |
| Models / seed | six / 42 | six / 42 |
| Search ranges, feature-choice grid, split and preprocessing | original | unchanged |

The reduced budget may select different or less optimized models. Report this
deviation; do not combine original and fast results or select between profiles
using held-out test scores. Independent supervisor sign-off remains unverified.
Original `config_r2.yaml`, frozen split, extraction hashes and audit history stay
unchanged. Fast profiles explicitly accept the original Phase 5 hash only while
the exact feature contract matches. Historical split receipts are copied byte for
byte, never regenerated. No re-extraction or acquisition is necessary.

```python
# From the checkout, after pinned setup. This verifies reviewed Phase 5 evidence.
subprocess.run([sys.executable, 'scripts/prepare_fast_training.py',
                '--config', 'config_r2_fast.yaml'], check=True)
# TRAIN-only timing; add --include-cuda only on a GPU runtime.
subprocess.run([sys.executable, '-u', 'scripts/benchmark_training.py',
                '--config', 'config_r2_fast.yaml', '--output', '/content/timing.json'], check=True)
# Full authorized Phase 6; separate verified Drive checkpoints and backups.
subprocess.run([sys.executable, '-u', 'scripts/run_fast_training.py',
                '--config', 'config_r2_fast.yaml'], check=True)
```

Search workers are capped by available CPU and host memory, at most two in the
shipped profiles. Each estimator/BLAS worker uses one thread. CUDA searches use
one worker to avoid competing GPU fits; a synthetic probe rejects silent CPU
fallback. Timing comparisons must report the actual runtime hardware. No fixed
20–40% runtime or minute estimate is guaranteed; a representative fit is not a
full search benchmark.

Train/validation Parquet files can be copied to `/content/aidetect_train_val_cache`
with byte verification. The cache never enumerates or copies `test_sealed`.
The frozen split and audit continue to be read from their authoritative paths.
Cache files changing on disk cause an error. Local cache and outputs disappear
when Colab replaces the runtime; checkpoints and verified backups persist on Drive.

Each completed search candidate saves its sampled parameters and all fold scores
before the next candidate starts. Resume binds exact configuration, train/val
bytes, folds, code and versions. A partial candidate restarts; completed candidate
scores and completed calibrated models are reused. Feature selection still
checkpoints only after its complete grid. No search-space pruning or early stopping
is introduced. RF/XGBoost reuse the search's already fitted best pipeline instead
of fitting it a second time.

Outputs: `artifacts/r2_fast`, `reports/r2_fast`, and Drive
`aidetect/r2/r2_fast_phase6_checkpoints` (T4 uses `r2_fast_t4`). The independent
runner backs up the six completed models, reports, config and byte hashes under
`aidetect/r2/review_backup/r2_fast_training_TIMESTAMP`. Check `start.json`,
`training.log`, candidate receipts and checkpoint `status.json`. The launch lock
prevents concurrent fast-profile runners; never remove a stale lock until its old
process/runtime is confirmed stopped. Never start another original or fast job
while any training subprocess is active. Completed intact results refuse reruns.

## Other phases

| Phase | Efficient execution without changing the research protocol |
| --- | --- |
| 0 | CPU; install pinned dependencies once per runtime, restart if requested, then run checks once. |
| 1 | CPU; reuse the completed acquisition. Upstream source scans/transfers remain an unavoidable downloader limitation. |
| 2 | CPU; reuse the reviewed audit and immutable split. Never recreate them for a speed change. |
| 3–4 | CPU; run only the requested train-only review and retain its checksummed figures. Small 256×256 NumPy FFTs do not use T4 through this code. |
| 5 | CPU; optional `scripts/extract_features.py --config config_r2.yaml --split train --workers 2` (also val/test only when authorized). Two threads overlap Drive reads and Pillow/NumPy work; deterministic ordering, source checks and write-only test handling remain. Resume re-verifies source bytes. Existing completed R2 extraction needs no rerun. |
| 7 | CPU; freeze and evaluate once through the existing gates only when explicitly authorized. Never benchmark by opening the held-out set. |
| 8 | Local laptop CPU; reuse the reviewed serving bundle and shared inference path. |
| 9 | CPU; build the report from saved results only, when authorized. No retraining/evaluation for report generation. |

Changing Colab runtime type replaces the runtime and loses local files/processes.
Check active jobs and Drive backups first; mount Drive and restore the checkout
and pinned environment afterward. No paid plan, GPU requirement, or dataset
download is added. Phase 7 remains unauthorized by this performance task.

XGBoost's pinned-version GPU interface uses `tree_method='hist', device='cuda'`:
[official XGBoost 3.2 GPU documentation](https://xgboost.readthedocs.io/en/release_3.2.0/gpu/).

# Phase 6 performance amendment — 31 October 2026

The user approved a faster CPU/T4 execution plan, including reducing shared CV
from five to three folds and randomized search from 30 to ten candidates for each
of six models. SVM sigmoid calibration follows the three shared-fold budget.
Original R2 source/data/preprocessing/search ranges/feature-choice grid/seed remain.
Supervisor acceptance remains unverified. No Phase 7 execution is authorized.

## Implemented

Separate CPU/T4 YAML profiles, separate artifacts/reports/checkpoints, byte-for-byte
historical split-receipt staging behind the reviewed Phase 5 evidence, a TRAIN-only
timing command, train/validation-only verified local caching, CPU/memory bounded
CV parallelism with single-thread estimator/BLAS execution, serialized CUDA search
and an explicit GPU fallback guard. The fast search uses seeded ParameterSampler
and GridSearchCV per candidate with the same fold-local scaling/scoring and
first-best tie behavior as RandomizedSearchCV. Each completed candidate's scores
are committed/checksummed; interrupted candidates restart. Per-model checkpoints
and exact resume provenance remain. RF/XGBoost no longer repeat a completed best
pipeline refit. An independent Colab runner preserves completed outputs on Drive.

Phase 5 has optional bounded image extraction threads (`--workers 2`) with the
same output ordering/values/source-integrity checks. Existing completed extraction
is reused. Other-phase runtime guidance is in `docs/COLAB_PERFORMANCE.md`; no
completed data phase was rerun, and no research gate was relaxed.

## Local verification

Initial focused tests: 42 passed in 24.98s. New runtime/extraction focused suite:
22 passed in 8.67s. Full synthetic regression:
`.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-fast-full -p no:cacheprovider`:
**194 passed in 68.98s**. The full suite ran outside the Windows sandbox because
the existing local API socket test previously hung inside it. `pip check` passed.
Meaningful checks compare all three candidate searches against sklearn's seeded
RandomizedSearchCV, interruption/resume and corruption rejection, complete six-model
training with cache, no sealed-test cache, worker caps, and serial/threaded image
feature parity plus source-byte tampering. Notebook/code parsing also checked.

## Live observation before migration

The current browser Colab session exposed two CPU cores and a Tesla T4 at 0%
utilization / 0 MiB. No train_models.py process was present. Original run record
`phase_6_training_20261031T031201Z/start.json` still says running (PID 16432), but
the current replacement runtime has no such job or completed local results.
Its preserved log contains only an import warning. This does not establish the
old run completed. No completed checkpoints were observed. Original evidence was
left unchanged. Fast-profile readiness passed in the same live Colab runtime. Focused Colab
regressions: **51 passed, one optional bottleneck warning, in 31.47s**. TRAIN-only
benchmark completed with exit 0; XGBoost's booster configuration verified
`cuda:0` rather than silently falling back. Evidence directory:
`aidetect/r2/review_backup/performance_benchmark_20261031T085542Z` (`timing.json`,
`benchmark.log`, `pytest.txt`). One seeded candidate fitted 2,332 TRAIN rows:

| Feature set / model | CPU seconds | T4 seconds |
| --- | --- | --- |
| Frequency SVM | 0.621 | CPU only |
| Frequency RF | 2.567 | CPU only |
| Frequency XGBoost | 1.170 | 0.478 |
| Pixel SVM | 2.399 | CPU only |
| Pixel RF | 11.950 | CPU only |
| Pixel XGBoost | 23.816 | 2.390 |

These are representative individual fit timings, not full-search speedups or
accuracy measurements. Because the existing T4 accelerated XGBoost, selected the
approved T4 profile for this live run; CPU remains the default downloadable profile.
The live Phase 6 readiness/launch/status cells now use the fast helpers. Fast runner
PID **7238** started `2026-10-31T08:58:07.838561+00:00` (**14:28 IST**), explicitly
using `config_r2_fast_t4.yaml`. Its config digest is
`ca5ef99bd323928374356bf13e0ceff9cec729d3336bb92055082a633048e9a5`; original
feature config/split hashes matched reviewed R2 evidence. Verified the worker
remained active and checkpoint status reached `feature_selection` at
`08:58:14.959139Z`, test_features_read=false. Drive checkpoints:
`aidetect/r2/r2_fast_t4_phase6_checkpoints`; independent runner backup/log:
`aidetect/r2/review_backup/r2_fast_t4_training_20261031T085807Z`. Original record
and original config were not overwritten. Research completion remains pending.

Additional local guard/benchmark tests after the full suite: **11 passed in 7.34s**
(includes three added cases for TRAIN-only benchmark, CUDA fallback rejection and
changed-profile rejection before Drive access). No source behavior changed after
the 194-test full regression. No new dependency or installed library pin changed.

## Limitations / stop

180 versus 900 search fold fits is an 80% reduction in fit count, not a guaranteed
80% wall-clock reduction. Feature selection/calibration/refits/Drive overhead
remain, and Colab hardware varies. Reduced tuning can change accuracy and model
selection. Never select a profile using test results. Keep test values sealed;
review actual Phase 6 results and practical target before separately authorizing
Phase 7. No real accuracy, selected model or training completion is claimed.

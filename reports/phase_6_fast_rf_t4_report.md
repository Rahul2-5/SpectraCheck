# R2 Phase 6 completion — approved RF/T4 profile, 31 October 2026

**Training and independent machine verification are complete.** All six matched
pipelines are saved and checksummed on Drive. The validation-selected primary
model is **frequency RBF SVM**, with **80.67% validation accuracy (605/750)**,
ROC-AUC **0.883072** and Brier score **0.135568**. These are validation results;
no held-out test scores or significance claims are available.

## Done / final protocol

The user explicitly approved 3 shared stratified CV folds and 10 randomized
candidates per model, then selected **Switch to faster RF search** after the
pixel RF's second broad-search candidate took about eight minutes. Final config:
`config_r2_fast_rf_t4.yaml`. Both RFs use **100–300 trees and max_features=sqrt**;
other RF ranges and all SVM/XGBoost ranges remain unchanged. SVM sigmoid calibration
uses 3 folds. All six pipelines retain fold-local scaling and seed 42. This
amends the original PRD's 5-fold/30-candidate budget and broad RF ranges, and may
change the selected settings/accuracy. Supervisor acceptance is unverified.

The data remains the reviewed R2 Tiny-GenImage cohort: real ImageNet photos and
Stable Diffusion v1.5 images, 3,500 train / 750 validation; 750 test images remain
sealed. Shared version-2 256×256 centre-patch/JPEG treatment, all feature values,
frozen memberships, source receipts and original configs are unchanged. No new
acquisition, audit, split or extraction was run.

Unchanged frequency SVM, frequency XGBoost and pixel SVM fits were carried over
from the first fast T4 run using exact source code/environment/config/input/fold
contract checks. Their 30 candidate receipts were rechecked. Both RF searches
were rerun under the amended ranges; **no old RF model/candidate was reused**.
The final pixel XGBoost was trained on T4. The final run has 30 new candidate
receipts plus 30 verified reused receipts: 60 candidates × 3 folds = 180 CV fits
contributing to the final comparison. The discarded RF work is retained as
superseded evidence and was not used for selection.

Selected frequency settings: **r0=120, k_mad=5**, from the unchanged 75-candidate
train-fit/validation-score grid. Primary selection uses highest frequency-model
validation accuracy with the configured SVM/RF/XGB tie order.

## Actual validation results

CV percentages are mean ± SD across the three shared training folds. Validation
metrics were independently recomputed after loading all six saved pipelines and
matched the saved values to absolute tolerance 1e-12.

| Pipeline | CV accuracy (%) | Validation accuracy | Validation AUC | Brier score |
| --- | --- | --- | --- | --- |
| frequency_svm | 83.66 ± 2.10 | 80.67% | 0.883072 | 0.135568 |
| frequency_rf | 77.49 ± 2.83 | 76.80% | 0.856092 | 0.154932 |
| frequency_xgb | 79.34 ± 2.00 | 78.93% | 0.869468 | 0.155469 |
| pixel_svm | 64.09 ± 1.38 | 66.00% | 0.723371 | 0.212452 |
| pixel_rf | 67.09 ± 0.78 | 66.67% | 0.750471 | 0.208623 |
| pixel_xgb | 66.86 ± 0.71 | 67.33% | 0.762830 | 0.199505 |

Full best parameters, provenance and package versions are in
`artifacts/r2_fast_rf_t4/training_results.json`. RF/XGBoost frequency importance
CSVs and 300 dpi plots are in `reports/r2_fast_rf_t4/figures/`; both plots were
visually inspected. Importances are descriptive and were not used for selection.

## Actual timing and hardware

The revised run started **15:05:26 IST** and finished **15:15:09 IST**:
**583.40 seconds / 9 minutes 43 seconds**, reusing the three verified models.
Independent verification completed at **15:15:25 IST**. The complete workflow
from the first fast launch at **14:28:07 IST**, including the superseded broad RF
attempt and switch, took **2,821.56 seconds / 47 minutes 2 seconds**. The revised
run time is not a from-scratch six-model benchmark. The original full-budget
run's completion remains unverified; no measured whole-pipeline speedup factor
against it is claimed.

Hardware: two CPU cores plus Tesla T4. SVM/RF and feature choice use CPU, with
at most two CV workers and one estimator/BLAS thread per fit. XGBoost uses
`tree_method=hist, device=cuda`, one search worker to avoid GPU contention;
actual CUDA capability was checked by a synthetic probe and TRAIN-only timing.
The selected frequency SVM itself uses CPU. No paid service, new dependency or
GPU requirement for other phases was introduced.

## Verification / tests

- Full local synthetic regression: `.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-rf-amendment-full -p no:cacheprovider`: **204 passed in 74.56s**. The existing local API socket test required execution outside the Windows sandbox.
- Focused amendment/migration/runtime suite: **18 passed in 12.69s**, including unchanged-model prediction parity and rejection of changes to seed/folds/data/preprocessing/SVM/RF depth. Local `pip check` passed.
- Live Colab focused suite before migration: **51 passed in 31.47s**, one optional bottleneck-version warning. Real amended preflight and guarded migration succeeded.
- CI for source commit `3e1149d2037f9261c06f6999a12dbda90d3ca942`: frontend and Python 3.11/3.13 jobs all succeeded in [run 41189258815](https://github.com/Rahul2-5/SpectraCheck/actions/runs/41189258815). An earlier evidence commit's frontend browser step failed; both Python jobs passed there, and the latest source run passed all three jobs. No result is inferred from the earlier failure.
- Real completion verification checked all six model/receipt hashes, all 60 eligible candidate receipts and 3 scores per candidate, 20 backup-file hashes, every published output, original and revised exact code/input/version/fold contracts, config/split hashes, no active/stale writer locks, 3,500/750 rows, primary-model selection and both revised RF parameter ranges.
- Train/validation metrics for every saved model were reproduced; no test feature values, test predictions, freeze lock or evaluation marker were opened/created.
- Review export downloaded and unpacked with path containment checks; no model binary, source image or feature batch is in the Git/export pack. The downloaded completion JSON matches the exact Drive bytes.

## Evidence and outputs

- Config SHA-256: `46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766f`.
- Original feature config SHA-256: `ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6`.
- Frozen split SHA-256: `1e823b5df253951bc2a7221b618fc7c0e5716f85066d509a3d9442c56ccfef93`.
- Final checkpoint contract SHA-256: `eb2767373ddc9b1e140a25162fe8b2164b12eea9436f2260fc1c67aa4bf666e1`.
- Completion record: `reports/phase_6_fast_rf_t4_checks.json`, SHA-256 `5eb86a77101a2fdd7afc0967b3cf1cd388781415236782f6deb4549289296025` (same bytes on Drive).
- Review ZIP SHA-256: `065151c7b0a2ad0fcd2a5e0a6193556cdc55697cdca2f6d36149472e988cd108`.
- Models, config, reports and checksummed backup on Drive: `aidetect/r2/review_backup/r2_fast_rf_t4_training_20261031T093525Z`; the six models are in `artifacts/models` within that directory.
- Persistent Drive checkpoints: `aidetect/r2/r2_fast_rf_t4_phase6_checkpoints`; original checkpoint folder and the recorded RF-amendment decision remain preserved.
- Local review metadata: `artifacts/r2_fast_rf_t4/`, `reports/r2_fast_rf_t4/`, and `reports/phase_6_fast_rf_t4_*` records/logs. Exact live verification code: `notebooks/verify_fast_phase6_cell.py` (specific to this run).

## Team review / assumptions / stop

Machine completion is verified; team/supervisor acceptance and an agreed practical
use target are still pending. Review this narrower search protocol, the six CV/
validation rows, selection records, figures and original audit/amendment history.
The original raw shortcut audit remains blocked; the passing audit applies only
to treated R2 inputs. Findings are limited to the single-generator patch study.

**STOP before Phase 7.** Do not rerun training, choose another profile using test
results, create FREEZE.lock, or evaluate test features. Agree the practical target
for the configured report root and obtain explicit next-phase authorization first.

# SpectraCheck model card

Status: **Template; held-out performance and approvals are pending.**

## Model identity

- Primary model: `frequency_svm` (calibrated RBF SVM).
- Experiment: `config_r2_fast_rf_t4.yaml`.
- Configuration SHA-256: `46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766`.
- Feature selection: `r0=120`, `k_mad=5`.
- Model artifact: `artifacts/r2_fast_rf_t4/models/frequency_svm.joblib`.
- Artifact SHA-256: `f5eb064260349873ebeddb16dc090a0aa842ce7b844bfd0c0e203d97d578bbda`.

## Intended use

This is a local research prototype for comparing frequency-domain and pixel
baselines on the reviewed Stable Diffusion v1.5 R2 cohort. It is not an
authenticity service, production detector or cross-generator guarantee.

## Data and preprocessing

The R2 cohort contains real ImageNet photographs and Stable Diffusion v1.5
images. The shared version-2 preprocessing applies EXIF orientation, fixed
256x256 centre cropping with no upscaling, one JPEG quality-96 4:4:4 encoding,
grayscale conversion and float64 normalization to `[0, 1]`. Test data is sealed
until the authorized single evaluation.

## Features and training

The model consumes 68 ordered frequency features: energy ratio, spectral
entropy, 64 log-RAPSD ring values, peak count and peak strength. The centred FFT
uses the configured Hann window and epsilon. Training uses seed 42, fold-local
scaling, three shared CV/calibration folds and the recorded search contract.

## Validation evidence

The saved validation accuracy is 80.67% (605/750), ROC-AUC 0.883072 and Brier
score 0.135568. These values do not establish held-out performance.

## Held-out performance

- Test accuracy: **TBD - populate only from committed Phase 7 results.**
- Confidence interval: **TBD**.
- ROC-AUC: **TBD**.
- Brier score: **TBD**.
- Statistical decisions: **TBD**.

## Limitations and risks

- One generator and one reviewed dataset scope.
- R2 is a treated-input amendment; the original raw shortcut audit remains a
  separate blocked finding.
- Performance may differ for other generators, cameras, screenshots and
  compression settings.
- Joblib/library compatibility is checked by the serving bundle; mismatches are
  warnings or fail-closed conditions, not silently ignored changes.

## Governance

- Practical target: proposed 0.80 validation accuracy; approval **PENDING**.
- R2 scope acceptance: **PENDING**.
- Supervisor approval: **PENDING**.
- Phase 7 freeze/evaluation authorization: **PENDING**.

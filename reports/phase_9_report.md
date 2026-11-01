# SpectraCheck Phase 9 - final research report

Status: **Template. Held-out evaluation and final approvals are unavailable.**

## 1. Executive summary

SpectraCheck studies whether handcrafted frequency-domain features distinguish
real images from Stable Diffusion v1.5 images in the reviewed R2 cohort. The
primary model selected on validation data is the frequency RBF SVM. No held-out
test conclusion is recorded in this template.

## 2. Research scope and protocol

- Dataset scope: reviewed R2 Tiny-GenImage cohort, real ImageNet photographs and
  Stable Diffusion v1.5 images.
- R2 treatment: version-2 fixed 256x256 centre patch, common JPEG quality 96,
  4:4:4 subsampling, then grayscale conversion.
- Split: 3,500 train, 750 validation and 750 sealed test images, subject to the
  recorded split contract.
- Labels: real = 0; AI-generated = 1.
- Test access: **not documented here; populate only from the committed Phase 7
  evaluation record.**

## 3. System architecture

Training and evaluation use the Python `aidetect` package. The shared
`preprocess_image` entry point applies EXIF orientation, the R2 fixed-centre
encoding contract and float64 normalization to `[0, 1]`. Frequency extraction
computes a centred FFT power spectrum, energy ratio, non-DC spectral entropy,
64 log-RAPSD rings and robust local-peak summaries, producing a stable 68-value
vector. The application uses the same preprocessing and extraction path through
the FastAPI serving adapter; React is the local frontend.

## 4. Phase 6 model selection

Configuration: `config_r2_fast_rf_t4.yaml` (SHA-256:
`46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766`). Feature
selection chose `r0=120`, `k_mad=5`. The selected primary is `frequency_svm`.

Recorded validation evidence:

| Model | CV accuracy | Validation accuracy | Validation ROC-AUC | Validation Brier |
| --- | ---: | ---: | ---: | ---: |
| frequency_svm | 83.66 +/- 2.10% | 80.67% (605/750) | 0.883072 | 0.135568 |
| frequency_rf | 77.49 +/- 2.83% | 76.80% | 0.856092 | 0.154932 |
| frequency_xgb | 79.34 +/- 2.00% | 78.93% | 0.869468 | 0.155469 |
| pixel_svm | 64.09 +/- 1.38% | 66.00% | 0.723371 | 0.212452 |
| pixel_rf | 67.09 +/- 0.78% | 66.67% | 0.750471 | 0.208623 |
| pixel_xgb | 66.86 +/- 0.71% | 67.33% | 0.762830 | 0.199505 |

Source: `reports/phase_6_fast_rf_t4_report.md` and the corresponding R2
metadata. These are validation results, not held-out results.

## 5. Held-out evaluation results - complete after Phase 7

Source file: `reports/r2_fast_rf_t4/results.json`
Source SHA-256: **TBD**
Evaluation marker: **TBD**
Test predictions SHA-256: **TBD**

| Model | Test accuracy | 95% interval | ROC-AUC | Brier | H1/H1a/H1b decision |
| --- | ---: | --- | ---: | ---: | --- |
| frequency_svm | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| frequency_rf | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| frequency_xgb | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| pixel_svm | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| pixel_rf | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| pixel_xgb | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Do not replace `TBD` from validation metrics or application test requests.

## 6. Figures and tables

Existing Phase 6 descriptive frequency-importance figures are under
`reports/r2_fast_rf_t4/figures/`. Final held-out figures and their hashes:

- Figure/table: **TBD**; source: **TBD**; SHA-256: **TBD**.
- Figure/table: **TBD**; source: **TBD**; SHA-256: **TBD**.

## 7. Limitations and deviations

See [`deviations.md`](deviations.md) for the R2 scope amendment, faster RF/T4
search profile and implementation decisions. The study is limited to one
generator and the approved treated-input domain; it is not an authenticity
guarantee for other generators, screenshots or compressed images.

## 8. Reproducibility and application

Follow [`../docs/REPRODUCIBILITY_GUIDE.md`](../docs/REPRODUCIBILITY_GUIDE.md)
and [`../docs/PHASE_9_APPLICATION_USAGE.md`](../docs/PHASE_9_APPLICATION_USAGE.md).
The supervisor-demo workflow has been verified through the browser, Vite proxy
and FastAPI using the designated `frequency_svm` artifact and a non-sealed local
image. It displayed a model name, probability meter and development-only
warning. This is software evidence only; the final reproduction commit, lock,
held-out results and environment hashes are **TBD**.

## 9. Approval record

- Practical target: proposed 0.80 validation accuracy; approval: **PENDING**.
- R2 treated-input scope acceptance: **PENDING**.
- Supervisor approval: **PENDING**.
- Final report review date: **TBD**.

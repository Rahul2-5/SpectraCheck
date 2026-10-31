# Project decisions

## 31 October 2026 - Narrower RF search explicitly approved

After observing the pixel RF's second candidate take about eight minutes, the
user selected **"Switch to faster RF search"**. `config_r2_fast_rf_t4.yaml`
uses **100–300 RF trees and max_features='sqrt'** for both feature sets, retaining
the remaining RF ranges, all SVM/XGBoost settings, six models, three shared folds,
ten candidates, seed 42 and unchanged R2 inputs. This supersedes the original
RF search ranges for the new profile only. Supervisor sign-off is unverified.

A guarded migration pins the original fast-run checkpoint contract, checks exact
code/environment/input/fold provenance, and reuses only completed identical SVM/
XGBoost pipelines plus the unaffected feature choice. Both RF searches restart
under the revised ranges; no RF model/candidate is reused. Old checkpoints and
records remain intact. New outputs/checkpoints have separate roots. Synthetic
migration/runtime checks: 18 passed in 12.69s. Real migration/completion must be
verified separately. Test values remain sealed; no Phase 7 authorization.

## 31 October 2026 - Faster CPU/T4 execution approved by user

The user approved the proposed performance changes after requesting a list first.
Opt-in `config_r2_fast.yaml` / `config_r2_fast_t4.yaml` reduce training to three
shared CV folds and ten candidates per model (180 fold fits); sigmoid SVM
calibration uses three folds. Six models, search ranges, feature-choice grid,
seed 42, frozen split, preprocessing and sealed-test rules remain unchanged.
This is an explicit amendment to the original five-fold/30-candidate protocol;
original configs/records stay unchanged and fast results use separate roots.
Supervisor sign-off is unverified. Optional T4 acceleration applies to XGBoost
only; CPU is the default. See `COLAB_PERFORMANCE.md` and the performance report.
Other phases receive CPU/resume guidance and optional bounded Phase 5 extraction
threads; no completed data phase or held-out evaluation is authorized to rerun.

## 30 October 2026 - R2 fixed-patch amendment approved by user

The user replied "aproved" after being directed to the R2 amendment and its
explicit fixed-patch treatment and narrower audit domain. R2 is implemented as
an opt-in experiment in `config_r2.yaml`, with independent roots and the same
0.60 probe limit. The raw failed audit remains preserved. User approval is
recorded; independent supervisor sign-off remains unverified. This task covers
the revised Phase 2 implementation/audit/split, not training or held-out evaluation.
Use `notebooks/colab_r2.ipynb` and explicit `--config config_r2.yaml` commands.

## 30 October 2026 - Full GenImage metadata feasibility result

The user downloaded and supplied `genimage_metadata.csv` for inspection. Two
independent counting passes agree: 2,681,150 total metadata rows; 174,000 SD1.5
rows, all 512x512; only 51 real 512x512 entries across all folders, including 7
in the SD1.5 folder. The investigated GenImage candidate cannot meet draft R1's
proposed minimum. See [the feasibility report](../reports/genimage_metadata_feasibility_2026-10-30.md).
No image acquisition or revised experimental treatment is approved by this finding.

## 30 October 2026 - Revised data protocol draft requested

Following the live failed Phase 2 audit and unsuccessful geometry-matching
diagnostics, the user selected **"Prepare a revised data protocol"**.
See [the R1 proposal](REVISED_DATA_PROTOCOL_PROPOSAL.md). This authorizes drafting
only. Native-source feasibility, matched acquisition, source/license/capacity
limits and team/supervisor review remain pending; no dataset change, download,
preprocessing change or downstream run is approved by this decision.

## 27 October 2026 - React prototype

The user requested React instead of Streamlit. This replaces the original PRD's
Phase 8/D3 framework choice; the source PRD files remain historical references.

- Interface: local React frontend, planned under `app/frontend/`.
- Inference: local Python HTTP API, planned under `app/backend/`, calling the
  shared `aidetect` preprocessing, feature extraction, and frozen primary model.
- Display: Real/AI-generated label, calibrated probability, log spectrum, RAPSD
  plot, and the same single-generator limitations note specified in the PRD.
- Upload handling: in memory; no saved uploads. Clear errors for invalid or
  undersized images. Check model/library versions and prediction parity.
- Cost/scope: free tools, local laptop demo, no paid inference API or hosting.
- Timing: implement in Phase 8 after the research pipeline is ready. Choose and
  pin the frontend build tooling and Python API library then, with appropriate
  upload/inference tests and local run instructions.

Streamlit was removed from current direct requirements. Earlier environment
reports intentionally retain their historical installed versions and check counts.
No React app or Python HTTP endpoint has been implemented by this planning change.

## 28 October 2026 - Phase 8 implementation authorized before real-data review

The user explicitly chose: "Build Phase 8 code now; keep real-data checks pending."
This overrides the timing gate in the Phase 7 report for software development
only. No actual Phase 7 freeze/evaluation/review or model accuracy is implied.

Phase 8 now uses React/Vite and a FastAPI service. Raw request streaming keeps
uploads in memory. The service reuses shared Python preprocessing, features, and
evaluation probability handling and checks the frozen serving bundle. With no
real model artifacts, the frontend displays an unavailable-model state.

UI work applies the `emil-design-eng` and `mobile-native` skills: restrained
motion, clear feedback, keyboard/focus support, readable contrast, responsive
layouts, safe-area padding, retained zoom, touch-aware hover, and reduced motion.
The full research config/source remains unchanged by app-specific settings.

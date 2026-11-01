# Phase 9 documentation checklist

Status: **Not complete.** This checklist is a documentation structure only.
Held-out results, approvals and research conclusions must be added only from
committed Phase 7 outputs.

## Required deliverables

- [ ] Final research report: [`phase_9_report.md`](phase_9_report.md)
  - [x] Research question, scope and R2 protocol described.
  - [x] Phase 1-8 implementation evidence linked.
  - [ ] Phase 7 held-out metrics copied from `reports/r2_fast_rf_t4/results.json`.
  - [ ] Six-model results table and statistical decisions completed.
  - [ ] Limitations, deviations and conclusions reviewed by the team.
- [ ] Results tables and figures
  - [ ] Held-out accuracy intervals, ROC-AUC and Brier values.
  - [ ] Confusion matrices and prediction-error summaries.
  - [ ] H1, H1a and H1b results with exact-test decisions.
  - [x] Phase 6 validation table and frequency importance figures linked.
  - [ ] Figure-generation provenance and hashes recorded.
- [ ] Model card: [`model_card.md`](model_card.md)
  - [x] Model identity, preprocessing, features and intended scope.
  - [x] Validation evidence recorded without presenting it as test evidence.
  - [ ] Held-out performance and approval fields completed.
- [ ] Experimental deviations: [`deviations.md`](deviations.md)
  - [x] R2 treated-input scope and faster RF/T4 amendment recorded.
  - [x] React-over-Streamlit implementation decision recorded.
  - [ ] Team/supervisor acceptance and dates recorded.
- [ ] Reproducibility guide: [`../docs/REPRODUCIBILITY_GUIDE.md`](../docs/REPRODUCIBILITY_GUIDE.md)
  - [x] Environment, configuration, provenance and hash contracts described.
  - [ ] Final commit, freeze lock and result hashes added.
  - [ ] Reproduction commands marked with actual run evidence.
- [ ] Application usage guide: [`../docs/PHASE_9_APPLICATION_USAGE.md`](../docs/PHASE_9_APPLICATION_USAGE.md)
  - [x] React/FastAPI architecture and local setup described.
  - [x] No-model and fail-closed behavior documented.
  - [x] Explicit development-demo command and non-research labeling documented.
  - [ ] Reviewed frozen bundle activation and real-model smoke check completed.
  - [ ] Physical-device review recorded or explicitly marked unavailable.

## Evidence rules

- Validation metrics are limited to the Phase 6 train/validation evidence.
- Held-out metrics remain `TBD` until the single authorized Phase 7 evaluation
  produces committed results and predictions.
- Do not derive a test result from validation metrics, application requests or
  synthetic fixtures.
- Do not claim team agreement, supervisor approval, R2 scope acceptance or
  reproducibility until the decision is recorded with a date and evidence hash.
- Every final table and figure must name its source file and SHA-256 digest.

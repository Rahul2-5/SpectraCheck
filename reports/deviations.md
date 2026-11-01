# SpectraCheck experimental deviations

Status: **Template; decision records require team/supervisor completion.**

## R2 treated-input scope

The original raw-input shortcut audit was blocked. The approved R2 experiment
uses a separately documented treated-input domain: fixed 256x256 centre patches,
one common JPEG quality-96 4:4:4 encoding and the shared version-2 preprocessing
contract. The original raw finding is retained and must not be described as a
passed original audit.

- User implementation approval: recorded in project history.
- Team acceptance of the R2 scope: **PENDING**.
- Supervisor acceptance: **PENDING**.
- Decision date and rationale: **TBD**.

## Faster RF/T4 search profile

The final Phase 6 run used three shared CV/calibration folds and ten randomized
candidates per model. Both RF searches used the approved 100-300 tree range and
`max_features=sqrt`; the retained SVM/XGBoost settings, seed, six-model design
and frozen inputs were preserved. This differs from the original broader
5-fold/30-candidate plan.

- User approval: recorded in the Phase 6 completion evidence.
- Team/supervisor acceptance: **PENDING**.
- Final config SHA-256: `46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766`.

## Application framework decision

The application uses React/Vite with a local FastAPI service rather than the
PRD's earlier Streamlit direction. The backend reuses the Python preprocessing
and frequency-feature code; inference remains local and in memory.

- User decision: recorded in `docs/DECISIONS.md`.
- Supervisor/team review: **TBD**.

## Entries to add after Phase 7

- [ ] Record any freeze/evaluation interruption without rerunning the evaluator.
- [ ] Record any approved deviation from the preregistered statistical tests.
- [ ] Record final result/report commits and SHA-256 values.
- [ ] Record rejected or superseded analyses without deleting their evidence.

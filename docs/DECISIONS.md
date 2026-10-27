# Project decisions

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

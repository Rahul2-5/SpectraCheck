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

# Phase 2 exploratory fixed-crop diagnostic

Date: 30 October 2026. **Candidate treatment for review; official audit remains blocked.**

## Done

Added `scripts/diagnose_fixed_crop.py` and synthetic coverage in `tests/test_fixed_crop_diagnostic.py`. Ran the same tested diagnostic source through the live Chrome/Colab notebook as a separately labeled candidate. It validates config/manifest/image hashes, uses the existing 2,500-per-class audited population, and refuses to run after split/freeze/evaluation markers exist.

The fixed treatment is EXIF orientation, RGB, centre 256x256 crop with no resize/upscale, then JPEG quality 96, 4:4:4, non-progressive and no optimization. Quality 96 is taken from the existing audit's recorded real median, not selected by trying qualities. All transformed image bytes remain in memory and are discarded. Only diagnostic metadata and results are persisted outside research data/artifact/feature roots. The script does not call training preprocessing, extract frequency features or generate splits.

## Actual results

| Metadata probe | Existing full-image JPEG remedy | Exploratory fixed-crop treatment |
| --- | ---: | ---: |
| Logistic regression | 0.8308 | **0.5814** |
| Depth-3 decision tree | 0.9998 | **0.5814** |

Crop-candidate logistic folds: `[0.581, 0.567, 0.577, 0.591, 0.591]`.

Crop-candidate tree folds: `[0.592, 0.544, 0.587, 0.595, 0.589]`.

Both candidate means are <=0.60 with the unchanged balanced seed-42 five-fold probes. The official audit result was not replaced. Equal treated dimensions/format/quality are imposed by construction; primarily byte-size predictiveness remains measurable in this candidate. A below-limit probe therefore does not establish that the original dataset's confounding, compression history, object scale or content differences disappeared.

This is a data-adaptive exploratory analysis prompted by prior audit results, not a pristine preregistered pass. No held-out research split or detector scores existed or were inspected.

## Tests

- `.venv/Scripts/python.exe -m pytest tests/test_fixed_crop_diagnostic.py -q --basetemp .venv/pytest-fixed-crop-20261030-a -p no:cacheprovider`: **12 passed in 5.03s**.
- `.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-fixed-crop-full-20261030-a -p no:cacheprovider`: **165 passed in 52.61s**.

Both runs used the previously approved outside-sandbox test execution with fresh project-local temp directories. Tests cover known centre-crop content without resizing, deterministic bytes, invalid/undersized input, isolated outputs, unchanged original files, and rejection of sealed or tampered provenance. Existing dependencies were used; no pins were changed. No frontend changes were made.

## Outputs and provenance

Drive directory: `/content/drive/MyDrive/aidetect/r2_fixed_crop_candidate_20261030T134245Z/`.

- `candidate_metadata.csv` — per-image IDs/labels, original image checksums, transformed metadata and crop checksums; no image payloads or frequency features.
- `candidate_result.json` — actual fold scores, parameters, hashes and explicit diagnostic-only status.

Unchanged config hash: `ce79d3ef5f012768bbc377db9cdd9226377a1449a13a7725d9d49189f3549dce`.

Source audit SHA-256: `5c910e8958737edc363461db647131272dc7e541766a1441efe9d095e803e8db`.

Source active-manifest SHA-256: `db7b72907e8b0de7b5aa343a654c59aaee9b945080a4fd7c6122852abbc5cb81`.

The live notebook embeds the tested diagnostic source and actual output so no GitHub source pull was required for this experiment. It does not install the new script into the remote checkout or alter the research package there.

## Review and remaining implementation

See `docs/R2_FIXED_CROP_AMENDMENT.md` for the concrete proposal and the substantial change to interpretation of the audit gate. Adopting it requires a recorded team/supervisor protocol decision. Do not label the original audit passed, copy candidate results into `audit_result.json`, or run the existing Phase 3 cells on this result.

If approved, implement R2 in a separately versioned experiment with shared preprocessing/serving parity, raw plus treated metadata, original/canonical duplicate groups, full hash provenance, new immutable splits and the usual downstream reviews. Physical image quality, content-scale confounding and historical JPEG artifacts remain unverified limitations.

STOP before research split creation. Required next instruction is a reviewed decision to implement R2 or retain the current blocked protocol; the candidate result alone is not authorization.

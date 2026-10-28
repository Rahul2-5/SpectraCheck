# Verification through Phase 8 - 28 October 2026

**Software checks pass after fixing the gaps below. Real-data and model-quality validation remain pending.** Phase 9 was not started.

## Findings and fixes

| Before | After | Why |
| --- | --- | --- |
| Feature extraction used an inner join that could silently omit a frozen image ID | Require exact full manifest/split ID coverage and matching labels/groups | Preserve the frozen sample and class balance |
| Extraction did not verify its active-manifest provenance or source bytes at the processing boundary | Check the passing audit and frozen manifest fingerprint; verify image SHA-256 on fresh work and resume | Stop accidental source drift from entering an experiment |
| A relative image path could resolve outside the configured data directory | Resolve and check path containment before reading | Process only the intended collection |
| A model runtime error could escape as an unstructured server failure | Log diagnostics server-side and return a clear 503 model-service response | Keep the app usable without exposing internal errors |
| An HTML/proxy error could show `Unexpected token <` | Show a readable retry message without a fabricated result | Give users actionable feedback |
| Linux CI timed out on a page-level keyboard file-picker event | Scroll and focus the upload control, assert focus, and press Enter on that control; retain browser failure traces | Exercise the intended keyboard target and preserve useful diagnostics |

The failure cases were reproduced with new tests before the fixes. The existing
137-test baseline passed, showing why these additional boundary tests mattered.
The image-preparation and frequency math, research config, labels, model search,
statistical decision rules, and evaluation protocol were not changed.

## Final verification results

| Check | Result |
| --- | --- |
| Full Python suite | **147 passed**, 0 failed, 0 skipped, 65.69 seconds |
| React components | **6 passed** |
| Chrome browser/accessibility flows | **2 passed**, 7.9 seconds after keyboard-check update |
| Production frontend build | Passed |
| Clean `npm ci` | Passed; lockfile install completed, audit reported 0 vulnerabilities at check time |
| Python dependency check | `No broken requirements found.` |
| Notebook JSON / unique IDs / ordinary Python AST | Passed |
| Research configuration | Unchanged: `ce79d3ef5f012768bbc377db9cdd9226377a1449a13a7725d9d49189f3549dce` |
| Live production page | HTTP 200 |
| Live model readiness | Unavailable, correctly reflecting missing real artifacts |
| Live corrupt upload | Clear HTTP 422 |
| GitHub Actions | Frontend and Python 3.11/3.13 jobs all passed for `ebaedb3` ([run](https://github.com/Rahul2-5/SpectraCheck/actions/runs/40633198358)) |

Commands ran from the repository root (frontend commands from app/frontend):

```powershell
$reviewFinalRoot = Join-Path (Get-Location) '.venv/review-final-20261028a'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.venv/matplotlib'
.\.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp $reviewFinalRoot -o cache_dir=.venv/review-pytest-cache
.\.venv\Scripts\python.exe -m pip check
npm.cmd --prefix app/frontend test
npm.cmd --prefix app/frontend run build
npm.cmd --prefix app/frontend run test:browser
```

The explicit Python test directory was fresh. Future runs should use another
fresh name rather than a folder containing files to keep.

## Coverage by phase

| Phase | What was checked | Limit |
| --- | --- | --- |
| 0 | Config/hash/seed utilities, imports, dependency compatibility, clean frontend install | Installed tools/OS only |
| 1 | Original-byte collection, metadata mapping, deterministic sampling/resume, local Parquet round trip | Offline fixtures; full hosted download not executed |
| 2 | Format/size/quality probes, remediation, exact/perceptual grouping, exact-balanced group splits | Synthetic data, not actual dataset bias measurements |
| 3 | EXIF, crop/downsize, path/bytes parity, invalid/undersized inputs | Real-image visual review pending |
| 4 | FFT/radius/entropy/RAPSD/peaks and known-answer cases | Actual train-only spectra review pending |
| 5 | Summaries, pixels, batches, seal, resume, manifest/source integrity | Real batches/counts/timings pending |
| 6 | Six real classifier code paths, shared folds, tuning/calibration/provenance | Synthetic inputs and reduced test search budget |
| 7 | Clean-tree/freeze, one-use guard, metrics/statistics, lock hashes | Only generated fixture evaluation |
| 8 | Shared feature/probability parity, no upload writes, readiness/errors, UI flows, responsive layout/accessibility | Real model and physical-phone checks pending |

## Connected pipeline check

`tests/test_pipeline_end_to_end.py` creates a separate temporary repository and
generated smooth/noisy images with fixed-size PNG encoding. It runs the actual
audit and grouped split code, all three extraction commands, Phase 6's six model
searches/calibration, freeze, committed-lock evaluation, and the local API.

It verifies that a second evaluation is blocked, immutable splits keep their
fingerprint, and the API's selected-model class/probability matches the already
saved fixture evaluation result (probability tolerance 1e-12). It does not reopen
test feature rows for app parity, use any real held-out image, or publish fixture
metrics/models as research findings. The fixture's target/sign-off text exists
only inside its temporary repository, not in the real practical-target document.

## Still needed for real research validation

The configured real manifest/splits, primary model, FREEZE.lock, and evaluation
results are absent in this checkout. The following therefore cannot be marked
passed:

- Actual acquisition/counts, format-bias probes, exclusions, duplicates and split review in Colab.
- Actual preprocessing/spectrum review, feature extraction and full-budget training.
- Supervisor-agreed practical target and the real committed freeze/single evaluation.
- Real-image API parity, detector performance and physical mobile testing.

No real locked artifact or evaluation marker was modified/reset. Guard fixes do
not change numerical features, but existing frozen experiments must retain their
matching source checkout; do not rewrite a real lock to accommodate code edits.
Phase 9 remains separate and must use saved real outputs only.

## Outcome

The implemented software now passes the checked Phases 0-8 paths and the added
integrity/error cases. This is software verification; it does not establish
real-dataset accuracy or complete the pending team/supervisor gates.

The initial review CI run for `85c1540` passed both Python versions and the
frontend component/build steps, but its keyboard file-picker browser test timed
out on Linux Chrome. The targeted keyboard test update passes locally and on
Linux CI. All three replacement jobs passed for code commit
`ebaedb3c1e1223eff4b07551b880a307f1df63e6`; see the linked run above.

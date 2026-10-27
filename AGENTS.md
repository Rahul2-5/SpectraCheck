# AGENTS.md - Project memory and working rules

## Source and current state

- Source of truth: `PRD_AI_Image_Detection_Codex_Guide.docx` (PRD v1.0, 26 October 2026); a PDF version is also present. The files were observed moved into `docs/` during Phase 0; check there if absent at the root. Read the relevant phase before implementing it. This file summarizes the PRD; it does not replace its detailed prompts and checks.
- Project: **Detecting AI-Generated Images Using Frequency-Domain Features**.
- Team: Rahul Temkar, Shravani Rane, Jagruti Doke; MCA, MET Institute of Computer Science.
- Memory initialized: 27 October 2026. At initialization the workspace contained only the two PRD files. No implementation, Git repository, dataset, experiments, or completed phase was present.
- Phase 0 foundation is implemented and included on `main`. See `reports/phase_0_report.md` and `reports/phase_0_environment.json` for historical evidence.
- Phase 1 acquisition code is implemented: original-byte main/pilot collection, seeded reservoir sampling, all-generator counts, revision/config guards, manifests, interruption recovery, and Colab commands. See `reports/phase_1_report.md` and `reports/phase_1_checks.json`.
- Phase 2 audit/split code is implemented: integrity/metadata measurements, balanced metadata-only probes, conditional JPEG remediation for both classes, original+normalized duplicate union groups, and immutable class-balanced group splits. See `reports/phase_2_report.md` and `reports/phase_2_checks.json`.
- Phase 3 shared preprocessing is implemented: one path/bytes entry point with EXIF orientation, RGB-to-L conversion, centered square crop, conditional LANCZOS downsampling, no-upscaling rejection, float64 `[0, 1]` output, versioned settings, synthetic tests, and a Colab review cell. See `src/aidetect/preprocess.py`, `tests/test_preprocess.py`, and `reports/phase_3_report.md`.
- Phase 4 frequency features are implemented: centered FFT power spectrum, radius map, energy ratio, non-DC spectral entropy, K-ring log-RAPSD, robust local peaks, stable feature names, 68-value default extraction, synthetic known-answer tests, and a train-only Colab figure cell. See `src/aidetect/features.py`, `tests/test_features.py`, and `reports/phase_4_report.md`.
- Local checks: 94 tests passed on Python 3.12.1; `pip check` passed; installed NumPy FFT signatures were inspected; the Phase 3 and Phase 4 notebook cells parse successfully. Real Colab downloads/audit, actual generator/probe counts, real split hashes, the 20-per-class preprocessing review, the 200-per-class spectra review, supervisor decisions, and remote CI results have not been supplied/verified. Local synthetic success is not evidence the real-data gates passed.
- Configured Git remote: `https://github.com/Rahul2-5/SpectraCheck.git`. User instruction after Phase 0: **commit and push on `main` only**. This overrides the PRD's per-phase branch convention. Completed setup was fast-forwarded onto local `main`; keep notebook clone instructions on `main`.
- The user explicitly approved pushing the Phase 0/1 source code and reports to `https://github.com/Rahul2-5/SpectraCheck.git`, branch `main`, after the destination-specific approval question. The previous automatic-review approval block is resolved by that authorization. Push only `main`; retain real-data/CI checks as pending unless verified.
- The user explicitly approved publishing Phase 2 and the React-in-Phase-8 plan change to the configured GitHub repository on `main`, with commit message `Phase 2 completed ,(React in Phase 8)`. This resolves the earlier Phase 2 publication-approval block. Real-data/CI checks remain pending unless their outputs are verified.
- User decision on 27 October 2026: use **React instead of Streamlit** for D3/Phase 8. This overrides the PRD's framework choice. Build a local React frontend with a local Python prediction API that reuses `aidetect` preprocessing/features/models; do not port inference logic to JavaScript. See `docs/DECISIONS.md`.
- The environment requires Python 3.11+ (within the PRD's 3.10+ scope); direct pins were selected using Python 3.11 package metadata and installed/import-checked on Python 3.13. Full Python 3.11/Linux execution remains a CI check. Phase 1 makes the already installed `huggingface-hub==2.3.0` an explicit dependency because the downloader resolves source revisions using HfApi.
- Keep this section current with confirmed decisions, completed phases, evidence paths, and pending team checks. Never mark a phase passed on behalf of the team or record planned work as completed.

## Objective and scope

Build a reproducible Python package using handcrafted 2D FFT features to distinguish real images from AI-generated images. Compare frequency features against a matched pixel baseline using RBF SVM, Random Forest, and XGBoost (six combinations). Run a single held-out evaluation with pre-registered statistics, then demonstrate the validation-selected frequency model through a local React frontend and Python prediction API.

Deliverables: D1 frequency feature module; D2 evaluation and statistical comparisons; D3 local prototype; D4 final report supported by reproducible figures, tables, and a model card. Null and mixed findings are valid outcomes.

Out of scope: cross-generator evaluation, social-media compression robustness, deep-learning detectors, colour/wavelet/DCT/noise-residual features, edited-region localization, video, adversarial attacks, and production hosting. Metadata is used for the shortcut audit only, not as detector features.

## Working rules from the PRD

1. Implement exactly one requested phase per task. Per the user's override, commit and push to `main` only; do not push `phase-N` branches. Run the phase tests, write `reports/phase_N_report.md`, commit the phase work, and STOP. The user authorizes the next phase by saying proceed; preserve team checks as pending unless results are supplied, and do not claim they passed.
2. Cost must remain zero. No paid APIs, API keys, paid hosting, or GPU dependency. Permitted free services: GitHub, Google Colab, Google Drive, Kaggle, and Hugging Face public downloads. Recheck changing limits when needed; PRD estimates are not guarantees.
3. Codex writes code, synthetic-data tests, documentation, and Colab commands. The team runs real-data downloads, audits, extraction, training, and evaluation in Colab, with data on Drive. The prototype runs on a laptop.
4. Put settings in `config.yaml`; avoid magic numbers. Fixed seed is 42. Pin dependency versions actually installed and tested, and record config hashes in outputs.
5. Use exactly one shared image preprocessing entry point: `src/aidetect/preprocess.py:preprocess_image`. Training, evaluation, and the app must call it. No class-specific or app-specific preprocessing.
6. The test set is sealed. No exploratory reading, tuning, model selection, or training may use test rows or features. Test evaluation is through `scripts/evaluate_test.py` only, after `artifacts/FREEZE.lock` exists and all hashes match. See the Phase 5 clarification below before implementing the seal.
7. Every function receives appropriate pytest coverage using tiny generated images/data, without real datasets or network access. Use noise, sinusoids, checkerboards, constant images, and known-answer statistical examples.
8. Verify uncertain library signatures with installed-package inspection or `help()`; document what was checked. Do not invent parameters or claim unrun tests passed.
9. If a step fails or a required test cannot pass, report the precise failure and cause. Do not silently bypass a requirement or advance the phase.
10. Label convention everywhere: **AI-generated = 1 (positive); real = 0**. Dataset generator IDs are separate from these binary labels.

## Stack, configuration, and planned layout

Python 3.10+; planned CI uses Python 3.11. Match the Python minor version between Colab and laptop where possible. Libraries: numpy, scipy, pandas, pyarrow, pillow, scikit-learn, xgboost, statsmodels, imagehash, matplotlib, datasets, pyyaml, joblib, pytest. React/Node frontend tooling and the Python HTTP API dependency will be added and pinned when implementing Phase 8; Streamlit is no longer required. Historical environment reports retain the versions that were actually checked at the time.

PRD defaults to centralize in configuration:

| Setting | Default |
| --- | --- |
| seed | 42 |
| image_size | 256 |
| rapsd_rings | 64 |
| pixel_baseline_size | 32 |
| train / val / test | 0.70 / 0.15 / 0.15 |
| main dataset | TheKernel01/Tiny-GenImage |
| generator_name | SD15 |
| cv_folds / search_iters | 5 / 30 |
| alpha | 0.05 |
| batch size | 500, resumable |
| Drive path default | /content/drive/MyDrive/aidetect |

Planned folders: `src/aidetect/`, `scripts/`, `tests/`, `notebooks/`, `app/`, `reports/`, `artifacts/`, `docs/`. Planned modules: `io_utils`, `data`, `audit`, `splits`, `preprocess`, `features`, `pixel_baseline`, `train`, `evaluate`, `stats_tests`, `plots`.

Use `notebooks/colab_runner.ipynb`, `app/app.py`, and `docs/PHASE_REPORT_TEMPLATE.md`. Data and features live on Drive, not in Git. Ignore `data/`, `features/`, `*.parquet`, `*.npz`, and `artifacts/models/`. Do not redistribute dataset images.

## Data and validity safeguards

- Main data: Tiny-GenImage real ImageNet photos and Stable Diffusion v1.5 images only. Resolve `SD15` through the dataset's ClassLabel metadata; never hard-code generator ID 6.
- Acquisition outputs are `data_root/main/` and `data_root/pilot/`, each with original files, `manifest.csv`, and `acquisition.json`. Main collection uses a metadata pass and image pass; it bounds memory but may transfer the full upstream Parquet data and rescans on resume. A resolved dataset commit, source mappings, config hash, and completion state are recorded; changed settings/source or mismatched existing bytes must raise rather than silently mix data.
- CIFAKE is a small pilot only (planned 500 real + 500 fake). Its 32x32 images cannot support the main frequency study; never present pilot scores as research results.
- Stream original bytes with image decoding disabled. Preserve original JPEG/PNG evidence until the audit; infer format from magic bytes. Save a manifest and counts for every observed generator, including SD14 and SD15.
- Main sample size is unconfirmed: PRD assumes roughly 2,000 AI + 2,000 real. Initially sample up to 4,000 real images, then balance in Phase 2. If SD15 has fewer than roughly 1,000 images, the team must consult the supervisor about a scope change to full GenImage SD1.4.
- Audit format, width, height, byte size, aspect ratio, mode, and estimated JPEG quality. Shortcut probes: logistic regression and a depth-3 tree, using only metadata and stratified 5-fold CV.
- Assumed pass threshold: both metadata probes must have mean CV accuracy <= 0.60. If a probe exceeds it, re-encode BOTH classes identically to JPEG at the real images' median estimated quality with 4:4:4 subsampling, then re-probe and report before/after. Do not proceed if the audit still fails.
- Detect exact and near duplicates; perceptual-hash groups use Hamming distance <= 4 with union-find. No group may cross splits. Create balanced, stratified 70/15/15 splits once and save `splits.csv` plus `artifacts/splits.sha256`. A changed split hash invalidates downstream work.
- Phase 2 uses `audit_workflow.run_audit` then `run_split_freeze`, exposed by `scripts/run_audit.py` and `scripts/make_splits.py`. Measurements and the active manifest are under `data_root/audit/`; active paths are relative to `data_root`. Originals under `data_root/main/` are retained; normalized copies are separate. No real research split/hash has been generated locally.
- Probe samples are balanced before CV, so the 0.60 bar is evaluated against a 50% chance level. Undersized images are excluded and logged before balancing/freezing. JPEG quality is a nearest-standard quantization-table estimate with a residual, not recovered encoding history. Missing real JPEG quality blocks the preplanned fix instead of inventing a value.
- Exact per-split class balance plus group separation/proportion tolerance is enforced using SciPy integer allocation over group class-count patterns (instead of a heuristic grouped-CV split). Infeasible group structures fail explicitly. Freeze records are `data_root/splits.csv`, `artifacts/splits.sha256`, and `artifacts/splits.json`; identical reruns verify/reuse, changed/partial records fail. Once sealed, do not rerun the audit.
- Accuracy around or above 98% is a shortcut/leakage alarm: recheck metadata, duplicate separation, scaler leakage, and JPEG-grid spectral peaks.
- PRD identifies Tiny-GenImage's license as CC BY-NC-SA 4.0: retain attribution and document the source/license in the model card. Dataset counts, schema, and availability must be verified at acquisition time.

## Processing and features

- `preprocess_image(src, cfg)` accepts a path or bytes: Pillow decode, EXIF transpose, RGB then greyscale L, square centre crop, LANCZOS downsize to SxS, float64 divided by 255. No resize at equal size; reject smaller images instead of upscaling. Return finite [0,1] values and clear corrupt/unsupported-file errors. Record `PREPROCESS_VERSION` in the config hash.
- Phase 3 records the fixed preprocessing contract in `config.yaml` under `preprocess`; the active configuration hash is `bf1e6f3b4f4cdc5f95d0503e55533aee54f48fc054ba2a6ea89df305a0af2c6c`. The earlier Phase 2 hash in `reports/phase_2_report.md` remains historical. No real split is present locally, so no split file or audit output was regenerated.
- Phase 4 records `hann_window`, `epsilon`, and `mad_scale` under `config.yaml:frequency`; the active configuration hash is `fae16d16f5bc8891971219bc27a149e2fb4e2d4d8e2cddafe105ad07ae18fc10`. The previous Phase 3 hash remains historical. No real split, feature file, or research figure was regenerated locally.
- Frequency vector: energy ratio + normalized spectral entropy + K log RAPSD values + peak count/strength = **68 values at K=64**.
- Power spectrum uses mean removal, optional fixed 2D Hann window, fft2 then fftshift, squared magnitude, centre `(S//2, S//2)`. Exclude DC from energy/entropy; use epsilon handling for constant images and log zero.
- Energy ratio: energy at r > r0 divided by energy at 0 < r <= r0 plus epsilon. RAPSD averages K equal-width rings from radius 1 through S//2, ignores corner bins, and returns log10(mean + epsilon).
- Peaks use ring-median/MAD-normalized log power, local 3x3 maxima above `k_mad`, and one spectrum half to avoid counting conjugate pairs twice. Return count and max z (zero if none).
- Known-answer checks include a 256x256 cosine at frequency 16 with spectral maximum at radius 16, white-noise behavior, smooth gradients, deterministic output, matching feature names, and no NaN/Inf for constant inputs.
- Pixel baseline takes the shared preprocessed image, downsizes it using LANCZOS to 32x32, and flattens to **1,024 values**. This prescribed baseline downsampling is distinct from duplicating image preprocessing.
- Extract in resumable batches of 500. Persist summaries sufficient to recompute energy ratios and peak thresholds, plus pixel features. Parquet metadata records config and split hashes.

## Training, freeze, and statistics

- Pick `r0` from 8..120 step 8 and `k_mad` from [3,4,5,6,8] using a default RF fitted on train and scored on validation. Break ties by smallest r0, then smallest k_mad. Save `artifacts/feature_choice.json`.
- For all six pipelines, fit StandardScaler within CV; share the same shuffled, seeded five-fold StratifiedKFold splits and 30 RandomizedSearchCV trials scored by accuracy. Use PRD Phase 6 search spaces.
- Refit best models on train. SVM uses sigmoid CalibratedClassifierCV with five folds. Report validation accuracy, AUC, and Brier score for all six; save joblib models and `versions.json`.
- Choose the primary frequency model by validation accuracy before test access; save `artifacts/primary_model.json`. Do not choose a model using test results.
- Before Phase 7, the team must agree and record `reports/practical_target.md`. Statistical significance and practical usefulness are separate judgments.
- Freeze requires a clean Git tree and no existing lock. Hash config, splits, package source, feature choice, model files, and practical target; record UTC time. Also preserve the primary-model selection. Commit the lock before evaluating, and verify hashes before opening sealed features.
- Evaluate once, saving per-image predictions/probabilities for all six models. Report accuracy with exact Clopper-Pearson 95% CI, precision, recall, F1, ROC-AUC, and confusion matrices.
- H1: one-sided exact binomial test of primary-model accuracy against p=0.5; reject at p < 0.05. Other models' binomial results are descriptive.
- H1a: exact paired McNemar for frequency versus pixel within each classifier; Holm across three p-values. Claim frequency superiority only when it has more correct predictions and adjusted p < 0.05.
- H1b: Cochran's Q on the three frequency models' correctness matrix; if significant, pairwise exact McNemar with Holm across three pairs.
- Save statistics, raw/adjusted p-values, and explicit decisions, including non-rejections, to `reports/results.json` and `.md`. Recompute statistical reference values rather than treating rounded PRD examples as exact constants.
- After test opening, changes/analyses must be identified as exploratory. Phase 9 uses saved outputs only; never retrain or rerun the held-out test for the report pack.

## Prototype and reporting

The local React uploader supports jpg/jpeg/png/webp and sends the image to a local Python prediction API. The API processes uploads in memory only, reuses shared preprocessing and frozen feature choices, and loads the validation-selected primary model. React displays the label, calibrated probability, log spectrum, and RAPSD plot returned by the API. Reject corrupt and undersized inputs clearly; warn on saved-versus-installed version mismatch. Verify API prediction parity with evaluation helpers. Plan `app/frontend/` for React and `app/backend/` for the Python API; implement them in Phase 8, preserving the one-phase-at-a-time workflow. No paid API or production hosting is required.

Required app note: "Model score for one Stable Diffusion v1.5 dataset; not proof of authenticity; may be wrong on other generators, screenshots or compressed images."

Report pack: saved-result figures at 300 dpi, CSV/Markdown tables, model card, complete phase-by-phase README, and `reports/deviations.md`. State single-generator scope, no per-category labels/accuracy, untested compression robustness, and CIFAKE pilot-only status. Cite only literature actually read; do not fabricate references.

## Phase roadmap

| Phase | Work and main outputs |
| --- | --- |
| 0 | Repository skeleton, rules, config, pinned environment, synthetic tests, CI, Colab runner |
| 1 | Pilot/main acquisition scripts, original-byte manifest, actual generator counts |
| 2 | Audit, shortcut probes/fix, duplicate groups, balanced frozen splits/hash |
| 3 | Single shared preprocessing function and tests |
| 4 | Frequency module, known-answer tests, train-only example spectra (D1) |
| 5 | Pixel baseline, batched feature summaries, sealed test outputs |
| 6 | Equal-budget training/tuning, chosen features/primary model, practical target |
| 7 | Freeze, one test evaluation, hypothesis tests and decisions (D2) |
| 8 | Local React frontend + Python prediction API (D3) |
| 9 | Saved-output report pack, model card, reproduction docs, deviations (D4 support) |

Every phase report must contain: Done (files and purpose), Tests (exact command, pass/fail counts, summary), Outputs (paths), Checks for the team in Colab (exact cells/commands), Assumptions (and consequences if wrong), Open questions/unverified items, and STOP with the required next-phase instruction.

## Unresolved decisions and PRD inconsistencies

- Supervisor confirmations remain pending: main dataset choice; insufficient-SD15 fallback; shortcut threshold and format remediation; undersized-image policy; pixel baseline; primary-model rule; practical target. Defaults above are PRD choices, not recorded supervisor approvals. The user explicitly chose React for the prototype; do not ask again about Streamlit.
- Phase 5 explicitly requires creating test features before freezing, whereas the short rulebook prohibits reading any test rows before freeze. Preserve the intended distinction: deterministic, write-only test feature generation is planned in Phase 5; inspection/evaluation of sealed features waits until Phase 7. Surface and resolve this conflict with the team before implementing test access; do not silently weaken the seal.
- CIFAKE's 32x32 pilot conflicts with the main 256x256 no-upscaling policy. A pilot-specific configuration must be documented and kept separate from the main experiment; never silently upscale or weaken the main policy.
- The PRD's instruction to find every resize only in preprocessing has a prescribed exception for pixel-baseline downsampling in Phase 5.
- Phase 2 implements the default undersized-image exclusion before split freeze and records every exclusion. Any later change to image-size/eligibility policy affects the frozen experiment and must not silently drop rows or rewrite splits. Later phases may add config settings; preserve Phase 2's recorded config hash as historical provenance and verify the immutable split hash rather than regenerating splits.
- The one-time evaluation rule needs an explicit reuse guard; a freeze lock alone does not prevent running the evaluator twice. Preserve this requirement when implementing Phase 7.

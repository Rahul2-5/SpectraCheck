# AI Image FFT Detector

Research project: **Detecting AI-Generated Images Using Frequency-Domain Features**.
Team: Rahul Temkar, Shravani Rane, Jagruti Doke (MCA, MET Institute of Computer Science).

We will compare handcrafted frequency features against pixel features using
SVM, Random Forest, and XGBoost, evaluate once on a held-out test set, and build
a local React demonstration backed by a Python prediction API. A result near chance is still a valid research
finding. This is a single-generator study, not proof of image authenticity.

## Current stage

Phases 0-8 software is implemented, including the one-shot evaluation protocol
and local React/Python prototype. The user authorized Phase 8 code development
while real-data checks remain pending. No real model, held-out result, or team
review is claimed. The app permits previewing an image and reports model
unavailability honestly until its frozen model bundle is supplied. Read
[AGENTS.md](AGENTS.md) and the phase reports before proceeding. Phase 9 has not
started. User policy: commit and push changes to `main` only.

## What the folders are for

| Path | Purpose |
| --- | --- |
| `config.yaml` | One place for experiment settings and Drive paths |
| `src/aidetect/` | Shared configuration, acquisition, audit, split, feature, and training code |
| `tests/` | Small automatic checks that need no dataset or internet |
| `scripts/` | Acquisition, audit, split, feature-extraction, and Phase 6 training commands |
| `notebooks/colab_runner.ipynb` | Team's step-by-step runner in free Colab |
| `app/` | React frontend and local Python prediction API |
| `reports/` | Phase reports and, later, research results |
| `artifacts/` | Provenance hashes and, later, model-selection/freeze records |
| `docs/` | Shared report template |
| `.github/workflows/tests.yml` | Automatic tests on GitHub pushes and pull requests |

## Local setup (Windows PowerShell)

Use Python 3.11 or 3.13. Phase 0 is locally checked with the version recorded in
the phase report; GitHub is configured to check both versions on Linux.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

The virtual environment keeps project libraries separate from your other
projects. `requirements.txt` pins the direct libraries to installed versions;
transitive dependencies are resolved by pip and may change. A future experiment's
saved `versions.json` will record its runtime versions. These pins do not claim
that unimplemented training or app code has already been tested.

For macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` in place of
the Windows executable. No paid service, API key, or GPU is needed.

## Run setup in Colab

1. Upload `notebooks/colab_runner.ipynb` to Google Colab.
2. The clone cell uses `main` from the configured remote,
   `https://github.com/Rahul2-5/SpectraCheck.git`. Adjust the URL if the team uses
   a different repository. Phase branches are not used for publishing.
3. Run the Phase 0 cells in order: mount Drive, clone `main`, install, restart
   the Colab session if prompted, print versions, and run pytest.
4. Confirm no dependency conflicts and all tests pass. Keep the printed versions
   and results for team review. Run only the authorized phase's cells.

The notebook checks runtime version/pins and imports the installed libraries.
Its setup cells do not download research images or train models. Paths in
`config.yaml` target Drive; adjust them explicitly for future local data runs.

## GitHub check

The configured remote is `https://github.com/Rahul2-5/SpectraCheck.git`. Push `main`
and inspect the **Python tests** workflow under Actions. Both matrix jobs must
pass. Remote workflow success must be checked separately from local test success.

## Phase 3: shared preprocessing

`aidetect.preprocess.preprocess_image(src, cfg)` is the single image entry
point for later features, training, evaluation, and the local app. It accepts
an image path or encoded bytes, applies EXIF orientation, RGB then grayscale
`L`, center square cropping, conditional LANCZOS downsampling, and float64
normalization to `[0, 1]`. It rejects images smaller than `image_size` rather
than upscaling them. The fixed contract is recorded as `PREPROCESS_VERSION` in
`src/aidetect/preprocess.py` and mirrored under `preprocess` in `config.yaml`.
The Phase 3 notebook cell reviews 20 training images per class and reports
undersized inputs; it does not inspect test rows or create features.

## Phase 4: frequency features

`aidetect.features` turns one Phase 3 array into a centered 2D FFT power
spectrum, non-DC energy ratio, normalized spectral entropy, configurable-K
log-RAPSD vector, and robust conjugate-pair-safe peak count/strength. The
default configuration uses a fixed 2D Hann window, epsilon handling, and 64
RAPSD rings, producing 68 finite values. The Phase 4 notebook cell is
train-only: it samples 200 rows per class, saves mean log-spectrum and mean
RAPSD figures with 95% bands under `reports/figures/`, and does not create
features for validation or test rows.

## Phase 5: pixel baseline and batched feature extraction

`aidetect.pixel_baseline.extract_pixel_features` accepts only the shared
`preprocess_image` output, downsizes it with Pillow LANCZOS to the configured
32×32 baseline, and returns 1,024 finite brightness values. This prescribed
baseline resize is the one Phase 5 exception to the shared image-preprocessing
resize rule.

`scripts/extract_features.py` joins the immutable split file with the active
manifest, preprocesses each image once, and writes one Parquet file per
configured batch (500 by default). Each batch stores RAPSD, normalized entropy,
per-radius cumulative energy for integer energy cutoffs, peak count/strength
summaries for k in `[3, 4, 5, 6, 8]`, and the pixel vector. Parquet metadata
records the configuration and split hashes. Existing batches are verified and
skipped, so a disconnected Colab session can resume safely.

Run the authorized split commands only after the Phase 2 audit and immutable
split outputs exist:

```text
python scripts/extract_features.py --split train
python scripts/extract_features.py --split val
python scripts/extract_features.py --split test
```

The test command is deterministic write-only extraction into
`features_root/test_sealed/`; the official `read_feature_batches` reader raises
`PermissionError` until `artifacts/FREEZE.lock` exists. Phase 5 does not train,
select models, inspect test features, or evaluate the test set.

Phase 6 accepts either the active full-config hash or the recorded Phase 5 hash
`7ac0df38870f031ee724c7b9b1610d12b02890bd9fd2760adb1215facc498d0f`. The
historical hash is allowed only when the Phase 5 feature contract fingerprint
in `config.yaml` still matches the active preprocessing, frequency, and
extraction settings. This lets Phase 6 add training-only settings without
silently treating changed image features as compatible.

## Phase 6: train/validation model selection

Run Phase 6 only after the team has verified the frozen split and Phase 5
train/validation feature batches in Colab. `scripts/train_models.py` reads only
`features_root/train` and `features_root/val`, verifies their Parquet metadata,
batch ID hashes, the immutable split digest, and the active feature contract,
and refuses to run after `artifacts/FREEZE.lock` exists. It never opens
`features_root/test_sealed`.

The script selects `r0` and `k_mad` by fitting the default Random Forest on
train and comparing validation accuracy. It then runs the same configured
shuffled stratified folds and randomized-search budget for frequency and pixel
SVM, Random Forest, and XGBoost pipelines. Each pipeline scales inside CV. The
SVM is refit with sigmoid calibration using training rows only. Validation
accuracy, ROC-AUC, and Brier score are recorded for all six models; the primary
frequency model is selected by validation accuracy before test access.

```text
python scripts/train_models.py
```

Outputs include `artifacts/feature_choice.json`,
`artifacts/primary_model.json`, `artifacts/versions.json`, six ignored model
files under `artifacts/models/`, and the actual run tables under
`reports/phase_6_results.{json,md}`. Frequency Random Forest and XGBoost feature
importance plots and CSVs are descriptive only. Synthetic local tests exercise
the pipeline; no real-data results are recorded until the team runs Colab.
The practical-use target remains a team/supervisor decision in
`reports/practical_target.md`; do not invent a threshold or create the Phase 7
freeze until the team records it.

## Phase 7: freeze and one-time test evaluation

The local implementation is in `src/aidetect/freeze.py`,
`src/aidetect/evaluate.py`, and `src/aidetect/stats_tests.py`, with commands
`scripts/freeze.py` and `scripts/evaluate_test.py`. Local synthetic tests cover
the protocol; no real test set has been opened and no research result is
recorded in this checkout.

Before running Phase 7 in Colab, review the real Phase 6 results and models,
complete `reports/practical_target.md`, and commit the Phase 6 outputs and
reviewed decision on `main`. The Phase 7 notebook copies the six ignored model
files to Drive immediately after training, with a SHA-256 manifest, and can
restore them after a Colab runtime restart. The freeze command requires a clean `main` tree,
the verified split digest, all six trained models, compatible Phase 6
provenance, and the completed target. It locks the config, split, package and
script sources, selection records, models, versions, target, and raw SHA-256
digests of the sealed Parquet files. Freeze hashes the test files as bytes; it
does not parse test rows or feature values.

```text
python scripts/freeze.py
```

Review and commit/push `artifacts/FREEZE.lock` to `main` before test access. On
the resulting clean checkout, run the evaluator once:

```text
python scripts/evaluate_test.py
```

The evaluator rechecks every frozen hash, creates
`artifacts/EVALUATION_STARTED.json` before loading any sealed test rows, then
writes six models' metrics, exact accuracy confidence intervals, preregistered
tests and decisions to `reports/results.json` and `reports/results.md`, plus
per-image predictions to `reports/test_predictions.csv`. If it fails after
creating the marker, do not rerun it; preserve the marker and ask the supervisor
how to proceed. After a successful run, check accuracy against the confusion
matrix and commit/push the result files to `main`. Later analyses must be
labeled exploratory.

## Phase 1: collect images in Colab

The [Tiny-GenImage card](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)
documents `image`, `label`, and `generator` fields. The code checks the actual
ClassLabel metadata at runtime and resolves SD15 by name. It records the source
commit so a changed dataset cannot silently enter an interrupted run.

1. Download [CIFAKE from Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
   manually, then extract it on Drive. `--source-dir` must contain
   `train/REAL`, `train/FAKE`, `test/REAL`, and `test/FAKE`. No API key is needed.
2. In the notebook's Phase 1 cell, set `CIFAKE_SOURCE_DIR` to that extracted folder.
3. Run the pilot and main commands below (also supplied as notebook cells):

   ```text
   python scripts/download_pilot.py --source-dir /content/drive/MyDrive/cifake
   python scripts/download_main.py
   ```

4. Run the notebook's count and sample-image checks. Record actual SD15 and SD14
   source counts. If fewer than the configured 1,000 SD15 images are saved,
   the main command returns exit code 2 **after saving counts/files**. Stop and
   consult the supervisor before Phase 2; do not silently change generators.

Default outputs are separate `data_root/pilot/` and `data_root/main/` folders.
Each has `real/`, `ai/`, `manifest.csv`, and `acquisition.json`. The manifest
records ID, relative path, binary label, generator, source split, byte-derived
extension, byte count, SHA-256, and config hash. The acquisition record includes
all source-generator counts (including zero counts), source mappings, settings,
completion status, and the manifest hash.

The pilot copies 500 images per class, preserving originals. Its configured
image size is 32 for future pilot-only use; main experiment size stays unchanged.
The main collection keeps all SD15 rows and samples up to 4,000 real rows using
seeded reservoir sampling. Balancing and our final splits happen in Phase 2;
upstream `train`/`validation` names are provenance only.

If Colab disconnects, rerun the same commands with unchanged settings/output.
Valid saved files are hash-checked and left untouched. Missing images are restored;
conflicting existing bytes, changed settings/revision, or stray image files raise
an error for inspection. Interrupted runs remain `incomplete` until the final
manifest is written. Do not manually combine pilot and main files.

Acquisition uses a metadata pass followed by an image pass. It bounds memory and
avoids decoded image arrays; it does **not** guarantee small network transfers.
Streaming Parquet may transfer the full source, and resuming scans it again.
If the upstream `main` revision changes, use the recorded commit in
`dataset.revision` and a consistent config/output, or choose a fresh output folder.
No research images have been downloaded by the local tests.

## Phase 2: audit, duplicates, and frozen splits

After the main Phase 1 download/checks, run only the Phase 2 notebook section.
If Drive already has a complete collection, do not rerun acquisition just because
the project gained new audit settings. The audit verifies the saved Phase 1
manifest and file hashes while recording its own current configuration hash.

```text
python scripts/run_audit.py
python scripts/make_splits.py
```

`run_audit.py` reads `data_root/main/` and does the following:

- Checks original file integrity, format, dimensions, mode, byte size, estimated
  JPEG quality, and perceptual hashes. Corrupt files stop the run.
- Excludes images smaller than configured `image_size` before splitting, logging
  every exclusion. Keeps every eligible AI image and samples equally many real
  images. The configured minimum eligible AI count still applies.
- Runs five-fold logistic-regression and depth-3-tree probes using **only** file
  metadata. Balanced classes make the chance level 50%. Both mean accuracies must
  be <= the configured threshold, initially 0.60.
- If either fails, re-encodes both classes identically to RGB JPEG at the median
  estimated real-image quality, 4:4:4, in a separate normalized folder. EXIF
  orientation is applied before metadata is stripped. Original files stay intact.
  Both probes run again; a remaining shortcut returns exit code 2 and blocks splits.
- Groups exact and perceptually similar images using union-find. A Hamming
  distance <= 4 links images; transitive chains remain one group. When encoding
  changes, original and normalized duplicate relationships are both retained.

The JPEG-quality value is a nearest-standard quantization-table estimate, not a
record of the true encoding history. Table-matching residuals appear in the
report. If the probe fails and no real JPEG quality can be estimated, the code
records a blocked audit and waits for a supervisor decision; it does not invent
a target quality. Passing these probes does not prove every shortcut is absent.

Generated outputs:

| Path | Purpose |
| --- | --- |
| `data_root/audit/original_audit.csv` | Full source-file measurements |
| `data_root/audit/excluded.csv` | Undersized and excess-real exclusions with reasons |
| `data_root/audit/active_manifest.csv` | Balanced usable files, paths relative to data_root, current hashes and group IDs |
| `data_root/audit/audit_result.json` | Before/after probe scores and a verifiable pass/block gate |
| `data_root/audit/normalized/` | Separate JPEG copies, created only when remediation is required |
| `reports/audit_report.md`, `reports/audit_*.csv`, `reports/figures/audit_*.png` | Metadata tables, plots, decisions |
| `data_root/splits.csv` | Immutable id/label/split/group_id assignments |
| `artifacts/splits.sha256`, `artifacts/splits.json` | Split fingerprint and configuration/audit provenance |

`make_splits.py` requires a passing audit for the current config and unchanged
audited images. Each duplicate group stays together; train/val/test are
70/15/15 within one percentage point and **each split is exactly class-balanced**.
It uses SciPy integer allocation over group-size/class-count patterns because
ordinary grouped-CV heuristics do not guarantee all those constraints together.
If the group structure makes the requirements impossible, it stops explicitly.

Once written, the split CSV, SHA-256 and provenance cannot be silently replaced.
An identical rerun verifies and reuses them. Changed/partial records fail.
Do not rerun the audit after sealing splits. A split hash change invalidates all
downstream experiments. Never weaken the gate or separate duplicate groups just
to make a run pass.

Run the notebook's Phase 2 checks, review before/after scores and plots, and
retain the actual split fingerprint/provenance. The final cell backs up audit
reports and split records to Drive. Commit the actual `artifacts/splits.sha256`
and `artifacts/splits.json` to `main` after team review. Local synthetic checks
do not generate a fake research split hash in this repository.

## Reproducibility rules

- Seed 42; give future models the same seed through `random_state`.
- Use `load_config`, `set_seeds`, `sha256_of_file`, and `config_hash` from
  `aidetect.io_utils`. A SHA-256 hash is a fingerprint for detecting changes.
- The config hash covers the full sorted YAML configuration, including paths.
  Future preprocessing settings/version must be included before hashing outputs.
- AI-generated is label 1; real is label 0.
- Store datasets and features on Drive. Git ignores data, Parquet/NPZ files,
  model binaries, and the local environment.
- Main data will be Tiny-GenImage SD15; CIFAKE is pilot-only. Preserve attribution
  and the PRD's stated CC BY-NC-SA 4.0 terms; no dataset images are redistributed.
- Keep test data sealed until the Phase 7 freeze/evaluation protocol.
- Work on `main` only, one phase at a time; report actual checks and preserve
  pending real-data checks for the team.

## Remaining phases

Phase 9 (report pack and reproduction guide) is not started. Real-data execution
and team review from earlier phases still remain pending and must not be inferred
from the completed software.

## Run the React demo locally

```powershell
npm.cmd --prefix app/frontend ci
npm.cmd --prefix app/frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Install the Python requirements/editable package
first using the local setup instructions above. Node 22.12+ is required.
The interface works without a model for image preview; analysis stays disabled
until the reviewed frozen model bundle is present. No demonstration scores are
invented. See [app setup and API contract](app/README.md), the
[Phase 8 report](reports/phase_8_report.md), and [recorded decisions](docs/DECISIONS.md).

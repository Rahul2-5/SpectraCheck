# AI Image FFT Detector

Research project: **Detecting AI-Generated Images Using Frequency-Domain Features**.
Team: Rahul Temkar, Shravani Rane, Jagruti Doke (MCA, MET Institute of Computer Science).

We will compare handcrafted frequency features against pixel features using
SVM, Random Forest, and XGBoost, evaluate once on a held-out test set, and build
a local Streamlit demonstration. A result near chance is still a valid research
finding. This is a single-generator study, not proof of image authenticity.

## Current stage

Phase 0 foundation and Phase 1 acquisition code are implemented. Phase 1 adds
original-byte collection and a manifest; the team runs real downloads in Colab.
Image processing, training, and the app belong to later phases. Read
[AGENTS.md](AGENTS.md), the [Phase 1 report](reports/phase_1_report.md), and the PRD
before proceeding. User policy: commit and push changes to `main` only.

## What the folders are for

| Path | Purpose |
| --- | --- |
| `config.yaml` | One place for experiment settings and Drive paths |
| `src/aidetect/` | Shared code for configuration, seeds, hashes, and acquisition |
| `tests/` | Small automatic checks that need no dataset or internet |
| `scripts/` | Acquisition commands now; training/evaluation commands later |
| `notebooks/colab_runner.ipynb` | Team's step-by-step runner in free Colab |
| `app/` | Future local Streamlit demo |
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
3. Run the Phase 0 cells in order: mount Drive, clone `phase-0`, install, restart
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

## Next phases

1. Data acquisition scripts and actual dataset counts.
2. Shortcut audit, duplicates, and frozen balanced splits.
3. Shared preprocessing.
4. Frequency features.
5. Pixel baseline and batched extraction.
6. Fair training/tuning and primary-model selection.
7. Freeze, single evaluation, and hypothesis tests.
8. Local demo.
9. Report pack and reproduction guide.

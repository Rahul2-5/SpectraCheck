# AI Image FFT Detector

Research project: **Detecting AI-Generated Images Using Frequency-Domain Features**.
Team: Rahul Temkar, Shravani Rane, Jagruti Doke (MCA, MET Institute of Computer Science).

We will compare handcrafted frequency features against pixel features using
SVM, Random Forest, and XGBoost, evaluate once on a held-out test set, and build
a local Streamlit demonstration. A result near chance is still a valid research
finding. This is a single-generator study, not proof of image authenticity.

## Current stage

Phase 0 sets up the project only. Image processing, dataset downloads, model
training, and the app belong to later phases. Read [AGENTS.md](AGENTS.md) and the
PRD before working on a phase. See [the Phase 0 report](reports/phase_0_report.md)
for executed checks and remaining team checks.

## What the folders are for

| Path | Purpose |
| --- | --- |
| `config.yaml` | One place for experiment settings and Drive paths |
| `src/aidetect/` | Shared Python code; currently configuration, seeds, file hashes |
| `tests/` | Small automatic checks that need no dataset or internet |
| `scripts/` | Future commands for data, training, and evaluation |
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

## Run Phase 0 in Colab

1. Upload `notebooks/colab_runner.ipynb` to Google Colab.
2. The clone cell uses the configured remote,
   `https://github.com/Rahul2-5/SpectraCheck.git`. Push the local `phase-0` branch
   before running it; adjust the URL if the team uses a different repository.
3. Run the Phase 0 cells in order: mount Drive, clone `phase-0`, install, restart
   the Colab session if prompted, print versions, and run pytest.
4. Confirm no dependency conflicts and all tests pass. Keep the printed versions
   and results for the team review. Stop at the next-phase TODO.

The notebook checks runtime version/pins and imports the installed libraries.
It does not download research images or train models in this phase. Paths in
`config.yaml` target Drive; adjust them explicitly for future local data runs.

## GitHub check

The configured remote is `https://github.com/Rahul2-5/SpectraCheck.git`. Push `phase-0`
and inspect the **Python tests** workflow under Actions. Both matrix jobs must
pass. This phase has not pushed changes or run the remote workflow.

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
- Work one phase at a time, report what was actually checked, and wait for the
  team's review before continuing.

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

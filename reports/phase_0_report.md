# Phase 0 report

Status: **Implemented and verified locally; waiting for team checks.** Phase 1 has not started.

## What this phase did, in easy words

We built the empty project's foundation. Every later piece now has a folder.
The settings live in one file, so changing a setting later does not require
hunting through code. The helper code makes random runs repeatable and gives
files a digital fingerprint, so we can tell if the experiment changed.
Automatic tests check these helpers. A Colab notebook gives the team a place
to run the project, and a GitHub workflow will repeat the checks after pushes.
No images were downloaded, processed, or used to train a detector in this phase.

## Done

- `AGENTS.md`: updated persistent project state and recorded pending checks.
- `config.yaml`: centralized PRD defaults, separate data/features Drive folders, batch size, and initial real-image sampling limit.
- `src/aidetect/__init__.py`: created the importable package without image-processing logic.
- `src/aidetect/io_utils.py`: safe YAML loading, Python/NumPy seeds, chunked file SHA-256, and sorted-YAML config hashing.
- `tests/test_io_utils.py`: 15 tests covering configuration loading/failures, deterministic seeds, stable/changed config hashes, and binary file hashes.
- `requirements.txt`: pinned all 15 direct research libraries to versions installed in the local environment.
- `pyproject.toml`: editable package installation, src layout, Python minimum, and pytest configuration.
- `.gitignore`: keeps data, feature files, model binaries, local environment, and common secret files out of Git.
- `.gitattributes`: consistent text line endings and binary treatment for the source PRDs.
- `.github/workflows/tests.yml`: Linux test jobs for Python 3.11 and 3.13 on push/pull request.
- `notebooks/colab_runner.ipynb`: Drive mounting, safe clone of `phase-0`, pinned installation, version/import checks, pytest, STOP gate, and nine future-phase TODO cells.
- `docs/PHASE_REPORT_TEMPLATE.md`: standard report template for later phases.
- `README.md`: plain-language folder guide, Windows setup commands, Colab/GitHub checks, and project safeguards.
- `scripts/`, `app/`, `artifacts/`: empty tracked placeholders for later work; future modules are not falsely implemented as stubs.
- `.venv/`: isolated local environment, intentionally ignored by Git.
- `reports/phase_0_environment.json`: actual Python/platform/direct-version evidence, config hash, smoke-check results, and inspected library signatures.

## Tests

Final local environment: Windows, Python 3.13.5. Commands run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

Actual summaries:

```text
No broken requirements found.
15 passed in 0.72s
```

Passed: 15. Failed: 0. Skipped: 0 on the final run.

Additional executed checks:

- Installed the package with `python -m pip install --no-deps -e .` successfully.
- Imported all 15 direct libraries and asserted their installed versions equal `requirements.txt`.
- Parsed notebook JSON, verified unique cell IDs, compiled standard Python cells, checked blank outputs and nine future-phase TODOs. Colab-specific `%pip` cells were inspected but cannot be executed in plain Python.
- Parsed workflow YAML and verified the Python 3.11 matrix job is present.
- Checked sorted YAML and NumPy seed signatures from the installed packages; evidence is in the environment JSON.
- Searched for `256`: the main image size is in config; code references to SHA-256 and the binary fixture's byte range are unrelated constants.
- Reviewed `git diff --check` for whitespace errors.

Problems encountered and fixed:

- The sandbox initially blocked package downloads. The same project-local install succeeded with the required network access.
- Newest available NumPy/SciPy/XGBoost choices did not all match the Python 3.11 target. Selected compatible direct versions using pip's Python 3.11 metadata resolution, then installed those exact pins locally.
- An oversized binary test parameter produced an excessively long pytest test ID on Windows. Explicit short test IDs fixed it; all final tests pass.
- The first environment-evidence write found that `reports/` did not yet exist. Created it and reran the smoke checks successfully.

## Outputs produced (paths)

- `reports/phase_0_report.md` (this report)
- `reports/phase_0_environment.json` (verified environment and provenance)
- `notebooks/colab_runner.ipynb` (team runner)
- All foundation files listed under Done.

Current config SHA-256:

```text
e84c4f99730a922ca15e1382d309459bc6380066c5468356287ce9fef9d4733b
```

## Checks for the team to run in Colab

1. Make the local `phase-0` branch available on the configured GitHub repository. This setup has not pushed it. From a Git-enabled terminal, the team can run:

   ```text
   git push -u origin phase-0
   ```

2. Upload `notebooks/colab_runner.ipynb` to free Colab. Use a CPU runtime. Run the Phase 0 cells in order; they mount Drive, clone the configured repository/branch, and install the pinned environment. Restart the session after installation if Colab asks, then continue at the verification cell.
3. The verification cell prints the actual branch, commit, Python version, all 15 pinned library versions, and config hash. It runs pip check and raises on any mismatch/import error. Keep these outputs for review.
4. The exact test command in the notebook is:

   ```python
   subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_DIR, check=True)
   ```

   Expected: 15 passing tests. Do not continue if installation, pip check, or pytest fails; share the exact failing output for a Phase 0 fix.

5. In GitHub Actions, open **Python tests** for `phase-0` and verify both Linux jobs (3.11 and 3.13) pass.
6. Review `config.yaml` against the PRD. Data and features are intentionally stored in separate subfolders under `/content/drive/MyDrive/aidetect`.

## Assumptions I made

- The configured remote `https://github.com/Rahul2-5/SpectraCheck.git` is the intended team repository. If wrong, adjust origin and the notebook URL before pushing/running Colab.
- Python 3.11 is the common minimum for the chosen pins. The local machine only supplied Python 3.13.5. This narrows the PRD's Python 3.10+ allowance; Python 3.10 is not supported by this environment.
- Direct dependencies are pinned; pip resolves transitive dependencies. This is not a complete cross-platform lockfile. Before real experiments, runtime versions must also be saved with models as planned.
- Default paths separate `data/` and `features/` beneath the PRD's Drive base folder, preventing accidental mixing of images and feature files.
- The main dataset/generator defaults remain PRD choices, not evidence of supervisor approval. Their real schema/counts will be checked in Phase 1.

## Open questions / anything I could not verify

- No live Colab session or GitHub workflow was run here. Their results are pending team action; local success is not a claim that the whole gate passed.
- Python 3.11-compatible direct-package metadata was checked, but the actual 3.11/Linux install and test run remain the CI check.
- Dataset schema, availability, license details, and generator counts have not been verified against live data; this phase did not download any research images.
- Model training, calibration, statistics, and the app are unimplemented and untested, as required by Phase 0's limited scope.
- The source PRD files were observed moved from the root into `docs/` during this session, outside this implementation's edits. Their move is left untouched and unstaged; the phase commit stages only the implementation files explicitly.

## STOP

Waiting for team checks and `PROCEED TO PHASE 1`, as required by the PRD Phase 0 STOP gate and `AGENTS.md` working rule 1. Phase 1 will add the data-acquisition scripts and notebook commands.

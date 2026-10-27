# Phase 4 report

Status: **Frequency feature code and synthetic known-answer checks are complete locally. The required real-data train-only spectra review, feature files, split outputs, and remote CI remain pending.**

## Done

- `src/aidetect/features.py`: added centered FFT power spectra with optional fixed 2D Hann windows, shifted-DC radius maps, non-DC energy ratio, normalized spectral entropy, equal-width log-RAPSD, robust ring-MAD local peak count/strength, stable feature names, and the combined extractor.
- `config.yaml`: added `frequency.hann_window`, `frequency.epsilon`, and `frequency.mad_scale` settings. With `rapsd_rings: 64`, extraction returns 68 values: energy ratio, entropy, 64 RAPSD values, peak count, and peak strength.
- `tests/test_features.py`: added 15 synthetic tests covering the known 16-cycle cosine radius, optional window and constant inputs, DC exclusion, entropy, RAPSD ring count/logging, conjugate-pair peak counting, white noise, smooth gradients, deterministic finite vectors, feature-name alignment, and invalid inputs.
- `notebooks/colab_runner.ipynb`: replaced the Phase 4 placeholder with a training-only cell that processes 200 rows per class and saves mean log-spectrum and mean RAPSD figures with 95% bands under `reports/figures/`.
- `src/aidetect/__init__.py`, `README.md`, and `AGENTS.md`: recorded the Phase 4 architecture and current evidence state.
- `reports/phase_4_checks.json`: recorded local versions, signatures, test summaries, feature count, config hash, and pending real-data checks.

The fixed feature order is:

```text
energy_ratio, spectral_entropy, rapsd_ring_00 ... rapsd_ring_63,
peak_count, peak_strength
```

The active configuration SHA-256 is `fae16d16f5bc8891971219bc27a149e2fb4e2d4d8e2cddafe105ad07ae18fc10`. No real feature file, split output, or research figure was generated locally.

## Tests

- Exact focused command: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_features.py`
- Focused passed / failed / skipped: **15 / 0 / 0**
- Focused actual summary: `15 passed in 0.51s`
- Exact regression command: `.\\.venv\\Scripts\\python.exe -m pytest -q`
- Regression passed / failed / skipped: **94 / 0 / 0**
- Regression actual summary: `94 passed in 24.35s`
- Exact dependency check: `.\\.venv\\Scripts\\python.exe -m pip check`
- Actual summary: `No broken requirements found.`
- Notebook JSON parsing, unique IDs, and AST parsing of the Phase 3 and Phase 4 cells passed. The new Phase 4 cell has 64 source lines.
- Installed NumPy signatures for `fft2`, `fftshift`, and `hanning` were inspected and recorded in `reports/phase_4_checks.json`.
- The repository resize search found only `src/aidetect/preprocess.py`; no Phase 5 pixel-baseline resize has been implemented yet.
- During implementation, the first focused run found three issues: entropy included the DC bin in its normalization count, the tiny-image test used `K=64` with an overstrict ring-size restriction, and raw FFT residue created spurious no-window peaks. These were corrected; the focused and full results above are the final runs.

## Outputs produced (paths)

- `src/aidetect/features.py`
- `tests/test_features.py`
- `config.yaml` frequency settings
- Phase 4 review cell in `notebooks/colab_runner.ipynb`
- `reports/phase_4_checks.json`
- `reports/phase_4_report.md`
- Updated `README.md`, `AGENTS.md`, and `src/aidetect/__init__.py`

The Colab cell will produce these planned, real-data outputs after team review:

- `reports/figures/phase_4_mean_log_spectrum.png`
- `reports/figures/phase_4_mean_rapsd.png`

## Checks for the team to run in Colab

1. Clone/update `main`, install the pinned requirements, and run the notebook setup/test cells. Confirm the full test suite passes in the Colab runtime.
2. Complete the reviewed Phase 2 audit/split steps and Phase 3 preprocessing review first if their real Drive outputs do not already exist. Do not silently regenerate sealed split assignments.
3. Run the notebook cell under `## Phase 4 - Frequency features`. It uses only the first 200 deterministic `train` rows per label after joining `audit/active_manifest.csv` with `splits.csv`:

   ```python
   train = active.merge(
       frozen[["id", "split"]], on="id", how="inner", validate="one_to_one"
   )
   train = train[train["split"] == "train"].sort_values(["label", "id"])
   selected = train[train["label"] == label].head(200)
   power = power_spectrum(image, window=frequency.get("hann_window", True))
   ```

4. Review the saved mean log-spectrum and mean RAPSD plots for both labels. Confirm the figures are exploratory train-only diagnostics, not detector results or evidence of generalization. Record the exact selected counts and any preprocessing failures.
5. Confirm the output vector remains length 68 at `rapsd_rings: 64`, all values are finite, and feature names align with the vector order. Do not create or inspect validation/test features in this phase.
6. Check GitHub Actions on `main`; local success does not verify remote CI.

## Assumptions I made

- The configured Hann window is enabled for the project’s default extraction, while `power_spectrum(..., window=False)` remains available for known-answer tests and explicit ablations. Changing this setting changes the active config hash and downstream feature provenance.
- Epsilon is `1e-12` and the robust normalizer uses the configured `1.4826` MAD scale. Zero-power bins are clamped to epsilon before log peak scoring so floating-point FFT residue cannot become a false peak.
- RAPSD uses K equal-width intervals spanning radius 1 through `S//2`; bins outside that radius, including corners, are ignored. Empty synthetic rings return `log10(epsilon)` so the vector length and finiteness contract are preserved.
- The peak half-plane is the shifted upper half plus the positive horizontal axis, excluding DC. This selects one member of each conjugate pair for real images.
- The Phase 4 plot cell is diagnostic only and reads training rows. No real dataset or held-out test data was accessed locally.

## Open questions / anything I could not verify

- The team has not run the 200-real/200-AI Colab spectra review, so actual plots, image counts, and data-specific spectral behavior are unknown.
- Real dataset availability, generator counts, audit decisions, duplicate structure, split feasibility, and preprocessing rejection counts remain unverified.
- No real feature files were extracted, no model was trained, and no validation/test row was read.
- No remote GitHub Actions result was verified; the previous push authentication issue remains external to this code change.

## STOP

Phase 4 is complete locally. Waiting for team checks and your explicit **`PROCEED TO PHASE 5`** approval before starting Phase 5.

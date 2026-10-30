# Phase 4 report

Status: **Frequency feature code and synthetic checks are complete locally, and the R2 train-only real-data spectra review is complete. Feature extraction, training, freeze, test evaluation, and remote CI remain unverified or not authorized.**

## Done

- `src/aidetect/features.py`: added centered FFT power spectra with optional fixed 2D Hann windows, shifted-DC radius maps, non-DC energy ratio, normalized spectral entropy, equal-width log-RAPSD, robust ring-MAD local peak count/strength, stable feature names, and the combined extractor.
- `config.yaml`: added `frequency.hann_window`, `frequency.epsilon`, and `frequency.mad_scale` settings. With `rapsd_rings: 64`, extraction returns 68 values: energy ratio, entropy, 64 RAPSD values, peak count, and peak strength.
- `tests/test_features.py`: added 15 synthetic tests covering the known 16-cycle cosine radius, optional window and constant inputs, DC exclusion, entropy, RAPSD ring count/logging, conjugate-pair peak counting, white noise, smooth gradients, deterministic finite vectors, feature-name alignment, and invalid inputs.
- `notebooks/colab_runner.ipynb`: replaced the Phase 4 placeholder with a training-only cell that processes 200 rows per class and saves mean log-spectrum and mean RAPSD figures with 95% bands under `reports/figures/`.
- `src/aidetect/__init__.py`, `README.md`, and `AGENTS.md`: recorded the Phase 4 architecture and current evidence state.
- `reports/phase_4_checks.json`: records local implementation checks and the R2 live Colab review provenance, figures, and checksum-backed review JSON.

## R2 real-data review (30 October 2026)

The Phase 4 review ran in the live Colab notebook on `main` code commit `1170199bbbc62ad32e9c436c3cbad90d2f6d157f`. It verified these R2 provenance records before selecting or opening images:

- `config_r2.yaml`: `ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6`
- frozen R2 split: `1e823b5df253951bc2a7221b618fc7c0e5716f85066d509a3d9442c56ccfef93`
- active manifest: `2af4d01ee69adadc0f28c741ad2bafed846d73c958b0f114efca939122ee05e8`
- R2 audit result: `0ec413d2f64e3e9da9da6e97be41c5bf9071a061cf987a472bacdafaa7fce145`

The code selected the first 200 IDs per label from the sorted frozen TRAIN rows only, then opened those 400 image files and checked each source SHA-256 against the active manifest. Every image used shared preprocessing version 2 and the fixed 256x256 centre patch with the single JPEG Q96/4:4:4 treatment. The resulting arrays were finite `float64` in `[0,1]`. The cell computed aggregate log-power spectra and RAPSD with the configured Hann window, epsilon `1e-12`, and 64 rings. The saved plots show both class means and the RAPSD 95% mean bands; they were visually reviewed as descriptive diagnostics only.

The figures and JSON record, including selected TRAIN IDs and source hashes, were backed up with verified checksums under `aidetect/r2/review_backup/phase_4_spectrum_review_20261030T152916Z`. The review JSON SHA-256 is `3e794badb5ea3faa0e20bb1caa4eefb9ccae085109363fbf02e5815d08560fe5`. Figure SHA-256 values are recorded in `reports/phase_4_checks.json`. No validation/test image files or feature rows were accessed; no extraction batches, model, freeze, or evaluation outputs were created. The original raw-input audit remains blocked, the narrower treated-input audit is the one used by R2, and supervisor sign-off remains unverified.

The exported Colab output briefly encountered an inline-renderer login/cookie error. The latest notebook export retained both figures and the completion summary, and the figures were reviewed from that saved export; the Drive backup directory contains both PNGs and the JSON plus `.sha256` sidecar.

The fixed feature order is:

```text
energy_ratio, spectral_entropy, rapsd_ring_00 ... rapsd_ring_63,
peak_count, peak_strength
```

The original configuration SHA-256 is `fae16d16f5bc8891971219bc27a149e2fb4e2d4d8e2cddafe105ad07ae18fc10`. R2 uses the separate `config_r2.yaml` hash recorded below. The live review produced aggregate figures on Drive; it did not produce feature files or alter split outputs.

## Local implementation checks (historical)

These checks were run during the original Phase 4 implementation; they were not rerun during the R2 live-data review.

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

The live R2 Colab review saved these aggregate figures to Drive:

- `aidetect/r2/review_backup/phase_4_spectrum_review_20261030T152916Z/phase_4_mean_log_spectrum.png`
- `aidetect/r2/review_backup/phase_4_spectrum_review_20261030T152916Z/phase_4_mean_rapsd.png`
- `aidetect/r2/review_backup/phase_4_spectrum_review_20261030T152916Z/phase_4_spectrum_review_20261030T152916Z.json` and its `.sha256` sidecar

## R2 review procedure and scope

1. Verified the R2 config, frozen split, active-manifest, and audit-result hashes before image access.
2. Filtered frozen `TRAIN` membership, sorted by label and ID, and selected exactly 200 images per class. The review record includes all selected IDs and source checksums.
3. Called shared version-2 preprocessing and computed aggregate mean log-power spectra and mean RAPSD with 95% bands.
4. Visually reviewed both saved figures as train-only diagnostics. They are not detector results and do not support generalization claims.
5. The code explicitly records that no validation/test image files or feature rows were opened and no feature-extraction or batch files were written.

The remote GitHub Actions status remains unverified. The local synthetic checks below are historical implementation evidence, not a fresh Colab test-suite run.

## Assumptions I made

- The configured Hann window is enabled for the project’s default extraction, while `power_spectrum(..., window=False)` remains available for known-answer tests and explicit ablations. Changing this setting changes the active config hash and downstream feature provenance.
- Epsilon is `1e-12` and the robust normalizer uses the configured `1.4826` MAD scale. Zero-power bins are clamped to epsilon before log peak scoring so floating-point FFT residue cannot become a false peak.
- RAPSD uses K equal-width intervals spanning radius 1 through `S//2`; bins outside that radius, including corners, are ignored. Empty synthetic rings return `log10(epsilon)` so the vector length and finiteness contract are preserved.
- The peak half-plane is the shifted upper half plus the positive horizontal axis, excluding DC. This selects one member of each conjugate pair for real images.
- The Phase 4 plot cell is diagnostic only and reads training rows. No real dataset or held-out test data was accessed locally.

## Open questions / anything I could not verify

- The original raw-input audit remains blocked. R2 uses the reviewed treated-input domain; the original failure and the narrower scope must remain explicit.
- Supervisor sign-off remains unverified.
- No real feature files were extracted, no model was trained, and no validation/test image or feature row was read.
- No remote GitHub Actions result was verified.

## STOP

The R2 Phase 4 train-only spectra review is complete. Stop here and wait for your explicit Phase 5 authorization before starting Phase 5.

# Phase 3 report

Status: **R2 Phase 3 real-data preprocessing review is complete in live Colab. The original version-1 code and synthetic-check record below is retained as historical context.**

## R2 real-data review (30 October 2026)

The current R2 preprocessing gate completed against the frozen Drive data. The live notebook was `colab_runner_11.ipynb` (Drive ID `1h3v8pHEQmK6QwPHWIL9K0Max6GhnE1j7`), on `main`, source commit `1170199bbbc62ad32e9c436c3cbad90d2f6d157f`.

- Verified `config_r2.yaml` SHA-256 `ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6`, frozen split SHA-256 `1e823b5df253951bc2a7221b618fc7c0e5716f85066d509a3d9442c56ccfef93`, active manifest SHA-256 `2af4d01ee69adadc0f28c741ad2bafed846d73c958b0f114efca939122ee05e8`, and audit result SHA-256 `0ec413d2f64e3e9da9da6e97be41c5bf9071a061cf987a472bacdafaa7fce145`.
- The notebook confirmed the frozen split and active manifest have matching IDs, labels and groups, with no duplicate group crossing splits. It selected 20 deterministic rows per label only after filtering to research `split == "train"`, using seed 42, verified source hashes, and called the shared version-2 `preprocess_image`.
- Machine checks confirmed 256×256 fixed centre patches; exactly one JPEG Q96/4:4:4 treatment; no resize or upscaling; finite float64 outputs in `[0,1]`; and zero undersized TRAIN candidates for labels 0 and 1.
- I visually reviewed both rendered 4×5 grids: all 40 outputs were legible grayscale centre patches with no decode corruption or obvious stretching. Some IDs include `validation_`; these are upstream item IDs, not research validation membership. The selection code filters using the frozen research split before opening paths.
- Checksummed Drive evidence: `/content/drive/MyDrive/aidetect/r2/review_backup/phase_3_preprocessing_review_20261030T143633Z.json`, SHA-256 `47aaca58c83b3d8b0ddcbe3e8cd137f87e75764735f0dc4443c857b8b1bc7fb6`. The Drive JSON records the machine provenance, selected IDs, source hashes and output-array hashes; the rendered grids were shown by Colab, and the manual visual judgement is recorded here.
- The latest live Colab notebook replaces the removed canonical runner. Its executable review code and text outputs are preserved in Git; embedded sample-image pixels from Phase 1/3 outputs are omitted to avoid redistributing dataset images. The cells regenerate the previews when run in Colab.

The original raw audit remains blocked; the R2 treated-input audit is a narrower amended scope with residual confounding. Supervisor sign-off remains unverified. No feature extraction, training, freeze, test evaluation, or Phase 4 work was performed in this review. Stop here until Phase 4 is explicitly authorized.

## Historical version-1 implementation record

## Done

- `src/aidetect/preprocess.py`: added the single `preprocess_image(src, cfg)` entry point for paths and encoded bytes. It fully decodes with Pillow, applies EXIF orientation, converts RGB then grayscale `L`, center-crops to a square, downsizes only larger inputs with LANCZOS, rejects undersized inputs, and returns finite float64 arrays in `[0, 1]`.
- `config.yaml`: added the fixed `preprocess` contract so the preprocessing settings and `PREPROCESS_VERSION` are included in the active configuration hash.
- `tests/test_preprocess.py`: added nine synthetic tests for path/bytes parity, shape/range/dtype, determinism, center cropping, equal-size no-resize behavior, EXIF orientation, undersized/corrupt input errors, invalid settings, and config-hash sensitivity.
- `notebooks/colab_runner.ipynb`: replaced the Phase 3 placeholder with a train-only review cell that samples 20 rows per class, displays successful arrays, and counts undersized images.
- `src/aidetect/__init__.py`: updated package-stage documentation.
- `README.md`: documented the shared preprocessing contract and current remaining phases.
- `AGENTS.md`: recorded Phase 3 progress, evidence, active config hash, and pending team checks.
- `reports/phase_3_checks.json`: recorded the local runtime, test result, Pillow signatures, config hash, and notebook checks.

The fixed preprocessing contract is:

```text
EXIF transpose -> RGB -> grayscale L -> centered square crop
-> LANCZOS downsize only when larger than target -> float64 / 255.0
```

The active configuration SHA-256 is `bf1e6f3b4f4cdc5f95d0503e55533aee54f48fc054ba2a6ea89df305a0af2c6c`. The Phase 2 configuration hash in its historical report remains unchanged; no real audit, split, or data output was regenerated locally.

## Tests

- Exact command: `.\\.venv\\Scripts\\python.exe -m pytest -q`
- Passed / failed / skipped: **79 / 0 / 0**
- Actual summary: `79 passed in 21.17s`
- Exact command: `.\\.venv\\Scripts\\python.exe -m pip check`
- Actual summary: `No broken requirements found.`
- Exact notebook checks: JSON parsing, unique cell IDs, and AST parsing of the new `phase-3-review-code` cell; all passed. A generic AST parse of every notebook code cell is not applicable because the existing setup cell contains Colab `%pip` magic.
- Pillow API signatures were inspected for `Image.open`, `ImageOps.exif_transpose`, `Image.Image.resize`, and `Image.Resampling.LANCZOS`; the checked signatures are recorded in `reports/phase_3_checks.json`.
- The first focused test collection exposed an incorrect Pillow import (`PIL.UnidentifiedImageError`); it was corrected to Pillow's public `from PIL import UnidentifiedImageError`, after which `tests/test_preprocess.py` passed 9/9. The final full-suite result above is the authoritative result.

## Outputs produced (paths)

- `src/aidetect/preprocess.py`
- `tests/test_preprocess.py`
- `config.yaml` preprocessing settings
- Phase 3 review cell in `notebooks/colab_runner.ipynb`
- `reports/phase_3_checks.json`
- `reports/phase_3_report.md`
- Updated `README.md`, `AGENTS.md`, and `src/aidetect/__init__.py`

No dataset images, feature files, research split hash, model, or evaluation output was produced.

## Original version-1 checks for the team (historical; superseded for R2)

1. Clone/update `main`, install the pinned requirements, and run the notebook setup/test cells. Confirm the full test suite passes in the Colab runtime.
2. Complete the reviewed Phase 2 audit/split steps first if the real Drive outputs do not already exist. Do not silently regenerate a sealed split or overwrite an existing audit.
3. Run the Phase 3 notebook cell under `## Phase 3 - Shared preprocessing`. It executes the equivalent of:

   ```python
   cfg = load_config(PROJECT_DIR / "config.yaml")
   DATA_ROOT = Path(cfg["paths"]["data_root"])
   active = pd.read_csv(DATA_ROOT / "audit/active_manifest.csv", dtype={"id": str})
   frozen = pd.read_csv(DATA_ROOT / "splits.csv", dtype={"id": str})
   train = active.merge(frozen[["id", "split"]], on="id", how="inner", validate="one_to_one")
   train = train[train["split"] == "train"].sort_values(["label", "id"])
   # preprocess_image is called for the first 20 rows of each label only.
   ```

4. Review both displayed 4x5 grids for normal grayscale images with no stretching. Record `smaller than target` counts separately for real (`0`) and AI-generated (`1`); do not drop them silently. If more than a few percent are rejected, the team must decide whether to document exclusion or change `image_size` for the entire experiment before freezing data.
5. Search the repository and confirm the only executable image resize is the shared preprocessing resize. The Phase 5 pixel-baseline resize is a planned, documented exception and has not been implemented.
6. Check GitHub Actions on `main`; local success does not verify remote CI.

## Assumptions I made

- The fixed operations are represented by `PREPROCESS_VERSION` and mirrored exactly under `config.yaml:preprocess`. If the team changes any operation, version, output size, or undersized-image policy, the configuration hash changes and downstream artifacts must be treated as a new configuration.
- The PRD's no-upscaling policy is enforced in the shared function. Phase 2 already logs undersized files before split freezing; the notebook review is an additional visible check, not permission to silently alter the frozen dataset.
- Pillow's `RGB` then `L` conversion is the prescribed single grayscale conversion. No class-specific, feature-specific, or app-specific preprocessing was added.
- Since no real research split is present locally, adding the Phase 3 settings does not overwrite or regenerate team data. Any team-created Phase 2 audit/split records must retain their recorded historical hash and be reviewed against the active configuration before downstream work.

## Open questions at the original version-1 report date (historical)

- The team has not run the 20-real/20-AI Colab preprocessing review, so the actual undersized count and visual quality are unknown.
- Real dataset availability, generator counts, metadata-probe decisions, duplicate structure, and split feasibility remain unverified as recorded in Phase 2.
- No remote GitHub Actions result was checked in this phase.
- The planned Phase 4 frequency module has not been started.

## STOP

R2 Phase 3 is complete. Stop here and wait for explicit Phase 4 authorization; supervisor sign-off remains unverified.

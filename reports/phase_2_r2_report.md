# Phase 2 R2 implementation and execution

## Done

User approved R2 on 30 October 2026. Independent supervisor sign-off remains unverified. The amendment changes preprocessing to fixed 256x256 centre patches with one common JPEG Q96/4:4:4 generation and moves the numerical audit gate to treated inputs. Raw failure, residual source/content/compression confounding and exploratory amendment history remain disclosed.

Implemented `config_r2.yaml`, the shared R2 canonical treatment in `preprocess.py`, separate official audit/staging in `r2_audit.py`, canonical/original duplicate union, raw/treated hash checks in split creation and extraction/resume, configurable reviewed serving config and R2 scope note, and `notebooks/colab_r2.ipynb`. Historical `config.yaml`/version-1 behavior remains supported. New R2 roots prevent reuse of old batches/models. No treated image caches are used as extraction inputs; source hashes always bind original bytes, and canonical hashes bind one treatment of those originals.

R2 config SHA-256: `ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6`.

## Tests

- First full regression: `.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-r2-full-a -p no:cacheprovider`: **174 passed in 46.22s**.
- Final full regression after recovery/parity additions: `.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-r2-final-a -p no:cacheprovider`: **175 passed in 44.65s**.
- All new notebook code cells compile. `git diff --check` reviewed.

Synthetic checks cover exact crop coordinates and encoder bytes, app/shared feature parity, no-upscaling, separate roots, raw file preservation, config/manifest/image integrity, canonical hashes, group-separated splits, blocked probes, sealed-run refusal and interrupted staging recovery. The new code uses installed Pillow/sklearn signatures already exercised by tests. No new dependency pins or frontend code changes.

## Outputs

- Local implementation/config/notebook and tests above.
- Official Drive audit/split execution: pending at initial code publication; actual observations will be appended after execution.
- Intended R2 data root `/content/drive/MyDrive/aidetect/r2/data`, feature root `.../r2/features`, repository artifacts `artifacts/r2`, reports `reports/r2`.

## Team Colab checks

Use `notebooks/colab_r2.ipynb`. In the existing checkout, pull main with `git pull --ff-only origin main`, then run:

```python
subprocess.run([sys.executable, '-u', 'scripts/run_audit.py', '--config', 'config_r2.yaml'], cwd=PROJECT_DIR, check=True)
```

Review the R2-domain audit, raw failure disclosure, both CV means <=0.60, and zero mixed-label duplicate groups. Only after that pass/review:

```python
subprocess.run([sys.executable, '-u', 'scripts/make_splits.py', '--config', 'config_r2.yaml'], cwd=PROJECT_DIR, check=True)
```

Verify exact per-split class balance, no crossing groups, actual split proportions and the split hash. Preserve R2 artifacts with the Drive data before restarting the runtime. Do not run the old all-phase notebook as an R2 workflow.

## Assumptions

The previously recorded source audit SHA-256 and source cohort are unchanged. R2 copies and hash-verifies existing raw images to separate Drive storage; it does not redownload. Enough free Drive space is required for the copied selected cohort. Source and treated metadata predictiveness are different questions; the narrower gate is not evidence that source confounding vanished.

## Open items

Official R2 audit/splits, supervisor sign-off, train-only visual reviews, real extraction/training, practical target, freeze/evaluation and real-model app review remain pending until individually evidenced. The Colab-host pandas requirement conflict remains; successful software tests do not make host pip check pass.

STOP after revised Phase 2. Phase 3 real-data review requires the verified R2 splits and next-phase authorization; do not start extraction/training/evaluation in this task.

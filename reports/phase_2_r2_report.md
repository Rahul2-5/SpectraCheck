# Phase 2 R2 implementation and execution

**Completed in live Colab:** official R2 treated-input audit passed (both probes 0.5814), immutable class-balanced 70/15/15 splits verified, and 175 tests passed locally and in Colab. Original audit failure remains preserved. See the actual execution section and `phase_2_r2_checks.json` below. Stop before Phase 3 until authorized.

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
- Official Drive audit/split execution: completed and verified; exact observations are recorded below.
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

Supervisor sign-off, train-only visual reviews, real extraction/training, practical target, freeze/evaluation and real-model app review remain pending. The Colab-host pandas requirement conflict remains; successful software tests do not make host pip check pass.

STOP after revised Phase 2. Phase 3 real-data review requires the verified R2 splits and next-phase authorization; do not start extraction/training/evaluation in this task.

## Observed official Colab execution

Executed the published R2 implementation (`7f7bc13`) through the user's live Chrome/Colab session on 30 October 2026. The command exited 0 and the **official R2 treated-input audit passed** on 2,500 originals per class. Logistic regression and depth-3 tree each had mean accuracy **0.5814**, matching the earlier diagnostic. Independently remeasured raw-file probes remained 0.9998/1.0000. The raw prior audit is retained unchanged, with an exact-byte copy under the R2 audit folder. Mixed-label duplicate groups: **0**.

Then ran `scripts/make_splits.py --config config_r2.yaml` once. Verified raw and canonical hashes before splitting, exact class balance, split CSV hash and **zero groups crossing splits**:

| Split | Real | AI | Total |
| --- | ---: | ---: | ---: |
| Train | 1,750 | 1,750 | 3,500 |
| Validation | 375 | 375 | 750 |
| Test | 375 | 375 | 750 |

Actual split SHA-256: `1e823b5df253951bc2a7221b618fc7c0e5716f85066d509a3d9442c56ccfef93`.

R2 active-manifest SHA-256: `2af4d01ee69adadc0f28c741ad2bafed846d73c958b0f114efca939122ee05e8`.

R2 official audit-result SHA-256: `0ec413d2f64e3e9da9da6e97be41c5bf9071a061cf987a472bacdafaa7fce145`.

Fresh Colab regression command `python -m pytest -q`: **175 passed in 26.81s**, exit 0. Actual stdout/stderr saved to Drive. Verified checksum backups of config, raw prior audit, R2 result/report and both artifact split records are under `/content/drive/MyDrive/aidetect/r2/review_backup/`. The images and `data/splits.csv` already reside in the separate Drive R2 root. Local committed `artifacts/r2/splits.json` and `.sha256` reproduce the actual records observed in Colab; no split rows/images are published.

No feature extraction, model training, freeze or held-out evaluation was run. The remaining experimental limitations and unverified supervisor sign-off still apply. Use R2 configuration explicitly in the next phase; the historical runner's Phase 3 cell still targets the blocked original experiment.

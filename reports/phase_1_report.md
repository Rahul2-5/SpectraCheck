# Phase 1 report

Status: **Acquisition code implemented and verified offline; real downloads and team checks pending.** Phase 2 has not started.

## What this phase did, in easy words

The project can now collect the images it will study. One command copies a small
CIFAKE practice sample. Another streams the main dataset and saves Stable
Diffusion 1.5 images plus a random sample of real photos. Neither command edits
or re-saves image pixels, preserving the original evidence for the later audit.

Every saved image gets a row in a manifest: a list saying where it came from,
its label, file size, and fingerprint. Generator counts tell the team whether
enough suitable data exists. If Colab disconnects, rerunning can reuse intact
saved images, while catching changed settings and conflicting files.

## Done

- `src/aidetect/data.py`: original-byte acquisition, byte-signature format detection, bounded reservoir sampling, schema checks, commit resolution, manifests/provenance, and interruption-safe file writes.
- `scripts/download_main.py`: configuration-driven public Tiny-GenImage collection; returns exit code 2 after saving outputs if fewer than the configured minimum SD15 images are present.
- `scripts/download_pilot.py`: copies a balanced pilot from an existing extracted Kaggle CIFAKE folder; no API key or automated archive download.
- `config.yaml`: added source split names, requested revision, minimum SD15 count, and separate pilot size/count settings.
- `requirements.txt`: declared already installed `huggingface-hub==2.3.0` explicitly; there are now 16 direct pins. Refreshed editable package metadata successfully.
- `tests/test_data.py`: synthetic image and streaming checks, including an actual local Parquet round trip through the installed datasets library.
- `tests/test_download_commands.py`: tests CLI arguments, config forwarding, and insufficient-data exit behavior without network access.
- `notebooks/colab_runner.ipynb`: clones `main`; supplies pilot/main commands, manifest/file-size checks, generator-count output, and five random readable images per class. Phases 2-9 remain TODOs.
- `README.md`: acquisition commands, Kaggle folder layout, outputs, resume behavior, main-only Git policy, and network-transfer limitations.
- `AGENTS.md`: preserved current memory, pending real checks, and the user's branch override.
- `reports/phase_0_report.md`: updated actionable push/clone instructions to `main` while preserving the original test record.
- `reports/phase_1_checks.json`: local environment, config hash, signature checks, source links, and notebook static-check evidence.

User override: all commits and pushes go to `main` only. The completed Phase 0
history, including the existing PRD move into `docs/`, was fast-forwarded onto
local `main`. No phase branch is pushed by this work.

Publication status: the push was rejected by automatic approval review before
execution. Its reason was that the source/reports may be private and the user
must explicitly authorize upload to `https://github.com/Rahul2-5/SpectraCheck.git`.
Work is committed locally; publication requires that destination-specific approval.

## Tests

Final local commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/download_main.py --help
.\.venv\Scripts\python.exe scripts/download_pilot.py --help
```

Actual final summaries:

```text
42 passed in 2.85s
No broken requirements found.
```

Passed: 42 total (15 prior tests, 27 new cases). Failed: 0. Skipped: 0.

Tests require no dataset downloads or credentials. They verify original bytes,
format detection despite misleading filenames, actual ClassLabel resolution
with SD15 deliberately assigned ID 0, all-generator/zero counts, capped seeded
sampling, manifest/hash integrity, missing-file recovery, unchanged-file reuse,
interruption recovery, and rejection of conflicting bytes/settings/source commits.
Pilot tests check balancing, original-byte copying, source layout, and unsafe
source/output overlap. CLI tests verify the sample-count gate.

Additional checks passed: all 16 installed pin versions/imports; notebook JSON,
cell IDs, plain-Python cell syntax, clean outputs, `main` clone instruction, eight
future-phase TODOs; CLI help commands; whitespace review with `git diff --check`.

Integration-test environment issues fixed: the Hugging Face loader initially
tried an unwritable default cache; an explicit temporary cache then hit Windows'
long-filename limit. The test now uses a short, temporary workspace cache and
cleans it up. The production downloader was not changed to bypass these tests.

## Dataset fields and APIs checked

The official [Tiny-GenImage dataset card](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/blob/main/README.md?code=true)
documents `image` as Image, `label`/`generator` as ClassLabel, source splits
`train`/`validation`, real/fake binary labels, and the SD14/SD15 generator names.
It states the CC BY-NC-SA 4.0 license. Code checks mappings again at runtime and
does not hard-code the generator IDs. This review does not establish actual
per-generator counts in the full collection.

The [official image-loading documentation](https://huggingface.co/docs/datasets/image_load)
and [Image reference](https://huggingface.co/docs/datasets/package_reference/main_classes)
support obtaining encoded path/bytes with `Image(decode=False)`. Installed
datasets 5.1.0 and huggingface-hub 2.3.0 signatures were inspected for
`load_dataset`, `Image`, `ClassLabel`, `cast_column`, `select_columns`, and
`HfApi.dataset_info`; exact signatures are saved in the checks JSON.

The [Kaggle CIFAKE page](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
is linked for manual acquisition; the actual extracted folder still needs team verification.

## Outputs produced (paths)

Local deliverables: scripts, tests, notebook, docs, and
`reports/phase_1_checks.json`. No real image collection or real counts were
produced locally.

Outputs the team commands will produce on Drive:

```text
data_root/
  pilot/{real/, ai/, manifest.csv, acquisition.json}
  main/{real/, ai/, manifest.csv, acquisition.json}
```

Manifest columns: `id, rel_path, label, generator, source_split, ext, n_bytes,
sha256, config_hash`. The JSON records source revision/settings, source class
mappings, all generator counts, saved class counts, manifest hash, and completion
state. Temporary files are not counted as completed images.

Current config SHA-256:

```text
7bdbebb71fb20c71a233bcb6ca6e5aa61dc5e3fb6b1ad3a66bf6358e8eeed188
```

This differs from Phase 0 because Phase 1 adds acquisition/pilot settings; no
frozen splits or experimental feature files exist yet.

## Checks for the team to run in Colab

Run the updated notebook from `main` on a CPU runtime. Complete setup/version
verification first. Download/extract CIFAKE manually and set `CIFAKE_SOURCE_DIR`
to the parent of its `train/` and `test/` folders. Then run the notebook commands:

```python
subprocess.run([sys.executable, "scripts/download_pilot.py", "--source-dir", str(CIFAKE_SOURCE_DIR)], cwd=PROJECT_DIR, check=True)
subprocess.run([sys.executable, "scripts/download_main.py"], cwd=PROJECT_DIR, check=True)
```

Expected pilot counts: 500 real and 500 AI images. Main counts must be measured;
do not substitute PRD estimates. Record SD15 and SD14 counts in the team tracker.
If saved SD15 count is below 1,000, exit code 2 is an intentional STOP and
supervisor consultation is required before Phase 2. Outputs remain available.

Run `phase-1-checks` in the notebook: it compares manifest rows with image files,
checks recorded byte sizes/completion, prints generator counts, and opens five
seeded random images per class. They must all display correctly. Inspect GitHub
Actions for `main`; local tests are not evidence of remote CI success.

## Assumptions I made

- The user saying proceed authorizes Phase 1 implementation; it does not prove
  the previous Colab/GitHub checks passed. Those results were not supplied.
- CIFAKE is supplied already extracted by the team. No Kaggle credentials are
  handled. Pilot configuration remains separate from the main experiment.
- Two passes are acceptable to keep RAM usage bounded. Streaming/resume may
  still transfer the full upstream Parquet files; resume skips valid disk writes,
  not the entire preceding network scan.
- Only JPEG and PNG are accepted at acquisition, matching the planned sources.
  Other byte signatures cause a clear failure rather than re-encoding.
- Source rows and revision remain consistent within the resolved commit. Changed
  acquisition settings or an updated upstream revision require a fresh output
  folder, or an explicitly consistent pinned revision/configuration.

## Open questions / anything I could not verify

- Real Colab collection, exact generator counts, source-file readability, and
  full-download runtime/storage remain team checks. Local fixtures are synthetic.
- The live dataset card was reviewed, but the full hosted data path has not been
  executed in Colab. Runtime schema checks will fail clearly if it differs.
- Dataset choice and insufficient-data fallback remain supervisor decisions;
  no fallback or generator switch has been performed.
- Balancing, metadata-shortcut probes, duplicates, and final splits belong to
  Phase 2. No scientific conclusion or accuracy claim is made in Phase 1.

## STOP

Wait for the team download/check outputs and authorization to proceed to Phase 2.

# Phase 2 report

Status: **Audit and split code implemented and verified locally with synthetic data. Actual dataset gate and research splits pending Colab/team checks.** Phase 3 has not started.

## What this phase did, in easy words

This phase checks whether a detector could cheat. For example, if real photos
are JPEGs and AI images are PNGs, a model might learn file format instead of
anything about AI generation. Two small models try to classify images using
only format, quality, size, and file length. If either does too well, the code
applies the same planned JPEG conversion to both classes and checks again.
If the shortcut remains, the project stops before creating experiment splits.

It also finds exact and visually similar files. Similar files stay in one group,
so a near-copy cannot appear in training and later testing. The final split
allocator keeps each group together, gives each split equal real/AI counts, and
aims for the planned 70/15/15 proportions. A digital fingerprint locks the
assignments against silent changes.

## Done

- `src/aidetect/audit.py`: byte-based format detection, full decoding/corruption checks, dimensions/mode/file-size measurements, JPEG quantization-table quality estimation with residuals, file hashes, perceptual hashes, two metadata-only CV probes, deterministic identical JPEG remediation, and transitive duplicate grouping.
- `src/aidetect/splits.py`: seeded real-class balancing, whole-group integer split allocation, independent split validation, and once-only CSV/hash/provenance creation.
- `src/aidetect/audit_workflow.py`: verifies Phase 1 collection integrity, excludes undersized files before freezing, runs/re-runs probes, preserves originals, creates active manifests and audit reports/plots, and gates split creation on an unchanged passing audit.
- `scripts/run_audit.py`: runs the audit; exit code 2 means the metadata shortcut remains and splits must not proceed.
- `scripts/make_splits.py`: verifies audit/config/image hashes, allocates/validates splits, and writes their fingerprint/provenance.
- `config.yaml`: centralized audit, hash-distance, plotting, split tolerance/time-limit, report/artifact path settings.
- `tests/test_audit.py`, `tests/test_splits.py`, `tests/test_audit_workflow.py`, `tests/test_audit_commands.py`: 28 new cases, including full synthetic audit-to-freeze runs and deliberate blocked cases.
- `notebooks/colab_runner.ipynb`: Phase 2 commands, report/figure viewing, probe/count/group/hash checks, Drive backup, and STOP gate. Phases 3-9 remain TODOs.
- `README.md`: plain-language commands, paths, gate rules, provenance, remediation, and immutability instructions.
- `AGENTS.md`: updated current state, implemented eligibility policy, stronger split method, and remaining real-data checks.
- `reports/phase_2_checks.json`: inspected signatures, installed dependency versions, current config hash, and notebook static-check evidence.

Work remains on `main`, following the user's branch/push policy. No actual
dataset images or synthetic stand-in research hashes are included in this commit.

Publication status: automatic approval review rejected the Phase 2 push before
execution, stating that the previous explicit repository approval covers only
Phase 0/1. The Phase 2 work is saved locally; pushing these new source files,
tests, notebook, and reports requires user approval for that payload at
`https://github.com/Rahul2-5/SpectraCheck.git`, branch `main`.

## Tests

Final local commands:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.venv/matplotlib'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/run_audit.py --help
.\.venv\Scripts\python.exe scripts/make_splits.py --help
```

Actual summaries:

```text
70 passed in 16.29s
No broken requirements found.
```

Passed: 70 (42 previous + 28 new). Failed: 0. Skipped: 0.

New checks establish that:

- Known standard JPEG qualities are recovered from tables, custom-table estimates have nonzero residuals, and PNGs have no invented quality.
- Byte signatures override misleading filenames, corrupt files fail, and source hashes/path containment are checked.
- Balanced random metadata scores near chance, while a deliberate format shortcut is detected perfectly. Imbalanced/non-finite probe inputs fail.
- JPEG normalization is repeatable within the installed environment, yields RGB/4:4:4 output, and leaves original bytes untouched.
- Duplicate grouping is transitive, stable under row ordering, and preserves original relationships after normalization.
- Grouped splits are deterministic, exactly class-balanced, within tolerance, and never cross duplicate groups. Mixed-label groups remain together. Impossible/too-small group allocations fail.
- Split files/provenance cannot be silently overwritten; identical reruns preserve them, while tampering/partial freezes fail.
- Full synthetic audit-to-split flow works. Both normalization outcomes (pass and remaining shortcut) are exercised; blocked audits cannot produce splits. Missing real JPEG quality produces a blocked report. Config/image changes block freezing. Undersized files are logged before freezing.

Additional checks: all 16 installed library pins/imports, notebook syntax/unique
IDs/empty outputs/main clone/seven TODOs, command help, and whitespace review.
Reviewed a generated synthetic metadata plot visually: titles, labels, legends,
and axes are legible and unclipped. These images are test previews, not real findings.

## Outputs produced (paths)

Existing local deliverables are code, tests, the notebook, this report, and
`reports/phase_2_checks.json`. Real data outputs have **not** been produced here.

The team commands will generate:

- `data_root/audit/original_audit.csv`: measurements for original acquired files.
- `data_root/audit/excluded.csv`: undersized/excess-real exclusions and reasons.
- `data_root/audit/active_manifest.csv`: balanced eligible files, current/original hashes, metadata, and group IDs; paths are relative to `data_root`.
- `data_root/audit/audit_result.json`: input/config hashes, before/final probes, decisions, exclusions, duplicate counts, and active-manifest hash.
- `data_root/audit/normalized/`: separate identically encoded copies, only if needed.
- `reports/audit_report.md`, CSV tables, and three paired size/quality plots under `reports/figures/`.
- `data_root/splits.csv`: id, label, split, group_id.
- `artifacts/splits.sha256` and `artifacts/splits.json`: actual split fingerprint and source/config/audit provenance.
- Notebook backup under `data_root/audit/review_backup/`: review reports/plots and split records on Drive.

Current config SHA-256:

```text
e4a9a9cc79e2419805307ffa6348ea1a88f6aec1a0684867f71ff72a176c8e5c
```

No real `artifacts/splits.sha256` has been fabricated or committed. It must come
from the actual reviewed Colab run.

## Checks for the team to run in Colab

1. Clone/update `main`, run setup/tests, and use the complete Phase 1 collection on Drive. If it already exists, skip Phase 1 download cells; Phase 2 adds new config keys but can audit the preserved Phase 1 manifest and original-byte hashes.
2. Run these Phase 2 cells/commands in order:

   ```python
   subprocess.run([sys.executable, "scripts/run_audit.py"], cwd=PROJECT_DIR, check=True)
   subprocess.run([sys.executable, "scripts/make_splits.py"], cwd=PROJECT_DIR, check=True)
   ```

3. Review the generated audit tables/plots. Both final model means must be <= 0.60 (or a documented supervisor-set threshold). Record before/after scores and whether conversion happened. Passing the probe is a limited gate, not proof that every confound is removed.
4. Run `phase-2-checks`: each split must have equal class counts; split fractions must be within one percentage point of 70/15/15; crossing duplicate groups must be 0; the saved split fingerprint must match the CSV.
5. Review the exclusion reasons and mixed-label/duplicate groups. Do not silently drop or divide groups when an allocation is infeasible.
6. Run `phase-2-backup` to retain records on Drive. Commit the actual reviewed `artifacts/splits.sha256` and `artifacts/splits.json` to `main`. Preserve this hash for all later phases.
7. Check GitHub Actions for the pushed `main` commit. Local success does not establish remote CI success.

If an audit is blocked, a source file is corrupt/changed, or grouping makes the
constraints infeasible, share the exact error/report. Do not bypass the gate,
invent a quality, weaken the threshold, or regenerate frozen assignments.

## Assumptions and implementation decisions

- Supervisor confirmations remain pending. PRD defaults are implemented, not represented as approved decisions. The user authorized Phase 2 code work; no real Phase 1 results were supplied.
- **Balanced probe sample:** balance eligible classes before CV so 50% is the relevant chance level. Keeping the initial excess-real collection would give an inappropriate baseline for the 0.60 rule.
- **Eligibility timing:** exclude/log images smaller than the target before freezing. This resolves the PRD ordering conflict where later preprocessing exclusions could invalidate already-frozen splits.
- **Split allocator:** standard grouped-CV heuristics cannot guarantee exact balance in every split plus the specified tolerance. Use existing SciPy's free integer solver over class-count patterns, without a new dependency. It minimizes deviation from target proportions and reports infeasibility rather than breaking groups.
- **Quality uncertainty:** compare decoded quantization tables to standards generated by the installed Pillow JPEG encoder and report matching error. Custom tables and repeated compression cannot reveal an exact encoding history.
- **Remediation:** apply EXIF orientation before stripping metadata, then RGB JPEG/4:4:4 at the rounded median estimated real JPEG quality, identically for both classes. Originals remain available. The test of normalization determinism applies within the installed codec environment.
- **Group preservation:** retain exact/perceptual links from both original and normalized versions so conversion does not break a prior duplicate relationship. Mixed-label groups are reported, not silently discarded.

Installed signatures were inspected for the APIs used. The official
[SciPy milp reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
documents integer constraints and time limits. Pillow's
[JPEG documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
and [subsampling reference](https://pillow.readthedocs.io/en/latest/reference/JpegPresets.html)
support quantization-table access and subsampling 0 as 4:4:4.

## Open questions / anything I could not verify

- Actual metadata distributions, probe means, conversion need, eligible counts, duplicate structure, and split feasibility require real Colab data.
- No actual split fingerprint can be committed before that real run and team review. Current local workflow tests use synthetic fixtures only.
- Full Colab runtime and GitHub's Python 3.11/3.13 Linux jobs remain unverified here. Duplicate comparison uses O(n^2) time/O(n) memory for the planned small subset.
- A successful audit only checks the declared metadata features/models. It does not establish detector accuracy or generalization. No frequency features were computed and no held-out classifier test was run.

## STOP

Wait for team audit/split outputs and authorization to start Phase 3.

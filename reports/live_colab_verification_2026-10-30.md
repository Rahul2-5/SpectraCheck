# Live Colab verification — 30 October 2026

Observed directly through the user's existing Chrome/Colab session, rather than inferred from imported notebook outputs. Notebook: https://colab.research.google.com/drive/1h3v8pHEQmK6QwPHWIL9K0Max6GhnE1j7 (`colab_runner_11.ipynb`).

## Completed in Colab

- Ran the existing `scripts/run_audit.py` cell against the already acquired Drive images. No downloader was started by this verification task.
- Ran `python -m pytest -q`: **153 passed in 25.81s**, exit 0.
- Ran `python -m pip check`: **exit 1**. IPython requires missing `jedi`; `google-colab` requires pandas 2.2.3 while the pinned research environment has pandas 3.1.0. This environment check is not a pass; dependency pins were not changed.
- Ran `scripts/diagnose_metadata_shortcut.py` and `scripts/diagnose_geometry_bins.py` with the existing configuration: both exit 0, reports displayed in the notebook.

## Actual data audit: blocked

The configuration hash is `ce79d3ef5f012768bbc377db9cdd9226377a1449a13a7725d9d49189f3549dce`.

| Probe | Original mean accuracy | After identical JPEG encoding |
| --- | ---: | ---: |
| Logistic regression | 0.9998 | 0.8308 |
| Depth-3 decision tree | 1.0000 | 0.9998 |

Both classes were encoded at quality 96 with the prescribed remediation. Both final probes exceed the unchanged 0.60 limit. The original 6,500 rows yielded 6,148 eligible rows; 352 undersized images and 1,148 excess real images were excluded, leaving 2,500 images per class. The audit recorded zero duplicate groups and zero mixed-label groups.

Source manifest SHA-256: `61270a3c51cc70a8059407c1a705549b363f9c5ef11ec497685d94b853c0be61`.

Active manifest SHA-256: `db7b72907e8b0de7b5aa343a654c59aaee9b945080a4fd7c6122852abbc5cb81`.

The metadata diagnostic measured active width-only decision-tree accuracy of 0.9992 and aspect-ratio-only accuracy of 0.9886. Exact dimension matching retained zero pairs, including when portrait/landscape dimensions were treated equivalently. Approximate geometry candidates retained 0, 1, 2, 10, 50, or 50 images per class. The candidates large enough to probe still scored 0.95–0.99, so none passed. These diagnostic candidates were not adopted into the experiment.

## Drive outputs

- `/content/drive/MyDrive/aidetect/data/audit/audit_result.json`
- `/content/drive/MyDrive/aidetect/data/audit/active_manifest.csv`
- `/content/drive/MyDrive/aidetect/phase_2_live_review/audit_report.md` (copy of the runtime report)
- `/content/drive/MyDrive/aidetect/metadata_diagnostic/metadata_shortcut_diagnostic.md` and accompanying CSV/JSON files
- `/content/drive/MyDrive/aidetect/geometry_diagnostic/geometry_bin_matching_summary.md` and accompanying CSV/JSON files

The notebook contains the new verification cell and its actual test/audit/diagnostic outputs. No research split, feature extraction, training, freeze, or held-out evaluation was run by this task. A reviewed experimental decision is required to resolve the measured geometry confounding; successful diagnostic execution does not pass the audit.

## Local checks

- `npm.cmd --prefix app/frontend test`: **6 passed**.
- `npm.cmd --prefix app/frontend run build`: **passed**.
- `npm.cmd --prefix app/frontend run test:browser`: **2 passed**, isolated headless Chrome outside the sandbox.
- `.venv/Scripts/python.exe -m pip check`: **passed**, no broken requirements.
- Initial sandboxed Python/browser runs stalled and were interrupted. The browser rerun passed. The first completed local Python run had **72 passed, 81 setup errors**, all reported setup errors stemming from denied access to the default pytest temporary directory; cache writes were also denied.
- `.venv/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-live-20261030-1836-a -p no:cacheprovider`: **153 passed in 60.51s**, exit 0, outside the sandbox. The fresh project-local temporary directory resolved the fixture setup errors without changing source code or tests.

## Pending decision

STOP before split creation. The existing PRD remediation did not remove the shortcut. The team/supervisor must review a dataset or acquisition-policy change before downstream research work. Do not weaken the threshold, silently change preprocessing, or treat the geometry candidates as approved research cohorts.

## Follow-up notebook repair

The user subsequently ran the split cell and received a generic `CalledProcessError` from `scripts/make_splits.py`. Inspected that actual error in Chrome. Updated the live notebook cell to read the audit gate first, display the blocked status, threshold, final model accuracies and report locations, and invoke the unchanged split CLI only for a passing audit with the current configuration hash. Executed the repaired cell: it displayed **BLOCKED — no research splits were created**, with the expected scores. This improves notebook error reporting; it does not resolve or bypass the scientific audit.

Installed the missing Colab-only IPython dependency `jedi==0.19.2` and verified the installed version. Fresh `pip check` still exits 1, now reporting only the `google-colab` pandas 2.2.3 requirement versus the research pin 3.1.0. No research dependency was downgraded. This host-environment conflict remains unresolved; using a separate research virtual environment would be a distinct environment repair.

## Remaining-cell review

The user subsequently ran the Phase 2 checks cell and encountered its expected `assert audit_result['status'] == 'passed'` failure. This was inspected directly in Chrome. Added and ran a separate readiness/backup cell; the assertion was not removed or weakened.

Fresh results: **153 passed in 25.82s**; pip check still exits 1 for the Colab/pandas conflict. The audit configuration and active-manifest hashes match. The runtime has no `splits.csv`, split digest or split record; zero train/validation batches; no joblib models, freeze lock or evaluation marker. Test feature values were not opened.

Completed a checksum-verified backup of available audit JSON, active manifest, Markdown/CSV reports and audit figures to `/content/drive/MyDrive/aidetect/remaining_cells_review_20261030T131541Z`. This directory also contains `pytest.txt`, `pip_check.txt` and `readiness.json` recording the actual run.

| Remaining work | Current prerequisite |
| --- | --- |
| Phase 2 split checks and split backup | Passing audit and actual immutable splits |
| Phase 3/4 real-image review | Approved train split |
| Phase 5 extraction | Passing audit and immutable splits |
| Phase 6 training | Verified feature batches and review |
| Phase 7 freeze/evaluation | Reviewed training, agreed practical target, committed freeze |
| Phase 8 real-model demonstration | Reviewed serving bundle |
| Phase 9 report pack | Saved real research outputs |

No downstream command was invoked. Asked the user whether to prepare a revised data protocol for team/supervisor review or retain the current protocol and report its blocked result. No new experimental policy was assumed.

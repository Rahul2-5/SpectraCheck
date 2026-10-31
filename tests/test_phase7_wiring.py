"""Safe Phase 7 configuration, path, CLI, and serialization checks."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aidetect.io_utils import (
    project_relative_path,
    stable_json_bytes,
    stable_json_dumps,
    write_json,
)
from aidetect.phase7_config import (
    DEFAULT_PHASE7_CONFIG_NAME,
    EXPECTED_PHASE6_CONFIG_SHA256,
    load_phase7_config,
    phase7_artifact_paths,
    phase7_paths,
    validate_phase7_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_active_phase7_config_and_nested_roots_are_selected():
    path, cfg = load_phase7_config(ROOT)
    assert path.name == DEFAULT_PHASE7_CONFIG_NAME
    paths = phase7_paths(cfg, ROOT)
    assert paths["artifacts_root"] == (ROOT / "artifacts/r2_fast_rf_t4").resolve()
    assert paths["reports_root"] == (ROOT / "reports/r2_fast_rf_t4").resolve()
    assert cfg["paths"]["artifacts_root"] == "artifacts/r2_fast_rf_t4"
    assert cfg["paths"]["reports_root"] == "reports/r2_fast_rf_t4"


def test_active_config_hash_is_the_reviewed_phase6_identity():
    _, cfg = load_phase7_config(ROOT)
    from aidetect.io_utils import config_hash

    assert config_hash(cfg) == EXPECTED_PHASE6_CONFIG_SHA256


def test_explicit_legacy_config_remains_supported():
    path, cfg = load_phase7_config(ROOT, ROOT / "config.yaml")
    assert path.name == "config.yaml"
    assert cfg["paths"]["reports_root"] == "reports"


def test_missing_default_config_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 7 configuration is missing"):
        load_phase7_config(tmp_path)


def test_nested_artifact_paths_include_selected_model_and_feature_records():
    cfg = {"paths": {"artifacts_root": "artifacts/run", "reports_root": "reports/run"}}
    paths = phase7_artifact_paths(cfg, ROOT)
    assert paths["feature_choice"] == (ROOT / "artifacts/run/feature_choice.json").resolve()
    assert paths["model:frequency_svm"] == (
        ROOT / "artifacts/run/models/frequency_svm.joblib"
    ).resolve()
    assert paths["practical_target"] == (ROOT / "reports/run/practical_target.md").resolve()
    assert project_relative_path(ROOT, paths["results_json"]) == "reports/run/results.json"


def test_missing_phase7_artifacts_are_reported_without_opening_features(tmp_path):
    cfg = {"paths": {"artifacts_root": "artifacts/run", "reports_root": "reports/run"}}
    with pytest.raises(FileNotFoundError, match="practical_target") as error:
        validate_phase7_artifacts(cfg, tmp_path)
    assert "frequency_svm.joblib" in str(error.value)
    assert "pixel_xgb.joblib" in str(error.value)


def test_json_serialization_and_file_hash_are_lf_stable(tmp_path):
    first = {"z": [3, 2, 1], "nested": {"b": "é", "a": True}}
    reordered = {"nested": {"a": True, "b": "é"}, "z": [3, 2, 1]}
    expected = stable_json_bytes(first)
    assert expected == stable_json_bytes(reordered)
    assert b"\r" not in expected
    assert stable_json_dumps(first).endswith("\n")

    output = tmp_path / "nested" / "record.json"
    write_json(output, reordered)
    assert output.read_bytes() == expected
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("script", ["scripts/freeze.py", "scripts/evaluate_test.py"])
def test_phase7_cli_help_exposes_active_config(script):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert DEFAULT_PHASE7_CONFIG_NAME in result.stdout
    assert "config.yaml" not in result.stdout


def test_phase7_notebook_cells_are_explicit_and_parseable():
    notebook = json.loads((ROOT / "notebooks/colab_runner.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    code_cells = {
        cell_id: "".join(cells[cell_id]["source"])
        for cell_id in ("phase-7-preflight", "phase-7-freeze", "phase-7-evaluate-once")
    }
    for source in code_cells.values():
        ast.parse(source)
    assert "config_r2_fast_rf_t4.yaml" in code_cells["phase-7-preflight"]
    assert "config_r2_fast_rf_t4.yaml" in code_cells["phase-7-evaluate-once"]
    assert all("config.yaml" not in source for source in code_cells.values())
    assert "--config" in code_cells["phase-7-freeze"]
    assert "--config" in code_cells["phase-7-evaluate-once"]
    assert "artifacts/r2_fast_rf_t4/FREEZE.lock" in "".join(
        cells["phase-7-commit-lock"]["source"]
    )
    assert "reports/r2_fast_rf_t4/results.json" in "".join(
        cells["stop-phase-7"]["source"]
    )
    assert all("phase_6_models" not in source for source in code_cells.values())

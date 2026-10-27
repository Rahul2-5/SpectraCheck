from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from sklearn.dummy import DummyClassifier

from aidetect.evaluate import evaluate_test_once
from aidetect.freeze import freeze_phase7, validate_practical_target, verify_freeze_lock
from aidetect.io_utils import config_hash, sha256_of_file


_MODEL_KEYS = tuple(
    f"{feature_set}_{model}"
    for feature_set in ("frequency", "pixel")
    for model in ("svm", "rf", "xgb")
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _workspace(tmp_path: Path) -> tuple[Path, dict, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "src/aidetect").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "artifacts/models").mkdir(parents=True)
    (repo / "reports").mkdir()
    drive = tmp_path / "drive"
    data_root = drive / "data"
    features_root = drive / "features"
    test_dir = features_root / "test_sealed"
    data_root.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (repo / ".gitignore").write_text("artifacts/models/\n", encoding="utf-8")
    (repo / "src/aidetect/fixture.py").write_text("FIXTURE = True\n", encoding="utf-8")
    (repo / "scripts/evaluate_test.py").write_text("# fixture evaluator source\n", encoding="utf-8")
    (repo / "scripts/freeze.py").write_text("# fixture freeze source\n", encoding="utf-8")

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
    cfg.update({"image_size": 16, "rapsd_rings": 4, "pixel_baseline_size": 4})
    cfg["extraction"]["peak_k_mad"] = [3]
    cfg["training"]["r0"] = {"min": 8, "max": 8, "step": 8}
    cfg["paths"] = {
        "data_root": str(data_root),
        "features_root": str(features_root),
        "reports_root": "reports",
        "artifacts_root": "artifacts",
    }
    config_path = repo / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    config_digest = config_hash(cfg)

    split_rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            identifier = f"{split}-{label}"
            split_rows.append(
                {
                    "id": identifier,
                    "label": label,
                    "group_id": f"group-{identifier}",
                    "split": split,
                    "rel_path": f"{split}/{identifier}.png",
                }
            )
    split_path = data_root / "splits.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False, lineterminator="\n")
    splits_digest = sha256_of_file(split_path)
    (repo / "artifacts/splits.sha256").write_text(splits_digest + "\n", encoding="ascii")

    choice = {
        "r0": 8,
        "k_mad": 3,
        "config_sha256": config_digest,
        "splits_sha256": splits_digest,
    }
    primary = {
        "feature_set": "frequency",
        "model": "svm",
        "artifact_key": "frequency_svm",
        "selection_metric": "validation_accuracy",
        "config_sha256": config_digest,
        "splits_sha256": splits_digest,
    }
    results = {
        "config_sha256": config_digest,
        "splits_sha256": splits_digest,
        "test_features_read": False,
        "primary_model": primary,
        "models": [{"key": key} for key in _MODEL_KEYS],
    }
    for name, payload in (
        ("feature_choice.json", choice),
        ("primary_model.json", primary),
        ("versions.json", {"python": "test"}),
        ("training_results.json", results),
    ):
        (repo / "artifacts" / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    target = (
        "# Practical-use target for Phase 7\n\n"
        "Status: Agreed\n\n"
        "Minimum validation accuracy: 0.75\n"
        "Rationale: Synthetic fixture used only to exercise the Phase 7 software.\n"
        "Agreed by: Fixture team member A, Fixture team member B\n"
        "Supervisor: Fixture supervisor\n"
        "Decision date: 2026-10-27\n"
    )
    (repo / "reports/practical_target.md").write_text(target, encoding="utf-8")

    training_model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    for key in _MODEL_KEYS:
        joblib.dump(training_model, repo / "artifacts/models" / f"{key}.joblib")

    feature_rows = []
    for label in (0, 1):
        identifier = f"test-{label}"
        row = {
            "id": identifier,
            "label": label,
            "group_id": f"group-{identifier}",
            "split": "test",
            "rel_path": f"test/{identifier}.png",
            "image_side": 16,
            "entropy": 0.25 + 0.5 * label,
        }
        for index in range(13):
            row[f"cumulative_energy_{index:03d}"] = float(index * (label + 1))
        for index in range(4):
            row[f"rapsd_ring_{index:02d}"] = float(label + index / 10)
        row["peak_count_k3"] = label
        row["peak_strength_k3"] = float(label)
        for index in range(16):
            row[f"pixel_{index:04d}"] = float(label)
        feature_rows.append(row)
    feature_frame = pd.DataFrame(feature_rows)
    table = pa.Table.from_pandas(feature_frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"aidetect_schema_version": b"1",
            b"aidetect_config_hash": config_digest.encode("ascii"),
            b"aidetect_splits_sha256": splits_digest.encode("ascii"),
            b"aidetect_split": b"test",
            b"aidetect_batch_index": b"0",
            b"aidetect_batch_ids_sha256": hashlib.sha256(
                "\n".join(feature_frame["id"].astype(str)).encode("utf-8")
            ).hexdigest().encode("ascii"),
            b"aidetect_image_side": b"16",
            b"aidetect_rapsd_rings": b"4",
            b"aidetect_pixel_baseline_size": b"4",
            b"aidetect_peak_k_mad": b"[3]",
        }
    )
    table = table.replace_schema_metadata(metadata)
    test_batch = test_dir / "batch_00000.parquet"
    pq.write_table(table, test_batch)

    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Phase 7 fixture")
    _git(repo, "config", "user.email", "phase7-fixture@example.invalid")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "Create synthetic Phase 7 workspace")
    return repo, cfg, test_batch


def test_practical_target_rejects_template_and_accepts_complete_agreement(tmp_path):
    template = tmp_path / "target.md"
    template.write_text(
        "Status: Pending\n"
        "Minimum validation accuracy: [team value]\n"
        "Rationale: [why]\nAgreed by: [team]\nSupervisor: [name]\n"
        "Decision date: YYYY-MM-DD\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum validation accuracy"):
        validate_practical_target(template)

    repo, _, _ = _workspace(tmp_path / "complete")
    target = validate_practical_target(repo / "reports/practical_target.md")
    assert target["minimum_validation_accuracy"] == 0.75
    assert target["decision_date"] == "2026-10-27"


def test_freeze_records_integrity_without_reading_parquet_values(tmp_path, monkeypatch):
    repo, cfg, test_batch = _workspace(tmp_path)
    import aidetect.freeze as freeze_module

    def unexpected(*args, **kwargs):
        raise AssertionError("Freeze must not parse or read test feature rows")

    monkeypatch.setattr(freeze_module.pd, "read_parquet", unexpected)
    monkeypatch.setattr(freeze_module.pq, "read_table", unexpected)
    lock = freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)

    assert lock["test_feature_values_read_during_freeze"] is False
    assert lock["sealed_test_batches"][0]["sha256"] == sha256_of_file(test_batch)
    assert lock["expected_test_rows"] == 2
    assert (repo / "artifacts/FREEZE.lock").is_file()
    with pytest.raises(FileExistsError, match="single-use"):
        freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)


def test_freeze_refuses_dirty_tree_and_pending_target(tmp_path):
    repo, cfg, _ = _workspace(tmp_path)
    (repo / "uncommitted.txt").write_text("local change\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git tree"):
        freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)
    (repo / "uncommitted.txt").unlink()
    (repo / "reports/practical_target.md").write_text(
        "Status: Pending\n"
        "Minimum validation accuracy: 0.75\n"
        "Rationale: A complete synthetic fixture reason.\n"
        "Agreed by: Fixture team member A.\n"
        "Supervisor: Fixture supervisor.\n"
        "Decision date: 2026-10-27\n",
        encoding="utf-8",
    )
    _git(repo, "add", "reports/practical_target.md")
    _git(repo, "commit", "-m", "Set pending target template")
    with pytest.raises(ValueError, match="Status must be exactly 'Agreed'"):
        freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)


def test_freeze_will_not_reset_a_prior_evaluation_marker(tmp_path):
    repo, cfg, _ = _workspace(tmp_path)
    marker = repo / "artifacts/EVALUATION_STARTED.json"
    marker.write_text('{"state":"started"}\n', encoding="utf-8")
    _git(repo, "add", "artifacts/EVALUATION_STARTED.json")
    _git(repo, "commit", "-m", "Record previous evaluation attempt")

    with pytest.raises(FileExistsError, match="do not reset the one-time guard"):
        freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)


def test_verify_freeze_rejects_changed_sealed_batch_hash(tmp_path):
    repo, cfg, test_batch = _workspace(tmp_path)
    freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)
    _git(repo, "add", "artifacts/FREEZE.lock")
    _git(repo, "commit", "-m", "Commit synthetic freeze for integrity test")
    table = pq.read_table(test_batch)
    pq.write_table(table, test_batch, compression="gzip")

    with pytest.raises(ValueError, match="Sealed test feature batch list or file hash"):
        verify_freeze_lock(
            cfg,
            config_file=repo / "config.yaml",
            repo_root=repo,
            require_committed_lock=False,
        )


def test_phase7_eval_reads_synthetic_test_once_and_writes_all_outputs(tmp_path):
    repo, cfg, _ = _workspace(tmp_path)
    freeze_phase7(cfg, config_file=repo / "config.yaml", repo_root=repo)
    _git(repo, "add", "artifacts/FREEZE.lock")
    _git(repo, "commit", "-m", "Commit synthetic Phase 7 freeze")

    result = evaluate_test_once(cfg, config_file=repo / "config.yaml", repo_root=repo)

    assert result["test_features_read"] is True
    assert result["evaluation_count"] == 1
    assert result["test_rows"] == 2
    assert set(result["models"]) == set(_MODEL_KEYS)
    assert result["models"]["frequency_svm"]["metrics"]["confusion_matrix_labels_0_real_1_ai"] == [
        [1, 0],
        [1, 0],
    ]
    assert result["hypothesis_tests"]["H1b"]["p_value"] == 1.0
    assert (repo / "reports/results.json").is_file()
    markdown = (repo / "reports/results.md").read_text(encoding="utf-8")
    assert "H1: primary model vs chance" in markdown
    assert "H1a: paired frequency vs pixel comparisons" in markdown
    assert "H1b: frequency-model comparison" in markdown
    assert "Holm-adjusted p" in markdown
    predictions = pd.read_csv(repo / "reports/test_predictions.csv")
    assert len(predictions) == 12
    assert set(predictions.columns) == {"id", "true", "pred", "prob", "feature_set", "model"}
    marker = json.loads((repo / "artifacts/EVALUATION_STARTED.json").read_text(encoding="utf-8"))
    assert marker["state"] == "completed"
    assert marker["test_features_read"] is True
    with pytest.raises(RuntimeError, match="clean Git tree"):
        evaluate_test_once(cfg, config_file=repo / "config.yaml", repo_root=repo)

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import aidetect.train as train_module
from aidetect.io_utils import config_hash, sha256_of_file
from aidetect.train import (
    assert_scaler_is_fold_local,
    choose_frequency_features,
    load_training_features,
    make_frequency_matrix,
    make_pipeline,
    make_pixel_matrix,
    make_stratified_folds,
    run_phase6,
    search_space,
)


ROOT = Path(__file__).parents[1]


def _cfg(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = deepcopy(cfg)
    cfg.update({"image_size": 16, "rapsd_rings": 4, "pixel_baseline_size": 8})
    cfg["extraction"]["peak_k_mad"] = [3, 4]
    cfg["cv_folds"] = 3
    cfg["search_iters"] = 2
    cfg["training"]["r0"] = {"min": 8, "max": 16, "step": 8}
    cfg["training"]["search_space"] = {
        "svm": {"C_loguniform": [0.1, 10.0], "gamma_loguniform": [0.001, 1.0]},
        "rf": {
            "n_estimators": [8, 12],
            "max_depth": [None, 4],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.5],
        },
        "xgb": {
            "n_estimators": [8, 12],
            "max_depth": [2, 4],
            "learning_rate_uniform": [0.05, 0.2],
            "subsample_uniform": [0.8, 1.0],
            "colsample_bytree_uniform": [0.8, 1.0],
        },
        "xgb_fixed": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
        },
    }
    cfg["paths"] = {
        "data_root": str(tmp_path / "data"),
        "features_root": str(tmp_path / "features"),
        "reports_root": str(tmp_path / "reports"),
        "artifacts_root": str(tmp_path / "artifacts"),
    }
    return cfg


def _summary_frame(cfg: dict, split: str, rows_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        for index in range(rows_per_class):
            identifier = f"{split}-{label}-{index:03d}"
            row = {
                "id": identifier,
                "label": label,
                "group_id": f"group-{identifier}",
                "split": split,
                "rel_path": f"{split}/{identifier}.png",
                "image_side": cfg["image_size"],
                "entropy": float(np.clip(0.08 + 0.82 * label + rng.normal(0, 0.015), 0, 1)),
            }
            energy_count = math.ceil(math.sqrt(2.0) * (cfg["image_size"] // 2)) + 1
            for radial_index in range(energy_count):
                row[f"cumulative_energy_{radial_index:03d}"] = float(
                    radial_index * (1.0 + 0.35 * label)
                )
            for ring_index in range(cfg["rapsd_rings"]):
                row[f"rapsd_ring_{ring_index:02d}"] = float(
                    0.5 + label * 2.5 + rng.normal(0, 0.04)
                )
            for threshold in cfg["extraction"]["peak_k_mad"]:
                row[f"peak_count_k{threshold:g}"] = int(label * 3)
                row[f"peak_strength_k{threshold:g}"] = float(
                    label * 3.0 + rng.normal(0, 0.03)
                )
            pixels = np.clip(
                label + rng.normal(0, 0.03, cfg["pixel_baseline_size"] ** 2), 0.0, 1.0
            )
            for pixel_index, value in enumerate(pixels):
                row[f"pixel_{pixel_index:04d}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_training_inputs(
    cfg: dict,
    *,
    rows_per_class: int = 40,
    feature_config_hash_override: str | None = None,
) -> None:
    data_root = Path(cfg["paths"]["data_root"])
    features_root = Path(cfg["paths"]["features_root"])
    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    data_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    train = _summary_frame(cfg, "train", rows_per_class, 10)
    val = _summary_frame(cfg, "val", max(rows_per_class // 2, 10), 20)
    split_frame = pd.concat(
        [train[["id", "label", "split", "group_id"]], val[["id", "label", "split", "group_id"]]],
        ignore_index=True,
    )
    split_path = data_root / "splits.csv"
    split_frame.to_csv(split_path, index=False, lineterminator="\n")
    split_digest = sha256_of_file(split_path)
    (artifacts_root / "splits.sha256").write_text(split_digest + "\n", encoding="ascii")

    metadata_base = {
        b"aidetect_schema_version": b"1",
        b"aidetect_config_hash": (feature_config_hash_override or config_hash(cfg)).encode("ascii"),
        b"aidetect_splits_sha256": split_digest.encode("ascii"),
        b"aidetect_image_side": str(cfg["image_size"]).encode("ascii"),
        b"aidetect_rapsd_rings": str(cfg["rapsd_rings"]).encode("ascii"),
        b"aidetect_pixel_baseline_size": str(cfg["pixel_baseline_size"]).encode("ascii"),
        b"aidetect_peak_k_mad": json.dumps(cfg["extraction"]["peak_k_mad"]).encode("ascii"),
    }
    for split, frame in (("train", train), ("val", val)):
        directory = features_root / split
        directory.mkdir(parents=True)
        ids = frame["id"].astype(str).tolist()
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(metadata_base)
        metadata[b"aidetect_split"] = split.encode("ascii")
        metadata[b"aidetect_batch_index"] = b"0"
        metadata[b"aidetect_batch_ids_sha256"] = hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest().encode("ascii")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, directory / "batch_00000.parquet")


def test_frequency_matrix_reconstructs_selected_68_style_vector(tmp_path):
    cfg = _cfg(tmp_path)
    frame = _summary_frame(cfg, "train", 2, 42)

    matrix, names = make_frequency_matrix(frame, cfg, r0=8, k_mad=4)

    assert matrix.shape == (4, 8)
    assert names == [
        "energy_ratio",
        "spectral_entropy",
        "rapsd_ring_00",
        "rapsd_ring_01",
        "rapsd_ring_02",
        "rapsd_ring_03",
        "peak_count",
        "peak_strength",
    ]
    assert np.isfinite(matrix).all()
    assert matrix[0, 1] < matrix[-1, 1]


def test_feature_choice_ties_use_smallest_r0_then_smallest_k_mad(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    train = _summary_frame(cfg, "train", 2, 10)
    val = _summary_frame(cfg, "val", 2, 20)
    cfg["training"]["r0"] = {"min": 16, "max": 8, "step": 8}
    # The malformed order must be rejected before any model fit.
    with pytest.raises(ValueError, match="inclusive regular grid"):
        choose_frequency_features(train, val, cfg)

    cfg["training"]["r0"] = {"min": 8, "max": 16, "step": 8}

    class AlwaysZero:
        def __init__(self, **kwargs):
            pass

        def fit(self, values, labels):
            return self

        def predict(self, values):
            return np.zeros(len(values), dtype=np.int64)

    monkeypatch.setattr(train_module, "RandomForestClassifier", AlwaysZero)
    cfg["extraction"]["peak_k_mad"] = [4, 3]
    choice, _, _, _ = choose_frequency_features(train, val, cfg)
    assert choice["r0"] == 8
    assert choice["k_mad"] == 3


def test_search_spaces_match_configured_prd_ranges(tmp_path):
    cfg = _cfg(tmp_path)

    svm = search_space("svm", cfg)
    rf = search_space("rf", cfg)
    xgb = search_space("xgb", cfg)

    assert svm["classifier__C"].args == (0.1, 10.0)
    assert svm["classifier__gamma"].args == (0.001, 1.0)
    assert rf["classifier__n_estimators"].args == (8, 13)
    assert rf["classifier__min_samples_leaf"].args == (1, 3)
    assert rf["classifier__max_depth"] == [None, 4]
    assert xgb["classifier__n_estimators"].args == (8, 13)
    assert xgb["classifier__max_depth"].args == (2, 5)
    assert xgb["classifier__learning_rate"].args == (0.05, 0.15000000000000002)
    assert make_pipeline("xgb", cfg).get_params()["classifier__eval_metric"] == "logloss"


def test_shared_stratified_folds_are_seeded_and_scaler_is_fold_local(tmp_path):
    cfg = _cfg(tmp_path)
    labels = np.asarray([0] * 15 + [1] * 15)
    values = np.column_stack([np.arange(30, dtype=np.float64), labels.astype(float)])
    folds = make_stratified_folds(labels, folds=3, seed=42)
    repeated = make_stratified_folds(labels, folds=3, seed=42)

    assert len(folds) == 3
    assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(folds, repeated))
    assert_scaler_is_fold_local(values, labels, make_pipeline("rf", cfg), folds)
    with pytest.raises(ValueError, match="ordered 'scale'"):
        from sklearn.pipeline import Pipeline

        assert_scaler_is_fold_local(values, labels, Pipeline([("classifier", make_pipeline("rf", cfg))]), folds)


def test_phase6_runs_six_models_on_train_val_only_and_writes_provenance(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _write_training_inputs(cfg)
    visited = []
    original = train_module.feature_output_dir

    def no_sealed_directory(features_root, split):
        visited.append(split)
        if split == "test":
            raise AssertionError("Phase 6 must never access test_sealed")
        return original(features_root, split)

    monkeypatch.setattr(train_module, "feature_output_dir", no_sealed_directory)
    result = run_phase6(cfg, config_file=tmp_path / "config.yaml")

    assert set(visited) == {"train", "val"}
    assert result["test_features_read"] is False
    assert len(result["models"]) == 6
    assert all(item["validation_accuracy"] > 0.9 for item in result["models"])
    assert result["feature_choice"]["r0"] in (8, 16)
    assert result["feature_choice"]["k_mad"] in (3, 4)
    frequency_results = [item for item in result["models"] if item["feature_set"] == "frequency"]
    expected_primary = max(frequency_results, key=lambda item: item["validation_accuracy"])
    assert result["primary_model"]["artifact_key"] == expected_primary["key"]

    artifacts = Path(cfg["paths"]["artifacts_root"])
    assert json.loads((artifacts / "feature_choice.json").read_text())["r0"] == result["feature_choice"]["r0"]
    assert json.loads((artifacts / "primary_model.json").read_text())["artifact_key"] == result["primary_model"]["artifact_key"]
    versions = json.loads((artifacts / "versions.json").read_text())
    assert {"python", "scikit_learn", "xgboost", "numpy"} <= set(versions)
    assert (Path(cfg["paths"]["reports_root"]) / "phase_6_results.md").is_file()
    assert (Path(cfg["paths"]["reports_root"]) / "figures" / "phase_6_feature_importance_frequency_rf.png").is_file()
    for model in result["models"]:
        fitted = joblib.load(artifacts / "models" / f"{model['key']}.joblib")
        assert np.array_equal(fitted.predict(np.zeros((3, model["feature_count"]))), fitted.predict(np.zeros((3, model["feature_count"]))))


def test_load_training_features_checks_split_and_feature_hashes_and_stops_after_freeze(tmp_path):
    cfg = _cfg(tmp_path)
    _write_training_inputs(cfg)
    train, val, provenance = load_training_features(cfg, config_file="synthetic.yaml")
    assert len(train) == 80
    assert len(val) == 40
    assert provenance["splits_sha256"] == sha256_of_file(Path(cfg["paths"]["data_root"]) / "splits.csv")

    split_file = Path(cfg["paths"]["data_root"]) / "splits.csv"
    split_file.write_text(split_file.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_training_features(cfg, config_file="synthetic.yaml")

    split_file.write_text("synthetic split\n", encoding="utf-8")
    digest = sha256_of_file(split_file)
    (Path(cfg["paths"]["artifacts_root"]) / "splits.sha256").write_text(digest + "\n", encoding="ascii")
    (Path(cfg["paths"]["artifacts_root"]) / "FREEZE.lock").write_text("locked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="before the Phase 7"):
        load_training_features(cfg, config_file="synthetic.yaml")


def test_legacy_phase5_hash_is_accepted_only_with_the_frozen_feature_contract(tmp_path):
    cfg = deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    cfg["paths"] = {
        "data_root": str(tmp_path / "data"),
        "features_root": str(tmp_path / "features"),
        "reports_root": str(tmp_path / "reports"),
        "artifacts_root": str(tmp_path / "artifacts"),
    }
    legacy_hash = cfg["training"]["legacy_phase5_config_hashes"][0]
    _write_training_inputs(cfg, rows_per_class=2, feature_config_hash_override=legacy_hash)

    train, val, provenance = load_training_features(cfg, config_file="synthetic.yaml")
    assert len(train) == 4
    assert len(val) == 20
    assert provenance["feature_config_sha256"] == legacy_hash

    cfg["frequency"]["hann_window"] = False
    with pytest.raises(ValueError, match="feature contract remains unchanged"):
        load_training_features(cfg, config_file="synthetic.yaml")


def test_all_six_seeded_pipeline_fits_are_deterministic_and_separable(tmp_path):
    cfg = _cfg(tmp_path)
    train = _summary_frame(cfg, "train", 20, 35)
    labels = train["label"].to_numpy(dtype=np.int64)
    matrices = {
        "frequency": make_frequency_matrix(train, cfg, r0=8, k_mad=3)[0],
        "pixel": make_pixel_matrix(train, cfg)[0],
    }
    for values in matrices.values():
        for model_name in ("svm", "rf", "xgb"):
            first = make_pipeline(model_name, cfg).fit(values, labels)
            second = make_pipeline(model_name, cfg).fit(values, labels)
            first_predictions = first.predict(values)
            assert np.array_equal(first_predictions, second.predict(values))
            assert float(np.mean(first_predictions == labels)) > 0.9


def test_training_loader_rejects_wrong_config_hash_without_reading_test(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _write_training_inputs(cfg)
    batch_path = Path(cfg["paths"]["features_root"]) / "train" / "batch_00000.parquet"
    table = pq.read_table(batch_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"aidetect_config_hash"] = b"0" * 64
    pq.write_table(table.replace_schema_metadata(metadata), batch_path)
    monkeypatch.setattr(
        train_module,
        "feature_output_dir",
        lambda root, split: (_ for _ in ()).throw(AssertionError("test dir touched"))
        if split == "test"
        else Path(root) / split,
    )
    with pytest.raises(ValueError, match="aidetect_config_hash"):
        load_training_features(cfg, config_file="synthetic.yaml")

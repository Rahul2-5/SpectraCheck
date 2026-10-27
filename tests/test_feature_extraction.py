from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from aidetect.feature_extraction import (
    energy_ratio_from_cumulative,
    extract_split_features,
    read_feature_batches,
    summarize_preprocessed_image,
)
from aidetect.features import energy_ratio, power_spectrum
from aidetect.io_utils import sha256_of_file


def _cfg(**overrides):
    cfg = {
        "image_size": 32,
        "rapsd_rings": 8,
        "pixel_baseline_size": 8,
        "batch_size": 1,
        "frequency": {
            "hann_window": True,
            "epsilon": 1.0e-12,
            "mad_scale": 1.4826,
        },
        "extraction": {"peak_k_mad": [3, 4, 5, 6, 8]},
    }
    cfg.update(overrides)
    return cfg


def _rows(split="train"):
    return pd.DataFrame(
        [
            {"id": "b", "label": 1, "group_id": "gb", "split": split, "rel_path": "b.png"},
            {"id": "a", "label": 0, "group_id": "ga", "split": split, "rel_path": "a.png"},
        ]
    )


def _write_images(root: Path):
    root.mkdir(parents=True)
    Image.fromarray(np.full((48, 64), 96, dtype=np.uint8), mode="L").save(root / "a.png")
    pattern = np.indices((48, 64)).sum(axis=0) % 2
    Image.fromarray((pattern * 255).astype(np.uint8), mode="L").save(root / "b.png")


def test_lookup_energy_ratio_matches_direct_masks_for_integer_cutoffs():
    rng = np.random.default_rng(42)
    image = rng.random((32, 32))
    cfg = _cfg()
    summary = summarize_preprocessed_image(image, cfg)
    power = power_spectrum(image, window=True)

    for r0 in (1, 4, 8, 12):
        expected = energy_ratio(power, r0, epsilon=1.0e-12)
        actual = energy_ratio_from_cumulative(summary["cumulative_energy"], r0)
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_summary_contains_all_recomputable_phase5_components():
    image = np.full((32, 32), 0.5, dtype=np.float64)

    summary = summarize_preprocessed_image(image, _cfg())

    assert len(summary["rapsd"]) == 8
    assert len(summary["pixel"]) == 64
    assert set(summary["peak_counts"]) == {3, 4, 5, 6, 8}
    assert np.isfinite(summary["cumulative_energy"]).all()
    assert np.isfinite(summary["rapsd"]).all()
    assert np.isfinite(summary["pixel"]).all()


def test_feature_batches_resume_without_reprocessing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _write_images(data_root)
    cfg = _cfg()
    rows = _rows()

    first = extract_split_features(
        rows,
        data_root=data_root,
        features_root=tmp_path / "features",
        cfg=cfg,
        split="train",
        config_digest="config-digest",
        splits_digest="split-digest",
    )
    assert first["written_batches"] == 2
    assert first["skipped_batches"] == 0

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resume should not preprocess an existing batch")

    monkeypatch.setattr("aidetect.feature_extraction.preprocess_image", fail_if_called)
    second = extract_split_features(
        rows,
        data_root=data_root,
        features_root=tmp_path / "features",
        cfg=cfg,
        split="train",
        config_digest="config-digest",
        splits_digest="split-digest",
    )
    assert second["written_batches"] == 0
    assert second["skipped_batches"] == 2


def test_test_sealed_reader_requires_freeze_lock(tmp_path):
    data_root = tmp_path / "data"
    _write_images(data_root)
    features_root = tmp_path / "features"
    artifacts_root = tmp_path / "artifacts"
    cfg = _cfg()
    rows = _rows(split="test")

    result = extract_split_features(
        rows,
        data_root=data_root,
        features_root=features_root,
        cfg=cfg,
        split="test",
        config_digest="config-digest",
        splits_digest="split-digest",
    )
    assert result["output_dir"].endswith("test_sealed")
    with pytest.raises(PermissionError, match="FREEZE.lock"):
        read_feature_batches(features_root, "test", artifacts_root=artifacts_root)

    artifacts_root.mkdir()
    (artifacts_root / "FREEZE.lock").write_text("test lock\n", encoding="utf-8")
    loaded = read_feature_batches(features_root, "test", artifacts_root=artifacts_root)
    assert len(loaded) == 2
    assert len([column for column in loaded if column.startswith("pixel_")]) == 64


def test_extract_features_script_joins_manifest_and_split_hash(tmp_path):
    data_root = tmp_path / "data"
    _write_images(data_root)
    (data_root / "audit").mkdir()
    split_rows = _rows()
    split_path = data_root / "splits.csv"
    split_rows[["id", "label", "split", "group_id"]].to_csv(
        split_path, index=False, lineterminator="\n"
    )
    split_rows[["id", "rel_path"]].to_csv(
        data_root / "audit" / "active_manifest.csv", index=False, lineterminator="\n"
    )
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / "splits.sha256").write_text(
        sha256_of_file(split_path) + "\n", encoding="utf-8"
    )
    features_root = tmp_path / "features"
    cfg = _cfg()
    cfg["paths"] = {
        "data_root": str(data_root),
        "features_root": str(features_root),
        "artifacts_root": str(artifacts_root),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_features.py",
            "--config",
            str(config_path),
            "--split",
            "train",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output["rows"] == 2
    assert output["written_batches"] == 2
    assert output["splits_sha256"] == sha256_of_file(split_path)

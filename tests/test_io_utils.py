"""Known-answer and failure-path checks without data downloads."""

import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import yaml

from aidetect.io_utils import config_hash, load_config, set_seeds, sha256_of_file


def test_project_config_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg["positive_label"] == 1
    assert set(cfg["split"]) == {"train", "val", "test"}
    assert sum(cfg["split"].values()) == pytest.approx(1)
    assert cfg["dataset"]["generator_name"] == "SD15"
    assert cfg["paths"]["data_root"] != cfg["paths"]["features_root"]


@pytest.mark.parametrize("document", ["", "- item\n", "42\n"])
def test_config_rejects_non_mapping(tmp_path, document):
    path = tmp_path / "config.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_missing_config_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_malformed_yaml_is_rejected(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("split: [unterminated", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_config_hash_stable_across_nested_key_order():
    first = {"seed": 42, "paths": {"data": "photos", "features": "vectors"}}
    reordered = {"paths": {"features": "vectors", "data": "photos"}, "seed": 42}
    assert config_hash(first) == config_hash(first) == config_hash(reordered)
    assert len(config_hash(first)) == 64


def test_config_hash_changes_when_settings_change():
    assert config_hash({"seed": 42}) != config_hash({"seed": 43})
    assert config_hash({"preprocess_version": 1}) != config_hash(
        {"preprocess_version": 2}
    )


def test_config_hash_matches_sorted_yaml_contract():
    cfg = {"label": "real", "seed": 42}
    expected = hashlib.sha256(
        yaml.safe_dump(
            cfg, sort_keys=True, allow_unicode=True, default_flow_style=False
        ).encode("utf-8")
    ).hexdigest()
    assert config_hash(cfg) == expected


def test_seed_repeats_python_and_numpy_sequences():
    set_seeds(42)
    first_python = [random.random() for _ in range(5)]
    first_numpy = np.random.random(5)
    set_seeds(42)
    assert first_python == [random.random() for _ in range(5)]
    np.testing.assert_array_equal(first_numpy, np.random.random(5))
    set_seeds(43)
    assert first_python != [random.random() for _ in range(5)]
    assert not np.array_equal(first_numpy, np.random.random(5))


@pytest.mark.parametrize("payload", [b"", b"abc", bytes(range(256)) * 8193])
def test_file_hash_matches_known_bytes(tmp_path, payload):
    path = tmp_path / "sample.bin"
    path.write_bytes(payload)
    assert sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_file_hash_detects_modified_content(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"before")
    before = sha256_of_file(path)
    path.write_bytes(b"after")
    assert before != sha256_of_file(path)


def test_missing_file_hash_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_of_file(tmp_path / "missing.bin")

"""Synthetic checks for a proposed RF-only amendment; no real run is changed."""
from copy import deepcopy
import json

import joblib
import numpy as np
import pytest
import yaml

from aidetect.io_utils import config_hash
from aidetect.train import run_phase6
from scripts.migrate_fast_rf_checkpoints import validate_amendment, migrate
from test_train import _cfg, _write_training_inputs


def amended(original, tmp_path, source_contract="synthetic"):
    revised = deepcopy(original)
    revised["paths"]["artifacts_root"] = str(tmp_path / "revised_artifacts")
    revised["paths"]["reports_root"] = str(tmp_path / "revised_reports")
    revised["training"]["profile"] = "synthetic-rf-amendment"
    revised["training"]["rf_search_amendment"] = "user-approved-2026-10-31"
    revised["training"]["reuse_source_contract_sha256"] = source_contract
    revised["training"]["search_space"]["rf"]["n_estimators"] = [100, 300]
    revised["training"]["search_space"]["rf"]["max_features"] = ["sqrt"]
    return revised


@pytest.mark.parametrize("change", ["seed", "cv_folds", "preprocess", "svm_range", "data_root", "rf_depth"])
def test_amendment_rejects_changes_that_invalidate_reuse(tmp_path, change):
    original = _cfg(tmp_path)
    revised = amended(original, tmp_path)
    validate_amendment(original, revised)
    if change in {"seed", "cv_folds"}:
        revised[change] += 1
    elif change == "preprocess":
        revised["preprocess"]["normalization_divisor"] = 256.0
    elif change == "svm_range":
        revised["training"]["search_space"]["svm"]["C_loguniform"] = [0.1, 1.0]
    elif change == "data_root":
        revised["paths"]["data_root"] += "other"
    else:
        revised["training"]["search_space"]["rf"]["max_depth"] = [2]
    with pytest.raises(ValueError, match="explicitly reviewed"):
        validate_amendment(original, revised)


def test_migration_reuses_only_identical_non_rf_models_and_refits_both_forests(tmp_path):
    original = _cfg(tmp_path)
    original["training"]["profile"] = "synthetic-original-fast"
    original["training"]["resume_candidates"] = True
    original["training"]["legacy_phase5_config_hashes"] = ["a" * 64]
    original["training"]["phase5_feature_contract_sha256"] = config_hash({
        name: original[name] for name in ("image_size", "rapsd_rings", "pixel_baseline_size",
                                         "preprocess", "frequency", "extraction")})
    _write_training_inputs(original, rows_per_class=12, feature_config_hash_override="a" * 64)
    # The synthetic fixture normally needs only splits.sha256; migration retains
    # both historical receipts byte for byte.
    from pathlib import Path
    (Path(original["paths"]["artifacts_root"]) / "splits.json").write_text("{}")
    old_path = tmp_path / "original.yaml"
    old_path.write_text(yaml.safe_dump(original))
    source = tmp_path / "source_checkpoint"
    previous = run_phase6(original, config_file=old_path, checkpoint_dir=source)
    contract = json.loads((source / "manifest.json").read_text())["contract_sha256"]
    revised = amended(original, tmp_path, contract)
    new_path = tmp_path / "revised.yaml"
    new_path.write_text(yaml.safe_dump(revised))
    target = tmp_path / "revised_checkpoint"
    record = migrate(old_path, new_path, source, target)
    assert set(record["reused_models"]) == {"frequency_svm", "frequency_xgb", "pixel_svm", "pixel_xgb"}
    assert record["rf_models_and_candidates_reused"] is False
    assert set(json.loads((target / "manifest.json").read_text())["entries"]) == {"feature_choice", *record["reused_models"]}
    assert not (target / "search").exists()
    result = run_phase6(revised, config_file=new_path, checkpoint_dir=target)
    candidate_keys = json.loads((target / "manifest.json").read_text())["search_entries"]
    assert set(name.rsplit("_", 1)[0] for name in candidate_keys) == {"frequency_rf", "pixel_rf"}
    for old, new in zip(previous["models"], result["models"]):
        if old["model"] != "rf":
            old_model, new_model = joblib.load(old["model_path"]), joblib.load(new["model_path"])
            assert np.array_equal(old_model.predict_proba(np.zeros((2, old["feature_count"]))),
                                  new_model.predict_proba(np.zeros((2, new["feature_count"]))))
    assert previous["config_sha256"] != result["config_sha256"]

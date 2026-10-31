"""Verify the approved R2 inputs and stage historical split receipts for fast Phase 6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from aidetect.io_utils import config_hash, load_config, sha256_of_file
from aidetect.training_checkpoints import _atomic_copy


def prepare(config: Path, *, project: Path = PROJECT) -> dict:
    original = load_config(project / "config_r2.yaml")
    cfg = load_config(config)
    original_hash = config_hash(original)
    if original_hash != "ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6":
        raise ValueError("Historical R2 configuration changed")
    for key in original:
        if key not in {"training", "cv_folds", "search_iters", "paths"} and cfg[key] != original[key]:
            raise ValueError(f"Fast profile must preserve R2 {key}")
    if cfg["cv_folds"] != 3 or cfg["search_iters"] != 10:
        raise ValueError("Approved fast profile requires 3 folds and 10 candidates")
    for key in ("data_root", "features_root"):
        if cfg["paths"][key] != original["paths"][key]:
            raise ValueError("Fast profile must reuse original frozen R2 inputs")
    for key in ("r0", "model_order", "primary_model_tie_order", "calibration", "phase5_feature_contract_sha256"):
        if cfg["training"][key] != original["training"][key]:
            raise ValueError(f"Fast profile changed training {key}")
    for model in ("svm", "rf", "xgb"):
        expected_space = dict(original["training"]["search_space"][model])
        if model == "rf" and cfg["training"].get("rf_search_amendment") == "user-approved-2026-10-31":
            expected_space.update(n_estimators=[100, 300], max_features=["sqrt"])
            if cfg["training"].get("reuse_source_contract_sha256") != "141960dbe64eb5b59eae1763b36c90b2dc992243c86e781426a7cf277359b3b4":
                raise ValueError("RF amendment must pin the reviewed original fast-run contract")
        if cfg["training"]["search_space"][model] != expected_space:
            raise ValueError("Fast profile must preserve original search ranges")
    fixed = dict(cfg["training"]["search_space"]["xgb_fixed"])
    if fixed.pop("device", None) not in {"cpu", "cuda"} or fixed != original["training"]["search_space"]["xgb_fixed"]:
        raise ValueError("Fast XGBoost fixed parameters may add only cpu/cuda device")
    if cfg["training"]["legacy_phase5_config_hashes"] != [original_hash]:
        raise ValueError("Fast profile must pin the original feature configuration")
    summary = json.loads((project / "reports/phase_5_r2_checks.json").read_text())
    evidence_path = Path(summary["drive_backup"]) / "phase_5_r2_checks.json"
    if sha256_of_file(evidence_path) != summary["review_record_sha256"]:
        raise ValueError("Reviewed Phase 5 evidence changed")
    evidence = json.loads(evidence_path.read_text())
    if (evidence["status"] != "completed" or evidence["config_sha256"] != original_hash
        or evidence["test_feature_values_read"] or not evidence["sealed_reader_denied"]):
        raise ValueError("Phase 5 completion/seal evidence is invalid")
    data = Path(cfg["paths"]["data_root"])
    for relative, key in (("splits.csv", "splits_sha256"),
                          ("audit/active_manifest.csv", "active_manifest_sha256"),
                          ("audit/audit_result.json", "audit_result_sha256")):
        if sha256_of_file(data / relative) != evidence[key]:
            raise ValueError(f"Frozen R2 input changed: {relative}")
    audit = json.loads((data / "audit/audit_result.json").read_text())
    if audit["status"] != "passed" or audit["protocol"] != "R2":
        raise ValueError("R2 treated-input audit must have passed")
    target = project / cfg["paths"]["artifacts_root"]
    source = project / original["paths"]["artifacts_root"]
    for root in (source, target):
        if any((root / name).exists() for name in ("FREEZE.lock", "EVALUATION_STARTED.json")):
            raise RuntimeError("Cannot prepare training after freeze/evaluation")
    if target.resolve() == source.resolve():
        raise ValueError("Fast outputs must be separate from original R2 outputs")
    split_receipt = json.loads((source / "splits.json").read_text())
    if split_receipt["config_hash"] != original_hash:
        raise ValueError("Historical split configuration differs")
    for key in ("splits_sha256", "active_manifest_sha256", "audit_result_sha256"):
        if split_receipt[key] != evidence[key]:
            raise ValueError("Historical split provenance differs from Phase 5")
    for filename in ("splits.json", "splits.sha256"):
        if (target / filename).exists():
            if sha256_of_file(target / filename) != sha256_of_file(source / filename):
                raise ValueError("Fast split receipt differs from the original")
        else:
            _atomic_copy(source / filename, target / filename)
    if (target / "splits.sha256").read_text().strip() != evidence["splits_sha256"]:
        raise ValueError("Frozen split receipt differs from Phase 5")
    return {"profile": cfg["training"]["profile"], "config_sha256": config_hash(cfg),
            "feature_config_sha256": original_hash, "splits_sha256": evidence["splits_sha256"],
            "phase5_record_sha256": summary["review_record_sha256"], "test_features_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config_r2_fast.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config.resolve()), indent=2))

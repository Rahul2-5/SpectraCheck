"""Prepare a reviewed narrower RF search while retaining identical non-RF fits.

Only run after the user approves the RF search amendment and the old writer has
stopped. No test files are inspected; affected RF models/candidates are not reused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from aidetect.io_utils import load_config, config_hash, sha256_of_file
from aidetect.train import load_training_features, make_stratified_folds, _checkpoint_contract
from aidetect.training_checkpoints import TrainingCheckpoints, _atomic_copy, _atomic_json


def validate_amendment(original: dict, revised: dict) -> None:
    expected = deepcopy(original)
    expected["paths"]["artifacts_root"] = revised["paths"]["artifacts_root"]
    expected["paths"]["reports_root"] = revised["paths"]["reports_root"]
    for name in ("artifacts_root", "reports_root"):
        if Path(original["paths"][name]).resolve() == Path(revised["paths"][name]).resolve():
            raise ValueError("Amended RF outputs must have separate roots")
    expected["training"]["profile"] = revised["training"]["profile"]
    expected["training"]["rf_search_amendment"] = "user-approved-2026-10-31"
    expected["training"]["reuse_source_contract_sha256"] = revised["training"].get("reuse_source_contract_sha256")
    rf = expected["training"]["search_space"]["rf"]
    rf["n_estimators"] = [100, 300]
    rf["max_features"] = ["sqrt"]
    if revised != expected:
        raise ValueError("Only the explicitly reviewed RF ranges, output roots and lineage may change")


def migrate(original_path: Path, revised_path: Path, source: Path, target: Path) -> dict:
    original, revised = load_config(original_path), load_config(revised_path)
    validate_amendment(original, revised)
    if target.exists():
        raise FileExistsError("Migration requires a new checkpoint folder")
    for root in (Path(original["paths"]["data_root"]), Path(original["paths"]["features_root"]), source):
        if target.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(target.resolve()):
            raise ValueError("Migration target must be separate from inputs and original checkpoints")
    old_artifacts, new_artifacts = (Path(c["paths"]["artifacts_root"]) for c in (original, revised))
    for root in (old_artifacts, new_artifacts):
        if any((root / name).exists() for name in ("FREEZE.lock", "EVALUATION_STARTED.json")):
            raise RuntimeError("Cannot amend training after freeze/evaluation")
    for name in ("splits.json", "splits.sha256"):
        destination = new_artifacts / name
        if destination.exists() and sha256_of_file(destination) != sha256_of_file(old_artifacts / name):
            raise ValueError("New split receipt differs from the frozen source")
        if not destination.exists():
            _atomic_copy(old_artifacts / name, destination)
    old_train, _, old_provenance = load_training_features(original, config_file=original_path)
    new_train, _, new_provenance = load_training_features(revised, config_file=revised_path)
    folds = make_stratified_folds(old_train["label"], folds=original["cv_folds"], seed=original["seed"])
    if old_train["id"].tolist() != new_train["id"].tolist():
        raise ValueError("Input order changed during migration")
    old_contract = _checkpoint_contract(original, old_provenance, folds)
    new_contract = _checkpoint_contract(revised, new_provenance, folds)
    if revised["training"]["reuse_source_contract_sha256"] != config_hash(old_contract):
        raise ValueError("Explicitly pinned source checkpoint contract differs")
    with TrainingCheckpoints(source, old_contract) as previous:
        if not previous.manifest["entries"]:
            raise ValueError("No completed source work to migrate")
        source_digest = sha256_of_file(source / "manifest.json")
        with TrainingCheckpoints(target, new_contract) as following:
            choice = previous.load("feature_choice")
            if choice is None:
                raise ValueError("Completed feature choice is required")
            following.save("feature_choice", choice[0])
            reused = []
            for key in ("frequency_svm", "frequency_xgb", "pixel_svm", "pixel_xgb"):
                item = previous.load(key)
                if item is None:
                    continue
                payload, model = item
                payload = deepcopy(payload)
                payload["result"]["model_path"] = (new_artifacts / "models" / (key + ".joblib")).as_posix()
                following.save(key, payload, model)
                reused.append(key)
            lineage = {"source_contract_sha256": previous.digest,
                       "source_manifest_sha256": source_digest,
                       "source_config_sha256": config_hash(original),
                       "reused_models": reused,
                       "rf_models_and_candidates_reused": False,
                       "test_features_read": False}
            _atomic_json(target / "migration.json", lineage)
            following.manifest["migration_sha256"] = sha256_of_file(target / "migration.json")
            following._commit()
            following.progress("migrated_identical_non_rf_models")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--revised-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.original_config.resolve(), args.revised_config.resolve(),
                             args.source_checkpoint.resolve(), args.target_checkpoint.resolve()), indent=2))

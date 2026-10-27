"""Extract Phase 5 frequency summaries and pixel features for one split."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from aidetect.feature_extraction import extract_split_features
from aidetect.io_utils import config_hash, load_config, sha256_of_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    return parser


def _load_rows(data_root: Path, split: str) -> pd.DataFrame:
    active_path = data_root / "audit" / "active_manifest.csv"
    frozen_path = data_root / "splits.csv"
    active = pd.read_csv(active_path, dtype={"id": str})
    frozen = pd.read_csv(frozen_path, dtype={"id": str, "group_id": str})
    required_active = {"id", "rel_path"}
    required_frozen = {"id", "label", "split", "group_id"}
    if missing := required_active - set(active.columns):
        raise ValueError(f"Active manifest is missing columns: {sorted(missing)}")
    if missing := required_frozen - set(frozen.columns):
        raise ValueError(f"Frozen splits are missing columns: {sorted(missing)}")
    merged = frozen.merge(
        active[["id", "rel_path"]], on="id", how="inner", validate="one_to_one"
    )
    selected = merged[merged["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split {split}")
    return selected[["id", "label", "group_id", "split", "rel_path"]].sort_values("id")


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_root = Path(paths["data_root"])
    features_root = Path(paths["features_root"])
    artifacts_root = Path(paths["artifacts_root"])
    splits_path = data_root / "splits.csv"
    splits_hash_path = artifacts_root / "splits.sha256"
    rows = _load_rows(data_root, args.split)
    recorded_digest = splits_hash_path.read_text(encoding="utf-8").strip()
    actual_digest = sha256_of_file(splits_path)
    if recorded_digest != actual_digest:
        raise ValueError("splits.csv does not match artifacts/splits.sha256")
    result = extract_split_features(
        rows,
        data_root=data_root,
        features_root=features_root,
        cfg=cfg,
        split=args.split,
        config_digest=config_hash(cfg),
        splits_digest=recorded_digest,
    )
    result.update(
        {
            "config_hash": config_hash(cfg),
            "splits_sha256": recorded_digest,
            "test_features_written_only": args.split == "test",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

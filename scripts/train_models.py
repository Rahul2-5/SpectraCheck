"""Run Phase 6 model selection from previously extracted train/val batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aidetect.io_utils import load_config
from aidetect.train import run_phase6


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select and train six matched models using train and validation features only."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to the project YAML configuration",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Optional persistent directory for verified per-model resume; use the same Drive path on restart",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    result = run_phase6(cfg, config_file=config_path, checkpoint_dir=args.checkpoint_dir)
    print(
        json.dumps(
            {
                "feature_choice": result["feature_choice"]["r0"],
                "k_mad": result["feature_choice"]["k_mad"],
                "primary_model": result["primary_model"]["artifact_key"],
                "rows": {
                    "train": result["train_rows"],
                    "val": result["validation_rows"],
                },
                "models": result["models"],
                "test_features_read": result["test_features_read"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

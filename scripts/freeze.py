"""Create the committed Phase 7 freeze lock without reading test features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aidetect.freeze import freeze_phase7
from aidetect.io_utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Phase 7 inputs and sealed feature hashes without parsing test rows."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to the project YAML configuration",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    result = freeze_phase7(cfg, config_file=config_path, repo_root=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "freeze_lock": "artifacts/FREEZE.lock",
                "created_at_utc": result["created_at_utc"],
                "git_commit": result["git_commit"],
                "config_sha256": result["config_sha256"],
                "splits_sha256": result["splits_sha256"],
                "hashed_test_batches": len(result["sealed_test_batches"]),
                "test_feature_values_read": result["test_feature_values_read_during_freeze"],
                "next_step": "Commit and push FREEZE.lock to main before running evaluate_test.py once.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

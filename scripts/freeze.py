"""Create the committed Phase 7 freeze lock without reading test features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aidetect.io_utils import project_relative_path, stable_json_dumps
from aidetect.phase7_config import (
    DEFAULT_PHASE7_CONFIG_NAME,
    load_phase7_config,
    phase7_artifact_paths,
    validate_phase7_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Phase 7 inputs and sealed feature hashes without parsing test rows."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PHASE7_CONFIG_NAME,
        help=(
            "Path to the reviewed Phase 7 YAML configuration "
            f"(default: {DEFAULT_PHASE7_CONFIG_NAME})"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path, cfg = load_phase7_config(PROJECT_ROOT, args.config)
    paths = phase7_artifact_paths(cfg, PROJECT_ROOT)
    validate_phase7_artifacts(cfg, PROJECT_ROOT)
    from aidetect.freeze import freeze_phase7

    result = freeze_phase7(cfg, config_file=config_path, repo_root=PROJECT_ROOT)
    print(
        stable_json_dumps(
            {
                "freeze_lock": project_relative_path(PROJECT_ROOT, paths["freeze_lock"]),
                "created_at_utc": result["created_at_utc"],
                "git_commit": result["git_commit"],
                "config_sha256": result["config_sha256"],
                "splits_sha256": result["splits_sha256"],
                "hashed_test_batches": len(result["sealed_test_batches"]),
                "test_feature_values_read": result["test_feature_values_read_during_freeze"],
                "next_step": "Commit and push FREEZE.lock to main before running evaluate_test.py once.",
            },
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

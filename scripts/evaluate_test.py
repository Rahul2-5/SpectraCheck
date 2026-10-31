"""Run the single preregistered Phase 7 evaluation of sealed test features."""

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
        description="Verify the committed Phase 7 freeze and evaluate the sealed test set once."
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
    if not paths["freeze_lock"].is_file():
        raise FileNotFoundError(
            f"Phase 7 freeze lock is missing: {paths['freeze_lock']}. "
            "Freeze and commit the reviewed Phase 6 inputs before evaluation."
        )
    from aidetect.evaluate import evaluate_test_once

    result = evaluate_test_once(cfg, config_file=config_path, repo_root=PROJECT_ROOT)
    print(
        stable_json_dumps(
            {
                "evaluation_count": result["evaluation_count"],
                "test_rows": result["test_rows"],
                "primary_model": result["primary_model"],
                "primary_accuracy": result["models"][result["primary_model"]]["metrics"]["accuracy"],
                "results": project_relative_path(PROJECT_ROOT, paths["results_json"]),
                "predictions": result["predictions_path"],
                "test_features_read": result["test_features_read"],
            },
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the single preregistered Phase 7 evaluation of sealed test features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aidetect.evaluate import evaluate_test_once
from aidetect.io_utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the committed Phase 7 freeze and evaluate the sealed test set once."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to the project YAML configuration",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    result = evaluate_test_once(cfg, config_file=config_path, repo_root=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "evaluation_count": result["evaluation_count"],
                "test_rows": result["test_rows"],
                "primary_model": result["primary_model"],
                "primary_accuracy": result["models"][result["primary_model"]]["metrics"]["accuracy"],
                "results": "reports/results.json",
                "predictions": result["predictions_path"],
                "test_features_read": result["test_features_read"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

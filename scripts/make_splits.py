"""Create balanced group-preserving splits only after the metadata audit passes."""

import argparse
from pathlib import Path

from aidetect.audit_workflow import run_split_freeze
from aidetect.io_utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    run_split_freeze(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

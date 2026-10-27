"""Audit main data, test metadata shortcuts, and apply the preplanned JPEG fix."""

import argparse
from pathlib import Path

from aidetect.audit_workflow import run_audit
from aidetect.io_utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    result = run_audit(load_config(args.config))
    if result["status"] != "passed":
        print("STOP: metadata shortcut remains. Review reports/audit_report.md before continuing.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

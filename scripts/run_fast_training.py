"""Run approved fast Phase 6 and independently preserve verified outputs on Drive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from prepare_fast_training import prepare
from aidetect.io_utils import load_config, sha256_of_file
from aidetect.training_checkpoints import _atomic_json, _atomic_copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config_r2_fast.yaml")
    args = parser.parse_args()
    config = args.config.resolve()
    os.chdir(PROJECT)
    readiness = prepare(config)
    cfg = load_config(config)
    name = Path(cfg["paths"]["artifacts_root"]).name
    root = Path(cfg["paths"]["data_root"]).parent
    checkpoint = root / (name + "_phase6_checkpoints")
    review = root / "review_backup" / (name + "_training_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    review.mkdir(parents=True, exist_ok=False)
    lock = root / "fast_training_launch.lock"
    # One cross-profile fast runner, including the input-loading window before
    # the checkpoint writer lock is acquired. Never auto-remove a stale lock.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "review": str(review)}, stream)
    record = {**readiness, "status": "running", "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_dir": str(checkpoint), "review_backup": str(review)}
    _atomic_json(review / "start.json", record)
    print(json.dumps(record, indent=2), flush=True)
    try:
        command = [sys.executable, "-u", str(PROJECT / "scripts/train_models.py"),
                   "--config", str(config), "--checkpoint-dir", str(checkpoint),
                   "--feature-cache-dir", "/content/aidetect_train_val_cache"]
        with (review / "training.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        artifacts = PROJECT / cfg["paths"]["artifacts_root"]
        result = json.loads((artifacts / "training_results.json").read_text())
        if len(result["models"]) != 6 or result["test_features_read"] is not False:
            raise ValueError("Incomplete training output")
        for source, folder in ((artifacts, "artifacts"), (PROJECT / cfg["paths"]["reports_root"], "reports")):
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    _atomic_copy(path, review / folder / path.relative_to(source))
        _atomic_copy(config, review / config.name)
        hashes = {str(p.relative_to(review)): sha256_of_file(p)
                  for p in review.rglob("*") if p.is_file() and p.name != "start.json"}
        _atomic_json(review / "sha256.json", hashes)
        record.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        print("Six models completed and backed up. STOP before Phase 7.", flush=True)
    except BaseException:
        record["status"] = "failed_or_interrupted"
        raise
    finally:
        _atomic_json(review / "start.json", record)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

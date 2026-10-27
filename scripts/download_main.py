"""Run in Colab: stream real + selected-generator original files to Drive."""

import argparse
from pathlib import Path

from aidetect.data import stream_generator_subset
from aidetect.io_utils import config_hash, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out-dir", type=Path, help="Override data_root/main (changes config hash)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["paths"]["data_root"] = str(args.out_dir.parent)
        cfg["acquisition_output_override"] = str(args.out_dir)
    output = args.out_dir or Path(cfg["paths"]["data_root"]) / "main"
    dataset = cfg["dataset"]
    summary = stream_generator_subset(dataset["main"], dataset["generator_name"],
        dataset["source_splits"], output, dataset["max_real"], cfg["seed"],
        revision=dataset["revision"], configuration_hash=config_hash(cfg))
    print("Manifest:", output / "manifest.csv")
    if summary["saved_counts"]["ai"] < dataset["minimum_ai_images"]:
        print("STOP: target-generator count is below the configured minimum. Consult the supervisor before Phase 2.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

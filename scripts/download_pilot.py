"""Copy a CIFAKE pilot from an existing, extracted Kaggle download folder.

Get the archive manually from https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
and extract train/REAL, train/FAKE, test/REAL, test/FAKE. No Kaggle API key is used.
"""

import argparse
from pathlib import Path

from aidetect.data import prepare_cifake_sample
from aidetect.io_utils import config_hash, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out-dir", type=Path, help="Override data_root/pilot")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["pilot_source_dir"] = str(args.source_dir.resolve())
    if args.out_dir:
        cfg["acquisition_output_override"] = str(args.out_dir)
    output = args.out_dir or Path(cfg["paths"]["data_root"]) / "pilot"
    prepare_cifake_sample(args.source_dir, output, cfg["pilot"]["samples_per_class"], cfg["seed"],
        split_names=cfg["pilot"]["source_splits"], configuration_hash=config_hash(cfg))
    print("Pilot manifest:", output / "manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

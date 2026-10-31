"""Time representative fits on TRAIN only; no validation/test based device selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import joblib
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from aidetect.io_utils import config_hash, load_config
from aidetect.train import make_pipeline, search_space, make_frequency_matrix, make_pixel_matrix
from sklearn.model_selection import ParameterSampler
import pandas as pd


def benchmark(cfg, *, include_cuda=False):
    paths = sorted((Path(cfg["paths"]["features_root"]) / "train").glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError("No TRAIN batches available")
    for path in paths:
        if pq.read_metadata(path).metadata[b"aidetect_config_hash"].decode() not in cfg["training"]["legacy_phase5_config_hashes"]:
            raise ValueError("Benchmark requires original R2 TRAIN features")
    frame = pd.concat([pq.read_table(path).to_pandas() for path in paths], ignore_index=True)
    if set(frame["split"]) != {"train"}:
        raise ValueError("Benchmark only accepts TRAIN rows")
    labels = frame["label"].to_numpy()
    # Match a single 3-fold training partition's size, preserving class balance.
    per_class = min(int((labels == value).sum()) for value in (0, 1)) * 2 // 3
    indices = np.concatenate([np.flatnonzero(labels == value)[:per_class] for value in (0, 1)])
    timings = []
    with threadpool_limits(limits=1):
        for feature, matrix in (("frequency", make_frequency_matrix(frame, cfg, r0=8, k_mad=3)[0]),
                                ("pixel", make_pixel_matrix(frame, cfg)[0])):
            for model in ("svm", "rf", "xgb"):
                parameters = next(iter(ParameterSampler(search_space(model, cfg), n_iter=1, random_state=cfg["seed"])))
                for device in (("cpu", "cuda") if model == "xgb" and include_cuda else ("cpu",)):
                    active = deepcopy(cfg)
                    active["training"]["search_space"]["xgb_fixed"]["device"] = device
                    estimator = make_pipeline(model, active).set_params(**parameters)
                    start = time.perf_counter()
                    estimator.fit(matrix[indices], labels[indices])
                    elapsed = time.perf_counter() - start
                    actual_device = "cpu"
                    if model == "xgb":
                        actual_device = json.loads(estimator.named_steps["classifier"].get_booster().save_config())["learner"]["generic_param"]["device"]
                        if device == "cuda" and not actual_device.startswith("cuda"):
                            raise RuntimeError("CUDA requested but XGBoost fell back to CPU")
                    item = {"feature_set": feature, "model": model, "device": actual_device,
                            "fit_rows": len(indices), "seconds": round(elapsed, 3)}
                    timings.append(item)
                    print(json.dumps(item), flush=True)
    return {"config_sha256": config_hash(cfg), "cpu_count": joblib.cpu_count(),
            "timings": timings, "test_features_read": False,
            "limitation": "One sampled candidate per model; not a full-run ETA or research score"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config_r2_fast.yaml")
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(load_config(args.config), include_cuda=args.include_cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

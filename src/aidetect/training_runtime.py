"""Bounded CPU execution and resumable, equally budgeted candidate searches."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterSampler

from aidetect.io_utils import sha256_of_file
from aidetect.training_checkpoints import _atomic_copy


def search_workers(training: dict, *, device: str = "cpu") -> int:
    """Cap configured search workers by CPU and available memory; serialize CUDA."""
    requested = training.get("search_n_jobs", 1)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("training.search_n_jobs must be a positive integer")
    if device.startswith("cuda"):
        return 1
    workers = min(requested, joblib.cpu_count())
    memory_per_worker = training.get("worker_memory_mb", 1024)
    if isinstance(memory_per_worker, bool) or not isinstance(memory_per_worker, int) or memory_per_worker < 1:
        raise ValueError("training.worker_memory_mb must be a positive integer")
    info = Path("/proc/meminfo")
    if info.is_file():
        fields = dict(line.split(":", 1) for line in info.read_text().splitlines())
        available_mb = int(fields["MemAvailable"].split()[0]) // 1024
        workers = min(workers, max(1, available_mb // memory_per_worker))
    return max(1, workers)


def cache_training_batches(source: Path, destination: Path) -> None:
    """Copy only train/val batches and verify bytes; never enumerate test_sealed."""
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Feature cache must be separate from feature inputs")
    for split in ("train", "val"):
        for path in sorted((source / split).glob("batch_*.parquet")):
            target = destination / split / path.name
            if target.exists():
                if sha256_of_file(target) != sha256_of_file(path):
                    raise ValueError(f"Cached training batch changed: {target}")
            else:
                _atomic_copy(path, target)


def verify_cuda_device(cfg: dict) -> None:
    """Reject silent CPU fallback when the optional CUDA profile is selected."""
    device = cfg["training"]["search_space"]["xgb_fixed"].get("device", "cpu")
    if device == "cpu":
        return
    if device != "cuda":
        raise ValueError("XGBoost device must be cpu or cuda")
    from xgboost import XGBClassifier
    probe = XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda", n_jobs=1)
    probe.fit(np.array([[0.], [1.], [2.], [3.]]), np.array([0, 1, 0, 1]))
    actual = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith("cuda"):
        raise RuntimeError("CUDA profile requested but XGBoost fell back to CPU; use the CPU profile")


def resumable_search(pipeline, distributions, *, iterations, seed, folds,
                     values, labels, workers, checkpoints, key):
    """Sample once, score candidates with fold-local pipelines, save each result.

    Same ParameterSampler order and first-best tie rule as RandomizedSearchCV.
    No fitted candidate models are cached; the selected pipeline is refitted once.
    """
    parameters = list(ParameterSampler(distributions, n_iter=iterations, random_state=seed))
    means, deviations = [], []
    for index, params in enumerate(parameters):
        # NumPy scalar parameters must have stable JSON representations.
        params = json.loads(json.dumps(params, default=lambda x: x.item()))
        entry = checkpoints.load_candidate(key, index) if checkpoints is not None else None
        if entry is not None:
            if entry["params"] != params or len(entry["fold_scores"]) != len(folds):
                raise ValueError("Cached search candidate differs from sampled parameters/folds")
            scores = np.asarray(entry["fold_scores"], dtype=float)
        else:
            search = GridSearchCV(pipeline, [{k: [v] for k, v in params.items()}],
                                  scoring="accuracy", cv=folds, n_jobs=workers,
                                  pre_dispatch=workers, refit=False, error_score="raise")
            search.fit(values, labels)
            scores = np.array([search.cv_results_[f"split{fold}_test_score"][0]
                               for fold in range(len(folds))])
            if checkpoints is not None:
                checkpoints.save_candidate(key, index, {"params": params, "fold_scores": scores.tolist()})
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Search candidate has invalid accuracy scores")
        means.append(float(scores.mean()))
        deviations.append(float(scores.std()))
        print(f"[Phase 6] {key}: candidate {index + 1}/{len(parameters)} "
              f"{'reused' if entry is not None else 'saved'}", flush=True)
    best = int(np.argmax(means))
    best_params = parameters[best]
    estimator = clone(pipeline).set_params(**best_params).fit(values, labels)
    return SimpleNamespace(best_index_=best, best_params_=best_params, best_estimator_=estimator,
                           cv_results_={"mean_test_score": means, "std_test_score": deviations})

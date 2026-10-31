from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from aidetect.train import make_pipeline, search_space, make_stratified_folds, run_phase6, load_training_features
from aidetect.training_checkpoints import TrainingCheckpoints
from aidetect.training_runtime import resumable_search, cache_training_batches, search_workers
from test_train import _cfg, _summary_frame, _write_training_inputs


@pytest.mark.parametrize("model", ["svm", "rf", "xgb"])
def test_resumable_search_matches_randomized_search(tmp_path, model):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 8))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    folds = make_stratified_folds(y, folds=3, seed=42)
    pipeline, space = make_pipeline(model, cfg), search_space(model, cfg)
    baseline = RandomizedSearchCV(pipeline, space, n_iter=3, cv=folds, random_state=42,
                                  scoring="accuracy", error_score="raise").fit(x, y)
    with TrainingCheckpoints(tmp_path / "checkpoints", {"synthetic": True}) as store:
        result = resumable_search(pipeline, space, iterations=3, seed=42, folds=folds,
                                  values=x, labels=y, workers=1, checkpoints=store, key=f"pixel_{model}")
    assert result.best_params_ == baseline.best_params_
    assert result.cv_results_["mean_test_score"] == pytest.approx(baseline.cv_results_["mean_test_score"])
    assert result.cv_results_["std_test_score"] == pytest.approx(baseline.cv_results_["std_test_score"])
    assert np.array_equal(result.best_estimator_.predict(x), baseline.predict(x))


def test_candidate_resume_and_corruption_rejection(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    x = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    folds = make_stratified_folds(y, folds=3, seed=42)
    original = GridSearchCV.fit
    calls = []
    def interrupted(search, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("disconnect")
        return original(search, *args, **kwargs)
    monkeypatch.setattr(GridSearchCV, "fit", interrupted)
    root = tmp_path / "checkpoints"
    def search(store):
        return resumable_search(make_pipeline("svm", cfg), search_space("svm", cfg), iterations=3,
                                seed=42, folds=folds, values=x, labels=y, workers=1,
                                checkpoints=store, key="frequency_svm")
    with pytest.raises(KeyboardInterrupt):
        with TrainingCheckpoints(root, {"synthetic": True}) as store:
            search(store)
    calls.clear()
    def counted(search, *args, **kwargs):
        calls.append(1)
        return original(search, *args, **kwargs)
    monkeypatch.setattr(GridSearchCV, "fit", counted)
    with TrainingCheckpoints(root, {"synthetic": True}) as store:
        search(store)
    assert len(calls) == 2
    (root / "search/frequency_svm_0000.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        with TrainingCheckpoints(root, {"synthetic": True}):
            pytest.fail("Corrupt candidate admitted")


def test_cache_copies_only_train_val_and_rejects_changed_bytes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache"
    for split in ("train", "val", "test_sealed"):
        (source / split).mkdir(parents=True)
        (source / split / "batch_00000.parquet").write_bytes(split.encode())
    cache_training_batches(source, target)
    assert {p.name for p in target.iterdir()} == {"train", "val"}
    cache_training_batches(source, target)
    (target / "train/batch_00000.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache_training_batches(source, target)
    with pytest.raises(ValueError, match="separate"):
        cache_training_batches(source, source / "cache")


def test_fast_end_to_end_with_local_cache_keeps_test_sealed(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["training"]["resume_candidates"] = True
    cfg["training"]["profile"] = "synthetic-fast"
    _write_training_inputs(cfg, rows_per_class=12)
    result = run_phase6(cfg, checkpoint_dir=tmp_path / "checkpoints", feature_cache_dir=tmp_path / "cache")
    assert len(result["models"]) == 6 and result["test_features_read"] is False
    assert result["training_profile"] == "synthetic-fast"
    assert len(json.loads((tmp_path / "checkpoints/manifest.json").read_text())["search_entries"]) == 12
    assert not (tmp_path / "cache/test_sealed").exists()


def test_worker_cap_prevents_gpu_contention_and_cpu_oversubscription(monkeypatch):
    monkeypatch.setattr(joblib, "cpu_count", lambda: 2)
    assert search_workers({"search_n_jobs": 8}) <= 2
    assert search_workers({"search_n_jobs": 8}, device="cuda") == 1
    with pytest.raises(ValueError):
        search_workers({"search_n_jobs": -1})


def test_fast_profiles_preserve_original_feature_contract():
    root = Path(__file__).parents[1]
    original = yaml.safe_load((root / "config_r2.yaml").read_text())
    for filename in ("config_r2_fast.yaml", "config_r2_fast_t4.yaml"):
        cfg = yaml.safe_load((root / filename).read_text())
        assert cfg["cv_folds"] == 3 and cfg["search_iters"] == 10
        for key in ("preprocess", "frequency", "extraction", "seed", "split", "dataset"):
            assert cfg[key] == original[key]
        assert cfg["paths"]["features_root"] == original["paths"]["features_root"]
        assert cfg["paths"]["artifacts_root"] != original["paths"]["artifacts_root"]

"""Leakage-conscious Phase 6 model selection using train and validation only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import platform
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import loguniform, randint, uniform
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost

from aidetect.feature_extraction import (
    SCHEMA_VERSION,
    energy_ratio_from_cumulative,
    feature_output_dir,
)
from aidetect.io_utils import config_hash, sha256_of_file


_DEFAULT_MODEL_ORDER = ("svm", "rf", "xgb")
_DEFAULT_FEATURE_SETS = ("frequency", "pixel")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_pair(value: object, name: str, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric bounds")
    low, high = (float(item) for item in value)
    if not np.isfinite([low, high]).all() or high < low or (positive and low <= 0.0):
        raise ValueError(f"{name} has invalid bounds")
    return low, high


def _integer_pair(value: object, name: str, *, positive: bool = False) -> tuple[int, int]:
    low, high = _finite_pair(value, name, positive=positive)
    if not low.is_integer() or not high.is_integer():
        raise ValueError(f"{name} bounds must be integers")
    return int(low), int(high)


def _training_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    training = _mapping(cfg.get("training"), "cfg['training']")
    order = tuple(training.get("model_order", _DEFAULT_MODEL_ORDER))
    if order != _DEFAULT_MODEL_ORDER:
        raise ValueError(f"training.model_order must be {_DEFAULT_MODEL_ORDER}")
    tie_order = tuple(training.get("primary_model_tie_order", order))
    if sorted(tie_order) != sorted(order):
        raise ValueError("training.primary_model_tie_order must list svm, rf, and xgb once")
    return training


def _peak_thresholds(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    extraction = _mapping(cfg.get("extraction", {}), "cfg['extraction']")
    raw = extraction.get("peak_k_mad")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("cfg['extraction']['peak_k_mad'] must be a non-empty sequence")
    values = tuple(_positive_int(value, "peak_k_mad value") for value in raw)
    if len(set(values)) != len(values):
        raise ValueError("peak_k_mad values must be unique")
    return values


def _r0_values(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    training = _training_config(cfg)
    values = _mapping(training.get("r0"), "cfg['training']['r0']")
    low = _positive_int(values.get("min"), "training.r0.min")
    high = _positive_int(values.get("max"), "training.r0.max")
    step = _positive_int(values.get("step"), "training.r0.step")
    if high < low or (high - low) % step:
        raise ValueError("training.r0 bounds must form an inclusive regular grid")
    return tuple(range(low, high + 1, step))


def _column_sequence(frame: pd.DataFrame, prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matches = [(int(match.group(1)), name) for name in frame.columns if (match := pattern.match(str(name)))]
    matches.sort()
    indexes = [index for index, _ in matches]
    if not matches or indexes != list(range(len(matches))):
        raise ValueError(f"Feature columns {prefix}<index> must be contiguous from zero")
    return [name for _, name in matches]


def _threshold_suffix(value: int) -> str:
    return f"{value:g}"


def make_frequency_matrix(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    r0: int,
    k_mad: int,
) -> tuple[np.ndarray, list[str]]:
    """Reconstruct a 68-value-style vector from Phase 5 summary columns."""
    rings = _positive_int(cfg.get("rapsd_rings"), "cfg['rapsd_rings']")
    image_size = _positive_int(cfg.get("image_size"), "cfg['image_size']")
    pixel_size = _positive_int(cfg.get("pixel_baseline_size"), "cfg['pixel_baseline_size']")
    energy_columns = _column_sequence(frame, "cumulative_energy_")
    expected_energy_count = math.ceil(math.sqrt(2.0) * (image_size // 2)) + 1
    if len(energy_columns) != expected_energy_count:
        raise ValueError(
            f"Expected {expected_energy_count} cumulative-energy columns for image_size={image_size}; "
            f"found {len(energy_columns)}"
        )
    rapsd_columns = _column_sequence(frame, "rapsd_ring_")
    if len(rapsd_columns) != rings:
        raise ValueError(f"Expected {rings} RAPSD columns; found {len(rapsd_columns)}")
    if "entropy" not in frame:
        raise ValueError("Feature rows are missing entropy")
    suffix = _threshold_suffix(k_mad)
    peak_count_column = f"peak_count_k{suffix}"
    peak_strength_column = f"peak_strength_k{suffix}"
    for name in (peak_count_column, peak_strength_column):
        if name not in frame:
            raise ValueError(f"Feature rows are missing {name}")

    cumulative = frame[energy_columns].to_numpy(dtype=np.float64, copy=True)
    frequency = _mapping(cfg.get("frequency", {}), "cfg['frequency']")
    epsilon = float(frequency.get("epsilon", 1.0e-12))
    energy_ratio = np.fromiter(
        (energy_ratio_from_cumulative(row, r0, epsilon) for row in cumulative),
        dtype=np.float64,
        count=len(frame),
    )
    values = np.column_stack(
        [
            energy_ratio,
            frame["entropy"].to_numpy(dtype=np.float64),
            frame[rapsd_columns].to_numpy(dtype=np.float64),
            frame[peak_count_column].to_numpy(dtype=np.float64),
            frame[peak_strength_column].to_numpy(dtype=np.float64),
        ]
    )
    names = [
        "energy_ratio",
        "spectral_entropy",
        *[f"rapsd_ring_{index:02d}" for index in range(rings)],
        "peak_count",
        "peak_strength",
    ]
    if values.shape != (len(frame), len(names)) or not np.isfinite(values).all():
        raise ValueError("Reconstructed frequency features are not finite or have an invalid shape")
    return values, names


def make_pixel_matrix(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    size = _positive_int(cfg.get("pixel_baseline_size"), "cfg['pixel_baseline_size']")
    columns = _column_sequence(frame, "pixel_")
    if len(columns) != size * size:
        raise ValueError(f"Expected {size * size} pixel columns; found {len(columns)}")
    values = frame[columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Pixel feature values contain NaN or Inf")
    return values, [f"pixel_{index:04d}" for index in range(len(columns))]


def make_stratified_folds(
    labels: Sequence[int], *, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize one reusable shuffled stratified fold set for all searches."""
    y = np.asarray(labels)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a one-dimensional binary vector using 0/1")
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if counts.min() < folds:
        raise ValueError(f"Each class needs at least {folds} training rows; counts are {counts.tolist()}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = [(train.copy(), valid.copy()) for train, valid in splitter.split(np.zeros(len(y)), y)]
    validation_counts = np.zeros(len(y), dtype=np.int64)
    for train, valid in result:
        if np.intersect1d(train, valid).size or len(train) + len(valid) != len(y):
            raise RuntimeError("A stratified fold has overlapping or missing indices")
        validation_counts[valid] += 1
    if not np.all(validation_counts == 1):
        raise RuntimeError("Stratified folds do not form one complete validation partition")
    return result


def assert_scaler_is_fold_local(
    values: np.ndarray,
    labels: Sequence[int],
    pipeline: Pipeline,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Prove each cloned pipeline scaler is fitted only on its CV training fold."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Leakage check requires an sklearn Pipeline")
    step_names = [name for name, _ in pipeline.steps]
    if "scale" not in step_names or "classifier" not in step_names:
        raise ValueError("Pipeline must contain ordered 'scale' and 'classifier' steps")
    if step_names.index("scale") > step_names.index("classifier"):
        raise ValueError("StandardScaler must run before the classifier")
    if not isinstance(pipeline.named_steps["scale"], StandardScaler):
        raise TypeError("Pipeline 'scale' step must be StandardScaler")
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("values and labels must have matching row counts")
    for train_indices, valid_indices in folds:
        if len(train_indices) >= len(x) or np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Each fold must hold out rows from scaler fitting")
        fitted = clone(pipeline).fit(x[train_indices], y[train_indices])
        scaler = fitted.named_steps["scale"]
        expected = StandardScaler().fit(x[train_indices])
        if scaler.n_samples_seen_ != len(train_indices):
            raise AssertionError("StandardScaler saw rows outside the current training fold")
        if not np.allclose(scaler.mean_, expected.mean_) or not np.allclose(
            scaler.var_, expected.var_
        ):
            raise AssertionError("StandardScaler statistics differ from the current training fold")


def _estimator(name: str, cfg: Mapping[str, Any]) -> Any:
    training = _training_config(cfg)
    seed = _positive_int(cfg.get("seed"), "cfg['seed']")
    n_jobs = _positive_int(training.get("estimator_n_jobs"), "training.estimator_n_jobs")
    if name == "svm":
        return SVC(kernel="rbf")
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if name == "xgb":
        fixed = _mapping(training.get("search_space", {}).get("xgb_fixed"), "training.search_space.xgb_fixed")
        return XGBClassifier(
            **dict(fixed),
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model {name!r}")


def make_pipeline(name: str, cfg: Mapping[str, Any]) -> Pipeline:
    """Build the common scaler-plus-classifier pipeline used inside CV."""
    return Pipeline([("scale", StandardScaler()), ("classifier", _estimator(name, cfg))])


def search_space(name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the configured PRD ranges to SciPy distributions/lists."""
    training = _training_config(cfg)
    spaces = _mapping(training.get("search_space"), "training.search_space")
    values = _mapping(spaces.get(name), f"training.search_space.{name}")
    if name == "svm":
        c_bounds = _finite_pair(values.get("C_loguniform"), "SVM C_loguniform", positive=True)
        gamma_bounds = _finite_pair(
            values.get("gamma_loguniform"), "SVM gamma_loguniform", positive=True
        )
        result = {
            "classifier__C": loguniform(*c_bounds),
            "classifier__gamma": loguniform(*gamma_bounds),
        }
    elif name == "rf":
        n_estimators = _integer_pair(
            values.get("n_estimators"), "RF n_estimators", positive=True
        )
        leaf = _integer_pair(
            values.get("min_samples_leaf"), "RF min_samples_leaf", positive=True
        )
        result = {
            "classifier__n_estimators": randint(n_estimators[0], n_estimators[1] + 1),
            "classifier__max_depth": list(values.get("max_depth", [])),
            "classifier__min_samples_leaf": randint(leaf[0], leaf[1] + 1),
            "classifier__max_features": list(values.get("max_features", [])),
        }
    elif name == "xgb":
        n_estimators = _integer_pair(
            values.get("n_estimators"), "XGB n_estimators", positive=True
        )
        max_depth = _integer_pair(
            values.get("max_depth"), "XGB max_depth", positive=True
        )
        lr = _finite_pair(values.get("learning_rate_uniform"), "XGB learning_rate_uniform")
        subsample = _finite_pair(values.get("subsample_uniform"), "XGB subsample_uniform")
        colsample = _finite_pair(
            values.get("colsample_bytree_uniform"), "XGB colsample_bytree_uniform"
        )
        result = {
            "classifier__n_estimators": randint(n_estimators[0], n_estimators[1] + 1),
            "classifier__max_depth": randint(max_depth[0], max_depth[1] + 1),
            "classifier__learning_rate": uniform(lr[0], lr[1] - lr[0]),
            "classifier__subsample": uniform(subsample[0], subsample[1] - subsample[0]),
            "classifier__colsample_bytree": uniform(colsample[0], colsample[1] - colsample[0]),
        }
    else:
        raise ValueError(f"Unknown model {name!r}")
    allowed = set(make_pipeline(name, cfg).get_params(deep=True))
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"Search parameters are not accepted by {name}: {unknown}")
    if any(not isinstance(item, (list, tuple)) or len(item) == 0 for item in result.values() if isinstance(item, list)):
        raise ValueError(f"Search space for {name} contains an empty parameter list")
    return result


def choose_frequency_features(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    """Choose r0/k_mad using a default RF fitted on train and scored on val."""
    train_y = train_frame["label"].to_numpy(dtype=np.int64)
    val_y = val_frame["label"].to_numpy(dtype=np.int64)
    thresholds = _peak_thresholds(cfg)
    seed = _positive_int(cfg.get("seed"), "cfg['seed']")
    training = _training_config(cfg)
    n_jobs = _positive_int(training.get("estimator_n_jobs"), "training.estimator_n_jobs")
    scores: list[dict[str, Any]] = []
    best: tuple[float, int, int, np.ndarray, np.ndarray, list[str]] | None = None
    for r0 in _r0_values(cfg):
        for k_mad in thresholds:
            train_x, names = make_frequency_matrix(train_frame, cfg, r0=r0, k_mad=k_mad)
            val_x, val_names = make_frequency_matrix(val_frame, cfg, r0=r0, k_mad=k_mad)
            if names != val_names:
                raise ValueError("Train and validation frequency feature names differ")
            selector = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
            selector.fit(train_x, train_y)
            score = float(accuracy_score(val_y, selector.predict(val_x)))
            scores.append({"r0": r0, "k_mad": k_mad, "validation_accuracy": score})
            current = (score, r0, k_mad, train_x, val_x, names)
            if best is None or score > best[0] or (
                score == best[0] and (r0, k_mad) < (best[1], best[2])
            ):
                best = current
    if best is None:
        raise RuntimeError("Feature-selection grid was empty")
    record = {
        "r0": best[1],
        "k_mad": best[2],
        "validation_accuracy": best[0],
        "selection_rule": "highest validation accuracy; ties use smallest r0, then smallest k_mad",
        "candidates": scores,
    }
    return record, best[3], best[4], best[5]


def _validate_frames(train: pd.DataFrame, val: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    required = {"id", "label", "group_id", "split"}
    for name, frame, expected_split in (("train", train, "train"), ("val", val, "val")):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} features are missing columns: {missing}")
        if frame.empty or set(frame["split"].astype(str)) != {expected_split}:
            raise ValueError(f"{name} feature rows must all have split={expected_split!r}")
        if frame["id"].isna().any() or frame["id"].astype(str).str.len().eq(0).any():
            raise ValueError(f"{name} feature rows contain a missing or empty ID")
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"{name} feature rows contain duplicate IDs")
        if frame["label"].isna().any():
            raise ValueError(f"{name} feature rows contain a missing label")
        numeric_labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0, 1]).all():
            raise ValueError(f"{name} labels must use the binary convention real=0, AI=1")
        labels = set(numeric_labels.astype(np.int64))
        if labels != {0, 1}:
            raise ValueError(f"{name} feature rows must contain both binary labels 0 and 1")
        counts = pd.Series(numeric_labels.astype(np.int64)).value_counts()
        if counts[0] != counts[1]:
            raise ValueError(f"{name} rows must be exactly class-balanced; got {counts.to_dict()}")
        if frame["group_id"].isna().any() or frame["group_id"].astype(str).str.len().eq(0).any():
            raise ValueError(f"{name} rows contain a missing or empty group_id")
        make_pixel_matrix(frame, cfg)

    if set(train["id"].astype(str)) & set(val["id"].astype(str)):
        raise ValueError("Image IDs cross the train/validation boundary")
    if set(train["group_id"].astype(str)) & set(val["group_id"].astype(str)):
        raise ValueError("Duplicate groups cross the train/validation boundary")


def _split_digest(artifacts_root: Path) -> str:
    path = artifacts_root / "splits.sha256"
    if not path.is_file():
        raise FileNotFoundError(f"Immutable split digest is required at {path}")
    digest = path.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid SHA-256 digest in {path}")
    return digest


def _validate_batch_metadata(
    features_root: Path,
    split: str,
    cfg: Mapping[str, Any],
    *,
    config_digest: str,
    splits_digest: str,
) -> tuple[list[Path], str]:
    directory = feature_output_dir(features_root, split)
    paths = sorted(directory.glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} feature batches found under {directory}")
    extraction = _peak_thresholds(cfg)
    legacy_hashes = tuple(
        str(value).lower()
        for value in _training_config(cfg).get("legacy_phase5_config_hashes", [])
    )
    current_hash = config_digest
    expected_contract = str(
        _training_config(cfg).get("phase5_feature_contract_sha256", "")
    ).lower()
    feature_contract = {
        name: cfg[name]
        for name in (
            "image_size",
            "rapsd_rings",
            "pixel_baseline_size",
            "preprocess",
            "frequency",
            "extraction",
        )
    }
    legacy_contract_matches = config_hash(feature_contract) == expected_contract
    expected = {
        b"aidetect_schema_version": str(SCHEMA_VERSION).encode("ascii"),
        b"aidetect_splits_sha256": splits_digest.encode("ascii"),
        b"aidetect_split": split.encode("ascii"),
        b"aidetect_image_side": str(_positive_int(cfg.get("image_size"), "image_size")).encode("ascii"),
        b"aidetect_rapsd_rings": str(_positive_int(cfg.get("rapsd_rings"), "rapsd_rings")).encode("ascii"),
        b"aidetect_pixel_baseline_size": str(
            _positive_int(cfg.get("pixel_baseline_size"), "pixel_baseline_size")
        ).encode("ascii"),
        b"aidetect_peak_k_mad": json.dumps(list(extraction)).encode("ascii"),
    }
    for batch_index, path in enumerate(paths):
        metadata = pq.read_metadata(path).metadata or {}
        expected[b"aidetect_batch_index"] = str(batch_index).encode("ascii")
        mismatches = [key.decode("ascii") for key, value in expected.items() if metadata.get(key) != value]
        stored_config_hash = metadata.get(b"aidetect_config_hash", b"").decode("ascii").lower()
        if stored_config_hash != current_hash and stored_config_hash not in legacy_hashes:
            mismatches.append("aidetect_config_hash")
        if stored_config_hash in legacy_hashes and not legacy_contract_matches:
            raise ValueError(
                "Legacy Phase 5 feature batches are compatible only while the recorded "
                "preprocessing/frequency feature contract remains unchanged"
            )
        if mismatches:
            raise ValueError(
                f"Feature batch {path} has incompatible provenance metadata: {', '.join(mismatches)}"
            )
        batch_ids = [str(value) for value in pq.read_table(path, columns=["id"]).column("id").to_pylist()]
        batch_digest = hashlib.sha256("\n".join(batch_ids).encode("utf-8")).hexdigest()
        if metadata.get(b"aidetect_batch_ids_sha256") != batch_digest.encode("ascii"):
            raise ValueError(f"Feature batch {path} ID digest does not match its row IDs")
    source_hashes = {
        pq.read_metadata(path).metadata[b"aidetect_config_hash"].decode("ascii").lower()
        for path in paths
    }
    if len(source_hashes) != 1:
        raise ValueError(f"{split} feature batches mix configuration hashes")
    return paths, source_hashes.pop()


def load_training_features(
    cfg: Mapping[str, Any],
    *,
    config_file: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and validate train/val only; this function never opens test_sealed."""
    paths = _mapping(cfg.get("paths", {}), "cfg['paths']")
    features_root = Path(paths.get("features_root", "features"))
    artifacts_root = Path(paths.get("artifacts_root", "artifacts"))
    data_root = Path(paths.get("data_root", "data"))
    if (artifacts_root / "FREEZE.lock").exists():
        raise RuntimeError("Phase 6 training must finish before the Phase 7 FREEZE.lock")
    digest = _split_digest(artifacts_root)
    split_path = data_root / "splits.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Immutable split file is required at {split_path}")
    if sha256_of_file(split_path) != digest:
        raise ValueError("data_root/splits.csv does not match artifacts/splits.sha256")
    config_digest = config_hash(dict(cfg))
    train_paths, train_config_digest = _validate_batch_metadata(
        features_root, "train", cfg, config_digest=config_digest, splits_digest=digest
    )
    val_paths, val_config_digest = _validate_batch_metadata(
        features_root, "val", cfg, config_digest=config_digest, splits_digest=digest
    )
    if train_config_digest != val_config_digest:
        raise ValueError("Train and validation features were generated from different configurations")
    from aidetect.feature_extraction import read_feature_batches

    train = read_feature_batches(features_root, "train", artifacts_root=artifacts_root)
    val = read_feature_batches(features_root, "val", artifacts_root=artifacts_root)
    _validate_frames(train, val, cfg)
    provenance = {
        "config_sha256": config_digest,
        "feature_config_sha256": train_config_digest,
        "splits_sha256": digest,
        "feature_batches": {
            split: [
                {"path": path.relative_to(features_root).as_posix(), "sha256": sha256_of_file(path)}
                for path in files
            ]
            for split, files in (("train", train_paths), ("val", val_paths))
        },
        "config_file": str(config_file),
    }
    return train, val, provenance


def _digest_ids(values: Sequence[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _folds_digest(folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for train, valid in folds:
        digest.update(np.asarray(train, dtype="<i8").tobytes())
        digest.update(b"|")
        digest.update(np.asarray(valid, dtype="<i8").tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _metrics(model: Any, values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(model.predict(values), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = list(np.asarray(model.classes_).tolist())
    if 1 not in classes or probabilities.ndim != 2:
        raise ValueError("Classifier does not expose a probability column for AI label 1")
    positive = probabilities[:, classes.index(1)]
    if not np.isfinite(positive).all() or np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError("Classifier returned invalid validation probabilities")
    return {
        "validation_accuracy": float(accuracy_score(labels, predictions)),
        "validation_roc_auc": float(roc_auc_score(labels, positive)),
        "validation_brier_score": float(brier_score_loss(labels, positive)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Cannot serialize non-finite training result")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(model, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_importance(
    reports_root: Path,
    model_name: str,
    values: np.ndarray,
    names: Sequence[str],
    top_n: int,
    figure_dpi: int,
) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(values) != len(names) or not np.isfinite(values).all():
        raise ValueError("Feature importance values and names are invalid")
    ordered = sorted(range(len(names)), key=lambda index: (-float(values[index]), str(names[index])))
    selected = ordered[: min(top_n, len(ordered))]
    report_dir = reports_root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"phase_6_feature_importance_frequency_{model_name}.csv"
    png_path = report_dir / f"phase_6_feature_importance_frequency_{model_name}.png"
    frame = pd.DataFrame(
        [{"feature": names[index], "importance": float(values[index])} for index in selected]
    )
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    figure, axis = plt.subplots(figsize=(9, max(4.0, len(selected) * 0.28)))
    display = frame.iloc[::-1]
    axis.barh(display["feature"], display["importance"], color="steelblue")
    axis.set_xlabel("Feature importance")
    axis.set_title(f"Top {len(selected)} frequency features: {model_name.upper()} (descriptive)")
    figure.tight_layout()
    figure.savefig(png_path, dpi=figure_dpi)
    plt.close(figure)
    return str(csv_path), str(png_path)


def _format_results_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Phase 6 training results",
        "",
        "Validation metrics are computed on validation rows only. No test features are read here.",
        "",
        "| Feature set | Model | CV accuracy (mean ± SD) | Validation accuracy | Validation ROC-AUC | Validation Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {feature_set} | {model} | {cv_mean:.4f} ± {cv_sd:.4f} | "
            "{validation_accuracy:.4f} | {validation_roc_auc:.4f} | {validation_brier_score:.4f} |".format(
                **result
            )
        )
    lines.append("")
    lines.append("Feature importance plots are descriptive and do not affect model selection.")
    return "\n".join(lines) + "\n"


def _assert_outputs_absent(artifacts_root: Path, reports_root: Path) -> None:
    expected = [
        artifacts_root / "feature_choice.json",
        artifacts_root / "primary_model.json",
        artifacts_root / "versions.json",
        artifacts_root / "training_results.json",
        reports_root / "phase_6_results.json",
        reports_root / "phase_6_results.md",
        reports_root / "figures" / "phase_6_feature_importance_frequency_rf.csv",
        reports_root / "figures" / "phase_6_feature_importance_frequency_rf.png",
        reports_root / "figures" / "phase_6_feature_importance_frequency_xgb.csv",
        reports_root / "figures" / "phase_6_feature_importance_frequency_xgb.png",
        *[
            artifacts_root / "models" / f"{feature_set}_{model}.joblib"
            for feature_set in _DEFAULT_FEATURE_SETS
            for model in _DEFAULT_MODEL_ORDER
        ],
    ]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            "Phase 6 outputs already exist; inspect and archive them before a new run: "
            + ", ".join(existing)
        )


def run_phase6(cfg: Mapping[str, Any], *, config_file: str | Path = "config.yaml") -> dict[str, Any]:
    """Run feature selection and six matched searches using train/validation only."""
    training = _training_config(cfg)
    paths_cfg = _mapping(cfg.get("paths", {}), "cfg['paths']")
    artifacts_root = Path(paths_cfg.get("artifacts_root", "artifacts"))
    reports_root = Path(paths_cfg.get("reports_root", "reports"))
    _assert_outputs_absent(artifacts_root, reports_root)
    train_frame, val_frame, provenance = load_training_features(cfg, config_file=config_file)
    _validate_frames(train_frame, val_frame, cfg)

    train_y = train_frame["label"].to_numpy(dtype=np.int64)
    val_y = val_frame["label"].to_numpy(dtype=np.int64)
    seed = _positive_int(cfg.get("seed"), "cfg['seed']")
    folds_count = _positive_int(cfg.get("cv_folds"), "cfg['cv_folds']")
    iterations = _positive_int(cfg.get("search_iters"), "cfg['search_iters']")
    scoring = str(training.get("scoring", "accuracy"))
    if scoring != "accuracy":
        raise ValueError("Phase 6 hyperparameter searches must use accuracy scoring")
    search_n_jobs = _positive_int(training.get("search_n_jobs"), "training.search_n_jobs")
    folds = make_stratified_folds(train_y, folds=folds_count, seed=seed)
    cv_digest = _folds_digest(folds)

    choice, frequency_train, frequency_val, frequency_names = choose_frequency_features(
        train_frame, val_frame, cfg
    )
    pixel_train, pixel_names = make_pixel_matrix(train_frame, cfg)
    pixel_val, val_pixel_names = make_pixel_matrix(val_frame, cfg)
    if pixel_names != val_pixel_names:
        raise ValueError("Train and validation pixel feature names differ")
    feature_matrices = {
        "frequency": (frequency_train, frequency_val, frequency_names),
        "pixel": (pixel_train, pixel_val, pixel_names),
    }

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for feature_set in _DEFAULT_FEATURE_SETS:
        x_train, x_val, names = feature_matrices[feature_set]
        for model_name in _DEFAULT_MODEL_ORDER:
            pipeline = make_pipeline(model_name, cfg)
            parameter_distributions = search_space(model_name, cfg)
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_distributions,
                n_iter=iterations,
                scoring=scoring,
                n_jobs=search_n_jobs,
                cv=folds,
                random_state=seed,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(x_train, train_y)
            if model_name == "svm":
                calibration_cfg = _mapping(
                    training.get("calibration", {}), "training.calibration"
                )
                fitted = CalibratedClassifierCV(
                    estimator=clone(search.best_estimator_),
                    method=str(calibration_cfg.get("method", "sigmoid")),
                    cv=StratifiedKFold(
                        n_splits=folds_count, shuffle=True, random_state=seed
                    ),
                    n_jobs=search_n_jobs,
                )
                fitted.fit(x_train, train_y)
            else:
                fitted = clone(search.best_estimator_).fit(x_train, train_y)

            metrics = _metrics(fitted, x_val, val_y)
            key = f"{feature_set}_{model_name}"
            model_path = artifacts_root / "models" / f"{key}.joblib"
            fitted_models[key] = fitted
            results.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "key": key,
                    "cv_mean": float(search.cv_results_["mean_test_score"][search.best_index_]),
                    "cv_sd": float(search.cv_results_["std_test_score"][search.best_index_]),
                    **metrics,
                    "best_params": _jsonable(search.best_params_),
                    "feature_count": len(names),
                    "model_path": model_path.as_posix(),
                }
            )

    tie_order = tuple(training.get("primary_model_tie_order", _DEFAULT_MODEL_ORDER))
    frequency_results = [result for result in results if result["feature_set"] == "frequency"]
    tie_rank = {name: index for index, name in enumerate(tie_order)}
    primary = min(
        frequency_results,
        key=lambda result: (-result["validation_accuracy"], tie_rank[result["model"]]),
    )
    choice.update(
        {
            **provenance,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(val_frame)),
            "train_ids_sha256": _digest_ids(train_frame["id"].astype(str).tolist()),
            "validation_ids_sha256": _digest_ids(val_frame["id"].astype(str).tolist()),
        }
    )
    primary_record = {
        "feature_set": "frequency",
        "model": primary["model"],
        "artifact_key": primary["key"],
        "selection_metric": "validation_accuracy",
        "validation_accuracy": primary["validation_accuracy"],
        "tie_rule": f"highest validation accuracy; ties follow {list(tie_order)}",
        **provenance,
    }
    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    run_record = {
        **provenance,
        "cv_folds": folds_count,
        "search_iterations_per_model": iterations,
        "scoring": scoring,
        "seed": seed,
        "cv_splits_sha256": cv_digest,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "feature_choice": choice,
        "primary_model": primary_record,
        "models": results,
        "versions": versions,
        "test_features_read": False,
        "feature_importance": "descriptive only; not used for selection",
    }

    # Write results only after all six searches, calibrations, and validation
    # metrics completed successfully.
    _write_json(artifacts_root / "feature_choice.json", choice)
    _write_json(artifacts_root / "primary_model.json", primary_record)
    _write_json(artifacts_root / "versions.json", versions)
    _write_json(artifacts_root / "training_results.json", run_record)
    for key, model in fitted_models.items():
        _write_model(artifacts_root / "models" / f"{key}.joblib", model)

    importance_paths: dict[str, dict[str, str]] = {}
    frequency_result_by_model = {
        result["model"]: fitted_models[result["key"]]
        for result in frequency_results
        if result["model"] in {"rf", "xgb"}
    }
    top_n = _positive_int(
        training.get("feature_importance_top_n"), "training.feature_importance_top_n"
    )
    figure_dpi = _positive_int(
        training.get("feature_importance_dpi"), "training.feature_importance_dpi"
    )
    for model_name, fitted in frequency_result_by_model.items():
        estimator = fitted.named_steps["classifier"]
        importance_paths[model_name] = dict(
            zip(
                ("csv", "figure"),
                _write_importance(
                    reports_root,
                    model_name,
                    np.asarray(estimator.feature_importances_, dtype=np.float64),
                    frequency_names,
                    top_n,
                    figure_dpi,
                ),
            )
        )
    run_record["feature_importance_outputs"] = importance_paths
    _write_json(artifacts_root / "training_results.json", run_record)
    _write_json(reports_root / "phase_6_results.json", run_record)
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "phase_6_results.md").write_text(
        _format_results_markdown(results), encoding="utf-8"
    )
    return run_record


__all__ = [
    "assert_scaler_is_fold_local",
    "choose_frequency_features",
    "load_training_features",
    "make_frequency_matrix",
    "make_pipeline",
    "make_pixel_matrix",
    "make_stratified_folds",
    "run_phase6",
    "search_space",
]

"""Run the committed Phase 7 sealed-test evaluation exactly once."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import statsmodels
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from aidetect.feature_extraction import read_feature_batches
from aidetect.freeze import verify_freeze_lock
from aidetect.io_utils import (
    config_hash,
    project_relative_path,
    resolve_project_path,
    sha256_of_file,
    stable_json_bytes,
    write_json,
)
from aidetect.phase7_config import DEFAULT_PHASE7_CONFIG_NAME
from aidetect.stats_tests import run_hypothesis_tests
from aidetect.train import _validate_batch_metadata, make_frequency_matrix, make_pixel_matrix


_MODEL_ORDER = ("svm", "rf", "xgb")
_FEATURE_SETS = ("frequency", "pixel")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_json_bytes(value)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _project_path(repo_root: Path, value: str | Path) -> Path:
    return resolve_project_path(repo_root, value)


def _check_test_rows(
    frame: pd.DataFrame, split_path: Path, expected_rows: int
) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"id", "label", "split", "group_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Sealed test batches are missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or len(frame) != expected_rows:
        raise ValueError(
            f"Sealed test rows ({len(frame)}) do not match the frozen split count ({expected_rows})"
        )
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Sealed test feature batches contain duplicate IDs")
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError("Sealed test feature batches contain a non-test split assignment")
    labels = frame["label"].to_numpy()
    if not np.isin(labels, (0, 1)).all() or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Sealed test labels must contain both 0=real and 1=AI classes")
    expected = pd.read_csv(split_path, dtype={"id": str, "group_id": str, "split": str})
    expected = expected.loc[expected["split"].eq("test"), ["id", "label", "group_id"]]
    actual = frame[["id", "label", "group_id"]].copy()
    actual["id"] = actual["id"].astype(str)
    expected["id"] = expected["id"].astype(str)
    if len(expected) != len(actual) or set(expected["id"]) != set(actual["id"]):
        raise ValueError("Sealed test feature IDs do not exactly match the frozen test split")
    paired = actual.merge(
        expected,
        on="id",
        suffixes=("_feature", "_split"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Sealed test feature IDs differ from frozen test IDs")
    if not paired["label_feature"].astype(int).equals(paired["label_split"].astype(int)):
        raise ValueError("Sealed test labels differ from the frozen split assignments")
    if not paired["group_id_feature"].astype(str).equals(paired["group_id_split"].astype(str)):
        raise ValueError("Sealed test duplicate-group IDs differ from the frozen split assignments")
    counts = frame.groupby(frame["label"].astype(int)).size().to_dict()
    if counts.get(0) != counts.get(1):
        raise ValueError("The frozen test rows must remain exactly class balanced")
    return frame["label"].to_numpy(dtype=np.int64), expected


def _positive_probabilities(model: Any, values: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(model.predict(values))
    if predicted.ndim != 1 or not np.isin(predicted, (0, 1)).all():
        raise ValueError(f"{key} returned predictions outside the 0/1 label convention")
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", []))
    if probabilities.shape != (len(values), len(classes)) or 1 not in classes:
        raise ValueError(f"{key} must expose a probability column for AI label 1")
    positive = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    if not np.isfinite(positive).all() or np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError(f"{key} returned invalid AI probabilities")
    return predicted.astype(np.int64, copy=False), positive


def _model_metrics(y_true: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    correct = int(np.count_nonzero(y_true == predicted))
    n = int(y_true.size)
    exact = binomtest(correct, n=n, p=0.5)
    interval = exact.proportion_ci(confidence_level=0.95, method="exact")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=[1],
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "correct": correct,
        "n": n,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "accuracy_ci95_exact": {"low": float(interval.low), "high": float(interval.high)},
        "precision_positive_ai": float(precision),
        "recall_positive_ai": float(recall),
        "f1_positive_ai": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix_labels_0_real_1_ai": confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).tolist(),
    }


def _format_results_markdown(
    result: dict[str, Any], practical_target_path: str = "reports/practical_target.md"
) -> str:
    tests = result["hypothesis_tests"]
    h1 = tests["H1"]
    h1b = tests["H1b"]
    q_p_text = "undefined" if h1b["p_value"] is None else f"{h1b['p_value']:.8g}"
    q_stat_text = "undefined" if h1b["statistic"] is None else f"{h1b['statistic']:.8g}"
    lines = [
        "# Phase 7 held-out test results",
        "",
        "This is the single preregistered evaluation of the frozen test split. The test set was not used for training, feature choice, or model selection.",
        "",
        f"- Evaluation started (UTC): `{result['evaluation_started_at_utc']}`",
        f"- Freeze lock SHA-256: `{result['freeze_lock_sha256']}`",
        f"- Frozen test rows: {result['test_rows']}",
        f"- Primary model selected on validation: `{result['primary_model']}`",
        f"- Test prediction file: `{result['predictions_path']}`",
        "",
        "## Metrics",
        "",
        "Accuracy intervals are exact Clopper-Pearson 95% confidence intervals. Positive label is AI-generated (1).",
        "",
        "| Feature set | Model | Correct/n | Accuracy (95% exact CI) | Precision | Recall | F1 | ROC-AUC | Confusion matrix (true rows, predicted columns; labels 0,1) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in result["models"].values():
        metric = model["metrics"]
        lines.append(
            f"| {model['feature_set']} | {model['model']} | {metric['correct']}/{metric['n']} | {metric['accuracy']:.4f} "
            f"({metric['accuracy_ci95_exact']['low']:.4f}, {metric['accuracy_ci95_exact']['high']:.4f}) | "
            f"{metric['precision_positive_ai']:.4f} | {metric['recall_positive_ai']:.4f} | "
            f"{metric['f1_positive_ai']:.4f} | {metric['roc_auc']:.4f} | "
            f"`{metric['confusion_matrix_labels_0_real_1_ai']}` |"
        )
    lines.extend(
        [
            "",
            "## Statistical tests and preregistered decisions",
            "",
            f"**H1: primary model vs chance**: one-sided exact binomial test of H0 p=0.5, alpha={tests['alpha']:.3g}.",
            "",
            f"- Primary model `{h1['model']}`: k={h1['correct']}, n={h1['n']}, p={h1['p_value']:.8g}.",
            f"- Decision: {h1['decision']}",
            "",
            "One-sided exact binomial results for other models are descriptive only:",
            "",
            "| Model | Correct/n | Accuracy | One-sided exact p (descriptive unless primary) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, value in tests["binomial_tests"].items():
        lines.append(
            f"| {key} | {value['correct']}/{value['n']} | {value['accuracy']:.4f} | {value['p_value']:.8g} |"
        )
    lines.extend(
        [
            "",
            "**H1a: paired frequency vs pixel comparisons**: exact two-sided McNemar tests with Holm correction across the three classifiers.",
            "",
            "| Classifier | Frequency correct | Pixel correct | Frequency-only correct | Pixel-only correct | Exact statistic | Raw p | Holm-adjusted p | Decision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for model, comparison in tests["H1a"]["comparisons"].items():
        lines.append(
            f"| {model} | {comparison['frequency_correct']} | {comparison['pixel_correct']} | "
            f"{comparison['discordant_left_only']} | {comparison['discordant_right_only']} | "
            f"{comparison['statistic']:.4g} | {comparison['p_value']:.8g} | "
            f"{comparison['holm_adjusted_p_value']:.8g} | {comparison['decision']} |"
        )
    lines.extend(
        [
            "",
            f"**H1b: frequency-model comparison**: Cochran's Q, df={h1b['degrees_of_freedom']}, statistic={q_stat_text}, p={q_p_text}.",
            f"- Decision: {h1b['decision']}",
        ]
    )
    if h1b["pairwise_exact_mcnemar_holm"]:
        lines.extend(
            [
                "",
                "Conditional pairwise exact two-sided McNemar tests with Holm correction:",
                "",
                "| Pair | Exact statistic | Raw p | Holm-adjusted p | Decision |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for pair, comparison in h1b["pairwise_exact_mcnemar_holm"].items():
            lines.append(
                f"| {pair} | {comparison['statistic']:.4g} | {comparison['p_value']:.8g} | "
                f"{comparison['holm_adjusted_p_value']:.8g} | {comparison['decision']} |"
            )
    else:
        lines.append("- Conditional pairwise tests were not run because the Cochran's Q gate was not met.")
    practical = result["practical_use"]
    lines.extend(
        [
            "",
            "Other models' binomial p-values are descriptive only. H1a claims frequency superiority only when it has more correct predictions and adjusted p < alpha. H1b pairwise tests are conditional on significant Q.",
            "",
            "## Practical-use target",
            "",
            f"The team-agreed minimum validation accuracy was {practical['minimum_validation_accuracy']:.4f}; its rationale and agreement date are frozen in `{practical_target_path}`.",
            f"The primary model's held-out accuracy was {practical['primary_test_accuracy']:.4f}; its point estimate {'meets' if practical['primary_test_accuracy_meets_target'] else 'does not meet'} that target. Statistical significance and practical usefulness are separate decisions.",
            "",
            "No threshold, model, or analysis choice was changed after opening the test set. Any follow-up analysis must be labeled exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_test_once(
    cfg: dict[str, Any],
    *,
    config_file: str | Path = DEFAULT_PHASE7_CONFIG_NAME,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the committed freeze, then consume sealed test features once."""
    lock, root = verify_freeze_lock(
        cfg,
        config_file=config_file,
        repo_root=repo_root,
        require_committed_lock=True,
    )
    paths_cfg = cfg.get("paths", {})
    artifacts_root = _project_path(root, paths_cfg.get("artifacts_root", "artifacts"))
    reports_root = _project_path(root, paths_cfg.get("reports_root", "reports"))
    features_root = _project_path(root, paths_cfg.get("features_root", "features"))
    data_root = _project_path(root, paths_cfg.get("data_root", "data"))
    lock_path = artifacts_root / "FREEZE.lock"
    marker_path = artifacts_root / "EVALUATION_STARTED.json"
    output_paths = (
        reports_root / "results.json",
        reports_root / "results.md",
        reports_root / "test_predictions.csv",
    )
    existing = [str(path) for path in (marker_path, *output_paths) if path.exists()]
    if existing:
        raise FileExistsError(
            "Phase 7 evaluation is single-use; a marker or output already exists: "
            + ", ".join(existing)
        )

    started = _utc_now()
    marker = {
        "evaluation_started_at_utc": started,
        "freeze_lock_sha256": sha256_of_file(lock_path),
        "git_commit": _git_current_commit(root),
        "state": "started",
        "single_use": True,
    }
    # This exclusive marker is persisted before any test row or feature value
    # is opened. A crash after this point requires a supervisor-led decision;
    # silently rerunning would violate the one-time evaluation protocol.
    _write_json_exclusive(marker_path, marker)

    split_digest = lock["splits_sha256"]
    config_digest = config_hash(cfg)
    _, feature_config_hash = _validate_batch_metadata(
        features_root,
        "test",
        cfg,
        config_digest=config_digest,
        splits_digest=split_digest,
    )
    frame = read_feature_batches(features_root, "test", artifacts_root=artifacts_root)
    labels, _ = _check_test_rows(
        frame, data_root / "splits.csv", int(lock["expected_test_rows"])
    )

    artifacts = json.loads((artifacts_root / "feature_choice.json").read_text(encoding="utf-8"))
    primary_record = json.loads((artifacts_root / "primary_model.json").read_text(encoding="utf-8"))
    training_versions = json.loads((artifacts_root / "versions.json").read_text(encoding="utf-8"))
    training = cfg.get("training", {})
    model_order = tuple(training.get("model_order", _MODEL_ORDER))
    if set(model_order) != set(_MODEL_ORDER) or len(model_order) != 3:
        raise ValueError("Configured model_order must contain svm, rf, and xgb exactly once")
    frequency_values, frequency_names = make_frequency_matrix(
        frame, cfg, r0=int(artifacts["r0"]), k_mad=float(artifacts["k_mad"])
    )
    pixel_values, pixel_names = make_pixel_matrix(frame, cfg)
    if not frequency_names or not pixel_names:
        raise ValueError("Phase 7 feature matrices must contain at least one feature")
    matrices = {"frequency": frequency_values, "pixel": pixel_values}

    model_predictions: dict[str, np.ndarray] = {}
    result_models: dict[str, dict[str, Any]] = {}
    prediction_rows: list[pd.DataFrame] = []
    for feature_set in _FEATURE_SETS:
        for model_name in model_order:
            key = f"{feature_set}_{model_name}"
            model_path = artifacts_root / "models" / f"{key}.joblib"
            model = joblib.load(model_path)
            predicted, positive_probability = _positive_probabilities(
                model, matrices[feature_set], key
            )
            model_predictions[key] = predicted
            metrics = _model_metrics(labels, predicted, positive_probability)
            result_models[key] = {
                "feature_set": feature_set,
                "model": model_name,
                "metrics": metrics,
            }
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "id": frame["id"].astype(str).to_numpy(),
                        "true": labels,
                        "pred": predicted,
                        "prob": positive_probability,
                        "feature_set": feature_set,
                        "model": model_name,
                    }
                )
            )

    alpha = float(cfg.get("alpha", 0.05))
    hypothesis_tests = run_hypothesis_tests(
        labels,
        model_predictions,
        primary_model=str(primary_record["artifact_key"]),
        alpha=alpha,
        model_order=model_order,
    )
    primary_accuracy = float(result_models[primary_record["artifact_key"]]["metrics"]["accuracy"])
    target_accuracy = float(lock["practical_target"]["minimum_validation_accuracy"])
    record = {
        "schema_version": 1,
        "evaluation_started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "freeze_lock_sha256": marker["freeze_lock_sha256"],
        "config_sha256": config_digest,
        "splits_sha256": split_digest,
        "feature_config_sha256": feature_config_hash,
        "test_rows": int(len(frame)),
        "test_features_read": True,
        "evaluation_count": 1,
        "primary_model": str(primary_record["artifact_key"]),
        "models": result_models,
        "hypothesis_tests": hypothesis_tests,
        "practical_use": {
            "minimum_validation_accuracy": target_accuracy,
            "primary_test_accuracy": primary_accuracy,
            "primary_test_accuracy_meets_target": bool(primary_accuracy >= target_accuracy),
            "rationale": lock["practical_target"]["rationale"],
            "agreed_by": lock["practical_target"]["agreed_by"],
            "supervisor": lock["practical_target"]["supervisor"],
            "decision_date": lock["practical_target"]["decision_date"],
        },
        "versions": {
            "training": training_versions,
            "evaluation": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "joblib": joblib.__version__,
            },
        },
        "predictions_path": project_relative_path(root, output_paths[2]),
        "split_file_sha256": sha256_of_file(data_root / "splits.csv"),
        "report_note": "Any analyses after this single evaluation are exploratory.",
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(output_paths[2], index=False, lineterminator="\n")
    _write_json_exclusive(output_paths[0], record)
    markdown = _format_results_markdown(
        record, project_relative_path(root, reports_root / "practical_target.md")
    )
    descriptor = os.open(output_paths[1], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
        stream.flush()
        os.fsync(stream.fileno())
    marker["test_features_read"] = True
    marker["state"] = "completed"
    marker["completed_at_utc"] = record["completed_at_utc"]
    marker["results_sha256"] = sha256_of_file(output_paths[0])
    marker["predictions_sha256"] = sha256_of_file(output_paths[2])
    marker["report_sha256"] = sha256_of_file(output_paths[1])
    write_json(marker_path, marker)
    return record


def _git_current_commit(repo_root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


__all__ = ["evaluate_test_once"]

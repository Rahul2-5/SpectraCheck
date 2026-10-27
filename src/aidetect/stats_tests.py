"""Pre-registered paired and binomial tests for the single Phase 7 evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests


_MODELS = ("svm", "rf", "xgb")


def _as_correctness(values: Sequence[object], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} correctness must be a non-empty one-dimensional array")
    if not np.isin(array, (False, True, 0, 1)).all():
        raise ValueError(f"{name} correctness values must be boolean")
    return array.astype(bool, copy=False)


def exact_mcnemar(
    left_correct: Sequence[object], right_correct: Sequence[object]
) -> dict[str, Any]:
    """Compare paired correctness with an exact two-sided McNemar test.

    Rows represent left correctness (False, True), columns represent right
    correctness (False, True). The off-diagonal cells are discordant pairs.
    """
    left = _as_correctness(left_correct, "left")
    right = _as_correctness(right_correct, "right")
    if left.shape != right.shape:
        raise ValueError("Paired correctness arrays must have equal lengths")
    table = np.zeros((2, 2), dtype=np.int64)
    np.add.at(table, (left.astype(int), right.astype(int)), 1)
    result = mcnemar(table, exact=True)
    return {
        "table_left_rows_right_columns": table.tolist(),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "discordant_left_only": int(table[1, 0]),
        "discordant_right_only": int(table[0, 1]),
    }


def run_hypothesis_tests(
    y_true: Sequence[object],
    predictions: Mapping[str, Sequence[object]],
    *,
    primary_model: str,
    alpha: float = 0.05,
    model_order: Sequence[str] = _MODELS,
) -> dict[str, Any]:
    """Run H1, matched frequency-vs-pixel H1a, and frequency-only H1b.

    The accuracy comparisons use per-image correctness, so each table keeps
    the pairing between models. Only the pre-selected primary frequency model
    receives an inferential binomial decision; other binomial p-values are
    retained as descriptive values.
    """
    y = np.asarray(y_true)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true labels must use the project convention 0=real, 1=AI")
    y = y.astype(np.int64, copy=False)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("The held-out evaluation must contain both binary classes")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    order = tuple(str(name) for name in model_order)
    if len(order) != 3 or set(order) != set(_MODELS):
        raise ValueError("model_order must contain svm, rf, and xgb exactly once")

    expected = {f"{feature_set}_{model}" for feature_set in ("frequency", "pixel") for model in order}
    if set(predictions) != expected:
        raise ValueError("predictions must contain the six configured frequency/pixel model keys")
    if primary_model not in {f"frequency_{name}" for name in order}:
        raise ValueError("primary_model must identify one of the frequency models")

    correctness: dict[str, np.ndarray] = {}
    binomial_results: dict[str, dict[str, Any]] = {}
    for key in sorted(expected):
        predicted = np.asarray(predictions[key])
        if predicted.ndim != 1 or predicted.shape != y.shape:
            raise ValueError(f"{key} predictions must have the same row count as y_true")
        if not np.isin(predicted, (0, 1)).all():
            raise ValueError(f"{key} predictions must contain only labels 0 and 1")
        correct = predicted.astype(np.int64, copy=False) == y
        correctness[key] = correct
        successes = int(correct.sum())
        test = binomtest(successes, n=int(y.size), p=0.5, alternative="greater")
        binomial_results[key] = {
            "correct": successes,
            "n": int(y.size),
            "accuracy": successes / int(y.size),
            "null_probability": 0.5,
            "alternative": "greater",
            "p_value": float(test.pvalue),
            "role": "primary H1" if key == primary_model else "descriptive only",
        }

    primary = binomial_results[primary_model]
    h1_reject = primary["p_value"] < alpha
    h1 = {
        **primary,
        "model": primary_model,
        "alpha": float(alpha),
        "reject_null": bool(h1_reject),
        "decision": (
            "Reject H0: primary-model accuracy exceeds chance at the preregistered alpha."
            if h1_reject
            else "Do not reject H0: the data do not establish primary-model accuracy above chance at the preregistered alpha."
        ),
        "other_model_binomial_results": {
            key: value for key, value in binomial_results.items() if key != primary_model
        },
    }

    h1a_raw: dict[str, dict[str, Any]] = {}
    h1a_keys: list[str] = []
    h1a_p_values: list[float] = []
    for model in order:
        frequency_key = f"frequency_{model}"
        pixel_key = f"pixel_{model}"
        comparison = exact_mcnemar(correctness[frequency_key], correctness[pixel_key])
        frequency_correct = int(correctness[frequency_key].sum())
        pixel_correct = int(correctness[pixel_key].sum())
        h1a_raw[model] = {
            "frequency_model": frequency_key,
            "pixel_model": pixel_key,
            "frequency_correct": frequency_correct,
            "pixel_correct": pixel_correct,
            "frequency_has_more_correct": frequency_correct > pixel_correct,
            **comparison,
        }
        h1a_keys.append(model)
        h1a_p_values.append(comparison["p_value"])
    h1a_reject, h1a_adjusted, _, _ = multipletests(
        h1a_p_values, alpha=alpha, method="holm"
    )
    h1a: dict[str, Any] = {
        "method": "exact two-sided McNemar; Holm adjustment across three classifiers",
        "alpha": float(alpha),
        "comparisons": {},
    }
    for index, model in enumerate(h1a_keys):
        comparison = h1a_raw[model]
        adjusted_significant = bool(h1a_reject[index])
        superiority = bool(comparison["frequency_has_more_correct"] and adjusted_significant)
        h1a["comparisons"][model] = {
            **comparison,
            "holm_adjusted_p_value": float(h1a_adjusted[index]),
            "reject_equal_marginal_accuracy": adjusted_significant,
            "claim_frequency_superiority": superiority,
            "decision": (
                "Frequency has more correct predictions and the Holm-adjusted exact paired test is significant."
                if superiority
                else "No preregistered claim of frequency superiority."
            ),
        }

    frequency_matrix = np.column_stack(
        [correctness[f"frequency_{model}"] for model in order]
    ).astype(np.int64)
    if np.all(frequency_matrix == frequency_matrix[:, [0]]):
        # The null is exact when all three correctness vectors are identical;
        # Statsmodels returns NaN for the zero-denominator formula in this case.
        q_statistic, q_p_value, q_df, q_defined = 0.0, 1.0, 2, True
    else:
        q_result = cochrans_q(frequency_matrix)
        q_statistic = float(q_result.statistic)
        q_p_value = float(q_result.pvalue)
        q_df = int(q_result.df)
        q_defined = bool(np.isfinite(q_statistic) and np.isfinite(q_p_value))
        if not q_defined:
            q_statistic, q_p_value = None, None
    q_significant = bool(q_defined and q_p_value < alpha)
    h1b: dict[str, Any] = {
        "method": "Cochran's Q on per-image correctness for the three frequency models",
        "models": [f"frequency_{model}" for model in order],
        "statistic": q_statistic,
        "degrees_of_freedom": q_df,
        "p_value": q_p_value,
        "test_defined": q_defined,
        "alpha": float(alpha),
        "reject_equal_accuracy": bool(q_significant),
        "decision": (
            "Reject equal accuracy across frequency models; conditional pairwise exact McNemar tests follow."
            if q_significant
            else (
                "Cochran's Q was undefined for this correctness pattern; no decision or pairwise test was made."
                if not q_defined
                else "Do not reject equal accuracy across frequency models; no conditional pairwise tests were run."
            )
        ),
        "pairwise_exact_mcnemar_holm": {},
    }
    if q_significant:
        pair_names: list[str] = []
        pair_results: dict[str, dict[str, Any]] = {}
        pair_p_values: list[float] = []
        for left_index in range(len(order)):
            for right_index in range(left_index + 1, len(order)):
                left_model, right_model = order[left_index], order[right_index]
                pair_name = f"frequency_{left_model}_vs_frequency_{right_model}"
                pair = exact_mcnemar(
                    correctness[f"frequency_{left_model}"],
                    correctness[f"frequency_{right_model}"],
                )
                pair_results[pair_name] = {
                    "left_model": f"frequency_{left_model}",
                    "right_model": f"frequency_{right_model}",
                    "left_correct": int(correctness[f"frequency_{left_model}"].sum()),
                    "right_correct": int(correctness[f"frequency_{right_model}"].sum()),
                    **pair,
                }
                pair_names.append(pair_name)
                pair_p_values.append(pair["p_value"])
        pair_reject, pair_adjusted, _, _ = multipletests(
            pair_p_values, alpha=alpha, method="holm"
        )
        for index, pair_name in enumerate(pair_names):
            pair_results[pair_name]["holm_adjusted_p_value"] = float(pair_adjusted[index])
            pair_results[pair_name]["reject_equal_marginal_accuracy"] = bool(pair_reject[index])
            pair_results[pair_name]["decision"] = (
                "Reject equal marginal accuracy for this pair after Holm correction."
                if pair_reject[index]
                else "Do not reject equal marginal accuracy for this pair after Holm correction."
            )
        h1b["pairwise_exact_mcnemar_holm"] = pair_results

    return {"alpha": float(alpha), "binomial_tests": binomial_results, "H1": h1, "H1a": h1a, "H1b": h1b}


__all__ = ["exact_mcnemar", "run_hypothesis_tests"]

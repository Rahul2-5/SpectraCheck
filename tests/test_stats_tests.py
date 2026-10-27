import numpy as np
import pytest
from scipy.stats import binomtest

from aidetect.stats_tests import exact_mcnemar, run_hypothesis_tests


def _six_predictions(y_true, *, correct=None):
    y = np.asarray(y_true, dtype=np.int64)
    correct = y.copy() if correct is None else np.asarray(correct, dtype=np.int64)
    return {
        f"{feature_set}_{model}": correct.copy()
        for feature_set in ("frequency", "pixel")
        for model in ("svm", "rf", "xgb")
    }


def test_primary_binomial_matches_scipy_exact_one_sided_reference():
    y_true = np.tile([0, 1], 750)
    predictions = _six_predictions(y_true)
    predictions["frequency_svm"][:783] = y_true[:783]
    # Mark exactly 783 rows as correct without using any test dataset.
    predictions["frequency_svm"][:] = 1 - y_true
    predictions["frequency_svm"][:783] = y_true[:783]

    result = run_hypothesis_tests(
        y_true,
        predictions,
        primary_model="frequency_svm",
        alpha=0.05,
    )

    expected = binomtest(783, n=1500, p=0.5, alternative="greater").pvalue
    assert result["H1"]["correct"] == 783
    assert result["H1"]["p_value"] == pytest.approx(expected)
    assert result["H1"]["reject_null"] is True
    assert result["binomial_tests"]["pixel_svm"]["role"] == "descriptive only"


def test_exact_mcnemar_identical_correctness_has_p_one():
    correctness = [True, False, True, False, True]

    result = exact_mcnemar(correctness, correctness)

    assert result["table_left_rows_right_columns"] == [[2, 0], [0, 3]]
    assert result["p_value"] == 1.0


def test_h1a_claim_requires_more_frequency_correct_and_holm_significance():
    y_true = np.tile([0, 1], 20)
    predictions = _six_predictions(y_true)
    for model in ("svm", "rf", "xgb"):
        predictions[f"pixel_{model}"] = 1 - y_true
        predictions[f"frequency_{model}"] = 1 - y_true
        predictions[f"frequency_{model}"][:16] = y_true[:16]

    result = run_hypothesis_tests(
        y_true,
        predictions,
        primary_model="frequency_svm",
        alpha=0.05,
    )

    for comparison in result["H1a"]["comparisons"].values():
        assert comparison["frequency_correct"] > comparison["pixel_correct"]
        assert comparison["holm_adjusted_p_value"] < 0.05
        assert comparison["claim_frequency_superiority"] is True


def test_h1b_runs_pairwise_tests_only_after_significant_cochrans_q():
    y_true = np.tile([0, 1], 30)
    predictions = _six_predictions(y_true)
    predictions["frequency_svm"] = y_true.copy()
    predictions["frequency_rf"] = y_true.copy()
    predictions["frequency_rf"][:12] = 1 - y_true[:12]
    predictions["frequency_xgb"] = 1 - y_true

    result = run_hypothesis_tests(
        y_true,
        predictions,
        primary_model="frequency_svm",
        alpha=0.05,
    )

    assert result["H1b"]["test_defined"] is True
    assert result["H1b"]["reject_equal_accuracy"] is True
    assert len(result["H1b"]["pairwise_exact_mcnemar_holm"]) == 3


def test_identical_frequency_correctness_yields_defined_non_significant_q():
    y_true = np.tile([0, 1], 10)
    result = run_hypothesis_tests(
        y_true,
        _six_predictions(y_true),
        primary_model="frequency_svm",
    )

    assert result["H1b"]["test_defined"] is True
    assert result["H1b"]["statistic"] == 0.0
    assert result["H1b"]["p_value"] == 1.0
    assert result["H1b"]["pairwise_exact_mcnemar_holm"] == {}


def test_h1b_handles_equal_frequency_margins_with_constant_row_sums():
    y_true = np.tile([0, 1], 5)[:9]
    predictions = _six_predictions(y_true)
    correct_masks = {
        "svm": np.asarray([index % 3 == 0 for index in range(len(y_true))]),
        "rf": np.asarray([index % 3 == 1 for index in range(len(y_true))]),
        "xgb": np.asarray([index % 3 == 2 for index in range(len(y_true))]),
    }
    for model, correct in correct_masks.items():
        predictions[f"frequency_{model}"] = np.where(correct, y_true, 1 - y_true)

    result = run_hypothesis_tests(
        y_true,
        predictions,
        primary_model="frequency_svm",
    )

    assert result["H1b"]["test_defined"] is True
    assert result["H1b"]["statistic"] == 0.0
    assert result["H1b"]["p_value"] == 1.0
    assert result["H1b"]["reject_equal_accuracy"] is False
    assert result["H1b"]["pairwise_exact_mcnemar_holm"] == {}


@pytest.mark.parametrize(
    "y_true,predictions,kwargs",
    [
        ([0, 1], {}, {"primary_model": "frequency_svm"}),
        ([0, 2], _six_predictions([0, 2]), {"primary_model": "frequency_svm"}),
        ([0, 1], _six_predictions([0, 1]), {"primary_model": "pixel_svm"}),
        ([0, 1], _six_predictions([0, 1]), {"primary_model": "frequency_svm", "alpha": 1.0}),
    ],
)
def test_rejects_invalid_test_inputs(y_true, predictions, kwargs):
    with pytest.raises(ValueError):
        run_hypothesis_tests(y_true, predictions, **kwargs)

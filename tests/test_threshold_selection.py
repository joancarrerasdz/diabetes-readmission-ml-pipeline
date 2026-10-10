import numpy as np
import pandas as pd

from diabetes_readmission.threshold_selection import (
    build_summary,
    calculate_threshold_metrics,
    candidate_thresholds,
    select_threshold,
)


def test_threshold_grid_is_frozen():
    thresholds = (
        candidate_thresholds()
    )

    assert len(thresholds) == 99

    assert (
        thresholds[0]
        == 0.010
    )

    assert (
        thresholds[-1]
        == 0.500
    )

    assert np.allclose(
        np.diff(
            thresholds
        ),
        0.005,
    )


def test_binary_metrics_are_correct():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probability = np.array(
        [0.1, 0.8, 0.7, 0.2]
    )

    result = (
        calculate_threshold_metrics(
            y_true=y_true,
            probability=probability,
            threshold=0.5,
        )
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["fn"] == 1

    assert (
        result["sensitivity"]
        == 0.5
    )

    assert (
        result["specificity"]
        == 0.5
    )

    assert (
        result["precision"]
        == 0.5
    )

    assert (
        result["f1"]
        == 0.5
    )

    assert (
        result[
            "balanced_accuracy"
        ]
        == 0.5
    )

    assert (
        result[
            "predicted_positive_rate"
        ]
        == 0.5
    )


def test_summary_uses_five_folds():
    rows = []

    for fold in range(
        1,
        6,
    ):
        rows.append(
            {
                "outer_fold": fold,
                "threshold": 0.1,
                "sensitivity": 0.7,
                "specificity": 0.8,
                "precision": 0.3,
                "f1": 0.42,
                "balanced_accuracy": 0.75,
                "predicted_positive_rate": 0.25,
            }
        )

    summary = build_summary(
        pd.DataFrame(
            rows
        )
    )

    assert len(summary) == 1

    assert (
        summary.loc[
            0,
            "n_folds",
        ]
        == 5
    )

    assert np.isclose(
        summary.loc[
            0,
            "mean_balanced_accuracy",
        ],
        0.75,
    )


def test_selection_prioritises_balanced_accuracy():
    summary = pd.DataFrame(
        {
            "threshold": [
                0.1,
                0.2,
            ],
            "mean_balanced_accuracy": [
                0.60,
                0.61,
            ],
            "mean_sensitivity": [
                0.90,
                0.50,
            ],
            "mean_f1": [
                0.70,
                0.40,
            ],
        }
    )

    selected = (
        select_threshold(
            summary
        )
    )

    assert (
        selected[
            "threshold"
        ]
        == 0.2
    )


def test_selection_tie_breaks_deterministically():
    summary = pd.DataFrame(
        {
            "threshold": [
                0.10,
                0.15,
                0.20,
            ],
            "mean_balanced_accuracy": [
                0.60,
                0.60,
                0.60,
            ],
            "mean_sensitivity": [
                0.70,
                0.70,
                0.65,
            ],
            "mean_f1": [
                0.40,
                0.40,
                0.50,
            ],
        }
    )

    selected = (
        select_threshold(
            summary
        )
    )

    assert (
        selected[
            "threshold"
        ]
        == 0.10
    )

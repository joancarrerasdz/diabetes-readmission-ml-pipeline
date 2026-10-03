"""Quality gates for baseline OOF diagnostics."""

import pandas as pd

from diabetes_readmission.baseline_diagnostics import (
    calibration_summary,
    probability_summary,
    threshold_sweep,
)


def make_example_oof():
    return pd.DataFrame(
        {
            "readmitted_30d": [
                0, 0, 0, 0,
                1, 1, 1, 1,
            ],
            "logistic_regression_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.15,
                0.25,
                0.50,
                0.80,
            ],
        }
    )


def test_probability_summary_preserves_counts():
    frame = make_example_oof()

    summary = probability_summary(
        frame
    )

    counts = dict(
        zip(
            summary["group"],
            summary["n"],
        )
    )

    assert counts["all"] == 8
    assert counts["negative_30d"] == 4
    assert counts["positive_30d"] == 4


def test_threshold_sweep_metrics_are_bounded():
    frame = make_example_oof()

    result = threshold_sweep(
        frame
    )

    metrics = [
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]

    for metric in metrics:
        assert result[
            metric
        ].between(0, 1).all()


def test_calibration_summary_is_valid():
    frame = make_example_oof()

    result = calibration_summary(
        frame
    )

    assert len(result) > 0

    assert result[
        "mean_predicted_probability"
    ].between(0, 1).all()

    assert result[
        "observed_event_rate"
    ].between(0, 1).all()

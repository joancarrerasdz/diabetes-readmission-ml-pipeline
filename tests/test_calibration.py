"""Quality gates for Week-6 probability calibration."""

import numpy as np
import pandas as pd

from diabetes_readmission.calibration import (
    build_fold_metrics,
    build_summary,
    crossfit_calibration,
    expected_calibration_error,
    select_strategy,
)


def make_example_frame():
    rows = []

    rng = np.random.default_rng(42)

    patient = 1

    for fold in range(1, 6):
        for _ in range(80):
            probability = rng.uniform(
                0.02,
                0.35,
            )

            target = int(
                rng.random()
                < probability
            )

            rows.append(
                {
                    "patient_nbr": patient,
                    "outer_fold": fold,
                    "readmitted_30d": target,
                    "random_forest_probability": probability,
                }
            )

            patient += 1

    return pd.DataFrame(rows)


def test_crossfit_calibration_is_complete():
    frame = make_example_frame()

    result = crossfit_calibration(
        frame
    )

    for column in [
        "uncalibrated_probability",
        "sigmoid_probability",
        "isotonic_probability",
    ]:
        assert result[
            column
        ].notna().all()

        assert result[
            column
        ].between(
            0,
            1,
        ).all()


def test_fold_metrics_cover_all_methods_and_folds():
    frame = crossfit_calibration(
        make_example_frame()
    )

    metrics = build_fold_metrics(
        frame
    )

    assert set(
        metrics["method"]
    ) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }

    assert set(
        metrics["outer_fold"]
    ) == {
        1,
        2,
        3,
        4,
        5,
    }

    assert len(metrics) == 15


def test_calibration_summary_is_valid():
    frame = crossfit_calibration(
        make_example_frame()
    )

    fold_metrics = build_fold_metrics(
        frame
    )

    summary = build_summary(
        frame,
        fold_metrics,
    )

    assert len(summary) == 3

    assert summary[
        "brier_score"
    ].between(
        0,
        1,
    ).all()

    assert summary[
        "pr_auc"
    ].between(
        0,
        1,
    ).all()

    assert summary[
        "roc_auc"
    ].between(
        0,
        1,
    ).all()


def test_ece_is_bounded():
    target = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    ece = expected_calibration_error(
        target,
        probabilities,
    )

    assert 0 <= ece <= 1


def test_selection_returns_known_strategy():
    frame = crossfit_calibration(
        make_example_frame()
    )

    fold_metrics = build_fold_metrics(
        frame
    )

    summary = build_summary(
        frame,
        fold_metrics,
    )

    strategy, _ = select_strategy(
        summary,
        fold_metrics,
    )

    assert strategy in {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }

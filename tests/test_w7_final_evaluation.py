"""Tests for the protected Week-7 final held-out evaluator."""

from __future__ import annotations

import json

import numpy as np
import pytest

import diabetes_readmission.w7_final_evaluation as w7


def test_locked_estimator_matches_frozen_w6_specification():
    """The final estimator must exactly match the frozen W6 lock."""
    estimator = w7.build_locked_estimator()

    assert estimator.n_estimators == 300
    assert estimator.max_depth == 12
    assert estimator.min_samples_leaf == 5
    assert estimator.class_weight is None
    assert estimator.random_state == 42

    assert w7.LOCKED_THRESHOLD == 0.110


def test_final_evaluation_is_blocked_without_confirmation(
    monkeypatch,
    tmp_path,
):
    """
    The held-out evaluator must refuse accidental execution.

    This test exits before any held-out data are loaded.
    """
    monkeypatch.delenv(
        w7.CONFIRMATION_VARIABLE,
        raising=False,
    )

    monkeypatch.setattr(
        w7,
        "RUN_STATE_PATH",
        tmp_path
        / "w7_final_evaluation_run_state.json",
    )

    with pytest.raises(
        RuntimeError,
        match="FINAL HELD-OUT EVALUATION BLOCKED",
    ):
        w7.validate_confirmation()


def test_existing_run_state_blocks_repeat_execution(
    monkeypatch,
    tmp_path,
):
    """A previous run-state artifact must prevent a second evaluation."""
    run_state = (
        tmp_path
        / "w7_final_evaluation_run_state.json"
    )

    run_state.write_text(
        json.dumps(
            {
                "week": 7,
                "stage": "final_held_out_evaluation",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        w7.CONFIRMATION_VARIABLE,
        w7.CONFIRMATION_TOKEN,
    )

    monkeypatch.setattr(
        w7,
        "RUN_STATE_PATH",
        run_state,
    )

    with pytest.raises(
        RuntimeError,
        match="run-state already exists",
    ):
        w7.validate_confirmation()


def test_final_metric_calculation_uses_locked_threshold():
    """Threshold-dependent metrics must use the frozen 0.110 cutoff."""
    target = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=int,
    )

    probability = np.array(
        [
            0.02,
            0.08,
            0.20,
            0.15,
            0.40,
            0.80,
        ],
        dtype=float,
    )

    metrics = w7.calculate_final_metrics(
        target,
        probability,
    )

    assert metrics[
        "operating_threshold"
    ] == 0.110

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tp"] == 3

    assert metrics[
        "sensitivity"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "specificity"
    ] == pytest.approx(
        2 / 3
    )

    assert metrics[
        "predicted_positive_rate"
    ] == pytest.approx(
        4 / 6
    )

    for metric in [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "log_loss",
        "balanced_accuracy",
    ]:
        assert np.isfinite(
            metrics[metric]
        )


def test_calibration_table_preserves_all_observations():
    """Calibration aggregation must account for every prediction."""
    target = np.array(
        [
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
        ],
        dtype=int,
    )

    probability = np.array(
        [
            0.02,
            0.08,
            0.12,
            0.18,
            0.25,
            0.42,
            0.61,
            0.88,
        ],
        dtype=float,
    )

    table = w7.build_calibration_table(
        target,
        probability,
        n_bins=10,
    )

    assert table["n"].sum() == len(
        target
    )

    assert table[
        "mean_predicted_probability"
    ].between(
        0.0,
        1.0,
    ).all()

    assert table[
        "observed_event_rate"
    ].between(
        0.0,
        1.0,
    ).all()

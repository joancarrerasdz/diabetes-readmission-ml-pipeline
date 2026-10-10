"""Quality gates for the frozen Week-6 candidate-model decision."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCK_PATH = (
    PROJECT_ROOT
    / "reports"
    / "w6_selected_model.json"
)

DECISION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "w6_model_selection_decision.md"
)


def load_lock():
    with LOCK_PATH.open(
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def test_week6_candidate_model_is_locked():
    lock = load_lock()

    assert (
        lock["selection_status"]
        == "candidate_model_locked"
    )

    assert (
        lock["selected_model"]
        == "random_forest"
    )

    assert (
        lock["estimator"]
        == "RandomForestClassifier"
    )


def test_locked_random_forest_parameters():
    lock = load_lock()

    expected = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": None,
        "random_state": 42,
    }

    assert (
        lock["hyperparameters"]
        == expected
    )


def test_test_firewall_remains_closed():
    lock = load_lock()

    assert (
        lock["held_out_test_accessed"]
        is False
    )

    assert (
        lock["threshold_locked"]
        is False
    )

    assert (
        lock[
            "calibration_strategy_locked"
        ]
        is False
    )


def test_selection_decision_is_versioned():
    text = DECISION_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "Random Forest"
        in text
    )

    assert (
        "one-standard-error"
        in text
    )

    assert (
        "held-out test set has not been accessed"
        in text.lower()
    )

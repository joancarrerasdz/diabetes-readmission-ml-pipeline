"""Quality gates for patient-safe baseline modelling."""

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)

from diabetes_readmission.modelling import (
    DEFAULT_THRESHOLD,
    calculate_binary_metrics,
    build_baseline_pipeline,
    create_grouped_cv_splits,
    validate_grouped_cv_splits,
)


def test_grouped_cv_has_zero_patient_overlap():
    groups = pd.Series(
        np.repeat(
            np.arange(30),
            2,
        )
    )

    y = pd.Series(
        np.repeat(
            np.arange(30) % 2,
            2,
        )
    )

    splits = create_grouped_cv_splits(
        y=y,
        groups=groups,
        n_splits=5,
        random_state=42,
    )

    validate_grouped_cv_splits(
        splits=splits,
        groups=groups,
        n_rows=len(y),
    )

    assert len(splits) == 5

    for train_index, validation_index in splits:
        train_patients = set(
            groups.iloc[train_index]
        )

        validation_patients = set(
            groups.iloc[validation_index]
        )

        assert train_patients.isdisjoint(
            validation_patients
        )


def test_every_encounter_is_validation_once():
    groups = pd.Series(
        np.repeat(
            np.arange(25),
            2,
        )
    )

    y = pd.Series(
        np.repeat(
            np.arange(25) % 2,
            2,
        )
    )

    splits = create_grouped_cv_splits(
        y=y,
        groups=groups,
        n_splits=5,
        random_state=42,
    )

    validation_counts = np.zeros(
        len(y),
        dtype=int,
    )

    for _, validation_index in splits:
        validation_counts[
            validation_index
        ] += 1

    assert np.all(
        validation_counts == 1
    )


def test_binary_metrics_at_fixed_threshold():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.40, 0.35, 0.80]
    )

    metrics = calculate_binary_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.50,
    )

    assert metrics["roc_auc"] == 0.75
    assert metrics["sensitivity"] == 0.50
    assert metrics["specificity"] == 1.00
    assert metrics["precision"] == 1.00
    assert metrics["balanced_accuracy"] == 0.75


def test_baseline_pipeline_starts_unfitted():
    estimator = LogisticRegression(
        solver="liblinear",
        max_iter=100,
        random_state=42,
    )

    pipeline = build_baseline_pipeline(
        estimator
    )

    preprocessor = pipeline.named_steps[
        "preprocessing"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    assert not hasattr(
        preprocessor,
        "transformers_",
    )

    assert not hasattr(
        classifier,
        "classes_",
    )


def test_default_threshold_is_not_optimized():
    assert DEFAULT_THRESHOLD == 0.50

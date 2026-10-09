"""Quality gates for Week-6 nested model selection."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from diabetes_readmission.model_selection import (
    INNER_N_SPLITS,
    OUTER_N_SPLITS,
    PRIMARY_SCORING,
    add_selection_statistics,
    create_inner_splits,
    get_candidate_searches,
)


def make_grouped_example() -> tuple[
    pd.Series,
    pd.Series,
]:
    """Create a balanced synthetic patient-grouped target."""

    patients = np.arange(30)

    groups = pd.Series(
        np.repeat(
            patients,
            2,
        )
    )

    patient_outcome = (
        patients % 2
    ).astype(int)

    y = pd.Series(
        np.repeat(
            patient_outcome,
            2,
        )
    )

    return y, groups


def test_week6_validation_constants_are_frozen():
    assert OUTER_N_SPLITS == 5
    assert INNER_N_SPLITS == 3
    assert PRIMARY_SCORING == "average_precision"


def test_only_protocol_candidate_families_are_used():
    candidates = get_candidate_searches()

    assert set(
        candidates
    ) == {
        "logistic_regression",
        "random_forest",
    }

    logistic, _ = candidates[
        "logistic_regression"
    ]

    random_forest, _ = candidates[
        "random_forest"
    ]

    assert isinstance(
        logistic,
        LogisticRegression,
    )

    assert isinstance(
        random_forest,
        RandomForestClassifier,
    )


def test_logistic_search_space_matches_protocol():
    candidates = get_candidate_searches()

    logistic, grid = candidates[
        "logistic_regression"
    ]

    assert logistic.solver == "liblinear"
    assert logistic.max_iter == 2000
    assert logistic.random_state == 42

    assert grid[
        "classifier__penalty"
    ] == ["l2"]

    assert grid[
        "classifier__C"
    ] == [
        0.1,
        1.0,
        10.0,
    ]

    assert grid[
        "classifier__class_weight"
    ] == [
        None,
        "balanced",
    ]

    combinations = np.prod(
        [
            len(values)
            for values
            in grid.values()
        ]
    )

    assert combinations == 6


def test_random_forest_search_space_matches_protocol():
    candidates = get_candidate_searches()

    random_forest, grid = candidates[
        "random_forest"
    ]

    assert random_forest.n_estimators == 300
    assert random_forest.random_state == 42
    assert random_forest.n_jobs == -1

    assert grid[
        "classifier__n_estimators"
    ] == [300]

    assert grid[
        "classifier__max_depth"
    ] == [
        None,
        12,
    ]

    assert grid[
        "classifier__min_samples_leaf"
    ] == [
        1,
        5,
    ]

    assert grid[
        "classifier__class_weight"
    ] == [
        None,
        "balanced_subsample",
    ]

    combinations = np.prod(
        [
            len(values)
            for values
            in grid.values()
        ]
    )

    assert combinations == 8


def test_inner_cv_is_patient_grouped_and_complete():
    y, groups = make_grouped_example()

    splits = create_inner_splits(
        y=y,
        groups=groups,
    )

    assert len(splits) == INNER_N_SPLITS

    validation_counts = np.zeros(
        len(y),
        dtype=int,
    )

    for (
        train_index,
        validation_index,
    ) in splits:
        train_patients = set(
            groups.iloc[
                train_index
            ]
        )

        validation_patients = set(
            groups.iloc[
                validation_index
            ]
        )

        assert train_patients.isdisjoint(
            validation_patients
        )

        validation_counts[
            validation_index
        ] += 1

    assert np.all(
        validation_counts == 1
    )


def test_inner_cv_keeps_each_patient_in_one_validation_fold():
    y, groups = make_grouped_example()

    splits = create_inner_splits(
        y=y,
        groups=groups,
    )

    fold_assignment = np.zeros(
        len(y),
        dtype=int,
    )

    for fold, (
        _,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        fold_assignment[
            validation_index
        ] = fold

    audit = pd.DataFrame(
        {
            "patient_nbr": groups,
            "fold": fold_assignment,
        }
    )

    maximum_folds_per_patient = (
        audit
        .groupby(
            "patient_nbr"
        )[
            "fold"
        ]
        .nunique()
        .max()
    )

    assert (
        maximum_folds_per_patient
        == 1
    )


def test_selection_summary_adds_fold_standard_error():
    summary = pd.DataFrame(
        {
            "model": [
                "logistic_regression",
                "random_forest",
            ],
            "fold_sd_pr_auc": [
                0.010,
                0.020,
            ],
        }
    )

    result = (
        add_selection_statistics(
            summary
        )
    )

    expected = np.array(
        [
            0.010
            / np.sqrt(
                OUTER_N_SPLITS
            ),
            0.020
            / np.sqrt(
                OUTER_N_SPLITS
            ),
        ]
    )

    assert np.allclose(
        result[
            "fold_se_pr_auc"
        ].to_numpy(),
        expected,
    )

    assert (
        result[
            "primary_selection_metric"
        ]
        == "PR-AUC"
    ).all()

"""Nested patient-grouped model selection for 30-day readmission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from diabetes_readmission.modelling import (
    DEFAULT_THRESHOLD,
    ENGINEERED_TRAIN_PATH,
    GROUP_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_baseline_pipeline,
    build_model_summary,
    calculate_binary_metrics,
    create_grouped_cv_splits,
    validate_grouped_cv_splits,
)
from diabetes_readmission.model_preprocessing import (
    MODEL_FEATURES,
    validate_model_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

OOF_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_model_selection_oof_predictions.csv"
)

FOLD_METRICS_PATH = (
    REPORTS_DIR
    / "w6_nested_cv_fold_metrics.csv"
)

INNER_SEARCH_PATH = (
    REPORTS_DIR
    / "w6_inner_search_summary.csv"
)

MODEL_SUMMARY_PATH = (
    REPORTS_DIR
    / "w6_model_selection_summary.csv"
)


OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

PRIMARY_SCORING = "average_precision"


def get_candidate_searches() -> dict[
    str,
    tuple[
        BaseEstimator,
        dict[str, list[Any]],
    ],
]:
    """Return the frozen Week-6 candidate models and search spaces."""

    logistic_regression = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    logistic_grid = {
        "classifier__penalty": ["l2"],
        "classifier__C": [
            0.1,
            1.0,
            10.0,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest_grid = {
        "classifier__n_estimators": [
            300,
        ],
        "classifier__max_depth": [
            None,
            12,
        ],
        "classifier__min_samples_leaf": [
            1,
            5,
        ],
        "classifier__class_weight": [
            None,
            "balanced_subsample",
        ],
    }

    return {
        "logistic_regression": (
            logistic_regression,
            logistic_grid,
        ),
        "random_forest": (
            random_forest,
            random_forest_grid,
        ),
    }


def create_inner_splits(
    y: pd.Series,
    groups: pd.Series,
) -> list[
    tuple[np.ndarray, np.ndarray]
]:
    """Create and validate patient-grouped inner-CV folds."""

    splits = create_grouped_cv_splits(
        y=y,
        groups=groups,
        n_splits=INNER_N_SPLITS,
        random_state=RANDOM_STATE,
    )

    validate_grouped_cv_splits(
        splits=splits,
        groups=groups,
        n_rows=len(y),
    )

    return splits


def build_inner_search_rows(
    search: GridSearchCV,
    model_name: str,
    outer_fold: int,
) -> list[dict[str, Any]]:
    """Convert GridSearchCV results into versionable rows."""

    results = search.cv_results_

    rows: list[dict[str, Any]] = []

    for index, parameters in enumerate(
        results["params"]
    ):
        rows.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "inner_n_splits": INNER_N_SPLITS,
                "primary_metric": PRIMARY_SCORING,
                "mean_inner_pr_auc": float(
                    results[
                        "mean_test_score"
                    ][index]
                ),
                "sd_inner_pr_auc": float(
                    results[
                        "std_test_score"
                    ][index]
                ),
                "rank_inner_pr_auc": int(
                    results[
                        "rank_test_score"
                    ][index]
                ),
                "selected": int(
                    index == search.best_index_
                ),
                "parameters": json.dumps(
                    parameters,
                    sort_keys=True,
                ),
            }
        )

    return rows


def evaluate_candidate_nested_cv(
    dataframe: pd.DataFrame,
    outer_splits: Sequence[
        tuple[np.ndarray, np.ndarray]
    ],
    model_name: str,
    estimator: BaseEstimator,
    parameter_grid: dict[
        str,
        list[Any],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    """Evaluate one candidate using nested patient-grouped CV."""

    x = dataframe[
        list(MODEL_FEATURES)
    ]

    y = dataframe[
        TARGET_COLUMN
    ].astype(int)

    groups = dataframe[
        GROUP_COLUMN
    ]

    oof_probabilities = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    inner_rows: list[
        dict[str, Any]
    ] = []

    for fold, (
        outer_train_index,
        outer_validation_index,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        x_train = (
            x.iloc[
                outer_train_index
            ]
            .reset_index(
                drop=True
            )
        )

        y_train = (
            y.iloc[
                outer_train_index
            ]
            .reset_index(
                drop=True
            )
        )

        groups_train = (
            groups.iloc[
                outer_train_index
            ]
            .reset_index(
                drop=True
            )
        )

        x_validation = x.iloc[
            outer_validation_index
        ]

        y_validation = y.iloc[
            outer_validation_index
        ]

        train_patients = set(
            groups.iloc[
                outer_train_index
            ]
        )

        validation_patients = set(
            groups.iloc[
                outer_validation_index
            ]
        )

        patient_overlap = len(
            train_patients
            & validation_patients
        )

        if patient_overlap:
            raise RuntimeError(
                "Patient leakage detected "
                f"in outer fold {fold}."
            )

        inner_splits = (
            create_inner_splits(
                y=y_train,
                groups=groups_train,
            )
        )

        pipeline = (
            build_baseline_pipeline(
                estimator
            )
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=PRIMARY_SCORING,
            cv=inner_splits,
            refit=True,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )

        print(
            f"\n{model_name}: "
            f"outer fold "
            f"{fold}/{OUTER_N_SPLITS}"
        )

        search.fit(
            x_train,
            y_train,
        )

        probabilities = (
            search.best_estimator_
            .predict_proba(
                x_validation
            )[:, 1]
        )

        oof_probabilities[
            outer_validation_index
        ] = probabilities

        metrics = (
            calculate_binary_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                threshold=DEFAULT_THRESHOLD,
            )
        )

        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "train_encounters": len(
                    outer_train_index
                ),
                "validation_encounters": len(
                    outer_validation_index
                ),
                "train_patients": len(
                    train_patients
                ),
                "validation_patients": len(
                    validation_patients
                ),
                "patient_overlap": (
                    patient_overlap
                ),
                "validation_prevalence_30d": (
                    y_validation.mean()
                ),
                "best_inner_pr_auc": (
                    search.best_score_
                ),
                "best_parameters": (
                    json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    )
                ),
                **metrics,
            }
        )

        inner_rows.extend(
            build_inner_search_rows(
                search=search,
                model_name=model_name,
                outer_fold=fold,
            )
        )

        print(
            "  best inner PR-AUC: "
            f"{search.best_score_:.6f}"
        )

        print(
            "  outer PR-AUC: "
            f"{metrics['pr_auc']:.6f}"
        )

        print(
            "  outer ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            "  outer Brier: "
            f"{metrics['brier_score']:.6f}"
        )

        print(
            "  best parameters: "
            f"{search.best_params_}"
        )

    if np.isnan(
        oof_probabilities
    ).any():
        raise RuntimeError(
            "Missing nested-CV OOF "
            f"probabilities for {model_name}."
        )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            inner_rows
        ),
        oof_probabilities,
    )


def add_selection_statistics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add standard-error statistics used by the selection protocol."""

    result = summary.copy()

    result[
        "fold_se_pr_auc"
    ] = (
        result[
            "fold_sd_pr_auc"
        ]
        / np.sqrt(
            OUTER_N_SPLITS
        )
    )

    result[
        "primary_selection_metric"
    ] = "PR-AUC"

    return result


def main() -> None:
    """Run frozen Week-6 nested model selection."""

    data = pd.read_csv(
        ENGINEERED_TRAIN_PATH,
        low_memory=False,
    )

    if len(data) != 79_473:
        raise RuntimeError(
            "Unexpected training "
            f"encounter count: {len(data):,}"
        )

    if (
        data[
            GROUP_COLUMN
        ].nunique()
        != 55_952
    ):
        raise RuntimeError(
            "Unexpected training "
            "patient count: "
            f"{data[GROUP_COLUMN].nunique():,}"
        )

    if set(
        data[
            TARGET_COLUMN
        ].unique()
    ) != {0, 1}:
        raise RuntimeError(
            "Target must contain "
            "exactly {0, 1}."
        )

    validate_model_schema(
        data
    )

    y = data[
        TARGET_COLUMN
    ].astype(int)

    groups = data[
        GROUP_COLUMN
    ]

    outer_splits = (
        create_grouped_cv_splits(
            y=y,
            groups=groups,
            n_splits=OUTER_N_SPLITS,
            random_state=RANDOM_STATE,
        )
    )

    validate_grouped_cv_splits(
        splits=outer_splits,
        groups=groups,
        n_rows=len(data),
    )

    print(
        "\nW6 NESTED MODEL SELECTION "
        "— TRAINING ONLY"
    )
    print("=" * 72)

    print(
        f"Encounters: {len(data):,}"
    )

    print(
        "Unique patients: "
        f"{groups.nunique():,}"
    )

    print(
        "30-day prevalence: "
        f"{y.mean():.2%}"
    )

    print(
        "Outer grouped folds: "
        f"{OUTER_N_SPLITS}"
    )

    print(
        "Inner grouped folds: "
        f"{INNER_N_SPLITS}"
    )

    print(
        "Primary selection metric: "
        "PR-AUC / Average Precision"
    )

    all_fold_metrics = []
    all_inner_results = []

    oof_probabilities: dict[
        str,
        np.ndarray,
    ] = {}

    candidates = (
        get_candidate_searches()
    )

    for model_name, (
        estimator,
        parameter_grid,
    ) in candidates.items():
        (
            fold_metrics,
            inner_results,
            probabilities,
        ) = (
            evaluate_candidate_nested_cv(
                dataframe=data,
                outer_splits=outer_splits,
                model_name=model_name,
                estimator=estimator,
                parameter_grid=(
                    parameter_grid
                ),
            )
        )

        all_fold_metrics.append(
            fold_metrics
        )

        all_inner_results.append(
            inner_results
        )

        oof_probabilities[
            model_name
        ] = probabilities

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    inner_results = pd.concat(
        all_inner_results,
        ignore_index=True,
    )

    model_summary = (
        build_model_summary(
            y=y,
            fold_metrics=fold_metrics,
            oof_probabilities=(
                oof_probabilities
            ),
        )
    )

    model_summary = (
        add_selection_statistics(
            model_summary
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OOF_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    inner_results.to_csv(
        INNER_SEARCH_PATH,
        index=False,
    )

    model_summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
    )

    oof = data[
        [
            "encounter_id",
            GROUP_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    outer_fold_assignment = np.zeros(
        len(data),
        dtype=int,
    )

    for fold, (
        _,
        validation_index,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        outer_fold_assignment[
            validation_index
        ] = fold

    oof[
        "outer_fold"
    ] = outer_fold_assignment

    for (
        model_name,
        probabilities,
    ) in oof_probabilities.items():
        oof[
            f"{model_name}_probability"
        ] = probabilities

    if (
        oof.groupby(
            GROUP_COLUMN
        )[
            "outer_fold"
        ]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError(
            "A patient was assigned to "
            "multiple outer validation folds."
        )

    oof.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\nNESTED CV MODEL SUMMARY"
    )
    print("=" * 72)

    display_columns = [
        "model",
        "oof_pr_auc",
        "fold_mean_pr_auc",
        "fold_sd_pr_auc",
        "fold_se_pr_auc",
        "oof_roc_auc",
        "fold_mean_roc_auc",
        "oof_brier_score",
        "fold_mean_brier_score",
    ]

    print(
        model_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nVersionable reports saved:"
    )

    print(
        f"- {FOLD_METRICS_PATH}"
    )

    print(
        f"- {INNER_SEARCH_PATH}"
    )

    print(
        f"- {MODEL_SUMMARY_PATH}"
    )

    print(
        "\nLocal OOF predictions saved:"
    )

    print(
        f"- {OOF_PREDICTIONS_PATH}"
    )

    print(
        "\nNo operating threshold "
        "was selected."
    )

    print(
        "The held-out test set "
        "was not accessed."
    )


if __name__ == "__main__":
    main()

"""Patient-safe baseline modelling for 30-day readmission."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from diabetes_readmission.model_preprocessing import (
    MODEL_FEATURES,
    build_model_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENGINEERED_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_engineered.csv"
)

OOF_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "baseline_oof_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

CV_SPLIT_SUMMARY_PATH = (
    REPORTS_DIR
    / "baseline_cv_split_summary.csv"
)

CV_FOLD_METRICS_PATH = (
    REPORTS_DIR
    / "baseline_cv_fold_metrics.csv"
)

CV_SUMMARY_PATH = (
    REPORTS_DIR
    / "baseline_cv_summary.csv"
)


TARGET_COLUMN = "readmitted_30d"
GROUP_COLUMN = "patient_nbr"

N_SPLITS = 5
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def create_grouped_cv_splits(
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reproducible stratified patient-grouped CV splits."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    dummy_x = np.zeros(
        shape=(len(y), 1),
        dtype=float,
    )

    return list(
        splitter.split(
            dummy_x,
            y,
            groups=groups,
        )
    )


def validate_grouped_cv_splits(
    splits: Sequence[
        tuple[np.ndarray, np.ndarray]
    ],
    groups: pd.Series | np.ndarray,
    n_rows: int,
) -> None:
    """Validate zero patient overlap and one validation assignment per row."""
    groups_array = np.asarray(groups)

    validation_counts = np.zeros(
        n_rows,
        dtype=int,
    )

    for fold_index, (
        train_index,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        train_patients = set(
            groups_array[train_index]
        )

        validation_patients = set(
            groups_array[validation_index]
        )

        overlap = (
            train_patients
            & validation_patients
        )

        if overlap:
            raise RuntimeError(
                "Patient leakage detected in "
                f"CV fold {fold_index}: "
                f"{len(overlap)} overlapping patients."
            )

        validation_counts[
            validation_index
        ] += 1

    if not np.all(validation_counts == 1):
        raise RuntimeError(
            "Every training encounter must appear "
            "exactly once in a validation fold."
        )


def build_split_summary(
    y: pd.Series,
    groups: pd.Series,
    splits: Sequence[
        tuple[np.ndarray, np.ndarray]
    ],
) -> pd.DataFrame:
    """Summarize grouped cross-validation folds."""
    rows = []

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        train_y = y.iloc[train_index]
        validation_y = y.iloc[
            validation_index
        ]

        train_groups = groups.iloc[
            train_index
        ]
        validation_groups = groups.iloc[
            validation_index
        ]

        patient_overlap = len(
            set(train_groups)
            & set(validation_groups)
        )

        rows.append(
            {
                "fold": fold,
                "train_encounters": len(
                    train_index
                ),
                "validation_encounters": len(
                    validation_index
                ),
                "train_patients": (
                    train_groups.nunique()
                ),
                "validation_patients": (
                    validation_groups.nunique()
                ),
                "train_prevalence_30d": (
                    train_y.mean()
                ),
                "validation_prevalence_30d": (
                    validation_y.mean()
                ),
                "patient_overlap": (
                    patient_overlap
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Calculate discrimination, calibration and threshold metrics."""
    y_array = np.asarray(
        y_true,
        dtype=int,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probability_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "roc_auc": roc_auc_score(
            y_array,
            probability_array,
        ),
        "pr_auc": average_precision_score(
            y_array,
            probability_array,
        ),
        "brier_score": brier_score_loss(
            y_array,
            probability_array,
        ),
        "sensitivity": recall_score(
            y_array,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "precision": precision_score(
            y_array,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_array,
            predictions,
            zero_division=0,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_array,
                predictions,
            )
        ),
    }


def build_baseline_pipeline(
    estimator: BaseEstimator,
) -> Pipeline:
    """
    Combine fold-safe preprocessing with an estimator.

    No preprocessing component is fitted before Pipeline.fit().
    """
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_model_preprocessor(),
            ),
            (
                "classifier",
                clone(estimator),
            ),
        ]
    )


def get_baseline_estimators() -> dict[
    str,
    BaseEstimator,
]:
    """Return the predefined Week-5 baseline estimators."""
    return {
        "dummy_prior": DummyClassifier(
            strategy="prior",
        ),
        "logistic_regression": (
            LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_model_cv(
    dataframe: pd.DataFrame,
    splits: Sequence[
        tuple[np.ndarray, np.ndarray]
    ],
    model_name: str,
    estimator: BaseEstimator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate patient-safe out-of-fold probabilities and fold metrics."""
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

    rows = []

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        splits,
        start=1,
    ):
        train_patients = set(
            groups.iloc[train_index]
        )

        validation_patients = set(
            groups.iloc[validation_index]
        )

        overlap = len(
            train_patients
            & validation_patients
        )

        if overlap != 0:
            raise RuntimeError(
                f"Patient leakage in fold {fold}."
            )

        pipeline = build_baseline_pipeline(
            estimator
        )

        pipeline.fit(
            x.iloc[train_index],
            y.iloc[train_index],
        )

        probabilities = (
            pipeline.predict_proba(
                x.iloc[validation_index]
            )[:, 1]
        )

        oof_probabilities[
            validation_index
        ] = probabilities

        metrics = calculate_binary_metrics(
            y.iloc[validation_index],
            probabilities,
        )

        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "train_encounters": len(
                    train_index
                ),
                "validation_encounters": len(
                    validation_index
                ),
                "train_patients": len(
                    train_patients
                ),
                "validation_patients": len(
                    validation_patients
                ),
                "patient_overlap": overlap,
                "validation_prevalence_30d": (
                    y.iloc[
                        validation_index
                    ].mean()
                ),
                **metrics,
            }
        )

    if np.isnan(
        oof_probabilities
    ).any():
        raise RuntimeError(
            f"Missing OOF probabilities for "
            f"{model_name}."
        )

    return (
        pd.DataFrame(rows),
        oof_probabilities,
    )


def build_model_summary(
    y: pd.Series,
    fold_metrics: pd.DataFrame,
    oof_probabilities: dict[
        str,
        np.ndarray,
    ],
) -> pd.DataFrame:
    """Create one summary row per baseline model."""
    rows = []

    metric_names = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
    ]

    for (
        model_name,
        probabilities,
    ) in oof_probabilities.items():
        oof_metrics = (
            calculate_binary_metrics(
                y,
                probabilities,
            )
        )

        model_folds = (
            fold_metrics.loc[
                fold_metrics["model"]
                == model_name
            ]
        )

        row = {
            "model": model_name,
            "threshold": (
                DEFAULT_THRESHOLD
            ),
        }

        for metric in metric_names:
            row[
                f"oof_{metric}"
            ] = oof_metrics[metric]

            row[
                f"fold_mean_{metric}"
            ] = model_folds[
                metric
            ].mean()

            row[
                f"fold_sd_{metric}"
            ] = model_folds[
                metric
            ].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run patient-safe CV for the predefined baseline models."""
    data = pd.read_csv(
        ENGINEERED_TRAIN_PATH,
        low_memory=False,
    )

    if len(data) != 79_473:
        raise RuntimeError(
            "Unexpected training encounter count: "
            f"{len(data):,}"
        )

    if data[
        GROUP_COLUMN
    ].nunique() != 55_952:
        raise RuntimeError(
            "Unexpected training patient count: "
            f"{data[GROUP_COLUMN].nunique():,}"
        )

    if set(
        data[TARGET_COLUMN].unique()
    ) != {0, 1}:
        raise RuntimeError(
            "Target must contain exactly "
            "binary values {0, 1}."
        )

    y = data[
        TARGET_COLUMN
    ].astype(int)

    groups = data[
        GROUP_COLUMN
    ]

    splits = create_grouped_cv_splits(
        y=y,
        groups=groups,
    )

    validate_grouped_cv_splits(
        splits=splits,
        groups=groups,
        n_rows=len(data),
    )

    split_summary = build_split_summary(
        y=y,
        groups=groups,
        splits=splits,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary.to_csv(
        CV_SPLIT_SUMMARY_PATH,
        index=False,
    )

    print(
        "\nW5 BASELINE MODELLING — "
        "TRAINING ONLY"
    )

    print("=" * 72)

    print(
        f"Encounters: "
        f"{len(data):,}"
    )

    print(
        f"Unique patients: "
        f"{groups.nunique():,}"
    )

    print(
        f"30-day prevalence: "
        f"{y.mean():.2%}"
    )

    print(
        f"Grouped CV folds: "
        f"{len(splits)}"
    )

    print(
        "\nPatient-grouped CV audit"
    )

    print("-" * 72)

    print(
        split_summary.to_string(
            index=False
        )
    )

    all_fold_metrics = []
    oof_probabilities = {}

    estimators = (
        get_baseline_estimators()
    )

    for (
        model_name,
        estimator,
    ) in estimators.items():
        print(
            f"\nEvaluating: "
            f"{model_name}"
        )

        fold_metrics, probabilities = (
            evaluate_model_cv(
                dataframe=data,
                splits=splits,
                model_name=model_name,
                estimator=estimator,
            )
        )

        all_fold_metrics.append(
            fold_metrics
        )

        oof_probabilities[
            model_name
        ] = probabilities

        print(
            fold_metrics[
                [
                    "fold",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                    "sensitivity",
                    "specificity",
                ]
            ].to_string(
                index=False
            )
        )

    fold_metrics = pd.concat(
        all_fold_metrics,
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

    fold_metrics.to_csv(
        CV_FOLD_METRICS_PATH,
        index=False,
    )

    model_summary.to_csv(
        CV_SUMMARY_PATH,
        index=False,
    )

    oof = data[
        [
            "encounter_id",
            GROUP_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    fold_assignment = np.zeros(
        len(data),
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

    oof["cv_fold"] = (
        fold_assignment
    )

    for (
        model_name,
        probabilities,
    ) in oof_probabilities.items():
        oof[
            f"{model_name}_probability"
        ] = probabilities

    oof.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\nOOF model summary"
    )

    print("-" * 72)

    print(
        model_summary.to_string(
            index=False
        )
    )

    print(
        "\nVersionable reports saved:"
    )

    print(
        f"- {CV_SPLIT_SUMMARY_PATH}"
    )

    print(
        f"- {CV_FOLD_METRICS_PATH}"
    )

    print(
        f"- {CV_SUMMARY_PATH}"
    )

    print(
        "\nLocal OOF predictions saved:"
    )

    print(
        f"- {OOF_PREDICTIONS_PATH}"
    )

    print(
        "\nThreshold-dependent metrics "
        f"use the predefined threshold "
        f"{DEFAULT_THRESHOLD:.2f}."
    )

    print(
        "No threshold optimization "
        "was performed."
    )

    print(
        "The held-out test set "
        "was not accessed."
    )


if __name__ == "__main__":
    main()

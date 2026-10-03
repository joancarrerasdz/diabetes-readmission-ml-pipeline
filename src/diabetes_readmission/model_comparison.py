"""Patient-safe non-linear model comparison for Week 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from diabetes_readmission.modelling import (
    ENGINEERED_TRAIN_PATH,
    GROUP_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_model_summary,
    create_grouped_cv_splits,
    evaluate_model_cv,
    validate_grouped_cv_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

BASELINE_SUMMARY_PATH = (
    REPORTS_DIR
    / "baseline_cv_summary.csv"
)

BASELINE_OOF_PATH = (
    INTERMEDIATE_DIR
    / "baseline_oof_predictions.csv"
)

RF_FOLD_METRICS_PATH = (
    REPORTS_DIR
    / "random_forest_cv_fold_metrics.csv"
)

COMPARISON_SUMMARY_PATH = (
    REPORTS_DIR
    / "w5_model_comparison_summary.csv"
)

COMPARISON_OOF_PATH = (
    INTERMEDIATE_DIR
    / "w5_model_comparison_oof_predictions.csv"
)

RANDOM_FOREST_NAME = "random_forest"


def get_random_forest_estimator() -> RandomForestClassifier:
    """
    Return the predefined untuned Week-5 Random Forest.

    Hyperparameters are fixed before evaluation and are not
    selected using cross-validation performance.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_comparison_summary(
    baseline_summary: pd.DataFrame,
    random_forest_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine baseline and Random Forest results.

    Positive delta values indicate improvement for ROC-AUC,
    PR-AUC and Brier improvement.
    """
    combined = pd.concat(
        [
            baseline_summary,
            random_forest_summary,
        ],
        ignore_index=True,
    )

    logistic_rows = combined.loc[
        combined["model"]
        == "logistic_regression"
    ]

    if len(logistic_rows) != 1:
        raise RuntimeError(
            "Exactly one logistic_regression "
            "summary row is required."
        )

    logistic = logistic_rows.iloc[0]

    combined[
        "delta_roc_auc_vs_logistic"
    ] = (
        combined["oof_roc_auc"]
        - logistic["oof_roc_auc"]
    )

    combined[
        "delta_pr_auc_vs_logistic"
    ] = (
        combined["oof_pr_auc"]
        - logistic["oof_pr_auc"]
    )

    combined[
        "brier_improvement_vs_logistic"
    ] = (
        logistic["oof_brier_score"]
        - combined["oof_brier_score"]
    )

    return combined


def validate_baseline_oof_alignment(
    data: pd.DataFrame,
    baseline_oof: pd.DataFrame,
) -> None:
    """Ensure baseline OOF rows align exactly with training data."""
    required = {
        "encounter_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
        "cv_fold",
        "dummy_prior_probability",
        "logistic_regression_probability",
    }

    missing = required - set(
        baseline_oof.columns
    )

    if missing:
        raise RuntimeError(
            "Baseline OOF file is missing columns: "
            f"{sorted(missing)}"
        )

    if len(baseline_oof) != len(data):
        raise RuntimeError(
            "Baseline OOF row count does not "
            "match engineered training data."
        )

    for column in [
        "encounter_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
    ]:
        if not np.array_equal(
            baseline_oof[column].to_numpy(),
            data[column].to_numpy(),
        ):
            raise RuntimeError(
                "Baseline OOF alignment failed "
                f"for column: {column}"
            )


def main() -> None:
    """Run the predefined Random Forest OOF comparison."""
    data = pd.read_csv(
        ENGINEERED_TRAIN_PATH,
        low_memory=False,
    )

    if len(data) != 79_473:
        raise RuntimeError(
            "Unexpected training encounter count: "
            f"{len(data):,}"
        )

    if (
        data[GROUP_COLUMN].nunique()
        != 55_952
    ):
        raise RuntimeError(
            "Unexpected unique patient count: "
            f"{data[GROUP_COLUMN].nunique():,}"
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

    estimator = (
        get_random_forest_estimator()
    )

    print(
        "\nW5-C1 — NON-LINEAR MODEL COMPARISON"
    )

    print("=" * 72)

    print(
        "\nEvaluating predefined "
        "Random Forest..."
    )

    fold_metrics, probabilities = (
        evaluate_model_cv(
            dataframe=data,
            splits=splits,
            model_name=(
                RANDOM_FOREST_NAME
            ),
            estimator=estimator,
        )
    )

    random_forest_summary = (
        build_model_summary(
            y=y,
            fold_metrics=fold_metrics,
            oof_probabilities={
                RANDOM_FOREST_NAME:
                    probabilities
            },
        )
    )

    baseline_summary = pd.read_csv(
        BASELINE_SUMMARY_PATH
    )

    required_baselines = {
        "dummy_prior",
        "logistic_regression",
    }

    if not required_baselines.issubset(
        set(baseline_summary["model"])
    ):
        raise RuntimeError(
            "Required baseline models "
            "are not present."
        )

    comparison = (
        build_comparison_summary(
            baseline_summary=(
                baseline_summary
            ),
            random_forest_summary=(
                random_forest_summary
            ),
        )
    )

    baseline_oof = pd.read_csv(
        BASELINE_OOF_PATH
    )

    validate_baseline_oof_alignment(
        data=data,
        baseline_oof=baseline_oof,
    )

    comparison_oof = (
        baseline_oof.copy()
    )

    comparison_oof[
        "random_forest_probability"
    ] = probabilities

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERMEDIATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        RF_FOLD_METRICS_PATH,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_SUMMARY_PATH,
        index=False,
    )

    comparison_oof.to_csv(
        COMPARISON_OOF_PATH,
        index=False,
    )

    columns = [
        "model",
        "oof_roc_auc",
        "oof_pr_auc",
        "oof_brier_score",
        "fold_mean_roc_auc",
        "fold_sd_roc_auc",
        "fold_mean_pr_auc",
        "fold_sd_pr_auc",
        "delta_roc_auc_vs_logistic",
        "delta_pr_auc_vs_logistic",
        "brier_improvement_vs_logistic",
    ]

    print(
        "\nOOF model comparison"
    )

    print("-" * 72)

    print(
        comparison[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nRandom Forest fold metrics"
    )

    print("-" * 72)

    print(
        fold_metrics[
            [
                "fold",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "sensitivity",
                "specificity",
                "patient_overlap",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nVersionable reports saved:"
    )

    print(
        f"- {RF_FOLD_METRICS_PATH}"
    )

    print(
        f"- {COMPARISON_SUMMARY_PATH}"
    )

    print(
        "\nLocal OOF predictions saved:"
    )

    print(
        f"- {COMPARISON_OOF_PATH}"
    )

    print(
        "\nNo hyperparameter search "
        "was performed."
    )

    print(
        "No operating threshold "
        "was selected."
    )

    print(
        "The held-out test set "
        "was not accessed."
    )


if __name__ == "__main__":
    main()

"""Reproducible Week-6 candidate-model selection lock."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = (
    REPORTS_DIR
    / "w6_model_selection_summary.csv"
)

INNER_PATH = (
    REPORTS_DIR
    / "w6_inner_search_summary.csv"
)

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_model_selection_oof_predictions.csv"
)

LOCK_PATH = (
    REPORTS_DIR
    / "w6_selected_model.json"
)

DECISION_PATH = (
    REPORTS_DIR
    / "w6_model_selection_decision.md"
)


SELECTED_MODEL = "random_forest"

LOCKED_PARAMETERS = {
    "classifier__class_weight": None,
    "classifier__max_depth": 12,
    "classifier__min_samples_leaf": 5,
    "classifier__n_estimators": 300,
}


def main() -> None:
    summary = pd.read_csv(
        SUMMARY_PATH
    )

    inner = pd.read_csv(
        INNER_PATH
    )

    oof = pd.read_csv(
        OOF_PATH
    )

    best = (
        summary
        .sort_values(
            "fold_mean_pr_auc",
            ascending=False,
        )
        .iloc[0]
    )

    one_se_cutoff = (
        best["fold_mean_pr_auc"]
        - best["fold_se_pr_auc"]
    )

    summary = summary.copy()

    summary[
        "within_one_se"
    ] = (
        summary[
            "fold_mean_pr_auc"
        ]
        >= one_se_cutoff
    )

    selected_inner = inner.loc[
        inner["selected"] == 1
    ].copy()

    rf_selected = selected_inner.loc[
        selected_inner["model"]
        == SELECTED_MODEL
    ]

    if len(rf_selected) != 5:
        raise RuntimeError(
            "Expected one selected RF "
            "configuration per outer fold."
        )

    rf_parameter_sets = {
        row
        for row
        in rf_selected[
            "parameters"
        ]
    }

    if len(rf_parameter_sets) != 1:
        raise RuntimeError(
            "Random-forest hyperparameters "
            "were not stable across outer folds."
        )

    selected_parameters = json.loads(
        next(
            iter(
                rf_parameter_sets
            )
        )
    )

    if (
        selected_parameters
        != LOCKED_PARAMETERS
    ):
        raise RuntimeError(
            "Observed selected RF parameters "
            "do not match the proposed lock."
        )

    logistic_selected = (
        selected_inner.loc[
            selected_inner["model"]
            == "logistic_regression"
        ]
    )

    logistic_parameter_sets = (
        logistic_selected[
            "parameters"
        ].nunique()
    )

    probability_means = (
        oof.groupby(
            "outer_fold"
        )[
            [
                "logistic_regression_probability",
                "random_forest_probability",
            ]
        ]
        .mean()
    )

    logistic_probability_range = (
        probability_means[
            "logistic_regression_probability"
        ].max()
        - probability_means[
            "logistic_regression_probability"
        ].min()
    )

    rf_probability_range = (
        probability_means[
            "random_forest_probability"
        ].max()
        - probability_means[
            "random_forest_probability"
        ].min()
    )

    selected_row = summary.loc[
        summary["model"]
        == SELECTED_MODEL
    ].iloc[0]

    logistic_row = summary.loc[
        summary["model"]
        == "logistic_regression"
    ].iloc[0]

    lock = {
        "week": 6,
        "selection_status": (
            "candidate_model_locked"
        ),
        "selected_model": (
            SELECTED_MODEL
        ),
        "estimator": (
            "RandomForestClassifier"
        ),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "class_weight": None,
            "random_state": 42,
        },
        "primary_metric": "PR-AUC",
        "selection_rule": (
            "one-standard-error rule "
            "followed by probability quality, "
            "ROC-AUC, stability and "
            "methodological considerations"
        ),
        "mean_outer_pr_auc": float(
            selected_row[
                "fold_mean_pr_auc"
            ]
        ),
        "sd_outer_pr_auc": float(
            selected_row[
                "fold_sd_pr_auc"
            ]
        ),
        "se_outer_pr_auc": float(
            selected_row[
                "fold_se_pr_auc"
            ]
        ),
        "mean_outer_roc_auc": float(
            selected_row[
                "fold_mean_roc_auc"
            ]
        ),
        "mean_outer_brier_score": float(
            selected_row[
                "fold_mean_brier_score"
            ]
        ),
        "hyperparameter_stability": (
            "same configuration selected "
            "in 5/5 outer folds"
        ),
        "threshold_locked": False,
        "calibration_strategy_locked": False,
        "held_out_test_accessed": False,
    }

    with LOCK_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            lock,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    report = f"""# Week 6 model-selection decision

## Decision

The selected candidate model family is **Random Forest**.

The candidate is locked with:

- `n_estimators = 300`
- `max_depth = 12`
- `min_samples_leaf = 5`
- `class_weight = None`
- `random_state = 42`

No operating threshold has been selected.

No calibration strategy has been selected.

The held-out test set remains untouched.

## Primary selection criterion

The pre-specified primary model-selection metric is mean outer-fold
PR-AUC from patient-grouped nested cross-validation.

Random Forest:

- mean outer PR-AUC: {selected_row["fold_mean_pr_auc"]:.6f}
- SD: {selected_row["fold_sd_pr_auc"]:.6f}
- SE: {selected_row["fold_se_pr_auc"]:.6f}

Logistic Regression:

- mean outer PR-AUC: {logistic_row["fold_mean_pr_auc"]:.6f}
- SD: {logistic_row["fold_sd_pr_auc"]:.6f}
- SE: {logistic_row["fold_se_pr_auc"]:.6f}

Best-model 1-SE cutoff:

- {one_se_cutoff:.6f}

Both candidate families fall within the one-standard-error region and
are therefore treated as competitive on the primary discrimination
criterion.

## Secondary evidence

Random Forest:

- mean outer ROC-AUC: {selected_row["fold_mean_roc_auc"]:.6f}
- mean outer Brier score: {selected_row["fold_mean_brier_score"]:.6f}

Logistic Regression:

- mean outer ROC-AUC: {logistic_row["fold_mean_roc_auc"]:.6f}
- mean outer Brier score: {logistic_row["fold_mean_brier_score"]:.6f}

The Random Forest therefore provides modestly stronger discrimination
and substantially better probability accuracy.

## Hyperparameter stability

Random Forest selected the same configuration in all five outer folds.

Logistic Regression produced {logistic_parameter_sets} distinct selected
configurations across the five outer folds.

Four Logistic Regression folds selected `class_weight=None`, whereas
one fold selected `class_weight="balanced"`.

That fold produced a substantial shift in the probability scale and a
markedly worse Brier score. This result is retained as observed; the
search space is not modified retrospectively.

## Cross-fold probability-scale stability

Range of mean predicted probability across outer folds:

- Logistic Regression: {logistic_probability_range:.6f}
- Random Forest: {rf_probability_range:.6f}

The Random Forest predictions are substantially more consistent in
probability scale across the patient-grouped outer folds.

## Selection rationale

Because both candidates satisfy the one-standard-error rule, the
decision is not based on the small difference in PR-AUC alone.

Random Forest is selected because the nested-CV evidence jointly shows:

1. slightly higher PR-AUC;
2. slightly higher ROC-AUC;
3. substantially better Brier score;
4. identical hyperparameter selection across all five outer folds;
5. substantially more stable probability scale across outer folds.

Logistic Regression remains the principal interpretable reference
baseline.

Its interpretability advantage is acknowledged, but it does not
outweigh the probability-quality and stability evidence observed in the
pre-specified nested evaluation.

## Methodological safeguard

The decision uses training data and patient-safe nested
cross-validation only.

No retrospective modification of the candidate search spaces was made.

The held-out test set has not been accessed.

Threshold selection, calibration decisions and final held-out
evaluation remain separate later stages.
"""

    DECISION_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "\nW6 CANDIDATE MODEL LOCK"
    )
    print("=" * 72)

    print(
        "Selected model:",
        SELECTED_MODEL,
    )

    print(
        "Locked parameters:",
        selected_parameters,
    )

    print(
        "\n1-SE cutoff:",
        round(
            one_se_cutoff,
            6,
        ),
    )

    print(
        "\nModels within 1-SE:"
    )

    print(
        summary[
            [
                "model",
                "fold_mean_pr_auc",
                "within_one_se",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nMean-probability range:"
    )

    print(
        "Logistic Regression:",
        round(
            logistic_probability_range,
            6,
        ),
    )

    print(
        "Random Forest:",
        round(
            rf_probability_range,
            6,
        ),
    )

    print(
        "\nSaved:"
    )

    print(
        f"- {LOCK_PATH}"
    )

    print(
        f"- {DECISION_PATH}"
    )

    print(
        "\nNo threshold selected."
    )

    print(
        "No calibration strategy selected."
    )

    print(
        "Held-out test set not accessed."
    )


if __name__ == "__main__":
    main()

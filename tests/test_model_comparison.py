"""Quality gates for Week-5 model comparison."""

import pandas as pd

from diabetes_readmission.model_comparison import (
    build_comparison_summary,
    get_random_forest_estimator,
)


def test_random_forest_configuration_is_fixed():
    estimator = (
        get_random_forest_estimator()
    )

    assert estimator.n_estimators == 200
    assert estimator.max_depth == 16
    assert estimator.min_samples_leaf == 10
    assert estimator.max_features == "sqrt"
    assert estimator.random_state == 42
    assert estimator.class_weight is None


def test_random_forest_starts_unfitted():
    estimator = (
        get_random_forest_estimator()
    )

    assert not hasattr(
        estimator,
        "estimators_",
    )


def test_comparison_delta_convention():
    baseline = pd.DataFrame(
        {
            "model": [
                "dummy_prior",
                "logistic_regression",
            ],
            "oof_roc_auc": [
                0.50,
                0.64,
            ],
            "oof_pr_auc": [
                0.11,
                0.20,
            ],
            "oof_brier_score": [
                0.10,
                0.098,
            ],
        }
    )

    random_forest = pd.DataFrame(
        {
            "model": [
                "random_forest",
            ],
            "oof_roc_auc": [
                0.66,
            ],
            "oof_pr_auc": [
                0.22,
            ],
            "oof_brier_score": [
                0.096,
            ],
        }
    )

    result = build_comparison_summary(
        baseline_summary=baseline,
        random_forest_summary=(
            random_forest
        ),
    )

    rf = result.loc[
        result["model"]
        == "random_forest"
    ].iloc[0]

    assert round(
        rf[
            "delta_roc_auc_vs_logistic"
        ],
        3,
    ) == 0.020

    assert round(
        rf[
            "delta_pr_auc_vs_logistic"
        ],
        3,
    ) == 0.020

    assert round(
        rf[
            "brier_improvement_vs_logistic"
        ],
        3,
    ) == 0.002

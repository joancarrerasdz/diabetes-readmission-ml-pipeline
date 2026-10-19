"""Week-7 one-time final held-out evaluation.

The complete modelling specification was frozen before held-out access.

IMPORTANT
---------
This module must not be executed casually.

The held-out evaluation is protected by an explicit confirmation token
and by a persistent run-state artifact. Once the final evaluation has
started, rerunning it requires explicit investigation and documentation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from diabetes_readmission.calibration import (
    apply_sigmoid_calibrator,
    calibration_intercept_slope,
    expected_calibration_error,
    fit_sigmoid_calibrator,
)
from diabetes_readmission.features import (
    engineer_features,
    validate_engineered_data,
)
from diabetes_readmission.model_preprocessing import (
    MODEL_FEATURES,
    validate_model_schema,
)
from diabetes_readmission.modelling import (
    build_baseline_pipeline,
)
from diabetes_readmission.preprocessing import (
    clean_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort.csv"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "split_assignments.csv"
)

TRAIN_ENGINEERED_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_engineered.csv"
)

W6_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_model_selection_oof_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

RUN_STATE_PATH = (
    REPORTS_DIR
    / "w7_final_evaluation_run_state.json"
)

FINAL_METRICS_PATH = (
    REPORTS_DIR
    / "w7_final_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
    REPORTS_DIR
    / "w7_final_confusion_matrix.csv"
)

CALIBRATION_PATH = (
    REPORTS_DIR
    / "w7_final_calibration.csv"
)

FINDINGS_PATH = (
    REPORTS_DIR
    / "w7_final_evaluation_findings.md"
)

LOCAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w7_final_heldout_predictions.csv"
)


TARGET_COLUMN = "readmitted_30d"
GROUP_COLUMN = "patient_nbr"

EXPECTED_TRAIN_ENCOUNTERS = 79_473
EXPECTED_TRAIN_PATIENTS = 55_952

EXPECTED_TEST_ENCOUNTERS = 19_870
EXPECTED_TEST_PATIENTS = 14_038

LOCKED_THRESHOLD = 0.110

CONFIRMATION_VARIABLE = (
    "W7_FINAL_EVALUATION_CONFIRM"
)

CONFIRMATION_TOKEN = (
    "CONSUME_HELD_OUT_ONCE"
)


def require(
    condition: bool,
    message: str,
) -> None:
    """Fail immediately when a frozen invariant is violated."""
    if not condition:
        raise RuntimeError(message)


def build_locked_estimator() -> RandomForestClassifier:
    """Return the frozen Week-6 candidate estimator."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )


def validate_confirmation() -> None:
    """Protect the one-time held-out evaluation from accidental execution."""
    value = os.environ.get(
        CONFIRMATION_VARIABLE
    )

    if value != CONFIRMATION_TOKEN:
        raise RuntimeError(
            "FINAL HELD-OUT EVALUATION BLOCKED. "
            f"Set {CONFIRMATION_VARIABLE}="
            f"{CONFIRMATION_TOKEN} only when the "
            "frozen Week-7 implementation has been "
            "reviewed, tested and explicitly approved."
        )

    if RUN_STATE_PATH.exists():
        raise RuntimeError(
            "Week-7 held-out evaluation run-state "
            "already exists. Do not rerun the held-out "
            "evaluation without explicit technical "
            "investigation and documentation."
        )


def load_split_assignments() -> pd.DataFrame:
    assignments = pd.read_csv(
        SPLIT_PATH,
        low_memory=False,
    )

    required = {
        "encounter_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
        "split",
    }

    missing = required - set(
        assignments.columns
    )

    require(
        not missing,
        "Split assignment columns missing: "
        f"{sorted(missing)}",
    )

    require(
        set(
            assignments["split"].unique()
        )
        == {"train", "test"},
        "Unexpected split labels.",
    )

    require(
        not assignments[
            "encounter_id"
        ].duplicated().any(),
        "Duplicate encounter assignments.",
    )

    return assignments


def load_training_data() -> pd.DataFrame:
    train = pd.read_csv(
        TRAIN_ENGINEERED_PATH,
        low_memory=False,
    )

    require(
        len(train)
        == EXPECTED_TRAIN_ENCOUNTERS,
        "Unexpected engineered training size.",
    )

    require(
        train[
            GROUP_COLUMN
        ].nunique()
        == EXPECTED_TRAIN_PATIENTS,
        "Unexpected training patient count.",
    )

    require(
        set(
            train[
                TARGET_COLUMN
            ].unique()
        )
        == {0, 1},
        "Training target must contain exactly {0, 1}.",
    )

    validate_model_schema(
        train
    )

    return train


def load_held_out_data(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct held-out engineered features using only
    deterministic transformations frozen before W7.
    """
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    require(
        not cohort[
            "encounter_id"
        ].duplicated().any(),
        "Duplicate encounters in analytic cohort.",
    )

    merged = cohort.merge(
        assignments[
            [
                "encounter_id",
                GROUP_COLUMN,
                "split",
            ]
        ],
        on=[
            "encounter_id",
            GROUP_COLUMN,
        ],
        how="inner",
        validate="one_to_one",
    )

    require(
        len(merged) == len(cohort),
        "Split assignments do not cover "
        "the analytic cohort.",
    )

    test = (
        merged.loc[
            merged["split"] == "test"
        ]
        .drop(
            columns="split"
        )
        .copy()
    )

    require(
        len(test)
        == EXPECTED_TEST_ENCOUNTERS,
        "Unexpected held-out encounter count.",
    )

    require(
        test[
            GROUP_COLUMN
        ].nunique()
        == EXPECTED_TEST_PATIENTS,
        "Unexpected held-out patient count.",
    )

    require(
        set(
            test[
                TARGET_COLUMN
            ].dropna()
            .unique()
        )
        <= {0, 1},
        "Unexpected held-out target coding.",
    )

    cleaned = clean_dataframe(
        test
    )

    require(
        len(cleaned) == len(test),
        "Frozen cleaning changed held-out "
        "row count.",
    )

    engineered = engineer_features(
        cleaned
    )

    validate_engineered_data(
        before=cleaned,
        after=engineered,
    )

    validate_model_schema(
        engineered
    )

    require(
        len(engineered)
        == EXPECTED_TEST_ENCOUNTERS,
        "Feature engineering changed "
        "held-out row count.",
    )

    require(
        len(MODEL_FEATURES) == 42,
        "Unexpected frozen model-feature count.",
    )

    return engineered


def validate_patient_firewall(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    overlap = (
        set(
            train[
                GROUP_COLUMN
            ]
        )
        & set(
            test[
                GROUP_COLUMN
            ]
        )
    )

    require(
        len(overlap) == 0,
        "Patient leakage detected between "
        "development and held-out cohorts.",
    )


def load_calibration_training_data() -> pd.DataFrame:
    """
    Load patient-safe W6 OOF probabilities used to fit the
    already selected sigmoid calibration strategy.
    """
    oof = pd.read_csv(
        W6_OOF_PATH,
        low_memory=False,
    )

    required = {
        "encounter_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
        "random_forest_probability",
    }

    missing = required - set(
        oof.columns
    )

    require(
        not missing,
        "W6 OOF calibration input is missing "
        f"columns: {sorted(missing)}",
    )

    require(
        len(oof)
        == EXPECTED_TRAIN_ENCOUNTERS,
        "Unexpected W6 OOF row count.",
    )

    require(
        oof[
            GROUP_COLUMN
        ].nunique()
        == EXPECTED_TRAIN_PATIENTS,
        "Unexpected W6 OOF patient count.",
    )

    require(
        not oof[
            "random_forest_probability"
        ].isna().any(),
        "Missing W6 Random Forest OOF "
        "probabilities.",
    )

    require(
        oof[
            "random_forest_probability"
        ].between(
            0.0,
            1.0,
        ).all(),
        "Invalid W6 OOF probabilities.",
    )

    return oof


def calculate_final_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(
        target,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    predicted = (
        probability
        >= LOCKED_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target,
        predicted,
        labels=[0, 1],
    ).ravel()

    intercept, slope = (
        calibration_intercept_slope(
            target,
            probability,
        )
    )

    return {
        "n_encounters": int(
            len(target)
        ),
        "n_positive_30d": int(
            target.sum()
        ),
        "prevalence_30d": float(
            target.mean()
        ),
        "mean_predicted_probability": float(
            probability.mean()
        ),
        "pr_auc": float(
            average_precision_score(
                target,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                target,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                target,
                probability,
                labels=[0, 1],
            )
        ),
        "calibration_intercept": (
            intercept
        ),
        "calibration_slope": slope,
        "ece_10_equal_width": float(
            expected_calibration_error(
                target,
                probability,
                n_bins=10,
            )
        ),
        "operating_threshold": (
            LOCKED_THRESHOLD
        ),
        "sensitivity": float(
            recall_score(
                target,
                predicted,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / (tn + fp)
        ),
        "precision": float(
            precision_score(
                target,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                predicted,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                predicted,
            )
        ),
        "predicted_positive_rate": float(
            predicted.mean()
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def build_calibration_table(
    target: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    target = np.asarray(
        target,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        probability,
        edges[1:-1],
        right=False,
    )

    rows = []

    for bin_id in range(
        n_bins
    ):
        mask = (
            bin_ids == bin_id
        )

        if not mask.any():
            continue

        rows.append(
            {
                "bin": (
                    bin_id + 1
                ),
                "lower_bound": float(
                    edges[
                        bin_id
                    ]
                ),
                "upper_bound": float(
                    edges[
                        bin_id + 1
                    ]
                ),
                "n": int(
                    mask.sum()
                ),
                "mean_predicted_probability": float(
                    probability[
                        mask
                    ].mean()
                ),
                "observed_event_rate": float(
                    target[
                        mask
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_run_state(
    status: str,
) -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "week": 7,
        "stage": (
            "final_held_out_evaluation"
        ),
        "status": status,
        "candidate_model": (
            "random_forest"
        ),
        "calibration_strategy": (
            "sigmoid"
        ),
        "operating_threshold": (
            LOCKED_THRESHOLD
        ),
    }

    RUN_STATE_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """
    Execute the single final held-out evaluation.

    Do not call until implementation and tests have been reviewed.
    """
    validate_confirmation()

    assignments = (
        load_split_assignments()
    )

    train = load_training_data()

    test = load_held_out_data(
        assignments
    )

    validate_patient_firewall(
        train,
        test,
    )

    calibration_data = (
        load_calibration_training_data()
    )

    # Mark the one-time evaluation as started
    # before predictive results are generated.
    write_run_state(
        "started"
    )

    estimator = (
        build_locked_estimator()
    )

    pipeline = (
        build_baseline_pipeline(
            estimator
        )
    )

    x_train = train[
        list(
            MODEL_FEATURES
        )
    ]

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    x_test = test[
        list(
            MODEL_FEATURES
        )
    ]

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

    # Frozen model fit:
    # development data only.
    pipeline.fit(
        x_train,
        y_train,
    )

    raw_test_probability = (
        pipeline.predict_proba(
            x_test
        )[:, 1]
    )

    # Frozen sigmoid calibration:
    # calibrator fitted on development OOF evidence only.
    calibrator = (
        fit_sigmoid_calibrator(
            calibration_data[
                "random_forest_probability"
            ].to_numpy(
                dtype=float
            ),
            calibration_data[
                TARGET_COLUMN
            ].to_numpy(
                dtype=int
            ),
        )
    )

    calibrated_probability = (
        apply_sigmoid_calibrator(
            calibrator,
            raw_test_probability,
        )
    )

    metrics = (
        calculate_final_metrics(
            y_test.to_numpy(),
            calibrated_probability,
        )
    )

    calibration_table = (
        build_calibration_table(
            y_test.to_numpy(),
            calibrated_probability,
        )
    )

    predictions = test[
        [
            "encounter_id",
            GROUP_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    predictions[
        "raw_random_forest_probability"
    ] = raw_test_probability

    predictions[
        "sigmoid_probability"
    ] = calibrated_probability

    predictions[
        "predicted_30d"
    ] = (
        calibrated_probability
        >= LOCKED_THRESHOLD
    ).astype(int)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        FINAL_METRICS_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "actual_negative_predicted_negative":
                    metrics["tn"],
                "actual_negative_predicted_positive":
                    metrics["fp"],
                "actual_positive_predicted_negative":
                    metrics["fn"],
                "actual_positive_predicted_positive":
                    metrics["tp"],
            }
        ]
    ).to_csv(
        CONFUSION_MATRIX_PATH,
        index=False,
    )

    calibration_table.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    predictions.to_csv(
        LOCAL_PREDICTIONS_PATH,
        index=False,
    )

    findings = f"""# Week 7 — Final held-out evaluation

## Frozen specification

- candidate model: Random Forest
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 5
- class_weight: None
- random_state: 42
- probability calibration: sigmoid
- operating threshold: {LOCKED_THRESHOLD:.3f}

## Independent held-out cohort

- encounters: {metrics["n_encounters"]:,}
- positive 30-day readmissions: {metrics["n_positive_30d"]:,}
- prevalence: {metrics["prevalence_30d"]:.6f}

## Final discrimination and probability performance

- PR-AUC: {metrics["pr_auc"]:.6f}
- ROC-AUC: {metrics["roc_auc"]:.6f}
- Brier score: {metrics["brier_score"]:.6f}
- log loss: {metrics["log_loss"]:.6f}

## Calibration

- mean predicted probability: {metrics["mean_predicted_probability"]:.6f}
- calibration intercept: {metrics["calibration_intercept"]:.6f}
- calibration slope: {metrics["calibration_slope"]:.6f}
- ECE (10 equal-width bins): {metrics["ece_10_equal_width"]:.6f}

## Locked operating threshold

Threshold: **{LOCKED_THRESHOLD:.3f}**

- sensitivity: {metrics["sensitivity"]:.6f}
- specificity: {metrics["specificity"]:.6f}
- precision: {metrics["precision"]:.6f}
- F1: {metrics["f1"]:.6f}
- balanced accuracy: {metrics["balanced_accuracy"]:.6f}
- predicted-positive rate: {metrics["predicted_positive_rate"]:.6f}

## Confusion matrix

- TP: {metrics["tp"]}
- FP: {metrics["fp"]}
- TN: {metrics["tn"]}
- FN: {metrics["fn"]}

## Methodological status

This is the one-time independent held-out evaluation of the frozen
development pipeline.

The held-out data were not used to change the feature schema,
preprocessing, model family, hyperparameters, calibration strategy or
operating threshold.

The held-out test set is now considered consumed for this development
iteration.

Observed performance must not be used for retrospective tuning.
"""

    FINDINGS_PATH.write_text(
        findings,
        encoding="utf-8",
    )

    write_run_state(
        "completed"
    )

    print(
        "\nW7 FINAL HELD-OUT EVALUATION"
    )
    print("=" * 72)

    print(
        pd.DataFrame(
            [metrics]
        ).to_string(
            index=False
        )
    )

    print(
        "\nFinal held-out evaluation completed."
    )

    print(
        "The held-out test set is now CONSUMED "
        "for this development iteration."
    )

    print(
        "\nVersionable outputs:"
    )

    print(
        f"- {FINAL_METRICS_PATH}"
    )

    print(
        f"- {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"- {CALIBRATION_PATH}"
    )

    print(
        f"- {FINDINGS_PATH}"
    )

    print(
        f"- {RUN_STATE_PATH}"
    )

    print(
        "\nLocal row-level predictions:"
    )

    print(
        f"- {LOCAL_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()

"""Week-7 pre-flight audit before final held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from diabetes_readmission.features import (
    engineer_features,
    validate_engineered_data,
)
from diabetes_readmission.model_preprocessing import (
    MODEL_FEATURES,
    validate_model_schema,
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

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_LOCK_PATH = (
    REPORTS_DIR
    / "w6_selected_model.json"
)

CALIBRATION_LOCK_PATH = (
    REPORTS_DIR
    / "w6_calibration_lock.json"
)

THRESHOLD_LOCK_PATH = (
    REPORTS_DIR
    / "w6_threshold_lock.json"
)

AUDIT_PATH = (
    REPORTS_DIR
    / "w7_preflight_audit.csv"
)

EXPECTED_TRAIN_ENCOUNTERS = 79_473
EXPECTED_MODEL_FEATURES = 42
EXPECTED_MODEL = "random_forest"
EXPECTED_CALIBRATION = "sigmoid"
EXPECTED_THRESHOLD = 0.110

EXPECTED_HYPERPARAMETERS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": None,
    "random_state": 42,
}


def load_json(path: Path) -> dict:
    """Load a JSON lock artifact."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def require(
    condition: bool,
    message: str,
) -> None:
    """Fail immediately when a pre-flight invariant is violated."""
    if not condition:
        raise RuntimeError(message)


def add_check(
    rows: list[dict],
    check: str,
    value: object,
) -> None:
    """Append a passed audit check."""
    rows.append(
        {
            "check": check,
            "value": value,
            "status": "PASS",
        }
    )


def validate_w6_locks(
    rows: list[dict],
) -> None:
    """Validate consistency of the frozen Week-6 decisions."""
    model_lock = load_json(
        MODEL_LOCK_PATH
    )

    calibration_lock = load_json(
        CALIBRATION_LOCK_PATH
    )

    threshold_lock = load_json(
        THRESHOLD_LOCK_PATH
    )

    require(
        model_lock["selected_model"]
        == EXPECTED_MODEL,
        "Unexpected locked candidate model.",
    )

    require(
        model_lock["hyperparameters"]
        == EXPECTED_HYPERPARAMETERS,
        "Locked hyperparameters do not match "
        "the frozen W6 specification.",
    )

    require(
        calibration_lock["candidate_model"]
        == EXPECTED_MODEL,
        "Calibration lock candidate model mismatch.",
    )

    require(
        calibration_lock["calibration_strategy"]
        == EXPECTED_CALIBRATION,
        "Unexpected calibration strategy.",
    )

    require(
        calibration_lock["status"]
        == "locked",
        "Calibration strategy is not locked.",
    )

    require(
        threshold_lock["candidate_model"]
        == EXPECTED_MODEL,
        "Threshold lock candidate model mismatch.",
    )

    require(
        threshold_lock["calibration_strategy"]
        == EXPECTED_CALIBRATION,
        "Threshold lock calibration strategy mismatch.",
    )

    require(
        abs(
            float(
                threshold_lock[
                    "operating_threshold"
                ]
            )
            - EXPECTED_THRESHOLD
        )
        < 1e-12,
        "Unexpected operating threshold.",
    )

    require(
        threshold_lock["status"]
        == "locked",
        "Operating threshold is not locked.",
    )

    require(
        threshold_lock[
            "clinical_threshold"
        ]
        is False,
        "Threshold is unexpectedly marked "
        "as clinically validated.",
    )

    require(
        model_lock[
            "held_out_test_accessed"
        ]
        is False,
        "Model-selection lock records prior "
        "held-out access.",
    )

    require(
        calibration_lock[
            "held_out_test_accessed"
        ]
        is False,
        "Calibration lock records prior "
        "held-out access.",
    )

    require(
        threshold_lock[
            "held_out_test_accessed"
        ]
        is False,
        "Threshold lock records prior "
        "held-out access.",
    )

    add_check(
        rows,
        "candidate_model",
        EXPECTED_MODEL,
    )

    add_check(
        rows,
        "hyperparameters_locked",
        True,
    )

    add_check(
        rows,
        "calibration_strategy",
        EXPECTED_CALIBRATION,
    )

    add_check(
        rows,
        "operating_threshold",
        EXPECTED_THRESHOLD,
    )

    add_check(
        rows,
        "w6_held_out_access_record",
        False,
    )


def validate_split_firewall(
    rows: list[dict],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Validate the frozen patient-level train/test split."""
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        SPLIT_PATH,
        low_memory=False,
    )

    required_assignment_columns = {
        "encounter_id",
        "patient_nbr",
        "readmitted_30d",
        "split",
    }

    missing = (
        required_assignment_columns
        - set(assignments.columns)
    )

    require(
        not missing,
        "Split assignments are missing columns: "
        f"{sorted(missing)}",
    )

    require(
        set(assignments["split"].unique())
        == {"train", "test"},
        "Unexpected split labels.",
    )

    require(
        not assignments[
            "encounter_id"
        ].duplicated().any(),
        "Duplicate encounters in split assignments.",
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
                "patient_nbr",
                "split",
            ]
        ],
        on=[
            "encounter_id",
            "patient_nbr",
        ],
        how="inner",
        validate="one_to_one",
    )

    require(
        len(merged) == len(cohort),
        "Split assignments do not cover "
        "the complete analytic cohort.",
    )

    train = (
        merged.loc[
            merged["split"] == "train"
        ]
        .drop(columns="split")
        .copy()
    )

    test = (
        merged.loc[
            merged["split"] == "test"
        ]
        .drop(columns="split")
        .copy()
    )

    require(
        len(train)
        == EXPECTED_TRAIN_ENCOUNTERS,
        "Unexpected number of training encounters.",
    )

    require(
        len(test) > 0,
        "Held-out test set is empty.",
    )

    train_patients = set(
        train["patient_nbr"]
    )

    test_patients = set(
        test["patient_nbr"]
    )

    patient_overlap = (
        train_patients
        & test_patients
    )

    require(
        len(patient_overlap) == 0,
        "Patient leakage detected between "
        "development and held-out sets.",
    )

    encounter_overlap = (
        set(train["encounter_id"])
        & set(test["encounter_id"])
    )

    require(
        len(encounter_overlap) == 0,
        "Encounter leakage detected between "
        "development and held-out sets.",
    )

    require(
        set(
            test[
                "readmitted_30d"
            ].dropna().unique()
        )
        <= {0, 1},
        "Unexpected held-out target coding.",
    )

    require(
        not test[
            "patient_nbr"
        ].isna().any(),
        "Held-out patient identifiers contain "
        "missing values.",
    )

    add_check(
        rows,
        "training_encounters",
        len(train),
    )

    add_check(
        rows,
        "held_out_encounters",
        len(test),
    )

    add_check(
        rows,
        "training_unique_patients",
        train["patient_nbr"].nunique(),
    )

    add_check(
        rows,
        "held_out_unique_patients",
        test["patient_nbr"].nunique(),
    )

    add_check(
        rows,
        "patient_overlap",
        len(patient_overlap),
    )

    add_check(
        rows,
        "encounter_overlap",
        len(encounter_overlap),
    )

    add_check(
        rows,
        "held_out_target_binary",
        True,
    )

    return train, test


def validate_training_snapshot(
    train: pd.DataFrame,
    rows: list[dict],
) -> None:
    """Confirm the frozen engineered training snapshot is aligned."""
    engineered_train = pd.read_csv(
        TRAIN_ENGINEERED_PATH,
        low_memory=False,
    )

    require(
        len(engineered_train)
        == EXPECTED_TRAIN_ENCOUNTERS,
        "Unexpected engineered training size.",
    )

    require(
        set(
            engineered_train[
                "encounter_id"
            ]
        )
        == set(
            train[
                "encounter_id"
            ]
        ),
        "Engineered training encounters do not "
        "match the frozen training assignment.",
    )

    validate_model_schema(
        engineered_train
    )

    add_check(
        rows,
        "engineered_training_alignment",
        True,
    )

    add_check(
        rows,
        "training_model_features",
        len(MODEL_FEATURES),
    )


def validate_held_out_schema(
    test: pd.DataFrame,
    rows: list[dict],
) -> None:
    """
    Apply only previously frozen deterministic transformations.

    No estimator, encoder, scaler, calibrator or threshold is fitted here.
    No predictive result is generated.
    """
    cleaned = clean_dataframe(
        test
    )

    require(
        len(cleaned) == len(test),
        "Deterministic cleaning changed "
        "held-out row count.",
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
        len(MODEL_FEATURES)
        == EXPECTED_MODEL_FEATURES,
        "Unexpected number of frozen "
        "model features.",
    )

    require(
        set(MODEL_FEATURES)
        <= set(engineered.columns),
        "One or more frozen predictors are "
        "missing from held-out data.",
    )

    add_check(
        rows,
        "held_out_cleaning_row_preservation",
        True,
    )

    add_check(
        rows,
        "held_out_feature_engineering_row_preservation",
        True,
    )

    add_check(
        rows,
        "held_out_model_schema_valid",
        True,
    )

    add_check(
        rows,
        "held_out_model_features",
        len(MODEL_FEATURES),
    )


def main() -> None:
    """Run the Week-7 pre-flight gate."""
    rows: list[dict] = []

    validate_w6_locks(
        rows
    )

    train, test = (
        validate_split_firewall(
            rows
        )
    )

    validate_training_snapshot(
        train,
        rows,
    )

    validate_held_out_schema(
        test,
        rows,
    )

    audit = pd.DataFrame(
        rows
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    print(
        "\nW7 PRE-FLIGHT GATE"
    )
    print("=" * 72)

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\nW7 PRE-FLIGHT: PASS"
    )

    print(
        "\nNo predictive model was fitted."
    )

    print(
        "No held-out predictions were generated."
    )

    print(
        "No held-out performance metric "
        "was calculated."
    )

    print(
        "\nVersionable audit saved:"
    )

    print(
        f"- {AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()

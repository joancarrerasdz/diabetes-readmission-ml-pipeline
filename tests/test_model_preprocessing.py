"""Tests for the model-ready feature schema."""

import pandas as pd

from diabetes_readmission.model_preprocessing import (
    GENERAL_CATEGORICAL_FEATURES,
    MEDICAL_SPECIALTY_MIN_FREQUENCY,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SPECIALTY_FEATURES,
    build_feature_schema,
    build_model_preprocessor,
    validate_model_schema,
)


def make_model_ready_data() -> pd.DataFrame:
    rows = 2

    data = {}

    for feature in NUMERIC_FEATURES:
        data[feature] = [1.0, 2.0]

    for feature in GENERAL_CATEGORICAL_FEATURES:
        data[feature] = ["A", "B"]

    data["medical_specialty"] = [
        "InternalMedicine",
        "Cardiology",
    ]

    data["encounter_id"] = [1, 2]
    data["patient_nbr"] = [101, 102]
    data["discharge_disposition_id"] = [1, 1]
    data["readmitted"] = ["NO", "<30"]
    data["readmitted_30d"] = [0, 1]

    frame = pd.DataFrame(data)

    assert len(frame) == rows

    return frame


def test_model_feature_schema_contains_42_features():
    assert len(MODEL_FEATURES) == 42
    assert len(set(MODEL_FEATURES)) == 42


def test_model_feature_groups_do_not_overlap():
    numeric = set(NUMERIC_FEATURES)
    categorical = set(
        GENERAL_CATEGORICAL_FEATURES
    )
    specialty = set(SPECIALTY_FEATURES)

    assert numeric.isdisjoint(categorical)
    assert numeric.isdisjoint(specialty)
    assert categorical.isdisjoint(specialty)


def test_model_schema_validation_passes():
    frame = make_model_ready_data()

    validate_model_schema(frame)


def test_feature_schema_documents_all_features():
    schema = build_feature_schema()

    assert len(schema) == 42

    assert set(schema["feature"]) == set(
        MODEL_FEATURES
    )


def test_model_preprocessor_is_created_unfitted():
    preprocessor = build_model_preprocessor()

    assert not hasattr(
        preprocessor,
        "transformers_",
    )

    assert (
        MEDICAL_SPECIALTY_MIN_FREQUENCY
        == 0.005
    )


def test_preprocessor_handles_infrequent_and_unseen_categories():
    """Smoke-test fitted preprocessing using synthetic data only."""
    n_rows = 1000

    data = {}

    for feature in NUMERIC_FEATURES:
        data[feature] = [
            float(index % 10)
            for index in range(n_rows)
        ]

    for feature in GENERAL_CATEGORICAL_FEATURES:
        data[feature] = [
            "A" if index % 2 == 0 else "B"
            for index in range(n_rows)
        ]

    # With min_frequency=0.005 and 1000 rows,
    # categories occurring fewer than 5 times are infrequent.
    data["medical_specialty"] = (
        ["InternalMedicine"] * 995
        + ["Cardiology"] * 4
        + ["RareSpecialty"]
    )

    frame = pd.DataFrame(data)

    preprocessor = build_model_preprocessor()

    transformed = preprocessor.fit_transform(
        frame[list(MODEL_FEATURES)]
    )

    assert transformed.shape[0] == n_rows

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    assert transformed.shape[1] == len(
        feature_names
    )

    assert len(feature_names) == len(
        set(feature_names)
    )

    specialty_encoder = (
        preprocessor
        .named_transformers_["medical_specialty"]
    )

    infrequent = set(
        specialty_encoder
        .infrequent_categories_[0]
        .tolist()
    )

    assert {
        "Cardiology",
        "RareSpecialty",
    }.issubset(infrequent)

    # Simulate completely unseen levels at inference time.
    unseen = frame.iloc[[0]].copy()

    for feature in GENERAL_CATEGORICAL_FEATURES:
        unseen[feature] = "UNSEEN_LEVEL"

    unseen["medical_specialty"] = (
        "UnseenSpecialty"
    )

    unseen_transformed = preprocessor.transform(
        unseen[list(MODEL_FEATURES)]
    )

    assert unseen_transformed.shape[0] == 1

    assert (
        unseen_transformed.shape[1]
        == transformed.shape[1]
    )

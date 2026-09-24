"""Model-ready feature schema and fold-safe preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diabetes_readmission.preprocessing import (
    get_model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENGINEERED_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_engineered.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

SCHEMA_PATH = (
    REPORTS_DIR
    / "model_feature_schema.csv"
)


NUMERIC_FEATURES = (
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)


SPECIALTY_FEATURES = (
    "medical_specialty",
)


GENERAL_CATEGORICAL_FEATURES = (
    "race",
    "gender",
    "admission_type_id",
    "admission_source_id",
    "payer_code",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
)


MODEL_FEATURES = (
    NUMERIC_FEATURES
    + SPECIALTY_FEATURES
    + GENERAL_CATEGORICAL_FEATURES
)


FORBIDDEN_MODEL_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "discharge_disposition_id",
    "readmitted",
    "readmitted_30d",
    "weight",
    "age",
    "diag_1",
    "diag_2",
    "diag_3",
    "citoglipton",
    "examide",
    "glimepiride-pioglitazone",
}


MEDICAL_SPECIALTY_MIN_FREQUENCY = 0.005


def validate_model_schema(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the complete model feature schema."""
    numeric = set(NUMERIC_FEATURES)
    specialty = set(SPECIALTY_FEATURES)
    categorical = set(
        GENERAL_CATEGORICAL_FEATURES
    )

    if numeric & specialty:
        raise RuntimeError(
            "Numeric and specialty feature groups overlap."
        )

    if numeric & categorical:
        raise RuntimeError(
            "Numeric and categorical feature groups overlap."
        )

    if specialty & categorical:
        raise RuntimeError(
            "Specialty and categorical feature groups overlap."
        )

    configured = set(MODEL_FEATURES)

    if len(configured) != 42:
        raise RuntimeError(
            f"Expected 42 configured model features, "
            f"found {len(configured)}."
        )

    available = set(
        get_model_feature_columns(dataframe)
    )

    missing = configured - available

    if missing:
        raise RuntimeError(
            "Configured model features missing from data: "
            f"{sorted(missing)}"
        )

    unexpected = available - configured

    if unexpected:
        raise RuntimeError(
            "Unexpected candidate model features: "
            f"{sorted(unexpected)}"
        )

    forbidden_present = (
        FORBIDDEN_MODEL_COLUMNS
        & configured
    )

    if forbidden_present:
        raise RuntimeError(
            "Forbidden columns configured as predictors: "
            f"{sorted(forbidden_present)}"
        )

    null_cells = int(
        dataframe[
            list(MODEL_FEATURES)
        ]
        .isna()
        .sum()
        .sum()
    )

    if null_cells:
        raise RuntimeError(
            f"Model features contain "
            f"{null_cells:,} missing cells."
        )


def build_model_preprocessor() -> ColumnTransformer:
    """
    Build fold-safe preprocessing.

    The returned transformer is intentionally unfitted.
    It must be fitted only inside the modelling pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    specialty_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=(
            MEDICAL_SPECIALTY_MIN_FREQUENCY
        ),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                categorical_encoder,
                list(
                    GENERAL_CATEGORICAL_FEATURES
                ),
            ),
            (
                "medical_specialty",
                specialty_encoder,
                list(SPECIALTY_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_schema() -> pd.DataFrame:
    """Create a versionable description of model feature roles."""
    rows = []

    for feature in NUMERIC_FEATURES:
        rows.append(
            {
                "feature": feature,
                "feature_group": "numeric",
                "preprocessing": "StandardScaler",
            }
        )

    for feature in GENERAL_CATEGORICAL_FEATURES:
        rows.append(
            {
                "feature": feature,
                "feature_group": "categorical",
                "preprocessing": (
                    "OneHotEncoder("
                    "handle_unknown=ignore)"
                ),
            }
        )

    for feature in SPECIALTY_FEATURES:
        rows.append(
            {
                "feature": feature,
                "feature_group": (
                    "categorical_high_cardinality"
                ),
                "preprocessing": (
                    "OneHotEncoder("
                    "handle_unknown="
                    "infrequent_if_exist,"
                    "min_frequency=0.005)"
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Validate and document the model-ready feature schema."""
    data = pd.read_csv(
        ENGINEERED_TRAIN_PATH,
        low_memory=False,
    )

    if len(data) != 79_473:
        raise RuntimeError(
            f"Unexpected training size: "
            f"{len(data):,}"
        )

    validate_model_schema(data)

    schema = build_feature_schema()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema.to_csv(
        SCHEMA_PATH,
        index=False,
    )

    preprocessor = build_model_preprocessor()

    if hasattr(
        preprocessor,
        "transformers_",
    ):
        raise RuntimeError(
            "Preprocessor was unexpectedly fitted."
        )

    print("\nMODEL-READY FEATURE SCHEMA")
    print("=" * 70)

    print(
        f"Numeric features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print(
        "General categorical features: "
        f"{len(GENERAL_CATEGORICAL_FEATURES)}"
    )

    print(
        "High-cardinality specialty features: "
        f"{len(SPECIALTY_FEATURES)}"
    )

    print(
        f"Total model features: "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        "\nmedical_specialty rare-category rule:"
    )

    print(
        "- learned inside model fitting"
    )

    print(
        "- min_frequency = "
        f"{MEDICAL_SPECIALTY_MIN_FREQUENCY}"
    )

    print(
        "\nPreprocessor state: UNFITTED"
    )

    print(
        "\nFeature schema saved:"
    )

    print(f"- {SCHEMA_PATH}")

    print(
        "\nThe held-out test set was not accessed."
    )


if __name__ == "__main__":
    main()

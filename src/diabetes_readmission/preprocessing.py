"""Deterministic data-cleaning rules established from training-only EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEAN_TRAIN_PATH = INTERMEDIATE_DIR / "train_cleaned.csv"
CLEANING_SUMMARY_PATH = REPORTS_DIR / "cleaning_summary.csv"


DROP_COLUMNS = {
    "weight",
    "citoglipton",
    "examide",
    "glimepiride-pioglitazone",
}

LAB_NOT_MEASURED_COLUMNS = {
    "max_glu_serum",
    "A1Cresult",
}

EXPLICIT_MISSING_CATEGORY_COLUMNS = {
    "race",
    "gender",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
}

NON_PREDICTOR_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "discharge_disposition_id",
    "readmitted",
    "readmitted_30d",
}


def load_training_data() -> pd.DataFrame:
    """Load only encounters assigned to the frozen training split."""
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        SPLIT_PATH,
        usecols=["encounter_id", "split"],
    )

    merged = cohort.merge(
        assignments,
        on="encounter_id",
        how="inner",
        validate="one_to_one",
    )

    train = (
        merged.loc[merged["split"] == "train"]
        .drop(columns="split")
        .copy()
    )

    if len(train) != 79_473:
        raise RuntimeError(
            f"Unexpected training size: {len(train):,}"
        )

    return train


def clean_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply deterministic cleaning without learning from the data."""
    data = dataframe.copy()

    missing_drop_columns = DROP_COLUMNS - set(data.columns)

    if missing_drop_columns:
        raise ValueError(
            "Expected columns missing before cleaning: "
            f"{sorted(missing_drop_columns)}"
        )

    data = data.drop(
        columns=sorted(DROP_COLUMNS)
    )

    for column in LAB_NOT_MEASURED_COLUMNS:
        data[column] = (
            data[column]
            .astype("string")
            .fillna("Not_measured")
        )

    explicit_missing_markers = {
        "?",
        "Unknown/Invalid",
        "Not Available",
        "NULL",
        "",
    }

    for column in EXPLICIT_MISSING_CATEGORY_COLUMNS:
        data[column] = (
            data[column]
            .astype("string")
            .replace(list(explicit_missing_markers), pd.NA)
            .fillna("Missing")
        )

    return data


def build_cleaning_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """Create an auditable summary of deterministic cleaning."""
    rows = [
        {
            "metric": "rows_before",
            "value": len(before),
        },
        {
            "metric": "rows_after",
            "value": len(after),
        },
        {
            "metric": "columns_before",
            "value": before.shape[1],
        },
        {
            "metric": "columns_after",
            "value": after.shape[1],
        },
        {
            "metric": "columns_dropped",
            "value": before.shape[1] - after.shape[1],
        },
        {
            "metric": "remaining_null_cells",
            "value": int(after.isna().sum().sum()),
        },
    ]

    for column in sorted(DROP_COLUMNS):
        rows.append(
            {
                "metric": f"dropped::{column}",
                "value": 1,
            }
        )

    return pd.DataFrame(rows)


def validate_cleaned_data(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Validate core invariants after deterministic cleaning."""
    if len(before) != len(after):
        raise RuntimeError(
            "Cleaning changed the number of encounters."
        )

    if DROP_COLUMNS & set(after.columns):
        raise RuntimeError(
            "One or more planned drop columns remain."
        )

    for column in LAB_NOT_MEASURED_COLUMNS:
        if after[column].isna().any():
            raise RuntimeError(
                f"{column} still contains missing values."
            )

        if "Not_measured" not in set(after[column]):
            raise RuntimeError(
                f"{column} does not contain the expected "
                "'Not_measured' category."
            )

    for column in EXPLICIT_MISSING_CATEGORY_COLUMNS:
        if after[column].isna().any():
            raise RuntimeError(
                f"{column} still contains missing values."
            )

    if after["encounter_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate encounter_id values detected after cleaning."
        )


def get_model_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return predictor columns after removing non-predictor fields."""
    return [
        column
        for column in dataframe.columns
        if column not in NON_PREDICTOR_COLUMNS
    ]


def save_outputs(
    cleaned: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save local cleaned training data and versionable audit summary."""
    INTERMEDIATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        CLEAN_TRAIN_PATH,
        index=False,
    )

    summary.to_csv(
        CLEANING_SUMMARY_PATH,
        index=False,
    )


def main() -> None:
    """Clean the frozen training set and report the transformation."""
    train = load_training_data()

    cleaned = clean_dataframe(train)

    validate_cleaned_data(
        before=train,
        after=cleaned,
    )

    summary = build_cleaning_summary(
        before=train,
        after=cleaned,
    )

    features = get_model_feature_columns(cleaned)

    save_outputs(
        cleaned=cleaned,
        summary=summary,
    )

    print("\nTRAINING CLEANING AUDIT")
    print("=" * 70)

    print(summary.to_string(index=False))

    print(
        f"\nModel candidate features after cleaning: "
        f"{len(features)}"
    )

    print("\nExplicit laboratory missingness:")
    for column in sorted(LAB_NOT_MEASURED_COLUMNS):
        print(
            f"- {column}: "
            f"{(cleaned[column] == 'Not_measured').sum():,} "
            "Not_measured"
        )

    print("\nCleaned training data saved locally:")
    print(f"- {CLEAN_TRAIN_PATH}")

    print("\nVersionable cleaning summary:")
    print(f"- {CLEANING_SUMMARY_PATH}")

    print(
        "\nNo held-out test data were used to derive "
        "these cleaning decisions."
    )


if __name__ == "__main__":
    main()

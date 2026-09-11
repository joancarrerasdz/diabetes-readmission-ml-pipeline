"""Dataset acquisition utilities for the diabetes readmission project."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 296
EXPECTED_N_ROWS = 101_766
EXPECTED_N_COLUMNS = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_PATH = RAW_DIR / "diabetes_130_us_hospitals.csv"
VARIABLES_PATH = RAW_DIR / "uci_variables.csv"
METADATA_PATH = RAW_DIR / "uci_metadata.json"


def fetch_diabetes_dataset():
    """Fetch the UCI Diabetes 130-US Hospitals dataset."""
    return fetch_ucirepo(id=DATASET_ID)


def build_raw_dataframe(dataset) -> pd.DataFrame:
    """Combine UCI identifiers, features and target without altering values."""
    identifiers = dataset.data.ids.copy()
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    all_columns = (
        identifiers.columns.tolist()
        + features.columns.tolist()
        + targets.columns.tolist()
    )

    duplicated_columns = sorted(
        column
        for column, count in Counter(all_columns).items()
        if count > 1
    )

    if duplicated_columns:
        raise ValueError(
            f"Duplicated columns detected across UCI data blocks: "
            f"{duplicated_columns}"
        )

    dataframe = pd.concat(
        [identifiers, features, targets],
        axis=1,
    )

    if len(dataframe) != EXPECTED_N_ROWS:
        raise ValueError(
            f"Unexpected number of rows: {len(dataframe):,}. "
            f"Expected {EXPECTED_N_ROWS:,}."
        )

    if dataframe.shape[1] != EXPECTED_N_COLUMNS:
        raise ValueError(
            f"Unexpected number of columns: {dataframe.shape[1]}. "
            f"Expected {EXPECTED_N_COLUMNS}."
        )

    required_columns = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    return dataframe


def save_raw_snapshot(dataset, dataframe: pd.DataFrame) -> None:
    """Save an immutable local snapshot of source data and metadata."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = [
        path
        for path in (DATA_PATH, VARIABLES_PATH, METADATA_PATH)
        if path.exists()
    ]

    if existing_files:
        formatted = "\n".join(f"- {path}" for path in existing_files)
        raise FileExistsError(
            "Raw dataset snapshot already exists. "
            "Raw files must remain immutable:\n"
            f"{formatted}"
        )

    dataframe.to_csv(DATA_PATH, index=False)
    dataset.variables.to_csv(VARIABLES_PATH, index=False)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            dataset.metadata,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def print_dataset_audit(dataframe: pd.DataFrame) -> None:
    """Print basic structural checks without preprocessing the data."""
    print("\nDataset audit")
    print("-" * 60)

    print(f"Rows:            {len(dataframe):,}")
    print(f"Columns:         {dataframe.shape[1]}")
    print(f"Duplicate rows:  {dataframe.duplicated().sum():,}")

    encounters_per_patient = dataframe.groupby("patient_nbr").size()

    print(f"Unique patients: {encounters_per_patient.size:,}")
    print(
        "Patients with >1 encounter: "
        f"{(encounters_per_patient > 1).sum():,}"
    )
    print(
        "Maximum encounters for one patient: "
        f"{encounters_per_patient.max():,}"
    )

    print("\nOriginal readmitted values:")
    print(
        dataframe["readmitted"]
        .value_counts(dropna=False)
        .to_string()
    )


def main() -> None:
    """Fetch, validate and save the raw UCI dataset."""
    print(f"Fetching UCI dataset ID {DATASET_ID}...")

    dataset = fetch_diabetes_dataset()
    dataframe = build_raw_dataframe(dataset)

    print_dataset_audit(dataframe)
    save_raw_snapshot(dataset, dataframe)

    print("\nRaw snapshot saved:")
    print(f"- {DATA_PATH}")
    print(f"- {VARIABLES_PATH}")
    print(f"- {METADATA_PATH}")

    print(
        "\nNo cleaning, exclusions or target transformation "
        "were applied."
    )


if __name__ == "__main__":
    main()

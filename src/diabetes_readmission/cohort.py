"""Cohort definition for the diabetes readmission project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "diabetes_130_us_hospitals.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

COHORT_PATH = PROCESSED_DIR / "cohort.csv"
FLOW_PATH = REPORTS_DIR / "cohort_flow.csv"

DISCHARGE_EXCLUSION_CODES = {
    11,  # Expired
    13,  # Hospice / home
    14,  # Hospice / medical facility
    19,  # Expired at home, Medicaid only, hospice
    20,  # Expired in medical facility, Medicaid only, hospice
    21,  # Expired, place unknown, Medicaid only, hospice
}


def load_raw_data() -> pd.DataFrame:
    """Load the immutable raw snapshot."""
    return pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False,
    )


def apply_primary_cohort(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-specified encounter-level cohort exclusions."""
    required = {
        "encounter_id",
        "patient_nbr",
        "discharge_disposition_id",
        "readmitted",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Required cohort columns missing: {sorted(missing)}"
        )

    data = dataframe.copy()

    data["discharge_disposition_id"] = pd.to_numeric(
        data["discharge_disposition_id"],
        errors="coerce",
    )

    exclusion_mask = data["discharge_disposition_id"].isin(
        DISCHARGE_EXCLUSION_CODES
    )

    excluded = data.loc[exclusion_mask].copy()
    cohort = data.loc[~exclusion_mask].copy()

    return cohort, excluded


def build_cohort_flow(
    raw: pd.DataFrame,
    cohort: pd.DataFrame,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact audit table describing cohort construction."""
    excluded_patients = set(excluded["patient_nbr"])
    retained_patients = set(cohort["patient_nbr"])

    rows = [
        {
            "stage": "raw_dataset",
            "encounters": len(raw),
            "unique_patients": raw["patient_nbr"].nunique(),
        },
        {
            "stage": "excluded_non_interpretable_discharge",
            "encounters": len(excluded),
            "unique_patients": excluded["patient_nbr"].nunique(),
        },
        {
            "stage": "primary_analytic_cohort",
            "encounters": len(cohort),
            "unique_patients": cohort["patient_nbr"].nunique(),
        },
        {
            "stage": "patients_with_both_excluded_and_retained_encounters",
            "encounters": pd.NA,
            "unique_patients": len(
                excluded_patients & retained_patients
            ),
        },
        {
            "stage": "patients_only_in_excluded_encounters",
            "encounters": pd.NA,
            "unique_patients": len(
                excluded_patients - retained_patients
            ),
        },
    ]

    return pd.DataFrame(rows)


def add_binary_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create the binary <30-day readmission outcome."""
    data = dataframe.copy()

    valid_values = {"NO", ">30", "<30"}

    observed = set(
        data["readmitted"]
        .dropna()
        .unique()
    )

    unexpected = observed - valid_values

    if unexpected:
        raise ValueError(
            f"Unexpected readmitted values: {sorted(unexpected)}"
        )

    data["readmitted_30d"] = (
        data["readmitted"] == "<30"
    ).astype("int8")

    return data


def save_outputs(
    cohort: pd.DataFrame,
    flow: pd.DataFrame,
) -> None:
    """Save derived cohort outputs without modifying raw data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cohort.to_csv(COHORT_PATH, index=False)
    flow.to_csv(FLOW_PATH, index=False)


def main() -> None:
    """Construct and audit the primary analytic cohort."""
    raw = load_raw_data()

    cohort, excluded = apply_primary_cohort(raw)
    cohort = add_binary_target(cohort)

    flow = build_cohort_flow(
        raw=raw,
        cohort=cohort,
        excluded=excluded,
    )

    save_outputs(
        cohort=cohort,
        flow=flow,
    )

    print("\nPrimary cohort audit")
    print("=" * 60)

    print(flow.to_string(index=False))

    print("\nBinary target")
    print("=" * 60)

    print(
        cohort["readmitted_30d"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    prevalence = cohort["readmitted_30d"].mean()

    print(
        f"\n30-day readmission prevalence: "
        f"{prevalence:.2%}"
    )

    print("\nCohort saved:")
    print(f"- {COHORT_PATH}")

    print("\nCohort flow saved:")
    print(f"- {FLOW_PATH}")


if __name__ == "__main__":
    main()

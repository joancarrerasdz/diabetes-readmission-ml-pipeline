"""Reproducible feature engineering derived from the training cohort only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diabetes_readmission.preprocessing import (
    get_model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_cleaned.csv"
)

ENGINEERED_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_engineered.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = (
    REPORTS_DIR
    / "feature_engineering_summary.csv"
)

DIAGNOSIS_DISTRIBUTION_PATH = (
    REPORTS_DIR
    / "diagnosis_group_distribution.csv"
)

SPECIALTY_FREQUENCY_PATH = (
    REPORTS_DIR
    / "medical_specialty_frequency.csv"
)


AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

RAW_FEATURES_REPLACED = {
    "age",
    "diag_1",
    "diag_2",
    "diag_3",
}

ENGINEERED_FEATURES = {
    "age_midpoint",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
}

CLINICAL_GROUPS = {
    "Diabetes",
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Genitourinary",
    "Musculoskeletal",
    "Neoplasms",
    "Injury",
    "Other",
    "Missing",
}


def icd9_to_clinical_group(
    value: object,
) -> str:
    """Map an ICD-9 diagnosis code to a broad clinical group."""
    if pd.isna(value):
        return "Missing"

    code = str(value).strip()

    if code in {"", "Missing"}:
        return "Missing"

    upper = code.upper()

    # Supplementary V and E codes are retained as Other.
    if upper.startswith(("V", "E")):
        return "Other"

    try:
        major = int(float(code))
    except (ValueError, TypeError):
        return "Other"

    if major == 250:
        return "Diabetes"

    if 390 <= major <= 459 or major == 785:
        return "Circulatory"

    if 460 <= major <= 519 or major == 786:
        return "Respiratory"

    if 520 <= major <= 579 or major == 787:
        return "Digestive"

    if 580 <= major <= 629 or major == 788:
        return "Genitourinary"

    if 710 <= major <= 739:
        return "Musculoskeletal"

    if 140 <= major <= 239:
        return "Neoplasms"

    if 800 <= major <= 999:
        return "Injury"

    return "Other"


def engineer_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Replace selected raw variables with model-ready features."""
    data = dataframe.copy()

    required = {
        "age",
        "diag_1",
        "diag_2",
        "diag_3",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Missing required feature-engineering columns: "
            f"{sorted(missing)}"
        )

    unexpected_age = (
        set(data["age"].dropna().unique())
        - set(AGE_MIDPOINTS)
    )

    if unexpected_age:
        raise ValueError(
            "Unexpected age categories: "
            f"{sorted(unexpected_age)}"
        )

    data["age_midpoint"] = (
        data["age"]
        .map(AGE_MIDPOINTS)
        .astype("int64")
    )

    for column in [
        "diag_1",
        "diag_2",
        "diag_3",
    ]:
        data[f"{column}_group"] = (
            data[column]
            .map(icd9_to_clinical_group)
            .astype("string")
        )

    data = data.drop(
        columns=sorted(RAW_FEATURES_REPLACED)
    )

    return data


def build_diagnosis_distribution(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize engineered diagnosis groups."""
    outputs = []

    for variable in [
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]:
        counts = (
            data[variable]
            .value_counts(dropna=False)
            .rename_axis("clinical_group")
            .reset_index(name="encounters")
        )

        counts.insert(
            0,
            "variable",
            variable,
        )

        counts["proportion"] = (
            counts["encounters"]
            / len(data)
        )

        outputs.append(counts)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def build_specialty_frequency(
    cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """Audit medical-specialty frequencies before consolidation."""
    counts = (
        cleaned["medical_specialty"]
        .value_counts(dropna=False)
        .rename_axis("medical_specialty")
        .reset_index(name="encounters")
    )

    counts["proportion"] = (
        counts["encounters"]
        / len(cleaned)
    )

    counts["cumulative_proportion"] = (
        counts["proportion"].cumsum()
    )

    return counts


def build_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """Create an auditable feature-engineering summary."""
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
            "metric": "candidate_features_before",
            "value": len(
                get_model_feature_columns(before)
            ),
        },
        {
            "metric": "candidate_features_after",
            "value": len(
                get_model_feature_columns(after)
            ),
        },
    ]

    for feature in sorted(RAW_FEATURES_REPLACED):
        rows.append(
            {
                "metric": f"replaced_raw::{feature}",
                "value": 1,
            }
        )

    for feature in sorted(ENGINEERED_FEATURES):
        rows.append(
            {
                "metric": f"engineered::{feature}",
                "value": 1,
            }
        )

    return pd.DataFrame(rows)


def validate_engineered_data(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Validate feature-engineering invariants."""
    if len(before) != len(after):
        raise RuntimeError(
            "Feature engineering changed the number of encounters."
        )

    remaining_raw = (
        RAW_FEATURES_REPLACED
        & set(after.columns)
    )

    if remaining_raw:
        raise RuntimeError(
            "Raw replaced features remain: "
            f"{sorted(remaining_raw)}"
        )

    missing_engineered = (
        ENGINEERED_FEATURES
        - set(after.columns)
    )

    if missing_engineered:
        raise RuntimeError(
            "Engineered features missing: "
            f"{sorted(missing_engineered)}"
        )

    if after["age_midpoint"].isna().any():
        raise RuntimeError(
            "age_midpoint contains missing values."
        )

    for column in [
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]:
        if after[column].isna().any():
            raise RuntimeError(
                f"{column} contains missing values."
            )

        unexpected_groups = (
            set(after[column].unique())
            - CLINICAL_GROUPS
        )

        if unexpected_groups:
            raise RuntimeError(
                f"Unexpected groups in {column}: "
                f"{sorted(unexpected_groups)}"
            )

    if after["encounter_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate encounter_id detected."
        )


def save_outputs(
    engineered: pd.DataFrame,
    summary: pd.DataFrame,
    diagnosis_distribution: pd.DataFrame,
    specialty_frequency: pd.DataFrame,
) -> None:
    """Save local engineered data and versionable audit outputs."""
    ENGINEERED_TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    engineered.to_csv(
        ENGINEERED_TRAIN_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    diagnosis_distribution.to_csv(
        DIAGNOSIS_DISTRIBUTION_PATH,
        index=False,
    )

    specialty_frequency.to_csv(
        SPECIALTY_FREQUENCY_PATH,
        index=False,
    )


def main() -> None:
    """Run feature engineering on cleaned training data only."""
    cleaned = pd.read_csv(
        TRAIN_CLEAN_PATH,
        low_memory=False,
    )

    if len(cleaned) != 79_473:
        raise RuntimeError(
            f"Unexpected training size: {len(cleaned):,}"
        )

    engineered = engineer_features(cleaned)

    validate_engineered_data(
        before=cleaned,
        after=engineered,
    )

    summary = build_summary(
        before=cleaned,
        after=engineered,
    )

    diagnosis_distribution = (
        build_diagnosis_distribution(
            engineered
        )
    )

    specialty_frequency = (
        build_specialty_frequency(
            cleaned
        )
    )

    save_outputs(
        engineered=engineered,
        summary=summary,
        diagnosis_distribution=diagnosis_distribution,
        specialty_frequency=specialty_frequency,
    )

    print("\nFEATURE ENGINEERING — TRAINING ONLY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\nPrimary diagnosis groups")
    print("-" * 70)

    print(
        diagnosis_distribution.loc[
            diagnosis_distribution["variable"]
            == "diag_1_group"
        ].to_string(index=False)
    )

    print("\nMedical specialty — 20 most frequent")
    print("-" * 70)

    print(
        specialty_frequency
        .head(20)
        .to_string(index=False)
    )

    print("\nOutputs saved:")
    print(f"- {ENGINEERED_TRAIN_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {DIAGNOSIS_DISTRIBUTION_PATH}")
    print(f"- {SPECIALTY_FREQUENCY_PATH}")

    print(
        "\nThe held-out test set was not accessed."
    )


if __name__ == "__main__":
    main()

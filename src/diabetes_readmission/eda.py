"""Structured exploratory data-quality audit using training data only."""

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

REPORTS_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = REPORTS_DIR / "eda_train_summary.csv"
MISSINGNESS_PATH = REPORTS_DIR / "eda_missingness.csv"
CARDINALITY_PATH = REPORTS_DIR / "eda_cardinality.csv"
NEAR_CONSTANT_PATH = REPORTS_DIR / "eda_near_constant.csv"

SPECIAL_MISSING_MARKERS = {
    "?",
    "Unknown/Invalid",
    "Not Available",
    "NULL",
}


def load_training_data() -> pd.DataFrame:
    """Load the analytic cohort and retain training encounters only."""
    cohort = pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )

    assignments = pd.read_csv(
        SPLIT_PATH,
        usecols=[
            "encounter_id",
            "split",
        ],
    )

    merged = cohort.merge(
        assignments,
        on="encounter_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(cohort):
        raise RuntimeError(
            "Split assignments do not cover the full analytic cohort."
        )

    train = (
        merged.loc[merged["split"] == "train"]
        .drop(columns="split")
        .copy()
    )

    if train.empty:
        raise RuntimeError("Training dataset is empty.")

    return train


def build_dataset_summary(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Create high-level training-set audit statistics."""
    positive = int(train["readmitted_30d"].sum())

    rows = [
        {
            "metric": "encounters",
            "value": len(train),
        },
        {
            "metric": "unique_patients",
            "value": train["patient_nbr"].nunique(),
        },
        {
            "metric": "columns",
            "value": train.shape[1],
        },
        {
            "metric": "duplicate_encounter_id",
            "value": train["encounter_id"].duplicated().sum(),
        },
        {
            "metric": "duplicate_rows",
            "value": train.duplicated().sum(),
        },
        {
            "metric": "positive_30d",
            "value": positive,
        },
        {
            "metric": "negative_30d",
            "value": len(train) - positive,
        },
        {
            "metric": "prevalence_30d",
            "value": train["readmitted_30d"].mean(),
        },
    ]

    return pd.DataFrame(rows)


def build_missingness_summary(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Audit actual nulls and selected explicit missing-value markers."""
    rows = []

    for column in train.columns:
        series = train[column]

        string_series = series.astype("string")

        null_n = int(series.isna().sum())

        blank_n = int(
            string_series
            .str.strip()
            .eq("")
            .fillna(False)
            .sum()
        )

        special_marker_n = int(
            string_series
            .isin(SPECIAL_MISSING_MARKERS)
            .fillna(False)
            .sum()
        )

        rows.append(
            {
                "variable": column,
                "dtype": str(series.dtype),
                "null_n": null_n,
                "null_pct": null_n / len(train),
                "blank_n": blank_n,
                "special_missing_marker_n": special_marker_n,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "null_pct",
            "special_missing_marker_n",
        ],
        ascending=False,
    )


def build_cardinality_summary(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize variable cardinality and most common values."""
    rows = []

    for column in train.columns:
        series = train[column]

        counts = series.value_counts(
            dropna=False,
        )

        if len(counts):
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            top_pct = top_count / len(train)
        else:
            top_value = pd.NA
            top_count = 0
            top_pct = pd.NA

        rows.append(
            {
                "variable": column,
                "dtype": str(series.dtype),
                "n_unique_including_na": series.nunique(
                    dropna=False
                ),
                "most_common_value": top_value,
                "most_common_n": top_count,
                "most_common_pct": top_pct,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        "n_unique_including_na",
        ascending=False,
    )


def build_near_constant_summary(
    cardinality: pd.DataFrame,
) -> pd.DataFrame:
    """Flag constant and highly dominant variables for review."""
    result = cardinality.copy()

    result["constant"] = (
        result["n_unique_including_na"] <= 1
    )

    result["near_constant_99pct"] = (
        result["most_common_pct"] >= 0.99
    )

    return result.loc[
        result["constant"]
        | result["near_constant_99pct"]
    ].copy()


def save_reports(
    summary: pd.DataFrame,
    missingness: pd.DataFrame,
    cardinality: pd.DataFrame,
    near_constant: pd.DataFrame,
) -> None:
    """Save versionable EDA audit outputs."""
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    missingness.to_csv(
        MISSINGNESS_PATH,
        index=False,
    )

    cardinality.to_csv(
        CARDINALITY_PATH,
        index=False,
    )

    near_constant.to_csv(
        NEAR_CONSTANT_PATH,
        index=False,
    )


def main() -> None:
    """Run the structured training-only data-quality audit."""
    train = load_training_data()

    summary = build_dataset_summary(train)
    missingness = build_missingness_summary(train)
    cardinality = build_cardinality_summary(train)
    near_constant = build_near_constant_summary(cardinality)

    save_reports(
        summary=summary,
        missingness=missingness,
        cardinality=cardinality,
        near_constant=near_constant,
    )

    print("\nTRAINING-ONLY EDA AUDIT")
    print("=" * 70)

    print("\nDataset summary")
    print("-" * 70)
    print(summary.to_string(index=False))

    print("\nHighest missingness")
    print("-" * 70)
    print(
        missingness.head(15).to_string(
            index=False
        )
    )

    print("\nConstant / >=99% dominant variables")
    print("-" * 70)

    if near_constant.empty:
        print("None detected.")
    else:
        print(
            near_constant[
                [
                    "variable",
                    "n_unique_including_na",
                    "most_common_value",
                    "most_common_pct",
                ]
            ].to_string(index=False)
        )

    print("\nReports saved:")
    print(f"- {SUMMARY_PATH}")
    print(f"- {MISSINGNESS_PATH}")
    print(f"- {CARDINALITY_PATH}")
    print(f"- {NEAR_CONSTANT_PATH}")

    print(
        "\nHeld-out test encounters were not used "
        "for this exploratory audit."
    )


if __name__ == "__main__":
    main()

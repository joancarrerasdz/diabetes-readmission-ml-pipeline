"""Clinical exploratory analysis using the cleaned training cohort only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "train_cleaned.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

CATEGORICAL_REPORT_PATH = (
    REPORTS_DIR / "eda_outcome_by_category.csv"
)

NUMERIC_REPORT_PATH = (
    REPORTS_DIR / "eda_numeric_by_outcome.csv"
)


CATEGORICAL_VARIABLES = [
    "age",
    "gender",
    "race",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "insulin",
]

NUMERIC_VARIABLES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def load_training_data() -> pd.DataFrame:
    """Load cleaned training data only."""
    data = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
    )

    if len(data) != 79_473:
        raise RuntimeError(
            f"Unexpected training size: {len(data):,}"
        )

    return data


def categorical_outcome_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate outcome prevalence across selected categories."""
    outputs = []

    for variable in CATEGORICAL_VARIABLES:
        summary = (
            data
            .groupby(variable, dropna=False)
            ["readmitted_30d"]
            .agg(
                encounters="size",
                positive_30d="sum",
                prevalence_30d="mean",
            )
            .reset_index()
        )

        summary.insert(
            0,
            "variable",
            variable,
        )

        summary = summary.rename(
            columns={variable: "level"}
        )

        outputs.append(summary)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def numeric_outcome_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Describe numeric variables separately by outcome."""
    outputs = []

    for variable in NUMERIC_VARIABLES:
        summary = (
            data
            .groupby("readmitted_30d")[variable]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        summary.insert(
            0,
            "variable",
            variable,
        )

        outputs.append(summary)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def plot_target_distribution(
    data: pd.DataFrame,
) -> None:
    counts = (
        data["readmitted_30d"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        ["No <30-day readmission", "<30-day readmission"],
        counts.values,
    )

    ax.set_ylabel("Encounters")
    ax.set_title("30-day readmission outcome — training set")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "eda_target_distribution.png",
        dpi=200,
    )

    plt.close(fig)


def plot_age_prevalence(
    categorical: pd.DataFrame,
) -> None:
    age = categorical.loc[
        categorical["variable"] == "age"
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        age["level"],
        age["prevalence_30d"],
    )

    ax.set_ylabel("30-day readmission prevalence")
    ax.set_xlabel("Age band")
    ax.set_title(
        "Observed 30-day readmission prevalence by age — training set"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "eda_age_readmission_prevalence.png",
        dpi=200,
    )

    plt.close(fig)


def plot_prior_inpatient_prevalence(
    data: pd.DataFrame,
) -> None:
    grouped = data.copy()

    grouped["prior_inpatient_group"] = pd.cut(
        grouped["number_inpatient"],
        bins=[
            -1,
            0,
            1,
            2,
            5,
            float("inf"),
        ],
        labels=[
            "0",
            "1",
            "2",
            "3-5",
            "6+",
        ],
    )

    summary = (
        grouped
        .groupby(
            "prior_inpatient_group",
            observed=True,
        )["readmitted_30d"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        summary.index.astype(str),
        summary.values,
    )

    ax.set_xlabel(
        "Inpatient visits in preceding year"
    )

    ax.set_ylabel(
        "30-day readmission prevalence"
    )

    ax.set_title(
        "Observed readmission prevalence by prior inpatient use"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "eda_prior_inpatient_readmission_prevalence.png",
        dpi=200,
    )

    plt.close(fig)


def save_reports(
    categorical: pd.DataFrame,
    numeric: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    categorical.to_csv(
        CATEGORICAL_REPORT_PATH,
        index=False,
    )

    numeric.to_csv(
        NUMERIC_REPORT_PATH,
        index=False,
    )


def main() -> None:
    """Run descriptive clinical EDA on training data only."""
    data = load_training_data()

    categorical = categorical_outcome_summary(data)
    numeric = numeric_outcome_summary(data)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_reports(
        categorical=categorical,
        numeric=numeric,
    )

    plot_target_distribution(data)
    plot_age_prevalence(categorical)
    plot_prior_inpatient_prevalence(data)

    print("\nCLINICAL EDA — TRAINING SET ONLY")
    print("=" * 70)

    print(
        f"Encounters analysed: {len(data):,}"
    )

    print(
        f"Unique patients: "
        f"{data['patient_nbr'].nunique():,}"
    )

    print(
        "30-day readmission prevalence: "
        f"{data['readmitted_30d'].mean():.2%}"
    )

    print("\nAge")
    print("-" * 70)

    print(
        categorical.loc[
            categorical["variable"] == "age"
        ].to_string(index=False)
    )

    print("\nGender")
    print("-" * 70)

    print(
        categorical.loc[
            categorical["variable"] == "gender"
        ].to_string(index=False)
    )

    print("\nRace")
    print("-" * 70)

    print(
        categorical.loc[
            categorical["variable"] == "race"
        ].to_string(index=False)
    )

    print("\nPrior inpatient utilization")
    print("-" * 70)

    print(
        numeric.loc[
            numeric["variable"] == "number_inpatient"
        ].to_string(index=False)
    )

    print("\nReports saved:")
    print(f"- {CATEGORICAL_REPORT_PATH}")
    print(f"- {NUMERIC_REPORT_PATH}")

    print("\nFigures saved:")
    print("- figures/eda_target_distribution.png")
    print("- figures/eda_age_readmission_prevalence.png")
    print(
        "- figures/"
        "eda_prior_inpatient_readmission_prevalence.png"
    )

    print(
        "\nThese are descriptive associations in the "
        "training data, not causal effects."
    )

    print(
        "The held-out test set was not accessed."
    )


if __name__ == "__main__":
    main()

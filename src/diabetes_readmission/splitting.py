"""Patient-level data splitting for model development and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort.csv"
)

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
REPORTS_DIR = PROJECT_ROOT / "reports"

ASSIGNMENTS_PATH = INTERMEDIATE_DIR / "split_assignments.csv"
SUMMARY_PATH = REPORTS_DIR / "split_summary.csv"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_FOLD = 0


def load_cohort() -> pd.DataFrame:
    """Load the primary analytic cohort."""
    return pd.read_csv(
        COHORT_PATH,
        low_memory=False,
    )


def create_patient_level_split(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Assign encounters to train or test without patient overlap."""
    required = {
        "encounter_id",
        "patient_nbr",
        "readmitted_30d",
    }

    missing = required - set(cohort.columns)

    if missing:
        raise ValueError(
            f"Required split columns missing: {sorted(missing)}"
        )

    if cohort["patient_nbr"].isna().any():
        raise ValueError("patient_nbr contains missing values.")

    if cohort["readmitted_30d"].isna().any():
        raise ValueError("readmitted_30d contains missing values.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        splitter.split(
            X=cohort[["encounter_id"]],
            y=cohort["readmitted_30d"],
            groups=cohort["patient_nbr"],
        )
    )

    train_index, test_index = folds[TEST_FOLD]

    assignments = cohort[
        [
            "encounter_id",
            "patient_nbr",
            "readmitted_30d",
        ]
    ].copy()

    assignments["split"] = "unassigned"

    assignments.loc[
        assignments.index[train_index],
        "split",
    ] = "train"

    assignments.loc[
        assignments.index[test_index],
        "split",
    ] = "test"

    if (assignments["split"] == "unassigned").any():
        raise RuntimeError(
            "Some encounters were not assigned to a split."
        )

    return assignments


def validate_split(
    assignments: pd.DataFrame,
) -> None:
    """Validate patient independence and encounter assignment."""
    train = assignments.loc[
        assignments["split"] == "train"
    ]

    test = assignments.loc[
        assignments["split"] == "test"
    ]

    train_patients = set(train["patient_nbr"])
    test_patients = set(test["patient_nbr"])

    patient_overlap = train_patients & test_patients

    if patient_overlap:
        raise RuntimeError(
            "Patient leakage detected: "
            f"{len(patient_overlap)} patients appear in both splits."
        )

    train_encounters = set(train["encounter_id"])
    test_encounters = set(test["encounter_id"])

    encounter_overlap = train_encounters & test_encounters

    if encounter_overlap:
        raise RuntimeError(
            "Encounter leakage detected: "
            f"{len(encounter_overlap)} encounters appear in both splits."
        )

    if len(assignments) != len(train) + len(test):
        raise RuntimeError(
            "Split sizes do not sum to the full cohort."
        )


def build_split_summary(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sample size and target prevalence by split."""
    rows = []

    for split_name in ["train", "test"]:
        subset = assignments.loc[
            assignments["split"] == split_name
        ]

        positives = int(subset["readmitted_30d"].sum())

        rows.append(
            {
                "split": split_name,
                "encounters": len(subset),
                "unique_patients": subset["patient_nbr"].nunique(),
                "positive_30d": positives,
                "negative_30d": len(subset) - positives,
                "prevalence_30d": subset["readmitted_30d"].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save local assignments and versionable split audit summary."""
    INTERMEDIATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        ASSIGNMENTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )


def main() -> None:
    """Create and audit the frozen patient-level split."""
    cohort = load_cohort()

    assignments = create_patient_level_split(cohort)

    validate_split(assignments)

    summary = build_split_summary(assignments)

    save_outputs(
        assignments=assignments,
        summary=summary,
    )

    train_patients = set(
        assignments.loc[
            assignments["split"] == "train",
            "patient_nbr",
        ]
    )

    test_patients = set(
        assignments.loc[
            assignments["split"] == "test",
            "patient_nbr",
        ]
    )

    print("\nPatient-level split audit")
    print("=" * 70)

    print(summary.to_string(index=False))

    print(
        "\nPatient overlap between train and test: "
        f"{len(train_patients & test_patients)}"
    )

    print(
        "Total encounters assigned: "
        f"{len(assignments):,}"
    )

    print(
        "Random state: "
        f"{RANDOM_STATE}"
    )

    print(
        "Held-out fold: "
        f"{TEST_FOLD + 1}/{N_SPLITS}"
    )

    print("\nSplit assignments saved locally:")
    print(f"- {ASSIGNMENTS_PATH}")

    print("\nVersionable split summary:")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

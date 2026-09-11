"""Tests for patient-level splitting."""

import pandas as pd

from diabetes_readmission.splitting import (
    create_patient_level_split,
    validate_split,
)


def make_synthetic_cohort() -> pd.DataFrame:
    rows = []

    encounter_id = 1

    for patient_nbr in range(1000, 1030):
        target = 1 if patient_nbr % 5 == 0 else 0

        for _ in range(2):
            rows.append(
                {
                    "encounter_id": encounter_id,
                    "patient_nbr": patient_nbr,
                    "readmitted_30d": target,
                }
            )

            encounter_id += 1

    return pd.DataFrame(rows)


def test_patient_level_split_has_zero_patient_overlap():
    cohort = make_synthetic_cohort()

    assignments = create_patient_level_split(cohort)

    validate_split(assignments)

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

    assert train_patients.isdisjoint(test_patients)
    assert len(assignments) == len(cohort)
    assert set(assignments["split"]) == {"train", "test"}


def test_all_encounters_from_one_patient_stay_together():
    cohort = make_synthetic_cohort()

    assignments = create_patient_level_split(cohort)

    splits_per_patient = (
        assignments
        .groupby("patient_nbr")["split"]
        .nunique()
    )

    assert (splits_per_patient == 1).all()

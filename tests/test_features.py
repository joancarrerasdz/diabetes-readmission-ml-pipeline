"""Tests for deterministic clinical feature engineering."""

import pandas as pd

from diabetes_readmission.features import (
    AGE_MIDPOINTS,
    CLINICAL_GROUPS,
    engineer_features,
    icd9_to_clinical_group,
)


def test_age_midpoint_mapping():
    assert AGE_MIDPOINTS["[0-10)"] == 5
    assert AGE_MIDPOINTS["[50-60)"] == 55
    assert AGE_MIDPOINTS["[90-100)"] == 95


def test_icd9_clinical_group_mapping():
    cases = {
        "250.6": "Diabetes",
        "428": "Circulatory",
        "486": "Respiratory",
        "530": "Digestive",
        "585": "Genitourinary",
        "715": "Musculoskeletal",
        "174": "Neoplasms",
        "820": "Injury",
        "V45": "Other",
        "E849": "Other",
        "Missing": "Missing",
        None: "Missing",
    }

    for code, expected in cases.items():
        assert icd9_to_clinical_group(code) == expected


def test_engineer_features_replaces_raw_variables():
    raw = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [101, 102],
            "readmitted_30d": [0, 1],
            "age": ["[50-60)", "[70-80)"],
            "diag_1": ["250.6", "428"],
            "diag_2": ["486", "Missing"],
            "diag_3": ["715", "820"],
            "medical_specialty": [
                "InternalMedicine",
                "Missing",
            ],
        }
    )

    engineered = engineer_features(raw)

    assert len(engineered) == len(raw)

    for raw_column in {
        "age",
        "diag_1",
        "diag_2",
        "diag_3",
    }:
        assert raw_column not in engineered.columns

    assert engineered["age_midpoint"].tolist() == [55, 75]

    assert engineered["diag_1_group"].tolist() == [
        "Diabetes",
        "Circulatory",
    ]

    assert engineered["diag_2_group"].tolist() == [
        "Respiratory",
        "Missing",
    ]

    assert engineered["diag_3_group"].tolist() == [
        "Musculoskeletal",
        "Injury",
    ]


def test_engineered_diagnosis_groups_are_valid():
    raw = pd.DataFrame(
        {
            "age": ["[60-70)"],
            "diag_1": ["250.13"],
            "diag_2": ["786"],
            "diag_3": ["E849"],
        }
    )

    engineered = engineer_features(raw)

    for column in [
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]:
        assert set(engineered[column]).issubset(
            CLINICAL_GROUPS
        )

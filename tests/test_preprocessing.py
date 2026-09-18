"""Tests for deterministic preprocessing rules."""

import pandas as pd

from diabetes_readmission.preprocessing import (
    DROP_COLUMNS,
    clean_dataframe,
    get_model_feature_columns,
)


def make_example_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [101, 102],
            "discharge_disposition_id": [1, 1],
            "readmitted": ["NO", "<30"],
            "readmitted_30d": [0, 1],
            "weight": [pd.NA, "[75-100)"],
            "citoglipton": ["No", "No"],
            "examide": ["No", "No"],
            "glimepiride-pioglitazone": ["No", "No"],
            "max_glu_serum": [pd.NA, "Norm"],
            "A1Cresult": [pd.NA, ">8"],
            "race": [pd.NA, "Caucasian"],
            "gender": ["Unknown/Invalid", "Female"],
            "payer_code": [pd.NA, "MC"],
            "medical_specialty": [pd.NA, "InternalMedicine"],
            "diag_1": [pd.NA, "250.6"],
            "diag_2": ["401", pd.NA],
            "diag_3": [pd.NA, "428"],
            "age": ["[60-70)", "[70-80)"],
        }
    )


def test_cleaning_preserves_rows_and_drops_expected_columns():
    raw = make_example_data()

    cleaned = clean_dataframe(raw)

    assert len(cleaned) == len(raw)
    assert DROP_COLUMNS.isdisjoint(cleaned.columns)


def test_laboratory_missingness_becomes_not_measured():
    cleaned = clean_dataframe(make_example_data())

    assert cleaned.loc[0, "max_glu_serum"] == "Not_measured"
    assert cleaned.loc[0, "A1Cresult"] == "Not_measured"


def test_explicit_missing_categories_are_normalized():
    cleaned = clean_dataframe(make_example_data())

    assert cleaned.loc[0, "race"] == "Missing"
    assert cleaned.loc[0, "gender"] == "Missing"
    assert cleaned.loc[0, "payer_code"] == "Missing"
    assert cleaned.loc[0, "medical_specialty"] == "Missing"
    assert cleaned.loc[0, "diag_1"] == "Missing"
    assert cleaned.loc[1, "diag_2"] == "Missing"


def test_non_predictor_columns_are_not_model_features():
    cleaned = clean_dataframe(make_example_data())

    features = get_model_feature_columns(cleaned)

    forbidden = {
        "encounter_id",
        "patient_nbr",
        "discharge_disposition_id",
        "readmitted",
        "readmitted_30d",
    }

    assert forbidden.isdisjoint(features)

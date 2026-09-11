"""Tests for cohort definition and target construction."""

import pandas as pd

from diabetes_readmission.cohort import (
    DISCHARGE_EXCLUSION_CODES,
    add_binary_target,
    apply_primary_cohort,
)


def test_primary_cohort_excludes_only_prespecified_dispositions():
    data = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [101, 102, 103, 104],
            "discharge_disposition_id": [1, 11, 13, 6],
            "readmitted": ["NO", "<30", ">30", "<30"],
        }
    )

    cohort, excluded = apply_primary_cohort(data)

    assert set(excluded["encounter_id"]) == {2, 3}
    assert set(cohort["encounter_id"]) == {1, 4}

    assert set(DISCHARGE_EXCLUSION_CODES) == {
        11,
        13,
        14,
        19,
        20,
        21,
    }


def test_binary_target_mapping():
    data = pd.DataFrame(
        {
            "readmitted": ["<30", ">30", "NO"]
        }
    )

    result = add_binary_target(data)

    assert result["readmitted_30d"].tolist() == [1, 0, 0]

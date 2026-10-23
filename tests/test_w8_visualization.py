from pathlib import Path

import numpy as np
import pytest

from diabetes_readmission.w8_visualization import (
    CALIBRATION_FIGURE,
    CONFUSION_FIGURE,
    PERFORMANCE_FIGURE,
    load_calibration,
    load_confusion_matrix,
    load_final_metrics,
)


EXPECTED_ROC_AUC = 0.6450059118364249
EXPECTED_PR_AUC = 0.19238337331108202
EXPECTED_BRIER = 0.09801970553557973
EXPECTED_BALANCED_ACCURACY = 0.6020187884237682
EXPECTED_THRESHOLD = 0.110

EXPECTED_CONFUSION_MATRIX = np.array(
    [
        [10766, 6841],
        [922, 1341],
    ],
    dtype=int,
)


def test_final_metrics_match_locked_w7_results():
    metrics = load_final_metrics()

    assert float(metrics["roc_auc"]) == pytest.approx(
        EXPECTED_ROC_AUC
    )
    assert float(metrics["pr_auc"]) == pytest.approx(
        EXPECTED_PR_AUC
    )
    assert float(metrics["brier_score"]) == pytest.approx(
        EXPECTED_BRIER
    )
    assert float(
        metrics["balanced_accuracy"]
    ) == pytest.approx(
        EXPECTED_BALANCED_ACCURACY
    )
    assert float(
        metrics["operating_threshold"]
    ) == pytest.approx(
        EXPECTED_THRESHOLD
    )


def test_final_confusion_matrix_matches_w7_results():
    matrix = load_confusion_matrix()

    assert matrix.shape == (2, 2)

    np.testing.assert_array_equal(
        matrix,
        EXPECTED_CONFUSION_MATRIX,
    )

    assert int(matrix.sum()) == 19870


def test_final_calibration_data_are_valid_probabilities():
    predicted, observed = load_calibration()

    assert len(predicted) == len(observed)
    assert len(predicted) > 0

    assert predicted.notna().all()
    assert observed.notna().all()

    assert predicted.between(0, 1).all()
    assert observed.between(0, 1).all()


@pytest.mark.parametrize(
    "figure_path",
    [
        PERFORMANCE_FIGURE,
        CONFUSION_FIGURE,
        CALIBRATION_FIGURE,
    ],
)
def test_final_figure_exists_and_is_nonempty(
    figure_path: Path,
):
    assert figure_path.exists()
    assert figure_path.is_file()

    assert figure_path.stat().st_size > 10_000

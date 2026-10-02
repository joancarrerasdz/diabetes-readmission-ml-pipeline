"""Diagnostics for baseline out-of-fold predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "baseline_oof_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

PROBABILITY_SUMMARY_PATH = (
    REPORTS_DIR
    / "baseline_probability_summary.csv"
)

THRESHOLD_SWEEP_PATH = (
    REPORTS_DIR
    / "baseline_threshold_sweep.csv"
)

CALIBRATION_PATH = (
    REPORTS_DIR
    / "baseline_calibration.csv"
)

PROBABILITY_FIGURE_PATH = (
    FIGURES_DIR
    / "baseline_logistic_probability_distribution.png"
)

CALIBRATION_FIGURE_PATH = (
    FIGURES_DIR
    / "baseline_logistic_calibration.png"
)

TARGET = "readmitted_30d"
PROBABILITY = "logistic_regression_probability"


def probability_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize OOF probabilities overall and by outcome."""
    rows = []

    groups = [
        ("all", dataframe),
        (
            "negative_30d",
            dataframe.loc[
                dataframe[TARGET] == 0
            ],
        ),
        (
            "positive_30d",
            dataframe.loc[
                dataframe[TARGET] == 1
            ],
        ),
    ]

    for label, subset in groups:
        values = subset[PROBABILITY]

        rows.append(
            {
                "group": label,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate candidate thresholds descriptively on OOF predictions."""
    y_true = dataframe[TARGET].to_numpy(
        dtype=int
    )

    probabilities = dataframe[
        PROBABILITY
    ].to_numpy(dtype=float)

    thresholds = np.arange(
        0.05,
        0.505,
        0.025,
    )

    rows = []

    for threshold in thresholds:
        predicted = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted,
            labels=[0, 1],
        ).ravel()

        sensitivity = recall_score(
            y_true,
            predicted,
            zero_division=0,
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        precision = precision_score(
            y_true,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predicted,
            zero_division=0,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                predicted,
            )
        )

        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "predicted_positive_rate": (
                    predicted.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def calibration_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a quantile-binned calibration table."""
    y_true = dataframe[TARGET].to_numpy(
        dtype=int
    )

    probabilities = dataframe[
        PROBABILITY
    ].to_numpy(dtype=float)

    observed, predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
    )

    return pd.DataFrame(
        {
            "bin": np.arange(
                1,
                len(observed) + 1,
            ),
            "mean_predicted_probability": (
                predicted
            ),
            "observed_event_rate": observed,
            "calibration_difference": (
                observed - predicted
            ),
        }
    )


def save_probability_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Plot OOF probability distributions by observed outcome."""
    negative = dataframe.loc[
        dataframe[TARGET] == 0,
        PROBABILITY,
    ]

    positive = dataframe.loc[
        dataframe[TARGET] == 1,
        PROBABILITY,
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        negative,
        bins=50,
        alpha=0.55,
        density=True,
        label="No readmission <30d",
    )

    plt.hist(
        positive,
        bins=50,
        alpha=0.55,
        density=True,
        label="Readmission <30d",
    )

    plt.xlabel(
        "Logistic regression OOF probability"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Baseline logistic regression — OOF risk distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PROBABILITY_FIGURE_PATH,
        dpi=180,
    )

    plt.close()


def save_calibration_figure(
    calibration: pd.DataFrame,
) -> None:
    """Plot observed versus predicted OOF risk."""
    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        calibration[
            "mean_predicted_probability"
        ],
        calibration[
            "observed_event_rate"
        ],
        marker="o",
        label="Logistic regression OOF",
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed 30-day readmission rate"
    )

    plt.title(
        "Baseline logistic regression — OOF calibration"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=180,
    )

    plt.close()


def main() -> None:
    """Generate training-only baseline diagnostic artifacts."""
    oof = pd.read_csv(
        OOF_PATH
    )

    required = {
        "encounter_id",
        "patient_nbr",
        TARGET,
        "cv_fold",
        PROBABILITY,
    }

    missing = required - set(
        oof.columns
    )

    if missing:
        raise RuntimeError(
            "Missing required OOF columns: "
            f"{sorted(missing)}"
        )

    if len(oof) != 79_473:
        raise RuntimeError(
            "Unexpected OOF encounter count."
        )

    if (
        oof["patient_nbr"].nunique()
        != 55_952
    ):
        raise RuntimeError(
            "Unexpected OOF patient count."
        )

    if oof[PROBABILITY].isna().any():
        raise RuntimeError(
            "OOF probabilities contain missing values."
        )

    if not oof[
        PROBABILITY
    ].between(0, 1).all():
        raise RuntimeError(
            "Invalid probability outside [0, 1]."
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities = probability_summary(
        oof
    )

    thresholds = threshold_sweep(
        oof
    )

    calibration = calibration_summary(
        oof
    )

    probabilities.to_csv(
        PROBABILITY_SUMMARY_PATH,
        index=False,
    )

    thresholds.to_csv(
        THRESHOLD_SWEEP_PATH,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    save_probability_figure(
        oof
    )

    save_calibration_figure(
        calibration
    )

    print(
        "\nBASELINE OOF DIAGNOSTICS"
    )

    print("=" * 72)

    print(
        "\nProbability summary"
    )

    print("-" * 72)

    print(
        probabilities.to_string(
            index=False
        )
    )

    print(
        "\nCalibration table"
    )

    print("-" * 72)

    print(
        calibration.to_string(
            index=False
        )
    )

    print(
        "\nSelected threshold rows "
        "(diagnostic only)"
    )

    print("-" * 72)

    selected = thresholds.loc[
        thresholds["threshold"].round(3).isin(
            [
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
            ]
        )
    ]

    print(
        selected.to_string(
            index=False
        )
    )

    print(
        "\nNo operating threshold "
        "was selected or locked."
    )

    print(
        "The held-out test set "
        "was not accessed."
    )


if __name__ == "__main__":
    main()

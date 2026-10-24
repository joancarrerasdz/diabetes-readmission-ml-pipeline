from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

METRICS_PATH = REPORTS_DIR / "w7_final_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "w7_final_confusion_matrix.csv"
CALIBRATION_PATH = REPORTS_DIR / "w7_final_calibration.csv"

PERFORMANCE_FIGURE = (
    FIGURES_DIR / "w7_final_performance_summary.png"
)
CONFUSION_FIGURE = (
    FIGURES_DIR / "w7_final_confusion_matrix.png"
)
CALIBRATION_FIGURE = (
    FIGURES_DIR / "w7_final_calibration.png"
)


def first_existing_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str:
    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "None of the expected columns were found: "
        f"{candidates}. Available columns: "
        f"{frame.columns.tolist()}"
    )


def load_final_metrics() -> pd.Series:
    frame = pd.read_csv(METRICS_PATH)

    if len(frame) != 1:
        raise ValueError(
            "Expected exactly one row in "
            f"{METRICS_PATH}, found {len(frame)}."
        )

    return frame.iloc[0]


def load_confusion_matrix() -> np.ndarray:
    frame = pd.read_csv(CONFUSION_PATH)

    if len(frame) != 1:
        raise ValueError(
            "Expected exactly one row in "
            f"{CONFUSION_PATH}, found {len(frame)}."
        )

    required = [
        "actual_negative_predicted_negative",
        "actual_negative_predicted_positive",
        "actual_positive_predicted_negative",
        "actual_positive_predicted_positive",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing confusion-matrix columns: "
            f"{missing}"
        )

    row = frame.iloc[0]

    return np.array(
        [
            [
                row["actual_negative_predicted_negative"],
                row["actual_negative_predicted_positive"],
            ],
            [
                row["actual_positive_predicted_negative"],
                row["actual_positive_predicted_positive"],
            ],
        ],
        dtype=int,
    )


def load_calibration() -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(CALIBRATION_PATH)

    predicted_column = first_existing_column(
        frame,
        (
            "mean_predicted_probability",
            "mean_probability",
            "predicted_probability",
            "mean_predicted_risk",
        ),
    )

    observed_column = first_existing_column(
        frame,
        (
            "observed_event_rate",
            "observed_rate",
            "event_rate",
            "observed_probability",
        ),
    )

    predicted = pd.to_numeric(
        frame[predicted_column],
        errors="raise",
    )

    observed = pd.to_numeric(
        frame[observed_column],
        errors="raise",
    )

    if not predicted.between(0, 1).all():
        raise ValueError(
            "Calibration predicted probabilities "
            "must be between 0 and 1."
        )

    if not observed.between(0, 1).all():
        raise ValueError(
            "Calibration observed event rates "
            "must be between 0 and 1."
        )

    return predicted, observed


def plot_performance_summary(
    metrics: pd.Series,
) -> None:
    selected_metrics = {
        "PR-AUC": float(metrics["pr_auc"]),
        "ROC-AUC": float(metrics["roc_auc"]),
        "Sensitivity": float(metrics["sensitivity"]),
        "Specificity": float(metrics["specificity"]),
        "Precision": float(metrics["precision"]),
        "F1": float(metrics["f1"]),
        "Balanced\naccuracy": float(
            metrics["balanced_accuracy"]
        ),
    }

    labels = list(selected_metrics.keys())
    values = list(selected_metrics.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title(
        "Final held-out performance — Week 7"
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.97),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    threshold = float(
        metrics["operating_threshold"]
    )

    prevalence = float(metrics["prevalence_30d"])

    fig.text(
        0.5,
        0.01,
        (
            f"Locked operating threshold = "
            f"{threshold:.3f} | "
            f"Held-out prevalence = "
            f"{prevalence:.3f}"
        ),
        ha="center",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(
        PERFORMANCE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix)

    fig.colorbar(
        image,
        ax=ax,
        label="Number of encounters",
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Predicted negative", "Predicted positive"]
    )
    ax.set_yticklabels(
        ["Actual negative", "Actual positive"]
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")

    ax.set_title(
        "Final held-out confusion matrix "
        "(threshold = 0.110)"
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_calibration() -> None:
    predicted, observed = load_calibration()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.plot(
        predicted,
        observed,
        marker="o",
        label="Final sigmoid-calibrated model",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed 30-day readmission rate")

    ax.set_title(
        "Final held-out calibration — Week 7"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        CALIBRATION_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_final_metrics()
    confusion_matrix = load_confusion_matrix()

    plot_performance_summary(metrics)
    plot_confusion_matrix(confusion_matrix)
    plot_calibration()

    print("W8 FINAL VISUALIZATION")
    print("=" * 72)

    print(
        f"ROC-AUC: "
        f"{float(metrics['roc_auc']):.6f}"
    )
    print(
        f"PR-AUC: "
        f"{float(metrics['pr_auc']):.6f}"
    )
    print(
        f"Brier score: "
        f"{float(metrics['brier_score']):.6f}"
    )
    print(
        f"Balanced accuracy: "
        f"{float(metrics['balanced_accuracy']):.6f}"
    )
    print(
        f"Operating threshold: "
        f"{float(metrics['operating_threshold']):.3f}"
    )

    print("\nFigures saved:")

    for path in (
        PERFORMANCE_FIGURE,
        CONFUSION_FIGURE,
        CALIBRATION_FIGURE,
    ):
        print(f"- {path}")

    print(
        "\nNo model fitting, calibration fitting, "
        "threshold optimisation or held-out "
        "decision making was performed."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_calibrated_oof_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

FOLD_METRICS_PATH = (
    REPORTS_DIR
    / "w6_threshold_fold_metrics.csv"
)

SUMMARY_PATH = (
    REPORTS_DIR
    / "w6_threshold_summary.csv"
)

FINDINGS_PATH = (
    REPORTS_DIR
    / "w6_threshold_findings.md"
)


TARGET_COLUMN = "readmitted_30d"
GROUP_COLUMN = "patient_nbr"
FOLD_COLUMN = "outer_fold"
PROBABILITY_COLUMN = "sigmoid_probability"

EXPECTED_ROWS = 79_473
EXPECTED_PATIENTS = 55_952
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

THRESHOLD_START = 0.010
THRESHOLD_STOP = 0.500
THRESHOLD_STEP = 0.005

SELECTION_DECIMALS = 12


def candidate_thresholds() -> np.ndarray:
    """
    Return the frozen threshold grid:
    0.010 to 0.500 inclusive in increments of 0.005.
    """
    n_steps = int(
        round(
            (
                THRESHOLD_STOP
                - THRESHOLD_START
            )
            / THRESHOLD_STEP
        )
    )

    return np.round(
        THRESHOLD_START
        + np.arange(n_steps + 1)
        * THRESHOLD_STEP,
        3,
    )


def validate_input(
    frame: pd.DataFrame,
) -> None:
    """
    Validate the patient-safe OOF input used for
    threshold selection.
    """
    required = {
        "encounter_id",
        GROUP_COLUMN,
        TARGET_COLUMN,
        FOLD_COLUMN,
        PROBABILITY_COLUMN,
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected row count: "
            f"{len(frame):,}; "
            f"expected {EXPECTED_ROWS:,}."
        )

    unique_patients = (
        frame[GROUP_COLUMN]
        .nunique()
    )

    if (
        unique_patients
        != EXPECTED_PATIENTS
    ):
        raise ValueError(
            "Unexpected unique-patient "
            "count: "
            f"{unique_patients:,}; "
            f"expected "
            f"{EXPECTED_PATIENTS:,}."
        )

    observed_folds = set(
        frame[FOLD_COLUMN]
        .unique()
    )

    if (
        observed_folds
        != EXPECTED_FOLDS
    ):
        raise ValueError(
            "Unexpected outer folds: "
            f"{sorted(observed_folds)}."
        )

    max_folds_per_patient = (
        frame
        .groupby(
            GROUP_COLUMN
        )[FOLD_COLUMN]
        .nunique()
        .max()
    )

    if (
        max_folds_per_patient
        != 1
    ):
        raise ValueError(
            "Patient leakage detected "
            "across outer folds: "
            "maximum folds per patient "
            f"= {max_folds_per_patient}."
        )

    audit_columns = [
        GROUP_COLUMN,
        TARGET_COLUMN,
        FOLD_COLUMN,
        PROBABILITY_COLUMN,
    ]

    if (
        frame[audit_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing values detected "
            "in threshold-selection "
            "columns."
        )

    target_values = set(
        frame[TARGET_COLUMN]
        .unique()
    )

    if not target_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Target must contain "
            "only 0/1."
        )

    if not (
        frame[
            PROBABILITY_COLUMN
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):
        raise ValueError(
            f"{PROBABILITY_COLUMN} "
            "must be bounded in [0, 1]."
        )


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    Calculate threshold-dependent metrics.
    """
    predicted = (
        probability
        >= threshold
    ).astype(
        np.int8
    )

    tp = int(
        (
            (y_true == 1)
            & (predicted == 1)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (predicted == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (predicted == 0)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (predicted == 0)
        ).sum()
    )

    sensitivity = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    f1 = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2.0

    predicted_positive_rate = (
        float(
            predicted.mean()
        )
    )

    return {
        "threshold": float(
            threshold
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy":
            balanced_accuracy,
        "predicted_positive_rate":
            predicted_positive_rate,
    }


def build_fold_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate every frozen threshold
    independently in each outer fold.
    """
    records = []

    for fold in sorted(
        EXPECTED_FOLDS
    ):
        fold_frame = frame.loc[
            frame[
                FOLD_COLUMN
            ]
            == fold
        ]

        y_true = (
            fold_frame[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=np.int8
            )
        )

        probability = (
            fold_frame[
                PROBABILITY_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        for threshold in (
            candidate_thresholds()
        ):
            metrics = (
                calculate_threshold_metrics(
                    y_true=y_true,
                    probability=probability,
                    threshold=float(
                        threshold
                    ),
                )
            )

            records.append(
                {
                    "outer_fold":
                        int(fold),
                    "n":
                        int(
                            len(
                                fold_frame
                            )
                        ),
                    "positive_30d":
                        int(
                            y_true.sum()
                        ),
                    "prevalence_30d":
                        float(
                            y_true.mean()
                        ),
                    **metrics,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def build_summary(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate threshold performance
    across the five outer folds.
    """
    metric_names = [
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]

    grouped = (
        fold_metrics
        .groupby(
            "threshold",
            sort=True,
        )
    )

    rows = []

    for threshold, part in grouped:
        row = {
            "threshold":
                float(threshold),
            "n_folds":
                int(
                    part[
                        "outer_fold"
                    ]
                    .nunique()
                ),
        }

        for metric in metric_names:
            row[
                f"mean_{metric}"
            ] = float(
                part[metric]
                .mean()
            )

            row[
                f"sd_{metric}"
            ] = float(
                part[metric]
                .std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "threshold"
        )
        .reset_index(
            drop=True
        )
    )


def select_threshold(
    summary: pd.DataFrame,
) -> pd.Series:
    """
    Apply the frozen deterministic
    selection hierarchy:

    1. highest mean balanced accuracy;
    2. highest mean sensitivity;
    3. highest mean F1;
    4. lowest threshold.
    """
    required = {
        "threshold",
        "mean_balanced_accuracy",
        "mean_sensitivity",
        "mean_f1",
    }

    missing = (
        required
        .difference(
            summary.columns
        )
    )

    if missing:
        raise ValueError(
            "Summary is missing "
            "selection columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    ranked = (
        summary.copy()
    )

    ranked[
        "_rank_balanced_accuracy"
    ] = ranked[
        "mean_balanced_accuracy"
    ].round(
        SELECTION_DECIMALS
    )

    ranked[
        "_rank_sensitivity"
    ] = ranked[
        "mean_sensitivity"
    ].round(
        SELECTION_DECIMALS
    )

    ranked[
        "_rank_f1"
    ] = ranked[
        "mean_f1"
    ].round(
        SELECTION_DECIMALS
    )

    ranked = ranked.sort_values(
        by=[
            "_rank_balanced_accuracy",
            "_rank_sensitivity",
            "_rank_f1",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    )

    return (
        ranked
        .iloc[0]
        .drop(
            labels=[
                "_rank_balanced_accuracy",
                "_rank_sensitivity",
                "_rank_f1",
            ]
        )
    )


def write_findings(
    selected: pd.Series,
    fold_metrics: pd.DataFrame,
) -> None:
    threshold = float(
        selected["threshold"]
    )

    selected_folds = (
        fold_metrics.loc[
            np.isclose(
                fold_metrics[
                    "threshold"
                ],
                threshold,
            )
        ]
        .copy()
    )

    lines = [
        "# Week 6 — Operating-threshold sweep findings",
        "",
        "## Frozen input",
        "",
        "- candidate model: Random Forest",
        "- calibration strategy: sigmoid",
        "- probability source: `sigmoid_probability`",
        "- held-out test set: not accessed",
        "",
        "## Frozen threshold grid",
        "",
        "- minimum: 0.010",
        "- maximum: 0.500",
        "- increment: 0.005",
        (
            "- candidate thresholds: "
            f"{len(candidate_thresholds())}"
        ),
        "",
        "## Protocol-derived recommendation",
        "",
        (
            "**Operating threshold: "
            f"{threshold:.3f}**"
        ),
        "",
        "The threshold was selected mechanically by:",
        "",
        "1. highest mean balanced accuracy across the five outer folds;",
        "2. higher mean sensitivity if tied;",
        "3. higher mean F1 if still tied;",
        "4. lower threshold if still tied.",
        "",
        "Aggregate metrics at the selected threshold:",
        "",
        (
            "- mean balanced accuracy: "
            f"{selected['mean_balanced_accuracy']:.6f}"
        ),
        (
            "- SD balanced accuracy: "
            f"{selected['sd_balanced_accuracy']:.6f}"
        ),
        (
            "- mean sensitivity: "
            f"{selected['mean_sensitivity']:.6f}"
        ),
        (
            "- mean specificity: "
            f"{selected['mean_specificity']:.6f}"
        ),
        (
            "- mean precision: "
            f"{selected['mean_precision']:.6f}"
        ),
        (
            "- mean F1: "
            f"{selected['mean_f1']:.6f}"
        ),
        (
            "- mean predicted-positive rate: "
            f"{selected['mean_predicted_positive_rate']:.6f}"
        ),
        "",
        "## Fold-level stability",
        "",
        (
            "| outer fold | balanced accuracy | "
            "sensitivity | specificity | F1 | "
            "predicted-positive rate |"
        ),
        (
            "| ---: | ---: | ---: | ---: | "
            "---: | ---: |"
        ),
    ]

    for _, row in (
        selected_folds
        .sort_values(
            "outer_fold"
        )
        .iterrows()
    ):
        lines.append(
            "| "
            f"{int(row['outer_fold'])} | "
            f"{row['balanced_accuracy']:.6f} | "
            f"{row['sensitivity']:.6f} | "
            f"{row['specificity']:.6f} | "
            f"{row['f1']:.6f} | "
            f"{row['predicted_positive_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Safeguards",
            "",
            "- No model-family change was made.",
            "- No hyperparameter change was made.",
            "- No preprocessing or feature change was made.",
            "- No recalibration was performed.",
            "- The held-out test set was not accessed.",
            "",
            (
                "This is the protocol-derived threshold "
                "recommendation. It is not the final "
                "threshold lock until explicitly "
                "versioned in a separate lock artifact."
            ),
            "",
        ]
    )

    FINDINGS_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    frame = pd.read_csv(
        OOF_PATH
    )

    validate_input(
        frame
    )

    fold_metrics = (
        build_fold_metrics(
            frame
        )
    )

    summary = (
        build_summary(
            fold_metrics
        )
    )

    selected = (
        select_threshold(
            summary
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
        float_format="%.12f",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        float_format="%.12f",
    )

    write_findings(
        selected,
        fold_metrics,
    )

    print(
        "W6 OPERATING-THRESHOLD SWEEP"
    )
    print(
        "=" * 72
    )

    print(
        f"OOF encounters: "
        f"{len(frame):,}"
    )

    print(
        "Unique patients: "
        f"{frame[GROUP_COLUMN].nunique():,}"
    )

    print(
        "Outer folds:",
        sorted(
            frame[
                FOLD_COLUMN
            ].unique()
        ),
    )

    print(
        "Candidate thresholds:",
        len(
            candidate_thresholds()
        ),
        (
            f"({THRESHOLD_START:.3f}"
            f"–{THRESHOLD_STOP:.3f}, "
            f"step "
            f"{THRESHOLD_STEP:.3f})"
        ),
    )

    print(
        "\nProtocol-derived recommendation"
    )
    print(
        "-" * 72
    )

    display_columns = [
        "threshold",
        "mean_balanced_accuracy",
        "sd_balanced_accuracy",
        "mean_sensitivity",
        "mean_specificity",
        "mean_precision",
        "mean_f1",
        "mean_predicted_positive_rate",
    ]

    print(
        selected[
            display_columns
        ].to_string()
    )

    print(
        "\nVersionable reports saved:"
    )

    print(
        f"- {FOLD_METRICS_PATH}"
    )

    print(
        f"- {SUMMARY_PATH}"
    )

    print(
        f"- {FINDINGS_PATH}"
    )

    print(
        "\nNo threshold has been locked "
        "by this command."
    )

    print(
        "The held-out test set "
        "was not accessed."
    )


if __name__ == "__main__":
    main()

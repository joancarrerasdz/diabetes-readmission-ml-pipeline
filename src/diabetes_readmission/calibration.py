"""Week-6 leakage-safe probability calibration audit.

Calibration is evaluated exclusively from patient-safe out-of-fold
predictions generated during nested cross-validation.

The held-out test set is never accessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_model_selection_oof_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

FOLD_METRICS_PATH = (
    REPORTS_DIR
    / "w6_calibration_fold_metrics.csv"
)

SUMMARY_PATH = (
    REPORTS_DIR
    / "w6_calibration_summary.csv"
)

DECISION_PATH = (
    REPORTS_DIR
    / "w6_calibration_findings.md"
)

CALIBRATED_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "w6_calibrated_oof_predictions.csv"
)


TARGET_COLUMN = "readmitted_30d"
FOLD_COLUMN = "outer_fold"
PATIENT_COLUMN = "patient_nbr"
BASE_PROBABILITY_COLUMN = "random_forest_probability"

METHODS = (
    "uncalibrated",
    "sigmoid",
    "isotonic",
)

EPSILON = 1e-6
N_ECE_BINS = 10
RANDOM_STATE = 42


def clip_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        EPSILON,
        1.0 - EPSILON,
    )


def probabilities_to_logit(
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = clip_probabilities(
        probabilities
    )

    return np.log(
        probabilities
        / (1.0 - probabilities)
    )


def fit_sigmoid_calibrator(
    probabilities: np.ndarray,
    target: np.ndarray,
) -> LogisticRegression:
    """Fit logistic recalibration on model log-odds."""

    predictor = probabilities_to_logit(
        probabilities
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        predictor,
        target,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    predictor = probabilities_to_logit(
        probabilities
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        predictor
    )[:, 1]


def expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = N_ECE_BINS,
) -> float:
    """Equal-width expected calibration error."""

    target = np.asarray(
        target,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=False,
    )

    ece = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id

        if not mask.any():
            continue

        observed = target[mask].mean()
        predicted = probabilities[mask].mean()

        ece += (
            mask.mean()
            * abs(
                observed
                - predicted
            )
        )

    return float(ece)


def calibration_intercept_slope(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope."""

    predictor = probabilities_to_logit(
        probabilities
    ).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        predictor,
        target,
    )

    return (
        float(
            model.intercept_[0]
        ),
        float(
            model.coef_[0][0]
        ),
    )


def calculate_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    probabilities = clip_probabilities(
        probabilities
    )

    intercept, slope = (
        calibration_intercept_slope(
            target,
            probabilities,
        )
    )

    return {
        "brier_score": float(
            brier_score_loss(
                target,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                target,
                probabilities,
                labels=[0, 1],
            )
        ),
        "pr_auc": float(
            average_precision_score(
                target,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                probabilities,
            )
        ),
        "calibration_intercept": (
            intercept
        ),
        "calibration_slope": slope,
        "ece_10_equal_width": (
            expected_calibration_error(
                target,
                probabilities,
            )
        ),
    }


def validate_oof_frame(
    frame: pd.DataFrame,
) -> None:
    required = {
        TARGET_COLUMN,
        FOLD_COLUMN,
        PATIENT_COLUMN,
        BASE_PROBABILITY_COLUMN,
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Missing required OOF columns: "
            f"{sorted(missing)}"
        )

    if len(frame) != 79_473:
        raise RuntimeError(
            "Unexpected number of OOF "
            f"encounters: {len(frame):,}"
        )

    if (
        frame[
            PATIENT_COLUMN
        ].nunique()
        != 55_952
    ):
        raise RuntimeError(
            "Unexpected number of "
            "OOF patients."
        )

    folds_per_patient = (
        frame.groupby(
            PATIENT_COLUMN
        )[FOLD_COLUMN]
        .nunique()
        .max()
    )

    if folds_per_patient != 1:
        raise RuntimeError(
            "Patient leakage detected "
            "between outer folds."
        )

    if frame[
        BASE_PROBABILITY_COLUMN
    ].isna().any():
        raise RuntimeError(
            "Missing Random Forest "
            "OOF probabilities."
        )

    if not frame[
        BASE_PROBABILITY_COLUMN
    ].between(
        0,
        1,
    ).all():
        raise RuntimeError(
            "Invalid Random Forest "
            "OOF probabilities."
        )


def crossfit_calibration(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Generate cross-fitted calibrated probabilities."""

    result = frame.copy()

    result[
        "uncalibrated_probability"
    ] = result[
        BASE_PROBABILITY_COLUMN
    ].astype(float)

    result[
        "sigmoid_probability"
    ] = np.nan

    result[
        "isotonic_probability"
    ] = np.nan

    folds = sorted(
        result[
            FOLD_COLUMN
        ].unique()
    )

    for fold in folds:
        calibration_mask = (
            result[
                FOLD_COLUMN
            ]
            != fold
        )

        validation_mask = (
            result[
                FOLD_COLUMN
            ]
            == fold
        )

        train_probability = result.loc[
            calibration_mask,
            BASE_PROBABILITY_COLUMN,
        ].to_numpy()

        train_target = result.loc[
            calibration_mask,
            TARGET_COLUMN,
        ].to_numpy(
            dtype=int
        )

        validation_probability = (
            result.loc[
                validation_mask,
                BASE_PROBABILITY_COLUMN,
            ].to_numpy()
        )

        sigmoid = (
            fit_sigmoid_calibrator(
                train_probability,
                train_target,
            )
        )

        result.loc[
            validation_mask,
            "sigmoid_probability",
        ] = apply_sigmoid_calibrator(
            sigmoid,
            validation_probability,
        )

        isotonic = IsotonicRegression(
            y_min=EPSILON,
            y_max=1.0 - EPSILON,
            out_of_bounds="clip",
        )

        isotonic.fit(
            train_probability,
            train_target,
        )

        result.loc[
            validation_mask,
            "isotonic_probability",
        ] = isotonic.predict(
            validation_probability
        )

    probability_columns = [
        "uncalibrated_probability",
        "sigmoid_probability",
        "isotonic_probability",
    ]

    for column in probability_columns:
        if result[
            column
        ].isna().any():
            raise RuntimeError(
                "Cross-fitted calibration "
                f"incomplete for {column}."
            )

        if not result[
            column
        ].between(
            0,
            1,
        ).all():
            raise RuntimeError(
                "Invalid calibrated "
                f"probabilities: {column}."
            )

    return result


def build_fold_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for method in METHODS:
        probability_column = (
            f"{method}_probability"
        )

        for fold in sorted(
            frame[
                FOLD_COLUMN
            ].unique()
        ):
            part = frame.loc[
                frame[
                    FOLD_COLUMN
                ]
                == fold
            ]

            metrics = calculate_metrics(
                part[
                    TARGET_COLUMN
                ].to_numpy(
                    dtype=int
                ),
                part[
                    probability_column
                ].to_numpy(),
            )

            rows.append(
                {
                    "method": method,
                    "outer_fold": int(
                        fold
                    ),
                    "encounters": int(
                        len(part)
                    ),
                    "patients": int(
                        part[
                            PATIENT_COLUMN
                        ].nunique()
                    ),
                    "prevalence_30d": float(
                        part[
                            TARGET_COLUMN
                        ].mean()
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def build_summary(
    frame: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    raw_fold = (
        fold_metrics.loc[
            fold_metrics[
                "method"
            ]
            == "uncalibrated"
        ]
        .set_index(
            "outer_fold"
        )
    )

    for method in METHODS:
        probability_column = (
            f"{method}_probability"
        )

        overall = calculate_metrics(
            frame[
                TARGET_COLUMN
            ].to_numpy(
                dtype=int
            ),
            frame[
                probability_column
            ].to_numpy(),
        )

        method_fold = (
            fold_metrics.loc[
                fold_metrics[
                    "method"
                ]
                == method
            ]
            .set_index(
                "outer_fold"
            )
        )

        brier_values = (
            method_fold[
                "brier_score"
            ]
        )

        if (
            method
            == "uncalibrated"
        ):
            paired_improvement = (
                pd.Series(
                    np.zeros(
                        len(
                            brier_values
                        )
                    ),
                    index=(
                        brier_values.index
                    ),
                )
            )
        else:
            paired_improvement = (
                raw_fold[
                    "brier_score"
                ]
                - method_fold[
                    "brier_score"
                ]
            )

        improvement_mean = float(
            paired_improvement.mean()
        )

        improvement_sd = float(
            paired_improvement.std(
                ddof=1
            )
        )

        improvement_se = float(
            improvement_sd
            / np.sqrt(
                len(
                    paired_improvement
                )
            )
        )

        rows.append(
            {
                "method": method,
                **overall,
                "fold_mean_brier_score": float(
                    brier_values.mean()
                ),
                "fold_sd_brier_score": float(
                    brier_values.std(
                        ddof=1
                    )
                ),
                "paired_brier_improvement_mean": (
                    improvement_mean
                ),
                "paired_brier_improvement_sd": (
                    improvement_sd
                ),
                "paired_brier_improvement_se": (
                    improvement_se
                ),
            }
        )

    summary = pd.DataFrame(rows)

    raw_log_loss = float(
        summary.loc[
            summary[
                "method"
            ]
            == "uncalibrated",
            "log_loss",
        ].iloc[0]
    )

    summary[
        "eligible_vs_uncalibrated"
    ] = False

    for method in (
        "sigmoid",
        "isotonic",
    ):
        row = summary.loc[
            summary[
                "method"
            ]
            == method
        ].iloc[0]

        eligible = (
            row[
                "paired_brier_improvement_mean"
            ]
            > 0
            and row[
                "paired_brier_improvement_mean"
            ]
            > row[
                "paired_brier_improvement_se"
            ]
            and row[
                "log_loss"
            ]
            <= raw_log_loss
        )

        summary.loc[
            summary[
                "method"
            ]
            == method,
            "eligible_vs_uncalibrated",
        ] = eligible

    return summary


def select_strategy(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> tuple[str, str]:
    sigmoid_eligible = bool(
        summary.loc[
            summary["method"]
            == "sigmoid",
            "eligible_vs_uncalibrated",
        ].iloc[0]
    )

    isotonic_eligible = bool(
        summary.loc[
            summary["method"]
            == "isotonic",
            "eligible_vs_uncalibrated",
        ].iloc[0]
    )

    if (
        not sigmoid_eligible
        and not isotonic_eligible
    ):
        return (
            "uncalibrated",
            (
                "Neither calibration strategy "
                "satisfied the frozen "
                "eligibility rule."
            ),
        )

    if (
        sigmoid_eligible
        and not isotonic_eligible
    ):
        return (
            "sigmoid",
            (
                "Only sigmoid calibration "
                "satisfied the frozen "
                "eligibility rule."
            ),
        )

    if (
        isotonic_eligible
        and not sigmoid_eligible
    ):
        return (
            "isotonic",
            (
                "Only isotonic calibration "
                "satisfied the frozen "
                "eligibility rule."
            ),
        )

    sigmoid = (
        fold_metrics.loc[
            fold_metrics[
                "method"
            ]
            == "sigmoid"
        ]
        .set_index(
            "outer_fold"
        )
    )

    isotonic = (
        fold_metrics.loc[
            fold_metrics[
                "method"
            ]
            == "isotonic"
        ]
        .set_index(
            "outer_fold"
        )
    )

    isotonic_vs_sigmoid = (
        sigmoid[
            "brier_score"
        ]
        - isotonic[
            "brier_score"
        ]
    )

    mean_difference = float(
        isotonic_vs_sigmoid.mean()
    )

    se_difference = float(
        isotonic_vs_sigmoid.std(
            ddof=1
        )
        / np.sqrt(
            len(
                isotonic_vs_sigmoid
            )
        )
    )

    if (
        mean_difference > 0
        and mean_difference
        > se_difference
    ):
        return (
            "isotonic",
            (
                "Both strategies were eligible "
                "and isotonic exceeded sigmoid "
                "by more than one standard "
                "error in paired fold-level "
                "Brier improvement."
            ),
        )

    return (
        "sigmoid",
        (
            "Both strategies were eligible; "
            "isotonic did not improve over "
            "sigmoid by more than one standard "
            "error, so sigmoid is preferred "
            "for simplicity."
        ),
    )


def write_findings(
    strategy: str,
    reason: str,
    summary: pd.DataFrame,
) -> None:
    raw = summary.loc[
        summary[
            "method"
        ]
        == "uncalibrated"
    ].iloc[0]

    sigmoid = summary.loc[
        summary[
            "method"
        ]
        == "sigmoid"
    ].iloc[0]

    isotonic = summary.loc[
        summary[
            "method"
        ]
        == "isotonic"
    ].iloc[0]

    text = f"""# Week 6 calibration audit

## Scope

This analysis evaluates probability calibration for the locked
Random Forest candidate using patient-safe cross-fitted OOF
predictions only.

The held-out test set was not accessed.

## Cross-fitting

The five existing patient-grouped outer folds were reused.

For each validation fold, calibration was fitted using predictions
from the other four folds only.

No observation was calibrated using a calibrator fitted on its own
outer fold.

## Overall results

| Strategy | Brier | Log loss | PR-AUC | ROC-AUC | Intercept | Slope | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated | {raw["brier_score"]:.6f} | {raw["log_loss"]:.6f} | {raw["pr_auc"]:.6f} | {raw["roc_auc"]:.6f} | {raw["calibration_intercept"]:.6f} | {raw["calibration_slope"]:.6f} | {raw["ece_10_equal_width"]:.6f} |
| Sigmoid | {sigmoid["brier_score"]:.6f} | {sigmoid["log_loss"]:.6f} | {sigmoid["pr_auc"]:.6f} | {sigmoid["roc_auc"]:.6f} | {sigmoid["calibration_intercept"]:.6f} | {sigmoid["calibration_slope"]:.6f} | {sigmoid["ece_10_equal_width"]:.6f} |
| Isotonic | {isotonic["brier_score"]:.6f} | {isotonic["log_loss"]:.6f} | {isotonic["pr_auc"]:.6f} | {isotonic["roc_auc"]:.6f} | {isotonic["calibration_intercept"]:.6f} | {isotonic["calibration_slope"]:.6f} | {isotonic["ece_10_equal_width"]:.6f} |

## Paired Brier evidence

Sigmoid:

- mean paired improvement: {sigmoid["paired_brier_improvement_mean"]:.8f}
- SE: {sigmoid["paired_brier_improvement_se"]:.8f}
- eligible: {bool(sigmoid["eligible_vs_uncalibrated"])}

Isotonic:

- mean paired improvement: {isotonic["paired_brier_improvement_mean"]:.8f}
- SE: {isotonic["paired_brier_improvement_se"]:.8f}
- eligible: {bool(isotonic["eligible_vs_uncalibrated"])}

## Protocol-derived recommendation

**{strategy}**

{reason}

This recommendation is generated mechanically from the frozen
Week-6 calibration protocol.

It is not yet the final calibration lock.

## Safeguards

- Candidate family was not changed.
- Candidate hyperparameters were not changed.
- No operating threshold was selected.
- No held-out test data were accessed.
- PR-AUC and ROC-AUC were not used as the primary calibration-selection
  criterion.
"""

    DECISION_PATH.write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    frame = pd.read_csv(
        OOF_PATH
    )

    validate_oof_frame(
        frame
    )

    calibrated = (
        crossfit_calibration(
            frame
        )
    )

    fold_metrics = (
        build_fold_metrics(
            calibrated
        )
    )

    summary = build_summary(
        calibrated,
        fold_metrics,
    )

    strategy, reason = (
        select_strategy(
            summary,
            fold_metrics,
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CALIBRATED_OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    calibrated.to_csv(
        CALIBRATED_OOF_PATH,
        index=False,
    )

    write_findings(
        strategy,
        reason,
        summary,
    )

    print(
        "\nW6 CALIBRATION AUDIT"
    )
    print("=" * 72)

    print(
        summary[
            [
                "method",
                "brier_score",
                "log_loss",
                "pr_auc",
                "roc_auc",
                "calibration_intercept",
                "calibration_slope",
                "ece_10_equal_width",
                "paired_brier_improvement_mean",
                "paired_brier_improvement_se",
                "eligible_vs_uncalibrated",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nProtocol-derived recommendation:"
    )
    print(
        strategy
    )

    print(
        "\nReason:"
    )
    print(
        reason
    )

    print(
        "\nSaved versionable reports:"
    )
    print(
        f"- {FOLD_METRICS_PATH}"
    )
    print(
        f"- {SUMMARY_PATH}"
    )
    print(
        f"- {DECISION_PATH}"
    )

    print(
        "\nLocal cross-fitted predictions:"
    )
    print(
        f"- {CALIBRATED_OOF_PATH}"
    )

    print(
        "\nNo threshold selected."
    )
    print(
        "Held-out test set not accessed."
    )


if __name__ == "__main__":
    main()

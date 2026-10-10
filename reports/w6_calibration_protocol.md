# Week 6 — Probability Calibration Protocol

## Purpose

This protocol defines how probability calibration will be evaluated for
the Week-6 locked candidate model before any held-out test data are
accessed.

The locked candidate model is:

- estimator: RandomForestClassifier
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 5
- class_weight: None
- random_state: 42

This protocol is frozen before calibration results are examined.

## Data source

Calibration analysis will use only the patient-safe out-of-fold
predictions generated during the nested cross-validation model-selection
stage.

Source:

`data/intermediate/w6_model_selection_oof_predictions.csv`

The held-out test set must not be accessed.

## Candidate calibration strategies

Three strategies will be compared:

1. no additional calibration;
2. sigmoid calibration (Platt scaling);
3. isotonic calibration.

No additional calibration method may be introduced after results have
been inspected unless a technical failure requires an explicitly
documented amendment.

## Cross-fitting design

Calibration must itself be evaluated out of sample.

The existing five patient-grouped outer-fold assignments will be reused.

For each fold:

1. fit the calibrator using OOF predictions from the other four folds;
2. apply the fitted calibrator to the held-out fold;
3. store the calibrated probability for that fold.

This produces cross-fitted calibrated probabilities for all training
encounters without using a patient's own calibration fold for fitting.

No calibrator may be fitted and evaluated on the same observations.

## Primary calibration criterion

Brier score is the primary calibration-selection criterion.

For each calibrated strategy, the fold-level paired improvement relative
to the uncalibrated Random Forest will be calculated as:

`Brier_uncalibrated - Brier_calibrated`

Positive values indicate improvement.

The mean paired improvement and its standard error across the five outer
folds will be reported.

A calibration strategy is eligible to replace the uncalibrated model only
if:

1. its mean Brier improvement is positive;
2. the mean Brier improvement exceeds one standard error of the paired
   fold-level improvement;
3. overall cross-fitted log loss is not worse than the uncalibrated model.

## Selection between sigmoid and isotonic

If neither calibrated strategy satisfies the eligibility rule, the model
will remain uncalibrated.

If only one satisfies the rule, that strategy will be selected.

If both satisfy the rule, sigmoid calibration will be preferred for
simplicity unless isotonic calibration demonstrates a Brier improvement
over sigmoid that itself exceeds one standard error of the paired
fold-level difference.

## Secondary descriptive metrics

The following will also be reported:

- overall Brier score;
- log loss;
- calibration intercept;
- calibration slope;
- expected calibration error;
- fold-level Brier scores;
- probability distributions.

PR-AUC and ROC-AUC may be reported as safeguards but will not determine
the calibration strategy.

## Threshold separation

Probability calibration and operating-threshold selection are separate
decisions.

No operating threshold will be selected during this calibration stage.

Threshold selection may begin only after the calibration strategy has
been locked.

## Test-set firewall

The held-out test set remains frozen.

It may not be used for:

- calibration fitting;
- calibration-method selection;
- threshold selection;
- calibration diagnostics;
- model comparison.

The held-out test set may only be accessed after:

1. candidate model family is locked;
2. hyperparameters are locked;
3. preprocessing is locked;
4. calibration strategy is locked;
5. operating-threshold strategy is locked;
6. methodological documentation is complete.

## Freeze rule

This document is the frozen Week-6 calibration-selection protocol.

Changes are permitted only when technically necessary and must be
explicitly documented with the reason for the amendment.

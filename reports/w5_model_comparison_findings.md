# Week 5 — Baseline model comparison

## Scope

This comparison was performed exclusively within the training cohort.

All models were evaluated using the same five patient-grouped
cross-validation folds.

The held-out test set was not accessed.

No hyperparameter search or operating-threshold optimisation was
performed.

## Models

Three reference models are available:

- Dummy classifier using the training prevalence
- Logistic regression
- Random forest

The random forest configuration was defined before evaluation and was
not selected using cross-validation performance.

## Out-of-fold discrimination

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Dummy prior | 0.4999 | 0.1139 |
| Logistic regression | 0.6407 | 0.2021 |
| Random forest | 0.6486 | 0.2062 |

Relative to logistic regression, the random forest produced:

- ROC-AUC difference: +0.0079
- PR-AUC difference: +0.0040

The non-linear model therefore provides a modest improvement in
out-of-fold discrimination.

## Probability accuracy

OOF Brier scores were approximately:

- Dummy prior: 0.10092
- Logistic regression: 0.09769
- Random forest: 0.09778

The random forest therefore does not improve Brier score relative to
logistic regression. The difference is very small.

## Fold stability

Random forest validation ROC-AUC values were approximately:

- Fold 1: 0.6331
- Fold 2: 0.6490
- Fold 3: 0.6432
- Fold 4: 0.6522
- Fold 5: 0.6661

The discrimination improvement is therefore not attributable to one
single validation fold.

## Fixed-threshold behaviour

At the predefined threshold of 0.50, the random forest classified no
encounters as positive in any validation fold:

- sensitivity: 0
- specificity: 1

This behaviour reflects the low predicted probabilities in an
imbalanced outcome setting and does not indicate a pipeline failure.

Threshold-dependent metrics at 0.50 are therefore not used to select
the preferred model.

Operating-threshold selection remains a separate later modelling
decision.

## Interpretation

The random forest demonstrates that non-linear relationships provide
some additional discriminatory information beyond the logistic
regression baseline.

However, the improvement is modest, while probability accuracy measured
by the Brier score is essentially unchanged and slightly worse.

The logistic regression therefore remains the principal interpretable
baseline, while the random forest is retained as a non-linear comparator.

No final model has been selected at this stage.

## Methodological safeguard

These results derive entirely from patient-safe out-of-fold predictions.

The held-out test set remains untouched and will only be accessed after
all model-development, model-selection and threshold decisions have
been locked.

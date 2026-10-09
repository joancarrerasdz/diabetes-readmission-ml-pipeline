# Week 6 model-selection protocol

## Status

This protocol is frozen before Week 6 controlled optimisation results are generated.

Any subsequent change to the selection rules must be explicitly justified and documented rather than silently adapted to observed results.

## Objective

Week 6 will select and lock one candidate model for 30-day hospital readmission prediction.

The purpose is controlled model selection, not unrestricted model exploration.

The held-out test set remains completely inaccessible during this stage.

## Development data

All model-development decisions use only the previously defined training cohort:

- 79,473 encounters
- 55,952 unique patients
- approximately 11.39% positive 30-day readmission prevalence

The held-out test set must not be used for:

- hyperparameter selection
- feature selection
- preprocessing decisions
- model-family selection
- calibration decisions
- threshold selection
- interpretation of candidate performance

## Candidate models

Only the model families already established during Week 5 are eligible for controlled optimisation:

1. Logistic Regression
2. Random Forest

The dummy-prior classifier remains a reference baseline and is not an optimisation candidate.

No additional model family may be introduced during Week 6 without an explicit protocol amendment.

## Validation design

Model selection must remain patient-safe.

The outer evaluation uses five patient-grouped, approximately stratified folds.

Requirements:

- all encounters belonging to one patient remain in the same fold;
- patient overlap between training and validation partitions is zero;
- preprocessing is fitted exclusively within the relevant training partition;
- validation data never influence fitted preprocessing parameters;
- random states are fixed and reproducible.

Hyperparameter optimisation must occur only inside the training portion of each outer fold.

Therefore, Week 6 controlled optimisation uses nested patient-grouped cross-validation:

- outer loop: unbiased model comparison and OOF prediction generation;
- inner loop: hyperparameter selection using only the outer-training data.

The held-out test set is outside both loops.

## Primary selection metric

The primary model-selection metric is PR-AUC / Average Precision.

Rationale:

The positive outcome prevalence is approximately 11.39%, so precision-recall performance is especially informative for assessing discrimination of the minority readmission class.

Inner-loop hyperparameter selection will therefore optimise mean PR-AUC.

## Secondary evaluation metrics

The following metrics will also be reported:

- ROC-AUC
- Brier score
- calibration diagnostics
- fold-to-fold variability

Threshold-dependent metrics such as sensitivity, specificity, precision, F1 and balanced accuracy may be reported diagnostically, but they must not determine hyperparameter or model-family selection while the operating threshold remains unlocked.

## Model-selection rule

After nested cross-validation:

1. Candidate models are ranked primarily by mean outer-fold PR-AUC.
2. Models whose PR-AUC performance lies within one standard error of the best-performing candidate are treated as practically competitive rather than automatically distinct.
3. Among practically competitive candidates, probability quality is compared using mean Brier score and calibration behaviour.
4. ROC-AUC is used as an additional discrimination check.
5. Fold-to-fold stability is considered when performance is otherwise comparable.
6. If models remain effectively equivalent, the simpler and more interpretable model is preferred.

This rule prevents selection based on trivial numerical differences.

## Logistic Regression search scope

The Logistic Regression search is intentionally limited.

Candidate values:

- penalty: L2
- C: 0.1, 1.0, 10.0
- class_weight: None, balanced
- random_state: 42 where applicable

The existing fold-safe preprocessing pipeline must remain inside the fitted estimator pipeline.

## Random Forest search scope

The Random Forest search is intentionally limited.

Candidate values:

- n_estimators: 300
- max_depth: None, 12
- min_samples_leaf: 1, 5
- class_weight: None, balanced_subsample
- random_state: 42
- n_jobs: -1

The purpose is controlled optimisation rather than exhaustive hyperparameter search.

## Calibration

Calibration is evaluated using out-of-fold predictions only.

Brier score and calibration diagnostics must be reported.

No calibration method may be fitted using the held-out test set.

If additional probability calibration is justified, it must itself be learned exclusively inside the training/CV framework and documented as an additional modelling step.

## Operating threshold

No operating threshold is selected during hyperparameter optimisation.

Model-family and hyperparameter selection must be completed first.

Threshold selection, if performed later, will use only out-of-fold predictions from the locked candidate model and will remain completely independent of the held-out test set.

## Test-set firewall

The held-out test set remains frozen until:

1. the candidate model family is selected;
2. hyperparameters are locked;
3. preprocessing is locked;
4. any calibration strategy is locked;
5. the operating-threshold strategy is locked;
6. the model card and methodological decisions are updated.

Only then may a final one-time held-out evaluation be performed.

## Reproducibility requirements

Week 6 must produce versionable evidence sufficient to reconstruct the selection decision.

Expected outputs include:

- outer-fold model metrics
- selected inner-loop hyperparameters
- model-comparison summary
- candidate-selection rationale
- out-of-fold predictions stored locally
- automated quality-gate tests

All versioned reports must be derived reproducibly from code.

## Interpretation principle

Week 6 evaluates predictive performance, not causal effects.

A numerically higher metric does not by itself establish clinical superiority.

Model selection must consider discrimination, probability quality, stability, interpretability and methodological simplicity together.

## Freeze rule

This protocol is the Week 6 model-selection baseline.

Changes are permitted only when technically necessary and must be explicitly documented with the reason for the amendment.

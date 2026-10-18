# Week 7 — Frozen Final Held-out Evaluation Protocol

## Purpose

Week 7 performs the final one-time evaluation of the fully locked
30-day diabetes readmission prediction pipeline on the independent
held-out test set.

This evaluation estimates generalisation performance only.

The held-out test set must not be used to modify, select, tune,
recalibrate or otherwise adapt any component of the predictive pipeline.

## Pre-test locked state

All development decisions were completed and frozen before Week 7.

The locked candidate model is:

- model family: Random Forest;
- estimator: `RandomForestClassifier`;
- `n_estimators = 300`;
- `max_depth = 12`;
- `min_samples_leaf = 5`;
- `class_weight = None`;
- `random_state = 42`.

The following components are also locked:

- feature schema;
- preprocessing specification;
- candidate model family;
- candidate hyperparameters;
- probability calibration strategy: sigmoid;
- operating threshold: 0.110.

The operating threshold is a predictive operating point.

It is not a clinically validated treatment or intervention threshold.

No model, preprocessing, calibration, feature or threshold decision may
be changed during Week 7 because of held-out performance.

## Evaluation population

The final evaluation uses only the independent patient-level held-out
test set reserved before model development.

The development and held-out cohorts must remain patient-disjoint.

Before predictive results are inspected, the held-out cohort must be
audited to confirm:

- expected cohort structure;
- unique patient identifiers;
- no overlap with development patients;
- valid binary outcome coding;
- required model features are available;
- no unexpected schema violation.

No exclusion or inclusion criterion may be changed on the basis of model
predictions or held-out outcomes.

## Evaluation pipeline

The final predictive pipeline must reproduce the frozen Week-6
specification.

All fitted components must use development data only.

The evaluation sequence is:

1. load the frozen development and held-out cohorts;
2. verify the patient-level firewall;
3. reconstruct the locked feature schema and preprocessing;
4. fit the locked Random Forest using development data only;
5. apply the locked sigmoid probability-calibration strategy using
   development data only;
6. generate held-out predicted probabilities exactly once;
7. apply the locked operating threshold of 0.110;
8. calculate the predefined final evaluation metrics;
9. save the required versionable evidence.

The held-out set must not be used for fitting preprocessing,
hyperparameters, model parameters, calibration parameters or threshold
selection.

## Primary final metrics

The final held-out evaluation must report threshold-independent
performance using:

- PR-AUC;
- ROC-AUC;
- Brier score;
- log loss.

At the locked operating threshold of 0.110, the evaluation must report:

- sensitivity;
- specificity;
- precision;
- F1 score;
- balanced accuracy;
- predicted-positive rate.

The corresponding confusion-matrix counts must also be reported:

- true positives;
- false positives;
- true negatives;
- false negatives.

Probability diagnostics must include:

- held-out outcome prevalence;
- mean predicted probability;
- calibration intercept;
- calibration slope;
- expected calibration error;
- Brier score.

All metrics are descriptive estimates of generalisation performance.

## Interpretation

Week 7 evaluates predictive generalisation.

It does not establish causal effects or clinical utility.

The operating threshold of 0.110 was selected during development using
patient-safe calibrated out-of-fold predictions.

Its final held-out sensitivity, specificity, precision or other
threshold-dependent metrics must not be used to retrospectively select a
different threshold.

Likewise, held-out calibration results must not be used to choose a new
calibration method.

Any discrepancy between development and held-out performance is itself a
result and must be reported rather than corrected through additional
model development.

## Comparison with development evidence

Final held-out performance will be compared descriptively with the
previously generated patient-safe development evidence.

Relevant development references include:

- nested-CV model-selection performance;
- cross-fitted calibration performance;
- locked operating-threshold performance.

The purpose of this comparison is to assess apparent generalisation
rather than to trigger additional tuning.

Differences between development and held-out performance must be
documented without modifying the frozen pipeline.

## One-time evaluation rule

The independent held-out test set is intended for one final predictive
evaluation.

Once held-out predictive results have been generated and inspected, the
test set is considered consumed for this development iteration.

It must not subsequently be reused for:

- feature engineering;
- model-family selection;
- hyperparameter optimisation;
- probability-calibration selection;
- operating-threshold selection;
- retrospective tuning.

## Technical-failure rule

If execution fails before held-out predictive results are generated,
a strictly technical correction may be made only when necessary to
execute the already frozen methodology.

Such a correction must:

1. not change the statistical or modelling decision;
2. not depend on held-out performance;
3. be explicitly documented;
4. preserve the original locked specification.

If held-out predictive results have already been observed, no
performance-driven correction is permitted.

## Required safeguards

Before final evaluation:

- all automated tests must pass;
- `git diff --check` must pass;
- the working tree must be clean before new W7 implementation work;
- W6 selection, calibration and threshold locks must be internally
  consistent;
- the held-out patient firewall must pass.

## Required versionable outputs

Week 7 will produce versionable evidence including:

- final held-out cohort audit;
- final held-out metrics;
- final confusion matrix;
- final calibration diagnostics;
- comparison with development OOF performance;
- final evaluation findings;
- updated model card.

Raw or row-level prediction files containing dataset-level outputs may
remain local when appropriate and must not be committed merely for
convenience.

## Post-evaluation rule

After the one-time held-out evaluation, the development pipeline is
closed.

Any future model improvement must be treated as a new development
iteration requiring new independent validation data.

The existing held-out test set must not be reused as a tuning set.

## Freeze rule

This document is the frozen Week 7 final-evaluation protocol.

Changes after this point are permitted only for technically necessary
reasons and must be explicitly documented.

No amendment may be justified by observed held-out model performance.

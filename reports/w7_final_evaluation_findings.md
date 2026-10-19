# Week 7 — Final held-out evaluation

## Frozen specification

- candidate model: Random Forest
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 5
- class_weight: None
- random_state: 42
- probability calibration: sigmoid
- operating threshold: 0.110

## Independent held-out cohort

- encounters: 19,870
- positive 30-day readmissions: 2,263
- prevalence: 0.113890

## Final discrimination and probability performance

- PR-AUC: 0.192383
- ROC-AUC: 0.645006
- Brier score: 0.098020
- log loss: 0.340966

## Calibration

- mean predicted probability: 0.114198
- calibration intercept: -0.098333
- calibration slope: 0.951253
- ECE (10 equal-width bins): 0.006471

## Locked operating threshold

Threshold: **0.110**

- sensitivity: 0.592576
- specificity: 0.611461
- precision: 0.163896
- F1: 0.256774
- balanced accuracy: 0.602019
- predicted-positive rate: 0.411777

## Confusion matrix

- TP: 1341
- FP: 6841
- TN: 10766
- FN: 922

## Methodological status

This is the one-time independent held-out evaluation of the frozen
development pipeline.

The held-out data were not used to change the feature schema,
preprocessing, model family, hyperparameters, calibration strategy or
operating threshold.

The held-out test set is now considered consumed for this development
iteration.

Observed performance must not be used for retrospective tuning.

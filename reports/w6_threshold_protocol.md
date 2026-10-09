# Week 6 — Operating-Threshold Selection Protocol

## Purpose

This document defines the operating-threshold selection procedure for the
locked Week-6 model before threshold-dependent development results are
examined.

The purpose is to prevent retrospective threshold selection.

## Locked upstream state

The following development decisions are already frozen:

1. candidate model family: Random Forest;
2. model hyperparameters;
3. preprocessing specification;
4. probability calibration strategy: sigmoid.

These decisions must not be changed during threshold selection.

## Development data

Threshold selection will use only patient-safe cross-fitted out-of-fold
predictions from the locked Random Forest with locked sigmoid calibration.

The held-out test set must not be accessed.

Each patient belongs to a single outer validation fold.

## Probability source

The threshold-selection input is:

`sigmoid_probability`

from the Week-6 cross-fitted calibration output.

No uncalibrated, isotonic or held-out probabilities may be used to select
the operating threshold.

## Candidate thresholds

Candidate operating thresholds will be evaluated over the interval:

0.01 to 0.50 inclusive

using increments of:

0.005

This grid is frozen before inspecting threshold-dependent performance.

## Metrics

For every candidate threshold and every patient-grouped outer fold, the
following metrics will be calculated:

- sensitivity;
- specificity;
- precision;
- F1 score;
- balanced accuracy;
- predicted-positive rate.

Confusion-matrix counts will also be retained:

- true positives;
- false positives;
- true negatives;
- false negatives.

For each threshold, fold-level mean and standard deviation will be
reported.

## Primary selection criterion

The primary criterion is:

**mean balanced accuracy across the five patient-grouped outer folds.**

Balanced accuracy is used because the outcome is substantially imbalanced
and no externally validated clinical cost ratio has been specified.

The objective is therefore to give equal weight to sensitivity and
specificity during development.

## Deterministic tie-breaking

If more than one threshold has the same maximum mean balanced accuracy at
the stored numerical precision, the following tie-breaking rules will be
applied sequentially:

1. higher mean sensitivity;
2. higher mean F1 score;
3. lower operating threshold.

No retrospective modification of these rules is allowed after the
threshold sweep has been examined.

## Stability assessment

The selected threshold must be accompanied by:

- fold-level balanced accuracy;
- fold-level sensitivity;
- fold-level specificity;
- fold-level F1;
- mean and standard deviation across folds;
- predicted-positive rate across folds.

Threshold-performance curves may be generated for interpretation, but
they must not override the frozen selection rule.

## Calibration separation

The calibration strategy is already locked as sigmoid.

Threshold selection must not trigger:

- recalibration;
- model-family changes;
- hyperparameter changes;
- preprocessing changes;
- feature changes.

## Interpretation

The selected threshold is an operating point for this predictive
development pipeline.

It must not be interpreted as a clinically validated treatment or
intervention threshold.

Clinical deployment would require independent validation and an explicit
clinical utility or cost framework.

## Test-set firewall

The held-out test set remains frozen.

It must not be used for:

- threshold selection;
- threshold tuning;
- threshold comparison;
- calibration;
- feature engineering;
- model selection;
- hyperparameter tuning.

The test set may only be accessed after the operating threshold has been
locked and all remaining methodological decisions have been documented.

## Freeze rule

This document is the frozen Week-6 operating-threshold selection protocol.

Changes are permitted only when technically necessary and must be
explicitly documented with the reason for the amendment before threshold
results are inspected.

# Week 6 — Calibration Strategy Lock

## Locked decision

The probability-calibration strategy for the locked Week-6 Random Forest
candidate is:

**Sigmoid calibration (Platt scaling).**

This decision is now frozen for subsequent model-development stages.

## Evidence

The decision was derived from the previously frozen calibration protocol
and patient-safe cross-fitted OOF predictions.

Overall Brier scores:

- uncalibrated: 0.097992
- sigmoid: 0.097329
- isotonic: 0.097422

Paired fold-level Brier improvement versus the uncalibrated model:

- sigmoid:
  - mean improvement: 0.00066302
  - standard error: 0.00008955
  - eligible: yes

- isotonic:
  - mean improvement: 0.00056951
  - standard error: 0.00011513
  - eligible: yes

Both calibration approaches satisfied the eligibility rule.

Under the frozen protocol, sigmoid is preferred when both strategies are
eligible unless isotonic improves over sigmoid by more than one standard
error of their paired fold-level Brier difference.

That condition was not met.

Therefore sigmoid calibration is selected.

## Calibration diagnostics

For the cross-fitted sigmoid probabilities:

- Brier score: 0.097329
- log loss: 0.339160
- calibration intercept: -0.002429
- calibration slope: 0.998789
- ECE: 0.003203

These diagnostics are consistent with substantially improved probability
calibration relative to the uncalibrated Random Forest.

## Methodological safeguard

This lock does not select an operating threshold.

Threshold selection will use patient-safe cross-fitted probabilities from
the locked model/calibration strategy and remains a separate modelling
decision.

The held-out test set has not been accessed.

The held-out test set remains frozen until all remaining development
decisions have been locked.

## Freeze rule

The following components are now fixed:

1. candidate model family: Random Forest;
2. candidate model hyperparameters;
3. model preprocessing;
4. calibration strategy: sigmoid.

Changing the calibration strategy after this lock would require a
documented technical justification and protocol amendment.

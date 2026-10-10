# Week 6 calibration audit

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
| Uncalibrated | 0.097992 | 0.341641 | 0.205786 | 0.647181 | 1.338518 | 1.669590 | 0.014666 |
| Sigmoid | 0.097329 | 0.339160 | 0.205706 | 0.647076 | -0.002429 | 0.998789 | 0.003203 |
| Isotonic | 0.097422 | 0.339878 | 0.201660 | 0.645247 | -0.094671 | 0.951813 | 0.001301 |

## Paired Brier evidence

Sigmoid:

- mean paired improvement: 0.00066302
- SE: 0.00008955
- eligible: True

Isotonic:

- mean paired improvement: 0.00056951
- SE: 0.00011513
- eligible: True

## Protocol-derived recommendation

**sigmoid**

Both strategies were eligible; isotonic did not improve over sigmoid by more than one standard error, so sigmoid is preferred for simplicity.

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

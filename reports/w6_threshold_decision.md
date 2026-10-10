# Week 6 — Operating-threshold lock

## Locked operating point

The operating threshold is locked at:

**0.110**

The probability source is the previously locked sigmoid-calibrated
Random Forest output.

## Selection basis

The threshold was selected mechanically from the frozen Week-6
threshold-selection protocol using patient-safe cross-fitted
out-of-fold predictions only.

The predefined threshold grid contained 99 values:

- minimum: 0.010
- maximum: 0.500
- increment: 0.005

The deterministic selection hierarchy was:

1. highest mean balanced accuracy across the five outer folds;
2. higher mean sensitivity if tied;
3. higher mean F1 if still tied;
4. lower threshold if still tied.

No selection criterion was modified after inspecting the results.

## Performance at the locked threshold

Across the five outer validation folds:

- mean balanced accuracy: 0.602984
- SD balanced accuracy: 0.012366
- mean sensitivity: 0.591758
- mean specificity: 0.614211
- mean precision: 0.164594
- mean F1: 0.257535
- mean predicted-positive rate: 0.409246

Fold-level balanced accuracy ranged approximately from 0.584 to 0.619,
supporting acceptable stability of the selected operating point across
the patient-grouped outer folds.

## Interpretation

The threshold of 0.110 is an operating point for this predictive
development pipeline.

It is not a clinically validated intervention or treatment threshold.

The relatively high predicted-positive rate compared with the observed
30-day readmission prevalence reflects the frozen balanced-accuracy
selection objective and the sensitivity/specificity trade-off implied
by that objective.

Clinical deployment would require independent validation and an
explicit clinical utility or cost framework.

## Freeze rule

The following components are now fixed:

1. candidate model family: Random Forest;
2. candidate hyperparameters;
3. preprocessing and feature schema;
4. calibration strategy: sigmoid;
5. operating threshold: 0.110.

Changing the operating threshold after this lock would require a
documented technical justification and protocol amendment.

The held-out test set has not been accessed.

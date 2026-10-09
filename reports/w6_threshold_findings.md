# Week 6 — Operating-threshold sweep findings

## Frozen input

- candidate model: Random Forest
- calibration strategy: sigmoid
- probability source: `sigmoid_probability`
- held-out test set: not accessed

## Frozen threshold grid

- minimum: 0.010
- maximum: 0.500
- increment: 0.005
- candidate thresholds: 99

## Protocol-derived recommendation

**Operating threshold: 0.110**

The threshold was selected mechanically by:

1. highest mean balanced accuracy across the five outer folds;
2. higher mean sensitivity if tied;
3. higher mean F1 if still tied;
4. lower threshold if still tied.

Aggregate metrics at the selected threshold:

- mean balanced accuracy: 0.602984
- SD balanced accuracy: 0.012366
- mean sensitivity: 0.591758
- mean specificity: 0.614211
- mean precision: 0.164594
- mean F1: 0.257535
- mean predicted-positive rate: 0.409246

## Fold-level stability

| outer fold | balanced accuracy | sensitivity | specificity | F1 | predicted-positive rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.584175 | 0.544751 | 0.623598 | 0.243517 | 0.395571 |
| 2 | 0.605125 | 0.595580 | 0.614669 | 0.259290 | 0.409274 |
| 3 | 0.602231 | 0.592490 | 0.611971 | 0.257037 | 0.411324 |
| 4 | 0.604592 | 0.595580 | 0.613604 | 0.258824 | 0.410218 |
| 5 | 0.618800 | 0.630387 | 0.607214 | 0.269009 | 0.419844 |

## Safeguards

- No model-family change was made.
- No hyperparameter change was made.
- No preprocessing or feature change was made.
- No recalibration was performed.
- The held-out test set was not accessed.

This is the protocol-derived threshold recommendation. It is not the final threshold lock until explicitly versioned in a separate lock artifact.

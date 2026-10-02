# Baseline modelling findings

## Scope

Baseline modelling was performed exclusively on the frozen training split.

- Encounters: 79,473
- Unique patients: 55,952
- 30-day readmission prevalence: approximately 11.39%
- Held-out test set: not accessed

The objective of this stage is model development and internal validation,
not final generalization assessment.

## Cross-validation design

Five-fold `StratifiedGroupKFold` cross-validation was used.

`patient_nbr` is the grouping variable, ensuring that encounters from the
same patient cannot appear in both the training and validation portions of
the same fold.

Audit results:

- 5 validation folds
- zero patient overlap in every fold
- every encounter receives exactly one out-of-fold prediction
- every patient is assigned to exactly one validation fold
- class prevalence remains approximately stable across folds

All preprocessing is fitted inside the modelling pipeline within each
training fold.

## Baseline models

Two predefined baselines were evaluated:

1. `DummyClassifier(strategy="prior")`
2. Logistic regression

No threshold optimization was performed.

Threshold-dependent metrics use the predefined threshold of 0.50.

## Out-of-fold results

### Dummy prior

Approximate out-of-fold performance:

- ROC-AUC: 0.500
- PR-AUC: 0.114
- Brier score: 0.101
- sensitivity at threshold 0.50: 0
- specificity at threshold 0.50: 1
- balanced accuracy: 0.500

This provides the expected non-informative reference model.

### Logistic regression

Approximate out-of-fold performance:

- ROC-AUC: 0.641
- PR-AUC: 0.202
- Brier score: 0.098
- sensitivity at threshold 0.50: approximately 0.017
- specificity at threshold 0.50: approximately 0.998
- balanced accuracy: approximately 0.508

Fold-level ROC-AUC values ranged approximately from 0.624 to 0.650,
indicating reasonably stable discrimination across the five patient-grouped
validation folds.

## Interpretation

The logistic regression demonstrates predictive signal beyond the
prevalence-only dummy baseline.

The improvement in ROC-AUC and PR-AUC indicates that routinely collected
hospital variables contain information associated with 30-day readmission
risk.

The Brier score also improves relative to the dummy prior model, although
calibration requires dedicated evaluation later in model development.

The predefined 0.50 threshold produces very low sensitivity because the
outcome is uncommon and predicted probabilities are generally well below
0.50.

Therefore, threshold-dependent metrics at 0.50 must not be interpreted as
the final operating performance of the model.

Threshold selection will be performed later using development data only.
The held-out test set will remain untouched until the complete modelling
strategy and operating threshold have been locked.

## Methodological safeguards

- Patient-level grouped validation
- Zero patient overlap between fold training and validation sets
- Fold-safe preprocessing
- Out-of-fold prediction for every training encounter
- No test-set access
- No test-driven model selection
- No threshold optimization at this stage
- No causal interpretation of predictive associations

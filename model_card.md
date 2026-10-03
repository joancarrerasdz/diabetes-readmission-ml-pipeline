# Model Card — Diabetes 30-Day Readmission

## Project

`diabetes-readmission-ml-pipeline`

This project develops and evaluates machine-learning models for predicting
30-day hospital readmission among encounters in the UCI Diabetes
130-US Hospitals dataset.

The project is designed as a reproducible clinical machine-learning
workflow with explicit safeguards against patient-level leakage.

## Prediction task

The prediction target is binary 30-day readmission.

Positive class:

- original `readmitted == "<30"`

Negative class:

- original `readmitted == ">30"`
- original `readmitted == "NO"`

The prediction point is hospital discharge.

Only information available by discharge is eligible for use as a
candidate predictor.

The original `readmitted` variable is never used as a predictor.

## Unit of analysis

The modelling unit is the hospital encounter.

Because individual patients may have multiple encounters,
`patient_nbr` is retained only for patient-level grouping and is never
used as a predictive feature.

## Primary analytic cohort

The source dataset contains:

- 101,766 encounters
- 71,518 unique patients

Encounters with discharge dispositions representing death or hospice
care are excluded because subsequent hospital readmission is not
meaningfully comparable.

The resulting primary analytic cohort contains:

- 99,343 encounters
- 69,990 unique patients
- 11,314 positive 30-day readmissions
- 11.39% readmission prevalence

## Development split

A patient-level grouped split was created before exploratory analysis
and model development.

Training split:

- 79,473 encounters
- 55,952 unique patients
- 9,051 positive outcomes
- approximately 11.39% prevalence

Held-out test split:

- 19,870 encounters
- 14,038 unique patients
- 2,263 positive outcomes
- approximately 11.39% prevalence

Patient overlap between training and held-out test sets is zero.

The held-out test set has not been used for:

- exploratory data analysis
- data-quality decisions
- feature engineering decisions
- preprocessing decisions
- model fitting
- model comparison
- hyperparameter tuning
- threshold selection
- model interpretation

It remains reserved for final locked-model evaluation.

## Data-quality and preprocessing decisions

All data-quality decisions were derived from the training split only.

The deterministic cleaning procedure:

- removes `weight` because approximately 97% of training values are missing;
- removes constant medication variables:
  - `citoglipton`
  - `examide`
  - `glimepiride-pioglitazone`;
- preserves semantic laboratory non-measurement explicitly as
  `Not_measured`;
- preserves missing categorical information explicitly instead of
  silently applying mode imputation;
- excludes identifiers, cohort-definition variables and target variables
  from the model feature set.

After deterministic cleaning:

- encounters remain unchanged at 79,473;
- columns decrease from 51 to 47;
- no unresolved null cells remain under the documented transformations;
- 42 candidate model features remain.

## Feature engineering

Clinical feature engineering includes:

- age converted to a numeric midpoint representation;
- ICD-9 diagnosis codes mapped into broader clinical diagnosis groups;
- preservation of selected utilization and treatment variables;
- explicit handling of categorical clinical information.

The three diagnosis variables are transformed into:

- `diag_1_group`
- `diag_2_group`
- `diag_3_group`

Each derived diagnosis feature maps every training encounter into one of
the predefined clinical groups.

High-cardinality `medical_specialty` is handled using a fold-fitted rare
category strategy with:

`min_frequency = 0.005`

## Fold-safe preprocessing

Model preprocessing is implemented inside a scikit-learn
`Pipeline` / `ColumnTransformer`.

Numeric features use:

- `StandardScaler`

General categorical features use:

- `OneHotEncoder(handle_unknown="ignore")`

`medical_specialty` uses:

- `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005)`

All fitted preprocessing state is learned independently inside each
cross-validation training fold.

No validation-fold information is used to fit preprocessing operations.

## Cross-validation design

Model development uses five patient-grouped cross-validation folds.

Each patient belongs to exactly one validation fold.

Across all folds:

- total OOF encounters: 79,473
- unique OOF patients: 55,952
- maximum number of validation folds per patient: 1
- patient overlap between fold-training and fold-validation sets: 0

Out-of-fold predictions therefore cover every training encounter exactly
once without patient leakage.

## Reference models

Three reference models have been evaluated.

### Dummy prior classifier

The dummy model predicts the training prevalence.

Approximate OOF performance:

- ROC-AUC: 0.4999
- PR-AUC: 0.1139
- Brier score: 0.10092

### Logistic regression

Logistic regression is the principal interpretable baseline.

Approximate OOF performance:

- ROC-AUC: 0.6407
- PR-AUC: 0.2021
- Brier score: 0.09769

Five-fold ROC-AUC is approximately 0.641 with low between-fold variation.

### Random forest

A predefined Random Forest was evaluated as a non-linear comparator.

Its configuration was fixed before evaluation:

- 200 trees
- `max_depth = 16`
- `min_samples_leaf = 10`
- `max_features = "sqrt"`
- `random_state = 42`

No hyperparameter search was performed.

Approximate OOF performance:

- ROC-AUC: 0.6486
- PR-AUC: 0.2062
- Brier score: 0.09778

Relative to logistic regression:

- ROC-AUC difference: approximately +0.0079
- PR-AUC difference: approximately +0.0040
- Brier score: essentially unchanged and marginally worse

The Random Forest therefore provides a modest improvement in
discrimination but not in probability accuracy.

## Calibration

Out-of-fold logistic-regression probabilities have been examined using
calibration bins.

Observed and predicted risks are reasonably aligned over much of the
probability range, although deviations remain, particularly at higher
predicted risks.

Calibration assessment remains descriptive at this stage.

No post-hoc calibration model has been fitted.

## Classification threshold

No operating threshold has been selected or locked.

The default threshold of 0.50 is used only for diagnostic reporting.

For the Random Forest, a threshold of 0.50 produces:

- sensitivity: 0
- specificity: 1

This reflects the low predicted-probability range under an imbalanced
outcome and is not interpreted as evidence of pipeline failure.

Threshold-dependent metrics are therefore not used for final model
selection at this stage.

Threshold selection will remain separate from model-development and
model-comparison decisions.

## Current model-selection status

No final production model has been selected.

Current evidence indicates:

- logistic regression provides a strong, interpretable baseline;
- random forest provides a modest improvement in discrimination;
- Brier-score performance is essentially equivalent between the two.

The logistic regression therefore remains the principal interpretable
reference model, while the Random Forest is retained as a non-linear
comparator.

Any later model-selection decision must be made before accessing the
held-out test set.

## Evaluation metrics

Model development currently reports:

- ROC-AUC
- PR-AUC
- Brier score
- sensitivity
- specificity
- precision
- F1 score
- balanced accuracy

ROC-AUC, PR-AUC and Brier score are prioritised during pre-threshold
model comparison.

## Interpretation limits

This project evaluates prediction, not causation.

Observed predictor-outcome associations must not be interpreted as
causal effects.

Performance differences across demographic or clinical subgroups are
descriptive model-performance audits and do not by themselves establish
clinical fairness or causal disparity.

## Reproducibility and quality gates

The repository contains automated tests for:

- cohort construction
- patient-level splitting
- deterministic preprocessing
- clinical feature engineering
- fold-safe model preprocessing
- baseline modelling
- baseline diagnostics
- linear versus non-linear model comparison

At the end of Week 5:

- 29 automated tests pass;
- patient-level leakage checks pass;
- model-development outputs are reproducible;
- the held-out test set remains untouched.

## Current development stage

Weeks 1–5 have established:

1. reproducible data acquisition;
2. clinical outcome and cohort definition;
3. patient-safe development splitting;
4. training-only EDA and deterministic cleaning;
5. clinical feature engineering;
6. fold-safe model preprocessing;
7. baseline model development;
8. OOF diagnostics;
9. linear versus non-linear baseline comparison.

The next modelling stage should begin only after this Week-5 baseline
state is merged and treated as the new reproducible project baseline.

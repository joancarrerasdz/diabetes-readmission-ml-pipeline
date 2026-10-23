# Diabetes 30-Day Readmission — Reproducible Machine-Learning Pipeline

A reproducible machine-learning project for predicting **30-day hospital
readmission among patients with diabetes**, with particular emphasis on
patient-level leakage prevention, probability calibration, transparent model
selection and protected final evaluation.

The project follows a staged development workflow in which model selection,
calibration and operating-threshold decisions were completed using development
data before a single final evaluation on an untouched held-out test set.

---

## Project status

**Development cycle completed.**

The final Week-7 held-out evaluation has been performed and the test set is now
considered consumed for this development iteration.

The frozen final specification is:

| Component | Locked value |
|---|---|
| Model family | Random Forest |
| Trees | 300 |
| Maximum depth | 12 |
| Minimum samples per leaf | 5 |
| Class weighting | None |
| Random state | 42 |
| Model features | 42 |
| Probability calibration | Sigmoid |
| Operating threshold | 0.110 |
| Final evaluation | One-time protected held-out assessment |

No post-hoc model, calibration or threshold modification was performed after
observing held-out performance.

---

## Clinical prediction problem

Hospital readmission is an important outcome in healthcare because it may
reflect disease complexity, comorbidity burden, treatment requirements and
care-transition challenges.

This project addresses the following predictive question:

> Using information available from a hospital encounter, can a reproducible
> machine-learning pipeline estimate the probability of readmission within
> 30 days for patients with diabetes?

The binary target used throughout the modelling pipeline is:

```text
readmitted_30d
```

where:

```text
1 = readmission within 30 days
0 = no readmission within 30 days
```

This project evaluates **prediction**, not causal effects or clinical
interventions.

---

## Dataset and cohort

The project uses the Diabetes 130-US hospitals dataset and constructs a
patient-aware modelling cohort.

A central methodological requirement is that encounters from the same patient
must not appear simultaneously in training and validation/test partitions.

The final development structure contained:

| Cohort | Encounters | Unique patients |
|---|---:|---:|
| Development / training | 79,473 | 55,952 |
| Final held-out test | 19,870 | 14,038 |

Patient overlap between the development and held-out cohorts was verified to be:

```text
0
```

The held-out cohort remained inaccessible during model development,
hyperparameter selection, calibration selection and operating-threshold
selection.

---

## Why patient-level splitting matters

The dataset contains repeated encounters for some patients.

A conventional encounter-level random split could therefore allow different
encounters belonging to the same individual to appear in both development and
validation data.

That would create a clinically important form of information leakage.

To prevent this, the project uses the patient identifier:

```text
patient_nbr
```

as the grouping variable throughout the development workflow.

All major validation procedures therefore preserve patient independence.

---

## Project workflow

The project was developed incrementally.

### Week 1 — Data acquisition and cohort construction

The initial stage established:

- reproducible dataset loading;
- target construction;
- cohort definition;
- deterministic data processing;
- project structure and version control.

### Week 2 — Patient-safe partitioning

The dataset was separated into development and held-out cohorts while ensuring:

- zero patient overlap;
- deterministic splitting;
- preservation of the final held-out cohort for later evaluation.

### Week 3 — Training-only clinical EDA and cleaning

Exploratory analysis and cleaning decisions were performed using development
data only.

This prevented information from the held-out test set influencing feature or
cleaning decisions.

### Week 4 — Clinical feature engineering

A deterministic modelling schema was created from clinically interpretable
encounter variables.

The final modelling schema contains:

```text
42 features
```

including numerical, categorical and medical-specialty information.

### Week 5 — Baseline modelling

Two initial model families were evaluated:

- Logistic Regression
- Random Forest

Evaluation used patient-grouped out-of-fold predictions.

This stage established baseline discrimination, probability quality and
between-fold stability.

### Week 6 — Nested model selection

Candidate model configurations were evaluated using patient-grouped nested
cross-validation.

Model selection used development data only.

The primary model-selection metric was:

```text
PR-AUC
```

with additional consideration of:

- ROC-AUC;
- Brier score;
- between-fold stability;
- probability behaviour;
- interpretability;
- methodological simplicity.

The selected model family was:

```text
Random Forest
```

with:

```text
n_estimators = 300
max_depth = 12
min_samples_leaf = 5
class_weight = None
random_state = 42
```

The same configuration was selected across all five outer folds.

Approximate nested-CV performance of the selected Random Forest was:

| Metric | Development estimate |
|---|---:|
| PR-AUC | 0.2070 |
| ROC-AUC | 0.6473 |
| Brier score | 0.0980 |

These values represent development performance and are not final held-out
results.

### Week 6 — Probability calibration

Calibration was evaluated using cross-fitted development predictions.

The compared strategies were:

- uncalibrated probabilities;
- sigmoid calibration;
- isotonic calibration.

Both calibrated approaches satisfied the pre-specified eligibility criterion.

Because isotonic calibration did not improve sufficiently over sigmoid
calibration under the frozen decision rule, the simpler strategy was selected.

The locked calibration strategy was therefore:

```text
sigmoid
```

This decision was completed before the held-out test set was accessed.

### Week 6 — Operating-threshold selection

Operating-threshold selection was performed using patient-safe calibrated
out-of-fold predictions.

The frozen primary criterion was:

```text
mean balanced accuracy
```

with deterministic tie-breaking based on:

1. higher mean sensitivity;
2. higher mean F1 score;
3. lower threshold.

The selected operating threshold was:

```text
0.110
```

Development estimates at this threshold were approximately:

| Metric | Development estimate |
|---|---:|
| Balanced accuracy | 0.6030 |
| Sensitivity | 0.5918 |
| Specificity | 0.6142 |
| Precision | 0.1646 |
| F1 | 0.2575 |
| Predicted-positive rate | 0.4092 |

This threshold is an operating point for the predictive-development pipeline.

It is **not** a clinically validated intervention threshold.

### Week 7 — Protected final evaluation

Only after the model family, hyperparameters, preprocessing, calibration
strategy and operating threshold were locked was the held-out test set
accessed.

A pre-flight gate verified:

- model configuration consistency;
- calibration lock consistency;
- threshold lock consistency;
- zero patient overlap;
- schema validity;
- target validity;
- feature-engineering consistency;
- absence of previous held-out access.

The final model was then evaluated once on the held-out cohort.

---

## Final held-out performance

The final held-out cohort contained:

```text
19,870 encounters
2,263 positive 30-day readmissions
```

Observed 30-day readmission prevalence:

```text
0.11389
```

Mean predicted probability:

```text
0.11420
```

### Discrimination and probability quality

| Metric | Final held-out result |
|---|---:|
| PR-AUC | 0.1924 |
| ROC-AUC | 0.6450 |
| Brier score | 0.0980 |
| Log loss | 0.3410 |

### Calibration

| Metric | Final held-out result |
|---|---:|
| Calibration intercept | -0.0983 |
| Calibration slope | 0.9513 |
| ECE, 10 equal-width bins | 0.00647 |

The mean predicted probability closely matched the observed prevalence at the
cohort level.

The calibration slope was close to 1, although calibration performance should
still be interpreted in the context of this single held-out cohort.

### Performance at the locked threshold

At the pre-specified threshold of:

```text
0.110
```

the final held-out results were:

| Metric | Final result |
|---|---:|
| Sensitivity | 0.5926 |
| Specificity | 0.6115 |
| Precision | 0.1639 |
| F1 score | 0.2568 |
| Balanced accuracy | 0.6020 |
| Predicted-positive rate | 0.4118 |

### Final confusion matrix

|  | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 10,766 | 6,841 |
| Actual positive | 922 | 1,341 |

Therefore:

```text
TN = 10,766
FP = 6,841
FN = 922
TP = 1,341
```

---

## Development versus held-out performance

The final held-out discrimination was broadly consistent with the
patient-grouped development estimates.

| Metric | Development | Held-out |
|---|---:|---:|
| PR-AUC | ~0.2070 | 0.1924 |
| ROC-AUC | ~0.6473 | 0.6450 |
| Brier score | ~0.0980 | 0.0980 |

This consistency suggests that the development procedure produced a reasonably
stable estimate of final predictive performance.

It does not establish external validity.

---

## Interpretation

The final model demonstrates **modest discrimination** for 30-day readmission.

Its ROC-AUC of approximately 0.645 indicates useful but limited ranking ability.

PR-AUC must be interpreted relative to the imbalanced outcome prevalence.

Probability quality remained comparatively stable between development and
held-out evaluation, and the calibrated mean risk closely approximated the
observed event prevalence.

At the frozen operating threshold, sensitivity and specificity are both close
to 0.60.

However, because the outcome is imbalanced, precision remains low and the
predicted-positive rate is substantially higher than the observed readmission
prevalence.

This trade-off follows from the pre-specified balanced-accuracy objective.

The results should therefore be interpreted as a methodological demonstration
of a disciplined predictive-development workflow rather than evidence of
clinical deployment readiness.

---

## Preprocessing

All modelling preprocessing is performed inside the machine-learning pipeline
to prevent information leakage.

The final feature schema contains:

```text
42 modelling features
```

including:

- numerical encounter features;
- demographic categorical variables;
- admission characteristics;
- diagnosis-group variables;
- medication variables;
- diabetes-treatment variables;
- medical specialty.

Numerical variables are processed within the modelling pipeline.

Categorical variables are encoded using fold-safe transformations.

Medical specialty is handled separately because of its comparatively high
cardinality.

Preprocessing parameters are learned only from the relevant training portion
of each fold.

---

## Leakage-control strategy

Leakage prevention is a central design feature of this repository.

The workflow includes:

- patient-grouped development splitting;
- zero patient overlap between development and held-out data;
- fold-local preprocessing;
- training-only EDA;
- nested cross-validation for model selection;
- cross-fitted calibration assessment;
- cross-fitted threshold selection;
- frozen model specifications before final evaluation;
- a protected one-time held-out test;
- explicit post-evaluation closure of the development cycle.

The final held-out test set must not be reused for additional optimization.

---

## Repository structure

A simplified view of the project is:

```text
diabetes-readmission-ml-pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── intermediate/
│
├── figures/
│
├── reports/
│   ├── model_selection_protocol.md
│   ├── model_feature_schema.csv
│   ├── w5_model_comparison_findings.md
│   ├── w5_model_comparison_summary.csv
│   ├── w6_model_selection_protocol.md
│   ├── w6_model_selection_decision.md
│   ├── w6_model_selection_summary.csv
│   ├── w6_selected_model.json
│   ├── w6_calibration_protocol.md
│   ├── w6_calibration_findings.md
│   ├── w6_calibration_lock.json
│   ├── w6_threshold_protocol.md
│   ├── w6_threshold_findings.md
│   ├── w6_threshold_lock.json
│   ├── w7_final_evaluation_protocol.md
│   ├── w7_preflight_audit.csv
│   ├── w7_final_metrics.csv
│   ├── w7_final_confusion_matrix.csv
│   ├── w7_final_calibration.csv
│   ├── w7_final_evaluation_findings.md
│   ├── w7_final_evaluation_run_state.json
│   └── w8_project_finalization_protocol.md
│
├── src/
│   └── diabetes_readmission/
│
├── tests/
│
├── model_card.md
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

Dataset-level intermediate prediction files may remain local when appropriate
and are not committed solely for convenience.

---

## Reproducibility

Create or activate the project Python environment and install the required
dependencies.

Example:

```bash
pip install -r requirements.txt
```

The repository also contains a locked dependency specification:

```text
requirements-lock.txt
```

Run the automated test suite with:

```bash
python -m pytest -q
```

The repository reached Week 8 with:

```text
55 tests passing
```

before final documentation work began.

Whitespace and patch integrity can be checked with:

```bash
git diff --check
```

---

## Reproducing project stages

The repository contains dedicated modules for the major modelling stages.

Examples include:

```text
src/diabetes_readmission/model_preprocessing.py
src/diabetes_readmission/modelling.py
src/diabetes_readmission/model_comparison.py
src/diabetes_readmission/model_selection.py
src/diabetes_readmission/selection_lock.py
src/diabetes_readmission/calibration.py
src/diabetes_readmission/threshold_selection.py
src/diabetes_readmission/w7_preflight.py
src/diabetes_readmission/w7_final_evaluation.py
```

Many of the corresponding outputs are stored under:

```text
reports/
```

The report artifacts provide a versioned audit trail of the major modelling
decisions.

---

## Final model card

A detailed methodological description of the completed model-development cycle
is available in:

```text
model_card.md
```

The model card documents:

- baseline modelling;
- model selection;
- final model specification;
- calibration;
- operating-threshold selection;
- final held-out evaluation;
- limitations;
- frozen development status.

---

## Limitations

Several limitations should be considered when interpreting this project.

### Dataset and transportability

The analysis is based on a historical hospital dataset.

Performance in different hospitals, healthcare systems, time periods or patient
populations cannot be assumed from the present evaluation.

### Outcome imbalance

Thirty-day readmission is substantially less frequent than non-readmission.

Consequently, metrics such as accuracy alone would be misleading.

This project therefore emphasizes PR-AUC, ROC-AUC, Brier score, calibration,
sensitivity, specificity and balanced accuracy.

### Moderate discrimination

The final model provides only moderate discrimination.

It should not be interpreted as a high-performance clinical decision system.

### Threshold interpretation

The operating threshold of 0.110 was selected according to a predefined
statistical objective.

It was not derived from clinical costs, treatment effects, resource
constraints or prospective clinical utility.

### No prospective validation

The project does not include:

- prospective validation;
- temporal external validation;
- multi-centre external validation;
- intervention testing;
- clinical-impact analysis.

### Predictive rather than causal modelling

Model features may be associated with readmission risk without being causal
determinants of readmission.

Feature importance or predictive association must therefore not be interpreted
as evidence of causality.

---

## Methodological safeguards

The repository deliberately separates:

```text
model development
        ↓
model selection
        ↓
probability calibration
        ↓
operating-threshold selection
        ↓
frozen specification
        ↓
one-time held-out evaluation
```

This separation reduces the risk of optimistic performance estimates caused by
repeatedly adapting the model to the test set.

After Week 7, the held-out test set is considered consumed.

Any future performance improvement should begin as a new development iteration
using new independent validation data.

---

## What this project demonstrates

This repository is intended to demonstrate practical skills in:

- biomedical data science;
- clinical machine learning;
- Python;
- pandas;
- scikit-learn;
- reproducible modelling;
- patient-level data partitioning;
- leakage prevention;
- nested cross-validation;
- imbalanced classification;
- probability calibration;
- threshold analysis;
- model evaluation;
- automated testing;
- Git and GitHub workflows;
- scientific documentation.

The emphasis is not simply on obtaining the highest possible metric.

The primary objective is to build a **traceable, reproducible and
methodologically defensible predictive workflow**.

---

## Development closure

The Week-7 protected evaluation completed the current predictive-development
cycle.

The following components are now frozen:

1. Random Forest model family;
2. selected hyperparameters;
3. preprocessing and 42-feature schema;
4. sigmoid calibration;
5. operating threshold of 0.110;
6. final held-out evaluation.

Week 8 is restricted to documentation, visualization, QA, reproducibility and
project packaging.

No Week-8 modification may be justified by improving the already observed
held-out performance.

---

## Disclaimer

This repository is a data-science and machine-learning portfolio project.

The model is **not a clinically validated medical device or clinical
decision-support system** and must not be used to make patient-care decisions
without appropriate independent validation, clinical evaluation and regulatory
assessment.

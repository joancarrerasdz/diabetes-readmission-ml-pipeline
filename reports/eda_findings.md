# Training-Set Exploratory Data Analysis

## Scope

This exploratory analysis was performed exclusively on the frozen training split.

- Encounters: 79,473
- Unique patients: 55,952
- Observed 30-day readmission prevalence: 11.39%

The held-out test set was not accessed for exploratory analysis, cleaning decisions, feature selection, or interpretation.

All findings below are descriptive associations in the training data. They must not be interpreted as causal effects.

## Outcome distribution

Thirty-day readmission is an imbalanced outcome, with approximately 11.39% of training encounters classified as readmitted within 30 days.

This supports the planned use of discrimination and class-sensitive metrics such as PR-AUC, ROC-AUC, sensitivity/recall, precision, F1, specificity, and balanced accuracy rather than relying primarily on raw accuracy.

## Age

Observed readmission prevalence varies across age bands and does not show a strictly monotonic relationship.

Among the larger age groups:

- [40-50): approximately 10.80%
- [50-60): approximately 9.94%
- [60-70): approximately 11.34%
- [70-80): approximately 12.04%
- [80-90): approximately 12.34%
- [90-100): approximately 11.88%

Older age bands generally show somewhat higher observed prevalence than the [50-60) group, but the relationship is not linear across all age categories.

Age should therefore be retained as an ordered predictor and later included in subgroup performance auditing.

## Gender

Observed readmission prevalence is very similar between the two main gender groups:

- Female: approximately 11.41%
- Male: approximately 11.36%

Only two encounters were assigned to the explicit Missing category.

No meaningful unadjusted difference is apparent from this descriptive comparison alone.

## Race

Observed prevalence differs modestly across race categories:

- AfricanAmerican: approximately 11.58%
- Caucasian: approximately 11.49%
- Hispanic: approximately 10.36%
- Asian: approximately 10.32%
- Other: approximately 10.22%
- Missing: approximately 8.32%

Group sizes are highly unequal, so these unadjusted proportions must not be interpreted as evidence of causal demographic effects or comprehensive fairness.

Race will be retained for later subgroup performance auditing.

## Prior healthcare utilization

Prior utilization shows some of the clearest descriptive separation between outcome groups.

Mean number of inpatient visits in the preceding year:

- No 30-day readmission: approximately 0.55
- 30-day readmission: approximately 1.24

Median number of prior inpatient visits:

- No 30-day readmission: 0
- 30-day readmission: 1

Other utilization measures are also higher among encounters followed by 30-day readmission:

- Mean prior emergency visits: approximately 0.18 vs 0.36
- Mean prior outpatient visits: approximately 0.36 vs 0.44

These findings support retaining prior healthcare-utilization variables as candidate predictors.

## Encounter intensity and complexity

Several encounter-level measures are modestly higher among encounters followed by 30-day readmission:

- Mean time in hospital: approximately 4.32 vs 4.76 days
- Mean number of laboratory procedures: approximately 42.73 vs 44.20
- Mean number of medications: approximately 15.84 vs 16.98
- Mean number of diagnoses: approximately 7.36 vs 7.69

The mean number of non-laboratory procedures is slightly lower in the readmitted group:

- approximately 1.34 vs 1.29

These are descriptive differences and may reflect underlying clinical complexity rather than independent predictive effects.

## HbA1c

Observed 30-day readmission prevalence by A1Cresult:

- Norm: approximately 9.94%
- >7: approximately 10.15%
- >8: approximately 10.06%
- Not_measured: approximately 11.66%

Because A1C was not measured for most encounters, absence of testing is retained explicitly as a Not_measured category rather than being imputed with the most frequent measured result.

## Maximum serum glucose

Observed 30-day readmission prevalence by max_glu_serum:

- Norm: approximately 11.24%
- >200: approximately 12.54%
- >300: approximately 14.78%
- Not_measured: approximately 11.33%

The >300 category has relatively few encounters, so its higher observed prevalence should be interpreted cautiously.

As with A1Cresult, absence of measurement is retained explicitly as Not_measured.

## Diabetes medication

Observed prevalence is higher when diabetic medication was prescribed:

- diabetesMed = No: approximately 9.90%
- diabetesMed = Yes: approximately 11.83%

Encounters with a recorded change in diabetic medication also have higher observed prevalence:

- change = No: approximately 10.83%
- change = Ch: approximately 12.04%

These associations may reflect treatment intensity or underlying disease severity and should not be interpreted causally.

## Insulin

Observed 30-day readmission prevalence varies across insulin-status categories:

- No: approximately 10.22%
- Steady: approximately 11.36%
- Up: approximately 13.54%
- Down: approximately 14.03%

The higher prevalence in the Up and Down groups may reflect greater treatment complexity or clinical severity.

These values are descriptive and do not imply that insulin adjustment itself causes readmission.

## Data-quality implications

The training-only audit supports the following preprocessing decisions:

- Drop `weight` because approximately 97% of values are missing.
- Drop constant medication variables:
  - `citoglipton`
  - `examide`
  - `glimepiride-pioglitazone`
- Preserve missing laboratory measurements as explicit `Not_measured` categories.
- Preserve missing categorical values explicitly rather than silently applying mode imputation.
- Retain rare but non-constant medication variables for later review.
- Preserve raw `diag_1`, `diag_2`, and `diag_3` codes for later ICD-9 clinical grouping.
- Preserve `medical_specialty` for later rare-category consolidation.

## Interpretation limits

This EDA is exploratory and unadjusted.

Observed differences may reflect:

- disease severity,
- comorbidity burden,
- treatment intensity,
- healthcare utilization,
- coding practices,
- demographic composition,
- or other measured and unmeasured factors.

No causal conclusions are drawn from these descriptive associations.

The held-out test set remains untouched and will only be used for final locked-model evaluation.

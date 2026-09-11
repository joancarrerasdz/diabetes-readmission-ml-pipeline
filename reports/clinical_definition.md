# Clinical Prediction Definition

## Project question

Can routinely collected hospital information available at discharge identify patients with diabetes at increased risk of hospital readmission within 30 days?

## Intended use

Retrospective educational and research prototype for portfolio purposes.

This project is not a clinically validated decision-support system and must not be used for patient care.

## Unit of analysis

One hospital encounter.

The source dataset contains 101,766 encounters from 71,518 unique patients.

Because patients may have multiple encounters, observations from the same patient are not statistically independent for model evaluation.

## Patient grouping

`patient_nbr` is used only as a grouping key.

It must never be included as a predictive feature.

All train/validation/test splitting and cross-validation must prevent the same patient from appearing in both training and held-out evaluation data.

## Prediction time

The prediction point is hospital discharge.

Candidate predictors must therefore represent information known by, or available at, discharge from the index encounter.

Variables containing information occurring after discharge are not eligible predictors.

## Primary outcome

Binary 30-day readmission:

- Positive class (`1`): original `readmitted == "<30"`
- Negative class (`0`): original `readmitted == ">30"` or `"NO"`

The original `readmitted` variable is never used as a predictor.

## Identifiers

`encounter_id` is retained only for traceability and is excluded from modelling.

`patient_nbr` is retained only for patient-level grouping and is excluded from modelling.

## Discharge disposition

`discharge_disposition_id` is available at discharge but is reserved for cohort definition in the primary analysis.

It will not be used as a predictor in the primary model because some discharge dispositions represent situations such as death or hospice care for which subsequent readmission risk is not meaningfully comparable.

Exact exclusion codes must be verified against the official UCI ID mapping before cohort filtering is implemented.

## Cohort exclusions

The primary analysis excludes encounters whose discharge disposition makes subsequent hospital readmission non-interpretable.

The exclusion is applied at the encounter level, not the patient level.

Excluded `discharge_disposition_id` values:

- `11`: Expired
- `13`: Hospice / home
- `14`: Hospice / medical facility
- `19`: Expired at home. Medicaid only, hospice.
- `20`: Expired in a medical facility. Medicaid only, hospice.
- `21`: Expired, place unknown. Medicaid only, hospice.

These codes were verified against the official UCI `IDS_mapping.csv`.

Starting cohort:

- 101,766 encounters
- 71,518 unique patients

After exclusion:

- 99,343 encounters
- 69,990 unique patients

A total of 2,423 encounters are excluded.

Importantly, 871 patients have both excluded and retained encounters. Therefore, only non-interpretable encounters are removed; patients are not globally excluded because of another encounter.

## Evaluation principle

The project evaluates prediction, not causation.

Demographic subgroup analyses will be treated as model performance audits and not as evidence of causal effects or comprehensive fairness.

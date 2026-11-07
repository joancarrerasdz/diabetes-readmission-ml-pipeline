# Data provenance and governance

## Source dataset

This project uses the **Diabetes 130-US Hospitals for Years 1999-2008**
dataset from the UCI Machine Learning Repository.

Dataset metadata:

- Repository: UCI Machine Learning Repository
- UCI dataset ID: 296
- DOI: `10.24432/C5230J`
- Creators: John Clore, Krzysztof Cios, Jon DeShazo and Beata Strack
- Original donation year: 2014
- Source period: 1999–2008
- Source population: hospital encounters involving patients diagnosed with diabetes
- Source size: 101,766 encounters
- UCI-reported features: 47
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Source metadata and licensing were re-verified on 2026-11-07.

The original UCI distribution contains:

- `diabetic_data.csv`
- `IDS_mapping.csv`

## Project use

The project studies prediction of hospital readmission within 30 days.

The original `readmitted` field is transformed into the binary project target:

- `<30` → positive class (`1`)
- `>30` or `NO` → negative class (`0`)

The modelling unit is the hospital encounter.

`patient_nbr` is retained only as a grouping identifier so that encounters
from the same patient cannot cross development and held-out evaluation
partitions.

`encounter_id` is retained only for traceability and is not used as a
predictive feature.

## Data-governance rules

Raw source files are treated as immutable inputs.

The intended data flow is:

    raw -> intermediate -> processed

Raw data must not be manually edited to obtain modelling results.

Patient-level and row-level data files are not committed merely for
convenience. Version-controlled reports contain aggregate results,
metadata and reproducibility evidence where appropriate.

The repository `.gitignore` excludes local raw-data contents while
retaining the directory structure.

## Dataset documentation

Project-specific variable definitions and modelling roles are documented in:

- `data/data_dictionary.csv`
- `reports/clinical_definition.md`
- `reports/leakage_map.csv`

These files document:

- identifiers;
- target construction;
- demographic variables;
- candidate predictors;
- prediction-time availability;
- cohort-definition variables;
- leakage controls.

## License and attribution

The source dataset is distributed by the UCI Machine Learning Repository
under the Creative Commons Attribution 4.0 International license
(CC BY 4.0).

Dataset citation:

Clore J, Cios K, DeShazo J, Strack B. (2014).
Diabetes 130-US Hospitals for Years 1999-2008.
UCI Machine Learning Repository.
DOI: 10.24432/C5230J.

This portfolio repository contains project code, aggregate reports and
derived documentation. Use of the source dataset remains subject to its
original license and attribution requirements.

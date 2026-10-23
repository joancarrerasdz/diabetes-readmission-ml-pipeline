# Week 8 — Project Finalization Protocol

## Purpose

Week 8 closes the diabetes readmission modelling project as a reproducible
portfolio artifact.

The predictive-development cycle was completed during Week 7 with the
protected one-time evaluation of the held-out test set.

Week 8 is therefore a project-finalization stage, not a new modelling stage.

Its purpose is to:

- consolidate the final project narrative;
- improve repository documentation;
- produce final presentation-quality visualizations;
- verify reproducibility and repository integrity;
- document the final scientific and methodological conclusions;
- prepare the repository for portfolio use and versioned release.

No new predictive-development decisions may be introduced during Week 8.

## Frozen development state

The following components were selected and locked before the held-out test set
was accessed and must remain unchanged:

1. model family: Random Forest;
2. model preprocessing pipeline;
3. model feature schema;
4. selected model hyperparameters;
5. probability-calibration strategy: sigmoid;
6. operating threshold: 0.110;
7. patient-safe data-partitioning methodology;
8. final Week-7 held-out evaluation results.

The Week-7 held-out evaluation closed the current model-development cycle.

## Final locked model

The final candidate model is a Random Forest classifier with the locked
configuration documented by the Week-6 selection artifacts.

The model was selected exclusively using the development cohort and
patient-grouped nested cross-validation.

Model selection did not use the held-out test set.

## Probability calibration

Probability calibration was selected before final evaluation.

The locked calibration strategy is:

- sigmoid calibration.

Calibration-method selection was based on cross-fitted development predictions
according to the frozen Week-6 calibration protocol.

The held-out test set was not used to select the calibration strategy.

## Operating threshold

The locked operating threshold is:

- 0.110.

The threshold was selected before final evaluation using patient-safe,
cross-fitted and calibrated development predictions.

The frozen selection criterion was mean balanced accuracy across the
development folds, with the pre-specified tie-break rules documented in the
Week-6 threshold protocol.

The held-out test set was not used to select or modify the operating threshold.

## Held-out test-set status

The held-out test set has been consumed exactly once for the final Week-7
evaluation.

Its results may now be:

- reported;
- visualized;
- summarized;
- compared descriptively with pre-specified development estimates;
- documented in the model card and README.

Its results must not be used for:

- hyperparameter tuning;
- feature engineering;
- feature selection;
- model-family selection;
- recalibration;
- threshold optimization;
- threshold replacement;
- performance-driven modification of the current model.

Any future predictive improvement must be treated as a new development
iteration and must use new independent validation data.

## Week-8 permitted changes

Week 8 may modify or add:

- README documentation;
- model-card documentation;
- scientific interpretation;
- repository documentation;
- reproducibility instructions;
- static visualizations derived from frozen outputs;
- summary tables derived from frozen outputs;
- project architecture documentation;
- metadata and release documentation;
- portfolio-facing presentation material;
- non-modelling QA tests;
- code used exclusively to reproduce presentation artifacts from frozen
  results.

These changes must not alter the frozen predictive specification.

## Week-8 prohibited changes

Week 8 must not:

- retrain candidate models for model selection;
- search new hyperparameters;
- introduce new features for performance improvement;
- remove features based on held-out performance;
- change the selected model family;
- change model hyperparameters;
- change preprocessing;
- change the calibration strategy;
- change the operating threshold;
- repeatedly evaluate alternative models on the held-out test set;
- optimize any model component using Week-7 held-out results.

The final held-out test result is an evaluation result, not a development
dataset.

## Source-of-truth artifacts

The final project narrative must remain consistent with the versioned
development artifacts.

### Week 6 — model selection

Primary artifacts include:

- `reports/w6_selected_model.json`
- `reports/w6_model_selection_decision.md`
- `reports/w6_model_selection_summary.csv`
- `reports/w6_nested_cv_fold_metrics.csv`
- `reports/w6_inner_search_summary.csv`

### Week 6 — probability calibration

Primary artifacts include:

- `reports/w6_calibration_protocol.md`
- `reports/w6_calibration_summary.csv`
- `reports/w6_calibration_findings.md`
- `reports/w6_calibration_lock.json`
- `reports/w6_calibration_decision.md`

### Week 6 — operating threshold

Primary artifacts include:

- `reports/w6_threshold_protocol.md`
- `reports/w6_threshold_summary.csv`
- `reports/w6_threshold_findings.md`
- `reports/w6_threshold_lock.json`
- `reports/w6_threshold_decision.md`

### Week 7 — final evaluation

Primary artifacts include:

- `reports/w7_final_evaluation_protocol.md`
- `reports/w7_preflight_audit.csv`
- `reports/w7_final_metrics.csv`
- `reports/w7_final_confusion_matrix.csv`
- `reports/w7_final_calibration.csv`
- `reports/w7_final_evaluation_findings.md`
- `reports/w7_final_evaluation_run_state.json`
- `model_card.md`

These artifacts constitute the frozen evidence base for Week 8.

## Week-8 deliverables

Week 8 will produce, at minimum:

1. this project-finalization protocol;
2. a complete repository README;
3. final presentation-quality figures;
4. a concise final-results summary;
5. reproducibility and repository QA;
6. a final portfolio-facing repository audit;
7. a versioned project release.

Additional documentation may be produced when it improves transparency or
reproducibility without altering the frozen model-development state.

## Final figures

New figures may be created from already frozen development and final-evaluation
outputs.

Figures may visualize, for example:

- project workflow;
- cohort composition;
- development versus held-out discrimination;
- probability calibration;
- final confusion matrix;
- operating-threshold performance;
- final model performance summary.

Figures must faithfully reproduce existing versioned numerical results.

No new figure may be used as justification for retrospective model tuning.

## README requirements

The final README must explain:

- the clinical prediction problem;
- the dataset and cohort;
- the patient-safe splitting strategy;
- the preprocessing and feature schema;
- baseline modelling;
- nested model selection;
- probability calibration;
- operating-threshold selection;
- protected held-out evaluation;
- final results;
- limitations;
- reproducibility;
- repository structure.

The README must clearly distinguish:

- development performance;
- final held-out performance;
- predictive modelling from clinical validation.

## Scientific interpretation rule

Week-8 interpretation must remain proportional to the evidence.

The project demonstrates a reproducible machine-learning workflow for
predicting 30-day readmission in the analysed dataset.

It does not establish:

- causal effects;
- clinical effectiveness;
- prospective clinical utility;
- transportability to new institutions or populations;
- readiness for clinical deployment.

The selected operating threshold is a methodological operating point for this
development exercise and is not a clinically validated intervention threshold.

## Reproducibility gate

Before final release, the repository must satisfy:

- automated tests pass;
- `git diff --check` passes;
- the working tree is clean;
- documented commands reproduce the intended project artifacts;
- frozen W6 and W7 artifacts remain internally consistent;
- the README reports numerical results accurately;
- no local-only dataset-level prediction files are accidentally committed;
- no held-out performance-driven model modification has occurred.

## Release rule

The final release may be created only after Week-8 QA passes.

The release represents the completed development iteration documented by this
repository.

Any subsequent modelling work intended to improve predictive performance must
begin as a new development iteration rather than silently modifying this
release.

## Freeze rule

This document is the Week-8 project-finalization baseline.

From this point onward, Week-8 changes are restricted to documentation,
visualization, reproducibility, QA and packaging unless a technically necessary
correction is identified.

Any technically necessary correction affecting frozen methodology must be
explicitly documented.

No amendment may be justified by improving the already observed held-out test
performance.

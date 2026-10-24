# Week 8 — Final QA and Reproducibility Audit

## Status

**PASS**

This audit records the final quality-control state of the
`diabetes-readmission-ml-pipeline` after completion of the protected
Week-7 held-out evaluation and the Week-8 documentation,
visualisation, reproducibility and packaging stage.

No model fitting, hyperparameter tuning, feature modification,
recalibration, threshold optimisation or held-out performance-driven
modification was performed during Week 8.

---

## Frozen predictive specification

The final development specification remains:

| Component | Locked value |
|---|---|
| Model family | Random Forest |
| Number of trees | 300 |
| Maximum depth | 12 |
| Minimum samples per leaf | 5 |
| Class weighting | None |
| Random state | 42 |
| Model features | 42 |
| Probability calibration | Sigmoid |
| Operating threshold | 0.110 |
| Final evaluation | One-time protected held-out assessment |

The Week-7 held-out test set has already been consumed and must not be
reused for model development or optimisation.

---

## Final held-out cohort

The protected held-out cohort contained:

- 19,870 encounters
- 14,038 unique patients
- 2,263 positive 30-day readmission outcomes
- 30-day readmission prevalence: approximately 11.39%
- zero patient overlap with the development cohort

Patient-level separation was preserved throughout development and final
evaluation.

---

## Final held-out performance

The locked model achieved approximately:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6450 |
| PR-AUC | 0.1924 |
| Brier score | 0.0980 |
| Log loss | 0.3410 |
| Calibration intercept | -0.0983 |
| Calibration slope | 0.9513 |
| ECE | 0.0065 |
| Operating threshold | 0.110 |
| Sensitivity | 0.5926 |
| Specificity | 0.6115 |
| Precision | 0.1639 |
| F1 | 0.2568 |
| Balanced accuracy | 0.6020 |
| Predicted-positive rate | 0.4118 |

Confusion-matrix counts:

- true negatives: 10,766
- false positives: 6,841
- false negatives: 922
- true positives: 1,341

These results describe predictive performance on the protected held-out
cohort and do not establish clinical utility or transportability.

---

## Automated test status

Final Week-8 test suite: **61 passed**.

The test suite covers the development pipeline together with the
Week-7 protected evaluation and Week-8 visualisation safeguards.

---

## Repository QA

The final repository checks confirmed:

- automated tests pass;
- `git diff --check` passes;
- final Week-7 result artifacts are present;
- final visualisation artifacts are present;
- final run-state artifact records the completed held-out evaluation;
- README and model card describe the frozen final specification;
- raw held-out prediction rows remain excluded from version control;
- no model-development decision was changed during Week 8;
- the working tree was clean after the final documentation commit.

---

## Final versioned artifacts

Core final-evaluation artifacts:

- `reports/w7_final_metrics.csv`
- `reports/w7_final_confusion_matrix.csv`
- `reports/w7_final_calibration.csv`
- `reports/w7_final_evaluation_findings.md`
- `reports/w7_final_evaluation_run_state.json`
- `reports/w7_preflight_audit.csv`

Final visualisations:

- `figures/w7_final_performance_summary.png`
- `figures/w7_final_confusion_matrix.png`
- `figures/w7_final_calibration.png`

Final project documentation:

- `README.md`
- `model_card.md`
- `reports/w8_project_finalization_protocol.md`
- `reports/w8_final_qa_audit.md`

---

## Reproducibility conclusion

The repository preserves a traceable modelling history from data
preparation through patient-safe validation, model selection,
probability calibration, operating-threshold selection and protected
held-out evaluation.

The final Week-8 state is internally consistent with the methodological
locks established before the held-out test set was accessed.

The current predictive-development cycle is therefore considered
**closed**.

Any future attempt to improve model performance must begin as a new
development iteration and must use new independent validation data.

---

## QA conclusion

**W8 FINAL QA: PASS**

The project is ready for final integration into `main`.

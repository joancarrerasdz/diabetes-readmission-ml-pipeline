# Week 6 model-selection decision

## Decision

The selected candidate model family is **Random Forest**.

The candidate is locked with:

- `n_estimators = 300`
- `max_depth = 12`
- `min_samples_leaf = 5`
- `class_weight = None`
- `random_state = 42`

No operating threshold has been selected.

No calibration strategy has been selected.

The held-out test set remains untouched.

## Primary selection criterion

The pre-specified primary model-selection metric is mean outer-fold
PR-AUC from patient-grouped nested cross-validation.

Random Forest:

- mean outer PR-AUC: 0.207017
- SD: 0.010200
- SE: 0.004562

Logistic Regression:

- mean outer PR-AUC: 0.203404
- SD: 0.007016
- SE: 0.003137

Best-model 1-SE cutoff:

- 0.202455

Both candidate families fall within the one-standard-error region and
are therefore treated as competitive on the primary discrimination
criterion.

## Secondary evidence

Random Forest:

- mean outer ROC-AUC: 0.647307
- mean outer Brier score: 0.097992

Logistic Regression:

- mean outer ROC-AUC: 0.641604
- mean outer Brier score: 0.123974

The Random Forest therefore provides modestly stronger discrimination
and substantially better probability accuracy.

## Hyperparameter stability

Random Forest selected the same configuration in all five outer folds.

Logistic Regression produced 2 distinct selected
configurations across the five outer folds.

Four Logistic Regression folds selected `class_weight=None`, whereas
one fold selected `class_weight="balanced"`.

That fold produced a substantial shift in the probability scale and a
markedly worse Brier score. This result is retained as observed; the
search space is not modified retrospectively.

## Cross-fold probability-scale stability

Range of mean predicted probability across outer folds:

- Logistic Regression: 0.358280
- Random Forest: 0.001999

The Random Forest predictions are substantially more consistent in
probability scale across the patient-grouped outer folds.

## Selection rationale

Because both candidates satisfy the one-standard-error rule, the
decision is not based on the small difference in PR-AUC alone.

Random Forest is selected because the nested-CV evidence jointly shows:

1. slightly higher PR-AUC;
2. slightly higher ROC-AUC;
3. substantially better Brier score;
4. identical hyperparameter selection across all five outer folds;
5. substantially more stable probability scale across outer folds.

Logistic Regression remains the principal interpretable reference
baseline.

Its interpretability advantage is acknowledged, but it does not
outweigh the probability-quality and stability evidence observed in the
pre-specified nested evaluation.

## Methodological safeguard

The decision uses training data and patient-safe nested
cross-validation only.

No retrospective modification of the candidate search spaces was made.

The held-out test set has not been accessed.

Threshold selection, calibration decisions and final held-out
evaluation remain separate later stages.

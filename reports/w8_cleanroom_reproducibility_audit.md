# Week 8 — Clean-room Reproducibility Certification

## Status

**PASS**

This audit records an independent clean-room verification of the
`diabetes-readmission-ml-pipeline` repository under Python 3.11 after
completion of the protected Week-7 held-out evaluation.

The purpose of this verification was to assess reproducibility,
environment reconstruction and automated-test execution without reopening
or repeating the final held-out evaluation.

---

## Scope

The clean-room verification covered:

- fresh repository clone;
- fresh Python 3.11 virtual environment;
- dependency installation;
- project package installation;
- package import without manual `PYTHONPATH`;
- safe automated-test collection;
- safe automated-test execution;
- explicit verification of the Week-7 final-evaluation firewall;
- preservation of the completed final-evaluation run state;
- repository cleanliness after testing.

No model fitting, calibration fitting, operating-threshold optimisation or
held-out predictive evaluation was performed during this certification.

---

## Clean-room environment

The verification was performed from an independent clone located outside
the development working copy.

Environment:

- Python 3.11;
- isolated virtual environment;
- repository cloned from the version-controlled project state;
- project installed as a Python package;
- no manual `PYTHONPATH` required after project installation.

The project package was installed after dependency installation using:

    python -m pip install --no-deps -e .

The `diabetes_readmission` package was subsequently importable directly
from the clean-room environment.

---

## Safe automated-test execution

The protected Week-7 final-evaluation test module was excluded from the
general suite to prevent accidental re-execution of the consumed held-out
evaluation.

Safe test collection:

    56 tests collected

Safe test execution:

    56 passed

The Week-7 execution firewall was then tested independently using:

    tests/test_w7_final_evaluation.py::test_final_evaluation_is_blocked_without_confirmation

Result:

    1 passed

The independent clean-room verification therefore executed:

    56 safe pipeline tests
    + 1 final-evaluation firewall test
    ---------------------------------
    57 successful QA checks

---

## Held-out protection

The environment variable required to authorize the protected final
evaluation was explicitly unset before the firewall test.

The one-time final held-out evaluation was **not re-executed**.

The existing run-state artifact remained:

    {
      "calibration_strategy": "sigmoid",
      "candidate_model": "random_forest",
      "operating_threshold": 0.11,
      "stage": "final_held_out_evaluation",
      "status": "completed",
      "week": 7
    }

This confirms that the protected held-out evaluation remains recorded as
completed and was not reopened for model development, feature engineering,
model selection, calibration selection or threshold optimisation.

---

## Frozen predictive specification

The clean-room QA did not modify the frozen predictive specification.

The final locked state remains:

- model family: Random Forest;
- number of trees: 300;
- maximum depth: 12;
- minimum samples per leaf: 5;
- class weighting: none;
- random state: 42;
- model features: 42;
- probability calibration: sigmoid;
- operating threshold: 0.110;
- final evaluation: one-time protected held-out assessment.

No post-hoc predictive modification was performed.

---

## Repository integrity

Following clean-room testing:

    git diff --check

completed successfully.

Likewise:

    git status --short

returned no output.

Therefore the independent verification did not modify the cloned
repository state.

---

## Reproducibility finding

The initial clean-room attempt showed that installing the dependency lock
alone does not install the local `src/` package.

This was an environment/package-discovery issue rather than a modelling,
analysis or test-correctness failure.

The repository already contains Python packaging configuration. Installing
the project package into the isolated environment makes
`diabetes_readmission` importable without manually setting `PYTHONPATH`.

The reproducible clean-room procedure is therefore:

1. clone the repository;
2. create and activate an isolated Python 3.11 virtual environment;
3. install the locked project dependencies;
4. install the repository package;
5. execute the safe QA test suite;
6. execute only the final-evaluation firewall test from the protected W7
   test module.

This makes the package-installation requirement explicit and reproducible.

---

## Independent QA evidence

The clean-room execution independently confirmed:

- 56 safe tests collected;
- 56 safe tests passed;
- Week-7 final-evaluation firewall test passed;
- final run state remained `completed`;
- candidate model remained `random_forest`;
- calibration strategy remained `sigmoid`;
- operating threshold remained `0.11`;
- the held-out evaluation was not repeated;
- no repository files were modified by the verification.

---

## QA interpretation

This clean-room run provides independent execution evidence supporting the
Week-8 reproducibility and testing requirements.

It demonstrates that:

- the repository can be reconstructed in an independent Python 3.11
  environment;
- the project package can be installed and imported;
- the safe automated-test suite passes independently;
- the final-evaluation firewall blocks unauthorized execution;
- the consumed held-out state remains unchanged;
- no model-development decision was modified;
- no held-out performance was used to justify a correction;
- repository state remains clean after verification.

This certification is limited to reproducibility and safe QA execution.
It does not constitute additional model validation and does not establish
clinical utility or external transportability.

---

## Conclusion

**CLEAN-ROOM REPRODUCIBILITY QA: PASS**

The current predictive-development cycle remains closed.

The clean-room certification confirms reproducible and safe execution of
the available automated QA without repeating the protected Week-7 held-out
evaluation.

Any future attempt to improve predictive performance must begin as a new
development iteration and use new independent validation data.

# ML Testing & Validation — 6 branches

# 🍏 Eva's Tools

  

> 💡 _A lighthouse for developers, scientists & researchers — small tools that make the everyday a little brighter._

  

💖 **Welcome!** This is my main public repository: a growing collection of examples and functions to help with your day-to-day work as a **developer**, **scientist**, or **researcher**.

## What's inside

A fresh git repo with `main` (shared scaffold) and 6 feature branches:

  

| Branch                     | Module                     | Tests                      | Stack                                                    |
| -------------------------- | -------------------------- | -------------------------- | -------------------------------------------------------- |
| `feature/unit-tests`       | — (tests the pipeline)     | `tests/test_pipeline.py`   | pytest                                                   |
| `feature/input-validation` | `src/mlpipe/validation.py` | `tests/test_validation.py` | Pydantic v2, Great Expectations                          |
| `feature/drift-detection`  | `src/mlpipe/drift.py`      | `tests/test_drift.py`      | Evidently AI (+ PSI fallback)                            |
| `feature/ab-testing`       | `src/mlpipe/ab_testing.py` | `tests/test_ab_testing.py` | scipy two-proportion z-test                              |
| `feature/fairness-testing` | `src/mlpipe/fairness.py`   | `tests/test_fairness.py`   | Fairlearn, AIF360 (+ numpy fallback), lifespan guardrail |
| `feature/cv-tuning`        | `src/mlpipe/tuning.py`     | `tests/test_tuning.py`     | Optuna, GridSearchCV, StratifiedKFold                    |

  

Shared on `main`: `src/mlpipe/data.py` (deterministic synthetic data, with a `drift=` knob)

and `src/mlpipe/model.py` (a small PyTorch MLP wrapped in an sklearn-compatible `TorchClassifier`).

  

## Restore the repo (recommended — keeps all branches)

```bash

git clone ml-testing-validation.bundle ml-testing

cd ml-testing

git branch -a # see all 6 feature branches

git checkout feature/unit-tests

```

  

## Or use the plain tarball (main branch tree only)

```bash

tar xzf ml-testing-validation-repo.tar.gz

```

`ml-testing-branches-readable.tar.gz` contains each branch's files in its own folder

(`feature_unit-tests/`, etc.) if you just want to read them without git.

  

## Run any branch

```bash

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

pytest

```

  

## Heavy-library tests are guarded

Tests for Great Expectations, Evidently, Fairlearn, and Optuna use

`pytest.importorskip`, so they skip cleanly if a lib isn't installed and your

core suite still runs. Markers: `ge`, `evidently`, `fairlearn`, `optuna`.

Run only fast tests, e.g.: `pytest -m "not optuna and not evidently"`.

  

## CI

Each branch ships its own `.github/workflows/ci.yml` that installs

requirements and runs pytest on push/PR.

  

## Notes / things to wire to your real setup

- `requirements.txt` accumulates per branch. When you merge branches, dedupe it.

- The PyTorch model is intentionally tiny (4-feature MLP). Swap `TorchClassifier`

for your real estimator — the interface (`fit/predict/predict_proba/get_params`)

is what GridSearchCV, Optuna, Fairlearn, and Evidently rely on.

- AIF360 has heavier system deps; Fairlearn covers the same metrics with a

lighter footprint, which is why the example leans on it.

- I couldn't run pytest in this environment (no network to install it), but I

validated the core logic of each module directly: synthetic data + drift knob,

PSI drift separation, the A/B z-test (correctly picks the better arm at p≈0),

and the fairness metrics (DPD/EOD + lifespan guardrail). The torch/heavy-lib

paths follow each library's current documented API.
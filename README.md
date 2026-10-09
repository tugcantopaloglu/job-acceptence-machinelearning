# Exam-score logistic regression experiment

An educational binary classification exercise implementing logistic regression with scalar stochastic gradient descent. The included data has two exam-score features (`Exam1`, `Exam2`) and an admission label (`Admitted`). Despite the repository name, it does not contain job offer acceptance features, a recruiting API, notebooks, or a saved-model prediction interface.

## Setup and run

Use Python 3.10 or newer; the regression checks use Python 3.12 and the versions in `requirements.txt`. Run commands from the repository root because data and output paths are relative to the working directory.

```bash
git clone https://github.com/tugcantopaloglu/job-acceptence-machinelearning.git
cd job-acceptence-machinelearning
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS), then:

```bash
python -m pip install -r requirements.txt
python main.py
```

The default run displays Matplotlib plots, trains a baseline for 50 epochs and a regularized model for up to 1,000 epochs. To run a small, headless check:

```bash
python main.py --no-plots --epochs 2 --regularized-epochs 3
```

## Data and evaluation

`data/hw1Data.txt` is a headerless CSV in `Exam1,Exam2,Admitted` order. The cached split files have headers. If all three split files exist, they are reused. Otherwise, the loader reads the raw file and creates stratified 60% training, 20% validation and 20% test splits using random state 42. To change the dataset, prepare a separate working copy with matching raw data and regenerate all three split files together.

Both models fit only the training split. Each model selects its own probability threshold from 0.10 through 0.99 in steps of 0.01 by maximizing validation F1. It then reports test accuracy, precision, recall and F1 at both 0.5 and the selected threshold. Test labels do not influence threshold selection. Regularized early stopping monitors the training epoch loss, rather than validation loss.

The hand-written model has no intercept or feature scaling. This preserves the educational algorithm and can limit its accuracy on the included positive exam scores. The test checks below validate behavior, not a claimed model-quality benchmark.

## Files and outputs

| Path | Purpose |
| --- | --- |
| `DataHandler.py` | CSV loading, cached splitting and training-data scatter plot |
| `LogisticRegression.py` | Sigmoid, cross-entropy, SGD, L2 regularization and binary metrics |
| `main.py` | Training, validation threshold selection and held-out test evaluation |
| `results/epoch_loss_output.txt` | Appended epoch loss logs |
| `results/weights.txt` | Appended model weights |
| `results/scores.txt` | Appended test metric logs |
| `IEEE_Report.pdf` | Historical course report |
| `tests/test_regressions.py` | Small regression fixtures and an actual CLI run |

The `results` directory is created when needed. Plots are displayed interactively; this code does not automatically export plot PNG files.

## Historical result provenance

The checked-in reports, plots and text logs are historical artifacts. Their exact environment and run identity are not recorded. The earlier script trained additional models on the validation split and optimized thresholds on the test split, including applying another model's threshold to the baseline. Historical tuned scores therefore cannot be treated as unbiased held-out estimates, and are not claims about the current code. Runs append logs; use a separate working copy for a clean experiment.

This repository and [job-acceptence-logistic-regression-machine-learning](https://github.com/tugcantopaloglu/job-acceptence-logistic-regression-machine-learning) contain related copies of the educational experiment and remain separate projects with their own licenses.

## Regression checks

```bash
python -m pip install pytest
python -m pytest -q
python -m pip check
```

The tests cover stable sigmoid values at extreme inputs, metric length validation, validation-only threshold selection, missing split/output handling, and a two-model CLI run on 40 fixture rows. They do not regenerate historical research results.

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).

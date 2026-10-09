import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from LogisticRegression import LogisticRegression
from main import CalculateBestThreshold


@pytest.mark.parametrize("value, expected", [(-1000, 0.0), (0, 0.5), (1000, 1.0)])
def test_sigmoid_handles_extreme_scores(value, expected):
    assert LogisticRegression("fixture").Sigmoid(value) == expected


def test_binary_metrics_and_length_validation():
    model = LogisticRegression("fixture")
    assert model.Metrics([1, 0, 1, 0], [1, 1, 0, 0]) == (0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        model.Metrics([1, 0], [1])


def test_threshold_search_uses_validation_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    scores = np.log(np.array([0.2, 0.4]) / (1 - np.array([0.2, 0.4])))
    validation = pd.DataFrame({"Exam1": scores, "Exam2": [0, 0], "Admitted": [0, 1]})
    test = validation.assign(Admitted=[1, 0])
    validation.to_csv("data/train_data.txt", index=False)
    validation.to_csv("data/validate_data.txt", index=False)
    test.to_csv("data/test_data.txt", index=False)
    model = LogisticRegression("fixture")
    model.weights = [1, 0]
    threshold = CalculateBestThreshold(model)
    assert 0.2 < threshold <= 0.4
    assert model.CalculateMetricsOnValidationData(threshold) == 1
    assert model.CalculateMetricsOnTestData(threshold, check_threshold=True) == 0


@pytest.mark.parametrize("regularized", [False, True])
def test_training_creates_results_and_exposes_losses(tmp_path, monkeypatch, regularized):
    monkeypatch.chdir(tmp_path)
    model = LogisticRegression("fixture")
    train = model.StochasticGradientDescentWithRegularization if regularized else model.StochasticGradientDescent
    losses = train([[0], [1]], [0, 1], [0], 0.1, 3)
    assert len(losses) == 3
    assert np.isfinite(losses).all()
    assert Path("results/weights.txt").exists()
    assert model.CalculateStoppingEpoch(100) == 2


def test_cli_rebuilds_missing_splits_and_evaluates_only_training_models(tmp_path):
    Path(tmp_path, "data").mkdir()
    rows = pd.DataFrame({"Exam1": np.arange(40) / 40, "Exam2": np.arange(40) / 40, "Admitted": [0, 1] * 20})
    rows.to_csv(tmp_path / "data/hw1Data.txt", header=False, index=False)
    entrypoint = Path(__file__).resolve().parents[1] / "main.py"
    result = subprocess.run(
        [sys.executable, str(entrypoint), "--no-plots", "--epochs", "2", "--regularized-epochs", "3"],
        cwd=tmp_path, env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert "Validate Model" not in result.stdout
    assert result.stdout.count("Model Evaluation on Test Set") == 4
    assert [len(pd.read_csv(tmp_path / "data" / name)) for name in ("train_data.txt", "validate_data.txt", "test_data.txt")] == [24, 8, 8]
    assert (tmp_path / "results/scores.txt").exists()

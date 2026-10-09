import argparse
import numpy as np
import DataHandler
import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the exam-score logistic regression experiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--regularized-epochs", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.regularized_epochs < 1:
        parser.error("Epoch counts must be positive")

    data = DataHandler.DataHandler("./data/hw1Data.txt")
    data.DataSplitter()
    if not args.no_plots:
        data.DataVisualazation()
    x_train = data.TrainDataGet()[["Exam1", "Exam2"]].to_numpy()
    y_train = data.TrainDataGet()["Admitted"].to_numpy()

    baseline = LogisticRegression.LogisticRegression("Train Model")
    baseline_losses = baseline.StochasticGradientDescent(x_train, y_train, [0.0, 0.0], 0.001, args.epochs)
    regularized = LogisticRegression.LogisticRegression("Improved Train Model")
    regularized_losses = regularized.StochasticGradientDescentWithRegularization(
        x_train, y_train, [0.0, 0.0], 0.001, args.regularized_epochs, lambda_reg=0.01, patience=5
    )
    if not args.no_plots:
        baseline.GenerateLossGraph(baseline_losses)
        regularized.GenerateLossGraph(regularized_losses)

    for model in (baseline, regularized):
        threshold = CalculateBestThreshold(model)
        model.CalculateMetricsOnTestData()
        model.CalculateMetricsOnTestData(threshold=threshold)


def CalculateBestThreshold(model, precision=0.01):
    if not 0 < precision < 1:
        raise ValueError("Threshold step must be between zero and one")
    best_f1 = -float("inf")
    best_threshold = 0.5
    for threshold in np.arange(0.1, 1.0, precision):
        value = model.CalculateMetricsOnValidationData(threshold=threshold)
        if value > best_f1:
            best_threshold = float(threshold)
            best_f1 = value
    print(f"Best validation threshold for {model.model_name}: {best_threshold}")
    return best_threshold


if __name__ == "__main__":
    main()

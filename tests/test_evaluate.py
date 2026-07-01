import numpy as np

from src.evaluate import calculate_mape, evaluate_metrics


def test_calculate_mape_basic():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 300])
    mape = calculate_mape(y_true, y_pred)
    # errors: 10/100=10%, 10/200=5%, 0/300=0% -> mean = 5%
    assert round(mape, 2) == 5.0


def test_calculate_mape_all_zero_actuals_returns_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([5, 10, 1])
    assert calculate_mape(y_true, y_pred) == 0.0


def test_calculate_mape_skips_zero_actuals_but_uses_rest():
    y_true = np.array([0, 100])
    y_pred = np.array([50, 90])
    # zero-actual row must be excluded, not treated as div-by-zero -> 0%
    mape = calculate_mape(y_true, y_pred)
    assert round(mape, 2) == 10.0


def test_evaluate_metrics_perfect_prediction():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])
    metrics = evaluate_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0


def test_evaluate_metrics_returns_expected_keys():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    metrics = evaluate_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}
    assert all(isinstance(v, float) for v in metrics.values())

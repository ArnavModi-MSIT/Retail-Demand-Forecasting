import numpy as np
import pandas as pd

from src.future_forecast import _compute_lag_rolling, _compute_customer_features


def _naive_lag_rolling(sales_buf: np.ndarray) -> dict:
    """
    Reference implementation using plain indexing/slicing instead of the
    buffer-walk in _compute_lag_rolling, as a correctness oracle for the
    O(1) rewrite. Computes features for the *next* (not-yet-existing) day:
    lag_k is simply the k-th value from the end of history, and rolling
    stats are taken over the trailing window of *known* history — no
    extra shift is applied, since the day being forecast isn't in the
    buffer yet (unlike engineer_features(), which shifts because it's
    computing features for rows that already have a sales value).
    """
    s = pd.Series(sales_buf)
    features = {}
    for lag in [1, 7, 14, 28]:
        features[f"lag_{lag}"] = float(s.iloc[-lag]) if len(s) >= lag else 0.0
    for window in [7, 14, 30]:
        if len(s) >= window:
            features[f"rolling_mean_{window}"] = float(s.iloc[-window:].mean())
        else:
            features[f"rolling_mean_{window}"] = 0.0
    for window in [7, 30]:
        if len(s) >= window:
            features[f"rolling_std_{window}"] = float(s.iloc[-window:].std())
        else:
            features[f"rolling_std_{window}"] = 0.0
    return features


def test_compute_lag_rolling_matches_pandas_reference_full_history():
    rng = np.random.default_rng(42)
    sales_buf = rng.integers(10, 200, size=60).astype(float)

    fast = _compute_lag_rolling(sales_buf)
    naive = _naive_lag_rolling(sales_buf)

    for key in naive:
        assert fast[key] == pytest_approx(naive[key]), f"mismatch on {key}"


def test_compute_lag_rolling_short_history_returns_zeros_not_errors():
    # Fewer than 30 points — every lag/rolling window should degrade to 0.0
    # instead of raising or returning NaN (which would break model.predict).
    sales_buf = np.array([10.0, 20.0, 30.0])
    features = _compute_lag_rolling(sales_buf)
    assert features["lag_28"] == 0.0
    assert features["rolling_mean_30"] == 0.0
    assert features["lag_1"] == 30.0  # last value, since n >= 1


def test_compute_lag_rolling_no_nans_in_output():
    sales_buf = np.array([5.0, 10.0])
    features = _compute_lag_rolling(sales_buf)
    assert not any(pd.isna(v) for v in features.values())


def test_compute_customer_features_basic():
    cust_buf = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    features = _compute_customer_features(cust_buf, lag_1_sales=100.0)
    assert features["customers_lag_1"] == 70.0
    assert features["customers_lag_7"] == 10.0
    assert features["customer_to_sales_ratio"] == pytest_approx(70.0 / 100.0)


def test_compute_customer_features_short_buffer_defaults_lag7_to_zero():
    cust_buf = np.array([10.0, 20.0])
    features = _compute_customer_features(cust_buf, lag_1_sales=50.0)
    assert features["customers_lag_7"] == 0.0
    assert features["customers_lag_1"] == 20.0


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol
    return _Approx()

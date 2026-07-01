import numpy as np
import pandas as pd

from src.inventory import compute_inventory


def _make_history_future(store_sales, forecast_total, n_history_days=60):
    """Build minimal history_df / future_df for one store/dept combo."""
    dates = pd.date_range("2024-01-01", periods=n_history_days)
    history_df = pd.DataFrame({
        "store_id": "S1",
        "dept_id": "D1",
        "date": dates,
        "sales": store_sales,
    })
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7)
    future_df = pd.DataFrame({
        "store_id": "S1",
        "dept_id": "D1",
        "date": future_dates,
        "predicted_sales": forecast_total / 7,
    })
    return history_df, future_df


def test_low_risk_when_demand_is_stable():
    # Constant demand -> std=0 -> CV=0 -> Low Risk
    sales = [100] * 60
    history_df, future_df = _make_history_future(sales, forecast_total=700)
    inv = compute_inventory(history_df, future_df, service_level=0.95, lead_time_days=7)
    assert inv.loc[0, "risk_level"] == "Low Risk"


def test_critical_risk_when_demand_is_highly_volatile():
    # Alternate between very low and very high demand -> high CV -> Critical
    rng = np.random.default_rng(0)
    sales = list(rng.choice([10, 500], size=60))
    history_df, future_df = _make_history_future(sales, forecast_total=700)
    inv = compute_inventory(history_df, future_df, service_level=0.95, lead_time_days=7)
    assert inv.loc[0, "risk_level"] in ("High Risk", "Critical")


def test_zero_sales_days_excluded_from_cv_calculation():
    # Closure days (sales == 0) should not inflate volatility for open days.
    sales = [100] * 50 + [0] * 10  # 10 closure days mixed in
    history_df, future_df = _make_history_future(sales, forecast_total=700)
    inv = compute_inventory(history_df, future_df, service_level=0.95, lead_time_days=7)
    # avg_daily_demand should reflect only the open days (100), not be
    # dragged toward 0 by closure days.
    assert inv.loc[0, "avg_daily_demand"] == pytest_approx(100, abs_tol=1)


def pytest_approx(value, abs_tol):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= abs_tol
    return _Approx()


def test_safety_stock_scales_with_service_level():
    sales = [100, 120, 80, 110, 90] * 12
    history_df, future_df = _make_history_future(sales, forecast_total=700)

    inv_90 = compute_inventory(history_df, future_df, service_level=0.90, lead_time_days=7)
    inv_99 = compute_inventory(history_df, future_df, service_level=0.99, lead_time_days=7)

    # Higher service level -> higher safety stock (larger z-score)
    assert inv_99.loc[0, "safety_stock"] >= inv_90.loc[0, "safety_stock"]


def test_reorder_point_equals_lead_time_demand_plus_safety_stock():
    sales = [100, 120, 80, 110, 90] * 12
    history_df, future_df = _make_history_future(sales, forecast_total=700)
    inv = compute_inventory(history_df, future_df, service_level=0.95, lead_time_days=7)

    expected = inv.loc[0, "expected_lead_time_demand"] + inv.loc[0, "safety_stock"]
    assert inv.loc[0, "reorder_point"] == round(expected, 2)


def test_no_group_cols_falls_back_to_ungrouped_aggregation():
    dates = pd.date_range("2024-01-01", periods=30)
    history_df = pd.DataFrame({"date": dates, "sales": [50] * 30})
    future_df = pd.DataFrame({
        "date": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7),
        "predicted_sales": [50] * 7,
    })
    inv = compute_inventory(history_df, future_df, service_level=0.95, lead_time_days=7)
    assert len(inv) == 1
    assert inv.loc[0, "avg_daily_demand"] == 50.0

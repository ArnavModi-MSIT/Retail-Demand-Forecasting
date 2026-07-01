import numpy as np
import pandas as pd

from src.error_analysis import compute_error_segments, MIN_SEGMENT_ROWS


def _make_predictions(n_per_store=20, weekend_bias_store=None):
    """
    Build a synthetic predictions_df spanning two stores. If
    weekend_bias_store is set, that store's weekend predictions carry a
    deliberate additional error so the weekend segment is distinguishable
    from weekday in tests.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_per_store)
    rows = []
    for store in ["S1", "S2"]:
        for d in dates:
            is_weekend = d.dayofweek in (5, 6)
            actual = 100 + (30 if is_weekend else 0) + rng.normal(0, 2)
            bias = 25 if (is_weekend and store == weekend_bias_store) else rng.normal(0, 1)
            rows.append(dict(
                date=d, store_id=store, dept_id="D1",
                actual_sales=actual, predicted_sales=actual + bias,
            ))
    return pd.DataFrame(rows)


def _make_df_featured(n_per_store=20):
    dates = pd.date_range("2024-01-01", periods=n_per_store)
    rows = []
    for store in ["S1", "S2"]:
        for d in dates:
            rows.append(dict(date=d, store_id=store, is_weekend=int(d.dayofweek in (5, 6))))
    return pd.DataFrame(rows)


def _make_inventory(avg_daily_demand=(200.0, 50.0)):
    return pd.DataFrame({
        "store_id": ["S1", "S2"],
        "dept_id": ["D1", "D1"],
        "avg_daily_demand": list(avg_daily_demand),
    })


# ── Empty / missing input handling ───────────────────────────────────────────

def test_compute_error_segments_empty_predictions_returns_empty_dict():
    result = compute_error_segments(pd.DataFrame())
    assert result["by_store"].empty
    assert result["by_weekend"].empty
    assert result["by_demand_tier"].empty


def test_compute_error_segments_missing_date_column_returns_empty():
    df = pd.DataFrame({"store_id": ["S1"], "actual_sales": [10], "predicted_sales": [12]})
    result = compute_error_segments(df)
    assert result["by_store"].empty


def test_compute_error_segments_without_df_featured_skips_weekend_segment():
    preds = _make_predictions(n_per_store=20)
    result = compute_error_segments(preds, df_featured=None, inventory_df=None)
    assert result["by_weekend"].empty
    # by_store should still work — it only needs predictions_df
    assert not result["by_store"].empty


def test_compute_error_segments_without_inventory_skips_demand_tier_segment():
    preds = _make_predictions(n_per_store=20)
    df_featured = _make_df_featured(n_per_store=20)
    result = compute_error_segments(preds, df_featured=df_featured, inventory_df=None)
    assert result["by_demand_tier"].empty
    assert not result["by_weekend"].empty


# ── Minimum row-count guard ───────────────────────────────────────────────────

def test_segments_with_too_few_rows_are_dropped():
    # Only 5 rows per store — below MIN_SEGMENT_ROWS (10) — should be dropped.
    preds = _make_predictions(n_per_store=5)
    result = compute_error_segments(preds)
    assert result["by_store"].empty


def test_segments_at_or_above_min_rows_are_kept():
    preds = _make_predictions(n_per_store=MIN_SEGMENT_ROWS)
    result = compute_error_segments(preds)
    assert not result["by_store"].empty
    assert set(result["by_store"]["n_rows"]) == {MIN_SEGMENT_ROWS}


# ── Correctness of segment metrics ───────────────────────────────────────────

def test_by_store_flags_the_higher_error_store():
    # Give S1 a large, obvious bias so it's unambiguously the worst store.
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=30)
    rows = []
    for store, bias in [("S1", 50.0), ("S2", 1.0)]:
        for d in dates:
            actual = 100 + rng.normal(0, 2)
            rows.append(dict(date=d, store_id=store, dept_id="D1",
                              actual_sales=actual, predicted_sales=actual + bias))
    preds = pd.DataFrame(rows)

    result = compute_error_segments(preds)
    by_store = result["by_store"]
    # sorted descending by MAE — worst store should be first
    assert by_store.iloc[0]["store_id"] == "S1"
    assert by_store.iloc[0]["MAE"] > by_store.iloc[1]["MAE"]


def test_weekend_segment_shows_higher_error_when_bias_injected():
    preds = _make_predictions(n_per_store=60, weekend_bias_store="S1")
    df_featured = _make_df_featured(n_per_store=60)
    result = compute_error_segments(preds, df_featured=df_featured)

    by_weekend = result["by_weekend"]
    assert not by_weekend.empty
    weekend_mae = by_weekend.loc[by_weekend["day_type"] == "Weekend", "MAE"].values[0]
    weekday_mae = by_weekend.loc[by_weekend["day_type"] == "Weekday", "MAE"].values[0]
    assert weekend_mae > weekday_mae


def test_demand_tier_segment_splits_high_and_low_correctly():
    preds = _make_predictions(n_per_store=30)
    inventory_df = _make_inventory(avg_daily_demand=(300.0, 20.0))  # S1 high, S2 low

    result = compute_error_segments(preds, df_featured=None, inventory_df=inventory_df)
    by_tier = result["by_demand_tier"]
    assert not by_tier.empty
    assert set(by_tier["demand_tier"]) == {"High Demand", "Low Demand"}


def test_mape_excludes_zero_actual_rows():
    dates = pd.date_range("2024-01-01", periods=15)
    rows = []
    for i, d in enumerate(dates):
        actual = 0 if i == 0 else 100.0
        rows.append(dict(date=d, store_id="S1", dept_id="D1",
                          actual_sales=actual, predicted_sales=actual + 10))
    preds = pd.DataFrame(rows)

    result = compute_error_segments(preds)
    by_store = result["by_store"]
    assert not by_store.empty
    # MAPE should be finite (10/100 * 100 = 10%), not NaN/inf from the zero-actual row
    mape = by_store.iloc[0]["MAPE"]
    assert mape == 10.0

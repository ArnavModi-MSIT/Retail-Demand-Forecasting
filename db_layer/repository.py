"""
Repository layer — all database reads/writes go through these functions.

Keeps SQL out of the UI and pipeline code. Every function takes plain
Python/pandas objects in, and returns plain Python/pandas objects out.
"""
import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from db_layer.connection import get_session


# ── Runs ──────────────────────────────────────────────────────────────────────

def save_run(
    dataset_name: str,
    n_rows: int,
    n_stores: int,
    n_depts: int,
    model_choice: str,
    horizon_days: int,
    service_level: float,
    lead_time: int,
    test_days: int,
) -> str:
    """Insert a new run record. Returns the generated run_id (str)."""
    run_id = str(uuid.uuid4())
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO runs (
                    run_id, dataset_name, n_rows, n_stores, n_depts,
                    model_choice, horizon_days, service_level, lead_time, test_days
                ) VALUES (
                    :run_id, :dataset_name, :n_rows, :n_stores, :n_depts,
                    :model_choice, :horizon_days, :service_level, :lead_time, :test_days
                )
            """),
            dict(
                run_id=run_id, dataset_name=dataset_name, n_rows=n_rows,
                n_stores=n_stores, n_depts=n_depts, model_choice=model_choice,
                horizon_days=horizon_days, service_level=service_level,
                lead_time=lead_time, test_days=test_days,
            ),
        )
    return run_id


def list_runs(limit: int = 20) -> pd.DataFrame:
    """Return the most recent runs, newest first."""
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT run_id, created_at, dataset_name, n_rows, n_stores,
                       model_choice, horizon_days, service_level, lead_time
                FROM runs
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            dict(limit=limit),
        )
        rows = result.mappings().all()
    return pd.DataFrame(rows)


def get_run(run_id: str) -> dict | None:
    """Return a single run's config, or None if not found."""
    with get_session() as session:
        result = session.execute(
            text("SELECT * FROM runs WHERE run_id = :run_id"),
            dict(run_id=run_id),
        )
        row = result.mappings().first()
    return dict(row) if row else None


# ── Model metrics ─────────────────────────────────────────────────────────────

def save_model_metrics(run_id: str, results_df: pd.DataFrame) -> None:
    """Save one row per model from a results_df (columns: Model, MAE, RMSE, MAPE)."""
    if results_df.empty:
        return
    records = [
        dict(
            run_id=run_id, model_name=row["Model"],
            mae=float(row["MAE"]), rmse=float(row["RMSE"]), mape=float(row["MAPE"]),
        )
        for _, row in results_df.iterrows()
    ]
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO model_metrics (run_id, model_name, mae, rmse, mape)
                VALUES (:run_id, :model_name, :mae, :rmse, :mape)
            """),
            records,
        )


def get_model_metrics(run_id: str) -> pd.DataFrame:
    with get_session() as session:
        result = session.execute(
            text("SELECT model_name AS \"Model\", mae AS \"MAE\", rmse AS \"RMSE\", mape AS \"MAPE\" "
                 "FROM model_metrics WHERE run_id = :run_id"),
            dict(run_id=run_id),
        )
        rows = result.mappings().all()
    df = pd.DataFrame(rows)
    for col in ["MAE", "RMSE", "MAPE"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── Forecasts ─────────────────────────────────────────────────────────────────

def save_forecasts(run_id: str, future_df: pd.DataFrame) -> None:
    """Bulk insert future forecast rows."""
    if future_df.empty:
        return
    records = []
    for _, row in future_df.iterrows():
        store_id = row.get("store_id") if "store_id" in future_df.columns else None
        dept_id = row.get("dept_id") if "dept_id" in future_df.columns else None
        records.append(dict(
            run_id=run_id,
            store_id=str(store_id) if pd.notna(store_id) else None,
            dept_id=str(dept_id) if pd.notna(dept_id) else None,
            forecast_date=pd.to_datetime(row["date"]).date(),
            predicted_sales=float(row["predicted_sales"]),
            lower_bound=float(row.get("lower_bound", 0)) if pd.notna(row.get("lower_bound")) else None,
            upper_bound=float(row.get("upper_bound", 0)) if pd.notna(row.get("upper_bound")) else None,
        ))
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO forecasts (
                    run_id, store_id, dept_id, forecast_date,
                    predicted_sales, lower_bound, upper_bound
                ) VALUES (
                    :run_id, :store_id, :dept_id, :forecast_date,
                    :predicted_sales, :lower_bound, :upper_bound
                )
            """),
            records,
        )


def get_forecasts(run_id: str) -> pd.DataFrame:
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT store_id, dept_id, forecast_date AS date,
                       predicted_sales, lower_bound, upper_bound
                FROM forecasts WHERE run_id = :run_id
                ORDER BY forecast_date
            """),
            dict(run_id=run_id),
        )
        rows = result.mappings().all()
    df = pd.DataFrame(rows)
    # Postgres NUMERIC → Python Decimal; cast to float for downstream arithmetic
    for col in ["predicted_sales", "lower_bound", "upper_bound"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── Inventory ──────────────────────────────────────────────────────────────────

def save_inventory(run_id: str, inventory_df: pd.DataFrame) -> None:
    """Bulk insert inventory recommendation rows."""
    if inventory_df.empty:
        return
    records = []
    for _, row in inventory_df.iterrows():
        store_id = row.get("store_id") if "store_id" in inventory_df.columns else None
        dept_id = row.get("dept_id") if "dept_id" in inventory_df.columns else None
        records.append(dict(
            run_id=run_id,
            store_id=str(store_id) if pd.notna(store_id) else None,
            dept_id=str(dept_id) if pd.notna(dept_id) else None,
            avg_daily_demand=float(row.get("avg_daily_demand", 0)),
            demand_std=float(row.get("demand_std", 0)),
            forecast_demand=float(row.get("forecast_demand", 0)),
            safety_stock=float(row.get("safety_stock", 0)),
            reorder_point=float(row.get("reorder_point", 0)),
            recommended_inventory=float(row.get("recommended_inventory", 0)),
            risk_level=str(row.get("risk_level", "")),
            expected_stockout_probability=float(row.get("expected_stockout_probability", 0)),
        ))
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO inventory (
                    run_id, store_id, dept_id, avg_daily_demand, demand_std,
                    forecast_demand, safety_stock, reorder_point,
                    recommended_inventory, risk_level, expected_stockout_probability
                ) VALUES (
                    :run_id, :store_id, :dept_id, :avg_daily_demand, :demand_std,
                    :forecast_demand, :safety_stock, :reorder_point,
                    :recommended_inventory, :risk_level, :expected_stockout_probability
                )
            """),
            records,
        )


def get_inventory(run_id: str) -> pd.DataFrame:
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT store_id, dept_id, avg_daily_demand, demand_std,
                       forecast_demand, safety_stock, reorder_point,
                       recommended_inventory, risk_level,
                       expected_stockout_probability
                FROM inventory WHERE run_id = :run_id
            """),
            dict(run_id=run_id),
        )
        rows = result.mappings().all()
    df = pd.DataFrame(rows)
    # Postgres NUMERIC → Python Decimal; cast all numeric cols to float
    numeric_cols = [
        "avg_daily_demand", "demand_std", "forecast_demand", "safety_stock",
        "reorder_point", "recommended_inventory", "expected_stockout_probability",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── Actuals (Phase 5 — forecast accuracy tracking) ────────────────────────────

def save_actuals(actuals_df: pd.DataFrame) -> int:
    """
    Upsert real sales data. Returns number of rows inserted/updated.
    Expects columns: store_id, dept_id, date, sales.
    """
    if actuals_df.empty:
        return 0
    records = [
        dict(
            store_id=str(row.get("store_id", "")),
            dept_id=str(row.get("dept_id", "")),
            actual_date=pd.to_datetime(row["date"]).date(),
            actual_sales=float(row["sales"]),
        )
        for _, row in actuals_df.iterrows()
    ]
    with get_session() as session:
        session.execute(
            text("""
                INSERT INTO actuals (store_id, dept_id, actual_date, actual_sales)
                VALUES (:store_id, :dept_id, :actual_date, :actual_sales)
                ON CONFLICT (store_id, dept_id, actual_date)
                DO UPDATE SET actual_sales = EXCLUDED.actual_sales
            """),
            records,
        )
    return len(actuals_df)


def get_forecast_accuracy(run_id: str) -> pd.DataFrame:
    """
    Join a run's forecasts against any actuals uploaded for the same
    store/dept/date. Returns error metrics per row where actuals exist.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    f.store_id, f.dept_id, f.forecast_date,
                    f.predicted_sales, a.actual_sales,
                    (a.actual_sales - f.predicted_sales)            AS error,
                    ABS(a.actual_sales - f.predicted_sales)         AS abs_error,
                    CASE WHEN a.actual_sales != 0
                         THEN ABS(a.actual_sales - f.predicted_sales) / a.actual_sales * 100
                         ELSE NULL END                              AS pct_error
                FROM forecasts f
                JOIN actuals a
                  ON f.store_id = a.store_id
                 AND f.dept_id  = a.dept_id
                 AND f.forecast_date = a.actual_date
                WHERE f.run_id = :run_id
                ORDER BY f.forecast_date
            """),
            dict(run_id=run_id),
        )
        rows = result.mappings().all()
    return pd.DataFrame(rows)
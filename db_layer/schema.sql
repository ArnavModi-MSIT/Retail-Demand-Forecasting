-- ═══════════════════════════════════════════════════════════════════════════
-- Retail Demand Forecasting — Database Schema
-- ═══════════════════════════════════════════════════════════════════════════
-- Run this once against a fresh Postgres database to create all tables.
-- Safe to re-run: uses CREATE TABLE IF NOT EXISTS.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── runs ──────────────────────────────────────────────────────────────────
-- One row per pipeline execution. Everything else references this.
CREATE TABLE IF NOT EXISTS runs (
    run_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_name  TEXT,
    n_rows        INTEGER,
    n_stores      INTEGER,
    n_depts       INTEGER,
    model_choice  TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    service_level NUMERIC NOT NULL,
    lead_time     INTEGER NOT NULL,
    test_days     INTEGER NOT NULL
);

-- ── model_metrics ─────────────────────────────────────────────────────────
-- Test-period model quality. One row per model evaluated in a run
-- (selected model + naive baseline).
CREATE TABLE IF NOT EXISTS model_metrics (
    id         SERIAL PRIMARY KEY,
    run_id     UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    mae        NUMERIC,
    rmse       NUMERIC,
    mape       NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_run ON model_metrics(run_id);

-- ── forecasts ─────────────────────────────────────────────────────────────
-- Future demand predictions. The core output of generate_future_forecast().
CREATE TABLE IF NOT EXISTS forecasts (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    store_id        TEXT,
    dept_id         TEXT,
    forecast_date   DATE NOT NULL,
    predicted_sales NUMERIC NOT NULL,
    lower_bound     NUMERIC,
    upper_bound     NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_forecasts_run        ON forecasts(run_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_store_date ON forecasts(store_id, forecast_date);

-- ── inventory ─────────────────────────────────────────────────────────────
-- Inventory recommendations. Output of compute_inventory().
CREATE TABLE IF NOT EXISTS inventory (
    id                     SERIAL PRIMARY KEY,
    run_id                 UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    store_id               TEXT,
    dept_id                TEXT,
    avg_daily_demand       NUMERIC,
    demand_std             NUMERIC,
    forecast_demand        NUMERIC,
    safety_stock           NUMERIC,
    reorder_point          NUMERIC,
    recommended_inventory  NUMERIC,
    risk_level             TEXT,
    expected_stockout_probability NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_inventory_run ON inventory(run_id);

-- ── actuals ───────────────────────────────────────────────────────────────
-- Real sales uploaded later, used to score past forecasts (Phase 5).
-- Not tied to a specific run — actuals belong to the store/date, and can be
-- joined against ANY run's forecasts for the same store/date.
CREATE TABLE IF NOT EXISTS actuals (
    id           SERIAL PRIMARY KEY,
    uploaded_at  TIMESTAMP NOT NULL DEFAULT now(),
    store_id     TEXT,
    dept_id      TEXT,
    actual_date  DATE NOT NULL,
    actual_sales NUMERIC NOT NULL,
    UNIQUE (store_id, dept_id, actual_date)
);

CREATE INDEX IF NOT EXISTS idx_actuals_store_date ON actuals(store_id, actual_date);
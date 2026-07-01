# Retail Demand Forecasting & Inventory Optimization

An end-to-end retail demand forecasting and inventory optimization application — from raw CSV upload through feature engineering, model training, recursive multi-day forecasting, SHAP explainability, and inventory recommendations, backed by a Postgres persistence layer.

**Live demo:** https://retail-demand-forecasting-a.streamlit.app/

---

## Overview

Retail businesses need to answer two operational questions from historical sales data: how much will we sell in the coming weeks, and how much stock should we hold to avoid running out or overstocking. This project implements a full pipeline addressing both, starting from an arbitrary user-uploaded CSV (column names are auto-detected and mapped) through model training and evaluation, a recursive multi-day forecast, and inventory recommendations with per-SKU risk classification.

The application is designed as a public Streamlit demo, but architected with production concerns in mind: a normalized Postgres schema, session-scoped multi-tenant data isolation, best-effort persistence that never blocks the core pipeline, and a Dockerized deployment using a multi-stage build and a non-root user.

---

## Architecture

```
CSV Upload
    │
    ▼
Column Detection & Mapping  (dataset.py — handles arbitrary column names/aliases)
    │
    ▼
Data Validation             (dataset.py — blocking errors vs. warnings)
    │
    ▼
Demo Capacity Guard         (capacity.py — time-series-safe row sampling for shared demo)
    │
    ▼
Feature Engineering         (features.py, features_lag.py, features_calendar.py, features_retail.py)
    │
    ▼
Train/Test Split            (splitter.py — time-based split, no leakage)
    │
    ▼
Model Training               (models.py, train.py — XGBoost / LightGBM / Random Forest)
    │
    ▼
Evaluation & SHAP            (evaluate.py — MAE / RMSE / MAPE + explainability)
    │
    ▼
Recursive Future Forecast    (future_forecast.py — O(1) per-step buffer computation)
    │
    ▼
Inventory Optimization       (inventory.py — safety stock, reorder point, CV-based risk)
    │
    ▼
Executive Summary & Export   (reporting.py — KPI rollup, Excel export)
    │
    ▼
Postgres Persistence         (db_layer/ — best-effort, session-scoped, never blocks pipeline)
```

`pipeline.py` is a thin orchestrator that re-exports every module's public functions so the app can import from a single place.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML / Data | XGBoost, LightGBM, Random Forest, SHAP, pandas, NumPy, scikit-learn |
| Application | Streamlit, Plotly |
| Persistence | SQLAlchemy, PostgreSQL (Neon serverless in production) |
| Infrastructure | Docker (multi-stage build), Docker Compose |
| Testing | pytest |

---

## Key Features

**Flexible data ingestion**
Upload any retail sales CSV — the app detects `date`/`sales` and a wide range of optional columns (`store_id`, `dept_id`, `sell_price`, `promo`, `competition_distance`, calendar/holiday flags, etc.) via alias matching, and validates the data before it reaches the model.

**Leakage-safe feature engineering**
Lag features (1/7/14/28 days), rolling mean/std (7/14/30 days), cyclical calendar encoding, price/promo/competition covariates — all derived strictly from `t-1` or earlier.

**Multi-model training**
Choice of XGBoost (with `RandomizedSearchCV` + `TimeSeriesSplit` hyperparameter search), LightGBM (fast defaults, used for the live demo), or Random Forest — evaluated against a naive 7-day baseline.

**Recursive multi-day forecasting, optimized for speed**
Forecasting 28+ days ahead recursively (each day's prediction feeds the next day's lag features) was originally an O(n²) operation, re-running full feature engineering on a growing history at every step. It has been rewritten to compute lag/rolling features directly from a NumPy buffer in O(1) per step, reducing forecast runtime from approximately 40 minutes to approximately 12 seconds at scale.

**SHAP explainability**
Feature importance via SHAP `TreeExplainer`, with a graceful fallback to native `feature_importances_` if SHAP fails on a given model.

**Inventory optimization with per-SKU risk classification**
Safety stock and reorder points computed from lead-time demand and its standard deviation (z-score scaled by service level). Risk level is classified by **coefficient of variation** (std/mean) on open-trading days only, rather than a constant service-level-derived probability — this means SKUs get genuinely differentiated risk levels instead of all showing identical stockout probability.

**Postgres persistence, safely scoped for a public demo**
- Every pipeline run is saved (forecasts, model metrics, inventory recommendations) under a random per-browser-session ID
- "Load Previous Run" only ever shows runs from the same session — no cross-user data leakage on the shared public deployment
- All DB writes are best-effort: if Postgres is unreachable, the pipeline still completes and the user sees their results; only persistence is skipped
- Actuals can later be uploaded and joined against past forecasts (`get_forecast_accuracy`) using window functions (`AVG() OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`, `RANK()`) to compute rolling forecast-error trends per store/dept, computed in Postgres rather than pandas

**Demo capacity guard**
On the memory-constrained free-tier deployment, oversized uploads are sampled down via `MAX_DEMO_ROWS` — never by randomly dropping rows (which would corrupt lag/rolling features), but by keeping whole stores' history intact or the most recent contiguous date window.

**Excel + CSV export**
A formatted, color-coded multi-sheet Excel workbook (Executive Summary, Forecast Results, Inventory Recommendations, Model Performance, Feature Importance, Future Forecast) plus individual CSV downloads.

---

## Project Structure

```
.
├── app.py                      # Streamlit entrypoint
├── ui/
│   ├── sidebar.py               # Configuration controls
│   ├── upload.py                 # CSV upload stage
│   ├── running.py                 # Pipeline execution + progress
│   ├── results.py                  # Results routing (retrain vs. reforecast detection)
│   └── tabs/
│       ├── forecast.py              # Forecast chart + actual-vs-predicted
│       ├── inventory.py              # Risk KPI cards + color-coded table
│       ├── explainability.py          # SHAP feature importance
│       ├── model.py                    # Model performance metrics
│       └── download.py                  # Excel / CSV export
├── src/
│   ├── dataset.py                # Column detection, mapping, validation
│   ├── capacity.py                # Demo row-cap sampling
│   ├── features.py                 # Feature engineering orchestrator
│   ├── features_lag.py              # Lag / rolling features
│   ├── features_calendar.py          # Calendar / cyclical / event features
│   ├── features_retail.py             # Price / promo / competition features
│   ├── preprocessing.py                # Label encoding, feature column selection
│   ├── splitter.py                      # Time-based train/test split
│   ├── models.py                         # Model fitting, hyperparameter search
│   ├── train.py                           # Training orchestrator
│   ├── evaluate.py                         # Metrics + SHAP
│   ├── future_forecast.py                   # Recursive O(1) multi-day forecast
│   ├── inventory.py                           # Safety stock, reorder point, risk classification
│   ├── reporting.py                             # Executive summary, Excel export
│   ├── session_helper.py                         # Per-browser-session ID for demo isolation
│   └── pipeline.py                                # Thin re-export orchestrator
├── db_layer/
│   ├── connection.py               # SQLAlchemy engine, env/secrets-based config
│   ├── repository.py                # All SQL — reads/writes, session-scoped
│   └── schema.sql                     # Postgres schema (runs, forecasts, inventory, actuals)
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   ├── test_inventory.py
│   └── test_future_forecast.py
├── Dockerfile                       # Multi-stage build, non-root user
├── docker-compose.yml                # App + Postgres for local full-stack dev
├── requirements.txt
├── pytest.ini
└── .streamlit/config.toml            # Theme config
```

---

## Getting Started

### Quick local run (no Docker, no Postgres required)

```bash
pip install -r requirements.txt --break-system-packages
streamlit run app.py
```

The application runs fully without a database connection. Persistence is best-effort and is silently skipped if Postgres is unreachable.

### Full stack with Docker Compose

```bash
docker-compose up --build
```

Requires a `.env` file in the repo root with at least `DB_PASSWORD` set (used by `docker-compose.yml`). Visit `http://localhost:8501`.

### Configuration

Database credentials are read from environment variables (local/Docker) or `st.secrets` (Streamlit Community Cloud) — see `db_layer/connection.py`. Required keys: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, and optionally `DB_SSLMODE` (set to `require` for managed Postgres like Neon or RDS).

---

## Testing

```bash
pip install pytest --break-system-packages
pytest tests/ -v
```

29 tests covering:
- Column detection/aliasing and data validation (`test_dataset.py`)
- Forecast evaluation metrics, including zero-actual edge cases (`test_evaluate.py`)
- Inventory risk classification, safety stock scaling, and reorder point math (`test_inventory.py`)
- The O(1) recursive forecast buffer computation, verified against a from-scratch reference implementation (`test_future_forecast.py`)

Writing tests for the recursive forecast buffer surfaced two defects that were subsequently fixed in `future_forecast.py`: rolling mean and standard deviation were computed one day stale relative to the training-time feature definitions, and standard deviation used population variance (`ddof=0`) rather than sample variance (`ddof=1`, matching the training-time convention). Both issues affected live forecast quality and downstream inventory risk scoring prior to the fix.

---

## Known Limitations (by design, for a public demo)

- Session isolation via a random per-browser-session ID is a privacy boundary, not authentication — acceptable for a public demo, not a substitute for real multi-tenant auth.
- The `MAX_DEMO_ROWS` capacity guard trades off dataset completeness for staying within free-tier memory limits; local/full-scale runs are uncapped.
- No scheduling/orchestration layer (e.g. Airflow) — the pipeline runs synchronously on user interaction, not as a recurring batch job, so an orchestrator isn't the right fit for this deployment model.

---

## Author

**Arnav Modi** — B.Tech Information Technology, Maharaja Surajmal Institute of Technology

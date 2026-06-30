import pandas as pd
import numpy as np

from src.features_calendar import add_calendar_features
from src.features_retail import add_retail_features


# ── Confidence intervals ──────────────────────────────────────────────────────

def calculate_forecast_confidence(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Compute prediction interval margin using test set residuals.

    Returns:
        (margin, std_residual)
    """
    X_num = X_test.copy()
    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(0)

    residuals = y_test.values - model.predict(X_num)
    std_residual = np.std(residuals)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    return z * std_residual, std_residual


# ── Direct lag/rolling computation (replaces engineer_features re-call) ───────

def _compute_lag_rolling(sales_buf: np.ndarray) -> dict:
    """
    Compute lag and rolling features directly from a sales buffer.

    Parameters
    ----------
    sales_buf
        1-D array of historical sales in chronological order.
        Must have at least 30 elements for all features to be valid.

    Returns
    -------
    Dict of feature_name → scalar value.
    """
    n = len(sales_buf)
    features = {}

    for lag in [1, 7, 14, 28]:
        features[f"lag_{lag}"] = float(sales_buf[-lag]) if n >= lag else 0.0

    for window in [7, 14, 30]:
        if n >= window + 1:
            features[f"rolling_mean_{window}"] = float(sales_buf[-(window + 1):-1].mean())
        else:
            features[f"rolling_mean_{window}"] = 0.0

    for window in [7, 30]:
        if n >= window + 1:
            features[f"rolling_std_{window}"] = float(sales_buf[-(window + 1):-1].std())
        else:
            features[f"rolling_std_{window}"] = 0.0

    return features


def _compute_customer_features(cust_buf: np.ndarray, lag_1_sales: float) -> dict:
    """Compute customer lag features from a customer buffer."""
    n = len(cust_buf)
    features = {}
    cl1 = float(cust_buf[-1]) if n >= 1 else 0.0
    features["customers_lag_1"] = cl1
    features["customers_lag_7"] = float(cust_buf[-7]) if n >= 7 else 0.0
    features["customer_to_sales_ratio"] = cl1 / (lag_1_sales + 1e-6)
    return features


# ── Encoder helper ────────────────────────────────────────────────────────────

def _encode_row(row: dict, encoders: dict) -> dict:
    """Apply label encoders to categorical fields in a feature dict."""
    if not encoders:
        return row
    for col, encoder in encoders.items():
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val) if not isinstance(val, str) else False:
            row[col] = 0
            continue
        sval = str(val)
        row[col] = int(encoder.transform([sval])[0]) if sval in encoder.classes_ else 0
    return row


# ── Future forecast ───────────────────────────────────────────────────────────

def generate_future_forecast(
    model,
    df_featured: pd.DataFrame,
    feature_cols: list,
    horizon_days: int = 28,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    confidence_level: float = 0.95,
    encoders: dict = None,
) -> pd.DataFrame:
    """
    Forecast future demand recursively, updating lag and rolling features
    after each predicted day.

    Key optimisations vs original:
    - Lag/rolling features computed directly from a numpy sales buffer —
      no full engineer_features() re-call per step (was O(n²), now O(1)).
    - History kept as a growing list; single DataFrame built once per group
      at the end — no pd.concat inside the loop.

    Returns DataFrame with:
        date, (store_id), (dept_id),
        predicted_sales, lower_bound, upper_bound
    """
    group_cols = [c for c in ["store_id", "dept_id"] if c in df_featured.columns]
    empty = pd.DataFrame(columns=group_cols + ["date", "predicted_sales", "lower_bound", "upper_bound"])

    if df_featured.empty or "date" not in df_featured.columns:
        return empty

    last_date = pd.to_datetime(df_featured["date"]).max()
    if pd.isna(last_date):
        return empty

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    margin = 0.0
    if X_test is not None and y_test is not None and len(X_test) > 0:
        margin, _ = calculate_forecast_confidence(model, X_test, y_test, confidence_level)

    # ── Identify static columns (non-sales, non-date, non-lag) ───────────────
    # These are carried forward unchanged from the last historical row.
    dynamic_cols = (
        {f"lag_{l}" for l in [1, 7, 14, 28]}
        | {f"rolling_mean_{w}" for w in [7, 14, 30]}
        | {f"rolling_std_{w}" for w in [7, 30]}
        | {"customers_lag_1", "customers_lag_7", "customer_to_sales_ratio"}
        | {"month", "quarter", "wday", "weekend", "year",
           "day_of_week", "is_weekend",
           "month_sin", "month_cos", "wday_sin", "wday_cos"}
    )

    if group_cols:
        groups = [
            grp.sort_values("date").reset_index(drop=True)
            for _, grp in df_featured.groupby(group_cols, dropna=False)
        ]
    else:
        groups = [df_featured.sort_values("date").reset_index(drop=True)]

    records = []

    for history_df in groups:
        history_df = history_df.copy()
        history_df["date"] = pd.to_datetime(history_df["date"])

        # Sales buffer — grows by one each step
        sales_buf = history_df["sales"].to_numpy(dtype=float, na_value=0.0).tolist()

        # Customer buffer (optional)
        has_customers = "customers" in history_df.columns
        cust_buf = (
            history_df["customers"].fillna(0).to_numpy(dtype=float).tolist()
            if has_customers else []
        )

        # Static feature values from last historical row
        last_row = history_df.iloc[-1].to_dict()
        static = {
            k: v for k, v in last_row.items()
            if k not in dynamic_cols and k not in {"date", "sales"}
        }

        group_id = {col: last_row[col] for col in group_cols if col in last_row}

        for current_date in future_dates:
            # 1. Compute lag/rolling directly — O(1), no DataFrame ops
            row = dict(static)
            row.update(_compute_lag_rolling(np.array(sales_buf)))

            # 2. Customer features
            if has_customers and cust_buf:
                row.update(_compute_customer_features(
                    np.array(cust_buf), row.get("lag_1", 0.0)
                ))

            # 3. Calendar features directly (no DataFrame call)
            dow = current_date.dayofweek + 1
            row["day_of_week"] = dow
            row["is_weekend"]  = int(dow in (6, 7))
            row["month"]       = current_date.month
            row["quarter"]     = current_date.quarter
            row["wday"]        = dow
            row["weekend"]     = int(dow in (6, 7))
            row["year"]        = current_date.year
            row["month_sin"]   = float(np.sin(2 * np.pi * current_date.month / 12))
            row["month_cos"]   = float(np.cos(2 * np.pi * current_date.month / 12))
            row["wday_sin"]    = float(np.sin(2 * np.pi * dow / 7))
            row["wday_cos"]    = float(np.cos(2 * np.pi * dow / 7))

            # 4. Encode categoricals
            row = _encode_row(row, encoders)

            # 5. Build model input and predict
            X_row = np.array(
                [float(row.get(col, 0) if not isinstance(row.get(col, 0), str) else 0)
                 for col in feature_cols],
                dtype=float,
            ).reshape(1, -1)
            pred = float(np.clip(model.predict(X_row)[0], 0, None))

            # 6. Append prediction to sales buffer (no DataFrame concat)
            sales_buf.append(pred)
            if has_customers:
                cust_buf.append(cust_buf[-1] if cust_buf else 0.0)

            # 7. Record result
            rec = dict(group_id)
            rec["date"]            = current_date
            rec["predicted_sales"] = pred
            rec["lower_bound"]     = max(0.0, pred - margin)
            rec["upper_bound"]     = pred + margin
            records.append(rec)

    future_df = pd.DataFrame(records)
    if future_df.empty:
        return empty

    sort_cols = [*group_cols, "date"]
    future_df = future_df.sort_values(sort_cols).reset_index(drop=True)
    return future_df[group_cols + ["date", "predicted_sales", "lower_bound", "upper_bound"]]
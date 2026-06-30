import numpy as np
import pandas as pd


def compute_inventory(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    service_level: float = 0.95,
    lead_time_days: int = 7,
) -> pd.DataFrame:

    z_map = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(service_level, 1.645)

    group_cols = [
        c for c in ["store_id", "dept_id"]
        if c in history_df.columns and c in future_df.columns
    ]

    if group_cols:
        history = history_df.copy()
        future = future_df.copy()
        for col in group_cols:
            history[col] = history[col].astype(str)
            future[col] = future[col].astype(str)

        # CV computed on open days only to avoid zero-sales closure days
        # inflating std and making every SKU appear Critical.
        open_history = history[history["sales"] > 0] if "sales" in history.columns else history

        hist = (
            open_history.groupby(group_cols)
            .agg(
                avg_daily_demand=("sales", "mean"),
                demand_std=("sales", "std"),
            )
            .reset_index()
        )

        future_agg = (
            future.groupby(group_cols)
            .agg(forecast_demand=("predicted_sales", "sum"))
            .reset_index()
        )

        for col in group_cols:
            if col in hist.columns:
                hist[col] = hist[col].astype(str)
            if col in future_agg.columns:
                future_agg[col] = future_agg[col].astype(str)

        inv = hist.merge(future_agg, on=group_cols, how="inner")
    else:
        open_history = history_df[history_df["sales"] > 0] if "sales" in history_df.columns else history_df
        mean_val = open_history["sales"].mean()
        std_val = open_history["sales"].std()
        inv = pd.DataFrame({
            "avg_daily_demand": [mean_val if pd.notna(mean_val) else 0.0],
            "demand_std":       [std_val if pd.notna(std_val) else 0.0],
            "forecast_demand":  [future_df["predicted_sales"].sum()],
        })

    inv["demand_std"] = inv["demand_std"].fillna(0)

    inv["expected_lead_time_demand"] = inv["avg_daily_demand"] * lead_time_days
    inv["lead_time_demand_std"]      = inv["demand_std"] * np.sqrt(lead_time_days)
    inv["safety_stock"]              = z * inv["lead_time_demand_std"]
    inv["reorder_point"]             = inv["expected_lead_time_demand"] + inv["safety_stock"]
    inv["recommended_inventory"]     = inv["forecast_demand"] + inv["safety_stock"]

    # ── Risk classification via Coefficient of Variation (open days only) ────
    # CV = std / mean on trading days. Reflects true demand volatility per SKU.
    # Thresholds: Low <15%, Medium 15-25%, High 25-40%, Critical >40%
    inv["cv"] = np.where(
        inv["avg_daily_demand"] > 0,
        inv["demand_std"] / inv["avg_daily_demand"],
        0,
    )
    inv["risk_level"] = np.select(
        [inv["cv"] > 0.40, inv["cv"] > 0.25, inv["cv"] > 0.15],
        ["Critical", "High Risk", "Medium Risk"],
        default="Low Risk",
    )

    # Per-SKU stockout probability derived from CV rather than a constant
    # service-level complement (which made every row identical).
    inv["expected_stockout_probability"] = round(
        np.clip(inv["cv"] / (inv["cv"] + z), 0.0, 1.0), 6,
    )

    inv["expected_overstock_quantity"] = np.maximum(
        inv["recommended_inventory"] - inv["forecast_demand"], 0
    )
    inv["inventory_days_of_supply"] = np.where(
        inv["forecast_demand"] > 0,
        inv["recommended_inventory"] / (inv["forecast_demand"] / max(lead_time_days, 1)),
        np.nan,
    )
    inv["safety_stock_percentage"] = np.where(
        inv["expected_lead_time_demand"] > 0,
        inv["safety_stock"] / inv["expected_lead_time_demand"],
        np.nan,
    )
    inv["fill_rate_estimate"] = service_level
    inv = inv.drop(columns=["cv"])

    numeric_cols = [
        "forecast_demand", "avg_daily_demand", "demand_std",
        "expected_lead_time_demand", "lead_time_demand_std",
        "safety_stock", "reorder_point", "recommended_inventory",
        "expected_stockout_probability", "expected_overstock_quantity",
        "inventory_days_of_supply", "safety_stock_percentage", "fill_rate_estimate",
    ]
    inv[numeric_cols] = inv[numeric_cols].round(2)

    return inv.sort_values("recommended_inventory", ascending=False).reset_index(drop=True)
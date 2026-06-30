import pandas as pd


def add_retail_features(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add price, promo, customer, competition, and store attribute features."""

    # ── Price ─────────────────────────────────────────────────────────────────
    if "sell_price" in df.columns and df["sell_price"].notna().any():
        if group_cols:
            df["price_lag_1"] = df.groupby(group_cols)["sell_price"].shift(1)
        else:
            df["price_lag_1"] = df["sell_price"].shift(1)
        df["price_change"]     = df["sell_price"] - df["price_lag_1"]
        df["price_change_pct"] = df["price_change"] / (df["price_lag_1"] + 1e-6)

    # ── Customers ─────────────────────────────────────────────────────────────
    if "customers" in df.columns:
        if group_cols:
            df["customers_lag_1"] = df.groupby(group_cols)["customers"].shift(1)
            df["customers_lag_7"] = df.groupby(group_cols)["customers"].shift(7)
        else:
            df["customers_lag_1"] = df["customers"].shift(1)
            df["customers_lag_7"] = df["customers"].shift(7)
        # Leakage-safe: uses lag_1 (already historical)
        df["customer_to_sales_ratio"] = df["customers_lag_1"] / (df["lag_1"] + 1e-6)

    # ── Promotion flags ───────────────────────────────────────────────────────
    if "open" in df.columns:
        df["is_open"] = df["open"].fillna(0).astype(int)
    if "promo" in df.columns:
        df["is_promo"] = df["promo"].fillna(0).astype(int)
    if "state_holiday" in df.columns:
        df["is_state_holiday"] = (
            df["state_holiday"].fillna("0").astype(str).ne("0")
        ).astype(int)
    if "school_holiday" in df.columns:
        df["is_school_holiday"] = df["school_holiday"].fillna(0).astype(int)
    if "promo2" in df.columns:
        df["is_promo2"] = df["promo2"].fillna(0).astype(int)

    # ── Competition ───────────────────────────────────────────────────────────
    if "competition_distance" in df.columns:
        df["competition_distance"] = pd.to_numeric(
            df["competition_distance"], errors="coerce"
        )
        median_dist = df["competition_distance"].median()
        df["competition_distance"] = df["competition_distance"].fillna(
            median_dist if not pd.isna(median_dist) else 0
        )

    if "competition_open_month" in df.columns and "competition_open_year" in df.columns:
        df["competition_open_month"] = pd.to_numeric(
            df["competition_open_month"], errors="coerce"
        ).fillna(1)
        df["competition_open_year"] = pd.to_numeric(
            df["competition_open_year"], errors="coerce"
        ).fillna(df["date"].dt.year)
        df["competition_age_months"] = (
            (df["date"].dt.year  - df["competition_open_year"]).astype("Int64") * 12
            + (df["date"].dt.month - df["competition_open_month"]).astype("Int64")
        ).fillna(0).clip(lower=0)

    # ── Store attributes ──────────────────────────────────────────────────────
    if "store_type" in df.columns:
        df["store_type"] = df["store_type"].fillna("Unknown").astype(str)
    if "assortment" in df.columns:
        df["assortment"] = df["assortment"].fillna("Unknown").astype(str)

    return df

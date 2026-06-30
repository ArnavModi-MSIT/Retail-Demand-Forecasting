import pandas as pd

from src.features_lag import add_lag_features
from src.features_calendar import add_calendar_features
from src.features_retail import add_retail_features


def _get_group_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in ["store_id", "dept_id"]
        if c in df.columns and df[c].notna().any()
    ]


def _fillna_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaNs so the model never sees missing values."""
    for col in df.columns:
        if col in {"date", "sales"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
        lag/rolling → retail covariates → calendar/cyclical

    Works with or without optional columns.
    Leakage-safe: all features derived from t-1 or earlier.
    """
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    group_cols = _get_group_cols(df)
    if group_cols:
        df = df.sort_values(group_cols + ["date"])
    else:
        df = df.sort_values("date")

    df = add_lag_features(df, group_cols)
    df = add_retail_features(df, group_cols)
    df = add_calendar_features(df)

    df = df.dropna(subset=["sales"]).reset_index(drop=True)
    df = _fillna_features(df)
    return df


def engineer_features_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy feature set (pre retail-covariate expansion).
    Kept for compare_feature_sets() compatibility.
    Uses only lag, rolling, price, and calendar features.
    """
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    group_cols = _get_group_cols(df)
    if group_cols:
        df = df.sort_values(group_cols + ["date"])
    else:
        df = df.sort_values("date")

    df = add_lag_features(df, group_cols)
    df = add_retail_features(df, group_cols)   # price only; other cols absent in legacy data
    df = add_calendar_features(df)

    df = df.dropna().reset_index(drop=True)
    return df
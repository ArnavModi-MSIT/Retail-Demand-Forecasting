import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical, and event features derived from the date column."""

    # ── Day of week ──────────────────────────────────────────────────────────
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")
        df["day_of_week"] = df["day_of_week"].fillna(df["date"].dt.dayofweek + 1)
    else:
        df["day_of_week"] = df["date"].dt.dayofweek + 1

    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

    # ── Calendar fields ───────────────────────────────────────────────────────
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["wday"]    = df["date"].dt.dayofweek + 1
    df["weekend"] = df["wday"].isin([6, 7]).astype(int)
    df["year"]    = df["date"].dt.year

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wday_sin"]  = np.sin(2 * np.pi * df["wday"] / 7)
    df["wday_cos"]  = np.cos(2 * np.pi * df["wday"] / 7)

    # ── Event features ────────────────────────────────────────────────────────
    if "event_type_1" in df.columns and df["event_type_1"].notna().any():
        df["is_event"] = (
            df["event_type_1"].fillna("No_Event").astype(str) != "No_Event"
        ).astype(int)
    if "event_name_1" in df.columns:
        df["has_event_name"] = df["event_name_1"].notna().astype(int)

    return df

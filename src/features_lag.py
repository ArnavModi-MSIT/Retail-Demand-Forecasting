import pandas as pd


LAG_WINDOWS     = [1, 7, 14, 28]
ROLLING_MEANS   = [7, 14, 30]
ROLLING_STDS    = [7, 30]


def add_lag_features(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add lag and rolling mean/std features, grouped by store/dept if present."""
    grp = df.groupby(group_cols)["sales"] if group_cols else df["sales"]

    for lag in LAG_WINDOWS:
        df[f"lag_{lag}"] = grp.shift(lag) if group_cols else df["sales"].shift(lag)

    for window in ROLLING_MEANS:
        if group_cols:
            df[f"rolling_mean_{window}"] = grp.transform(
                lambda x, w=window: x.shift(1).rolling(w).mean()
            )
        else:
            df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()

    for window in ROLLING_STDS:
        if group_cols:
            df[f"rolling_std_{window}"] = grp.transform(
                lambda x, w=window: x.shift(1).rolling(w).std()
            )
        else:
            df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    return df

import pandas as pd
import numpy as np

print("Loading master dataset...")

df = pd.read_csv("data/master_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store_id", "dept_id", "date"]
)

group_cols = ["store_id", "dept_id"]

print("Creating lag features...")

for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = (
        df.groupby(group_cols)["sales"]
        .shift(lag)
    )

print("Creating rolling mean features...")

for window in [7, 14, 30]:
    df[f"rolling_mean_{window}"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    )

print("Creating rolling std features...")

for window in [7, 30]:
    df[f"rolling_std_{window}"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    )

print("Creating price features...")

df["price_lag_1"] = (
    df.groupby(group_cols)["sell_price"]
    .shift(1)
)

df["price_change"] = (
    df["sell_price"] - df["price_lag_1"]
)

df["price_change_pct"] = (
    df["price_change"]
    / (df["price_lag_1"] + 1e-6)
)

print("Creating calendar features...")

df["quarter"] = df["date"].dt.quarter

df["weekend"] = (
    df["wday"].isin([1, 7])
).astype(int)

print("Creating event features...")

df["is_event"] = (
    df["event_type_1"] != "No_Event"
).astype(int)

print("Dropping rows with null lag values...")


print("\nFinal Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["wday_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
df["wday_cos"] = np.cos(2 * np.pi * df["wday"] / 7)

df = df.dropna()

df.to_csv(
    "data/features_dataset.csv",
    index=False
)

print("\nSaved: features_dataset.csv")

print("\nSample:")
print(df.head())
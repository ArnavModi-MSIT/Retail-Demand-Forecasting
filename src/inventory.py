import pandas as pd

print("Loading data...")

forecast_df = pd.read_csv(
    "outputs/future_forecasts.csv"
)

features_df = pd.read_csv(
    "features_dataset.csv"
)

latest_stats = (
    features_df
    .sort_values("date")
    .groupby(
        ["store_id", "dept_id"]
    )
    .tail(1)
)

inventory_df = forecast_df.merge(
    latest_stats[
        [
            "store_id",
            "dept_id",
            "rolling_std_7"
        ]
    ],
    on=[
        "store_id",
        "dept_id"
    ],
    how="left"
)

inventory_df["rolling_std_7"] = (
    inventory_df["rolling_std_7"]
    .fillna(0)
)

inventory_df["safety_stock"] = (
    inventory_df["rolling_std_7"]
    * 1.65
)

inventory_df["recommended_inventory"] = (
    inventory_df["forecast_sales"]
    + inventory_df["safety_stock"]
)

inventory_df["inventory_risk"] = "Low"

inventory_df.loc[
    inventory_df["safety_stock"] > 100,
    "inventory_risk"
] = "High"

inventory_df.loc[
    (
        inventory_df["safety_stock"] > 50
    )
    &
    (
        inventory_df["safety_stock"] <= 100
    ),
    "inventory_risk"
] = "Medium"

inventory_df = inventory_df[
    [
        "date",
        "store_id",
        "dept_id",
        "forecast_sales",
        "safety_stock",
        "recommended_inventory",
        "inventory_risk"
    ]
]

inventory_df.to_csv(
    "outputs/inventory_recommendations.csv",
    index=False
)

print("\nShape:")
print(inventory_df.shape)

print("\nRisk Distribution:")
print(
    inventory_df["inventory_risk"]
    .value_counts()
)

print("\nTop Inventory Requirements:")
print(
    inventory_df
    .sort_values(
        "recommended_inventory",
        ascending=False
    )
    .head(20)
)

print("\nSaved:")
print("inventory_recommendations.csv")
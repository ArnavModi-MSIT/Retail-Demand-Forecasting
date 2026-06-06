import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

print("Loading datasets...")

sales = pd.read_csv("dataset/sales_train_evaluation.csv")
calendar = pd.read_csv("dataset/calendar.csv")
prices = pd.read_csv("dataset/sell_prices.csv")

print("\nDatasets Loaded Successfully")

# =========================
# BASIC INFO
# =========================

print("\n" + "=" * 60)
print("SALES DATASET")
print("=" * 60)

print("Shape:", sales.shape)

print("\nCategories")
print(sales["cat_id"].value_counts())

print("\nDepartments")
print(sales["dept_id"].value_counts())

print("\nStores")
print(sales["store_id"].value_counts())

print("\nStates")
print(sales["state_id"].value_counts())

# =========================
# SALES SUMMARY
# =========================

day_cols = [col for col in sales.columns if col.startswith("d_")]

sales["total_sales"] = sales[day_cols].sum(axis=1)

print("\n" + "=" * 60)
print("TOTAL SALES BY CATEGORY")
print("=" * 60)

print(
    sales.groupby("cat_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOTAL SALES BY DEPARTMENT")
print("=" * 60)

print(
    sales.groupby("dept_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOTAL SALES BY STORE")
print("=" * 60)

print(
    sales.groupby("store_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOTAL SALES BY STATE")
print("=" * 60)

print(
    sales.groupby("state_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
)

# =========================
# TOP PRODUCTS
# =========================

print("\n" + "=" * 60)
print("TOP 20 PRODUCTS")
print("=" * 60)

top_products = (
    sales[["item_id", "store_id", "cat_id", "total_sales"]]
    .sort_values("total_sales", ascending=False)
    .head(20)
)

print(top_products)

# =========================
# CALENDAR ANALYSIS
# =========================

print("\n" + "=" * 60)
print("CALENDAR DATA")
print("=" * 60)

print("Shape:", calendar.shape)

print("\nEvent Type Distribution")
print(calendar["event_type_1"].value_counts(dropna=False))

print("\nTop Events")
print(calendar["event_name_1"].value_counts().head(20))

print("\nSNAP DAYS")

print(
    {
        "CA": calendar["snap_CA"].sum(),
        "TX": calendar["snap_TX"].sum(),
        "WI": calendar["snap_WI"].sum(),
    }
)

# =========================
# PRICE ANALYSIS
# =========================

print("\n" + "=" * 60)
print("PRICE DATA")
print("=" * 60)

print("Shape:", prices.shape)

print("\nPrice Statistics")
print(prices["sell_price"].describe())

print("\nHighest Average Price Products")

avg_prices = (
    prices.groupby("item_id")["sell_price"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

print(avg_prices)

print("\nLowest Average Price Products")

print(
    prices.groupby("item_id")["sell_price"]
    .mean()
    .sort_values()
    .head(20)
)

# =========================
# PRICE VARIABILITY
# =========================

price_variation = (
    prices.groupby("item_id")["sell_price"]
    .agg(["mean", "std"])
    .sort_values("std", ascending=False)
)

print("\n" + "=" * 60)
print("MOST VOLATILE PRICES")
print("=" * 60)

print(price_variation.head(20))

# =========================
# BUSINESS INSIGHTS
# =========================

print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

food_sales = sales[sales["cat_id"] == "FOODS"]["total_sales"].sum()
total_sales = sales["total_sales"].sum()

print(
    f"Food Contribution: {(food_sales/total_sales)*100:.2f}% of total demand"
)

best_store = (
    sales.groupby("store_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

print(f"Top Performing Store: {best_store}")

best_dept = (
    sales.groupby("dept_id")["total_sales"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

print(f"Top Department: {best_dept}")

print("\nEDA Complete.")
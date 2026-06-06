import pandas as pd

print("Loading datasets...")

sales = pd.read_csv("dataset/sales_train_evaluation.csv")
calendar = pd.read_csv("dataset/calendar.csv")
prices = pd.read_csv("dataset/sell_prices.csv")

print("Datasets loaded.")

day_cols = [col for col in sales.columns if col.startswith("d_")]

print(f"Sales days: {len(day_cols)}")

sales_long = sales.melt(
id_vars=[
"item_id",
"dept_id",
"cat_id",
"store_id",
"state_id"
],
value_vars=day_cols,
var_name="d",
value_name="sales"
)

print("Long sales shape:", sales_long.shape)

sales_long = sales_long.merge(
calendar,
on="d",
how="left"
)

print("Shape after calendar merge:", sales_long.shape)

sales_long = sales_long.merge(
prices,
on=["store_id", "item_id", "wm_yr_wk"],
how="left"
)

print("Shape after price merge:", sales_long.shape)

sales_long["date"] = pd.to_datetime(sales_long["date"])

sales_long["sell_price"] = sales_long["sell_price"].fillna(
sales_long["sell_price"].median()
)

sales_long["event_name_1"] = sales_long["event_name_1"].fillna(
"No_Event"
)

sales_long["event_type_1"] = sales_long["event_type_1"].fillna(
"No_Event"
)

master_df = (
sales_long
.groupby(
[
"date",
"store_id",
"dept_id",
"state_id",
"month",
"year",
"wday",
"event_name_1",
"event_type_1",
"snap_CA",
"snap_TX",
"snap_WI"
],
as_index=False
)
.agg(
sales=("sales", "sum"),
sell_price=("sell_price", "mean")
)
)

master_df = master_df.sort_values(
["store_id", "dept_id", "date"]
)

series_count = (
master_df[["store_id", "dept_id"]]
.drop_duplicates()
.shape[0]
)

days_count = master_df["date"].nunique()

print("\nValidation")
print("-" * 40)
print("Series:", series_count)
print("Days:", days_count)
print("Expected Rows:", series_count * days_count)
print("Actual Rows:", len(master_df))

master_df.to_csv(
"data/master_dataset.csv",
index=False
)

print("\nMaster Dataset Shape:")
print(master_df.shape)

print("\nSaved:")
print("master_dataset.csv")

print("\nSample:")
print(master_df.head())

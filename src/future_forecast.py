import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

print("Loading dataset...")

df = pd.read_csv("data/features_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

categorical_cols = [
    "store_id",
    "dept_id",
    "state_id",
    "event_name_1",
    "event_type_1"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = [
    col for col in df.columns
    if col not in ["date", "sales"]
]

X = df[feature_cols]
y = df["sales"]

print("Training final XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Creating future forecasts...")

future_rows = []

latest_date = df["date"].max()

group_cols = ["store_id", "dept_id"]

for (store, dept), group in df.groupby(group_cols):

    group = group.sort_values("date")

    latest = group.iloc[-1].copy()

    current_date = latest_date

    lag_1 = latest["lag_1"]
    lag_7 = latest["lag_7"]
    lag_14 = latest["lag_14"]
    lag_28 = latest["lag_28"]

    rolling_mean_7 = latest["rolling_mean_7"]
    rolling_mean_14 = latest["rolling_mean_14"]
    rolling_mean_30 = latest["rolling_mean_30"]

    for day in range(1, 29):

        current_date = current_date + pd.Timedelta(days=1)

        row = latest.copy()

        row["month"] = current_date.month
        row["year"] = current_date.year
        row["wday"] = current_date.weekday() + 1

        row["quarter"] = current_date.quarter

        row["weekend"] = int(
            current_date.weekday() >= 5
        )

        row["month_sin"] = np.sin(
            2 * np.pi * current_date.month / 12
        )

        row["month_cos"] = np.cos(
            2 * np.pi * current_date.month / 12
        )

        row["wday_sin"] = np.sin(
            2 * np.pi * current_date.weekday() / 7
        )

        row["wday_cos"] = np.cos(
            2 * np.pi * current_date.weekday() / 7
        )

        row["lag_1"] = lag_1
        row["lag_7"] = lag_7
        row["lag_14"] = lag_14
        row["lag_28"] = lag_28

        row["rolling_mean_7"] = rolling_mean_7
        row["rolling_mean_14"] = rolling_mean_14
        row["rolling_mean_30"] = rolling_mean_30

        row["event_name_1"] = latest["event_name_1"]
        row["event_type_1"] = latest["event_type_1"]

        prediction = model.predict(
            pd.DataFrame([row[feature_cols]])
        )[0]

        prediction = max(0, prediction)

        lag_28 = lag_14
        lag_14 = lag_7
        lag_7 = lag_1
        lag_1 = prediction

        rolling_mean_7 = (
            rolling_mean_7 * 6 + prediction
        ) / 7

        rolling_mean_14 = (
            rolling_mean_14 * 13 + prediction
        ) / 14

        rolling_mean_30 = (
            rolling_mean_30 * 29 + prediction
        ) / 30

        future_rows.append({
            "date": current_date,
            "store_id": store,
            "dept_id": dept,
            "forecast_sales": round(prediction, 2)
        })

forecast_df = pd.DataFrame(
    future_rows
)

forecast_df["store_id"] = (
    encoders["store_id"]
    .inverse_transform(
        forecast_df["store_id"]
        .astype(int)
    )
)

forecast_df["dept_id"] = (
    encoders["dept_id"]
    .inverse_transform(
        forecast_df["dept_id"]
        .astype(int)
    )
)

forecast_df.to_csv(
    "outputs/future_forecasts.csv",
    index=False
)

print("\nForecast Shape:")
print(forecast_df.shape)

print("\nSample:")
print(forecast_df.head())

print("\nSaved:")
print("future_forecasts.csv")
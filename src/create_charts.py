import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(10,5))

forecasts = pd.read_csv("outputs/future_forecasts.csv")
results = pd.read_csv("outputs/model_results.csv")
inventory = pd.read_csv("outputs/inventory_recommendations.csv")

forecasts["date"] = pd.to_datetime(forecasts["date"])

plt.figure(figsize=(12, 6))

daily_forecast = (
    forecasts.groupby("date")["forecast_sales"]
    .sum()
)

plt.plot(
    daily_forecast.index,
    daily_forecast.values,
    linewidth=3
)

plt.title("28 Day Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Forecast Sales")

plt.tight_layout()

plt.savefig(
    "static/forecast_chart.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(8, 5))

plt.barh(
    results["Model"],
    results["RMSE"]
)

plt.title("Model RMSE Comparison")
colors = ["#2563eb","#10b981","#f59e0b","#ef4444"]

plt.tight_layout()

plt.savefig(
    "static/model_chart.png",
    dpi=300
)

plt.close()

risk_counts = (
    inventory["inventory_risk"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

plt.bar(
    risk_counts.index,
    risk_counts.values
)

plt.title("Inventory Risk Distribution")

plt.tight_layout()

plt.savefig(
    "static/inventory_chart.png",
    dpi=300
)

plt.close()

print("Charts Saved")
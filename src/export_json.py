import pandas as pd
import os

print("Creating dashboard JSON files...")

os.makedirs("dashboard/data", exist_ok=True)

files = {
    "outputs/future_forecasts.csv":
        "dashboard/data/future_forecasts.json",

    "outputs/inventory_recommendations.csv":
        "dashboard/data/inventory_recommendations.json",

    "outputs/model_results.csv":
        "dashboard/data/model_results.json",

    "outputs/feature_importance.csv":
        "dashboard/data/feature_importance.json",

    "outputs/shap_feature_importance.csv":
        "dashboard/data/shap_feature_importance.json"
}

for csv_file, json_file in files.items():

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)

        df.to_json(
            json_file,
            orient="records",
            indent=4
        )

        print(f"✓ {json_file}")

    else:
        print(f"✗ Missing: {csv_file}")

print("\nDone.")
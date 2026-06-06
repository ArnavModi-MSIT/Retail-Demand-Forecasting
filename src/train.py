import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("Loading dataset...")

df = pd.read_csv("data/features_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

print(f"Dataset Shape: {df.shape}")

categorical_cols = [
    "store_id",
    "dept_id",
    "state_id",
    "event_name_1",
    "event_type_1",
]

print("Encoding categorical features...")

encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

test_days = 90

cutoff_date = (
    df["date"].max()
    - pd.Timedelta(days=test_days)
)

train_df = df[df["date"] <= cutoff_date].copy()
test_df = df[df["date"] > cutoff_date].copy()

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

target = "sales"

drop_cols = [
    "sales",
    "date",
]

feature_cols = [
    col
    for col in df.columns
    if col not in drop_cols
]

X_train = train_df[feature_cols]
y_train = train_df[target]

X_test = test_df[feature_cols]
y_test = test_df[target]

print(f"Number of Features: {len(feature_cols)}")

results = []


def calculate_mape(y_true, y_pred):
    mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mape = calculate_mape(
        y_true,
        y_pred,
    )

    results.append(
        {
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
        }
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}")


print("\nRunning Naive Baseline...")

naive_pred = X_test["lag_7"]

evaluate_model(
    "Naive_7Day",
    y_test,
    naive_pred,
)

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train,
)

rf_pred = rf.predict(X_test)

evaluate_model(
    "RandomForest",
    y_test,
    rf_pred,
)

print("\nTraining XGBoost...")

print("\nHyperparameter Tuning XGBoost...")

tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

base_xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(search.best_params_)

best_params = pd.DataFrame(
    [search.best_params_]
)

best_params.to_csv(
    "outputs/best_xgb_params.csv",
    index=False
)

xgb = search.best_estimator_

xgb_pred = xgb.predict(X_test)

predictions_df = pd.DataFrame({
    "date": test_df["date"],
    "actual_sales": y_test,
    "predicted_sales": xgb_pred
})

predictions_df.to_csv(
    "outputs/predictions.csv",
    index=False
)

evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred,
)


print("\nTraining LightGBM...")

lgbm = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

lgbm.fit(
    X_train,
    y_train,
)

lgbm_pred = lgbm.predict(X_test)

evaluate_model(
    "LightGBM",
    y_test,
    lgbm_pred,
)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False,
)

importance_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": xgb.feature_importances_,
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False,
)

print("\n")
print("=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)
print(importance_df.head(20))

best_model = results_df.iloc[0]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)
print(best_model)

print("\nSaved Files:")
print("model_results.csv")
print("feature_importance.csv")

import shap
import matplotlib.pyplot as plt

print("\nRunning SHAP Analysis...")

sample_size = min(2000, len(X_test))

X_shap = X_test.sample(
    sample_size,
    random_state=42
)

explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_shap)

shap_importance = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
})

shap_importance = (
    shap_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

shap_importance.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    show=False
)

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTop SHAP Features")
print(shap_importance.head(20))

import joblib

joblib.dump(
    xgb,
    "models/xgb_model.pkl"
)

print("Saved: xgb_model.pkl")
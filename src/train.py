import numpy as np
import pandas as pd

from src.preprocessing import prepare_training_frame
from src.splitter import split_time_series
from src.models import fit_model
from src.evaluate import evaluate_metrics, run_shap


def train_models(
    df: pd.DataFrame,
    model_choice: str = "XGBoost",
    test_days: int = 90,
    progress_callback=None,
) -> dict:
    """
    Full training pipeline: preprocess → split → train → evaluate.

    Parameters
    ----------
    df
        Raw mapped dataframe (date + sales + optional columns).
    model_choice
        One of "XGBoost", "LightGBM", "Random Forest".
    test_days
        Days reserved for test evaluation.
    progress_callback
        Optional callable(pct: float, msg: str) for UI progress updates.

    Returns
    -------
    Dict with keys:
        model, model_name, predictions_df, results_df,
        feature_cols, encoders, X_test, y_test,
        feature_importance_df, train_size, test_size, cutoff_date
    """
    if progress_callback:
        progress_callback(0.05, "Preparing features...")

    featured, feature_cols, encoders = prepare_training_frame(df.copy(), feature_set="expanded")

    train_df, test_df, X_train, y_train, X_test, y_test, cutoff_date = split_time_series(
        featured, feature_cols, test_days=test_days
    )

    results = []

    # Naive baseline
    if "lag_7" in X_test.columns:
        naive_metrics = evaluate_metrics(y_test.values, X_test["lag_7"].values)
        results.append({"Model": "Naive (7-day)", **naive_metrics})

    if progress_callback:
        progress_callback(0.2, f"Training {model_choice}...")

    model = fit_model(model_choice, X_train, y_train)
    importance = (
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else np.zeros(len(feature_cols))
    )

    if progress_callback:
        progress_callback(0.7, "Evaluating model...")

    y_pred = model.predict(X_test)
    results.append({"Model": model_choice, **evaluate_metrics(y_test.values, y_pred)})

    # Decode store/dept IDs back to original labels
    store_vals = test_df["store_id"].values if "store_id" in test_df.columns else "All"
    dept_vals  = test_df["dept_id"].values  if "dept_id"  in test_df.columns else "All"

    if "store_id" in encoders and "store_id" in test_df.columns:
        store_vals = encoders["store_id"].inverse_transform(store_vals.astype(int))
    if "dept_id" in encoders and "dept_id" in test_df.columns:
        dept_vals = encoders["dept_id"].inverse_transform(dept_vals.astype(int))

    predictions_df = pd.DataFrame({
        "date":            test_df["date"].values,
        "store_id":        store_vals,
        "dept_id":         dept_vals,
        "actual_sales":    y_test.values,
        "predicted_sales": y_pred,
    })

    feature_importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    results_df = pd.DataFrame(results).sort_values("RMSE")

    return {
        "model":                 model,
        "model_name":            model_choice,
        "predictions_df":        predictions_df,
        "results_df":            results_df,
        "feature_cols":          feature_cols,
        "encoders":              encoders,
        "X_test":                X_test,
        "y_test":                y_test,
        "feature_importance_df": feature_importance_df,
        "train_size":            len(train_df),
        "test_size":             len(test_df),
        "cutoff_date":           cutoff_date,
    }
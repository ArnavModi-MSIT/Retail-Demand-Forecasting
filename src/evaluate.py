import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, skipping zero-valued actuals.
    Returns 0.0 if no non-zero actuals exist.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and MAPE for a set of predictions.

    Parameters
    ----------
    y_true, y_pred
        Arrays of true and predicted values.

    Returns
    -------
    Dict with keys MAE, RMSE, MAPE — all rounded to 4 decimal places.
    """
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE": round(calculate_mape(y_true, y_pred), 4),
    }


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(model, X_test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Run SHAP TreeExplainer on a sample of X_test.

    Falls back to model.feature_importances_ if SHAP raises an exception.

    Parameters
    ----------
    model
        Fitted tree model (XGBoost, LightGBM, RandomForest).
    X_test
        Test feature matrix. Non-numeric columns are coerced to float.
    model_name
        Model identifier string (unused internally, kept for API consistency).

    Returns
    -------
    DataFrame with columns: feature, mean_abs_shap — sorted descending.
    """
    X_clean = X_test.copy()
    for col in X_clean.columns:
        if not pd.api.types.is_numeric_dtype(X_clean[col]):
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
    X_clean = X_clean.fillna(0)

    if len(X_clean) == 0:
        raise ValueError("X_test is empty.")

    sample = X_clean.sample(min(1000, len(X_clean)), random_state=42)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return pd.DataFrame({
            "feature":       sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

    except Exception as e:
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame({
                "feature":       sample.columns,
                "mean_abs_shap": model.feature_importances_[: len(sample.columns)],
            }).sort_values("mean_abs_shap", ascending=False)
        raise e

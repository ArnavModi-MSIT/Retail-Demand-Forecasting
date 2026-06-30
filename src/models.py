import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.preprocessing import prepare_training_frame
from src.splitter import split_time_series
from src.evaluate import evaluate_metrics


# ── Per-algorithm hyperparameters ─────────────────────────────────────────────

XGBOOST_PARAM_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [4, 6],
    "learning_rate":    [0.03, 0.05],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

LIGHTGBM_DEFAULTS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

RANDOM_FOREST_DEFAULTS = dict(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
)


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_model(model_choice: str, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit the selected model on training data.

    Parameters
    ----------
    model_choice
        One of "XGBoost", "LightGBM", "Random Forest".
    X_train, y_train
        Training feature matrix and target.

    Returns
    -------
    Fitted model instance.
    """
    if model_choice == "XGBoost":
        n_splits = min(3, max(2, len(X_train) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Subsample for HPO when training set is large — refit best params on
        # full data. Keeps search fast without sacrificing final model quality.
        HPO_SAMPLE_LIMIT = 50_000
        if len(X_train) > HPO_SAMPLE_LIMIT:
            sample_idx = np.random.default_rng(42).choice(
                len(X_train), size=HPO_SAMPLE_LIMIT, replace=False
            )
            sample_idx.sort()          # preserve time order
            X_hpo = X_train.iloc[sample_idx]
            y_hpo = y_train.iloc[sample_idx]
        else:
            X_hpo, y_hpo = X_train, y_train

        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )

        # Defensive guard: XGBoost's sklearn wrapper raises an opaque,
        # deeply-nested ValueError if any column isn't numeric. Catch it
        # here with a clear, actionable message instead.
        non_numeric = [c for c in X_hpo.columns if not pd.api.types.is_numeric_dtype(X_hpo[c])]
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns reached XGBoost training: {non_numeric}. "
                "These should have been LabelEncoded in prepare_training_frame() — "
                "check src/preprocessing.py and src/features_retail.py for a column "
                "that bypasses encoding (e.g. a dtype reassignment after encoding, "
                "or a column not produced by engineer_features() at fit time)."
            )

        search = RandomizedSearchCV(
            base, XGBOOST_PARAM_GRID,
            n_iter=6,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_hpo, y_hpo)

        # Refit best params on full training data
        best = XGBRegressor(
            **search.best_params_,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        best.fit(X_train, y_train)
        return best

    if model_choice == "LightGBM":
        model = LGBMRegressor(**LIGHTGBM_DEFAULTS)
        model.fit(X_train, y_train)
        return model

    # Default: Random Forest
    model = RandomForestRegressor(**RANDOM_FOREST_DEFAULTS)
    model.fit(X_train, y_train)
    return model


# ── Feature set comparison utility ───────────────────────────────────────────

def compare_feature_sets(
    df: pd.DataFrame,
    test_days: int = 90,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Train every supported model on legacy vs expanded features and return
    a comparison metrics dataframe.

    Parameters
    ----------
    df
        Raw mapped dataframe.
    test_days
        Days reserved for evaluation.
    max_rows
        Optional row cap for faster experimentation.

    Returns
    -------
    DataFrame with columns: feature_set, model, MAE, RMSE, MAPE,
    baseline_rmse, baseline_improvement, feature_cols.
    """
    working = df.copy()
    if max_rows is not None and len(working) > max_rows:
        working = (
            working.sample(max_rows, random_state=42)
            .sort_values("date")
            .reset_index(drop=True)
        )

    rows = []
    for feature_set in ["legacy", "expanded"]:
        for model_choice in ["XGBoost", "LightGBM", "Random Forest"]:
            featured, feature_cols, _ = prepare_training_frame(working, feature_set=feature_set)
            _, test_df, X_train, y_train, X_test, y_test, _ = split_time_series(
                featured, feature_cols, test_days=test_days
            )

            model   = fit_model(model_choice, X_train, y_train)
            y_pred  = model.predict(X_test)
            metrics = evaluate_metrics(y_test.values, y_pred)

            naive_pred    = X_test["lag_7"].values if "lag_7" in X_test.columns else y_test.values * 0.0
            naive_metrics = evaluate_metrics(y_test.values, naive_pred)

            rows.append({
                "feature_set":          feature_set,
                "model":                model_choice,
                "MAE":                  metrics["MAE"],
                "RMSE":                 metrics["RMSE"],
                "MAPE":                 metrics["MAPE"],
                "baseline_rmse":        naive_metrics["RMSE"],
                "baseline_improvement": round(
                    (naive_metrics["RMSE"] - metrics["RMSE"]) / naive_metrics["RMSE"] * 100, 1
                ) if naive_metrics["RMSE"] else None,
                "feature_cols":         len(feature_cols),
            })

    return pd.DataFrame(rows)
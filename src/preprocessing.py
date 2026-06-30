import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features import engineer_features, engineer_features_legacy


def prepare_training_frame(
    df: pd.DataFrame,
    feature_set: str = "expanded",
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Run feature engineering and encode categorical columns with LabelEncoder.

    Parameters
    ----------
    df
        Raw mapped dataframe (date + sales + optional columns).
    feature_set
        "expanded" (default) or "legacy".

    Returns
    -------
    featured
        Fully engineered and encoded dataframe.
    feature_cols
        List of column names to use as model inputs.
    encoders
        Dict of {col: LabelEncoder} for inverse-transforming predictions.
    """
    if feature_set == "legacy":
        featured = engineer_features_legacy(df.copy())
    else:
        featured = engineer_features(df.copy())

    # "date" must never be encoded — it's neither numeric nor boolean per
    # pandas' dtype checks (it's datetime64), so it has to be excluded
    # explicitly or it gets silently converted to an int by LabelEncoder,
    # breaking downstream Timedelta arithmetic in splitter.py.
    categorical_cols = [
        col for col in featured.columns
        if col != "date"
        and not pd.api.types.is_numeric_dtype(featured[col])
        and not pd.api.types.is_bool_dtype(featured[col])
        and not pd.api.types.is_datetime64_any_dtype(featured[col])
    ]
    encoders = {}
    for col in categorical_cols:
        enc = LabelEncoder()
        featured[col] = enc.fit_transform(featured[col].fillna("Unknown").astype(str))
        encoders[col] = enc

    drop_cols = {"sales", "date"}
    if "price_lag_1" in featured.columns:
        drop_cols.add("price_lag_1")
    feature_cols = [c for c in featured.columns if c not in drop_cols]

    return featured, feature_cols, encoders
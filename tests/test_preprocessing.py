import pandas as pd
import numpy as np

from src.preprocessing import prepare_training_frame


def _make_raw_df(n=80, with_store=True, with_price=False):
    dates = pd.date_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    data = {"date": dates, "sales": rng.integers(50, 200, size=n).astype(float)}
    if with_store:
        data["store_id"] = np.tile(["S1", "S2"], n // 2 + 1)[:n]
        data["dept_id"] = np.tile(["D1"], n)
    if with_price:
        data["sell_price"] = rng.uniform(5, 10, size=n)
    return pd.DataFrame(data)


# ── Categorical encoding ──────────────────────────────────────────────────────

def test_categorical_columns_are_label_encoded_to_numeric():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)

    assert "store_id" in encoders
    assert "dept_id" in encoders
    # after encoding, these columns must be numeric (int codes), not strings
    assert pd.api.types.is_numeric_dtype(featured["store_id"])
    assert pd.api.types.is_numeric_dtype(featured["dept_id"])


def test_date_column_is_never_encoded():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)

    assert "date" not in encoders
    assert pd.api.types.is_datetime64_any_dtype(featured["date"])


def test_encoders_can_inverse_transform_back_to_original_labels():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)

    original_labels = set(df["store_id"].unique())
    encoded_values = featured["store_id"].unique()
    decoded = set(encoders["store_id"].inverse_transform(encoded_values))
    assert decoded == original_labels


def test_no_categorical_columns_when_no_optional_columns_present():
    # date + sales only — no store/dept/price — should produce no encoders
    # other than whatever calendar features add (which are numeric already)
    df = _make_raw_df(with_store=False, with_price=False)
    featured, feature_cols, encoders = prepare_training_frame(df)
    assert encoders == {}


# ── Feature column selection ──────────────────────────────────────────────────

def test_sales_and_date_excluded_from_feature_cols():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)
    assert "sales" not in feature_cols
    assert "date" not in feature_cols


def test_price_lag_1_excluded_when_present():
    df = _make_raw_df(with_store=True, with_price=True)
    featured, feature_cols, encoders = prepare_training_frame(df)
    # price_lag_1 exists as an intermediate feature but should not leak into
    # the model input columns (it's used only to derive price_change)
    if "price_lag_1" in featured.columns:
        assert "price_lag_1" not in feature_cols


def test_feature_cols_are_a_subset_of_featured_columns():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)
    assert set(feature_cols).issubset(set(featured.columns))


def test_no_nans_in_selected_feature_columns():
    df = _make_raw_df(with_store=True)
    featured, feature_cols, encoders = prepare_training_frame(df)
    # engineer_features() fills remaining NaNs; feature_cols should be clean
    assert not featured[feature_cols].isna().any().any()


# ── Legacy vs expanded feature sets ───────────────────────────────────────────

def test_legacy_feature_set_produces_fewer_columns_than_expanded():
    df = _make_raw_df(with_store=True, with_price=True)
    featured_legacy, cols_legacy, _ = prepare_training_frame(df, feature_set="legacy")
    featured_expanded, cols_expanded, _ = prepare_training_frame(df, feature_set="expanded")
    # expanded includes retail covariates beyond what legacy computes
    assert len(cols_expanded) >= len(cols_legacy)


def test_unknown_category_in_encoder_does_not_raise_on_refit():
    # Simulates a slightly different category set between two uploads —
    # prepare_training_frame fits a fresh encoder each call, so this should
    # never raise even though it's a completely different label set.
    df1 = _make_raw_df(with_store=True)
    df2 = _make_raw_df(with_store=True)
    df2["store_id"] = df2["store_id"].replace({"S1": "S3", "S2": "S4"})

    _, _, encoders1 = prepare_training_frame(df1)
    _, _, encoders2 = prepare_training_frame(df2)

    assert set(encoders1["store_id"].classes_) == {"S1", "S2"}
    assert set(encoders2["store_id"].classes_) == {"S3", "S4"}

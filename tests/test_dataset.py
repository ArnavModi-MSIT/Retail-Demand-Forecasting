import pandas as pd
import pytest

from src.dataset import detect_columns, apply_column_mapping, validate_data, detect_sales_type


# ── detect_columns ────────────────────────────────────────────────────────────

def test_detect_columns_required_present():
    df = pd.DataFrame({"date": ["2024-01-01"], "sales": [10]})
    mapping = detect_columns(df)
    assert mapping["date"] == "date"
    assert mapping["sales"] == "sales"


def test_detect_columns_missing_required_raises():
    df = pd.DataFrame({"date": ["2024-01-01"]})  # no sales column
    with pytest.raises(ValueError, match="sales"):
        detect_columns(df)


def test_detect_columns_alias_matching_case_insensitive():
    # "Store" should map to canonical "store_id" via OPTIONAL_COLS aliases
    df = pd.DataFrame({"Date": ["2024-01-01"], "Sales": [10], "Store": ["S1"]})
    mapping = detect_columns(df)
    assert mapping["date"] == "Date"
    assert mapping["sales"] == "Sales"
    assert mapping["store_id"] == "Store"


def test_detect_columns_optional_absent_is_none():
    df = pd.DataFrame({"date": ["2024-01-01"], "sales": [10]})
    mapping = detect_columns(df)
    assert mapping["store_id"] is None
    assert mapping["dept_id"] is None


def test_apply_column_mapping_renames_correctly():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Sales": [10], "Store": ["S1"]})
    mapping = detect_columns(df)
    renamed = apply_column_mapping(df, mapping)
    assert set(["date", "sales", "store_id"]).issubset(renamed.columns)


# ── validate_data ─────────────────────────────────────────────────────────────

def test_validate_data_flags_non_numeric_sales():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "sales": [1, 2, "bad", 4, 5],
    })
    warnings = validate_data(df)
    assert any("non-numeric" in w for w in warnings)


def test_validate_data_flags_negative_sales():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "sales": [10, -5, 20],
    })
    warnings = validate_data(df)
    assert any("negative" in w for w in warnings)


def test_validate_data_flags_small_dataset():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "sales": [1, 2, 3, 4, 5],
    })
    warnings = validate_data(df)
    assert any("only 5 rows" in w for w in warnings)


def test_validate_data_clean_data_has_no_blocking_errors():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=200),
        "sales": range(200),
    })
    warnings = validate_data(df)
    assert not any(w.startswith("❌") for w in warnings)


# ── detect_sales_type ─────────────────────────────────────────────────────────

def test_detect_sales_type_daily():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})
    assert detect_sales_type(df) == "daily"


def test_detect_sales_type_weekly():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="7D")})
    assert detect_sales_type(df) == "weekly"


def test_detect_sales_type_monthly():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="30D")})
    assert detect_sales_type(df) == "monthly"


def test_detect_sales_type_dedupes_across_groups():
    # Same dates repeated for multiple stores shouldn't look "daily" due to
    # near-zero diffs from row ordering across groups.
    dates = list(pd.date_range("2024-01-01", periods=5, freq="7D")) * 3
    df = pd.DataFrame({"date": dates})
    assert detect_sales_type(df) == "weekly"

import pandas as pd
import numpy as np

# ── Column detection ───────────────────────────────────────────────────────────

REQUIRED_COLS = ["date", "sales"]

OPTIONAL_COLS = {
    "store_id": ["store", "store_id", "storeid", "store id"],
    "dept_id": ["dept", "dept_id", "deptid", "department", "category"],
    "sell_price": ["price", "sell_price", "selling_price", "unit_price"],
    "day_of_week": ["day_of_week", "wday", "weekday", "dow"],
    "month": ["month"],
    "customers": ["customers", "customer_count", "customer_counts"],
    "open": ["open", "is_open"],
    "promo": ["promo", "promotion", "is_promo"],
    "state_holiday": ["state_holiday", "holiday_state", "is_state_holiday"],
    "school_holiday": ["school_holiday", "is_school_holiday"],
    "store_type": ["store_type", "storetype"],
    "assortment": ["assortment"],
    "competition_distance": ["competition_distance", "comp_distance"],
    "competition_open_month": ["competition_open_month", "comp_open_month"],
    "competition_open_year": ["competition_open_year", "comp_open_year"],
    "promo2": ["promo2", "promotion2"],
    "promo2_since_week": ["promo2_since_week", "promo2week"],
    "promo2_since_year": ["promo2_since_year", "promo2year"],
    "promo_interval": ["promo_interval", "promointerval"],
    "event_type_1": ["event_type", "event_type_1", "event"],
    "event_name_1": ["event_name", "event_name_1"],
    "state_id": ["state", "state_id"],
}


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Returns a mapping {canonical_name: actual_col_name}.
    Raises ValueError if required columns are missing.
    """

    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for req in REQUIRED_COLS:
        candidates = [
            req,
            req.replace("_", " "),
            req.replace("_", "")
        ]

        found = next((cols_lower[c] for c in candidates if c in cols_lower), None)

        if found is None:
            raise ValueError(
                f"Required column '{req}' not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )

        mapping[req] = found

    for canon, aliases in OPTIONAL_COLS.items():
        found = next(
            (cols_lower[a.lower()] for a in aliases if a.lower() in cols_lower),
            None,
        )
        mapping[canon] = found

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename = {
        actual: canonical
        for canonical, actual in mapping.items()
        if actual is not None and actual != canonical
    }
    return df.rename(columns=rename)


# ── Sales type detection ──────────────────────────────────────────────────────

def detect_sales_type(df: pd.DataFrame) -> str:
    """
    Detect whether the uploaded data appears to be:

    - daily
    - weekly
    - monthly
    - irregular

    Returns a string.
    """

    if "date" not in df.columns:
        return "unknown"

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()

    if len(dates) < 3:
        return "unknown"

    # Deduplicate: multiple stores/depts share the same dates.
    # Use unique dates only so cross-group row ordering doesn't produce
    # spurious near-zero diffs that misclassify weekly data as daily.
    dates = dates.drop_duplicates().sort_values()

    diffs = dates.diff().dt.days.dropna()

    if diffs.empty:
        return "unknown"

    median_gap = diffs.median()

    if median_gap <= 2:
        return "daily"

    if median_gap <= 10:
        return "weekly"

    if median_gap <= 35:
        return "monthly"

    return "irregular"


# ── Data validation ───────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> list[str]:
    """
    Returns:

    ❌ = blocking errors

    ⚠️ = warnings
    """

    warnings = []

    # Date parsing

    try:
        parsed_dates = pd.to_datetime(df["date"])
    except Exception:
        warnings.append("❌ 'date' column could not be parsed as dates.")
        return warnings

    # Duplicate dates

    dup = parsed_dates.duplicated().sum()

    if dup > 0:
        warnings.append(
            f"⚠️ {dup} duplicate dates found."
        )

    # Sales numeric

    try:
        sales = pd.to_numeric(df["sales"], errors="coerce")
    except Exception:
        warnings.append("❌ 'sales' column contains invalid values.")
        return warnings

    invalid = sales.isna().sum()

    if invalid:
        warnings.append(
            f"❌ {invalid} rows contain non-numeric sales values."
        )

    infinite = np.isinf(sales).sum()

    if infinite:
        warnings.append(
            f"❌ {infinite} rows contain infinite sales values."
        )

    negatives = (sales < 0).sum()

    if negatives:
        warnings.append(
            f"⚠️ {negatives} rows contain negative sales."
        )

    # Missing values

    null_pct = df.isnull().mean() * 100

    for col, pct in null_pct.items():
        if pct > 20:
            warnings.append(
                f"⚠️ Column '{col}' has {pct:.1f}% missing values."
            )

    # Dataset size

    if len(df) < 100:
        warnings.append(
            f"⚠️ Dataset contains only {len(df)} rows. "
            "Forecast accuracy may be limited."
        )

    return warnings
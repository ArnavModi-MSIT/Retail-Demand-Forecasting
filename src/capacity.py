"""
Demo capacity guard.

The live deployment runs on a memory-constrained instance (e.g. AWS EC2
t3.micro, 1 GB RAM). Training XGBoost with hyperparameter search on a
full multi-million-row dataset will OOM-kill the process. This module
caps the working dataset size for the live demo while leaving local /
full-scale runs completely unrestricted.

Controlled via the MAX_DEMO_ROWS environment variable:
    - Unset or 0  → no cap (local development, full-scale benchmarking)
    - Set to N    → datasets larger than N rows are sampled down to fit

Sampling strategy preserves time-series integrity — it never randomly
drops rows, which would corrupt lag/rolling features. Instead:
    - If store_id exists: keep a subset of stores in full (complete date
      range intact for each), enough to stay under the cap.
    - Otherwise: keep the most recent contiguous date range.
"""
import os

import pandas as pd


def get_demo_row_cap() -> int | None:
    """Return the configured row cap, or None if uncapped."""
    raw = os.environ.get("MAX_DEMO_ROWS", "0")
    try:
        cap = int(raw)
    except ValueError:
        cap = 0
    return cap if cap > 0 else None


def apply_demo_cap(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """
    Apply the demo row cap if configured and the dataset exceeds it.

    Parameters
    ----------
    df
        Raw mapped dataframe (already has canonical column names).

    Returns
    -------
    (df_capped, notice)
        df_capped: the dataframe to use downstream (unchanged if no cap
                   or dataset already fits).
        notice: dict with details for a UI message, or None if no
                sampling occurred.
    """
    cap = get_demo_row_cap()
    if cap is None or len(df) <= cap:
        return df, None

    original_rows = len(df)

    if "store_id" in df.columns:
        # Keep whole stores' history intact — never split a single
        # store's time series, which would break lag/rolling features.
        store_sizes = df.groupby("store_id").size().sort_values()
        kept_stores = []
        running_total = 0
        for store_id, size in store_sizes.items():
            if running_total + size > cap and kept_stores:
                break
            kept_stores.append(store_id)
            running_total += size

        sampled = df[df["store_id"].isin(kept_stores)].copy()
        notice = {
            "original_rows": original_rows,
            "sampled_rows": len(sampled),
            "strategy": "stores",
            "detail": f"kept {len(kept_stores)} of {df['store_id'].nunique()} stores",
        }
    else:
        # No grouping column — keep the most recent contiguous window.
        sampled = df.copy()
        sampled["__date_tmp"] = pd.to_datetime(sampled["date"])
        sampled = sampled.sort_values("__date_tmp").tail(cap).drop(columns="__date_tmp")
        notice = {
            "original_rows": original_rows,
            "sampled_rows": len(sampled),
            "strategy": "recent_window",
            "detail": "kept the most recent date range",
        }

    return sampled.reset_index(drop=True), notice
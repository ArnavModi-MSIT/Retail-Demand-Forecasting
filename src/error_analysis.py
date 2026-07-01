"""
Error segmentation analysis.

Breaks down test-period forecast error (already computed by train.py —
no new model training here) by store, weekend/weekday, and demand tier,
to answer "where is the model wrong, and why" rather than just "how
wrong is the model on average."

All inputs are DataFrames already held in memory by the pipeline
(predictions_df, df_featured, inventory_df) — this module does no I/O
and adds no load to the live app beyond a few pandas groupby calls.
"""
import numpy as np
import pandas as pd

MIN_SEGMENT_ROWS = 10  # segments with fewer rows than this are dropped —
                        # too few points for a mean error to be meaningful


def _segment_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """MAE/MAPE per group, dropping groups with too few rows to be reliable."""
    def _mape(g):
        mask = g["actual_sales"] != 0
        if mask.sum() == 0:
            return np.nan
        return float(
            np.mean(np.abs((g.loc[mask, "actual_sales"] - g.loc[mask, "predicted_sales"])
                            / g.loc[mask, "actual_sales"])) * 100
        )

    rows = []
    for key, g in df.groupby(group_col):
        if len(g) < MIN_SEGMENT_ROWS:
            continue
        mae = float(np.mean(np.abs(g["actual_sales"] - g["predicted_sales"])))
        rows.append({group_col: key, "n_rows": len(g), "MAE": round(mae, 2), "MAPE": round(_mape(g), 2)})

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("MAE", ascending=False).reset_index(drop=True)
    return result


def compute_error_segments(
    predictions_df: pd.DataFrame,
    df_featured: pd.DataFrame = None,
    inventory_df: pd.DataFrame = None,
) -> dict:
    """
    Segment test-period forecast error by store, weekend/weekday, and
    demand tier.

    Parameters
    ----------
    predictions_df
        Output of train.py — date, store_id, dept_id, actual_sales,
        predicted_sales for the test period.
    df_featured
        Full engineered dataframe (has is_weekend / calendar flags).
        Optional — weekend segmentation is skipped if not provided
        (e.g. for a run loaded from the database with no live features).
    inventory_df
        Output of compute_inventory (has avg_daily_demand per group).
        Optional — demand-tier segmentation is skipped if not provided.

    Returns
    -------
    Dict with keys "by_store", "by_weekend", "by_demand_tier" — each a
    DataFrame, empty if that segmentation wasn't possible or every
    segment had too few rows.
    """
    empty = {"by_store": pd.DataFrame(), "by_weekend": pd.DataFrame(), "by_demand_tier": pd.DataFrame()}

    if predictions_df is None or predictions_df.empty or "date" not in predictions_df.columns:
        return empty

    preds = predictions_df.copy()
    preds["date"] = pd.to_datetime(preds["date"])

    results = dict(empty)

    # ── By store ───────────────────────────────────────────────────────────
    if "store_id" in preds.columns:
        results["by_store"] = _segment_metrics(preds, "store_id")

    # ── By weekend / weekday ──────────────────────────────────────────────
    if df_featured is not None and not df_featured.empty and "is_weekend" in df_featured.columns:
        cal = df_featured[["date", "is_weekend"]].copy()
        cal["date"] = pd.to_datetime(cal["date"])
        join_cols = ["date"]
        merge_cols = ["date", "is_weekend"]
        if "store_id" in df_featured.columns and "store_id" in preds.columns:
            cal["store_id"] = df_featured["store_id"].astype(str)
            preds["store_id_str"] = preds["store_id"].astype(str)
            join_cols = ["date", "store_id"]
            merge_cols = ["date", "store_id", "is_weekend"]

        cal = cal[merge_cols].drop_duplicates()

        if join_cols == ["date", "store_id"]:
            merged = preds.merge(
                cal, left_on=["date", "store_id_str"], right_on=["date", "store_id"], how="left"
            )
        else:
            merged = preds.merge(cal, on="date", how="left")

        merged = merged.dropna(subset=["is_weekend"])
        if not merged.empty:
            merged["day_type"] = merged["is_weekend"].map({1: "Weekend", 0: "Weekday"})
            results["by_weekend"] = _segment_metrics(merged, "day_type")

    # ── By demand tier ────────────────────────────────────────────────────
    if (
        inventory_df is not None and not inventory_df.empty
        and "avg_daily_demand" in inventory_df.columns
    ):
        group_cols = [c for c in ["store_id", "dept_id"] if c in inventory_df.columns and c in preds.columns]
        if group_cols:
            demand_map = inventory_df[group_cols + ["avg_daily_demand"]].copy()
            for c in group_cols:
                demand_map[c] = demand_map[c].astype(str)
                preds[c] = preds[c].astype(str)

            median_demand = demand_map["avg_daily_demand"].median()
            demand_map["demand_tier"] = np.where(
                demand_map["avg_daily_demand"] >= median_demand, "High Demand", "Low Demand"
            )

            merged = preds.merge(demand_map[group_cols + ["demand_tier"]], on=group_cols, how="left")
            merged = merged.dropna(subset=["demand_tier"])
            if not merged.empty:
                results["by_demand_tier"] = _segment_metrics(merged, "demand_tier")

    return results

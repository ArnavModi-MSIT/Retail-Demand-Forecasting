import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.pipeline import (
    detect_columns,
    apply_column_mapping,
    validate_data,
)
from db_layer.repository import list_runs, get_run, get_model_metrics, get_forecasts, get_inventory
from db_layer.connection import check_connection


def _render_load_previous_run():
    """Show a dropdown of past runs and let the user reload one (view-only)."""
    if not check_connection():
        return  # silently skip — DB not reachable, don't block the upload flow

    runs_df = list_runs(limit=20)
    if runs_df.empty:
        return

    with st.expander("🕘 Load a Previous Run", expanded=False):
        st.caption(
            "Loaded runs are view-only — forecast/inventory settings won't "
            "update live since the trained model isn't reloaded. "
            "Re-run the pipeline to retrain."
        )

        def _fmt_local(ts):
            ts = pd.to_datetime(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")   # old naive rows are UTC
            # astimezone() with no arg converts to the machine's local timezone
            return ts.to_pydatetime().astimezone().strftime("%Y-%m-%d %H:%M")

        labels = [
            f"{row['dataset_name']}  •  {row['model_choice']}  •  "
            f"{row['n_rows']:,} rows  •  {_fmt_local(row['created_at'])}"
            for _, row in runs_df.iterrows()
        ]
        choice = st.selectbox("Select a run", ["(none)"] + labels)

        if choice != "(none)":
            idx = labels.index(choice)
            run_id = runs_df.iloc[idx]["run_id"]

            if st.button("📂 Load this run", width="stretch"):
                run_meta     = get_run(run_id)
                results_df   = get_model_metrics(run_id)
                future_df    = get_forecasts(run_id)
                inventory_df = get_inventory(run_id)

                horizon_days = run_meta["horizon_days"]
                total_demand = future_df["predicted_sales"].sum() if not future_df.empty else 0.0

                # Minimal summary dict — enough for results.py to render.
                # Loaded runs skip the live re-forecast path entirely.
                high_risk_demand = (
                    inventory_df.loc[
                        inventory_df["risk_level"].isin(["High Risk", "Critical"]), "forecast_demand"
                    ].sum()
                    if "risk_level" in inventory_df.columns and "forecast_demand" in inventory_df.columns
                    else 0.0
                )
                stockout_risk_pct = round(high_risk_demand / (total_demand + 1e-6) * 100, 1)

                summary = {
                    "total_demand":         round(total_demand, 0),
                    "average_daily_demand": round(total_demand / max(horizon_days, 1), 2),
                    "high_risk_combos":     int((inventory_df["risk_level"].isin(["High Risk", "Critical"])).sum())
                                            if "risk_level" in inventory_df.columns else 0,
                    "medium_risk_combos":   int((inventory_df["risk_level"] == "Medium Risk").sum())
                                            if "risk_level" in inventory_df.columns else 0,
                    "low_risk_combos":      int((inventory_df["risk_level"] == "Low Risk").sum())
                                            if "risk_level" in inventory_df.columns else 0,
                    "stockout_risk_pct":    stockout_risk_pct,
                    "safety_stock_total":   round(inventory_df["safety_stock"].sum(), 2)
                                            if "safety_stock" in inventory_df.columns else 0.0,
                    "horizon_days":         horizon_days,
                    "model_name":           run_meta["model_choice"],
                    "mape": (
                        results_df.loc[results_df["Model"] == run_meta["model_choice"], "MAPE"].values[0]
                        if not results_df.empty and run_meta["model_choice"] in results_df["Model"].values
                        else None
                    ),
                    "baseline_improvement": None,
                    "top_demand_combo":     None,
                    "reorder_point_total":  0.0,
                    "recommended_inventory_total": 0.0,
                    "total_skus": len(inventory_df),
                }

                st.session_state["results"] = {
                    "model_name":           run_meta["model_choice"],
                    "model_choice":         run_meta["model_choice"],
                    "results_df":           results_df,
                    "predictions_df":       pd.DataFrame(),   # not persisted — test-period detail unavailable
                    "future_df":            future_df,
                    "inventory_df":         inventory_df,
                    "feature_importance_df":pd.DataFrame(),   # not persisted
                    "shap_df":              pd.DataFrame(),   # not persisted
                    "summary":              summary,
                    "evaluation_metrics":   {"model_name": run_meta["model_choice"], "mape": summary["mape"],
                                              "baseline_improvement": None, "test_predictions_df": pd.DataFrame(),
                                              "results_df": results_df},
                    "forecasting_metrics":  {"horizon_days": horizon_days, "total_demand": summary["total_demand"],
                                              "average_daily_demand": summary["average_daily_demand"],
                                              "confidence_interval_width": None, "future_forecast_df": future_df},
                    "business_metrics":     {"high_risk_combos": summary["high_risk_combos"],
                                              "medium_risk_combos": summary["medium_risk_combos"],
                                              "low_risk_combos": summary["low_risk_combos"],
                                              "top_demand_combo": None, "stockout_risk_pct": 0.0,
                                              "inventory_df": inventory_df, "total_skus": len(inventory_df)},
                    "horizon_days":         horizon_days,
                    "service_level":        float(run_meta["service_level"]),
                    "lead_time":            run_meta["lead_time"],
                    "test_days":            run_meta["test_days"],
                    "run_id":               run_id,
                    "loaded_from_db":       True,   # flags results.py to skip live re-forecast
                }
                st.session_state["stage"] = "results"
                st.rerun()


def render_upload():
    st.markdown("### 📤 Upload your sales data")
    st.markdown("*CSV file with `date` and `sales` columns. Other columns are auto-detected.*")

    _render_load_previous_run()

    # Prominent upload box
    uploaded = st.file_uploader(
        "**Drag and drop your CSV here or click to browse**",
        type=["csv"],
        help="File size limit: 200MB. Accepted: date, sales, and optional columns like store_id, dept_id, sell_price, etc.",
    )

    if uploaded:
        with st.spinner("Reading file..."):
            df_raw = pd.read_csv(uploaded)

        st.session_state["uploaded_filename"] = uploaded.name
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

        # Column detection
        try:
            mapping = detect_columns(df_raw)
            st.info(
                "**Auto-detected columns:** " +
                " • ".join(f"`{k}`" for k, v in mapping.items() if v)
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # Manual override for unmapped columns — must run before stats/chart
        # below so they reflect any correction, since Streamlit reruns
        # top-to-bottom on every widget interaction.
        with st.expander("🔧 Adjust Column Mapping (if needed)"):
            st.caption("Override auto-detection if any column was mapped incorrectly.")
            all_cols = ["(none)"] + list(df_raw.columns)
            for canon in ["store_id", "dept_id", "sell_price", "event_type_1"]:
                current = mapping.get(canon)
                idx = all_cols.index(current) if current in all_cols else 0
                chosen = st.selectbox(f"`{canon}`", all_cols, index=idx, key=f"map_{canon}")
                mapping[canon] = None if chosen == "(none)" else chosen

        # Dataset stats (always visible)
        st.subheader("📊 Dataset Overview")
        df_mapped = apply_column_mapping(df_raw.copy(), mapping)
        c1, c2, c3, c4 = st.columns(4)
        _dates = pd.to_datetime(df_mapped["date"])
        c1.metric("Rows", f"{len(df_raw):,}")
        c2.metric(
            "Date Range",
            f"{_dates.min().strftime('%Y-%m-%d')} → {_dates.max().strftime('%Y-%m-%d')}",
        )
        c3.metric("Stores", df_mapped["store_id"].nunique() if "store_id" in df_mapped.columns else "1")
        c4.metric("Departments", df_mapped["dept_id"].nunique() if "dept_id" in df_mapped.columns else "1")

        # Sales trend — weekly avg per store, bar chart avoids daily noise
        n_stores = df_mapped["store_id"].nunique() if "store_id" in df_mapped.columns else 1
        _trend = df_mapped.copy()
        _trend["date"] = pd.to_datetime(_trend["date"])
        _weekly = (
            _trend.groupby(pd.Grouper(key="date", freq="W"))["sales"]
            .sum()
            .div(max(n_stores, 1))
            .reset_index()
        )
        fig = go.Figure(go.Bar(
            x=_weekly["date"],
            y=_weekly["sales"],
            marker_color="#1f77b4",
            name="Avg Weekly Sales per Store",
        ))
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Avg Sales per Store",
            height=280,
            margin=dict(t=10, b=10, l=0, r=0),
            hovermode="x unified",
            bargap=0.15,
        )
        st.plotly_chart(fig, width="stretch")

        # Preview (collapsible)
        with st.expander("📋 Preview Data (first 5 rows)"):
            st.dataframe(df_raw.head(5), width="stretch")

        # Validation
        warnings = validate_data(df_mapped)
        if warnings:
            for w in warnings:
                if w.startswith("❌"):
                    st.error(w)
                else:
                    st.warning(w)
            if any(w.startswith("❌") for w in warnings):
                st.stop()

        st.divider()

        if st.button("🚀 Run Forecasting Pipeline", type="primary", width="stretch"):
            st.session_state["df_mapped"] = df_mapped
            st.session_state["mapping"]   = mapping
            st.session_state["stage"]     = "running"
            st.rerun()

    else:
        # No file uploaded yet
        st.divider()

        st.subheader("📋 How it works")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.markdown("**1️⃣ Upload**\nYour sales CSV")
        col_b.markdown("**2️⃣ Validate**\nCheck for errors")
        col_c.markdown("**3️⃣ Train**\nBuild ML model")
        col_d.markdown("**4️⃣ Forecast**\nGet results")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("✅ Required Columns")
            st.markdown("""
- **`date`** — YYYY-MM-DD format
- **`sales`** — numeric (units sold)
            """)

        with col_right:
            st.subheader("➕ Optional Columns")
            st.markdown("""
- `store_id` — store identifier
- `dept_id` — department/category
- `sell_price` — unit price
- `event_type_1` — promotion/holiday
            """)

        st.divider()

        st.subheader("📝 Sample Data")
        sample = pd.DataFrame({
            "date":     ["2023-01-01", "2023-01-02", "2023-01-03"],
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "dept_id":  ["FOODS", "FOODS", "FOODS"],
            "sales":    [120, 95, 138],
            "sell_price": [2.50, 2.50, 2.75],
        })
        st.dataframe(sample, width="stretch", hide_index=True)

        col_dl, col_space = st.columns([1, 4])
        with col_dl:
            sample_csv = sample.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Sample",
                sample_csv,
                "sample_sales.csv",
                "text/csv",
                width="stretch",
            )
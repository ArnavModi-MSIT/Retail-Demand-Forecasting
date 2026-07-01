import streamlit as st

from src.pipeline import (
    generate_future_forecast,
    compute_inventory,
    generate_executive_summary,
)
from ui.tabs.forecast import render_forecast_tab
from ui.tabs.inventory import render_inventory_tab
from ui.tabs.explainability import render_explainability_tab
from ui.tabs.model import render_model_tab
from ui.tabs.download import render_download_tab


# ── Settings that require full re-train vs fast re-run ───────────────────────
RETRAIN_KEYS  = {"model_choice", "test_days"}
REFORECAST_KEYS = {"horizon_days", "service_level", "lead_time"}


def _reforecast(R: dict, config: dict) -> dict:
    """
    Re-run only the fast steps (forecast → inventory → summary) using the
    already-trained model. Returns an updated results dict.
    """
    horizon_days  = config["horizon_days"]
    service_level = config["service_level"]
    lead_time     = config["lead_time"]

    future_df = generate_future_forecast(
        R["model"],
        R["df_featured"],
        R["feature_cols"],
        horizon_days=horizon_days,
        X_test=R["X_test"],
        y_test=R["y_test"],
        confidence_level=0.95,
        encoders=R["encoders"],
    )

    inventory_df = compute_inventory(
        history_df=R["df_featured"],
        future_df=future_df,
        service_level=service_level,
        lead_time_days=lead_time,
    )

    reporting_bundle = generate_executive_summary(
        R["predictions_df"],
        inventory_df,
        R["results_df"],
        R["model_name"],
        horizon_days,
        future_forecast_df=future_df,
    )

    return {
        **R,
        "future_df":          future_df,
        "inventory_df":       inventory_df,
        "summary":            reporting_bundle["summary"],
        "evaluation_metrics": reporting_bundle["evaluation_metrics"],
        "forecasting_metrics":reporting_bundle["forecasting_metrics"],
        "business_metrics":   reporting_bundle["business_metrics"],
        "horizon_days":       horizon_days,
        "service_level":      service_level,
        "lead_time":          lead_time,
    }


def render_results(config: dict):
    R = st.session_state["results"]

    if R.get("loaded_from_db"):
        # Loaded runs have no model/encoders/df_featured in memory —
        # skip config-change detection entirely and show a notice instead.
        st.info(
            "Viewing a previously saved run. Sidebar changes won't update "
            "this view — upload data and run the pipeline to forecast live.",
            icon="Info:",
        )
    else:
        # ── Detect config changes ─────────────────────────────────────────────
        # Use .get() with sidebar defaults so missing keys never trigger a false alarm
        retrain_changed = (
            config["model_choice"] != R.get("model_choice", config["model_choice"])
            or config["test_days"] != R.get("test_days", config["test_days"])
        )
        reforecast_changed = (
            not retrain_changed and (
                config["horizon_days"]  != R.get("horizon_days",  config["horizon_days"])
                or config["service_level"] != R.get("service_level", config["service_level"])
                or config["lead_time"]     != R.get("lead_time",     config["lead_time"])
            )
        )

        if retrain_changed:
            st.warning(
                "⚠️ **Algorithm** or **Test window** changed. "
                "Click **Run Forecasting Pipeline** to retrain the model.",
                icon="⚠️",
            )

        elif reforecast_changed:
            with st.spinner("Updating forecast for new settings..."):
                R = _reforecast(R, config)
                st.session_state["results"] = R

    # ── KPI cards ─────────────────────────────────────────────────────────────
    summary             = R["summary"]
    forecasting_metrics = R["forecasting_metrics"]
    business_metrics    = R["business_metrics"]

    st.subheader("Executive Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Forecast Demand", f"{summary['total_demand']:,.0f} units")
    k2.metric("Avg Daily Demand",      f"{summary['average_daily_demand']:,.0f} units/day")
    k3.metric("High Risk SKUs",        summary["high_risk_combos"])
    k4.metric("High-Risk Demand Share",f"{summary['stockout_risk_pct']}%")
    k5.metric("Safety Stock",          f"{summary['safety_stock_total']:,.0f}")

    # ── Insight text ──────────────────────────────────────────────────────────
    insight_parts = [
        f"Over the next **{forecasting_metrics['horizon_days']} days**, "
        f"total forecast demand is **{summary['total_demand']:,.0f} units**.",
    ]
    if business_metrics["top_demand_combo"]:
        insight_parts.append(
            f"Highest demand segment: **{business_metrics['top_demand_combo']}**."
        )
    if summary["high_risk_combos"] > 0:
        insight_parts.append(
            f"**{summary['high_risk_combos']} store-department combinations** "
            "are at high stockout risk."
        )
    insight_parts.append(
        f"High-risk demand share: **{summary['stockout_risk_pct']}%** of total forecast demand "
        "comes from high-risk store-department combinations."
    )
    st.info("  ".join(insight_parts))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Forecast", "Inventory", "Explainability", "Model", "Download"
    ])

    with tab1:
        render_forecast_tab(R)
    with tab2:
        render_inventory_tab(R)
    with tab3:
        render_explainability_tab(R)
    with tab4:
        render_model_tab(R)
    with tab5:
        render_download_tab(R)
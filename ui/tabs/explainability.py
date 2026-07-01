import streamlit as st
import plotly.graph_objects as go

from src.error_analysis import compute_error_segments


def _render_segment_table(title, caption, df):
    if df is None or df.empty:
        return
    st.markdown(f"**{title}**")
    st.caption(caption)
    st.dataframe(df, width="stretch", hide_index=True)


def render_explainability_tab(R):

    shap_df = R["shap_df"]

    st.subheader("Feature Importance (SHAP)")
    st.caption(
        "SHAP values show how much each feature contributes to the model's predictions."
    )

    if shap_df is None or shap_df.empty or "mean_abs_shap" not in shap_df.columns:
        st.info(
            "SHAP analysis isn't available for a loaded run — "
            "this view requires the original pipeline run.",
        )
        return

    if len(shap_df) <= 5:
        top_n = len(shap_df)
    else:
        top_n = st.slider(
            "Show top N features",
            min_value=5,
            max_value=min(20, len(shap_df)),
            value=min(10, len(shap_df)),
        )

    shap_plot = (
        shap_df.head(top_n)
        .sort_values("mean_abs_shap")
    )

    top_feature = shap_plot.iloc[-1]["feature"]
    st.caption(
        f"**{top_feature}** is currently the most influential feature for this run — "
        "see the feature category explanations below for what it represents."
    )

    fig = go.Figure(
        go.Bar(
            x=shap_plot["mean_abs_shap"],
            y=shap_plot["feature"],
            orientation="h",
            marker_color="#1f77b4",
        )
    )

    fig.update_layout(
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=max(300, top_n * 28),
        margin=dict(
            t=20,
            l=180,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    with st.expander("Feature Explanation"):

        st.markdown(
            """
**Lag Features**

- `lag_1`, `lag_7`, `lag_14`, `lag_28`
- Sales from previous days.
- Capture short-term demand patterns.

---

**Rolling Statistics**

- `rolling_mean_*`
- Average demand over recent periods.

- `rolling_std_*`
- Demand volatility.
- Higher values usually require more safety stock.

---

**Calendar Features**

- `month_sin`
- `month_cos`
- `wday_sin`
- `wday_cos`

Capture seasonality and weekly patterns.

---

**Price Features**

- `price_change`
- `price_change_pct`

Measure price elasticity and promotional effects.
"""
        )

    # ── Error segmentation ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Error Analysis")
    st.caption(
        "Test-period forecast error broken down by store, day type, and demand "
        "tier — segments with fewer than 10 test-period rows are omitted as "
        "too small to be reliable."
    )

    predictions_df = R.get("predictions_df")
    df_featured = R.get("df_featured")
    inventory_df = R.get("inventory_df")

    if predictions_df is None or predictions_df.empty or "date" not in predictions_df.columns:
        st.info(
            "Error analysis isn't available for a loaded run — "
            "this view requires the original pipeline run.",
        )
        return

    segments = compute_error_segments(predictions_df, df_featured, inventory_df)

    any_segment_shown = False
    col1, col2 = st.columns(2)
    with col1:
        if not segments["by_store"].empty:
            any_segment_shown = True
            _render_segment_table(
                "By Store",
                "Stores with the highest average forecast error.",
                segments["by_store"],
            )
    with col2:
        if not segments["by_weekend"].empty:
            any_segment_shown = True
            _render_segment_table(
                "Weekday vs. Weekend",
                "Whether forecast error differs between weekdays and weekends.",
                segments["by_weekend"],
            )

    if not segments["by_demand_tier"].empty:
        any_segment_shown = True
        _render_segment_table(
            "High-Demand vs. Low-Demand SKUs",
            "Whether the model is more accurate for high-volume or low-volume store/department combinations.",
            segments["by_demand_tier"],
        )

    if not any_segment_shown:
        st.info(
            "Not enough test-period data to compute reliable error segments "
            "for this run (each segment needs at least 10 rows).",
        )
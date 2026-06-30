import streamlit as st
import plotly.graph_objects as go


def render_explainability_tab(R):

    shap_df = R["shap_df"]

    st.subheader("Feature Importance (SHAP)")
    st.caption(
        "SHAP values show how much each feature contributes to the model's predictions."
    )

    if shap_df is None or shap_df.empty or "mean_abs_shap" not in shap_df.columns:
        st.info(
            "📂 SHAP analysis isn't available for a loaded run — "
            "this view requires the original pipeline run.",
            icon="📂",
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

    with st.expander("📖 Feature Explanation"):

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
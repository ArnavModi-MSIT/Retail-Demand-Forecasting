import streamlit as st
import plotly.graph_objects as go


def render_model_tab(R):

    results_df = R["results_df"]
    evaluation_metrics = R["evaluation_metrics"]

    st.subheader("Model Performance")

    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Selected Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model",
            evaluation_metrics["model_name"],
        )

        st.metric(
            "Forecast Horizon",
            f"{R['horizon_days']} Days",
        )

    with col2:
        st.metric(
            "MAPE",
            f"{evaluation_metrics['mape']:.2f}%"
            if evaluation_metrics["mape"] is not None
            else "—",
        )

        st.metric(
            "Baseline Improvement",
            (
                f"{evaluation_metrics['baseline_improvement']}%"
                if evaluation_metrics["baseline_improvement"] is not None
                else "—"
            ),
        )

    st.divider()

    st.markdown(
        """
### Metric Definitions

- **MAPE** – Mean Absolute Percentage Error. Lower is better.

- **RMSE** – Root Mean Squared Error. Penalizes large forecasting errors.

- **MAE** – Mean Absolute Error. Average prediction error.

- **Baseline Improvement** – Percentage improvement over a naive forecast.
"""
    )

    # ── Residual plots ────────────────────────────────────────────────────────
    predictions_df = R.get("predictions_df")

    st.divider()
    st.subheader("Residual Analysis")

    if predictions_df is None or predictions_df.empty or "actual_sales" not in predictions_df.columns:
        st.info(
            "Residual analysis isn't available for a loaded run — "
            "this view requires the original pipeline run.",
        )
        return

    st.caption(
        "Residual = actual sales minus predicted sales. A well-calibrated model "
        "should show residuals scattered evenly around zero with no obvious "
        "pattern relative to the predicted value."
    )

    residuals = predictions_df["actual_sales"] - predictions_df["predicted_sales"]

    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=predictions_df["predicted_sales"],
            y=residuals,
            mode="markers",
            marker=dict(size=5, color="#1f77b4", opacity=0.5),
            name="Residual",
        ))
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#d62728")
        fig_scatter.update_layout(
            title="Residuals vs. Predicted Sales",
            xaxis_title="Predicted Sales",
            yaxis_title="Residual (Actual − Predicted)",
            height=360,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    with col_b:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            marker_color="#1f77b4",
            nbinsx=40,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#d62728")
        fig_hist.update_layout(
            title="Residual Distribution",
            xaxis_title="Residual (Actual − Predicted)",
            yaxis_title="Count",
            height=360,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_hist, width="stretch")

    mean_residual = float(residuals.mean())
    bias_note = (
        "The model shows negligible systematic bias — the mean residual is close to zero."
        if abs(mean_residual) < 0.05 * predictions_df["actual_sales"].mean()
        else (
            "The model tends to under-predict on average (mean residual is positive)."
            if mean_residual > 0
            else "The model tends to over-predict on average (mean residual is negative)."
        )
    )
    st.caption(f"Mean residual: **{mean_residual:,.2f}**. {bias_note}")
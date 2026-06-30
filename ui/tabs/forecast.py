import streamlit as st
import plotly.graph_objects as go


def render_forecast_tab(R):
    future = R["future_df"]
    preds = R["predictions_df"]
    summary = R["summary"]

    st.subheader("Future Demand Forecast")
    # Store selector
    store_options = ["All stores"]
    if "store_id" in future.columns:
         store_options += sorted(future["store_id"].astype(str).unique().tolist())
    selected_store = st.selectbox("Filter by store", store_options)

    if selected_store != "All stores" and "store_id" in future.columns:
        f_plot = future[future["store_id"].astype(str) == selected_store]
    else:
        # Aggregate across stores — sum predictions, sum bounds
        agg_cols = {"predicted_sales": "sum"}
        if "lower_bound" in future.columns:
            agg_cols["lower_bound"] = "sum"
        if "upper_bound" in future.columns:
            agg_cols["upper_bound"] = "sum"
        f_plot = future.groupby("date").agg(agg_cols).reset_index()

    fig1 = go.Figure()

    if {"upper_bound", "lower_bound"}.issubset(f_plot.columns):

        fig1.add_trace(
            go.Scatter(
                x=f_plot["date"],
                y=f_plot["upper_bound"],
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=f_plot["date"],
                y=f_plot["lower_bound"],
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.2)",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            )
        )
    fig1.add_trace(go.Scatter(
        x=f_plot["date"], y=f_plot["predicted_sales"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#1f77b4", width=2),
    ))
    fig1.update_layout(
        title=f"{summary['horizon_days']}-Day Demand Forecast (95% Confidence Interval)",
        xaxis_title="Date", yaxis_title="Predicted Sales",
        hovermode="x unified", height=380,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig1, width="stretch")

    # Actual vs Predicted (test period)
    st.subheader("Actual vs Predicted (test period)")
    if preds.empty or "date" not in preds.columns:
        st.info(
            "📂 Test-period detail isn't available for a loaded run — "
            "this view requires the original pipeline run.",
            icon="📂",
        )
    else:
        if selected_store != "All stores" and "store_id" in preds.columns:
            p_plot = preds[preds["store_id"].astype(str) == selected_store]
        else:
            p_plot = preds.groupby("date")[["actual_sales", "predicted_sales"]].sum().reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=p_plot["date"], y=p_plot["actual_sales"],    name="Actual",    line=dict(color="#2ca02c")))
        fig2.add_trace(go.Scatter(x=p_plot["date"], y=p_plot["predicted_sales"], name="Predicted", line=dict(color="#1f77b4", dash="dash")))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Sales", hovermode="x unified", height=340, margin=dict(t=20, b=40))
        st.plotly_chart(fig2, width="stretch")
import streamlit as st


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
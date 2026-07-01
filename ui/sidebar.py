import streamlit as st


def render_sidebar():
    """Renders sidebar and returns all configuration values."""

    with st.sidebar:

        st.title("Configuration")

        st.subheader("Quick Settings")

        model_choice = st.selectbox(
            "Algorithm",
            ["LightGBM", "XGBoost", "Random Forest"],
            index=0,
            help=(
                "LightGBM: fastest, recommended for this live demo\n"
                "XGBoost: best accuracy, slower hyperparameter search\n"
                "Random Forest: most interpretable"
            ),
        )

        horizon_days = st.selectbox(
            "Forecast horizon",
            [7, 14, 28, 56],
            index=2,
            help="How many days ahead to forecast.",
        )

        service_level = st.selectbox(
            "Service level",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{int(x*100)}%",
            help="Higher service level = more safety stock.",
        )

        with st.expander("Advanced Options", expanded=False):

            test_days = st.selectbox(
                "Test window (days)",
                [30, 60, 90],
                index=2,
                help="Days reserved for model evaluation.",
            )

            lead_time = st.number_input(
                "Lead time (days)",
                min_value=1,
                max_value=30,
                value=7,
                help="Average supplier lead time.",
            )

        st.divider()

        if st.button("Reset — upload new data", width="stretch"):
            # Preserve session_id — it's the privacy boundary that scopes
            # this browser session's runs. Deleting it would generate a new
            # ID and make all previously saved runs invisible.
            preserved_session_id = st.session_state.get("session_id")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            if preserved_session_id:
                st.session_state["session_id"] = preserved_session_id
            st.rerun()

    return {
        "model_choice":  model_choice,
        "horizon_days":  horizon_days,
        "service_level": service_level,
        "test_days":     test_days,
        "lead_time":     lead_time,
    }
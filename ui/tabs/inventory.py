import streamlit as st
import plotly.express as px


def render_inventory_tab(R):
    inv = R["inventory_df"]

    st.subheader("Inventory Recommendations")

    # KPI cards
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High Risk",
        int(((inv["risk_level"] == "High Risk") | (inv["risk_level"] == "Critical")).sum())
        if "risk_level" in inv.columns else "—",
        delta_color="inverse",
    )

    c2.metric(
        "Medium Risk",
        int((inv["risk_level"] == "Medium Risk").sum())
        if "risk_level" in inv.columns else "—",
    )

    c3.metric(
        "Low Risk",
        int((inv["risk_level"] == "Low Risk").sum())
        if "risk_level" in inv.columns else "—",
        delta_color="off",
    )

    # Risk distribution
    if "risk_level" in inv.columns:

        risk_counts = (
            inv["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Count"]

        fig_pie = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map={
                "Critical":    "#8b0000",
                "High Risk":   "#d62728",
                "Medium Risk": "#ff7f0e",
                "Low Risk":    "#2ca02c",
            },
            hole=0.4,
        )

        fig_pie.update_layout(
            height=300,
            margin=dict(t=20),
        )

        st.plotly_chart(
            fig_pie,
            width="stretch",
        )

    # Inventory table
    if "risk_level" in inv.columns:

        def _color_row(row):
            color_map = {
                "Critical":    "background-color: #ff4d4d",
                "High Risk":   "background-color: #ffcccc",
                "Medium Risk": "background-color: #fff2cc",
                "Low Risk":    "background-color: #ccffcc",
            }
            color = color_map.get(row["risk_level"], "")
            return [color] * len(row)

        styled_df = inv.style.apply(_color_row, axis=1)

        st.dataframe(
            styled_df,
            width="stretch",
        )

    else:

        st.dataframe(
            inv,
            width="stretch",
        )
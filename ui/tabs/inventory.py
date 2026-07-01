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

    st.divider()

    with st.expander("How these numbers are calculated"):
        st.markdown(
            """
**Safety Stock**

    Safety Stock = z × (demand_std × √lead_time_days)

Extra buffer stock held to absorb demand variability during the supplier
lead time. `z` is the service-level factor (1.28 / 1.645 / 2.326 for a
90% / 95% / 99% service level — higher service level, more buffer).
`demand_std` is the standard deviation of daily demand on open-trading
days, scaled up over the lead time window.

---

**Reorder Point**

    Reorder Point = (avg_daily_demand × lead_time_days) + Safety Stock

The stock level at which a new order should be placed — expected demand
during the lead time, plus the safety stock buffer above.

---

**Recommended Inventory**

    Recommended Inventory = forecast_demand + Safety Stock

Total forecast demand over the selected horizon, plus the same safety
stock buffer, to arrive at how much should be on hand going into that
period.

---

**Risk Level**

Risk is classified by **Coefficient of Variation** (CV = demand_std /
avg_daily_demand) on open-trading days only — not a fixed probability
tied to the service level. This means two SKUs at the same service
level can land in different risk tiers if one has far more volatile
demand than the other, rather than every SKU showing an identical
stockout probability.

| CV | Risk Level |
|---|---|
| ≤ 15% | Low Risk |
| 15% – 25% | Medium Risk |
| 25% – 40% | High Risk |
| > 40% | Critical |

**Expected Stockout Probability** is derived from the same CV
(`CV / (CV + z)`) as a relative, per-SKU indicator of volatility risk —
it is a heuristic proxy, not a statistically calibrated probability of
an actual stockout event.
"""
        )
import streamlit as st
from src.pipeline import export_excel

def render_download_tab(R):

    st.subheader("Download Results")

    preds = R["predictions_df"]
    future = R["future_df"]
    inv = R["inventory_df"]
    summary = R["summary"]
    results_df = R["results_df"]
    feat_imp = R["feature_importance_df"]   

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Excel Report**")
        st.caption("All results in one formatted workbook")
        excel_bytes = export_excel(
            summary,
            preds,
            inv,
            feat_imp,
            results_df,
            future_forecast_df=future,
        )
        st.download_button(
            "⬇️ Download Excel (.xlsx)",
            excel_bytes,
            "retail_forecast_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            type="primary",
            key="dl_excel",
        )

    with col2:
        st.markdown("**📄 CSV Downloads**")
        st.caption("Raw data for further analysis")

    st.divider()

    col_pred, col_inv, col_feat, col_future = st.columns(4)
    with col_pred:
        st.download_button(
            "Forecast (.csv)",
            preds.to_csv(index=False).encode(),
            "forecast_predictions.csv", "text/csv",
            width="stretch",
            key="dl_forecast_csv",
        )
    with col_inv:
        st.download_button(
            "Inventory (.csv)",
            inv.to_csv(index=False).encode(),
            "inventory_recommendations.csv", "text/csv",
            width="stretch",
            key="dl_inventory_csv",
        )
    with col_feat:
        st.download_button(
            "Features (.csv)",
            feat_imp.to_csv(index=False).encode(),
            "feature_importance.csv", "text/csv",
            width="stretch",
            key="dl_features_csv",
        )
    with col_future:
        st.download_button(
            "Future Forecast (.csv)",
            future.to_csv(index=False).encode(),
            "future_forecast.csv", "text/csv",
            width="stretch",
            key="dl_future_csv",
        )
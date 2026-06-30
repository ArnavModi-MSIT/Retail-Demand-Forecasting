import os
import logging
import traceback
import pandas as pd

import streamlit as st
from src.session_helper import get_session_id

from src.pipeline import (
    engineer_features,
    train_models,
    run_shap,
    compute_inventory,
    generate_future_forecast,
    generate_executive_summary,
)
from db_layer.repository import (
    save_run,
    save_model_metrics,
    save_forecasts,
    save_inventory,
)
from db_layer.connection import check_connection


def render_running(config):
    cfg = config

    model_choice  = cfg["model_choice"]
    test_days     = cfg["test_days"]
    service_level = cfg["service_level"]
    lead_time     = cfg["lead_time"]
    horizon_days  = cfg["horizon_days"]

    st.subheader("⏳ Processing your data...")

    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(min(pct, 1.0))
        status_text.text(msg)

    try:
        df_mapped = st.session_state["df_mapped"]

        # Extra validation: ensure numeric columns are actually numeric
        if "sales" in df_mapped.columns:
            try:
                df_mapped["sales"] = pd.to_numeric(df_mapped["sales"], errors="coerce")
                if df_mapped["sales"].isna().any():
                    st.error("❌ Some 'sales' values could not be converted to numbers. Check for invalid characters.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error converting sales column: {e}")
                st.stop()

        update_progress(0.05, "Step 1/6: Engineering features...")
        update_progress(0.2, f"Step 2/6: Training {model_choice}... (may take 1-2 minutes)")
        train_result = train_models(
            df_mapped,
            model_choice=model_choice,
            test_days=test_days,
            progress_callback=update_progress,
        )
        df_featured = engineer_features(df_mapped)

        update_progress(0.75, "Step 3/6: Running SHAP analysis...")
        shap_df = run_shap(
            train_result["model"],
            train_result["X_test"],
            model_choice,
        )

        update_progress(0.82, "Step 4/6: Generating future forecast...")
        future_df = generate_future_forecast(
            train_result["model"],
            df_featured,
            train_result["feature_cols"],
            horizon_days=horizon_days,
            X_test=train_result["X_test"],
            y_test=train_result["y_test"],
            confidence_level=0.95,
            encoders=train_result["encoders"],
        )

        update_progress(0.90, "Step 5/6: Computing inventory recommendations...")
        inventory_df = compute_inventory(
            history_df=df_featured,
            future_df=future_df,
            service_level=service_level,
            lead_time_days=lead_time,
        )

        update_progress(0.95, "Step 6/6: Building executive summary...")
        reporting_bundle = generate_executive_summary(
            train_result["predictions_df"],
            inventory_df,
            train_result["results_df"],
            model_choice,
            horizon_days,
            future_forecast_df=future_df,
        )

        # ── Persist to database (best-effort — never blocks the pipeline) ────
        run_id = None
        if check_connection():
            try:
                n_stores = df_mapped["store_id"].nunique() if "store_id" in df_mapped.columns else 1
                n_depts  = df_mapped["dept_id"].nunique()  if "dept_id"  in df_mapped.columns else 1

                run_id = save_run(
                    dataset_name=st.session_state.get("uploaded_filename", "unknown.csv"),
                    n_rows=len(df_mapped),
                    n_stores=int(n_stores),
                    n_depts=int(n_depts),
                    model_choice=model_choice,
                    horizon_days=horizon_days,
                    service_level=service_level,
                    lead_time=lead_time,
                    test_days=test_days,
                    session_id=get_session_id(),
                )
                save_model_metrics(run_id, train_result["results_df"])
                save_forecasts(run_id, future_df)
                save_inventory(run_id, inventory_df)
            except Exception:
                # Persistence failures must never break the user's pipeline run.
                logging.error("DB persistence failed:\n%s", traceback.format_exc())
                run_id = None
        else:
            logging.warning("Database not reachable — skipping persistence for this run.")

        st.session_state["results"] = {
            **train_result,
            "shap_df":            shap_df,
            "inventory_df":       inventory_df,
            "future_df":          future_df,
            "summary":            reporting_bundle["summary"],
            "evaluation_metrics": reporting_bundle["evaluation_metrics"],
            "forecasting_metrics":reporting_bundle["forecasting_metrics"],
            "business_metrics":   reporting_bundle["business_metrics"],
            "df_featured":        df_featured,
            "model_choice":       model_choice,
            "test_days":          test_days,
            "service_level":      service_level,
            "lead_time":          lead_time,
            "horizon_days":       horizon_days,
            "run_id":             run_id,
        }
        st.session_state["stage"] = "results"
        update_progress(1.0, "Done! 🎉")
        st.rerun()

    except Exception as e:
        # Create logs directory if needed
        os.makedirs("logs", exist_ok=True)

        # Configure logger
        logging.basicConfig(
            filename="logs/app.log",
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        # Save full traceback
        logging.error(traceback.format_exc())

        st.exception(e)
        st.error("❌ Forecast generation failed.")
        st.info("""
Possible reasons:

• Uploaded dataset is missing required information

• Model training did not complete successfully

• Forecast generation encountered an unexpected issue

Please try again or upload another dataset.
""")

        if st.button("← Back to Upload"):
            st.session_state["stage"] = "upload"
            st.rerun()
# ── pipeline.py ───────────────────────────────────────────────────────────────
# Thin orchestrator. Import everything from individual modules so any file
# in the project can do:
#
#   from src.pipeline import train_models, engineer_features, ...
#
# or import directly from the specific module:
#
#   from src.train import train_models
#   from src.models import fit_model, compare_feature_sets
#   from src.evaluate import calculate_mape, run_shap
#   from src.splitter import split_time_series
#   from src.preprocessing import prepare_training_frame
# ─────────────────────────────────────────────────────────────────────────────

from src.dataset import (
    detect_columns,
    apply_column_mapping,
    validate_data,
    detect_sales_type,
    REQUIRED_COLS,
    OPTIONAL_COLS,
)

from src.features import engineer_features

from src.preprocessing import prepare_training_frame

from src.splitter import split_time_series

from src.models import fit_model, compare_feature_sets

from src.evaluate import calculate_mape, evaluate_metrics, run_shap

from src.train import train_models

from src.inventory import compute_inventory

from src.future_forecast import (
    calculate_forecast_confidence,
    generate_future_forecast,
)

from src.reporting import (
    generate_executive_summary,
    export_excel,
)

__all__ = [
    # dataset
    "detect_columns",
    "apply_column_mapping",
    "validate_data",
    "detect_sales_type",
    "REQUIRED_COLS",
    "OPTIONAL_COLS",
    # features
    "engineer_features",
    # preprocessing
    "prepare_training_frame",
    # splitter
    "split_time_series",
    # models
    "fit_model",
    "compare_feature_sets",
    # evaluate
    "calculate_mape",
    "evaluate_metrics",
    "run_shap",
    # train (orchestrator)
    "train_models",
    # inventory
    "compute_inventory",
    # future_forecast
    "calculate_forecast_confidence",
    "generate_future_forecast",
    # reporting
    "generate_executive_summary",
    "export_excel",
]
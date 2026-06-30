import streamlit as st
import sys
import os

from ui.sidebar import render_sidebar
from ui.upload import render_upload
from ui.running import render_running
from ui.results import render_results

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

for key in ["results", "stage"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["stage"] is None:
    st.session_state["stage"] = "upload"


# ── Sidebar Configuration ─────────────────────────────────────────────────────

config = render_sidebar()

model_choice = config["model_choice"]
horizon_days = config["horizon_days"]
service_level = config["service_level"]
test_days = config["test_days"]
lead_time = config["lead_time"]
# ── Header ────────────────────────────────────────────────────────────────────

st.title("📦 Retail Demand Forecasting")
st.caption("Upload CSV → Validate → Train → Forecast → Download insights")

# ── Stage 1: Upload ───────────────────────────────────────────────────────────

if st.session_state["stage"] == "upload":
    render_upload()



# ── Stage 2: Running pipeline ─────────────────────────────────────────────────
elif st.session_state["stage"] == "running":
    render_running(config)
# ── Stage 3: Results ──────────────────────────────────────────────────────────

elif st.session_state["stage"] == "results":
    render_results(config)
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

import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — reduces Streamlit's oversized default scale ─────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
    h1 { font-size: 1.6rem !important; }
    h2, h3 { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem !important; }
    .stMarkdown p, .stCaption { font-size: 0.85rem; }
    button[data-baseweb="tab"] { font-size: 0.85rem !important; }
    button[kind="primary"], button[kind="secondary"] { font-size: 0.85rem !important; }
    [data-testid="stDataFrame"] { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

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
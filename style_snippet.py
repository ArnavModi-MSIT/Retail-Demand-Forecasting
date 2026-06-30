# ── Custom CSS to reduce Streamlit's oversized default component scale ───────
# Place this near the top of app.py, right after st.set_page_config(...).

CUSTOM_CSS = """
<style>
    /* Overall content width and padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Page title (st.title) — default is very large */
    h1 {
        font-size: 1.9rem !important;
    }

    /* Section headers (st.subheader) */
    h2, h3 {
        font-size: 1.3rem !important;
    }

    /* KPI metric cards — value and label */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        font-size: 1.4rem !important;
    }

    /* General body text */
    .stMarkdown p {
        font-size: 0.95rem;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
    }
</style>
"""

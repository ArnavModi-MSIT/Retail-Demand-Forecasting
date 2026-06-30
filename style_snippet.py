# ── Custom CSS to reduce Streamlit's oversized default component scale ───────
# Place this near the top of app.py, right after st.set_page_config(...).

CUSTOM_CSS = """
<style>
    /* Overall content width and padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Page title (st.title) — default is very large */
    h1 {
        font-size: 1.6rem !important;
    }

    /* Section headers (st.subheader) */
    h2, h3 {
        font-size: 1.1rem !important;
    }

    /* KPI metric cards — value and label */
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
    }

    /* General body text */
    .stMarkdown p, .stCaption {
        font-size: 0.85rem;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.85rem !important;
    }

    /* Buttons */
    button[kind="primary"], button[kind="secondary"] {
        font-size: 0.85rem !important;
    }

    /* Dataframe / table text */
    [data-testid="stDataFrame"] {
        font-size: 0.8rem;
    }
</style>
"""
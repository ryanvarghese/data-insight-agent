"""
app.py — Entry point for the Data-to-Insight Agent.

Intentionally minimal. Only handles page config
and top-level layout routing.

All business logic  → core/
All rendering logic → ui/
"""

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from ui.insights_tab import render_insights_tab
from ui.query_tab import render_query_tab
from ui.sidebar import render_sidebar

load_dotenv()

st.set_page_config(page_title="Data-to-Insight Agent", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f1117;
            color: #e8eaf0;
        }
        [data-testid="stSidebar"] {
            background-color: #1c2230;
            color: #e8eaf0;
        }
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #e8eaf0 !important;
        }
        .card {
            background-color: #1c2230;
            border: 1px solid #2b3347;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .accent-blue {
            color: #4f8ef7;
        }
        .accent-green {
            color: #3fb950;
        }
        .accent-red {
            color: #f78166;
        }
        .insight-card {
            background-color: #1c2230;
            border: 1px solid #2b3347;
            border-left: 4px solid #4f8ef7;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #e8eaf0;
        }
        .rec-card {
            background-color: #1c2230;
            border: 1px solid #2b3347;
            border-left: 4px solid #3fb950;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #e8eaf0;
        }
        .anomaly-card {
            background-color: #1c2230;
            border: 1px solid #2b3347;
            border-left: 4px solid #f78166;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #e8eaf0;
        }
        .summary-file-name {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .summary-line {
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            color: #cfd5e3;
        }
        .badge-row {
            margin-bottom: 0.4rem;
        }
        .sidebar-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            margin: 0 0.25rem 0.25rem 0;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .badge-num { background-color: rgba(79, 142, 247, 0.2); color: #9fc2ff; border: 1px solid #4f8ef7; }
        .badge-cat { background-color: rgba(63, 185, 80, 0.2); color: #8bdf96; border: 1px solid #3fb950; }
        .badge-dt { background-color: rgba(247, 166, 79, 0.2); color: #ffc17f; border: 1px solid #f7a64f; }
        .alert-badge {
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
            margin-bottom: 0.5rem;
            font-size: 0.84rem;
            font-weight: 700;
        }
        .alert-danger {
            background-color: rgba(247, 129, 102, 0.18);
            border: 1px solid #f78166;
            color: #ffb8a8;
        }
        .alert-success {
            background-color: rgba(63, 185, 80, 0.18);
            border: 1px solid #3fb950;
            color: #9be5a7;
        }
        .badge-legend-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-top: 0.35rem;
            flex-wrap: wrap;
        }
        .badge-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 11px;
            color: #8b949e;
        }
        .badge-legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .legend-num { background-color: #1f6feb; }
        .legend-cat { background-color: #196c2e; }
        .legend-dt { background-color: #9e6a03; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "anthropic_client" not in st.session_state:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    st.session_state["anthropic_client"] = (
        anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
    )

render_sidebar()

st.title("AI-Powered Data Analysis Agent")
st.caption("Upload your dataset and explore insights through AI assistance.")

if "schema" not in st.session_state:
    st.markdown(
        """
        <div class="card">
            <h3 class="accent-blue">Ready to analyze your data</h3>
            <p>Please upload a CSV or Excel file from the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

insights_tab, ask_tab = st.tabs(["Insights", "Ask a Question"])
with insights_tab:
    render_insights_tab()
with ask_tab:
    render_query_tab()

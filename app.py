"""SIGAP - aplikasi utama (router + sidebar). Jalankan dari root proyek: streamlit run app.py

Halaman: 1 Overview, 2 Air Quality & Spatial Distribution, 3 Pollution Patterns & Drivers,
4 Public Health Impact & Vulnerability, 5 Data Analysis (modul di views/, hitungan di utils/).
"""
import base64
import os
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_DIR = str(BASE / "data")
LOGO = BASE / "assets" / "logo_gelar_rasa.png"
os.environ.setdefault("SIGAP_DATA_DIR", DATA_DIR)  # dibaca utils/air_quality_data (Halaman 2)

from views.page1_overview import render_overview_page  # noqa: E402
from views.page2_air_quality import render_air_quality_page  # noqa: E402
from views.page3_pollution_patterns import render_pollution_patterns_page  # noqa: E402
from views.page4_public_health import render_public_health_page  # noqa: E402
from views.page5_data_analysis import render_data_analysis_page  # noqa: E402

st.set_page_config(page_title="SIGAP", layout="wide", initial_sidebar_state="expanded")


PAGES = [
    ("Overview", lambda: render_overview_page(data_dir=DATA_DIR)),
    ("Air Quality & Spatial Distribution", render_air_quality_page),
    ("Pollution Patterns & Drivers", lambda: render_pollution_patterns_page(data_dir=DATA_DIR)),
    ("Public Health Impact & Vulnerability", lambda: render_public_health_page(data_dir=DATA_DIR)),
    ("Data Analysis", lambda: render_data_analysis_page(data_dir=DATA_DIR)),
]
NAMES = [n for n, _ in PAGES]


@st.cache_data
def _logo_html() -> str:
    """Logo Gelar Rasa dalam kartu putih (logo bertulisan hitam, tidak terbaca di latar navy)."""
    if not LOGO.exists():
        return ""
    b64 = base64.b64encode(LOGO.read_bytes()).decode()
    return f'<div class="sg-logo"><img src="data:image/png;base64,{b64}" alt="Gelar Rasa"></div>'


st.markdown("""<style>
:root { color-scheme: light; }
.stApp, [data-testid="stApp"] { background: #F3F6FC; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 20px 14px; }
section[data-testid="stSidebar"] { width: 250px !important; min-width: 250px !important; background: #10264A; }
section[data-testid="stSidebar"] > div { width: 250px; }
/* section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none; } */
section[data-testid="stSidebar"] .sg-logo { background: #fff; border-radius: 10px; padding: 8px 12px; margin-bottom: 14px; }
section[data-testid="stSidebar"] .sg-logo img { width: 100%; display: block; }
section[data-testid="stSidebar"] .sg-brand { font-size: 22px; font-weight: 700; color: #fff; line-height: 28px; }
section[data-testid="stSidebar"] .sg-sub { font-size: 12px; color: #C7D4EA; margin-bottom: 14px; }
[class*="st-key-nav_"] button { width: 100%; justify-content: flex-start; text-align: left; background: transparent;
    border: none; color: #fff; font-size: 13px; padding: 9px 12px; border-radius: 8px; min-height: 0; }
[class*="st-key-nav_"] button:hover { background: #263E60; color: #fff; }
[class*="st-key-nav_"] button[kind="primary"],
[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] { background: #2563EB; font-weight: 600; }
[class*="st-key-nav_"] button p { color: #fff; text-align: left; }
[class*="st-key-nav_"] button { justify-content: flex-start !important; }
[class*="st-key-nav_"] button > div,
[class*="st-key-nav_"] button [data-testid="stMarkdownContainer"] { width: 100%; justify-content: flex-start; text-align: left; }
</style>""", unsafe_allow_html=True)

st.session_state.setdefault("page", NAMES[0])  # default: Halaman 1 (Overview)

with st.sidebar:
    st.markdown(_logo_html(), unsafe_allow_html=True)
    st.markdown('<div class="sg-brand">SIGAP</div>'
                '<div class="sg-sub">Air Quality &amp; Public Health Dashboard</div>', unsafe_allow_html=True)
    for i, name in enumerate(NAMES):
        active = st.session_state["page"] == name
        if st.button(name, key=f"nav_{i}", type="primary" if active else "secondary",
                     use_container_width=True) and not active:
            st.session_state["page"] = name
            st.rerun()

dict(PAGES)[st.session_state["page"]]()

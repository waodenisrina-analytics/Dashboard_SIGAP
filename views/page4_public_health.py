"""
SIGAP - Halaman 4: Public Health Impact & Vulnerability.

Pemakaian dari aplikasi utama:
    from views.page4_public_health import render_public_health_page
    render_public_health_page(data_dir="data")            # filter global dirender di header halaman
    render_public_health_page(data_dir="data", filters=f)  # atau pakai filter global milik app utama
        # f = {"date_range": (date, date), "provinsi": [...], "site_group": [...]}  (kosong/None = semua)

Fungsi ini tidak memanggil st.set_page_config() dan tidak membuat sidebar.
Semua key widget/CSS berawalan "p4_" agar tidak bentrok dengan halaman lain.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import health_utils as U

# ---- Design tokens (Master Design Specification) --------------------------
NAVY, NAVY2, BG = "#10264A", "#263E60", "#F3F6FC"
CARD, ACCENT, BORDER = "#FFFFFF", "#2563EB", "#DCE6F5"
TEXT, TEXT2 = "#102044", "#64748B"
C_CASE, C_MUTED = "#F59E0B", "#94A3B8"
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

# ---- Ukuran (target viewport 1440x900; sesuaikan di sini bila perlu) ------
PANEL_H = 328   # tinggi panel grafik (px)
CHART_H = 252   # tinggi plot di dalam panel (px)
KPI_H = 96      # tinggi metric card (px)
PLOT_CFG = {"displayModeBar": False, "responsive": True}

OUTCOME_OPTS = {"Pernapasan": "resp", "Kardiovaskular": "cvd", "Asma (IGD)": "asthma_emergency_cases",
                "PPOK": "copd_admissions", "Kematian": "death"}


# --------------------------------------------------------------------------
# Data & CSS
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(data_dir: str):
    return U.load_data(data_dir)


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.block-container {{padding: 24px !important; max-width: none !important;}}
header[data-testid="stHeader"] {{display: none;}}
.st-key-p4_root {{font-family: {FONT}; color: {TEXT};}}
.st-key-p4_root [data-testid="stVerticalBlock"] {{gap: 12px;}}
.p4-title {{font-size: 27px; font-weight: 700; color: {TEXT}; line-height: 1.2;}}
.p4-sub {{font-size: 13px; color: {TEXT2}; margin-top: 2px;}}
.p4-ptitle {{font-size: 16px; font-weight: 600; color: {TEXT}; line-height: 1.25;}}
.p4-psub {{font-size: 11px; color: {TEXT2}; margin-top: 1px;}}
[class*="st-key-p4_kpi"], [class*="st-key-p4_panel"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px;}}
[class*="st-key-p4_kpi"] {{height: {KPI_H}px; overflow: hidden;}}
[class*="st-key-p4_panel"] {{height: {PANEL_H}px;}}
[class*="st-key-p4_panel"] [data-testid="stVerticalBlock"] {{gap: 4px;}}
.st-key-p4_root [data-testid="stMetricLabel"] {{font-size: 13px; color: {TEXT2}; font-family: {FONT};}}
.st-key-p4_root [data-testid="stMetricValue"] {{font-size: 28px; font-weight: 700; color: {TEXT}; font-family: {FONT};}}
.st-key-p4_root [data-testid="stMetricDelta"] {{font-size: 12px; color: {TEXT2} !important; font-family: {FONT};}}
.st-key-p4_root [data-testid="stMetricDelta"] svg {{display: none;}}
.st-key-p4_root [data-baseweb="select"] > div {{min-height: 32px; font-size: 12px;}}
.st-key-p4_root [data-testid="stRadio"] label p {{font-size: 12px;}}
.st-key-p4_root [data-testid="stDateInput"] input {{font-size: 12px;}}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header + filter global
# --------------------------------------------------------------------------
def _header(h_all: pd.DataFrame, filters: dict | None) -> dict:
    title = ('<div class="p4-title">Public Health Impact &amp; Vulnerability</div>'
             '<div class="p4-sub">Analisis dampak PM2.5 terhadap kesehatan masyarakat '
             'dan kerentanan kelompok populasi.</div>')
    if filters is not None:
        st.markdown(title, unsafe_allow_html=True)
        return {"date_range": filters.get("date_range"), "provinsi": filters.get("provinsi"),
                "site_group": filters.get("site_group")}

    c_t, c_d, c_p, c_s = st.columns([4.0, 2.4, 2.0, 2.0], vertical_alignment="center")
    c_t.markdown(title, unsafe_allow_html=True)
    dmin, dmax = h_all["date"].min().date(), h_all["date"].max().date()
    dr = c_d.date_input("Periode", (dmin, dmax), min_value=dmin, max_value=dmax,
                        key="p4_f_date", label_visibility="collapsed")
    if not (isinstance(dr, (tuple, list)) and len(dr) == 2):  # rentang belum lengkap -> pakai penuh
        dr = (dmin, dmax)
    prov = c_p.multiselect("Provinsi", sorted(h_all["provinsi"].unique()), key="p4_f_prov",
                           placeholder="Semua provinsi", label_visibility="collapsed")
    site = c_s.multiselect("Tipe lokasi", sorted(h_all["site_type_grup"].unique()), key="p4_f_site",
                           placeholder="Semua tipe lokasi", label_visibility="collapsed")
    return {"date_range": dr, "provinsi": prov, "site_group": site}


# --------------------------------------------------------------------------
# Helper UI
# --------------------------------------------------------------------------
def _show(fig: go.Figure, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)
    except TypeError:  # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key=key)


def _base_layout(fig: go.Figure, **kw) -> go.Figure:
    # Gabungkan margin default, bisa ditimpa atau ditambah dari **kw jika ada
    margin_cfg = dict(l=8, r=8, t=24, b=8)
    if "margin" in kw:
        margin_cfg.update(kw.pop("margin"))
        
    fig.update_layout(
        height=CHART_H, 
        margin=margin_cfg, 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(family=FONT, size=11, color=TEXT),
        legend=dict(
            orientation="h", 
            y=1.18, 
            x=0, 
            font=dict(size=11, color=TEXT, weight=600), 
            bgcolor="rgba(255, 255, 255, 0.7)", 
            bordercolor=BORDER,
            borderwidth=1
        ), 
        **kw
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=13, color=TEXT2))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _base_layout(fig)


def _panel_head(title: str, sub: str, widths: list[float]):
    cols = st.columns(widths, vertical_alignment="center")
    cols[0].markdown(f'<div class="p4-ptitle">{title}</div><div class="p4-psub">{sub}</div>',
                     unsafe_allow_html=True)
    return cols[1:]


# --------------------------------------------------------------------------
# Komponen 1: Metric cards
# --------------------------------------------------------------------------
def _kpi_row(h: pd.DataFrame) -> None:
    t = U.health_totals(h)
    cards = [
        ("Rawat inap pernapasan balita", _fmt(t["under5"]), f"{t['under5_rate']:.1f} kasus per RS-hari",
         "Jumlah resp_admissions_under5 seluruh RS terpilih pada periode filter."),
        ("Rawat inap pernapasan lansia", _fmt(t["over60"]), f"{t['over60_rate']:.1f} kasus per RS-hari",
         "Jumlah resp_admissions_over60 seluruh RS terpilih pada periode filter."),
        ("Asma IGD + PPOK rawat inap", _fmt(t["asthma"] + t["copd"]),
         f"Asma {_fmt(t['asthma'])} · PPOK {_fmt(t['copd'])}",
         "asthma_emergency_cases + copd_admissions. Rincian ditampilkan di bawah angka."),
        ("Kasus kardiovaskular (CVD)", _fmt(t["cvd"]), f"Kematian {_fmt(t['death'])}",
         "Nilai utama = cvd. Kematian = death (definisi penyebab tidak didokumentasikan, "
         "tidak dijumlahkan dengan cvd karena satuannya berbeda)."),
    ]
    for i, (col, (label, value, sub, hlp)) in enumerate(zip(st.columns(4), cards)):
        with col, st.container(key=f"p4_kpi_{i}"):
            st.metric(label, value, sub, delta_color="off", help=hlp)


# --------------------------------------------------------------------------
# Komponen 2: PM2.5 vs Health Trend
# --------------------------------------------------------------------------
def _trend_panel(am: pd.DataFrame) -> None:
    with st.container(key="p4_panel_trend"):
        c1, c2 = _panel_head("PM2.5 vs Health Trend",
                             f"Rata-rata stasiun vs total kasus · {am['hospital_id'].nunique()} RS", [2.3, 1.5, 1.1])
        pm_label = c1.selectbox("Seri PM2.5", list(U.PM_SERIES), key="p4_trend_pm", label_visibility="collapsed")
        out_label = c2.selectbox("Kasus", ["Pernapasan", "Kardiovaskular"], key="p4_trend_out",
                                 label_visibility="collapsed")
        if am.empty:
            _show(_empty_fig("Tidak ada data pada filter ini"), "p4_trend_fig"); return
        d = U.daily_trend(am, U.PM_SERIES[pm_label], OUTCOME_OPTS[out_label])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=d["date"], y=d["pm"], name=pm_label, mode="lines",
                                 line=dict(color=ACCENT, width=1.6),
                                 hovertemplate="%{y:.1f} µg/m³"), secondary_y=False)
        fig.add_trace(go.Scatter(x=d["date"], y=d["cases"], name=f"Kasus {out_label.lower()} (total/hari)",
                                 mode="lines", line=dict(color=C_CASE, width=1.2),
                                 hovertemplate="%{y:,.0f} kasus"), secondary_y=True)
        _base_layout(fig, hovermode="x unified")
        fig.update_yaxes(title_text="PM2.5 (µg/m³)", secondary_y=False, gridcolor=BORDER, zeroline=False,
                       title_font=dict(size=10, color=TEXT), tickfont=dict(size=10, color=TEXT))
        fig.update_yaxes(title_text="Kasus/hari", secondary_y=True, showgrid=False,
                       title_font=dict(size=10, color=TEXT), tickfont=dict(size=10, color=TEXT))
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color=TEXT))
        _show(fig, "p4_trend_fig")


# --------------------------------------------------------------------------
# Komponen 3: Lag correlation heatmap
# --------------------------------------------------------------------------
def _heatmap_panel(am: pd.DataFrame) -> None:
    with st.container(key="p4_panel_lag"):
        (c1,) = _panel_head("Lag Correlation Heatmap",
                            f"Within-station · {am['hospital_id'].nunique()} RS", [2.3, 2.1])
        method = c1.radio("Metode", ["Pearson", "Spearman"], horizontal=True, key="p4_lag_method",
                          label_visibility="collapsed").lower()
        if len(am) < 30:
            _show(_empty_fig("Data tidak cukup pada filter ini"), "p4_lag_fig"); return
        corr = U.lag_correlation(am, method)
        ylab = [U.OUTCOME_LABELS[o] for o in corr.index]
        fig = go.Figure(go.Heatmap(z=corr.values, x=U.LAG_LABELS, y=ylab, zmin=-1, zmax=1, zmid=0,
                                   colorscale="RdBu", texttemplate="%{z:.2f}", textfont=dict(size=11),
                                   xgap=2, ygap=2, hovertemplate="%{y} × %{x}: r = %{z:.3f}<extra></extra>",
                                   colorbar=dict(thickness=8, len=0.9, tickfont=dict(size=9))))
        best = U.best_lag(corr)   # kotak tebal = lag dengan korelasi tertinggi tiap outcome
        for i, j in enumerate(best.values):
            fig.add_shape(type="rect", x0=j - .5, x1=j + .5, y0=i - .5, y1=i + .5,
                          line=dict(color=NAVY, width=2))
        _base_layout(fig)
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10, color=TEXT))
        fig.update_xaxes(tickfont=dict(size=10, color=TEXT))
        _show(fig, "p4_lag_fig")


# --------------------------------------------------------------------------
# Komponen 4: Vulnerability comparison
# --------------------------------------------------------------------------
def _vulnerability_panel(h: pd.DataFrame) -> None:
    with st.container(key="p4_panel_vuln"):
        (c1,) = _panel_head("Vulnerability Comparison",
                            f"Rata-rata kasus per RS-hari · {h['hospital_id'].nunique()} RS", [2.3, 2.1])
        dim_label = c1.selectbox("Dimensi", ["Tipe lokasi", "Provinsi"], key="p4_vuln_dim",
                                 label_visibility="collapsed")
        dim = {"Tipe lokasi": "site_type_grup", "Provinsi": "provinsi"}[dim_label]
        g = U.vulnerability_summary(h, dim)
        fig = go.Figure()
        for col, name, color in [("under5", "Balita (<5 th)", ACCENT), ("over60", "Lansia (>60 th)", NAVY)]:
            fig.add_trace(go.Bar(x=g[dim], y=g[col], name=name, marker_color=color, customdata=g["n_rs"],
                                 hovertemplate="%{x}<br>%{y:.1f} kasus/RS-hari · %{customdata} RS<extra>" + name + "</extra>"))
        _base_layout(fig, barmode="group", bargap=0.25)
        fig.update_yaxes(title_text="Kasus per RS-hari", gridcolor=BORDER, 
                       title_font=dict(size=10, color=TEXT), tickfont=dict(size=10, color=TEXT))
        fig.update_xaxes(tickangle=-50 if dim == "provinsi" else 0,
                       tickfont=dict(size=9 if dim == "provinsi" else 11, color=TEXT))
        _show(fig, "p4_vulnerability_fig") # sesuaikan dengan key chart Anda


# --------------------------------------------------------------------------
# Komponen 5: Hospital-level vulnerability
# --------------------------------------------------------------------------
def _hospital_panel(am: pd.DataFrame) -> None:
    with st.container(key="p4_panel_hosp"):
        c1, c2 = _panel_head("Hospital-Level Vulnerability",
                             f"Slope PM2.5 per RS · {am['hospital_id'].nunique()} RS", [2.3, 1.2, 1.2])
        out_label = c1.selectbox("Outcome", list(OUTCOME_OPTS), key="p4_hosp_out", label_visibility="collapsed")
        unit = c2.selectbox("Satuan", ["Kasus/hari", "% rata-rata"], key="p4_hosp_unit",
                            label_visibility="collapsed")
        s = U.hospital_sensitivity(am, OUTCOME_OPTS[out_label])
        if s.empty:
            _show(_empty_fig(f"Data tidak cukup (min. {U.MIN_PAIRS} hari PM2.5 valid per RS)"), "p4_hosp_fig"); return
        v, lo, hi = (("delta_per_10ug", "ci_lo", "ci_hi") if unit == "Kasus/hari"
                     else ("delta_pct", "pct_lo", "pct_hi"))
        s = s.sort_values(v, ascending=False)
        sig = s[lo] > 0
        fig = go.Figure()
        for mask, name, color in [(sig, "CI 95% di atas 0", ACCENT), (~sig, "CI 95% mencakup/di bawah 0", C_MUTED)]:
            d = s[mask]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["kode"], y=d[v], mode="markers", name=name, marker=dict(color=color, size=8),
                error_y=dict(type="data", symmetric=False, array=d[hi] - d[v], arrayminus=d[v] - d[lo],
                             color=color, thickness=1.2, width=0),
                customdata=d[["rs_nama", "hospital_id", "n"]].to_numpy(),
                hovertemplate="%{customdata[0]} (%{customdata[1]})<br>%{y:.2f} per +10 µg/m³<br>n = %{customdata[2]} hari<extra></extra>"))
        
        fig.add_hline(y=0, line=dict(color=TEXT2, width=1))
        _base_layout(fig, margin=dict(l=8, r=8, t=24, b=45))
        
        # Ubah weight menjadi integer (500) tanpa tanda kutip
        fig.update_xaxes(
            tickangle=-45, 
            tickfont=dict(size=10, color=TEXT, weight=500), 
            categoryorder="array", 
            categoryarray=list(s["kode"])
        )
        
        fig.update_yaxes(title_text=("Δ kasus/hari" if unit == "Kasus/hari" else "Δ % dari rata-rata") + " per +10 µg/m³",
                         gridcolor=BORDER, zeroline=False, title_font=dict(size=10, color=TEXT2))
        _show(fig, "p4_hosp_fig")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def render_public_health_page(data_dir: str = "data", filters: dict | None = None) -> None:
    """Render Halaman 4. `data_dir` berisi analysis_daily.csv, fact_health_daily.csv, dim_station.csv."""
    _inject_css()
    h_all, a_all = _load(str(data_dir))
    with st.container(key="p4_root"):
        flt = _header(h_all, filters)
        h = U.apply_filters(h_all, **flt)     # semua RS (27) -> kartu & kerentanan
        a = U.apply_filters(a_all, **flt)
        am = U.model_stations(a)              # stasiun 'model penuh' -> panel yang memakai PM2.5
        if h.empty:
            st.info("Tidak ada data untuk kombinasi filter ini. Ubah periode, provinsi, atau tipe lokasi.")
            return
        _kpi_row(h)
        left, right = st.columns(2)
        with left:
            _trend_panel(am)
        with right:
            _heatmap_panel(am)
        left, right = st.columns(2)
        with left:
            _vulnerability_panel(h)
        with right:
            _hospital_panel(am)

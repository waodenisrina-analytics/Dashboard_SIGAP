"""Halaman 2 SIGAP: Air Quality & Spatial Distribution.

Titik masuk: ``render_air_quality_page()``. Halaman ini tidak membuat konfigurasi
global (``st.set_page_config``), sidebar, maupun halaman lain; semua itu milik
aplikasi utama. Target tampilan: desktop 1440 x 900 px, konten 1142 px, tanpa scroll.

Aturan data (26 stasiun, hari valid, baku mutu, ISPU, IKU indikatif, agregasi bobot
sama per stasiun) didokumentasikan di ``utils/air_quality_data.py``.

Semua key widget/container berawalan ``p2_`` agar tidak bentrok dengan halaman lain.
"""
from __future__ import annotations

import inspect

import streamlit as st

from utils.air_quality_charts import (
    exceedance_figure, iku_box_figure, ispu_figure, map_figure, ranking_figure, trend_figure,
)
from utils.air_quality_data import (
    IKU_BAIK, POLLUTANT_ORDER, POLLUTANTS, build_kpis, daily_trend, date_bounds, exceedance,
    iku_station_year, ispu_distribution, load_stations, station_metrics,
)
from utils.common import fmt

METRICS = ["Konsentrasi", "IKU", "ISPU"]

# Anggaran tinggi (px): 24 + 56 header + 58 filter + 98 KPI + 2 x 285 grafik
# + 4 x 12 jarak + 24 = 878 (<= 900).
FILTER_H, KPI_H, CARD_H = 58, 98, 285
CHART_H = 196  # tinggi grafik di dalam kartu (kartu 285 - header panel 52 - padding)
MAP_H = 190    # peta sedikit lebih rendah: header panelnya memuat toggle

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stApp"], [data-testid="stMain"] { font-family: 'Inter', sans-serif; }
[data-testid="stApp"] { background: #F3F6FC; }
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"], .block-container {
    padding: 24px !important; max-width: 100% !important; }
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: 12px; }
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] { gap: 12px; }

[class*="st-key-p2_card"] {
    background: #FFFFFF; border: 1px solid #DCE6F5 !important; border-radius: 12px;
    padding: 8px 10px !important; box-sizing: border-box; }
[class*="st-key-p2_card"] [data-testid="stVerticalBlock"] { gap: 0; }
[class*="st-key-p2_card"] [data-testid="stHorizontalBlock"] { gap: 8px; align-items: center; }

.p2-header { height: 56px; }
.p2-title { font-size: 26px; font-weight: 700; line-height: 32px; color: #102044; }
.p2-sub { font-size: 13px; line-height: 20px; color: #64748B; }
.p2-fl { height: 40px; display: flex; align-items: center; font-size: 12px;
         font-weight: 600; color: #64748B; }
.p2-note { height: 40px; display: flex; align-items: center; justify-content: flex-end;
           font-size: 11px; color: #64748B; text-align: right; }
.p2-ph { height: 52px; overflow: hidden; }
.p2-pt { font-size: 16px; font-weight: 700; line-height: 20px; color: #102044; }
.p2-pc { font-size: 11px; line-height: 13px; color: #64748B; }

[class*="st-key-p2_card_kpi"] [data-testid="stMetric"] { gap: 0; }
[class*="st-key-p2_card_kpi"] [data-testid="stMetricLabel"],
[class*="st-key-p2_card_kpi"] [data-testid="stMetricLabel"] p {
    font-size: 12px; font-weight: 600; color: #64748B; }
[class*="st-key-p2_card_kpi"] [data-testid="stMetricValue"] {
    font-size: 26px; font-weight: 700; line-height: 32px; color: #102044; }
[class*="st-key-p2_card_kpi"] [data-testid="stMetricDelta"],
[class*="st-key-p2_card_kpi"] [data-testid="stMetricDelta"] * {
    font-size: 11px; color: #64748B !important; }
[data-testid="stMetricDeltaIcon-Up"], [data-testid="stMetricDeltaIcon-Down"] { display: none; }
.p2-k5-label { font-size: 12px; font-weight: 600; line-height: 16px; color: #64748B; }
.p2-k5-row { font-size: 13px; line-height: 22px; color: #102044; }
.p2-k5-row b { font-weight: 700; }

.st-key-p2_metric button, .st-key-p2_metric label {
    font-size: 11px !important; min-height: 26px !important; padding: 0 8px !important; }
</style>
"""


# --------------------------------------------------------------------------
# Helper UI
# --------------------------------------------------------------------------
def _stretch_kw(fn) -> dict:
    """Streamlit baru memakai width='stretch'; yang lama use_container_width=True."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False, "responsive": True},
                    key=key, theme=None, **_stretch_kw(st.plotly_chart))


def _panel_header(title: str, caption: str) -> None:
    st.markdown(f'<div class="p2-ph"><div class="p2-pt">{title}</div>'
                f'<div class="p2-pc">{caption}</div></div>', unsafe_allow_html=True)


def _label(text: str) -> None:
    st.markdown(f'<div class="p2-fl">{text}</div>', unsafe_allow_html=True)


def _select_all_stations(codes: list) -> None:
    st.session_state["p2_stations"] = list(codes)


# --------------------------------------------------------------------------
# Header + filter global
# --------------------------------------------------------------------------
def _header() -> None:
    st.markdown(
        '<div class="p2-header"><div class="p2-title">Air Quality &amp; Spatial Distribution</div>'
        '<div class="p2-sub">Analisis kondisi kualitas udara antarstasiun dan antarpulau, tingkat '
        'pelampauan baku mutu, serta kecenderungannya dari waktu ke waktu.</div></div>',
        unsafe_allow_html=True)


def _filters(stations) -> tuple[str, str, str, tuple]:
    """Satu baris filter global: polutan, rentang tanggal, stasiun."""
    lo, hi = date_bounds()
    codes = stations["kode"].tolist()
    names = dict(zip(stations["kode"], stations["lokasi"]))
    st.session_state.setdefault("p2_stations", codes)
    n_sel = len([c for c in st.session_state["p2_stations"] if c in codes])

    with st.container(border=True, height=FILTER_H, key="p2_card_filters"):
        c = st.columns([0.7, 1.4, 0.7, 2.6, 0.7, 2.2, 3.0], gap="small")
        with c[0]:
            _label("Polutan")
        with c[1]:
            pollutant = st.selectbox("Polutan", POLLUTANT_ORDER, index=0, key="p2_pollutant",
                                     label_visibility="collapsed")
        with c[2]:
            _label("Periode")
        with c[3]:
            rng = st.date_input("Rentang tanggal", value=(lo, hi), min_value=lo, max_value=hi,
                                format="DD/MM/YYYY", key="p2_dates", label_visibility="collapsed")
        with c[4]:
            _label("Stasiun")
        with c[5]:
            with st.popover(f"{n_sel} dari {len(codes)} stasiun", **_stretch_kw(st.popover)):
                st.multiselect("Stasiun", options=codes, key="p2_stations",
                               format_func=lambda k: f"{k} · {names[k]}")
                st.button("Pilih semua stasiun", key="p2_btn_all",
                          on_click=_select_all_stations, args=(codes,))
        with c[6]:
            st.markdown(f'<div class="p2-note">{len(codes)} stasiun terpantau · hari valid = '
                        f'≥18 jam data PM2.5</div>', unsafe_allow_html=True)

    # date_input mengembalikan 1 tanggal selama pengguna baru memilih awal rentang
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        start, end = rng
    else:
        start, end = lo, hi
    selected = [k for k in st.session_state.get("p2_stations", codes) if k in codes] or codes
    return pollutant, start.isoformat(), end.isoformat(), tuple(selected)


# --------------------------------------------------------------------------
# 2.1 KPI
# --------------------------------------------------------------------------
def _kpis(k: dict, pollutant: str) -> None:
    unit = "µg/m³"
    d = 0 if pollutant == "CO" else 1
    cols = st.columns(5, gap="small")

    with cols[0], st.container(border=True, height=KPI_H, key="p2_card_kpi1"):
        st.metric(f"Rata-rata {pollutant} ({unit})", fmt(k["mean"], d),
                  f"{fmt(k['n_days'], 0)} stasiun-hari valid", delta_color="off")

    with cols[1], st.container(border=True, height=KPI_H, key="p2_card_kpi2"):
        t = k["top_station"]
        st.metric(f"Stasiun tertinggi ({pollutant})", t[0] if t else "N/A",
                  f"{t[1]} · {fmt(t[2], d)} {unit}" if t else "tidak ada data valid", delta_color="off")

    with cols[2], st.container(border=True, height=KPI_H, key="p2_card_kpi3"):
        t = k["top_island"]
        st.metric(f"Pulau tertinggi ({pollutant})", t[0] if t else "N/A",
                  f"{fmt(t[1], d)} {unit} · rata-rata stasiun" if t else "tidak ada data valid",
                  delta_color="off")

    with cols[3], st.container(border=True, height=KPI_H, key="p2_card_kpi4"):
        st.metric("Rata-rata IKU (indikatif)", fmt(k["iku_mean"], 1),
                  f"{k['iku_n']} dari {k['iku_total']} stasiun memenuhi kelengkapan", delta_color="off")

    with cols[4], st.container(border=True, height=KPI_H, key="p2_card_kpi5"):
        hi, lo = k["iku_hi"], k["iku_lo"]
        row = lambda sym, v: (f'<div class="p2-k5-row">{sym} {v[0]} <b>{fmt(v[1], 1)}</b></div>'
                              if v else f'<div class="p2-k5-row">{sym} N/A</div>')
        st.markdown('<div class="p2-k5-label">IKU tertinggi · terendah (indikatif)</div>'
                    + row("▲", hi) + row("▼", lo), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Caption panel
# --------------------------------------------------------------------------
def _map_caption(metric: str, pollutant: str) -> str:
    return {"Konsentrasi": f"Rata-rata {pollutant}, µg/m³",
            "IKU": "IKU indikatif per stasiun",
            "ISPU": "Rata-rata ISPU maks., indikatif"}[metric]


def _rank_caption(metric: str, pollutant: str) -> str:
    if metric == "Konsentrasi":
        bm = POLLUTANTS[pollutant]["bm"]
        return f"Rata-rata {pollutant} (µg/m³) per stasiun · garis putus-putus = baku mutu {fmt(bm, 0)}"
    if metric == "ISPU":
        return "Rata-rata ISPU maks. per stasiun (indikatif) · garis = batas kategori 50 dan 100"
    return f"IKU indikatif per stasiun · garis = batas kategori Baik ({IKU_BAIK}) · N/A = hari valid &lt; 75%"


def _trend_caption(pollutant: str) -> str:
    cfg = POLLUTANTS[pollutant]
    s = (f"Rata-rata antarstasiun (bobot sama) + rata-rata bergerak 7 hari · garis putus-putus = "
         f"baku mutu {fmt(cfg['bm'], 0)} µg/m³ ({cfg['basis']})")
    if pollutant in ("CO", "O3"):
        s += "; dibandingkan dengan rata-rata harian, indikatif"
    return s


# --------------------------------------------------------------------------
# Halaman
# --------------------------------------------------------------------------
def render_air_quality_page() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    stations = load_stations()

    _header()
    pollutant, start, end, kodes = _filters(stations)

    sm = station_metrics(pollutant, start, end, kodes)
    _kpis(build_kpis(sm), pollutant)

    # ---- Baris 1: Peta (4) | Peringkat stasiun (5) | % melampaui baku mutu (3)
    r1 = st.columns([4, 5, 3], gap="small")

    with r1[0], st.container(border=True, height=CARD_H, key="p2_card_map"):
        st.markdown('<div class="p2-pt">Peta Sebaran Stasiun</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="small")
        with right:
            if hasattr(st, "segmented_control"):
                choice = st.segmented_control("Metrik", METRICS, default=METRICS[0],
                                              key="p2_metric", label_visibility="collapsed")
            else:
                choice = st.radio("Metrik", METRICS, horizontal=True, key="p2_metric",
                                  label_visibility="collapsed")
        metric = choice or METRICS[0]
        with left:
            st.markdown(f'<div class="p2-pc">{_map_caption(metric, pollutant)}</div>', unsafe_allow_html=True)
        _plot(map_figure(sm, metric, pollutant, MAP_H), "p2_fig_map")

    with r1[1], st.container(border=True, height=CARD_H, key="p2_card_rank"):
        _panel_header("Peringkat Stasiun", _rank_caption(metric, pollutant))
        _plot(ranking_figure(sm, metric, pollutant, CHART_H), "p2_fig_rank")

    with r1[2], st.container(border=True, height=CARD_H, key="p2_card_exceed"):
        _panel_header("% Melampaui Baku Mutu",
                      "Tidak terpengaruh filter polutan · CO &amp; O3: maks. rata-rata 8 jam")
        _plot(exceedance_figure(exceedance(start, end, kodes), CHART_H), "p2_fig_exceed")

    # ---- Baris 2: Tren harian (6) | Histogram ISPU (3) | Boxplot IKU (3)
    r2 = st.columns([6, 3, 3], gap="small")

    with r2[0], st.container(border=True, height=CARD_H, key="p2_card_trend"):
        _panel_header(f"Tren Harian {pollutant}", _trend_caption(pollutant))
        _plot(trend_figure(daily_trend(pollutant, start, end, kodes), pollutant, CHART_H), "p2_fig_trend")

    with r2[1], st.container(border=True, height=CARD_H, key="p2_card_ispu"):
        _panel_header("Distribusi Kategori ISPU",
                      "Stasiun-hari valid · tidak terpengaruh filter polutan")
        _plot(ispu_figure(ispu_distribution(start, end, kodes), CHART_H), "p2_fig_ispu")

    with r2[2], st.container(border=True, height=CARD_H, key="p2_card_box"):
        _panel_header("Boxplot IKU per Tahun",
                      "IKU indikatif · * = tahun parsial · tidak terpengaruh filter polutan")
        _plot(iku_box_figure(iku_station_year(start, end, kodes), CHART_H), "p2_fig_box")

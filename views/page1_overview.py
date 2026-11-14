"""Halaman 1 SIGAP: Overview.

Satu fungsi publik: render_overview_page(data_dir="data", filters=None).
Tanpa st.set_page_config / sidebar. Prefix key p1_, kelas CSS p1-. Asumsi tema terang.
Halaman ini TIDAK punya filter (periode pembanding tetap); ``filters`` diabaikan.
Metode agregasi: lihat docstring utils/overview_utils.py.
"""
import html
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import overview_utils as U
from utils.common import ISLAND_COLORS, fmt

BG, CARD, BORDER, BLUE, TXT, TXT2 = "#F3F6FC", "#FFFFFF", "#DCE6F5", "#2563EB", "#102044", "#64748B"
YEAR_COLORS = U.YEAR_COLORS
YEARS = U.YEARS
HEAD_H = 52
H_TITLE, H_KPI, H_CHARTS, H_TOP3, H_CONF = 64, 98, 400, 92, 64
CHART_H = H_CHARTS - HEAD_H - 28  # padding + border + cadangan


def _sg(x, d=1):
    """Angka bertanda (+/-) gaya Indonesia."""
    return ("+" if x > 0 else "") + fmt(x, d)


def _plot(fig, key):
    cfg = {"displayModeBar": False, "responsive": True}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def _metric(label, value, delta=None, help=None, arrow=True):
    kw = dict(delta_color="off", help=help)
    if not arrow:
        try:
            st.metric(label, value, delta, delta_arrow="off", **kw)
            return
        except TypeError:  # Streamlit lama: tanpa delta_arrow -> jangan tampilkan panah menyesatkan
            st.metric(label, value, **kw)
            return
    st.metric(label, value, delta, **kw)


def _head(title, caption, right_html=""):
    st.markdown(
        f"<div class='p1-head'><div class='p1-prow'><div class='p1-ptitle'>{title}</div>{right_html}</div>"
        f"<div class='p1-pcap'>{caption}</div></div>", unsafe_allow_html=True)


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stApp {{ font-family: 'Inter', sans-serif; }}
header[data-testid="stHeader"] {{ display:none; }}
.block-container {{ padding: 24px 24px 24px 24px !important; max-width: 100% !important; }}
div[data-testid="stVerticalBlock"] {{ gap: 12px; }}
div[data-testid="stHorizontalBlock"] {{ gap: 12px; }}
[class*="st-key-p1_card"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:8px 12px !important; }}
[class*="st-key-p1_card"] div[data-testid="stVerticalBlock"] {{ gap: 0; }}
.p1-title {{ font-size:26px; font-weight:700; color:{TXT}; line-height:1.15; }}
.p1-sub {{ font-size:13px; color:{TXT2}; margin-top:2px; line-height:1.3; }}
.p1-titlebox {{ height:{H_TITLE}px; overflow:hidden; }}
.p1-head {{ height:{HEAD_H}px; overflow:hidden; }}
.p1-prow {{ display:flex; justify-content:space-between; align-items:baseline; }}
.p1-ptitle {{ font-size:16px; font-weight:600; color:{TXT}; line-height:1.25; }}
.p1-pcap {{ font-size:11px; color:{TXT2}; line-height:1.3; max-height:29px; overflow:hidden; }}
.p1-leg {{ font-size:11px; color:{TXT2}; white-space:nowrap; }}
.p1-leg b {{ font-weight:600; }}
.p1-kcap {{ font-size:10.5px; color:{TXT2}; line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
div[data-testid="stMetric"] {{ padding:0; }}
div[data-testid="stMetricLabel"] {{ min-height:0; }}
div[data-testid="stMetricLabel"] p {{ font-size:12px; color:{TXT2}; line-height:1.3; }}
div[data-testid="stMetricValue"] {{ font-size:26px; font-weight:700; color:{TXT}; line-height:1.1; }}
div[data-testid="stMetricDelta"] {{ font-size:12px; color:{TXT2} !important; line-height:1.3; }}
div[data-testid="stMetricDelta"] * {{ color:{TXT2} !important; fill:{TXT2} !important; }}
.p1-find {{ border-left:4px solid {BLUE}; padding-left:10px; }}
.p1-flab {{ font-size:11px; font-weight:600; color:{BLUE}; text-transform:uppercase; letter-spacing:.03em; line-height:1.3; }}
.p1-ftxt {{ font-size:13px; font-weight:500; color:{TXT}; line-height:1.35; margin-top:2px; }}
.p1-conf {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }}
.p1-citem {{ border-left:1px solid {BORDER}; padding-left:12px; }}
.p1-citem:first-child {{ border-left:none; padding-left:0; }}
.p1-clab {{ font-size:11px; color:{TXT2}; line-height:1.3; }}
.p1-cval {{ font-size:14px; font-weight:600; color:{TXT}; line-height:1.3; white-space:nowrap; }}
.p1-csub {{ font-size:10.5px; color:{TXT2}; line-height:1.2; }}
.p1-foot {{ font-size:11px; color:{TXT2}; line-height:1.35; height:30px; overflow:hidden; }}
[class*="st-key-p1_card_kpi"] {{ padding:6px 12px !important; }}
[class*="st-key-p1_card_conf"] {{ padding:5px 12px !important; }}
</style>
"""


# ---------- data ----------
@st.cache_data(show_spinner="Memuat data ringkasan…")
def _load(data_dir):
    return U.build_all(data_dir)

def _resolve_dir(data_dir):
    if data_dir and os.path.exists(os.path.join(str(data_dir), "env_daily.parquet")):
        return str(data_dir)
    return os.environ.get("SIGAP_DATA_DIR", str(data_dir or "data"))

# ---------- builder grafik ----------
def _layout(fig, h, **kw):
    base = dict(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=10, color=TXT), separators=",.",
                hoverlabel=dict(font_size=11), showlegend=False)
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="#EEF2F9", linecolor=BORDER, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#EEF2F9", linecolor=BORDER, tickfont=dict(size=9), rangemode="tozero")
    return fig


def fig_trend(mt, h):
    groups = ["Nasional"] + U.ISLANDS
    n_by = mt.groupby("grup")["n"].max()
    titles = [f"{g} (n={int(n_by[g])})" for g in groups]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.075,
                        vertical_spacing=0.17)
    for i, g in enumerate(groups):
        r, c = divmod(i, 3)
        for y in YEARS:
            d = mt[(mt["grup"] == g) & (mt["tahun"] == y)].sort_values("bulan")
            if d.empty:
                continue
            labels = [U.BULAN_FULL[int(b) - 1] + (" (parsial)" if p else "")
                      for b, p in zip(d["bulan"], d["parsial"])]
            cd = np.column_stack([[g] * len(d), labels, d["n"].astype(int)])
            col = YEAR_COLORS[y]
            last_partial = bool(d["parsial"].iloc[-1])
            msize = [0] * (len(d) - 1) + [7] if last_partial else 0
            fig.add_trace(go.Scatter(
                x=d["bulan"], y=d["pm25"], mode="lines+markers", name=str(y),
                line=dict(color=col, width=2), customdata=cd,
                marker=dict(size=msize, color="white", line=dict(color=col, width=2)),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} " + str(y) +
                              "<br>PM2.5: %{y:,.1f} µg/m³<br>%{customdata[2]} stasiun<extra></extra>"),
                row=r + 1, col=c + 1)
    _layout(fig, h, margin=dict(l=34, r=6, t=20, b=20))
    fig.update_xaxes(tickvals=[1, 3, 5, 7, 9, 11], ticktext=["Jan", "Mar", "Mei", "Jul", "Sep", "Nov"],
                     range=[0.6, 12.4])
    fig.update_yaxes(tickformat=",.0f")
    for a, g in zip(fig.layout.annotations, groups):  # judul facet
        a.update(font=dict(size=11, color=ISLAND_COLORS.get(g, TXT), family="Inter, sans-serif"),
                 text=f"<b>{a.text}</b>")
    return fig


def fig_bar(bar_long, ic, h):
    islands = [i for i in U.ISLANDS if i in ic.index]
    xl = [f"{i}<br>n={int(ic.loc[i, 'n'])}" for i in islands]
    fig = go.Figure()
    ymax = 0
    for y in YEARS:
        vals = [float(bar_long[(bar_long["pulau"] == i) & (bar_long["tahun"] == y)]["pm25"].iloc[0])
                for i in islands]
        ymax = max(ymax, max(vals))
        fig.add_trace(go.Bar(x=xl, y=vals, name=str(y), marker_color=YEAR_COLORS[y],
                             text=[fmt(v, 1) for v in vals], textposition="outside",
                             textfont=dict(size=9, color=TXT), cliponaxis=False,
                             hovertemplate="<b>%{x}</b><br>" + str(y) +
                                           ": %{y:,.1f} µg/m³<extra></extra>"))
    for i, lab in zip(islands, xl):
        top = max(float(bar_long[(bar_long["pulau"] == i) & (bar_long["tahun"] == y)]["pm25"].iloc[0])
                  for y in YEARS)
        fig.add_annotation(x=lab, y=top, text=f"<b>{_sg(ic.loc[i, 'delta_pct'], 1)}%</b>", showarrow=False,
                           yshift=24, font=dict(size=10, color=TXT), xanchor="center")
    _layout(fig, h, margin=dict(l=36, r=6, t=26, b=34), barmode="group", bargap=0.16, bargroupgap=0.04,
            showlegend=True, legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=10)))
    fig.update_yaxes(range=[0, ymax * 1.3], tickformat=",.0f")
    fig.update_xaxes(tickfont=dict(size=10))
    return fig


# ---------- bagian halaman ----------
def _title():
    st.markdown("<div class='p1-titlebox'><div class='p1-title'>Overview</div>"
                "<div class='p1-sub'>Ringkasan kondisi PM2.5 dan beban kasus pernapasan jaringan SIGAP, "
                "serta perubahannya dibanding dua tahun sebelumnya pada periode yang sama.</div></div>",
                unsafe_allow_html=True)


def _kpis(R):
    k, tgl = R["kpi"], R["tgl_akhir"]
    per = f"1 Jan–{U.fmt_tgl(tgl)}"
    yr, pv = tgl.year, tgl.year - 1
    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True, height=H_KPI, key="p1_card_kpi1"):
            _metric("Rata-rata PM2.5 (µg/m³)", fmt(k["pm25"][yr]),
                    f"{_sg(k['pm25_delta'])} µg/m³ ({_sg(k['pm25_delta_pct'])}%) vs {pv}",
                    help=f"n = {k['n']} stasiun sepadan; periode {per} {yr}.")
            st.markdown(f"<div class='p1-kcap'>n = {k['n']} stasiun sepadan · {per}</div>", unsafe_allow_html=True)
    with cols[1]:
        with st.container(border=True, height=H_KPI, key="p1_card_kpi2"):
            _metric("% hari valid PM2.5 > 55 µg/m³", f"{fmt(k['pct55'][yr])}%",
                    f"{_sg(k['pct55_delta_pp'])} poin persentase vs {pv}",
                    help="Baku mutu PM2.5 24 jam = 55 µg/m³ (PP 22/2021). Rata-rata dari persentase per stasiun.")
            st.markdown("<div class='p1-kcap'>Baku mutu PP 22/2021 · rata-rata per stasiun</div>", unsafe_allow_html=True)
    with cols[2]:
        with st.container(border=True, height=H_KPI, key="p1_card_kpi3"):
            _metric("Stasiun dengan kenaikan > 10%", f"{k['n_naik10']} dari {k['n']} stasiun",
                    f"Turun > 10%: {k['n_turun10']} stasiun",
                    help=f"Rata-rata PM2.5 {per} {yr} dibanding {pv}, stasiun sepadan.", arrow=False)
            st.markdown(f"<div class='p1-kcap'>{yr} vs {pv} · stasiun sepadan</div>", unsafe_allow_html=True)
    with cols[3]:
        with st.container(border=True, height=H_KPI, key="p1_card_kpi4"):
            _metric("Total kasus pernapasan (resp)", fmt(k["resp"][yr], 0),
                    f"{_sg(k['resp_delta'], 0)} kasus ({_sg(k['resp_delta_pct'])}%) vs {pv}",
                    help=f"Jumlah resp seluruh RS ({k['n_rs']} RS; jumlah stasiun AQMS 26) periode {per}. "
                         "Asosiasi waktu saja: tanpa koreksi cuaca/musim, tanpa denominator populasi.")
            st.markdown(f"<div class='p1-kcap'>{k['n_rs']} RS · asosiasi waktu, bukan sebab-akibat</div>",
                        unsafe_allow_html=True)


def _charts(R):
    tgl = R["tgl_akhir"]
    k = R["kpi"]
    left, right = st.columns([7, 5])
    legend = "<div class='p1-leg'>" + " &nbsp; ".join(
        f"<b style='color:{YEAR_COLORS[y]}'>━</b> {y}" for y in YEARS) + " &nbsp; <b>○</b> parsial</div>"
    with left:
        with st.container(border=True, height=H_CHARTS, key="p1_card_trend"):
            _head("Tren Bulanan PM2.5: Perbandingan Tahun per Pulau",
                  f"Rata-rata bulanan stasiun sepadan (µg/m³); skala sumbu-y berbeda per panel. "
                  f"Marker berongga: {U.BULAN_FULL[tgl.month - 1]} {tgl.year} parsial (s.d. {tgl.day} "
                  f"{U.BULAN[tgl.month - 1]}).", legend)
            _plot(fig_trend(R["monthly"], CHART_H), "p1_chart_trend")
    with right:
        with st.container(border=True, height=H_CHARTS, key="p1_card_bar"):
            _head("Periode Sama per Pulau",
                  f"Rata-rata 1 Jan–{tgl.day} {U.BULAN[tgl.month - 1]}, stasiun sepadan. Angka di atas grup: "
                  f"perubahan {tgl.year} vs {tgl.year - 1}. Angka pulau sensitif terhadap jumlah stasiun (n).")
            _plot(fig_bar(R["bar_long"], R["island_bar"], CHART_H), "p1_chart_bar")


def _top3(R):
    cols = st.columns(3)
    for i, f in enumerate(R["findings"]):
        with cols[i]:
            with st.container(border=True, height=H_TOP3, key=f"p1_card_top{i + 1}"):
                st.markdown(f"<div class='p1-find'><div class='p1-flab'>{i + 1}. {html.escape(f['label'])}</div>"
                            f"<div class='p1-ftxt'>{html.escape(f['text'])}</div></div>", unsafe_allow_html=True)


def _confidence(R):
    c = R["conf"]
    excl = ", ".join(c["excluded"]) or "tidak ada"
    tip1 = html.escape(f"Dikeluarkan dari perbandingan (< 75% hari valid pada salah satu tahun): {excl}", quote=True)
    p, s = c["peran"], c["status"]
    items = [
        ("Stasiun sepadan", f"{c['n_comp']} dari {c['n_aqms']} stasiun", f"Tahan kursor: stasiun yang dikeluarkan", tip1),
        ("Peran stasiun", f"{p['model penuh']} model penuh · {p['jaringan saja']} jaringan saja · {p['dikeluarkan']} dikeluarkan",
         "", "Sumber: dim_station.peran_analisis (27 stasiun)"),
        ("Status operasional", f"{s['Active']} Aktif · {s['Intermittent']} Terbatas · {s['Defective']} Rusak", "",
         "Active/Intermittent/Defective; terjemahan literal, perlu konfirmasi dokumentasi dataset"),
        ("Data valid", f"{fmt(c['valid_pct'])}% stasiun-hari",
         f"{fmt(c['valid_n'], 0)} dari {fmt(c['total_n'], 0)} (26 stasiun)", "Hari valid: ≥ 18 jam data (valid_75pct)"),
    ]
    cells = "".join(
        f"<div class='p1-citem' title=\"{t}\"><div class='p1-clab'>{html.escape(a)}</div>"
        f"<div class='p1-cval'>{html.escape(b)}</div>"
        + (f"<div class='p1-csub'>{html.escape(s_)}</div>" if s_ else "") + "</div>"
        for a, b, s_, t in items)
    with st.container(border=True, height=H_CONF, key="p1_card_conf"):
        st.markdown(f"<div class='p1-conf'>{cells}</div>", unsafe_allow_html=True)


def _footnote(R):
    tgl = R["tgl_akhir"]
    st.markdown(f"<div class='p1-foot'>{html.escape(U.footnote_text(R['island_bar'], tgl))}</div>",
                unsafe_allow_html=True)


def render_overview_page(data_dir="data", filters=None):
    """Halaman 1: Overview. ``filters`` diabaikan (periode pembanding tetap)."""
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        R = _load(_resolve_dir(data_dir))
    except FileNotFoundError as e:
        st.error(f"Data tidak ditemukan: {e}")
        return
    _title()
    _kpis(R)
    _charts(R)
    _top3(R)
    _confidence(R)
    _footnote(R)

"""Halaman 3 SIGAP: Pollution Patterns & Drivers.

Satu fungsi publik: render_pollution_patterns_page(data_dir="data", filters=None).
Tanpa st.set_page_config / sidebar. Prefix key p3_, kelas CSS p3-. Asumsi tema terang.
Agregasi: rata-rata dari rata-rata stasiun (bobot sama); 3A.4 dan 3B.1 memakai titik stasiun-hari.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import pollution_patterns_utils as U

NAVY, BG, CARD, BORDER, BLUE, TXT, TXT2 = "#10264A", "#F3F6FC", "#FFFFFF", "#DCE6F5", "#2563EB", "#102044", "#64748B"
GREY = "#CBD5E1"
CARD_H, HEAD_H, CHART_H = 306, 52, 218
SEQ = [[0, "#FFF7EC"], [.25, "#FDD49E"], [.5, "#FC8D59"], [.75, "#D7301F"], [1, "#7F0000"]]
DIV = [[0, "#2563EB"], [.5, "#F8FAFC"], [1, "#DC2626"]]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
MET = {"suhu": "Suhu", "rh": "Kelembapan", "kec_angin": "Kecepatan angin"}
MET_UNIT = {"suhu": "Suhu (°C)", "rh": "Kelembapan (%)", "kec_angin": "Kecepatan angin (m/s)"}
COMPASS = ["U", "UTL", "TL", "TTL", "T", "TTG", "TG", "GTG", "S", "GBD", "BD", "BBD", "B", "BBL", "BL", "UBL"]


# ---------- util tampilan ----------
def fmt(x, n=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:+,.{n}f}" if sign else f"{x:,.{n}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def _sample(v, lo, hi, scale=SEQ):
    t = 0.0 if hi <= lo else min(max((v - lo) / (hi - lo), 0), 1)
    for (p0, c0), (p1, c1) in zip(scale[:-1], scale[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0)
            a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            return "#%02X%02X%02X" % tuple(round(x + (y - x) * f) for x, y in zip(a, b))
    return scale[-1][1]


def _layout(fig, h=CHART_H, **kw):
    base = dict(height=h, margin=dict(l=44, r=10, t=8, b=30), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=10, color=TXT),
                separators=",.", legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                hoverlabel=dict(font_size=11))
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="#EEF2F9", linecolor=BORDER, tickfont=dict(size=10), title_font=dict(size=10))
    fig.update_yaxes(gridcolor="#EEF2F9", linecolor=BORDER, tickfont=dict(size=10), title_font=dict(size=10))
    return fig


def _note(fig, text, x=0, y=-0.2, **kw):
    fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=text, showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=9, color=TXT2), **kw)


def _plot(fig, key):
    cfg = {"displayModeBar": False, "responsive": True}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def _head(title, caption):
    st.markdown(f"<div class='p3-head'><div class='p3-ptitle'>{title}</div>"
                f"<div class='p3-pcap'>{caption}</div></div>", unsafe_allow_html=True)


def _metric(label, value, delta=None, key=None):
    try:
        st.metric(label, value, delta, delta_color="off", delta_arrow="off")
    except TypeError:
        st.metric(label, value, delta, delta_color="off")


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stApp {{ font-family: 'Inter', sans-serif; }}
header[data-testid="stHeader"] {{ display:none; }}
.block-container {{ padding: 24px 24px 24px 24px !important; max-width: 100% !important; }}
div[data-testid="stVerticalBlock"] {{ gap: 12px; }}
div[data-testid="stHorizontalBlock"] {{ gap: 12px; }}
[class*="st-key-p3_card"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:8px 12px !important; }}
[class*="st-key-p3_card"] div[data-testid="stVerticalBlock"] {{ gap: 0; }}
.p3-title {{ font-size:26px; font-weight:700; color:{TXT}; line-height:1.15; }}
.p3-sub {{ font-size:13px; color:{TXT2}; margin-top:2px; }}
.p3-head {{ height:{HEAD_H}px; overflow:hidden; }}
.p3-ptitle {{ font-size:16px; font-weight:600; color:{TXT}; line-height:1.25; }}
.p3-pcap {{ font-size:11px; color:{TXT2}; line-height:1.3; max-height:29px; overflow:hidden; }}
[class*="st-key-p3_filter"] label, [class*="st-key-p3_filter"] p {{ font-size:12px; color:{TXT2}; }}
[class*="st-key-p3_filter"] div[data-baseweb="select"] > div {{ min-height:34px; }}
div[data-testid="stMetric"] {{ padding:0; }}
div[data-testid="stMetricLabel"] p {{ font-size:12px; color:{TXT2}; }}
div[data-testid="stMetricValue"] {{ font-size:26px; font-weight:700; color:{TXT}; }}
div[data-testid="stMetricDelta"] {{ font-size:12px; color:{TXT2} !important; }}
button[data-baseweb="tab"] {{ font-size:14px; font-weight:600; height:40px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{BLUE}; }}
div[data-baseweb="tab-list"] {{ gap:12px; }}
</style>
"""


# ---------- data (cache) ----------
@st.cache_data(show_spinner="Memuat data harian…")
def _daily(data_dir):
    d = U.load_daily(data_dir)
    return d


@st.cache_resource(show_spinner="Memuat data jam-jaman (±105 MB)…")
def _hourly(data_dir):
    """Dibaca sekali; dibuang jam qc_blank/kosong/imputasi; ISPU jam dihitung sekali."""
    d = U.load_daily(data_dir)
    return U.prep_hourly(U.load_hourly(d, data_dir))


def _hsel(h, start, end, kaw):
    m = (h.tanggal >= pd.Timestamp(start)) & (h.tanggal <= pd.Timestamp(end)) & h.site_type_grup.isin(kaw)
    return h[m]


# ---------- builder grafik ----------
def fig_hour_line(w):
    fig = go.Figure()
    for col, c in (("Hari kerja", BLUE), ("Akhir pekan", "#F97316")):
        fig.add_trace(go.Scatter(x=w.jam_ke, y=w[col], name=col, mode="lines", line=dict(color=c, width=2),
                                 hovertemplate="Jam %{x}:00 WIB<br>ISPU %{y:.1f}<extra>" + col + "</extra>"))
    top = w.loc[w[["Hari kerja", "Akhir pekan"]].max(axis=1).idxmax()]
    fig.add_annotation(x=top.jam_ke, y=max(top["Hari kerja"], top["Akhir pekan"]), text=f"Tertinggi: jam {int(top.jam_ke)}",
                       showarrow=True, arrowhead=0, ay=-18, ax=0, font=dict(size=10, color=TXT), arrowcolor=TXT2)
    lo, hi = w[["Hari kerja", "Akhir pekan"]].min().min(), w[["Hari kerja", "Akhir pekan"]].max().max()
    _layout(fig, margin=dict(l=44, r=10, t=8, b=52))
    fig.update_xaxes(title_text="Jam (WIB)", tickmode="linear", dtick=2, range=[-0.5, 23.5])
    fig.update_yaxes(title_text="ISPU rata-rata", range=[np.floor(lo) - 4, np.ceil(hi) + 8])
    _note(fig, "ISPU per jam = pendekatan eksplorasi (tabel konversi 24 jam diterapkan pada konsentrasi per jam), bukan ISPU resmi.", y=-0.3)
    return fig


def fig_weekend_bars(w):
    lab = [f"{k}<br>n={n}" for k, n in zip(w.site_type_grup, w.n)]
    fig = go.Figure()
    for col, c in (("Hari kerja", BLUE), ("Akhir pekan", "#F97316")):
        fig.add_trace(go.Bar(x=lab, y=w[col], name=col, marker_color=c,
                             hovertemplate="%{x}<br>ISPU %{y:.1f}<extra>" + col + "</extra>"))
    top = w[["Hari kerja", "Akhir pekan"]].max(axis=1)
    fig.add_trace(go.Scatter(x=lab, y=top, mode="text", showlegend=False, hoverinfo="skip",
                             text=[fmt(p, 1, True) + "%" for p in w.selisih_pct], textposition="top center",
                             textfont=dict(size=10, color=TXT)))
    _layout(fig, margin=dict(l=44, r=10, t=8, b=34))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="ISPU harian rata-rata", range=[0, top.max() * 1.2])
    return fig


def fig_heat_month_hour(hm):
    fig = go.Figure(go.Heatmap(z=hm.values, x=list(range(24)), y=BULAN, colorscale=SEQ, xgap=1, ygap=1,
                               colorbar=dict(thickness=8, len=0.95, tickfont=dict(size=9), title=dict(text="µg/m³", font=dict(size=9))),
                               hovertemplate="%{y}, jam %{x}:00<br>PM2.5 %{z:.1f} µg/m³<extra></extra>"))
    _layout(fig, margin=dict(l=40, r=10, t=8, b=30))
    fig.update_xaxes(title_text="Jam (WIB)", tickmode="linear", dtick=2)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_box_kawasan(d):
    n = U.kawasan_n(d)
    fig = go.Figure()
    for k in [k for k in U.KAWASAN if k in n]:
        fig.add_trace(go.Box(y=d.loc[d.site_type_grup == k, "ispu_max"], name=f"{k}<br>n={n[k]}",
                             marker_color=U.KAWASAN_COLORS[k], boxpoints=False, line=dict(width=1.2),
                             hovertemplate="%{x}<br>ISPU %{y:.0f}<extra></extra>"))
    _layout(fig, h=CHART_H - 50, margin=dict(l=44, r=10, t=4, b=34))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="ISPU harian")
    return fig


def fig_small_multiples(d):
    from plotly.subplots import make_subplots
    ymax = float(np.ceil(d.pm25.quantile(0.995) / 10) * 10)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.025,
                        subplot_titles=[MET_UNIT[v] for v in MET])
    stats = {}
    for i, v in enumerate(MET, start=1):
        x, y = d[v].to_numpy(), d["pm25"].to_numpy()
        m = y <= ymax
        fig.add_trace(go.Histogram2d(x=x[m], y=y[m], nbinsx=34, nbinsy=30, colorscale=[[0, "rgba(255,255,255,0)"], [.05, "#DBEAFE"], [1, "#1E3A8A"]],
                                     showscale=False, hovertemplate="n=%{z}<extra></extra>"), row=1, col=i)
        lx, ly = U.loess(x, y, frac=0.3)
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#DC2626", width=2.2), showlegend=False,
                                 hovertemplate="LOESS %{y:.1f}<extra></extra>"), row=1, col=i)
        r, n = U.spearman(x, y)
        stats[v] = (r, n)
        fig.add_annotation(xref=f"x{i} domain" if i > 1 else "x domain", yref="y domain", x=0.97, y=0.97, xanchor="right", yanchor="top",
                           text=f"ρ = {fmt(r, 2, True)}<br>n = {fmt(n, 0)}", showarrow=False, font=dict(size=11, color=TXT),
                           bgcolor="rgba(255,255,255,.85)", bordercolor=BORDER)
    _layout(fig, margin=dict(l=44, r=10, t=22, b=44), showlegend=False)
    fig.update_yaxes(range=[0, ymax])
    fig.update_yaxes(title_text="PM2.5 harian (µg/m³)", col=1)
    _note(fig, f"Gabungan antarstasiun (stasiun-hari valid). Sumbu-y dipotong di {fmt(ymax, 0)} µg/m³ (persentil 99,5) hanya untuk tampilan; ρ dan LOESS memakai semua data.", y=-0.17)
    return fig, stats


def fig_windrose(g, tot, label):
    theta = [i * 22.5 for i in range(16)]
    vals = g.loc[g.n >= U.MIN_JAM_WINDROSE, "pm25"].dropna()
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    fig = go.Figure()
    for kls in U.SPEED_LABELS:
        s = g[g.kelas == kls].set_index("sektor").reindex(range(16))
        cols = [GREY if (pd.isna(n) or n < U.MIN_JAM_WINDROSE or pd.isna(p)) else _sample(p, lo, hi)
                for n, p in zip(s.n, s.pm25)]
        fig.add_trace(go.Barpolar(r=s.freq.fillna(0), theta=theta, width=[20.5] * 16, name=kls + " m/s",
                                  marker=dict(color=cols, line=dict(color="#FFFFFF", width=0.4)), showlegend=False,
                                  customdata=np.stack([s.n.fillna(0), s.pm25.fillna(np.nan)], axis=1),
                                  hovertemplate="Dari " + "%{text}" + f" · {kls} m/s<br>Frekuensi %{{r:.2f}}%<br>"
                                  "PM2.5 %{customdata[1]:.1f} µg/m³ (n=%{customdata[0]:.0f} jam)<extra></extra>",
                                  text=COMPASS))
    fig.add_trace(go.Scatterpolar(r=[0, 0], theta=[0, 0], mode="markers", hoverinfo="skip", showlegend=False,
                                  marker=dict(size=0.1, opacity=0, color=[lo, hi], cmin=lo, cmax=hi, colorscale=SEQ, showscale=True,
                                              colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=9),
                                                            title=dict(text="PM2.5", font=dict(size=9))))))
    _layout(fig, margin=dict(l=24, r=10, t=14, b=14))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", barmode="stack",
                                 angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=theta[::2],
                                                  ticktext=COMPASS[::2], tickfont=dict(size=10), gridcolor="#EEF2F9"),
                                 radialaxis=dict(ticksuffix="%", tickfont=dict(size=9), angle=67.5, gridcolor="#EEF2F9")))
    return fig


def fig_corr_heat(c):
    z = c[list(MET)].T.values
    fig = go.Figure(go.Heatmap(z=z, x=c.kode, y=[MET[v] for v in MET], colorscale=DIV, zmin=-1, zmax=1, zmid=0, xgap=1, ygap=1,
                               customdata=np.tile(c.n.values, (3, 1)),
                               colorbar=dict(thickness=8, len=0.9, tickfont=dict(size=9), title=dict(text="ρ", font=dict(size=9))),
                               hovertemplate="%{x} · %{y}<br>ρ = %{z:.2f} (n=%{customdata} hari)<extra></extra>"))
    _layout(fig, h=CHART_H - 8, margin=dict(l=96, r=10, t=4, b=48))
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    return fig


def _scatter_station(sm, xcol, xtitle, note_extra=""):
    fig = go.Figure()
    for k in U.KAWASAN:
        s = sm[sm.site_type_grup == k]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s[xcol], y=s.pm25, mode="markers+text", name=f"{k} (n={len(s)})", text=s.kode,
                                 textposition="top center", textfont=dict(size=8, color=TXT2),
                                 marker=dict(color=U.KAWASAN_COLORS[k], size=8, line=dict(color="#fff", width=0.6)),
                                 customdata=np.stack([s.hari], axis=1),
                                 hovertemplate="%{text}<br>" + xtitle + ": %{x:,.0f}<br>PM2.5 %{y:.1f} µg/m³<br>%{customdata[0]} hari valid<extra>" + k + "</extra>"))
    x, y = sm[xcol].to_numpy(float), sm.pm25.to_numpy()
    b, a = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=TXT2, dash="dash", width=1.5), showlegend=False, hoverinfo="skip"))
    r, n = U.spearman(x, y)
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.99, xanchor="right", yanchor="top", showarrow=False,
                       text=f"ρ = {fmt(r, 2, True)} · n = {n} stasiun" + note_extra, font=dict(size=10, color=TXT),
                       bgcolor="rgba(255,255,255,.85)", bordercolor=BORDER, align="right")
    _layout(fig, margin=dict(l=44, r=10, t=8, b=34), legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=9)))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text="PM2.5 rata-rata (µg/m³)", rangemode="tozero")
    return fig, r, n


def gambut_note(sm):
    """Anotasi 3C.2 dibangun dari data (bukan klaim tetap)."""
    g = sm[sm.site_type_grup == "Gambut"]
    if g.empty:
        return ""
    return (f"<br>Gambut (n={len(g)}): lalu lintas {fmt(g.traffic_intensity_per_h.min(), 0)}–"
            f"{fmt(g.traffic_intensity_per_h.max(), 0)} kend./jam, PM2.5 {fmt(g.pm25.min(), 1)}–{fmt(g.pm25.max(), 1)}")


def fig_box_traffic(d, sm, batas):
    q1, q2 = batas
    lab = {"Rendah": f"Rendah (≤{fmt(round(q1, -1), 0)})", "Sedang": f"Sedang ({fmt(round(q1, -1), 0)}–{fmt(round(q2, -1), 0)})",
           "Tinggi": f"Tinggi (>{fmt(round(q2, -1), 0)})"}
    cols = {"Rendah": "#0D9488", "Sedang": "#F59E0B", "Tinggi": "#DC2626"}
    ymax = float(np.ceil(d.pm25.quantile(0.995) / 10) * 10)
    dd = d.merge(sm[["kode", "kat_lalin"]], on="kode")
    fig = go.Figure()
    for k in ["Rendah", "Sedang", "Tinggi"]:
        s = dd[dd.kat_lalin == k]
        fig.add_trace(go.Box(y=s.pm25, name=lab[k] + " kend./jam", text=s.kode, boxpoints="outliers", marker=dict(color=cols[k], size=3, opacity=0.5),
                             line=dict(width=1.2), hovertemplate="%{text}<br>PM2.5 %{y:.1f} µg/m³<extra>" + k + "</extra>"))
    _layout(fig, margin=dict(l=44, r=10, t=8, b=34), showlegend=False)
    fig.update_yaxes(title_text="PM2.5 harian (µg/m³)", range=[0, ymax])
    _note(fig, f"Sumbu-y dipotong di {fmt(ymax, 0)} µg/m³ (persentil 99,5) hanya untuk tampilan.", y=-0.2, x=0.45)
    return fig


# ---------- filter global (disinkronkan antar-tab) ----------
def _sync(i):
    ss = st.session_state
    ss["p3_range"] = ss[f"p3_date_{i}"]
    ss["p3_kaw"] = ss[f"p3_kaw_{i}"]
    for j in range(3):
        if j != i:
            ss[f"p3_date_{j}"] = ss["p3_range"]
            ss[f"p3_kaw_{j}"] = ss["p3_kaw"]


def _filters(i, dmin, dmax, key_prefix="p3_filter"):
    ss = st.session_state
    ss.setdefault("p3_range", (dmin, dmax))
    ss.setdefault("p3_kaw", list(U.KAWASAN))
    ss.setdefault(f"p3_date_{i}", ss["p3_range"])
    ss.setdefault(f"p3_kaw_{i}", ss["p3_kaw"])
    with st.container(key=f"{key_prefix}_{i}"):
        c1, c2 = st.columns([4, 8], gap="small", vertical_alignment="center")
        with c1:
            st.date_input("Rentang tanggal", min_value=dmin, max_value=dmax, key=f"p3_date_{i}", on_change=_sync, args=(i,))
        with c2:
            st.multiselect("Kawasan", U.KAWASAN, key=f"p3_kaw_{i}", on_change=_sync, args=(i,))
    rng = ss["p3_range"]
    start, end = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax)
    kaw = ss["p3_kaw"] or list(U.KAWASAN)
    return start, end, tuple(kaw)


def _empty(msg="Tidak ada data valid untuk filter ini."):
    st.info(msg)


# ---------- tab ----------
def _tab1(d, h, start, end, kaw):
    hs = _hsel(h, start, end, kaw)
    a, b = st.columns([7, 5], gap="small")
    with a:
        with st.container(border=True, height=CARD_H, key="p3_card_1a"):
            _head("ISPU rata-rata per jam: hari kerja vs akhir pekan", "Rata-rata antarstasiun (bobot sama), hari valid. Menunjukkan jam dengan ISPU tertinggi.")
            w = U.by_hour_weekend(hs)
            _plot(fig_hour_line(w), "p3_ch_1a") if len(w) else _empty()
    with b:
        with st.container(border=True, height=CARD_H, key="p3_card_1b"):
            _head("ISPU hari kerja vs akhir pekan per kawasan", "Label = selisih % akhir pekan terhadap hari kerja. Rata-rata antarstasiun; n = jumlah stasiun.")
            w2 = U.weekend_by_kawasan(d)
            _plot(fig_weekend_bars(w2), "p3_ch_1b") if len(w2) else _empty()
    a, b = st.columns([7, 5], gap="small")
    with a:
        with st.container(border=True, height=CARD_H, key="p3_card_1c"):
            _head("PM2.5 rata-rata menurut bulan × jam", "Bulan dirata-ratakan antartahun; Nov–Des hanya memuat 2024–2025 (data berakhir 14 Okt 2026).")
            hm = U.heat_month_hour(hs)
            _plot(fig_heat_month_hour(hm), "p3_ch_1c") if hm.notna().any().any() else _empty()
    with b:
        with st.container(border=True, height=CARD_H, key="p3_card_1d"):
            c1, c2 = st.columns([5, 4], gap="small", vertical_alignment="top")
            with c1:
                st.markdown("<div class='p3-ptitle'>ISPU harian per kawasan</div><div class='p3-pcap'>Sebaran stasiun-hari valid. Tepi jalan hanya 2 stasiun.</div>", unsafe_allow_html=True)
            with c2:
                if len(d):
                    k, v = U.kawasan_ispu_kpi(d)
                    _metric("Kawasan ISPU tertinggi", k, f"rata-rata ISPU {fmt(v, 1)}")
            if len(d):
                _plot(fig_box_kawasan(d), "p3_ch_1d")
            else:
                _empty()


def _tab2(d, h, start, end, kaw):
    with st.container(border=True, height=CARD_H, key="p3_card_2a"):
        _head("Asosiasi meteorologi dengan PM2.5 harian", "Kepadatan titik stasiun-hari (histogram 2D), garis LOESS merah, dan ρ Spearman.")
        if len(d) > 100:
            fig, _ = fig_small_multiples(d)
            _plot(fig, "p3_ch_2a")
        else:
            _empty()
    a, b = st.columns([5, 7], gap="small")
    with a:
        with st.container(border=True, height=CARD_H, key="p3_card_2b"):
            _head("Windrose: PM2.5 menurut arah asal angin", "Batang = frekuensi (% jam); lapisan dalam→luar: <1, 1–2, 2–3, 3–4, ≥4 m/s. Abu-abu: <30 jam.")
            hs = _hsel(h, start, end, kaw)
            if len(hs):
                best = hs.groupby("kode")["pm25_ug_m3"].mean().idxmax()
                opts = ["Semua stasiun"] + sorted(hs.kode.unique())
                if st.session_state.get("p3_wr_stn") not in opts:
                    st.session_state["p3_wr_stn"] = best
                st.selectbox("Stasiun", opts, key="p3_wr_stn", label_visibility="collapsed")
                g, tot = U.windrose(hs, st.session_state["p3_wr_stn"])
                _plot(fig_windrose(g, tot, st.session_state["p3_wr_stn"]), "p3_ch_2b") if tot else _empty()
            else:
                _empty()
    with b:
        with st.container(border=True, height=CARD_H, key="p3_card_2c"):
            c = U.met_corr(d) if len(d) else pd.DataFrame()
            c1, c2 = st.columns([5, 3], gap="small", vertical_alignment="top")
            with c1:
                st.markdown("<div class='p3-ptitle'>ρ Spearman per stasiun terhadap PM2.5</div>"
                            "<div class='p3-pcap'>Hanya stasiun dengan ≥100 hari valid. Stasiun outlier = sel berwarna kuat.</div>", unsafe_allow_html=True)
            if len(c):
                v, mean, same, n = U.met_kpi(c)
                with c2:
                    _metric(f"Terkuat: {MET[v]}", f"ρ̄ {fmt(mean, 2, True)}", f"{same} dari {n} stasiun searah")
                _plot(fig_corr_heat(c), "p3_ch_2c")
            else:
                _empty()


def _tab3(d):
    if not len(d):
        _empty()
        return
    tg, sm, batas = U.traffic_groups(d)
    a, b = st.columns(2, gap="small")
    with a:
        with st.container(border=True, height=CARD_H, key="p3_card_3a"):
            _head("Jarak ke jalan vs PM2.5 rata-rata", "Asosiasi antarstasiun (satu titik = satu stasiun), bukan sebab-akibat.")
            fig, _, _ = _scatter_station(sm, "distance_to_road_m", "Jarak ke jalan (m)")
            _plot(fig, "p3_ch_3a")
    with b:
        with st.container(border=True, height=CARD_H, key="p3_card_3b"):
            _head("Intensitas lalu lintas vs PM2.5 rata-rata", "Asosiasi antarstasiun; warna = kawasan. Tepi jalan hanya 2 stasiun.")
            fig, _, _ = _scatter_station(sm, "traffic_intensity_per_h", "Lalu lintas (kendaraan/jam)", gambut_note(sm))
            _plot(fig, "p3_ch_3b")
    a, b = st.columns([4, 8], gap="small")
    with a:
        with st.container(border=True, height=CARD_H, key="p3_card_3c"):
            st.markdown("<div class='p3-ptitle'>PM2.5 per kategori lalu lintas</div>"
                        "<div class='p3-pcap'>Tertil stasiun; rata-rata dari rata-rata stasiun.</div>", unsafe_allow_html=True)
            q1, q2 = batas
            rng = {"Rendah": f"≤{fmt(round(q1, -1), 0)}", "Sedang": f"{fmt(round(q1, -1), 0)}–{fmt(round(q2, -1), 0)}", "Tinggi": f">{fmt(round(q2, -1), 0)}"}
            for k in ["Rendah", "Sedang", "Tinggi"]:
                _metric(f"{k} ({rng[k]} kend./jam)", f"{fmt(tg.loc[k, 'pm25'], 1)} µg/m³", f"{int(tg.loc[k, 'n'])} stasiun")
    with b:
        with st.container(border=True, height=CARD_H, key="p3_card_3d"):
            _head("Sebaran PM2.5 harian menurut kategori lalu lintas", "Titik = pencilan stasiun-hari (hover: kode stasiun). Batas tertil dalam kendaraan/jam.")
            _plot(fig_box_traffic(d, sm, batas), "p3_ch_3d")


def render_pollution_patterns_page(data_dir="data", filters=None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='p3-title'>Pollution Patterns &amp; Drivers</div>"
                "<div class='p3-sub'>Analisis pola waktu dan kawasan, pengaruh meteorologi, serta asosiasi lalu lintas dan lokasi dengan konsentrasi polutan.</div>",
                unsafe_allow_html=True)
    daily = _daily(data_dir)
    dmin, dmax = daily.tanggal.min().date(), daily.tanggal.max().date()
    tabs = st.tabs(["Waktu & Kawasan", "Meteorologi", "Lalu Lintas & Lokasi"])
    h = None
    for i, tab in enumerate(tabs):
        with tab:
            start, end, kaw = _filters(i, dmin, dmax)
            d = U.filter_daily(daily, start, end, kaw)
            if i == 0:
                _tab1(d, _hourly(data_dir), start, end, kaw)
            elif i == 1:
                _tab2(d, _hourly(data_dir), start, end, kaw)
            else:
                _tab3(d)

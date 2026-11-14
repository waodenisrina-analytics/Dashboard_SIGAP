"""
SIGAP - Halaman 5: DATA ANALYSIS
Pemanggilan dari aplikasi utama:

    from views.page5_data_analysis import render_data_analysis_page
    render_data_analysis_page(filters=global_filters, data_dir="data")

`filters` (opsional, dict) mengikuti filter global aplikasi utama (format sama dengan Halaman 4):
    {"date_range": (date_awal, date_akhir),   # membatasi riwayat yang ditampilkan & data per jam
     "provinsi":   ["Riau", ...] | None,       # wilayah
     "site_group": ["Perkotaan", ...] | None,  # tipe lokasi (site_type_grup); alias: "site_type"
     "station":    ["JKT-01", ...] | None}     # kode stasiun (opsional)
Jika None, dibaca dari st.session_state["global_filters"] (bila ada); selain itu = tanpa filter.

Fungsi ini tidak memanggil st.set_page_config dan tidak membuat sidebar.
Semua key widget/CSS berawalan "p5_" agar tidak bentrok dengan halaman lain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analysis_utils as U

# ---- Design tokens (Master Design Specification; identik dengan Halaman 4) -----
NAVY, NAVY2, BG = "#10264A", "#263E60", "#F3F6FC"
CARD, ACCENT, BORDER = "#FFFFFF", "#2563EB", "#DCE6F5"
TEXT, TEXT2 = "#102044", "#64748B"
C_CASE, C_MUTED = "#F59E0B", "#94A3B8"
STATUS_COLOR = {"Aktif": "#16A34A", "Terbatas": C_CASE, "Rusak": "#DC2626"}
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

# ---- Ukuran (target viewport 1440x900; sesuaikan di sini bila perlu) ----------
PANEL_H = 376   # tinggi tiap panel (px): 24 padding + header ~58 + 2 panel + gap = muat 900 px
CHART_H = {"forecast": 270, "map": 255, "drivers": 240, "perf": 190}   # tinggi plot di dalam panel (px)
PLOT_CFG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- #
# CSS (scoped ke .st-key-p5_root, mengikuti pola Halaman 4)
# --------------------------------------------------------------------------- #
def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Tambahkan baris ini untuk mengubah background aplikasi menjadi terang */
.stApp {{ background: {BG} !important; }}

.block-container {{padding: 24px !important; max-width: none !important;}}
header[data-testid="stHeader"] {{display: none;}}
.st-key-p5_root {{font-family: {FONT}; color: {TEXT};}}
.st-key-p5_root [data-testid="stVerticalBlock"] {{gap: 12px;}}
.p5-title {{font-size: 27px; font-weight: 700; color: {TEXT}; line-height: 1.2;}}
.p5-sub {{font-size: 13px; color: {TEXT2}; margin-top: 2px;}}
.p5-ptitle {{font-size: 16px; font-weight: 600; color: {TEXT}; line-height: 1.25;}}
.p5-psub {{font-size: 11px; color: {TEXT2}; margin-top: 1px;}}
.p5-cap {{font-size: 11px; color: {TEXT2}; line-height: 1.3;}}
.p5-msg {{font-size: 13px; color: {TEXT2}; text-align: center; display: flex; align-items: center;
         justify-content: center;}}
[class*="st-key-p5_panel"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px;
  height: {PANEL_H}px;}}
.st-key-p5_root [class*="st-key-p5_panel"],
.st-key-p5_root [class*="st-key-p5_panel"] [data-testid="stVerticalBlock"] {{gap: 4px;}}
.p5-metrics {{display: flex; gap: 8px;}}
.p5-m {{flex: 1; background: {BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 6px 12px;}}
.p5-ml {{font-size: 12px; color: {TEXT2};}}
.p5-mv {{font-size: 26px; font-weight: 700; color: {TEXT}; line-height: 1.15;}}
.p5-mv small {{font-size: 12px; font-weight: 500; color: {TEXT2};}}
.st-key-p5_root [data-baseweb="select"] > div {{min-height: 32px; font-size: 12px;}}
.st-key-p5_root [data-testid="stRadio"] label p {{font-size: 12px;}}
.st-key-p5_root [data-testid="stDateInput"] input {{font-size: 12px;}}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cache data / model (kunci = argumen sederhana agar hashable)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_static(data_dir: str):
    return U.load_stations(data_dir), U.load_daily(data_dir)


@st.cache_data(show_spinner=False)
def _load_hourly(data_dir: str):
    return U.load_hourly(data_dir)


@st.cache_data(show_spinner=False)
def _fingerprint(data_dir: str) -> str:
    return U.data_fingerprint(data_dir)


@st.cache_data(show_spinner=False)
def _forecast_cached(data_dir: str, art_dir: str, codes: tuple):
    """Pakai artefak pra-hitung bila ada & cocok dengan data; jika tidak, latih langsung."""
    hit = U.load_artifact(art_dir, "forecast", codes, _fingerprint(data_dir))
    if hit is not None:
        return hit[0]                      # (bundle, dropped)
    _, daily = _load_static(data_dir)
    return U.compute_forecast(daily, codes)


@st.cache_data(show_spinner=False)
def _driver_cached(data_dir: str, art_dir: str, codes: tuple, date_range: tuple):
    """Artefak pra-hitung dipakai bila filter tanggal mencakup seluruh data; jika tidak, latih langsung."""
    hit = U.load_artifact(art_dir, "driver", codes, _fingerprint(data_dir))
    span = hit[1].get("span") if hit is not None else None
    if span is not None and U.range_covers_span(date_range, span):
        return hit[0]
    stations, _ = _load_static(data_dir)
    hourly = _load_hourly(data_dir)
    dr = date_range if date_range and len(date_range) == 2 else None
    return U.run_driver_model(hourly, stations, list(codes), dr)


# --------------------------------------------------------------------------- #
# Helper UI
# --------------------------------------------------------------------------- #
def _normalize_filters(filters) -> dict:
    if filters is None:
        filters = st.session_state.get("global_filters") or {}
    dr = filters.get("date_range")
    if dr is not None and len(dr) == 2 and dr[0] is not None and dr[1] is not None:
        dr = (pd.Timestamp(dr[0]), pd.Timestamp(dr[1]))
    else:
        dr = None
    return {
        "date_range": dr,
        "provinsi": filters.get("provinsi"),
        "site_type": filters.get("site_group") or filters.get("site_type"),
        "station": filters.get("station"),
    }


def _show(fig: go.Figure, key: str) -> None:
    """st.plotly_chart kompatibel lintas versi Streamlit (width='stretch' vs use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key=key)


def _base_layout(fig: go.Figure, height: int, legend: bool = True, **kw) -> go.Figure:
    """Gaya grafik seragam dengan Halaman 4: latar transparan (mengikuti kartu), teks TEXT, grid BORDER."""
    margin_cfg = dict(l=8, r=8, t=34, b=8)
    margin_cfg.update(kw.pop("margin", {}))
    fig.update_layout(
        height=height, margin=margin_cfg,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=TEXT),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=TEXT, weight=600),
                    bgcolor="rgba(255, 255, 255, 0.7)", bordercolor=BORDER, borderwidth=1),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, zeroline=False, automargin=True,
                     tickfont=dict(size=11, color=TEXT), title_font=dict(size=11, color=TEXT))
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER, zeroline=False, automargin=True,
                     tickfont=dict(size=11, color=TEXT), title_font=dict(size=11, color=TEXT))
    return fig


def _panel_head(title: str, sub: str, widths: list[float] | None = None):
    """Judul panel + subjudul (pola Halaman 4). Mengembalikan kolom kontrol di sebelah kanan judul."""
    html = f'<div class="p5-ptitle">{title}</div><div class="p5-psub">{sub}</div>'
    if not widths:
        st.markdown(html, unsafe_allow_html=True)
        return []
    cols = st.columns(widths, vertical_alignment="center")
    cols[0].markdown(html, unsafe_allow_html=True)
    return cols[1:]


def _caption(text: str) -> None:
    st.markdown(f'<div class="p5-cap">{text}</div>', unsafe_allow_html=True)


def _message(text: str, height: int) -> None:
    st.markdown(f'<div class="p5-msg" style="min-height:{height}px">{text}</div>', unsafe_allow_html=True)


def _metric_cards(items: list[tuple[str, str, str]]) -> None:
    html = '<div class="p5-metrics">' + "".join(
        f'<div class="p5-m"><div class="p5-ml">{label}</div><div class="p5-mv">{value}<small> {unit}</small></div></div>'
        for label, value, unit in items
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def _tgl(ts, with_day: bool = True) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {_BULAN[ts.month - 1]} {ts.year}" if with_day else f"{_BULAN[ts.month - 1]} {ts.year}"


def _fmt(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def _fig_forecast(bundle: U.ForecastBundle, fc: pd.DataFrame, date_range, height: int) -> go.Figure:
    y = bundle.y
    if date_range is not None:
        y = y[(y.index >= date_range[0]) & (y.index <= date_range[1])]
    last_obs = bundle.y.index[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(245,158,11,0.25)", name="Interval prediksi 90%", customdata=fc["upper"],
        hovertemplate="%{x|%d %b %Y}<br>Batas bawah: %{y:.1f} µg/m³<br>Batas atas: %{customdata:.1f} µg/m³<extra>Interval 90%</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, mode="lines", name="PM2.5 aktual", line=dict(color=ACCENT, width=1.8),
        hovertemplate="%{x|%d %b %Y}<br>Aktual: %{y:.1f} µg/m³<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["yhat"], mode="lines", name="Forecast PM2.5",
        line=dict(color=C_CASE, width=2.2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:.1f} µg/m³<extra></extra>",
    ))
    x_mark = last_obs.strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=x_mark, x1=x_mark, y0=0, y1=1, yref="paper",
                  line=dict(color=TEXT2, width=1, dash="dot"))
    fig.add_annotation(x=x_mark, y=1, yref="paper", text="Awal forecast", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(size=11, color=TEXT), xshift=-4)
    _base_layout(fig, height)
    fig.update_xaxes(title_text="Tanggal")
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero")
    return fig


def _fig_map(stations: pd.DataFrame, selected_codes: set, size_mode: str, height: int) -> go.Figure:
    col = "pm25_obs_pct" if size_mode == "obs" else "distance_to_road_m"
    vals = stations[col]
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    span = (vmax - vmin) if vmax > vmin else 1.0
    stations = stations.assign(_size=(7 + 17 * (vals - vmin) / span).fillna(6.0))

    def _hover(r):
        obs = "tidak ada data" if pd.isna(r["pm25_obs_pct"]) else f"{r['pm25_obs_pct']:.1f}%"
        return (f"<b>{r['kode']}</b> · {r['status_label']}<br>{r['kota_kabupaten']}, {r['provinsi']}"
                f"<br>Tipe: {r['site_type']}<br>Jarak ke jalan: {r['distance_to_road_m']:.0f} m"
                f"<br>Ketersediaan PM2.5: {obs}")

    fig = go.Figure()
    other = stations[~stations["kode"].isin(selected_codes)]
    if len(other):
        fig.add_trace(go.Scattergeo(
            lon=other["longitude_dd"], lat=other["latitude_dd"], mode="markers", name="Di luar filter",
            marker=dict(size=6, color=C_MUTED, line=dict(width=0.5, color="#FFFFFF")),
            text=[_hover(r) for _, r in other.iterrows()], hoverinfo="text"))
    sel = stations[stations["kode"].isin(selected_codes)]
    for status in U.STATUS_ORDER:
        s = sel[sel["status_label"] == status]
        if s.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=s["longitude_dd"], lat=s["latitude_dd"], mode="markers", name=f"{status} ({len(s)})",
            marker=dict(size=s["_size"], color=STATUS_COLOR[status], opacity=0.9,
                        line=dict(width=1, color="#FFFFFF")),
            text=[_hover(r) for _, r in s.iterrows()], hoverinfo="text"))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=34, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=TEXT), hoverlabel=dict(font=dict(family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=11, color=TEXT, weight=600),
                    bgcolor="rgba(255, 255, 255, 0.7)", bordercolor=BORDER, borderwidth=1),
        geo=dict(projection_type="mercator", lonaxis_range=[94, 142], lataxis_range=[-11.8, 6.8],
                 showland=True, landcolor="#E3EBF8", showocean=False,
                 showcountries=True, countrycolor="#B6C6E3", coastlinecolor="#B6C6E3",
                 showframe=False, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _fig_importance(db: U.DriverBundle, height: int) -> go.Figure:
    imp = db.importance.sort_values("mean", ascending=True)
    colors = {"Meteorologi": ACCENT, "Lokasi & transportasi": C_CASE}
    fig = go.Figure()
    for grp, c in colors.items():
        s = imp[imp["group"] == grp]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            y=s["label"], x=s["mean"], orientation="h", name=grp, marker_color=c,
            error_x=dict(type="data", array=s["std"], color=TEXT2, thickness=1, width=3),
            text=[f"{v:.3f}" for v in s["mean"]], textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color=TEXT),
            hovertemplate="<b>%{y}</b><br>Penurunan R²: %{x:.4f}<extra>" + grp + "</extra>",
        ))
    _base_layout(fig, height, margin=dict(r=40, b=36), barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=imp["label"].tolist(), showgrid=False)
    fig.update_xaxes(title_text="Penurunan R² (log1p PM2.5) saat variabel diacak", showgrid=True, gridcolor=BORDER)
    return fig


def _fig_actual_pred(actual, pred, height: int, log_axes: bool, unit_label: str) -> go.Figure:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    if log_axes:
        lo = max(lo, 1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Garis 1:1 (ideal)",
                             line=dict(color=TEXT2, width=1.4, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=a, y=p, mode="markers", name="Data uji",
        marker=dict(color=ACCENT, size=5, opacity=0.55),
        hovertemplate="Aktual: %{x:.1f}<br>Prediksi: %{y:.1f}<extra></extra>"))
    _base_layout(fig, height, margin=dict(b=36))
    axis = dict(type="log") if log_axes else {}
    fig.update_xaxes(title_text=f"Aktual ({unit_label})", showgrid=True, gridcolor=BORDER, **axis)
    fig.update_yaxes(title_text=f"Prediksi ({unit_label})", **axis)
    return fig


def _fig_residual(actual, pred, height: int, unit_label: str) -> go.Figure:
    r = np.asarray(actual, float) - np.asarray(pred, float)
    fig = go.Figure(go.Histogram(
        x=r, nbinsx=30, marker=dict(color=ACCENT, line=dict(color="#FFFFFF", width=0.5)), name="Residual",
        hovertemplate="Residual: %{x:.1f}<br>Frekuensi: %{y}<extra></extra>"))
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", line=dict(color=TEXT2, width=1.4, dash="dash"))
    _base_layout(fig, height, legend=False, margin=dict(t=8, b=36), bargap=0.05)
    fig.update_xaxes(title_text=f"Residual = aktual − prediksi ({unit_label})")
    fig.update_yaxes(title_text="Frekuensi")
    return fig


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def _panel_forecast(data_dir, art_dir, codes, f):
    with st.container(key="p5_panel_forecast"):
        c_h, c_s = _panel_head("Forecasting & Scenario Simulation",
                               "Aktual vs proyeksi PM2.5 harian · interval prediksi 90%", [2.2, 1.1, 1.7])
        horizon = c_h.selectbox("Horizon", [30, 60, 90], index=2, format_func=lambda v: f"Horizon {v} hari",
                                key="p5_horizon", label_visibility="collapsed")
        scen = c_s.selectbox("Skenario", [0, 15, 30], format_func=lambda v: "Skenario: baseline" if v == 0
                             else f"Skenario: kemarau +{v} hari", key="p5_scen", label_visibility="collapsed")
        if not codes:
            _message("Tidak ada stasiun yang cocok dengan filter.", CHART_H["forecast"])
            return
        try:
            with st.spinner("Memproses model forecasting…"):
                bundle, dropped = _forecast_cached(data_dir, art_dir, codes)
        except ValueError as e:
            _message(f"Forecasting tidak dapat dilakukan: {e}", CHART_H["forecast"])
            return
        if bundle is None:
            _message("Data PM2.5 stasiun terpilih tidak memadai (kelengkapan &lt; 95%) untuk forecasting. "
                     "Pilih stasiun lain atau perluas filter.", CHART_H["forecast"])
            return
        fc = U.make_forecast(bundle, horizon, scen)
        _show(_fig_forecast(bundle, fc, f["date_range"], CHART_H["forecast"]), "p5_fig_forecast")
        note = (f"Rata-rata {len(bundle.stations_used)} stasiun · ARX dua-rezim (kemarau/non-kemarau) · "
                f"interval 90% empiris (backtest rolling-origin, {bundle.n_calib_origins} origin)")
        if dropped:
            note += f" · {len(dropped)} stasiun dikeluarkan (data &lt; 95%)"
        if scen:
            note += f" · kalender musim digeser {scen} hari"
        _caption(note)


def _panel_map(stations, selected_codes):
    with st.container(key="p5_panel_map"):
        (c_m,) = _panel_head("Network Adequacy & Blind Spot",
                             "Sebaran stasiun menurut status operasional", [1.55, 1.0])
        mode = c_m.selectbox("Ukuran marker", ["obs", "road"], format_func=lambda v: "Ukuran: data PM2.5"
                             if v == "obs" else "Ukuran: jarak jalan", key="p5_size", label_visibility="collapsed")
        _show(_fig_map(stations, set(selected_codes), mode, CHART_H["map"]), "p5_fig_map")
        n = stations["status_label"].value_counts()
        _caption(f"Status dari status_operasional (Active/Intermittent/Defective): {n.get('Aktif', 0)} aktif · "
                 f"{n.get('Terbatas', 0)} terbatas · {n.get('Rusak', 0)} rusak dari {len(stations)} stasiun. "
                 "Area tanpa marker = belum ada stasiun.")


def _panel_drivers(data_dir, art_dir, codes, f):
    with st.container(key="p5_panel_drivers"):
        _panel_head("Feature Importance / Driver Attribution", "Random Forest · permutation importance (data uji)")
        if not codes:
            _message("Tidak ada stasiun yang cocok dengan filter.", CHART_H["drivers"])
            return
        try:
            with st.spinner("Melatih Random Forest…"):
                db = _driver_cached(data_dir, art_dir, codes, f["date_range"] or ())
        except ValueError as e:
            _message(f"Analisis driver tidak dapat dilakukan: {e}", CHART_H["drivers"])
            return
        _show(_fig_importance(db, CHART_H["drivers"]), "p5_fig_importance")
        r = db.corr
        note = (f"Split waktu 80/20. r suhu–RH={r.loc['temperature_c', 'humidity_pct']:.2f}; "
                f"lalu lintas–jarak={r.loc['traffic_intensity_per_h', 'distance_to_road_m']:.2f} "
                "(importance terbagi). Lalu lintas &amp; jarak = atribut tetap per stasiun.")
        if db.constant_features:
            note += " Konstan pada filter ini: " + ", ".join(db.constant_features) + "."
        _caption(note)


def _panel_perf(data_dir, art_dir, codes, f):
    with st.container(key="p5_panel_perf"):
        (c_m,) = _panel_head("Model Performance & Residual Analysis",
                             "Evaluasi pada data uji kronologis (tidak dipakai saat latih)", [1.7, 1.2])
        which = c_m.selectbox("Model", ["forecast", "driver"], format_func=lambda v: "Model: forecast harian"
                              if v == "forecast" else "Model: driver per jam (RF)", key="p5_model",
                              label_visibility="collapsed")
        if not codes:
            _message("Tidak ada stasiun yang cocok dengan filter.", CHART_H["perf"] + 80)
            return
        try:
            if which == "forecast":
                with st.spinner("Memproses model…"):
                    bundle, _ = _forecast_cached(data_dir, art_dir, codes)
                if bundle is None:
                    _message("Data PM2.5 stasiun terpilih tidak memadai untuk evaluasi forecast.", CHART_H["perf"] + 80)
                    return
                m, b, t = bundle.metrics, bundle.baseline_metrics, bundle.test
                actual, pred, unit, log_axes = t["actual"], t["pred"], "µg/m³", False
                _metric_cards([("RMSE", _fmt(m["rmse"]), "µg/m³"), ("MAE", _fmt(m["mae"]), "µg/m³"),
                               ("R²", _fmt(m["r2"], 3), ""), ("n uji", f"{m['n']:,}", "hari")])
                ts, te = bundle.split_dates["test"]
                cap = (f"Uji {_tgl(ts)}–{_tgl(te)} (prediksi 1 hari ke depan), latih "
                       f"{_tgl(bundle.split_dates['train'][0], False)}–{_tgl(bundle.split_dates['train'][1], False)}, "
                       f"tanpa shuffle. Baseline persistence: RMSE {_fmt(b['rmse'])}, MAE {_fmt(b['mae'])}, "
                       f"R² {_fmt(b['r2'], 3)}.")
            else:
                with st.spinner("Melatih Random Forest…"):
                    db = _driver_cached(data_dir, art_dir, codes, f["date_range"] or ())
                m, t = db.metrics, db.test
                actual, pred, unit, log_axes = t["actual"], t["pred"], "µg/m³", True
                _metric_cards([("RMSE", _fmt(m["rmse"]), "µg/m³"), ("MAE", _fmt(m["mae"]), "µg/m³"),
                               ("R² (log1p)", _fmt(m["r2_log"], 3), ""), ("n uji", f"{db.n_test:,}", "jam")])
                cap = (f"Uji = setelah {_tgl(db.split_time)} (split waktu 80/20); scatter {len(t):,} titik sampel, "
                       f"sumbu log. R² skala asli = {_fmt(regression_r2(t['actual'], t['pred']), 3)} "
                       "(puncak ekstrem periode uji tak terlihat saat latih).")
        except ValueError as e:
            _message(f"Evaluasi tidak dapat dilakukan: {e}", CHART_H["perf"] + 80)
            return
        g1, g2 = st.columns(2)
        with g1:
            _show(_fig_actual_pred(actual, pred, CHART_H["perf"], log_axes, unit), "p5_fig_avp")
        with g2:
            _show(_fig_residual(actual, pred, CHART_H["perf"], unit), "p5_fig_resid")
        _caption(cap)


def regression_r2(actual, pred) -> float:
    return U.regression_metrics(actual, pred)["r2"]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_data_analysis_page(filters: dict | None = None, data_dir: str | None = None,
                              artifact_dir: str | None = None) -> None:
    _inject_css()
    f = _normalize_filters(filters)

    with st.container(key="p5_root"):
        st.markdown(
            '<div class="p5-title">Data Analysis</div>'
            '<div class="p5-sub">Pemodelan prediktif, evaluasi jaringan pemantauan, identifikasi faktor penggerak '
            "polusi, dan validasi model kualitas udara.</div>",
            unsafe_allow_html=True,
        )

        try:
            data_dir = U.resolve_data_dir(data_dir)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        art_dir = U.resolve_artifact_dir(artifact_dir, data_dir)
        stations, _ = _load_static(data_dir)
        sel = U.select_stations(stations, f["provinsi"], f["site_type"], f["station"])
        codes = tuple(sel["kode"].tolist())

        left, right = st.columns([3, 2], gap="small")
        with left:
            _panel_forecast(data_dir, art_dir, codes, f)
        with right:
            _panel_map(stations, codes)

        left, right = st.columns([2, 3], gap="small")
        with left:
            _panel_drivers(data_dir, art_dir, codes, f)
        with right:
            _panel_perf(data_dir, art_dir, codes, f)

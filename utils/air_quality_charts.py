"""Pembangun grafik Plotly Halaman 2 (fungsi murni: DataFrame -> go.Figure)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.air_quality_data import IKU_BAIK, POLLUTANTS
from utils.common import ISLAND_COLORS, ISPU_CATS, ISPU_COLORS, fmt

FONT = "Inter, Arial, sans-serif"
TEXT, TEXT_SEC, BLUE = "#102044", "#64748B", "#2563EB"
GRID, BORDER, RED = "#E8EEF8", "#DCE6F5", "#DC2626"
NA_GRAY = "#94A3B8"


def _layout(fig: go.Figure, height: int, margin: dict, **extra) -> go.Figure:
    fig.update_layout(
        height=height, margin=margin,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=10, color=TEXT_SEC),
        separators=",.",  # desimal koma, ribuan titik
        hoverlabel=dict(font=dict(family=FONT, size=11)),
        **extra,
    )
    return fig


def _axes(fig: go.Figure, **ykw) -> None:
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(size=9), zeroline=False)
    kw = dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=9))
    kw.update(ykw)
    fig.update_yaxes(**kw)


def _legend_top(**kw) -> dict:
    base = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(size=9), itemsizing="constant", bgcolor="rgba(0,0,0,0)")
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# 2.2 Peta
# --------------------------------------------------------------------------
def _value_label(row: pd.Series, metric: str, pollutant: str) -> str:
    if metric == "Konsentrasi":
        d = 0 if pollutant == "CO" else 1
        return f"{pollutant}: {fmt(row['conc'], d)} µg/m³"
    if metric == "IKU":
        if pd.isna(row["iku"]):
            return "IKU: N/A (hari valid &lt; 75%)"
        return f"IKU (indikatif): {fmt(row['iku'], 1)}"
    if pd.isna(row["ispu"]):
        return "ISPU: N/A"
    return f"ISPU rata-rata (indikatif): {fmt(row['ispu'], 0)} · {row['ispu_kat']}"


def map_figure(sm: pd.DataFrame, metric: str, pollutant: str, height: int) -> go.Figure:
    new_api = hasattr(go, "Scattermap")
    Trace = go.Scattermap if new_api else go.Scattermapbox
    col = {"Konsentrasi": "conc", "IKU": "iku", "ISPU": "ispu"}[metric]

    d = sm.copy()
    d["label"] = d.apply(lambda r: _value_label(r, metric, pollutant), axis=1)
    hover = (
        "<b>%{customdata[0]}</b> · %{customdata[1]}<br>Tipe: %{customdata[2]}"
        "<br>Pulau: %{customdata[3]}<br>%{customdata[4]}"
        "<br>Hari data valid: %{customdata[5]}<extra></extra>"
    )

    def trace(sub: pd.DataFrame, **kw):
        cd = [[r.kode, r.lokasi, r.site_type, r.pulau, r.label, fmt(r.n_valid, 0)]
              for r in sub.itertuples()]
        return Trace(lat=sub["latitude_dd"], lon=sub["longitude_dd"], mode="markers",
                     customdata=cd, hovertemplate=hover, **kw)

    fig = go.Figure()
    has_val = d[col].notna()

    if metric == "ISPU":
        for cat in ISPU_CATS:
            sub = d[d["ispu_kat"] == cat]
            if len(sub):
                fig.add_trace(trace(sub, name=cat, marker=dict(size=11, color=ISPU_COLORS[cat], opacity=0.95)))
    else:
        sub = d[has_val]
        if len(sub):
            vals = sub[col]
            cmin, cmax = float(vals.min()), float(vals.max())
            if cmax - cmin < 1e-6:
                cmax = cmin + 1.0
            # Konsentrasi: tinggi = buruk (gelap). IKU: tinggi = baik (hijau).
            scale = ([[0, "#FDE68A"], [0.5, "#F97316"], [1, "#7F1D1D"]] if metric == "Konsentrasi"
                     else [[0, "#7F1D1D"], [0.5, "#FACC15"], [1, "#16A34A"]])
            fig.add_trace(trace(sub, name=metric, showlegend=False, marker=dict(
                size=11, color=vals, colorscale=scale, cmin=cmin, cmax=cmax, opacity=0.95,
                colorbar=dict(thickness=8, len=0.8, x=0.99, xanchor="right", outlinewidth=0,
                              tickfont=dict(size=9, family=FONT), bgcolor="rgba(255,255,255,.75)"))))
    na = d[~has_val]
    if len(na):
        fig.add_trace(trace(na, name="N/A", marker=dict(size=10, color=NA_GRAY, opacity=0.9)))

    view = dict(style="carto-positron", center=dict(lat=-2.0, lon=118.0), zoom=2.25)
    _layout(fig, height, dict(l=0, r=0, t=0, b=0), uirevision="p2map",
            legend=dict(orientation="h", x=0.01, y=0.01, xanchor="left", yanchor="bottom",
                        bgcolor="rgba(255,255,255,.85)", font=dict(size=9, family=FONT)),
            **{"map" if new_api else "mapbox": view})
    return fig


# --------------------------------------------------------------------------
# 2.3 Bar peringkat stasiun
# --------------------------------------------------------------------------
def ranking_figure(sm: pd.DataFrame, metric: str, pollutant: str, height: int) -> go.Figure:
    col = {"Konsentrasi": "conc", "IKU": "iku", "ISPU": "ispu"}[metric]
    d = sm.sort_values(col, ascending=False, na_position="last").reset_index(drop=True)
    order = d["kode"].tolist()
    unit = " µg/m³" if metric == "Konsentrasi" else ""
    dec = 0 if (metric == "ISPU" or pollutant == "CO") else 1

    fig = go.Figure()
    for pulau, color in ISLAND_COLORS.items():
        s = d[d["pulau"] == pulau]
        if s.empty:
            continue
        v = s[col]
        fig.add_bar(
            x=s["kode"], y=v.fillna(0), name=pulau, marker_color=color,
            text=["N/A" if np.isnan(x) else "" for x in v], textposition="outside",
            textangle=-90, textfont=dict(size=8, color=TEXT_SEC), cliponaxis=False,
            customdata=[[r.lokasi, fmt(getattr(r, col), dec) + unit, fmt(r.n_valid, 0)] for r in s.itertuples()],
            hovertemplate=("<b>%{x}</b> · %{customdata[0]}<br>" + pulau +
                           "<br>%{customdata[1]}<br>Hari data valid: %{customdata[2]}<extra></extra>"),
        )

    if metric == "Konsentrasi":
        bm = POLLUTANTS[pollutant]["bm"]
        refs = [(bm, f"Baku mutu {fmt(bm, 0)}")]
    elif metric == "ISPU":
        refs = [(50, "50 · batas Baik"), (100, "100 · batas Sedang")]
    else:
        refs = [(IKU_BAIK, f"{IKU_BAIK} · batas Baik")]
    for y, label in refs:
        fig.add_hline(y=y, line_dash="dash", line_color=RED, line_width=1,
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(size=9, color=RED))

    vmax = float(np.nanmax(d[col])) if d[col].notna().any() else 0.0
    ymax = max(vmax, max(r[0] for r in refs)) * 1.12

    _layout(fig, height, dict(l=36, r=6, t=20, b=4), barmode="overlay", legend=_legend_top())
    _axes(fig, range=[0, ymax])
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), automargin=True,
                     categoryorder="array", categoryarray=order, type="category")
    return fig


# --------------------------------------------------------------------------
# 2.4 Tren harian
# --------------------------------------------------------------------------
def trend_figure(tr: pd.DataFrame, pollutant: str, height: int) -> go.Figure:
    cfg = POLLUTANTS[pollutant]
    dec = 0 if pollutant == "CO" else 1
    fig = go.Figure()
    fig.add_scatter(x=tr["tanggal"], y=tr["avg"], name="Rata-rata harian", mode="lines",
                    line=dict(color="#93C5FD", width=1), connectgaps=False,
                    hovertemplate="%{y:,." + str(dec) + "f} µg/m³<extra>Harian</extra>")
    fig.add_scatter(x=tr["tanggal"], y=tr["ma7"], name="Rata-rata bergerak 7 hari", mode="lines",
                    line=dict(color=BLUE, width=2), connectgaps=False,
                    hovertemplate="%{y:,." + str(dec) + "f} µg/m³<extra>MA 7 hari</extra>")
    fig.add_hline(y=cfg["bm"], line_dash="dash", line_color=RED, line_width=1,
                  annotation_text=f"Baku mutu {fmt(cfg['bm'], 0)} ({cfg['basis']})",
                  annotation_position="top left", annotation_font=dict(size=9, color=RED))
    top = float(np.nanmax(tr["avg"])) if tr["avg"].notna().any() else 0.0
    _layout(fig, height, dict(l=44, r=8, t=20, b=4), hovermode="x unified",
            legend=_legend_top(x=1, xanchor="right"))
    _axes(fig, range=[0, max(top, cfg["bm"]) * 1.12], title=dict(text="µg/m³", font=dict(size=9)))
    fig.update_xaxes(tickformat="%b %Y", nticks=8, hoverformat="%d %b %Y")
    return fig


# --------------------------------------------------------------------------
# 2.5 % melampaui baku mutu
# --------------------------------------------------------------------------
def exceedance_figure(ex: pd.DataFrame, height: int) -> go.Figure:
    d = ex.sort_values("pct", ascending=True)  # terbesar di atas
    fig = go.Figure(go.Bar(
        y=d["polutan"], x=d["pct"], orientation="h", marker_color=BLUE,
        text=[f"{fmt(p, 1)}% · n={fmt(n, 0)}" for p, n in zip(d["pct"], d["n"])],
        textposition="outside", textfont=dict(size=9, color=TEXT), cliponaxis=False,
        customdata=d[["n", "n_lampaui", "bm", "basis"]].to_numpy(),
        hovertemplate=("<b>%{y}</b><br>Melampaui: %{x:.2f}%<br>Pengamatan valid (n): %{customdata[0]:,}"
                       "<br>Melampaui: %{customdata[1]:,}<br>Baku mutu: %{customdata[2]:,} µg/m³ "
                       "(%{customdata[3]})<extra></extra>"),
    ))
    top = float(np.nanmax(d["pct"])) if len(d) else 0.0
    _layout(fig, height, dict(l=40, r=6, t=4, b=4))
    _axes(fig, tickfont=dict(size=10, color=TEXT))
    fig.update_xaxes(range=[0, max(top * 2.1, 5)], ticksuffix="%", showgrid=True, gridcolor=GRID)
    return fig


# --------------------------------------------------------------------------
# 2.6 Histogram kategori ISPU
# --------------------------------------------------------------------------
def ispu_figure(dist: pd.DataFrame, height: int) -> go.Figure:
    labels = {"Tidak Sehat": "Tidak<br>Sehat", "Sangat Tidak Sehat": "Sangat<br>Tidak Sehat"}
    fig = go.Figure(go.Bar(
        x=dist["kategori"], y=dist["n"],
        marker_color=[ISPU_COLORS[c] for c in dist["kategori"]],
        text=[f"{fmt(n, 0)}<br>({fmt(p, 1)}%)" for n, p in zip(dist["n"], dist["pct"])],
        textposition="outside", textfont=dict(size=9, color=TEXT), cliponaxis=False,
        customdata=dist["pct"],
        hovertemplate="<b>%{x}</b><br>Stasiun-hari: %{y:,}<br>Persentase: %{customdata:.2f}%<extra></extra>",
    ))
    top = float(dist["n"].max()) if len(dist) else 0.0
    _layout(fig, height, dict(l=40, r=6, t=4, b=4))
    _axes(fig, range=[0, max(top * 1.3, 1)])
    fig.update_xaxes(tickvals=dist["kategori"], ticktext=[labels.get(c, c) for c in dist["kategori"]],
                     tickfont=dict(size=9))
    return fig


# --------------------------------------------------------------------------
# 2.7 Boxplot IKU
# --------------------------------------------------------------------------
def iku_box_figure(t: pd.DataFrame, height: int) -> go.Figure:
    t = t.sort_values(["tahun", "kode"])
    label = {y: f"{y}*" if p else str(y)
             for y, p in t.drop_duplicates("tahun").set_index("tahun")["parsial"].items()}
    xs = t["tahun"].map(label)
    fig = go.Figure(go.Box(
        x=xs, y=t["iku"], text=t["kode"], boxpoints="all", jitter=0.5, pointpos=0,
        hoveron="points", marker=dict(color=BLUE, size=4, opacity=0.7),
        line=dict(color="#1E3A8A", width=1), fillcolor="rgba(37,99,235,0.10)",
        hovertemplate="<b>%{text}</b> · %{x}<br>IKU (indikatif): %{y:.1f}<extra></extra>",
        showlegend=False,
    ))
    _layout(fig, height, dict(l=34, r=6, t=4, b=4))
    _axes(fig, range=[0, 100])
    fig.update_xaxes(categoryorder="array", categoryarray=[label[y] for y in sorted(label)], type="category")
    return fig

"""Smoke test Halaman 2.

Jalankan dari root proyek:  python tests/smoke_page2.py

1. Memeriksa lapisan data terhadap perhitungan independen dari CSV mentah.
2. Menjalankan render_air_quality_page(). Jika streamlit/plotly tidak terpasang,
   dipakai stub ringan (hanya menguji logika Python, BUKAN tampilan sebenarnya).
"""
import importlib.util
import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("SIGAP_DATA_DIR", str(ROOT / "data"))


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install_fake_streamlit():
    st = types.ModuleType("streamlit")
    st.calls = {"metric": [], "plot": [], "markdown": [], "popover": []}
    st.session_state = {}
    st.cache_data = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
    st.markdown = lambda html, **k: st.calls["markdown"].append(html)
    st.columns = lambda spec, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.container = lambda **k: _Ctx()

    def popover(label, width="stretch", **k):
        st.calls["popover"].append(label)
        return _Ctx()

    st.popover = popover
    st.selectbox = lambda label, options, index=0, **k: options[index]
    st.date_input = lambda label, value=None, **k: value
    st.multiselect = lambda label, options, key=None, **k: st.session_state.get(key, options)
    st.button = lambda *a, **k: False
    st.segmented_control = lambda label, options, default=None, key=None, **k: default
    st.metric = lambda label, value, delta=None, **k: st.calls["metric"].append((label, value, delta))
    st.plotly_chart = lambda fig, width="stretch", key=None, **k: st.calls["plot"].append(key)
    sys.modules["streamlit"] = st
    return st


def install_fake_plotly():
    sys.modules["plotly"] = MagicMock()
    sys.modules["plotly.graph_objects"] = MagicMock()


def check_data(data):
    from utils import air_quality_data as D

    raw = pd.read_csv(Path(D.DATA_DIR) / "env_daily.csv", parse_dates=["tanggal"])
    v = raw[raw["valid_75pct"]]
    all_codes = tuple(sorted(raw["kode"].unique()))
    lo, hi = D.date_bounds()
    s, e = lo.isoformat(), hi.isoformat()
    assert len(all_codes) == 26 and len(D.load_stations()) == 26, "harus 26 stasiun"

    # station_metrics vs perhitungan independen
    sm = D.station_metrics("PM2.5", s, e, all_codes).set_index("kode")
    ref = v.groupby("kode")["pm25"].mean()
    assert np.allclose(sm["conc"], ref.reindex(sm.index)), "rata-rata PM2.5 per stasiun"
    # bobot sama per stasiun untuk pulau
    isl = D.island_means(sm.reset_index(), "conc")
    assert np.isclose(isl["Kalimantan"], ref[sm["pulau"].eq("Kalimantan")[lambda x: x].index].mean())

    # pelampauan vs flag lampaui_* dari build_env_csv.py
    ex = D.exceedance(s, e, all_codes).set_index("polutan")
    for name, flag in [("PM2.5", "lampaui_pm25"), ("PM10", "lampaui_pm10"), ("SO2", "lampaui_so2"),
                       ("NO2", "lampaui_no2"), ("CO", "lampaui_co_8j"), ("O3", "lampaui_o3_8j")]:
        x = v[flag].dropna()
        assert ex.loc[name, "n"] == len(x) and ex.loc[name, "n_lampaui"] == int(x.sum()), name

    # ISPU
    dist = D.ispu_distribution(s, e, all_codes)
    assert dist["n"].sum() == v["kategori_ispu"].notna().sum()

    # IKU: stasiun intermiten -> N/A; BJB-01 2024/2025 tidak masuk boxplot
    kpi = D.build_kpis(sm.reset_index())
    na_codes = set(sm.index[sm["iku"].isna()])
    assert {"MER-01", "MDN-01", "PLB-01", "BJB-01"} <= na_codes, na_codes
    t = D.iku_station_year(s, e, all_codes)
    assert not ((t["kode"] == "BJB-01") & t["tahun"].isin([2024, 2025])).any()
    assert t.loc[t["tahun"] == 2026, "parsial"].all() and not t.loc[t["tahun"] == 2024, "parsial"].any()
    assert t["iku"].between(0, 100).all()

    # filter sempit tidak boleh error
    for kodes, a, b in [(("MER-01",), s, e), (("JKT-01", "JKT-02"), "2024-01-01", "2024-01-03")]:
        for pol in D.POLLUTANT_ORDER:
            D.build_kpis(D.station_metrics(pol, a, b, kodes))
            D.daily_trend(pol, a, b, kodes)
        D.exceedance(a, b, kodes), D.ispu_distribution(a, b, kodes), D.iku_station_year(a, b, kodes)

    print("DATA OK | 26 stasiun | hari valid:", len(v))
    for pol in D.POLLUTANT_ORDER:
        k = D.build_kpis(D.station_metrics(pol, s, e, all_codes))
        print(f"  {pol:5s} rata-rata={D.fmt(k['mean'])}  stasiun={k['top_station'][0]} "
              f"({D.fmt(k['top_station'][2])})  pulau={k['top_island'][0]} ({D.fmt(k['top_island'][1])})")
    print("  IKU rata-rata:", D.fmt(kpi["iku_mean"]), "| n stasiun:", kpi["iku_n"],
          "| tertinggi:", kpi["iku_hi"], "| terendah:", kpi["iku_lo"])
    print("  Pelampauan (%):", {n: round(r, 2) for n, r in ex["pct"].items()})
    print("  ISPU:", dict(zip(dist["kategori"], dist["n"])))


def main():
    have_st = importlib.util.find_spec("streamlit") is not None
    have_pl = importlib.util.find_spec("plotly") is not None
    st = None
    if not have_st:
        st = install_fake_streamlit()
    if not have_pl:
        install_fake_plotly()
    print(f"streamlit asli: {have_st} | plotly asli: {have_pl}")

    import streamlit  # noqa: F401  (asli atau stub)
    check_data(None)

    if st is not None:
        from views.page2_air_quality import render_air_quality_page
        render_air_quality_page()
        assert len(st.calls["plot"]) == 6, st.calls["plot"]
        assert len(st.calls["metric"]) == 4  # KPI 1-4; KPI 5 berupa HTML
        print("RENDER (stub) OK | grafik:", st.calls["plot"])
        print("  KPI:", st.calls["metric"])
    else:
        print("Streamlit asli terpasang: gunakan streamlit.testing.v1.AppTest atau jalankan app.")


if __name__ == "__main__":
    main()

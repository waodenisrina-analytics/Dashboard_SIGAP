"""Lapisan data Halaman 2 SIGAP: Air Quality & Spatial Distribution.

Sumber: ``env_daily.csv`` (stasiun x hari; sudah memuat ISPU, ``co_8j_max``,
``o3_8j_max``, ``valid_75pct``) dan ``dim_station.csv`` (koordinat, pulau, tipe
lokasi). Relasi lewat kolom ``kode``. ``env_iku_tahunan.csv`` TIDAK dipakai:
IKU dihitung ulang dari ``env_daily`` agar patuh pada filter tanggal/stasiun
dan pada aturan hari valid (lihat catatan 5).

Catatan data yang harus diketahui pembaca kode ini
--------------------------------------------------
1. Jumlah stasiun: booklet menyebut 27, data AQMS hanya memuat 26. JMB-02
   (status Defective) ada di ``dim_station`` tetapi tidak punya data harian,
   sehingga tidak muncul. Halaman ini menampilkan apa adanya (26).
2. Hari valid = ``valid_75pct`` True (>= 18 jam data valid). Jumlah jam valid
   hanya tersedia untuk PM2.5 (``pm25_n_jam``); validitas hari untuk polutan lain
   mengikuti hari valid PM2.5 tersebut. Ini keterbatasan data, bukan pilihan.
3. Satuan: semua gas sudah dalam ug/m3 (PSW-01, MDN-01, BJB-01 sudah dikonversi
   dari ppm di hulu). Tidak ada konversi di sini.
4. Baku mutu PP 22/2021 (ug/m3): PM2.5 55, PM10 75, SO2 75, NO2 65 (24 jam);
   CO 4.000 dan O3 100 (8 jam). Pelampauan CO/O3 memakai ``co_8j_max`` dan
   ``o3_8j_max`` (maksimum harian rata-rata bergerak 8 jam), bukan rata-rata harian.
5. IKU (Permen LHK 27/2021): I_EU = (NO2/40 + SO2/20)/2;
   IKU = 100 - (50/0.9) * (I_EU - 0.1), dibatasi 0-100. Dihitung PER STASIUN
   (bukan per kota/kabupaten dari >= 4 lokasi), jadi hanya indikatif. Stasiun
   dengan hari valid < 75% pada periode terkait -> N/A. Catatan: ``env_iku_tahunan``
   memuat BJB-01 2024/2025 dengan kelengkapan 100% (berbasis NO2/SO2 non-null),
   padahal menurut hari valid hanya 68,3% / 47,9%; di sini keduanya N/A.
6. ISPU (Permen LHK 14/2020): ISPU stasiun-hari = nilai tertinggi antarparameter
   (``ispu_max``); HC tidak tersedia. Untuk peta/bar dipakai rata-rata ``ispu_max``
   pada rentang filter (ukuran indikatif).
7. Agregasi: rata-rata pulau = rata-rata dari rata-rata stasiun (bobot sama);
   tren harian = rata-rata antarstasiun yang valid pada hari itu (bobot sama).
8. Periode 2024-01-01 s.d. 2026-10-14; 2026 parsial.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.common import ISLAND_COLORS, ISPU_CATS, ISPU_COLORS, fmt  # noqa: F401  (re-ekspor)

DATA_DIR = Path(os.environ.get("SIGAP_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")

MIN_COMPLETENESS = 0.75  # syarat kelengkapan hari valid untuk IKU

# key = label filter; col = kolom rata-rata harian; exc_col = kolom pembanding baku mutu
POLLUTANTS = {
    "PM2.5": {"col": "pm25", "exc_col": "pm25", "bm": 55, "basis": "24 jam"},
    "PM10": {"col": "pm10", "exc_col": "pm10", "bm": 75, "basis": "24 jam"},
    "SO2": {"col": "so2", "exc_col": "so2", "bm": 75, "basis": "24 jam"},
    "CO": {"col": "co", "exc_col": "co_8j_max", "bm": 4000, "basis": "8 jam"},
    "O3": {"col": "o3", "exc_col": "o3_8j_max", "bm": 100, "basis": "8 jam"},
    "NO2": {"col": "no2", "exc_col": "no2", "bm": 65, "basis": "24 jam"},
}
POLLUTANT_ORDER = list(POLLUTANTS)

IKU_BAIK = 70  # batas bawah kategori Baik (Permen LHK 27/2021)

_DAILY_COLS = [
    "kode", "tanggal", "tahun", "valid_75pct",
    "pm25", "pm10", "so2", "no2", "co", "o3", "co_8j_max", "o3_8j_max",
    "ispu_max", "kategori_ispu",
]


# --------------------------------------------------------------------------
# Utilitas kecil
# --------------------------------------------------------------------------
def ispu_category(x) -> str | None:
    """Kategori ISPU dari nilai ISPU (Permen LHK 14/2020)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x <= 50:
        return "Baik"
    if x <= 100:
        return "Sedang"
    if x <= 200:
        return "Tidak Sehat"
    if x <= 300:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


def iku_score(no2, so2):
    """IKU 0-100 dari rata-rata NO2 dan SO2 (ug/m3). Indikatif, per stasiun."""
    i_eu = (no2 / 40.0 + so2 / 20.0) / 2.0
    return np.clip(100.0 - (50.0 / 0.9) * (i_eu - 0.1), 0.0, 100.0)


def _short_location(name: str) -> str:
    return name.replace("Kota Administrasi ", "").replace("Kabupaten ", "Kab. ")


# --------------------------------------------------------------------------
# Pemuatan data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    # Menggunakan Parquet, pemfilteran kolom bisa langsung via parameter 'columns'
    df = pd.read_parquet(DATA_DIR / "env_daily.parquet", columns=_DAILY_COLS)
    # Pastikan kolom tanggal di-parse kembali jika formatnya belum terbaca sebagai datetime di parquet
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df


@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    """Dimensi stasiun, hanya stasiun yang punya data harian (26 dari 27)."""
    dim = pd.read_parquet(DATA_DIR / "dim_station.parquet")
    codes = load_daily()["kode"].unique()
    dim = dim[dim["kode"].isin(codes)].copy()
    dim["lokasi"] = dim["kota_kabupaten"].map(_short_location)
    cols = ["kode", "lokasi", "kota_kabupaten", "site_type", "pulau", "latitude_dd", "longitude_dd"]
    return dim[cols].sort_values("kode").reset_index(drop=True)


def date_bounds() -> tuple[date, date]:
    t = load_daily()["tanggal"]
    return t.min().date(), t.max().date()


def _window(start: str, end: str, kodes: tuple) -> pd.DataFrame:
    df = load_daily()
    m = (df["tanggal"] >= pd.Timestamp(start)) & (df["tanggal"] <= pd.Timestamp(end))
    m &= df["kode"].isin(list(kodes))
    return df.loc[m]


# --------------------------------------------------------------------------
# Agregasi per panel
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def station_metrics(pollutant: str, start: str, end: str, kodes: tuple) -> pd.DataFrame:
    """Satu baris per stasiun: konsentrasi, IKU, ISPU, jumlah hari valid (2.1-2.3)."""
    col = POLLUTANTS[pollutant]["col"]
    w = _window(start, end, kodes)
    n_rows = w.groupby("kode").size()
    g = w[w["valid_75pct"]].groupby("kode")
    out = pd.DataFrame({
        "n_valid": g[col].count(),
        "conc": g[col].mean(),
        "no2": g["no2"].mean(),
        "so2": g["so2"].mean(),
        "ispu": g["ispu_max"].mean(),
    }).reindex(list(kodes))
    out["n_valid"] = out["n_valid"].fillna(0).astype(int)
    out["kelengkapan"] = out["n_valid"] / n_rows.reindex(out.index)
    ok = out["kelengkapan"] >= MIN_COMPLETENESS
    out["iku"] = np.where(ok, iku_score(out["no2"], out["so2"]), np.nan)
    out["ispu_kat"] = out["ispu"].map(ispu_category)
    meta = load_stations().set_index("kode")[["lokasi", "site_type", "pulau", "latitude_dd", "longitude_dd"]]
    out = out.join(meta)
    out.index.name = "kode"
    return out.reset_index()


def island_means(sm: pd.DataFrame, value_col: str = "conc") -> pd.Series:
    """Rata-rata pulau = rata-rata dari rata-rata stasiun (bobot sama)."""
    return sm.dropna(subset=[value_col]).groupby("pulau")[value_col].mean()


def build_kpis(sm: pd.DataFrame) -> dict:
    """Nilai KPI 2.1. KPI 1-3 dari 'conc' (ikut filter polutan); 4-5 dari 'iku'."""
    k = {
        "mean": np.nan, "n_days": 0, "n_st": 0,
        "top_station": None, "top_island": None,
        "iku_mean": np.nan, "iku_n": 0, "iku_total": len(sm),
        "iku_hi": None, "iku_lo": None,
    }
    c = sm.dropna(subset=["conc"])
    if len(c):
        k["mean"] = c["conc"].mean()
        k["n_st"] = len(c)
        k["n_days"] = int(c["n_valid"].sum())
        r = c.loc[c["conc"].idxmax()]
        k["top_station"] = (r["kode"], r["lokasi"], float(r["conc"]))
        isl = island_means(sm, "conc")
        k["top_island"] = (isl.idxmax(), float(isl.max()))
    q = sm.dropna(subset=["iku"])
    if len(q):
        k["iku_mean"] = q["iku"].mean()
        k["iku_n"] = len(q)
        hi, lo = q.loc[q["iku"].idxmax()], q.loc[q["iku"].idxmin()]
        k["iku_hi"] = (hi["kode"], float(hi["iku"]))
        k["iku_lo"] = (lo["kode"], float(lo["iku"]))
    return k


@st.cache_data(show_spinner=False)
def daily_trend(pollutant: str, start: str, end: str, kodes: tuple) -> pd.DataFrame:
    """Rata-rata harian antarstasiun (bobot sama) + rata-rata bergerak 7 hari (2.4)."""
    col = POLLUTANTS[pollutant]["col"]
    w = _window(start, end, kodes)
    s = w[w["valid_75pct"]].groupby("tanggal")[col].mean()
    idx = pd.date_range(start, end, freq="D")
    s = s.reindex(idx)
    return pd.DataFrame({
        "tanggal": idx,
        "avg": s.to_numpy(),
        "ma7": s.rolling(7, min_periods=4).mean().to_numpy(),
    })


@st.cache_data(show_spinner=False)
def exceedance(start: str, end: str, kodes: tuple) -> pd.DataFrame:
    """% stasiun-hari valid di atas baku mutu, per polutan (2.5). Tidak ikut filter polutan."""
    v = _window(start, end, kodes)
    v = v[v["valid_75pct"]]
    rows = []
    for name, cfg in POLLUTANTS.items():
        x = v[cfg["exc_col"]].dropna()
        n, hit = len(x), int((x > cfg["bm"]).sum())
        rows.append({
            "polutan": name, "n": n, "n_lampaui": hit,
            "pct": 100.0 * hit / n if n else np.nan,
            "bm": cfg["bm"], "basis": cfg["basis"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def ispu_distribution(start: str, end: str, kodes: tuple) -> pd.DataFrame:
    """Jumlah dan persen stasiun-hari valid per kategori ISPU (2.6)."""
    v = _window(start, end, kodes)
    v = v[v["valid_75pct"]]
    cnt = v["kategori_ispu"].value_counts().reindex(ISPU_CATS, fill_value=0)
    total = int(cnt.sum())
    return pd.DataFrame({
        "kategori": ISPU_CATS,
        "n": cnt.to_numpy(),
        "pct": (100.0 * cnt / total).to_numpy() if total else np.nan,
    })


@st.cache_data(show_spinner=False)
def iku_station_year(start: str, end: str, kodes: tuple) -> pd.DataFrame:
    """IKU per stasiun-tahun (2.7). Hanya baris dengan kelengkapan >= 75% dan IKU terhitung.

    Kolom ``parsial`` True bila jendela analisis tahun itu tidak mencakup satu tahun
    kalender penuh (mis. 2026 yang berakhir 14 Okt, atau tahun terpotong filter tanggal).
    """
    w = _window(start, end, kodes)
    n_rows = w.groupby(["kode", "tahun"]).size().rename("n_rows")
    agg = (
        w[w["valid_75pct"]]
        .groupby(["kode", "tahun"])
        .agg(n_valid=("no2", "size"), no2=("no2", "mean"), so2=("so2", "mean"))
    )
    t = pd.concat([n_rows, agg], axis=1).reset_index()
    t["n_valid"] = t["n_valid"].fillna(0)
    t["kelengkapan"] = t["n_valid"] / t["n_rows"]
    t["iku"] = np.where(t["kelengkapan"] >= MIN_COMPLETENESS, iku_score(t["no2"], t["so2"]), np.nan)
    win = w.groupby("tahun")["tanggal"].agg(["min", "max"])
    parsial = {
        y: bool(r["min"] > pd.Timestamp(int(y), 1, 1) or r["max"] < pd.Timestamp(int(y), 12, 31))
        for y, r in win.iterrows()
    }
    t["parsial"] = t["tahun"].map(parsial)
    return t.dropna(subset=["iku"]).reset_index(drop=True)

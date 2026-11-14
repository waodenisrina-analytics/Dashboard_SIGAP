"""
Utilitas data Halaman 4 SIGAP (Public Health Impact & Vulnerability).

Modul ini sengaja TIDAK mengimpor streamlit/plotly, sehingga fungsi-fungsinya
dapat diuji langsung dengan pandas + numpy.

Relasi tabel (diverifikasi pada data):
  fact_health_daily (hospital_id, tanggal)  --m:1-->  dim_station (hospital_id)
  analysis_daily    (kode, tanggal)         --m:1-->  dim_station (kode)
  hospital_id <-> kode adalah 1:1 (27 RS, 27 stasiun; JMB-02 tanpa data AQMS).
  analysis_daily = fact_health_daily yang punya AQMS (26 RS) + PM2.5 + lag/MA.
  Nilai kesehatan di kedua tabel identik (selisih 0), jadi tidak dijumlah dua kali.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------
LAGS = [f"pm25_lag{i}" for i in range(1, 8)]
LAG_LABELS = [f"lag{i}" for i in range(1, 8)]

OUTCOMES = [
    "resp_admissions_under5",
    "resp_admissions_over60",
    "asthma_emergency_cases",
    "copd_admissions",
    "cvd",
    "death",
]
OUTCOME_LABELS = {
    "resp_admissions_under5": "Resp. balita (<5)",
    "resp_admissions_over60": "Resp. lansia (>60)",
    "asthma_emergency_cases": "Asma (IGD)",
    "copd_admissions": "PPOK (rawat inap)",
    "cvd": "Kardiovaskular",
    "death": "Kematian",
    "resp": "Pernapasan (total)",
}
# Moving average di dataset: ma01 = rata-rata 2 hari (t, t-1), ma02 = 3 hari (t..t-2)
# (diverifikasi terhadap rolling mean per stasiun, selisih <= 0.001).
PM_SERIES = {
    "PM2.5 harian": "pm25",
    "MA 2 hari (ma01)": "pm25_ma01",
    "MA 3 hari (ma02)": "pm25_ma02",
}
MODEL_ROLE = "model penuh"   # stasiun yang dipakai analisis berbasis PM2.5
MIN_PAIRS = 100              # minimum pasangan hari valid untuk regresi per RS


# --------------------------------------------------------------------------
# Muat & gabung data
# --------------------------------------------------------------------------
def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (health, analysis).

    health   : fact_health_daily + atribut wilayah dari dim_station (27 RS)
    analysis : analysis_daily + site_type/rs_nama dari dim_station (26 RS ber-AQMS)
    Kolom tanggal diganti nama menjadi `date`.
    """
    d = Path(data_dir)
    st_cols = ["kode", "hospital_id", "provinsi", "site_type", "site_type_grup",
               "rs_nama", "peran_analisis"]
    
    # Diubah ke read_parquet dengan parameter 'columns'
    stn = pd.read_parquet(d / "dim_station.parquet", columns=st_cols)
    assert stn["hospital_id"].is_unique and stn["kode"].is_unique, "dim_station tidak 1:1"

    # Diubah ke read_parquet
    h = pd.read_parquet(d / "fact_health_daily.parquet")
    if "tanggal" in h.columns:
        h["tanggal"] = pd.to_datetime(h["tanggal"])
        
    h = h.drop(columns=["kode"]).merge(stn, on="hospital_id", how="left", validate="m:1")
    assert h["provinsi"].notna().all(), "ada hospital_id tanpa pasangan di dim_station"

    # Diubah ke read_parquet
    a = pd.read_parquet(d / "analysis_daily.parquet")
    if "tanggal" in a.columns:
        a["tanggal"] = pd.to_datetime(a["tanggal"])
        
    a = a.merge(stn[["kode", "site_type", "rs_nama"]], on="kode", how="left", validate="m:1")

    h = h.rename(columns={"tanggal": "date"})
    a = a.rename(columns={"tanggal": "date"}).sort_values(["kode", "date"])
    assert not h.duplicated(["hospital_id", "date"]).any()
    assert not a.duplicated(["kode", "date"]).any()
    return h, a


def apply_filters(df: pd.DataFrame, date_range=None, provinsi=None, site_group=None) -> pd.DataFrame:
    """Filter global. Daftar kosong/None = semua."""
    m = pd.Series(True, index=df.index)
    if date_range:
        m &= (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
    if provinsi:
        m &= df["provinsi"].isin(provinsi)
    if site_group:
        m &= df["site_type_grup"].isin(site_group)
    return df[m]


def model_stations(a: pd.DataFrame) -> pd.DataFrame:
    """Stasiun untuk analisis PM2.5-kesehatan (peran_analisis == 'model penuh')."""
    return a[a["peran_analisis"] == MODEL_ROLE]


# --------------------------------------------------------------------------
# 1. Metric cards  (fact_health_daily, semua RS pada filter; dijumlah sekali)
# --------------------------------------------------------------------------
def health_totals(h: pd.DataFrame) -> dict:
    n_hosp = h["hospital_id"].nunique()
    n_days = h["date"].nunique()
    s = h[["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
           "copd_admissions", "cvd", "death"]].sum()
    hd = max(len(h), 1)  # jumlah baris = RS-hari
    return {
        "n_hosp": n_hosp, "n_days": n_days,
        "under5": s["resp_admissions_under5"], "over60": s["resp_admissions_over60"],
        "asthma": s["asthma_emergency_cases"], "copd": s["copd_admissions"],
        "cvd": s["cvd"], "death": s["death"],
        "under5_rate": s["resp_admissions_under5"] / hd,
        "over60_rate": s["resp_admissions_over60"] / hd,
        "ac_rate": (s["asthma_emergency_cases"] + s["copd_admissions"]) / hd,
        "cvd_rate": s["cvd"] / hd,
    }


# --------------------------------------------------------------------------
# 2. Tren PM2.5 vs kasus
# --------------------------------------------------------------------------
def daily_trend(a: pd.DataFrame, pm_col: str = "pm25", outcome: str = "resp") -> pd.DataFrame:
    """Per tanggal: PM2.5 = rata-rata stasiun yang punya data (NaN diabaikan);
    kasus = JUMLAH seluruh RS terpilih (kelengkapan kasus 100%, jadi jumlahnya
    tidak dipengaruhi hari PM2.5 kosong)."""
    a = model_stations(a)
    out = a.groupby("date").agg(pm=(pm_col, "mean"), cases=(outcome, "sum"),
                                n_station_pm=(pm_col, "count")).reset_index()
    return out


# --------------------------------------------------------------------------
# 3. Korelasi lag (within-station)
# --------------------------------------------------------------------------
def lag_correlation(a: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Korelasi lag1..lag7 PM2.5 vs outcome, dihitung WITHIN-STATION:
    tiap variabel dipusatkan per stasiun (Spearman: diberi peringkat per stasiun
    lalu dipusatkan) agar perbedaan level antar-RS/wilayah tidak ikut terhitung.
    Pasangan dengan NaN dibuang pairwise. Hanya stasiun 'model penuh'."""
    a = model_stations(a)
    cols = OUTCOMES + LAGS
    x = a[["kode"] + cols].copy()
    if method == "spearman":
        x[cols] = x.groupby("kode")[cols].rank()
    x[cols] = x[cols] - x.groupby("kode")[cols].transform("mean")
    corr = x[cols].corr(method="pearson")
    return corr.loc[OUTCOMES, LAGS]


def best_lag(corr: pd.DataFrame) -> pd.Series:
    """Indeks kolom (0..6) dengan korelasi positif tertinggi per outcome."""
    return pd.Series(np.nanargmax(corr.values, axis=1), index=corr.index)


# --------------------------------------------------------------------------
# 4. Perbandingan kerentanan balita vs lansia
# --------------------------------------------------------------------------
def vulnerability_summary(h: pd.DataFrame, dim: str = "site_type_grup") -> pd.DataFrame:
    """Rata-rata kasus per RS-hari. Langkah: (1) rata-rata harian per RS,
    (2) rata-rata antar-RS dalam kelompok -> tiap RS berbobot sama, tanpa
    double counting. Kolom n_rs = jumlah RS dalam kelompok."""
    per_h = h.groupby(["hospital_id", dim])[["resp_admissions_under5", "resp_admissions_over60"]].mean().reset_index()
    g = per_h.groupby(dim).agg(under5=("resp_admissions_under5", "mean"),
                               over60=("resp_admissions_over60", "mean"),
                               n_rs=("hospital_id", "nunique")).reset_index()
    return g.sort_values("under5", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Sensitivitas PM2.5 per rumah sakit
# --------------------------------------------------------------------------
def _ols_slope(y: np.ndarray, x: np.ndarray):
    """OLS y = a + b x; return (b, se_robust_HC1, n). SE robust heteroskedastisitas."""
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, n
    X = np.c_[np.ones(n), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X.T * r**2) @ X
    v = xtx_inv @ meat @ xtx_inv * n / (n - 2)
    return b[1], float(np.sqrt(v[1, 1])), n


def hospital_sensitivity(a: pd.DataFrame, outcome: str = "resp", min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """Per RS: regresi sederhana outcome ~ PM2.5 (hari yang sama) pada data RS tsb
    (within-station). delta_per_10ug = 10 * slope = perubahan kasus/hari per +10 ug/m3.
    delta_pct = delta_per_10ug / rata-rata kasus harian RS * 100 (agar tidak
    bergantung ukuran RS). RS dengan pasangan valid < min_pairs dikeluarkan."""
    rows = []
    for hid, g in model_stations(a).groupby("hospital_id"):
        b, se, n = _ols_slope(g[outcome].to_numpy(float), g["pm25"].to_numpy(float))
        if n < min_pairs or np.isnan(b):
            continue
        mean_y = g.loc[g["pm25"].notna(), outcome].mean()
        d, s10 = 10 * b, 10 * se
        rows.append({"hospital_id": hid, "kode": g["kode"].iloc[0], "rs_nama": g["rs_nama"].iloc[0],
                     "provinsi": g["provinsi"].iloc[0], "n": n, "mean_y": mean_y,
                     "delta_per_10ug": d, "ci_lo": d - 1.96 * s10, "ci_hi": d + 1.96 * s10,
                     "delta_pct": 100 * d / mean_y if mean_y else np.nan,
                     "pct_lo": 100 * (d - 1.96 * s10) / mean_y if mean_y else np.nan,
                     "pct_hi": 100 * (d + 1.96 * s10) / mean_y if mean_y else np.nan})
    return pd.DataFrame(rows)

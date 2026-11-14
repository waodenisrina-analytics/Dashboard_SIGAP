"""Utilitas data Halaman 3 SIGAP (tanpa streamlit/plotly; bisa diuji terpisah).

Metode agregasi (aturan data no. 4): tiap stasiun dihitung dulu, lalu dirata-ratakan
antarstasiun dengan bobot sama (rata-rata dari rata-rata stasiun) untuk 3A.1, 3A.2, 3A.3,
3C.1, 3C.2, 3C.3. Distribusi 3A.4 dan 3B.1 memakai titik stasiun-hari.
Semua statistik hanya memakai hari valid_75pct=True; data jam dibuang bila qc_blank=1,
nilai kosong, atau flag *_obs=0 (imputasi).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = os.environ.get("SIGAP_DATA_DIR", "data")
KAWASAN = ["Perkotaan", "Industri", "Latar bersih", "Gambut", "Tepi jalan"]
KAWASAN_COLORS = {"Perkotaan": "#2563EB", "Industri": "#F97316", "Latar bersih": "#0D9488",
                  "Gambut": "#92400E", "Tepi jalan": "#7C3AED"}
MIN_HARI_KORELASI = 100
MIN_JAM_WINDROSE = 30

# Segmen ISPU (Xb, Xa, Ib, Ia): Baik -> Sangat Tidak Sehat (Permen LHK 14/2020)
ISPU_SEG = {
    "pm10": [(0, 50, 0, 50), (51, 150, 51, 100), (151, 350, 101, 200), (351, 420, 201, 300)],
    "pm25": [(0, 15.5, 0, 50), (15.5, 55.4, 50, 100), (55.5, 150.4, 101, 200), (150.5, 250.4, 201, 300)],
    "so2": [(0, 52, 0, 50), (53, 180, 51, 100), (181, 400, 101, 200), (401, 800, 201, 300)],
    "co": [(0, 4000, 0, 50), (4001, 8000, 51, 100), (8001, 15000, 101, 200), (15001, 30000, 201, 300)],
    "o3": [(0, 120, 0, 50), (121, 235, 51, 100), (236, 400, 101, 200), (401, 800, 201, 300)],
    "no2": [(0, 80, 0, 50), (81, 200, 51, 100), (201, 1130, 101, 200), (1131, 2260, 201, 300)],
}


def hitung_ispu(param, x):
    """Sub-indeks ISPU vektor. Segmen dipilih dari batas atas (X <= Xa); di atas segmen
    terakhir diekstrapolasi dengan segmen terakhir (hasil > 300 = Berbahaya)."""
    x = np.asarray(x, dtype=float)
    segs = ISPU_SEG[param]
    out = np.full(x.shape, np.nan)
    idx = np.full(x.shape, len(segs) - 1)
    for i in range(len(segs) - 1, -1, -1):
        idx = np.where(x <= segs[i][1], i, idx)
    xb, xa, ib, ia = (np.array([s[k] for s in segs], dtype=float) for k in range(4))
    r = (ia[idx] - ib[idx]) / (xa[idx] - xb[idx]) * (x - xb[idx]) + ib[idx]
    return np.where(np.isnan(x), out, r)


def _p(*parts):
    return os.path.join(DATA_DIR, *parts)

def load_daily(data_dir=None):
    filename = "env_daily.parquet"
    path = _p(filename) if data_dir is None else os.path.join(data_dir, filename)
    
    cols = [
        "kode", "tanggal", "kota_kabupaten", "site_type", "site_type_grup",
        "distance_to_road_m", "traffic_intensity_per_h", "pm25", "suhu", "rh",
        "kec_angin", "arah_angin", "ispu_max", "kategori_ispu", "valid_75pct",
        "akhir_pekan", "bulan", "musim"
    ]
    
    # Diubah ke read_parquet dengan parameter 'columns'
    d = pd.read_parquet(path, columns=cols)
    
    # Pastikan kolom tanggal di-parse sebagai datetime
    if "tanggal" in d.columns:
        d["tanggal"] = pd.to_datetime(d["tanggal"])
        
    return d[d["valid_75pct"]].dropna(subset=["site_type_grup"]).copy()


def load_stations(daily):
    """Tabel stasiun (26) + tertil lalu lintas (aturan no. 10)."""
    s = (daily.drop_duplicates("kode")[["kode", "site_type_grup", "distance_to_road_m",
                                         "traffic_intensity_per_h"]].reset_index(drop=True))
    q1, q2 = s["traffic_intensity_per_h"].quantile([1 / 3, 2 / 3])
    s["kat_lalin"] = np.where(s.traffic_intensity_per_h <= q1, "Rendah",
                              np.where(s.traffic_intensity_per_h <= q2, "Sedang", "Tinggi"))
    s.attrs["batas"] = (q1, q2)
    return s


def filter_daily(daily, start, end, kawasan):
    m = (daily.tanggal >= pd.Timestamp(start)) & (daily.tanggal <= pd.Timestamp(end))
    return daily[m & daily.site_type_grup.isin(kawasan)]


def load_hourly(daily, data_dir=None):
    """Baca kolom perlu saja, gabung ke hari valid, buang jam qc_blank/kosong/imputasi."""
    cols = ["kode", "tanggal", "jam_ke", "qc_blank", "pm25_ug_m3", "pm10_ug_m3", "so2_ug_m3",
            "co_ug_m3", "o3_ug_m3", "no2_ug_m3", "wind_speed_ms", "wind_direction_deg",
            "pm25_ug_m3_obs", "pm10_ug_m3_obs", "so2_ug_m3_obs", "co_ug_m3_obs", "o3_ug_m3_obs",
            "no2_ug_m3_obs", "wind_speed_ms_obs", "akhir_pekan", "bulan"]
    
    # Menggunakan format .parquet
    filename = "fact_aqms_hourly.parquet"
    path = _p(filename) if data_dir is None else os.path.join(data_dir, filename)
    
    # Diubah ke read_parquet dengan parameter 'columns'
    h = pd.read_parquet(path, columns=cols)
    
    # Pastikan kolom tanggal di-parse sebagai datetime
    if "tanggal" in h.columns:
        h["tanggal"] = pd.to_datetime(h["tanggal"])
        
    h = h[h.qc_blank == 0]
    valid = daily[["kode", "tanggal", "site_type_grup"]]
    h = h.merge(valid, on=["kode", "tanggal"], how="inner")
    for p in ["pm25", "pm10", "so2", "co", "o3", "no2"]:
        c = f"{p}_ug_m3"
        h.loc[h[f"{c}_obs"] == 0, c] = np.nan
    h.loc[h.wind_speed_ms_obs == 0, "wind_speed_ms"] = np.nan
    return h.drop(columns=[c for c in h.columns if c.endswith("_obs")])


def ispu_hourly(h):
    """ISPU per jam (eksplorasi): maksimum sub-indeks 6 polutan per jam. Bukan ISPU resmi."""
    subs = np.vstack([hitung_ispu(p, h[f"{p}_ug_m3"].to_numpy()) for p in ISPU_SEG])
    with np.errstate(all="ignore"):
        return np.nanmax(np.where(np.isnan(subs), -np.inf, subs), axis=0), np.isfinite(
            np.where(np.isnan(subs), -np.inf, subs).max(axis=0))


def prep_hourly(h):
    """Tambahkan kolom ispu_jam, sektor angin, kelas kecepatan."""
    v, ok = ispu_hourly(h)
    h = h.assign(ispu_jam=np.where(ok, v, np.nan))
    return h


def by_hour_weekend(h):
    """3A.1: rata-rata ISPU jam per (stasiun, jam, akhir pekan) lalu rata-rata antarstasiun."""
    g = h.dropna(subset=["ispu_jam"]).groupby(["kode", "jam_ke", "akhir_pekan"])["ispu_jam"].mean()
    return g.groupby(["jam_ke", "akhir_pekan"]).mean().unstack("akhir_pekan").rename(
        columns={False: "Hari kerja", True: "Akhir pekan"}).reset_index()


def weekend_by_kawasan(d):
    """3A.2: ISPU harian rata-rata per stasiun x (kerja/akhir pekan) -> per kawasan (bobot sama)."""
    if d.empty:
        return pd.DataFrame(columns=["site_type_grup", "Hari kerja", "Akhir pekan", "n", "selisih_pct"])
    st = d.groupby(["kode", "site_type_grup", "akhir_pekan"])["ispu_max"].mean().reset_index()
    k = st.groupby(["site_type_grup", "akhir_pekan"]).agg(ispu=("ispu_max", "mean"),
                                                         n=("kode", "nunique")).reset_index()
    w = k.pivot(index="site_type_grup", columns="akhir_pekan", values="ispu").rename(
        columns={False: "Hari kerja", True: "Akhir pekan"}).reindex(columns=["Hari kerja", "Akhir pekan"])
    w["n"] = k.groupby("site_type_grup")["n"].max()
    w["selisih_pct"] = (w["Akhir pekan"] - w["Hari kerja"]) / w["Hari kerja"] * 100
    return w.reindex([k for k in KAWASAN if k in w.index]).reset_index()


def heat_month_hour(h):
    """3A.3: PM2.5 rata-rata per stasiun x bulan x jam, lalu antarstasiun (bobot sama).
    Bulan dirata-ratakan antartahun (Nov-Des hanya 2024-2025)."""
    g = h.dropna(subset=["pm25_ug_m3"]).groupby(["kode", "bulan", "jam_ke"])["pm25_ug_m3"].mean()
    return (g.groupby(["bulan", "jam_ke"]).mean().unstack("jam_ke")
            .reindex(index=range(1, 13), columns=range(24)))


def kawasan_ispu_kpi(d):
    """3A.4: kawasan dengan rata-rata ISPU tertinggi (rata-rata antarstasiun)."""
    st = d.groupby(["site_type_grup", "kode"])["ispu_max"].mean().groupby("site_type_grup").mean()
    return st.idxmax(), float(st.max())


def kawasan_n(d):
    return d.groupby("site_type_grup")["kode"].nunique().to_dict()


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, int(m.sum())
    r = spearmanr(x[m], y[m])[0]
    return float(r), int(m.sum())


def loess(x, y, frac=0.3, n=1500, seed=0):
    """LOESS pada sampel agar cepat. Pakai statsmodels bila ada, kalau tidak, fallback lowess sederhana."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) > n:
        i = np.random.default_rng(seed).choice(len(x), n, replace=False)
        x, y = x[i], y[i]
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        r = lowess(y, x, frac=frac, return_sorted=True)
        return r[:, 0], r[:, 1]
    except ImportError:
        o = np.argsort(x); x, y = x[o], y[o]
        k = max(int(frac * len(x)), 5)
        gx = np.linspace(x[0], x[-1], 60)
        gy = []
        for g in gx:
            dist = np.abs(x - g); idx = np.argsort(dist)[:k]
            w = (1 - (dist[idx] / max(dist[idx].max(), 1e-9)) ** 3) ** 3
            gy.append(np.polyval(np.polyfit(x[idx], y[idx], 1, w=np.sqrt(w)), g))
        return gx, np.array(gy)


def met_corr(d):
    """3B.3: rho Spearman per stasiun (>=100 hari valid) PM2.5 vs suhu/rh/kec_angin."""
    rows = []
    for kode, g in d.groupby("kode"):
        if len(g) < MIN_HARI_KORELASI:
            continue
        rows.append({"kode": kode, "n": len(g), **{v: spearman(g[v].to_numpy(), g["pm25"].to_numpy())[0]
                                                   for v in ["suhu", "rh", "kec_angin"]}})
    c = pd.DataFrame(rows)
    return c


def met_kpi(c):
    """Variabel terkuat = rata-rata |rho| antarstasiun terbesar; hitung stasiun searah."""
    mv = {v: c[v].abs().mean() for v in ["suhu", "rh", "kec_angin"]}
    v = max(mv, key=mv.get)
    mean = c[v].mean()
    same = int((np.sign(c[v]) == np.sign(mean)).sum())
    return v, float(mean), same, len(c)


SPEED_BINS = [0, 1, 2, 3, 4, np.inf]
SPEED_LABELS = ["<1", "1–2", "2–3", "3–4", "≥4"]


def windrose(h, kode=None):
    """3B.2: frekuensi (% jam) dan PM2.5 rata-rata per sektor x kelas kecepatan."""
    x = h if kode in (None, "Semua stasiun") else h[h.kode == kode]
    x = x.dropna(subset=["wind_direction_deg", "wind_speed_ms"])
    sektor = np.floor(((x.wind_direction_deg.to_numpy() + 11.25) % 360) / 22.5).astype(int) % 16
    kelas = pd.cut(x.wind_speed_ms, SPEED_BINS, labels=SPEED_LABELS, right=False)
    t = pd.DataFrame({"sektor": sektor, "kelas": kelas.to_numpy(), "pm25": x.pm25_ug_m3.to_numpy()})
    tot = len(t)
    g = t.groupby(["sektor", "kelas"], observed=False).agg(n=("pm25", "size"), pm25=("pm25", "mean")).reset_index()
    g["freq"] = g.n / tot * 100 if tot else 0
    return g, tot


def station_means(d):
    """3C: rata-rata PM2.5 per stasiun (hari valid) + n hari valid."""
    s = d.groupby("kode").agg(pm25=("pm25", "mean"), hari=("pm25", "size")).reset_index()
    return s.merge(load_stations(d), on="kode")


def traffic_groups(d):
    """3C.3: PM2.5 per kategori lalu lintas (rata-rata dari rata-rata stasiun) + n stasiun."""
    sm = station_means(d)
    g = sm.groupby("kat_lalin").agg(pm25=("pm25", "mean"), n=("kode", "nunique")).reindex(
        ["Rendah", "Sedang", "Tinggi"])
    return g, sm, load_stations(d).attrs.get("batas")

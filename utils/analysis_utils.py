"""
SIGAP - Halaman 5 (Data Analysis): utilitas analitik.

Modul ini TIDAK bergantung pada Streamlit/Plotly sehingga dapat diuji terpisah.
Seluruh angka berasal dari perhitungan pada data aktual (tidak ada nilai hard-coded).

Isi:
  - pemuatan data & resolusi folder data
  - pemilihan stasiun berdasarkan filter global
  - Forecasting PM2.5 harian : model ARX dua-rezim (kemarau / non-kemarau), split waktu,
                               interval prediksi empiris dari backtest rolling-origin
  - Driver attribution       : Random Forest + permutation importance (split waktu)
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #
DATA_FILES = {
    "stations": "dim_station.parquet",
    "daily": "analysis_daily.parquet",
    "hourly": "fact_aqms_hourly.parquet",
}

# Terjemahan literal nilai kolom dim_station.status_operasional.
# (Dokumentasi dataset tidak ikut diunggah: pemetaan ini perlu dikonfirmasi.)
STATUS_MAP = {"Active": "Aktif", "Intermittent": "Terbatas", "Defective": "Rusak"}
STATUS_ORDER = ["Aktif", "Terbatas", "Rusak"]

# Fitur penggerak PM2.5 (nama kolom -> label tampilan, kelompok)
DRIVER_FEATURES = {
    "temperature_c": ("Suhu (°C)", "Meteorologi"),
    "humidity_pct": ("Kelembapan (%)", "Meteorologi"),
    "wind_speed_ms": ("Kecepatan angin (m/s)", "Meteorologi"),
    "wind_direction_deg": ("Arah angin (°)", "Meteorologi"),
    "traffic_intensity_per_h": ("Intensitas lalu lintas (per jam)", "Lokasi & transportasi"),
    "distance_to_road_m": ("Jarak ke jalan (m)", "Lokasi & transportasi"),
}

LAGS = list(range(1, 8))          # lag 1..7 hari
MIN_HISTORY = 14                  # hari riwayat minimum untuk fitur MA14
HORIZON_BINS = [(1, 7), (8, 14), (15, 30), (31, 60), (61, 90)]
INTERVAL_Q = (0.05, 0.95)         # interval prediksi 90%
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Pemuatan data
# --------------------------------------------------------------------------- #
import os
import pandas as pd

def resolve_data_dir(data_dir: str | None = None) -> str:
    """Cari folder yang memuat Parquet: argumen -> env SIGAP_DATA_DIR -> ./data -> ./ -> /mnt/user-data/uploads."""
    candidates = [data_dir, os.environ.get("SIGAP_DATA_DIR"), "data", ".", "/mnt/user-data/uploads"]
    for c in candidates:
        if c and all(os.path.exists(os.path.join(c, f)) for f in DATA_FILES.values()):
            return c
    raise FileNotFoundError(
        "File data Parquet tidak ditemukan. Letakkan "
        + ", ".join(DATA_FILES.values())
        + " di folder ./data atau set variabel lingkungan SIGAP_DATA_DIR."
    )


def load_stations(data_dir: str) -> pd.DataFrame:
    # Diubah dari pd.read_csv ke pd.read_parquet
    df = pd.read_parquet(os.path.join(data_dir, DATA_FILES["stations"]))
    df["status_label"] = df["status_operasional"].map(STATUS_MAP).fillna("Tidak diketahui")
    return df


def load_daily(data_dir: str) -> pd.DataFrame:
    # Menggunakan read_parquet dengan parameter 'columns' (bukan usecols)
    df = pd.read_parquet(
        os.path.join(data_dir, DATA_FILES["daily"]),
        columns=["kode", "tanggal", "pm25"],
    )
    # Pastikan format tanggal terbaca dengan benar setelah load dari parquet
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df.sort_values(["kode", "tanggal"]).reset_index(drop=True)


def load_hourly(data_dir: str) -> pd.DataFrame:
    cols = [
        "kode", "jam", "pm25_ug_m3", "temperature_c", "humidity_pct", "wind_speed_ms",
        "wind_direction_deg", "pm25_ug_m3_obs", "temperature_c_obs", "humidity_pct_obs",
        "wind_speed_ms_obs", "qc_blank",
    ]
    # Diubah ke read_parquet dan menggunakan 'columns'
    df = pd.read_parquet(os.path.join(data_dir, DATA_FILES["hourly"]), columns=cols)
    
    # Pastikan kolom jam terbaca sebagai datetime
    if "jam" in df.columns:
        df["jam"] = pd.to_datetime(df["jam"])

    # Hanya observasi nyata: bukan hasil imputasi dan lolos QC (bukan blank).
    ok = (
        (df["pm25_ug_m3_obs"] == 1) & (df["temperature_c_obs"] == 1)
        & (df["humidity_pct_obs"] == 1) & (df["wind_speed_ms_obs"] == 1) & (df["qc_blank"] == 0)
    )
    df = df[ok].dropna(subset=["pm25_ug_m3", "temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"])
    keep = ["kode", "jam", "pm25_ug_m3", "temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]
    return df[keep].sort_values("jam").reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Filter global
# --------------------------------------------------------------------------- #
def select_stations(stations: pd.DataFrame, provinsi=None, site_type=None, station=None) -> pd.DataFrame:
    """Terapkan filter global (wilayah / tipe lokasi / stasiun). Nilai kosong atau 'Semua' = tanpa filter."""
    def _norm(v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [x for x in v if x not in ("Semua", "All", None, "")]

    out = stations
    p, t, s = _norm(provinsi), _norm(site_type), _norm(station)
    if p:
        out = out[out["provinsi"].isin(p)]
    if t:
        out = out[out["site_type_grup"].isin(t)]
    if s:
        out = out[out["kode"].isin(s)]
    return out


# --------------------------------------------------------------------------- #
# Metrik
# --------------------------------------------------------------------------- #
def regression_metrics(actual, pred) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    if len(a) < 3:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": int(len(a))}
    e = a - p
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mae": float(np.abs(e).mean()),
        "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(len(a)),
    }


# --------------------------------------------------------------------------- #
# 1) FORECASTING
# --------------------------------------------------------------------------- #
def build_daily_series(daily: pd.DataFrame, codes: list[str], min_share: float = 0.95):
    """
    Rata-rata harian PM2.5 lintas stasiun terpilih.
    Stasiun dengan kelengkapan < min_share dikeluarkan agar komposisi stasiun tidak berubah-ubah
    (mencegah artefak pada rata-rata) dan agar fitur lag tidak berisi celah.
    Return: (series harian, stasiun dipakai, stasiun dikeluarkan)
    """
    d = daily[daily["kode"].isin(codes)]
    if d.empty:
        return None, [], list(codes)
    share = d.groupby("kode")["pm25"].apply(lambda s: s.notna().mean())
    used = share[share >= min_share].index.tolist()
    dropped = [c for c in codes if c not in used]
    if not used:
        return None, [], dropped
    y = d[d["kode"].isin(used)].groupby("tanggal")["pm25"].mean().asfreq("D")
    y = y.interpolate(limit=3, limit_area="inside")  # celah <=3 hari saja (bila ada)
    y = y.dropna()
    if not y.index.equals(pd.date_range(y.index.min(), y.index.max(), freq="D")):
        # masih ada celah panjang: ambil segmen kontinu terakhir
        full = pd.date_range(y.index.min(), y.index.max(), freq="D")
        missing = full.difference(y.index)
        y = y[y.index > missing.max()]
    return y, used, dropped


def _calendar(dates, shift_days: int = 0):
    """Fitur kalender (dummy bulan, flag kemarau Jun-Okt) dari tanggal yang digeser `shift_days`."""
    d = pd.DatetimeIndex(dates) - pd.Timedelta(days=int(shift_days))
    month_dummies = np.zeros((len(d), 11))
    for j, m in enumerate(range(2, 13)):      # Januari = referensi
        month_dummies[:, j] = (d.month == m)
    kemarau = ((d.month >= 6) & (d.month <= 10)).astype(int)   # sesuai kolom `musim` dataset
    return month_dummies, kemarau


def _hist_features(L: np.ndarray) -> np.ndarray:
    """Fitur untuk hari berikutnya; hanya memakai L (riwayat sampai t-1): lag1..7, MA7, MA14."""
    return np.concatenate([[L[-k] for k in LAGS], [L[-7:].mean(), L[-14:].mean()]])


class RegimeARX:
    """
    Regresi autoregresif (ARX) pada log1p(PM2.5) dengan dua rezim terpisah:
    kemarau (Jun-Okt) dan non-kemarau. Fitur: lag 1-7, MA7 & MA14 (dihitung dari data t-1 ke belakang,
    tanpa nilai hari ini -> tidak ada leakage) dan dummy bulan.
    Peramalan multi-langkah dilakukan secara rekursif.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.coef_: dict[int, np.ndarray] = {}
        self.icpt_: dict[int, float] = {}

    # --- tabel fitur historis leak-free
    @staticmethod
    def _table(y: pd.Series):
        L = np.log1p(y.values.astype(float))
        n = len(L)
        X = np.full((n, len(LAGS) + 2 + 11), np.nan)
        M, kem = _calendar(y.index)
        for t in range(MIN_HISTORY, n):
            X[t] = np.concatenate([_hist_features(L[:t]), M[t]])
        return X, L, kem

    def fit(self, y: pd.Series) -> "RegimeARX":
        X, L, kem = self._table(y)
        for r in (0, 1):
            ok = (kem == r) & ~np.isnan(X).any(axis=1) & ~np.isnan(L)
            if ok.sum() < 30:
                raise ValueError(f"Data rezim {'kemarau' if r else 'non-kemarau'} terlalu sedikit untuk dimodelkan.")
            m = Ridge(alpha=self.alpha).fit(X[ok], L[ok])
            self.coef_[r], self.icpt_[r] = m.coef_.copy(), float(m.intercept_)
        return self

    def _predict_row(self, x: np.ndarray, r: int) -> float:
        return float(x @ self.coef_[r] + self.icpt_[r])

    def predict_one_step(self, y: pd.Series) -> pd.Series:
        """Prediksi 1 hari ke depan untuk setiap t, memakai data aktual sampai t-1."""
        X, L, kem = self._table(y)
        p = np.full(len(L), np.nan)
        for t in range(MIN_HISTORY, len(L)):
            p[t] = self._predict_row(X[t], int(kem[t]))
        return pd.Series(np.expm1(p), index=y.index)

    def forecast_log(self, y: pd.Series, horizon: int, shift_days: int = 0) -> pd.Series:
        """Forecast rekursif (skala log1p) dari akhir `y`."""
        dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        M, kem = _calendar(dates, shift_days)
        h = list(np.log1p(y.values[-MIN_HISTORY:].astype(float)))
        out = []
        for i in range(horizon):
            x = np.concatenate([_hist_features(np.asarray(h)), M[i]])
            p = self._predict_row(x, int(kem[i]))
            out.append(p)
            h.append(p)
        return pd.Series(out, index=dates)


@dataclass
class ForecastBundle:
    y: pd.Series
    stations_used: list
    stations_dropped: list
    alpha: float
    split_dates: dict
    test: pd.DataFrame                 # date, actual, pred (1-step), baseline (persistence)
    metrics: dict
    baseline_metrics: dict
    interval_log_q: dict               # (h_lo,h_hi) -> (q_lo, q_hi) pada skala log1p
    n_calib_origins: int
    final_model: RegimeARX = field(repr=False, default=None)


def _horizon_quantiles(model: RegimeARX, y: pd.Series, first_origin: int, max_h: int, step: int = 3):
    """Galat log1p (aktual - prediksi) per horizon dari backtest rolling-origin; kuantil per kelompok horizon."""
    n = len(y)
    L_all = np.log1p(y.values.astype(float))
    errs = {h: [] for h in range(1, max_h + 1)}
    origins = list(range(first_origin, n - 1, step))
    for o in origins:
        hmax = min(max_h, n - 1 - o)
        f = model.forecast_log(y.iloc[: o + 1], hmax).values
        e = L_all[o + 1: o + 1 + hmax] - f
        for h in range(1, hmax + 1):
            errs[h].append(e[h - 1])
    q = {}
    for lo, hi in HORIZON_BINS:
        pool = np.concatenate([errs[h] for h in range(lo, hi + 1) if errs[h]] or [np.array([0.0])])
        q[(lo, hi)] = (float(np.quantile(pool, INTERVAL_Q[0])), float(np.quantile(pool, INTERVAL_Q[1])))
    # interval tidak boleh menyempit saat horizon bertambah
    keys = list(q.keys())
    lows = np.minimum.accumulate([q[k][0] for k in keys])
    highs = np.maximum.accumulate([q[k][1] for k in keys])
    return {k: (float(lows[i]), float(highs[i])) for i, k in enumerate(keys)}, len(origins)


def run_forecast_pipeline(y: pd.Series, stations_used=None, stations_dropped=None, max_h: int = 90) -> ForecastBundle:
    """
    Alur:
      1) split WAKTU: 60% train | 20% kalibrasi | 20% test (urut kronologis, tanpa shuffle)
      2) tuning alpha Ridge: validasi waktu di dalam train (75/25), pilih RMSE 1-langkah terkecil
      3) refit di seluruh train, evaluasi 1-langkah pada TEST (tidak dipakai saat training)
      4) interval prediksi: kuantil galat backtest rolling-origin (periode kalibrasi+test) per horizon
      5) model akhir di-refit pada seluruh data untuk forecast ke depan
    """
    n = len(y)
    if n < 300:
        raise ValueError("Riwayat data < 300 hari: tidak cukup untuk forecasting yang dapat dipertanggungjawabkan.")
    n_test = int(round(n * 0.2))
    n_train = n - 2 * n_test
    train = y.iloc[:n_train]

    # (2) tuning alpha dengan validasi waktu di dalam train
    n_val = int(len(train) * 0.25)
    inner_tr, inner_va_idx = train.iloc[: len(train) - n_val], train.index[len(train) - n_val:]
    best_alpha, best_rmse = None, np.inf
    for a in (0.01, 0.1, 1.0, 10.0):
        try:
            m = RegimeARX(a).fit(inner_tr)
        except ValueError:
            continue
        pred = m.predict_one_step(train).loc[inner_va_idx]
        r = regression_metrics(train.loc[inner_va_idx], pred)["rmse"]
        if r < best_rmse:
            best_alpha, best_rmse = a, r
    if best_alpha is None:
        best_alpha = 0.1

    # (3) model pada train, evaluasi pada test
    model = RegimeARX(best_alpha).fit(train)
    pred_all = model.predict_one_step(y)
    test_idx = y.index[n - n_test:]
    test = pd.DataFrame({
        "date": test_idx,
        "actual": y.loc[test_idx].values,
        "pred": pred_all.loc[test_idx].values,
        "baseline": y.shift(1).loc[test_idx].values,   # persistence: PM2.5 kemarin
    })
    metrics = regression_metrics(test["actual"], test["pred"])
    baseline_metrics = regression_metrics(test["actual"], test["baseline"])

    # (4) interval prediksi empiris
    q, n_orig = _horizon_quantiles(model, y, first_origin=n - 2 * n_test, max_h=max_h)

    # (5) model akhir pada seluruh data
    final = RegimeARX(best_alpha).fit(y)

    return ForecastBundle(
        y=y, stations_used=stations_used or [], stations_dropped=stations_dropped or [],
        alpha=best_alpha,
        split_dates={
            "train": (y.index[0], y.index[n_train - 1]),
            "calib": (y.index[n_train], y.index[n - n_test - 1]),
            "test": (y.index[n - n_test], y.index[-1]),
        },
        test=test, metrics=metrics, baseline_metrics=baseline_metrics,
        interval_log_q=q, n_calib_origins=n_orig, final_model=final,
    )


def make_forecast(bundle: ForecastBundle, horizon: int, shift_days: int = 0) -> pd.DataFrame:
    """Forecast ke depan + batas interval 90% (kembali ke skala µg/m³)."""
    f = bundle.final_model.forecast_log(bundle.y, horizon, shift_days)
    lo, hi = np.zeros(horizon), np.zeros(horizon)
    for i in range(horizon):
        h = i + 1
        for (a, b), (qlo, qhi) in bundle.interval_log_q.items():
            if a <= h <= b:
                lo[i], hi[i] = qlo, qhi
                break
        else:  # horizon > 90: pakai kelompok terakhir
            lo[i], hi[i] = list(bundle.interval_log_q.values())[-1]
    out = pd.DataFrame({
        "date": f.index,
        "yhat": np.expm1(f.values),
        "lower": np.clip(np.expm1(f.values + lo), 0, None),
        "upper": np.expm1(f.values + hi),
    })
    return out


# --------------------------------------------------------------------------- #
# 3 & 4) DRIVER ATTRIBUTION (Random Forest + permutation importance)
# --------------------------------------------------------------------------- #
@dataclass
class DriverBundle:
    importance: pd.DataFrame           # feature, label, group, mean, std
    test: pd.DataFrame                 # actual, pred (µg/m³) sampel dari test set
    metrics: dict                      # rmse, mae (µg/m³), r2_log
    corr: pd.DataFrame
    n_train: int
    n_test: int
    split_time: pd.Timestamp
    constant_features: list
    n_stations: int


def run_driver_model(hourly: pd.DataFrame, stations: pd.DataFrame, codes: list[str],
                     date_range=None, max_train: int = 60_000, max_test: int = 30_000,
                     n_estimators: int = 80, n_repeats: int = 5) -> DriverBundle:
    """
    Random Forest pada log1p(PM2.5 per jam). Split WAKTU 80/20 berdasarkan stempel jam
    (latih = periode awal, uji = periode akhir). Permutation importance dihitung pada data uji.
    Variabel lalu lintas & jarak ke jalan dari dim_station (atribut tetap per stasiun).
    """
    h = hourly[hourly["kode"].isin(codes)]
    if date_range is not None and len(date_range) == 2 and date_range[0] is not None and date_range[1] is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        h = h[(h["jam"] >= start) & (h["jam"] < end)]
    static = stations[["kode", "traffic_intensity_per_h", "distance_to_road_m"]]
    h = h.merge(static, on="kode", how="left").dropna(subset=list(DRIVER_FEATURES))
    if len(h) < 2_000:
        raise ValueError("Data per jam pada filter ini terlalu sedikit (<2.000 baris) untuk Random Forest.")

    feats = list(DRIVER_FEATURES)
    cut = h["jam"].quantile(0.8)
    tr, te = h[h["jam"] <= cut], h[h["jam"] > cut]
    if len(te) < 500:
        raise ValueError("Data uji terlalu sedikit setelah split waktu.")
    trs = tr.sample(min(max_train, len(tr)), random_state=RANDOM_STATE)
    tes = te.sample(min(max_test, len(te)), random_state=RANDOM_STATE)

    rf = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=30, max_features=0.7,
        n_jobs=-1, random_state=RANDOM_STATE,
    ).fit(trs[feats], np.log1p(trs["pm25_ug_m3"]))

    y_te_log = np.log1p(tes["pm25_ug_m3"])
    pred_log = rf.predict(tes[feats])
    pi = permutation_importance(rf, tes[feats], y_te_log, n_repeats=n_repeats,
                                random_state=RANDOM_STATE, n_jobs=-1, scoring="r2")
    imp = pd.DataFrame({
        "feature": feats,
        "label": [DRIVER_FEATURES[f][0] for f in feats],
        "group": [DRIVER_FEATURES[f][1] for f in feats],
        "mean": pi.importances_mean,
        "std": pi.importances_std,
    }).sort_values("mean", ascending=False).reset_index(drop=True)

    pred = np.expm1(pred_log)
    m = regression_metrics(tes["pm25_ug_m3"], pred)
    m["r2_log"] = regression_metrics(y_te_log, pred_log)["r2"]

    constant = [f for f in feats if h[f].nunique() <= 1]
    test_out = pd.DataFrame({"actual": tes["pm25_ug_m3"].values, "pred": pred}).sample(
        min(3000, len(tes)), random_state=RANDOM_STATE)
    return DriverBundle(
        importance=imp, test=test_out, metrics=m, corr=h[feats].corr(),
        n_train=int(len(tr)), n_test=int(len(te)), split_time=cut,
        constant_features=constant, n_stations=int(h["kode"].nunique()),
    )


# --------------------------------------------------------------------------- #
# ARTEFAK PRA-HITUNG (model disimpan ke disk agar halaman tidak melatih ulang)
# --------------------------------------------------------------------------- #
ARTIFACT_VERSION = 1


def resolve_artifact_dir(artifact_dir: str | None = None, data_dir: str | None = None) -> str:
    """argumen -> env SIGAP_ARTIFACT_DIR -> <data_dir>/artifacts_page5"""
    return artifact_dir or os.environ.get("SIGAP_ARTIFACT_DIR") or os.path.join(data_dir or ".", "artifacts_page5")


def data_fingerprint(data_dir: str) -> str:
    """Sidik jari murah: ukuran + hash 64 KB awal & akhir tiap CSV. Berubah bila data berubah."""
    h = hashlib.sha1(str(ARTIFACT_VERSION).encode())
    for f in DATA_FILES.values():
        path = os.path.join(data_dir, f)
        size = os.path.getsize(path)
        h.update(f"{f}:{size}".encode())
        with open(path, "rb") as fh:
            h.update(fh.read(65536))
            fh.seek(max(size - 65536, 0))
            h.update(fh.read(65536))
    return h.hexdigest()


def codes_key(codes) -> str:
    return hashlib.sha1(",".join(sorted(codes)).encode()).hexdigest()[:16]


def _artifact_file(art_dir: str, kind: str, codes) -> str:
    return os.path.join(art_dir, f"{kind}_{codes_key(codes)}.joblib")


def save_artifact(art_dir: str, kind: str, codes, payload, fingerprint: str, **meta) -> str:
    import joblib
    os.makedirs(art_dir, exist_ok=True)
    path = _artifact_file(art_dir, kind, codes)
    joblib.dump({"fingerprint": fingerprint, "codes": sorted(codes), "payload": payload, "meta": meta}, path, compress=3)
    return path


def load_artifact(art_dir: str, kind: str, codes, fingerprint: str):
    """Return (payload, meta) bila ada & cocok dengan data saat ini; selain itu None."""
    import joblib
    path = _artifact_file(art_dir, kind, codes)
    if not os.path.exists(path):
        return None
    try:
        d = joblib.load(path)
    except Exception:
        return None
    if d.get("fingerprint") != fingerprint or d.get("codes") != sorted(codes):
        return None
    return d["payload"], d.get("meta", {})


def range_covers_span(date_range, span) -> bool:
    """True bila filter tanggal kosong atau mencakup seluruh rentang data yang dipakai saat pra-hitung."""
    if not date_range or len(date_range) != 2:
        return True
    # bandingkan per TANGGAL (filter global biasanya berupa tanggal, sedangkan span berisi stempel jam)
    return (pd.Timestamp(date_range[0]).normalize() <= pd.Timestamp(span[0]).normalize()
            and pd.Timestamp(date_range[1]).normalize() >= pd.Timestamp(span[1]).normalize())


def compute_forecast(daily: pd.DataFrame, codes):
    y, used, dropped = build_daily_series(daily, list(codes))
    if y is None:
        return None, list(dropped)
    return run_forecast_pipeline(y, used, dropped), list(dropped)

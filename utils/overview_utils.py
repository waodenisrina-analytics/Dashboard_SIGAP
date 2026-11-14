"""Lapisan data Halaman 1 SIGAP: Overview. Tanpa streamlit/plotly (mudah diuji).

Metode (ringkas)
----------------
* Periode pembanding: 1 Januari s.d. bulan-hari ``tgl_akhir`` pada tiap tahun
  (``tgl_akhir`` = tanggal maksimum ``env_daily``; dihitung dari data).
* Hanya hari ``valid_75pct = True`` yang dipakai.
* Stasiun sepadan: >= 75% hari periode (denominator = jumlah hari kalender periode)
  valid pada ketiga tahun. Komposisi stasiun sama antartahun.
* Agregasi tanpa double counting: rata-rata per stasiun dahulu, lalu rata-rata
  antarstasiun (bobot sama). Pulau = stasiun di pulau itu; Nasional = semua
  stasiun sepadan (bukan rata-rata pulau).
* Tren bulanan: rata-rata bulanan stasiun dihitung bila >= 50% hari yang tersedia
  di bulan itu valid (bulan terakhir dihitung terhadap hari yang ada s.d. tgl_akhir).
* Kasus pernapasan: jumlah ``resp`` seluruh RS (27), tanpa aturan stasiun sepadan.
* Hanya deskripsi perubahan/asosiasi; tidak ada klaim sebab-akibat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.common import fmt

BM_PM25 = 55.0
MIN_VALID_PERIOD = 0.75
MIN_VALID_MONTH = 0.50
YEARS = (2024, 2025, 2026)
ISLANDS = ["Sumatera", "Jawa", "Kalimantan", "Sulawesi", "Papua"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
BULAN_FULL = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
              "September", "Oktober", "November", "Desember"]
STATUS_ID = {"Active": "Aktif", "Intermittent": "Terbatas", "Defective": "Rusak"}
YEAR_COLORS = {2024: "#94A3B8", 2025: "#2563EB", 2026: "#DC2626"}


# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------
def fmt_tgl(ts: pd.Timestamp, year: bool = False) -> str:
    s = f"{ts.day} {BULAN_FULL[ts.month - 1]}"
    return f"{s} {ts.year}" if year else s


# --------------------------------------------------------------------------
# Muat data (hanya kolom yang diperlukan; fact_aqms_hourly TIDAK dibaca)
# --------------------------------------------------------------------------
from pathlib import Path
import pandas as pd

def load_env(data_dir) -> pd.DataFrame:
    # Diubah ke read_parquet dengan parameter 'columns'
    d = pd.read_parquet(
        Path(data_dir) / "env_daily.parquet",
        columns=["kode", "tanggal", "pulau", "site_type_grup", "pm25", "valid_75pct"]
    )
    # Pastikan kolom tanggal di-parse sebagai datetime
    if "tanggal" in d.columns:
        d["tanggal"] = pd.to_datetime(d["tanggal"])
        
    d["valid_75pct"] = d["valid_75pct"].astype(str).str.lower().isin(["true", "1"])
    d["tahun"] = d["tanggal"].dt.year
    return d


def load_health(data_dir) -> pd.DataFrame:
    # Diubah ke read_parquet
    h = pd.read_parquet(
        Path(data_dir) / "fact_health_daily.parquet",
        columns=["hospital_id", "tanggal", "resp", "kode", "aqms_tersedia"]
    )
    # Pastikan kolom tanggal di-parse sebagai datetime
    if "tanggal" in h.columns:
        h["tanggal"] = pd.to_datetime(h["tanggal"])
    return h


def load_stations(data_dir) -> pd.DataFrame:
    # Diubah ke read_parquet
    return pd.read_parquet(
        Path(data_dir) / "dim_station.parquet",
        columns=["kode", "pulau", "peran_analisis", "status_operasional"]
    )


# --------------------------------------------------------------------------
# Periode & stasiun sepadan
# --------------------------------------------------------------------------
def period_end(env: pd.DataFrame) -> pd.Timestamp:
    return env["tanggal"].max()


def in_period(dates: pd.Series, tgl_akhir: pd.Timestamp) -> pd.Series:
    """Hari 1 Jan s.d. bulan-hari tgl_akhir (tiap tahun)."""
    md = dates.dt.month * 100 + dates.dt.day
    return md <= tgl_akhir.month * 100 + tgl_akhir.day


def period_days(year: int, tgl_akhir: pd.Timestamp) -> int:
    return (pd.Timestamp(year, tgl_akhir.month, tgl_akhir.day) - pd.Timestamp(year, 1, 1)).days + 1


def period_frame(env: pd.DataFrame, tgl_akhir: pd.Timestamp) -> pd.DataFrame:
    return env[in_period(env["tanggal"], tgl_akhir) & env["tahun"].isin(YEARS)]


def comparable_stations(env: pd.DataFrame, tgl_akhir: pd.Timestamp) -> list[str]:
    """Stasiun dengan >= 75% hari periode valid pada ketiga tahun."""
    p = period_frame(env, tgl_akhir)
    v = p[p["valid_75pct"]].groupby(["kode", "tahun"]).size().unstack(fill_value=0)
    ok = []
    for kode, row in v.iterrows():
        if all(row.get(y, 0) >= MIN_VALID_PERIOD * period_days(y, tgl_akhir) for y in YEARS):
            ok.append(kode)
    return sorted(ok)


def _valid_period(env, tgl_akhir, stations):
    p = period_frame(env, tgl_akhir)
    return p[p["valid_75pct"] & p["kode"].isin(stations) & p["pm25"].notna()]


def station_year_stats(env, tgl_akhir, stations) -> pd.DataFrame:
    """Per stasiun x tahun: rata-rata PM2.5 dan % hari valid > 55."""
    p = _valid_period(env, tgl_akhir, stations)
    g = p.groupby(["kode", "pulau", "tahun"])
    out = g.agg(pm25=("pm25", "mean"), pct55=("pm25", lambda s: (s > BM_PM25).mean() * 100),
                n_hari=("pm25", "size")).reset_index()
    return out


# --------------------------------------------------------------------------
# Komponen 3 (bar) & KPI
# --------------------------------------------------------------------------
def island_year_means(env, tgl_akhir, stations) -> pd.DataFrame:
    """Rata-rata PM2.5 periode per pulau x tahun (rata-rata dari rata-rata stasiun)."""
    s = station_year_stats(env, tgl_akhir, stations)
    r = s.groupby(["pulau", "tahun"]).agg(pm25=("pm25", "mean"), n=("kode", "nunique")).reset_index()
    r["pulau"] = pd.Categorical(r["pulau"], ISLANDS, ordered=True)
    r = r.sort_values(["pulau", "tahun"]).reset_index(drop=True)
    r["pulau"] = r["pulau"].astype(str)
    return r


def island_change(env, tgl_akhir, stations) -> pd.DataFrame:
    """Per pulau: nilai 2025, 2026, delta absolut, delta %, n stasiun."""
    m = island_year_means(env, tgl_akhir, stations).pivot(index="pulau", columns="tahun", values="pm25")
    n = island_year_means(env, tgl_akhir, stations).groupby("pulau")["n"].max()
    df = pd.DataFrame({"v2024": m[2024], "v2025": m[2025], "v2026": m[2026], "n": n})
    df["delta"] = df["v2026"] - df["v2025"]
    df["delta_pct"] = df["delta"] / df["v2025"] * 100
    return df.reindex([i for i in ISLANDS if i in df.index])


def kpi_values(env, health, tgl_akhir, stations) -> dict:
    s = station_year_stats(env, tgl_akhir, stations)
    nat = s.groupby("tahun").agg(pm25=("pm25", "mean"), pct55=("pct55", "mean"))
    piv = s.pivot(index="kode", columns="tahun", values="pm25")
    chg = (piv[2026] - piv[2025]) / piv[2025] * 100
    hp = health[in_period(health["tanggal"], tgl_akhir) & health["tahun"].isin(YEARS)] \
        if "tahun" in health else None
    if hp is None:
        h = health.assign(tahun=health["tanggal"].dt.year)
        hp = h[in_period(h["tanggal"], tgl_akhir) & h["tahun"].isin(YEARS)]
    resp = hp.groupby("tahun")["resp"].sum()
    return {
        "n": len(piv),
        "pm25": nat["pm25"].to_dict(),
        "pm25_delta": nat.loc[2026, "pm25"] - nat.loc[2025, "pm25"],
        "pm25_delta_pct": (nat.loc[2026, "pm25"] / nat.loc[2025, "pm25"] - 1) * 100,
        "pct55": nat["pct55"].to_dict(),
        "pct55_delta_pp": nat.loc[2026, "pct55"] - nat.loc[2025, "pct55"],
        "n_naik10": int((chg > 10).sum()),
        "n_turun10": int((chg < -10).sum()),
        "resp": resp.to_dict(),
        "resp_delta": resp[2026] - resp[2025],
        "resp_delta_pct": (resp[2026] / resp[2025] - 1) * 100,
        "n_rs": int(hp["hospital_id"].nunique()),
    }


# --------------------------------------------------------------------------
# Komponen 2: tren bulanan
# --------------------------------------------------------------------------
def monthly_trend(env, tgl_akhir, stations) -> pd.DataFrame:
    """Rata-rata bulanan per grup (Nasional + pulau) x tahun x bulan.

    Kolom: grup, tahun, bulan, pm25, n (stasiun), parsial (bool).
    """
    e = env[env["kode"].isin(stations)].copy()
    e["bulan"] = e["tanggal"].dt.month
    # hari tersedia per bulan (kalender, dibatasi data min..max)
    tmin, tmax = env["tanggal"].min(), env["tanggal"].max()
    days = pd.Series(pd.date_range(tmin, tmax))
    avail = days.groupby([days.dt.year, days.dt.month]).size()
    avail.index.names = ["tahun", "bulan"]
    v = e[e["valid_75pct"] & e["pm25"].notna()]
    sm = v.groupby(["kode", "pulau", "tahun", "bulan"]).agg(pm25=("pm25", "mean"), nv=("pm25", "size")).reset_index()
    sm["avail"] = [avail.get((t, b), np.nan) for t, b in zip(sm["tahun"], sm["bulan"])]
    sm = sm[sm["nv"] >= MIN_VALID_MONTH * sm["avail"]]
    parts = [sm.assign(grup="Nasional")] + [sm[sm["pulau"] == i].assign(grup=i) for i in ISLANDS]
    allp = pd.concat(parts)
    out = allp.groupby(["grup", "tahun", "bulan"]).agg(pm25=("pm25", "mean"), n=("kode", "nunique")).reset_index()
    out["parsial"] = (out["tahun"] == tgl_akhir.year) & (out["bulan"] == tgl_akhir.month) & \
                     (tgl_akhir.day < tgl_akhir.days_in_month)
    return out


# --------------------------------------------------------------------------
# Komponen 4: Top 3 temuan (teks dari data)
# --------------------------------------------------------------------------
def top_findings(env, tgl_akhir, stations) -> list[dict]:
    ic = island_change(env, tgl_akhir, stations)
    tgl = fmt_tgl(tgl_akhir)
    yr = tgl_akhir.year
    prev = yr - 1
    out = []

    # 1. kenaikan absolut terbesar
    top = ic["delta"].idxmax()
    r = ic.loc[top]
    verb = "naik" if r["delta"] > 0 else "turun"
    out.append({"label": "Perubahan terbesar",
                "text": f"{top}: {fmt(r['v2026'])} µg/m³ pada 1 Jan–{tgl} {yr}, {verb} "
                        f"{fmt(abs(r['delta_pct']))}% dari {fmt(r['v2025'])} ({prev})."})

    # 2. puncak bulanan pulau
    mt = monthly_trend(env, tgl_akhir, stations)
    mt = mt[mt["grup"].isin(ISLANDS)]
    b = mt.loc[mt["pm25"].idxmax()]
    par = " (parsial)" if b["parsial"] else ""
    out.append({"label": "Puncak bulanan",
                "text": f"Rata-rata bulanan tertinggi: {b['grup']}, {BULAN_FULL[int(b['bulan']) - 1]} "
                        f"{int(b['tahun'])}: {fmt(b['pm25'])} µg/m³{par}."})

    # 3. pulau stabil
    st_ = [i for i in ic.index if abs(ic.loc[i, "delta_pct"]) < 5]
    if st_:
        txt = f"{', '.join(st_)}: perubahan < 5% dibanding {prev}."
    else:
        txt = "Tidak ada pulau yang stabil (< 5%)."
    out.append({"label": "Pulau stabil", "text": txt})
    return out


def footnote_text(ic: pd.DataFrame, tgl_akhir: pd.Timestamp) -> str:
    """Catatan keterbatasan D.9 (a, c, d); nama pulau diturunkan dari data."""
    naik = [i for i in ic.index if ic.loc[i, "delta_pct"] >= 5]
    datar = [i for i in ic.index if abs(ic.loc[i, "delta_pct"]) < 5]
    s = ("Angka Overview memakai periode tetap dan hanya stasiun sepadan, sehingga dapat berbeda dari Halaman 2 "
         "(semua stasiun valid, rentang pilihan pengguna). ")
    if naik:
        s += f"Perubahan {tgl_akhir.year} vs {tgl_akhir.year - 1} terpusat pada {', '.join(naik)}"
        s += f"; {', '.join(datar)} relatif datar. " if datar else ". "
    s += "Data kesehatan hanya menunjukkan asosiasi waktu (tanpa koreksi cuaca/musim, tanpa denominator populasi)."
    return s


# --------------------------------------------------------------------------
# Komponen 5: data confidence
# --------------------------------------------------------------------------
def confidence(env, stations_df, comparable: list[str]) -> dict:
    n_aqms = env["kode"].nunique()
    excl = sorted(set(env["kode"].unique()) - set(comparable))
    peran = stations_df["peran_analisis"].value_counts().to_dict()
    status = stations_df["status_operasional"].value_counts().to_dict()
    return {
        "n_aqms": n_aqms,
        "n_comp": len(comparable),
        "excluded": excl,
        "peran": {k: int(peran.get(k, 0)) for k in ("model penuh", "jaringan saja", "dikeluarkan")},
        "status": {k: int(status.get(k, 0)) for k in ("Active", "Intermittent", "Defective")},
        "valid_pct": float(env["valid_75pct"].mean() * 100),
        "valid_n": int(env["valid_75pct"].sum()),
        "total_n": int(len(env)),
    }


def build_all(data_dir) -> dict:
    """Satu pintu: semua hasil Halaman 1 (dipanggil dari view dan smoke test)."""
    env, health, stn = load_env(data_dir), load_health(data_dir), load_stations(data_dir)
    tgl = period_end(env)
    comp = comparable_stations(env, tgl)
    return {
        "tgl_akhir": tgl, "stations": comp,
        "kpi": kpi_values(env, health, tgl, comp),
        "island_bar": island_change(env, tgl, comp),
        "bar_long": island_year_means(env, tgl, comp),
        "monthly": monthly_trend(env, tgl, comp),
        "findings": top_findings(env, tgl, comp),
        "conf": confidence(env, stn, comp),
    }

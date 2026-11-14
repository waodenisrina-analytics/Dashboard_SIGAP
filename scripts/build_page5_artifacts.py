"""
Pra-hitung model Halaman 5 SIGAP dan simpan ke disk.

Jalankan sekali (dan ulangi bila CSV berubah):
    python scripts/build_page5_artifacts.py --data-dir data
    # opsional: --out folder_artefak

Cakupan: semua stasiun, tiap provinsi, tiap tipe lokasi (site_type_grup), dan tiap stasiun.
Himpunan stasiun yang identik dipakai ulang (kunci = hash kode stasiun).
Filter tanggal khusus untuk Random Forest tidak di-pra-hitung -> halaman melatih langsung untuk kasus itu.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analysis_utils as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data_dir = U.resolve_data_dir(a.data_dir)
    out = U.resolve_artifact_dir(a.out, data_dir)
    fp = U.data_fingerprint(data_dir)
    stations, daily = U.load_stations(data_dir), U.load_daily(data_dir)
    hourly = U.load_hourly(data_dir)

    scopes = {"semua": stations["kode"].tolist()}
    for k, g in stations.groupby("provinsi"):
        scopes[f"provinsi:{k}"] = g["kode"].tolist()
    for k, g in stations.groupby("site_type_grup"):
        scopes[f"tipe:{k}"] = g["kode"].tolist()
    for c in stations["kode"]:
        scopes[f"stasiun:{c}"] = [c]

    seen, t0 = set(), time.time()
    for name, codes in scopes.items():
        key = U.codes_key(codes)
        if key in seen:
            continue
        seen.add(key)
        t = time.time()
        bundle, dropped = U.compute_forecast(daily, codes)
        U.save_artifact(out, "forecast", codes, (bundle, dropped), fp)
        status = "forecast ok" if bundle is not None else "forecast: data tidak memadai"
        try:
            db = U.run_driver_model(hourly, stations, codes)
            span = (hourly[hourly["kode"].isin(codes)]["jam"].min(), hourly[hourly["kode"].isin(codes)]["jam"].max())
            U.save_artifact(out, "driver", codes, db, fp, span=span)
            status += " | driver ok"
        except ValueError as e:
            status += f" | driver dilewati ({e})"
        print(f"[{len(seen):>2}] {name:<38} {status}  ({time.time() - t:4.1f}s)")
    print(f"Selesai: {len(seen)} artefak di '{out}' dalam {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

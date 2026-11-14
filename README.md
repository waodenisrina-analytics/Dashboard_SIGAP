# SIGAP – Air Quality & Public Health Dashboard

**Link dashboard** : https://dashboard-sigap.streamlit.app/

Dashboard Streamlit lima halaman: kualitas udara (PM2.5 dan polutan lain) di 26 stasiun AQMS dan beban kasus pernapasan di 27 RS, periode 2024-01-01 s.d. 2026-10-14.

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py        
```

## Halaman
| # | Menu | Modul |
|---|---|---|
| 1 | Overview | `views/page1_overview.py` |
| 2 | Air Quality & Spatial Distribution | `views/page2_air_quality.py` |
| 3 | Pollution Patterns & Drivers | `views/page3_pollution_patterns.py` |
| 4 | Public Health Impact & Vulnerability | `views/page4_public_health.py` |
| 5 | Data Analysis | `views/page5_data_analysis.py` |

## Struktur
```
app.py            router + sidebar (logo Gelar Rasa)
views/            satu file per halaman (tampilan Streamlit/Plotly)
utils/            data & perhitungan per halaman, tanpa UI; common.py = konstanta/warna/fmt bersama
data/             Parquet sumber (env_daily, analysis_daily, fact_health_daily, fact_aqms_hourly, dim_station)
assets/           logo
scripts/          build_page5_artifacts.py (opsional: pra-hitung model Halaman 5 -> data/artifacts_page5)
tests/            smoke test (python tests/smoke_page1.py ... smoke_page3.py)
.streamlit/       tema terang
```

## Aturan data
- 26 stasiun AQMS (JMB-02 rusak, tanpa data); 27 RS. Hari valid = `valid_75pct` (≥ 18 jam data).
- Rata-rata wilayah = rata-rata dari rata-rata stasiun (bobot sama).
- Baku mutu PP 22/2021: PM2.5 55, PM10 75, SO₂ 75, NO₂ 65 µg/m³ (24 jam); CO 4.000 dan O₃ 100 µg/m³ (8 jam).
- Hasil kesehatan adalah asosiasi waktu, bukan sebab-akibat.
- Halaman 1 memakai periode tetap (1 Jan–tanggal akhir data) dan hanya stasiun sepadan, sehingga angkanya dapat berbeda dari Halaman 2.

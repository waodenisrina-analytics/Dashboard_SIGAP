"""Smoke test Halaman 1. Jalankan dari root proyek: python tests/smoke_page1.py
Bagian 1: logika data terhadap angka uji D.8 (tanpa streamlit/plotly).
Bagian 2: render dengan stub Streamlit/Plotly bila paket asli tidak terpasang
          (hanya menguji alur kode & data yang dikirim ke grafik, BUKAN tampilan).
"""
import os, sys, types
from contextlib import contextmanager
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = str(ROOT / "data")
os.environ.setdefault("SIGAP_DATA_DIR", DATA)
from utils import overview_utils as U

R = U.build_all(DATA)
k, ic, conf = R["kpi"], R["island_bar"], R["conf"]
ok = []
def chk(name, got, exp, tol=0.06):
    good = abs(got - exp) <= tol
    ok.append(good); print(("OK  " if good else "FAIL"), name, round(got, 2), "vs", exp)

chk("n sepadan", k["n"], 21, 0)
for y, e in zip(U.YEARS, (32.3, 35.7, 45.2)): chk(f"PM2.5 {y}", k["pm25"][y], e)
chk("delta PM2.5", k["pm25_delta"], 9.5); chk("delta % PM2.5", k["pm25_delta_pct"], 26.5)
for y, e in zip(U.YEARS, (15.0, 16.5, 17.7)): chk(f"%>55 {y}", k["pct55"][y], e)
chk("delta pp", k["pct55_delta_pp"], 1.1)
chk("naik>10", k["n_naik10"], 7, 0); chk("turun>10", k["n_turun10"], 0, 0)
exp = {"Jawa": (37.9, 37.9, 37.9), "Kalimantan": (31.4, 43.5, 87.4), "Papua": (18.2, 18.3, 18.2),
       "Sulawesi": (23.8, 23.9, 23.8), "Sumatera": (31.5, 35.0, 38.4)}
for i, (a, b, c) in exp.items():
    for col, e in zip(("v2024", "v2025", "v2026"), (a, b, c)): chk(f"{i} {col}", ic.loc[i, col], e)
# spesifikasi D.8 menghitung % dari nilai yang sudah dibulatkan (+100,9 / +9,7); kode ini dari nilai tak dibulatkan
chk("Kalimantan delta% (toleransi 0,3)", ic.loc["Kalimantan", "delta_pct"], 100.9, 0.3)
chk("Sumatera delta% (toleransi 0,3)", ic.loc["Sumatera", "delta_pct"], 9.7, 0.3)
m = R["monthly"]; g = m[m.grup == "Kalimantan"].set_index(["tahun", "bulan"]).pm25
for (y, b), e in {(2024, 9): 61.4, (2025, 9): 113.4, (2026, 9): 277.4, (2024, 10): 60.2, (2025, 10): 112.6, (2026, 10): 538.7}.items():
    chk(f"Kal bulanan {y}-{b}", g[(y, b)], e)
isl = m[m.grup.isin(U.ISLANDS)]; top = isl.loc[isl.pm25.idxmax()]
ok.append(top.grup == "Kalimantan" and top.tahun == 2026 and top.bulan == 10 and bool(top.parsial)); print("OK  " if ok[-1] else "FAIL", "puncak bulanan", dict(top))
for y, e in zip(U.YEARS, (197813, 206158, 221304)): chk(f"resp {y}", k["resp"][y], e, 0.5)
chk("resp delta %", k["resp_delta_pct"], 7.3); chk("n RS", k["n_rs"], 27, 0)
chk("excluded", len(conf["excluded"]), 5, 0); ok.append(conf["excluded"] == ["BJB-01", "MDN-01", "MER-01", "PLB-01", "SM2-01"])
ok.append(conf["peran"] == {"model penuh": 24, "jaringan saja": 2, "dikeluarkan": 1}); ok.append(conf["status"] == {"Active": 23, "Intermittent": 3, "Defective": 1})
chk("valid %", conf["valid_pct"], 90.2); chk("valid n", conf["valid_n"], 23863, 0); chk("total n", conf["total_n"], 26468, 0)
ok.append(str(R["tgl_akhir"].date()) == "2026-10-14")
ok.append(U.period_days(2024, R["tgl_akhir"]) == 288 and U.period_days(2025, R["tgl_akhir"]) == 287 and U.period_days(2026, R["tgl_akhir"]) == 287)
for f in R["findings"]: print("  >", f["label"], "|", f["text"])
print("DATA:", "SEMUA LULUS" if all(ok) else f"GAGAL {ok.count(False)}")

# ---------------- Bagian 2: render dengan stub ----------------
try:
    import streamlit, plotly  # noqa
    print("Streamlit/Plotly asli terpasang: jalankan juga streamlit run app.py"); sys.exit(0 if all(ok) else 1)
except ImportError:
    pass

class Obj:
    def __init__(s, **kw): s.__dict__.update(kw)
    def update(s, **kw): s.__dict__.update(kw)
class Fig:
    def __init__(s, *a, **k): s.traces, s.ann = [], []; s.layout = Obj(annotations=[])
    def add_trace(s, t, **k): s.traces.append(t)
    def add_annotation(s, **k): s.ann.append(k)
    def update_layout(s, **k): s.layout.__dict__.update(k)
    def update_xaxes(s, **k): pass
    def update_yaxes(s, **k): pass
def make_subplots(rows, cols, subplot_titles=(), **k):
    f = Fig(); f.layout.annotations = [Obj(text=t, font=None) for t in subplot_titles]; return f
go = types.SimpleNamespace(Figure=Fig, Scatter=lambda **k: Obj(**k), Bar=lambda **k: Obj(**k))
sys.modules["plotly"] = types.ModuleType("plotly")
sys.modules["plotly.graph_objects"] = go
sys.modules["plotly.subplots"] = types.SimpleNamespace(make_subplots=make_subplots)

LOG = {"html": [], "metrics": [], "charts": [], "containers": []}
@contextmanager
def _cm(): yield
class St:
    def cache_data(self, *a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
    def markdown(self, s, **k): LOG["html"].append(s)
    def metric(self, label, value, delta=None, **k): LOG["metrics"].append((label, value, delta, k))
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec); return [_cm() for _ in range(n)]
    def container(self, **k): LOG["containers"].append(k); return _cm()
    def plotly_chart(self, fig, **k): LOG["charts"].append((k.get("key"), fig))
    def error(self, m): raise RuntimeError(m)
sys.modules["streamlit"] = St()
from views import page1_overview as P
P.render_overview_page(data_dir=DATA, filters={"x": 1})
print("\nMETRIK:"); [print("  ", m[:3]) for m in LOG["metrics"]]
keys = [c["key"] for c in LOG["containers"]]; print("KARTU:", keys)
exp_keys = ["p1_card_kpi1","p1_card_kpi2","p1_card_kpi3","p1_card_kpi4","p1_card_trend","p1_card_bar","p1_card_top1","p1_card_top2","p1_card_top3","p1_card_conf"]
assert keys == exp_keys, keys
tr = dict(LOG["charts"])["p1_chart_trend"]; bar = dict(LOG["charts"])["p1_chart_bar"]
print("TREN: trace =", len(tr.traces), "(harapan 6 facet x 3 tahun = 18)", "| judul:", [a.text for a in tr.layout.annotations])
assert len(tr.traces) == 18 and len(bar.traces) == 3
last = [t for t in tr.traces if t.name == "2026"][0]; print("  2026 titik:", len(last.x), "marker:", last.marker["size"][-3:])
print("BAR x:", list(bar.traces[0].x)); print("BAR anotasi:", [a["text"] for a in bar.ann])
print("TINGGI grafik tren/bar:", tr.layout.height, bar.layout.height)
H = P.H_TITLE + P.H_KPI + P.H_CHARTS + P.H_TOP3 + P.H_CONF + 4 * 12 + 24 + 24 + 12 + 30
print("Anggaran tinggi (px) incl. catatan kaki 30:", H, "<= 900:", H <= 900)
print("STUB RENDER: LULUS")
sys.exit(0 if all(ok) else 1)

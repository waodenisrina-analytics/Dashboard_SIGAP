"""Smoke test Halaman 3 TANPA Streamlit/Plotly asli (stub MagicMock).
Menjalankan seluruh jalur data + builder grafik pada data nyata untuk menangkap error logika
(NameError, indeks, kolom hilang). TIDAK menguji API Plotly/Streamlit asli maupun tampilan."""
import sys, types
from unittest.mock import MagicMock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os; os.chdir(Path(__file__).resolve().parents[1])

class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

st = MagicMock()
st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.cache_resource = st.cache_data
st.session_state = SS()
def cols(spec, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(n)]
st.columns = cols
st.tabs = lambda names: [MagicMock() for _ in names]
st.selectbox = lambda *a, **k: st.session_state[k["key"]]
sys.modules["streamlit"] = st
for m in ("plotly", "plotly.graph_objects", "plotly.subplots"):
    sys.modules[m] = MagicMock()
import importlib
P = importlib.import_module("views.page3_pollution_patterns")
U = P.U
d0 = P._daily("data")
dmin, dmax = d0.tanggal.min().date(), d0.tanggal.max().date()
for kaw in (U.KAWASAN, ["Gambut"], ["Tepi jalan", "Industri"]):
    d = U.filter_daily(d0, dmin, dmax, kaw)
    print(kaw, len(d))
    h = P._hourly("data")
    P._tab1(d, h, dmin, dmax, tuple(kaw))
    P._tab2(d, h, dmin, dmax, tuple(kaw))
    P._tab3(d)
# filter kosong
e = U.filter_daily(d0, "2030-01-01", "2030-02-01", U.KAWASAN)
P._tab1(e, P._hourly("data"), "2030-01-01", "2030-02-01", tuple(U.KAWASAN)); P._tab2(e, P._hourly("data"), "2030-01-01", "2030-02-01", tuple(U.KAWASAN)); P._tab3(e)
print("smoke ok")

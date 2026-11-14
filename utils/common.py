"""Konstanta dan helper bersama seluruh halaman SIGAP (tanpa streamlit/plotly)."""
import numpy as np

ISLAND_COLORS = {
    "Sumatera": "#2563EB",
    "Jawa": "#F97316",
    "Kalimantan": "#0D9488",
    "Sulawesi": "#7C3AED",
    "Papua": "#DB2777",
}

ISPU_CATS = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
ISPU_COLORS = {
    "Baik": "#16A34A",
    "Sedang": "#2563EB",
    "Tidak Sehat": "#FACC15",
    "Sangat Tidak Sehat": "#DC2626",
    "Berbahaya": "#111827",
}


def fmt(x, d: int = 1, sign: bool = False) -> str:
    """Angka gaya Indonesia (titik ribuan, koma desimal); NaN/None -> 'N/A'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")

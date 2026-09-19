"""
modules/fred.py — FRED API Client (Federal Reserve Economic Data)
=================================================================
Fuente gratuita y oficial de la Reserva Federal de EE.UU.
  • Yield curve del Tesoro (nominal y real / TIPS)
  • Breakevens de inflación (5y, 10y)
  • Spreads de crédito (IG, High Yield)
  • Indicadores de curva (T10Y2Y, T10Y3M — inversión)
  • Historial de tipos Fed Funds

API key gratuita en: https://fred.stlouisfed.org/docs/api/api_key.html
Límite: 120 requests/min. Sin límite diario.
"""

import os
import datetime
import logging
import requests
import pandas as pd
import numpy as np
import streamlit as st

log = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

# ─────────────────────────────────────────────────────────────────────────────
# Series relevantes (ID → descripción, vencimiento en años)
# ─────────────────────────────────────────────────────────────────────────────

TREASURY_SERIES = {
    "DGS1MO":  ("1 mes",    1/12),
    "DGS3MO":  ("3 meses",  0.25),
    "DGS6MO":  ("6 meses",  0.5),
    "DGS1":    ("1 año",    1),
    "DGS2":    ("2 años",   2),
    "DGS3":    ("3 años",   3),
    "DGS5":    ("5 años",   5),
    "DGS7":    ("7 años",   7),
    "DGS10":   ("10 años",  10),
    "DGS20":   ("20 años",  20),
    "DGS30":   ("30 años",  30),
}

TIPS_SERIES = {
    "DFII5":   ("TIPS 5 años",   5),
    "DFII7":   ("TIPS 7 años",   7),
    "DFII10":  ("TIPS 10 años",  10),
    "DFII20":  ("TIPS 20 años",  20),
    "DFII30":  ("TIPS 30 años",  30),
}

BREAKEVEN_SERIES = {
    "T5YIE":   ("Inflación implícita 5 años",  5),
    "T10YIE":  ("Inflación implícita 10 años", 10),
}

SPREAD_SERIES = {
    "BAMLC0A0CM":   "Spread IG corporativo (OAS)",
    "BAMLH0A0HYM2": "Spread High Yield (OAS)",
    "BAMLC0A4CBBB": "Spread BBB (OAS)",
}

CURVE_INDICATORS = {
    "T10Y2Y":  "Curva 10Y - 2Y (inversión de curva)",
    "T10Y3M":  "Curva 10Y - 3M (indicador recesión)",
    "FEDFUNDS": "Fed Funds Rate",
    "SOFR":     "SOFR",
}


# ─────────────────────────────────────────────────────────────────────────────
# API key
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    return (os.getenv("FRED_API_KEY")
            or st.session_state.get("fred_api_key")
            or None)


def api_key_set() -> bool:
    return bool(_get_api_key())


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict | None:
    key = _get_api_key()
    if not key:
        return None
    try:
        r = requests.get(
            f"{FRED_BASE}/{endpoint}",
            params={"api_key": key, "file_type": "json", **params},
            timeout=12,
        )
        if not r.ok:
            log.debug("FRED %s → HTTP %d", endpoint, r.status_code)
            return None
        return r.json()
    except Exception as exc:
        log.debug("FRED error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: get one series
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_series(series_id: str,
               start_date: str | None = None,
               end_date: str | None = None,
               limit: int = 1000) -> pd.Series | None:
    """
    Descarga una serie FRED.
    Devuelve pd.Series con DatetimeIndex y valores float.
    """
    params: dict = {"series_id": series_id, "limit": limit, "sort_order": "asc"}
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    data = _get("series/observations", params)
    if not data or "observations" not in data:
        return None

    rows = [
        (obs["date"], float(obs["value"]))
        for obs in data["observations"]
        if obs["value"] not in (".", "", "NA")
    ]
    if not rows:
        return None

    dates, values = zip(*rows)
    s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    return s.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def get_latest(series_id: str) -> float | None:
    """Devuelve el último valor disponible de la serie."""
    s = get_series(series_id, limit=10)
    if s is None or s.empty:
        return None
    return float(s.iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Yield curve (snapshot actual + snapshot histórico)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_yield_curve(as_of: str | None = None) -> pd.DataFrame:
    """
    Yield curve nominal (Tesoro USA) para una fecha dada (o la más reciente).

    Devuelve DataFrame con columnas:
      tenor_label, tenor_years, yield_pct, date
    Ordenado por vencimiento.
    """
    rows = []
    end = as_of or datetime.date.today().isoformat()
    start = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    for sid, (label, tenor) in TREASURY_SERIES.items():
        s = get_series(sid, start_date=start, end_date=end, limit=10)
        if s is not None and not s.empty:
            rows.append({
                "series_id":   sid,
                "tenor_label": label,
                "tenor_years": tenor,
                "yield_pct":   float(s.iloc[-1]),
                "date":        s.index[-1].strftime("%Y-%m-%d"),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_tips_curve() -> pd.DataFrame:
    """Real yield curve (TIPS)."""
    rows = []
    today = datetime.date.today().isoformat()
    start = (pd.Timestamp(today) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    for sid, (label, tenor) in TIPS_SERIES.items():
        s = get_series(sid, start_date=start, end_date=today, limit=10)
        if s is not None and not s.empty:
            rows.append({
                "series_id":   sid,
                "tenor_label": label,
                "tenor_years": tenor,
                "real_yield":  float(s.iloc[-1]),
                "date":        s.index[-1].strftime("%Y-%m-%d"),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_breakevens() -> dict:
    """Breakeven inflation (nominal - TIPS) de 5 y 10 años."""
    result = {}
    for sid, (label, tenor) in BREAKEVEN_SERIES.items():
        v = get_latest(sid)
        result[tenor] = {"label": label, "value": v, "series_id": sid}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_credit_spreads() -> dict:
    """Spreads de crédito IG y HY en puntos básicos."""
    result = {}
    for sid, label in SPREAD_SERIES.items():
        v = get_latest(sid)
        result[sid] = {"label": label, "value": v}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_curve_indicators() -> dict:
    """T10Y2Y, T10Y3M, Fed Funds, SOFR."""
    result = {}
    for sid, label in CURVE_INDICATORS.items():
        v = get_latest(sid)
        result[sid] = {"label": label, "value": v}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Historial de la curva en fechas concretas (para overlay)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def get_curve_history_multi(years_back: int = 5) -> pd.DataFrame:
    """
    Descarga el historial de todos los tenores nominales.
    Devuelve DataFrame: index=date, columns=series_ids.
    Útil para graficar curvas históricas y calcular cambios.
    """
    end   = datetime.date.today().isoformat()
    start = (pd.Timestamp(end) - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")

    series_dict = {}
    for sid in TREASURY_SERIES:
        s = get_series(sid, start_date=start, end_date=end)
        if s is not None and not s.empty:
            series_dict[sid] = s

    if not series_dict:
        return pd.DataFrame()

    df = pd.DataFrame(series_dict)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_data(ttl=7200, show_spinner=False)
def get_spread_history(series_id: str, years_back: int = 5) -> pd.Series | None:
    """Historial de un spread de crédito."""
    end   = datetime.date.today().isoformat()
    start = (pd.Timestamp(end) - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")
    return get_series(series_id, start_date=start, end_date=end)


# ─────────────────────────────────────────────────────────────────────────────
# Estado de la API
# ─────────────────────────────────────────────────────────────────────────────

def api_status() -> dict:
    """Verifica la conexión a FRED. Devuelve {'ok', 'key_set', 'error'}."""
    if not api_key_set():
        return {"ok": False, "key_set": False, "error": "FRED_API_KEY no configurada"}

    v = get_latest("DGS10")
    if v is not None:
        return {"ok": True, "key_set": True, "error": None,
                "dgs10": v}
    return {"ok": False, "key_set": True,
            "error": "API key inválida o sin conexión a FRED"}


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: curva sintética si no hay API key
# ─────────────────────────────────────────────────────────────────────────────

def get_yield_curve_fallback() -> pd.DataFrame:
    """
    Devuelve la curva sintética del Tesoro USA sin API key.
    Valores de mercado aproximados (actualizados manualmente).
    Solo para demo / cuando FRED no está disponible.
    """
    data = [
        ("1 mes",  1/12,  5.33),
        ("3 meses", 0.25,  5.31),
        ("6 meses", 0.5,   5.18),
        ("1 año",   1,     4.91),
        ("2 años",  2,     4.59),
        ("3 años",  3,     4.44),
        ("5 años",  5,     4.36),
        ("7 años",  7,     4.41),
        ("10 años", 10,    4.47),
        ("20 años", 20,    4.80),
        ("30 años", 30,    4.67),
    ]
    df = pd.DataFrame(data, columns=["tenor_label", "tenor_years", "yield_pct"])
    df["date"] = "2025-01-15 (referencia)"
    df["series_id"] = ""
    return df

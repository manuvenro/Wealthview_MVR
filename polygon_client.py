"""
modules/polygon_client.py — Polygon.io API Client
===================================================
Fuente de precios en tiempo real (o 15min delay en plan free).

Plan gratuito: 5 req/min, datos 15min retrasados.
Plan Starter ($29/mes): ilimitado, tiempo real, WebSocket.

Funciones principales:
  get_snapshot(ticker)         → quote en tiempo real/retrasado
  get_snapshots(tickers)       → quotes de múltiples tickers (1 sola llamada)
  get_ohlcv(ticker, from, to)  → barras históricas
  get_ticker_details(ticker)   → info de la empresa
  get_market_status()          → si el mercado está abierto
  get_price(ticker)            → precio actual (float) — reemplaza yf.fast_info
  api_status()                 → {'ok', 'plan', 'key_set', 'error'}

Fallback automático a yfinance si no hay API key configurada.
"""

import os
import logging
import datetime
import requests
import streamlit as st

log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"

# ─────────────────────────────────────────────────────────────────────────────
# API key
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    return (os.getenv("POLYGON_API_KEY")
            or st.session_state.get("polygon_api_key")
            or None)


def api_key_set() -> bool:
    return bool(_get_api_key())


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None,
         timeout: int = 10) -> dict | list | None:
    key = _get_api_key()
    if not key:
        return None
    try:
        r = requests.get(
            f"{POLYGON_BASE}{path}",
            params={"apiKey": key, **(params or {})},
            timeout=timeout,
        )
        if r.status_code == 403:
            log.warning("Polygon: API key inválida o sin permiso para %s", path)
            return None
        if r.status_code == 429:
            log.warning("Polygon: rate limit alcanzado (5 req/min en plan free)")
            return None
        if not r.ok:
            log.debug("Polygon %s → HTTP %d", path, r.status_code)
            return None
        data = r.json()
        if data.get("status") in ("NOT_AUTHORIZED", "ERROR"):
            log.debug("Polygon API error: %s", data.get("message", ""))
            return None
        return data
    except requests.Timeout:
        log.debug("Polygon timeout en %s", path)
        return None
    except Exception as exc:
        log.debug("Polygon error en %s: %s", path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plan detection
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def detect_plan() -> str:
    """
    Detecta el plan de la API key:
    - 'free'    → plan gratuito (datos 15min retrasados, 5 req/min)
    - 'starter' → plan de pago (datos en tiempo real)
    - 'unknown' → no se puede determinar
    """
    if not api_key_set():
        return "none"
    data = _get("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
    if not data:
        return "unknown"
    # Free plan returns delayed=True in the snapshot
    ticker_data = (data.get("ticker") or
                   (data.get("tickers") or [{}])[0])
    delayed = ticker_data.get("lastQuote", {}).get("P") is None
    # Better heuristic: check if 'day' has today's data
    try:
        day_data = ticker_data.get("day", {})
        # If open/close are populated, likely real-time
        if day_data.get("o") and day_data.get("c"):
            return "starter"
    except Exception:
        pass
    return "free"


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot: single ticker
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)   # 1 min cache (real-time feel)
def get_snapshot(ticker: str) -> dict | None:
    """
    Snapshot completo de un ticker: precio, cambio %, volumen, OHLCV de hoy.

    Devuelve dict compatible con yfinance fast_info:
      currentPrice, previousClose, open, dayHigh, dayLow,
      volume, marketCap, changePercent, lastTradeTime,
      fiftyTwoWeekHigh, fiftyTwoWeekLow, _polygon_source=True
    """
    ticker = ticker.upper().replace("-", ".")  # BRK-B → BRK.B for Polygon
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    if not data:
        return None

    t = data.get("ticker") or {}
    day   = t.get("day") or {}
    prev  = t.get("prevDay") or {}
    last  = t.get("lastTrade") or {}
    min_  = t.get("min") or {}

    current_price = (
        _safe_float(last.get("p"))
        or _safe_float(min_.get("c"))
        or _safe_float(day.get("c"))
        or None
    )
    prev_close = _safe_float(prev.get("c"))

    change_pct = None
    if current_price and prev_close and prev_close > 0:
        change_pct = (current_price - prev_close) / prev_close * 100

    return {
        "currentPrice":      current_price,
        "previousClose":     prev_close,
        "open":              _safe_float(day.get("o")),
        "dayHigh":           _safe_float(day.get("h")),
        "dayLow":            _safe_float(day.get("l")),
        "volume":            _safe_float(day.get("v")),
        "vwap":              _safe_float(day.get("vw")),
        "changePercent":     change_pct,
        "todayChangeAbs":    (current_price - prev_close) if (current_price and prev_close) else None,
        "lastTradeTime":     last.get("t"),
        "fiftyTwoWeekHigh":  _safe_float(t.get("fiftytwoWeekHigh")),
        "fiftyTwoWeekLow":   _safe_float(t.get("fiftytwoWeekLow")),
        "_polygon_source":   True,
        "_ticker_raw":       ticker,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Snapshots: batch (up to 250 tickers in one call — huge saving vs 1-by-1)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_snapshots(tickers: tuple) -> dict:
    """
    Batch snapshot para una lista de tickers.
    Devuelve {ticker: snapshot_dict}.
    Una sola llamada API para todos los tickers (hasta 250).
    """
    if not tickers or not api_key_set():
        return {}

    # Polygon accepts comma-separated tickers in the tickers param
    tickers_str = ",".join(t.upper().replace("-", ".") for t in tickers)
    data = _get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        params={"tickers": tickers_str},
    )
    if not data or "tickers" not in data:
        return {}

    result = {}
    for t in data["tickers"]:
        sym   = t.get("ticker", "").replace(".", "-")
        day   = t.get("day") or {}
        prev  = t.get("prevDay") or {}
        last  = t.get("lastTrade") or {}
        min_  = t.get("min") or {}

        current_price = (
            _safe_float(last.get("p"))
            or _safe_float(min_.get("c"))
            or _safe_float(day.get("c"))
        )
        prev_close = _safe_float(prev.get("c"))
        change_pct = None
        if current_price and prev_close and prev_close > 0:
            change_pct = (current_price - prev_close) / prev_close * 100

        result[sym] = {
            "currentPrice":  current_price,
            "previousClose": prev_close,
            "changePercent": change_pct,
            "volume":        _safe_float(day.get("v")),
            "dayHigh":       _safe_float(day.get("h")),
            "dayLow":        _safe_float(day.get("l")),
            "open":          _safe_float(day.get("o")),
            "vwap":          _safe_float(day.get("vw")),
            "_polygon_source": True,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV histórico
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv(ticker: str,
              from_date: str,
              to_date: str,
              timespan: str = "day",
              multiplier: int = 1,
              limit: int = 5000) -> "pd.DataFrame | None":
    """
    Barras OHLCV históricas.

    timespan: "minute" | "hour" | "day" | "week" | "month"
    Devuelve DataFrame con columnas: Open, High, Low, Close, Volume, VWAP
    e índice DatetimeIndex.
    """
    import pandas as pd

    ticker_pg = ticker.upper().replace("-", ".")
    data = _get(
        f"/v2/aggs/ticker/{ticker_pg}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
        params={"limit": limit, "sort": "asc", "adjusted": "true"},
    )
    if not data or not data.get("results"):
        return None

    rows = []
    for bar in data["results"]:
        rows.append({
            "Open":   bar.get("o"),
            "High":   bar.get("h"),
            "Low":    bar.get("l"),
            "Close":  bar.get("c"),
            "Volume": bar.get("v"),
            "VWAP":   bar.get("vw"),
        })
        ts = bar.get("t")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Convert timestamps (milliseconds) to dates
    timestamps = [bar.get("t") for bar in data["results"]]
    df.index = pd.to_datetime(timestamps, unit="ms", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    df.index.name = "Date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ticker details (company info)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_details(ticker: str) -> dict | None:
    """
    Detalles del ticker: nombre, mercado, sector, descripción, empleados, web.
    Endpoint: /v3/reference/tickers/{ticker}
    """
    ticker_pg = ticker.upper().replace("-", ".")
    data = _get(f"/v3/reference/tickers/{ticker_pg}")
    if not data or "results" not in data:
        return None

    r = data["results"]
    return {
        "longName":            r.get("name"),
        "shortName":           r.get("ticker"),
        "exchange":            r.get("primary_exchange"),
        "market":              r.get("market"),
        "type":                r.get("type"),          # CS, ETF, ADRC, etc.
        "currency":            r.get("currency_name", "usd").upper(),
        "country":             r.get("locale", "").upper(),
        "description":         r.get("description"),
        "fullTimeEmployees":   r.get("total_employees"),
        "website":             r.get("homepage_url"),
        "listDate":            r.get("list_date"),
        "sharesOutstanding":   _safe_float(r.get("share_class_shares_outstanding")),
        "marketCap":           _safe_float(r.get("market_cap")),
        "iconUrl":             (r.get("branding") or {}).get("icon_url"),
        "logoUrl":             (r.get("branding") or {}).get("logo_url"),
        "_polygon_source":     True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market status
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_market_status() -> dict:
    """
    Estado actual del mercado USA.
    {'market': 'open'|'closed'|'extended-hours', 'serverTime': str}
    """
    data = _get("/v1/marketstatus/now")
    if not data:
        return {"market": "unknown", "serverTime": None}
    return {
        "market":     data.get("market", "unknown"),
        "serverTime": data.get("serverTime"),
        "exchanges":  data.get("exchanges", {}),
        "currencies": data.get("currencies", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simple price getter (replaces yf.Ticker().fast_info.last_price)
# ─────────────────────────────────────────────────────────────────────────────

def get_price(ticker: str) -> float | None:
    """
    Precio actual de un ticker.
    Intenta Polygon primero, fallback a yfinance.
    """
    if api_key_set():
        snap = get_snapshot(ticker)
        if snap and snap.get("currentPrice"):
            return float(snap["currentPrice"])

    # Fallback yfinance
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        p = getattr(fi, "last_price", None)
        return float(p) if p else None
    except Exception:
        return None


def get_prices_batch(tickers: list | tuple) -> dict:
    """
    Precios de múltiples tickers en una sola llamada (Polygon) o por separado (yfinance).
    Devuelve {ticker: float}.
    """
    if not tickers:
        return {}

    if api_key_set():
        snaps = get_snapshots(tuple(tickers))
        prices = {t: s["currentPrice"] for t, s in snaps.items() if s.get("currentPrice")}
        # Fill missing with yfinance
        missing = [t for t in tickers if t not in prices]
        if missing:
            prices.update(_yf_prices_batch(missing))
        return prices

    return _yf_prices_batch(list(tickers))


def _yf_prices_batch(tickers: list) -> dict:
    """yfinance batch price fetch — fallback."""
    try:
        import yfinance as yf
        result = {}
        for t in tickers:
            try:
                fi = yf.Ticker(t).fast_info
                p = getattr(fi, "last_price", None)
                if p:
                    result[t] = float(p)
            except Exception:
                pass
        return result
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# API status & plan check
# ─────────────────────────────────────────────────────────────────────────────

def api_status() -> dict:
    """
    Verifica la conexión a Polygon.io.
    Devuelve {ok, key_set, plan, error, realtime}.
    """
    if not api_key_set():
        return {
            "ok": False, "key_set": False,
            "plan": "none", "realtime": False,
            "error": "POLYGON_API_KEY no configurada",
        }

    snap = get_snapshot("AAPL")
    if snap is None:
        return {
            "ok": False, "key_set": True,
            "plan": "unknown", "realtime": False,
            "error": "API key inválida o sin conexión",
        }

    plan = detect_plan()
    realtime = plan in ("starter", "developer", "advanced", "enterprise")

    return {
        "ok":       True,
        "key_set":  True,
        "plan":     plan,
        "realtime": realtime,
        "error":    None,
        "aapl_price": snap.get("currentPrice"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None

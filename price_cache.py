"""
modules/price_cache.py
──────────────────────────────────────────────────────────────────────────────
Caché de precios thread-safe para WealthView.

Problema resuelto:
  Cada llamada a yf.Ticker(t).fast_info.last_price es una petición HTTP
  independiente. Un portfolio de 15 tickers genera 15 peticiones en serie.
  yf.download(tickers, period="1d") obtiene todos en UNA sola petición.

Solución:
  - Caché en memoria con TTL configurable (por defecto 5 min).
  - get_prices(tickers) → batch fetch de los que no están en caché.
  - get_price(ticker)   → individual, usa caché si disponible.
  - invalidate()        → limpia la caché (útil tras añadir/eliminar posición).
  - Thread-safe: Lock protege caché compartida entre Streamlit y APScheduler.

API pública:
  get_price(ticker: str) -> float
  get_prices(tickers: Iterable[str]) -> dict[str, float]
  invalidate(ticker: str | None = None)
  cache_stats() -> dict
  set_ttl(seconds: int)
"""

import threading
import time
from datetime import datetime
from typing import Iterable

from modules.logger import get_logger

_log = get_logger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────────
_DEFAULT_TTL_SECONDS = 300   # 5 minutos
_MAX_CACHE_SIZE      = 500   # máximo de tickers en caché

# ── Estado interno ─────────────────────────────────────────────────────────────
_lock   = threading.Lock()
_ttl    = _DEFAULT_TTL_SECONDS

# _cache: {ticker_upper: {"price": float, "ts": float, "source": str}}
_cache: dict[str, dict] = {}

# Estadísticas
_stats = {"hits": 0, "misses": 0, "batch_calls": 0, "single_calls": 0, "errors": 0}


# ── TTL ────────────────────────────────────────────────────────────────────────

def set_ttl(seconds: int) -> None:
    """Cambia el TTL global de la caché."""
    global _ttl
    with _lock:
        _ttl = max(30, int(seconds))
    _log.info("price_cache TTL cambiado a %d segundos", _ttl)


def get_ttl() -> int:
    return _ttl


# ── Helpers internos ───────────────────────────────────────────────────────────

def _is_fresh(entry: dict) -> bool:
    return (time.monotonic() - entry["ts"]) < _ttl


def _store(ticker: str, price: float, source: str = "yfinance") -> None:
    """Guarda un precio en la caché (sin lock — llamar con lock adquirido)."""
    if len(_cache) >= _MAX_CACHE_SIZE:
        # Evict el más antiguo
        oldest = min(_cache, key=lambda k: _cache[k]["ts"])
        del _cache[oldest]
    _cache[ticker] = {"price": price, "ts": time.monotonic(), "source": source}


# ── Fetch batch ───────────────────────────────────────────────────────────────

def _batch_fetch(tickers: list[str]) -> dict[str, float]:
    """
    Descarga precios de cierre del último día para todos los tickers en una
    sola llamada a yf.download(). Fallback individual si el batch falla.
    Devuelve dict {ticker: price}.
    """
    if not tickers:
        return {}

    import yfinance as yf
    import pandas as pd

    _stats["batch_calls"] += 1
    result: dict[str, float] = {}

    try:
        tickers_str = " ".join(tickers)
        df = yf.download(
            tickers_str,
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            show_errors=False,
            threads=False,
        )
        if df.empty:
            raise ValueError("yf.download devolvió DataFrame vacío")

        # Estructura del DataFrame cambia según si hay 1 o N tickers
        close = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=0)

        if isinstance(close, pd.Series):
            # Un solo ticker — la serie ya es el precio
            price = float(close.dropna().iloc[-1]) if not close.dropna().empty else 0.0
            result[tickers[0]] = price
        else:
            for t in tickers:
                col = t if t in close.columns else None
                if col is None:
                    continue
                series = close[col].dropna()
                result[t] = float(series.iloc[-1]) if not series.empty else 0.0

        _log.debug("Batch fetch OK: %d tickers en 1 llamada", len(result))

    except Exception as e:
        _log.warning("Batch fetch falló (%s), pasando a fetches individuales", e)
        # Fallback: fetch individual
        for t in tickers:
            try:
                _stats["single_calls"] += 1
                price = yf.Ticker(t).fast_info.last_price
                result[t] = float(price) if price and float(price) > 0 else 0.0
            except Exception as e2:
                _log.warning("Precio individual %s falló: %s", t, e2)
                _stats["errors"] += 1
                result[t] = 0.0

    return result


# ── API pública ────────────────────────────────────────────────────────────────

def get_prices(tickers: Iterable[str]) -> dict[str, float]:
    """
    Devuelve {ticker: precio} para todos los tickers solicitados.
    Los precios frescos se sirven desde caché; los caducados/ausentes se
    obtienen en batch (una sola petición HTTP).
    """
    tickers_upper = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers_upper:
        return {}

    with _lock:
        # Separar frescos de los que necesitan fetch
        fresh:  dict[str, float] = {}
        stale:  list[str]        = []

        for t in tickers_upper:
            if t in _cache and _is_fresh(_cache[t]):
                fresh[t] = _cache[t]["price"]
                _stats["hits"] += 1
            else:
                stale.append(t)
                _stats["misses"] += 1

        if stale:
            fetched = _batch_fetch(stale)
            for t, p in fetched.items():
                _store(t, p)
            fresh.update(fetched)

    _log.debug(
        "get_prices: %d hits / %d fetched (total %d)",
        len(tickers_upper) - len(stale), len(stale), len(tickers_upper)
    )
    return fresh


def get_price(ticker: str) -> float:
    """
    Devuelve el precio de un único ticker.
    Usa la caché si está fresco; si no, fetch individual.
    Devuelve 0.0 si no se puede obtener.
    """
    t = ticker.strip().upper()
    with _lock:
        if t in _cache and _is_fresh(_cache[t]):
            _stats["hits"] += 1
            return _cache[t]["price"]
        _stats["misses"] += 1

    # Fetch individual (fuera del lock para no bloquear otras peticiones)
    import yfinance as yf
    _stats["single_calls"] += 1
    try:
        price = yf.Ticker(t).fast_info.last_price
        p = float(price) if price and float(price) > 0 else 0.0
    except Exception as e:
        _log.warning("get_price('%s') falló: %s", t, e)
        _stats["errors"] += 1
        p = 0.0

    with _lock:
        _store(t, p)

    return p


def invalidate(ticker: str | None = None) -> None:
    """
    Invalida la caché.
    - ticker=None  → borra todo.
    - ticker='AAPL' → borra solo ese ticker.
    """
    with _lock:
        if ticker is None:
            n = len(_cache)
            _cache.clear()
            _log.info("price_cache: caché completa limpiada (%d entradas)", n)
        else:
            t = ticker.strip().upper()
            if t in _cache:
                del _cache[t]
                _log.debug("price_cache: '%s' invalidado", t)


def cache_stats() -> dict:
    """
    Devuelve estadísticas de la caché:
      hits, misses, hit_rate, size, batch_calls, single_calls, errors,
      ttl_seconds, entries (lista de entradas con edad)
    """
    with _lock:
        total = _stats["hits"] + _stats["misses"]
        hit_rate = round(_stats["hits"] / total * 100, 1) if total > 0 else 0.0
        now = time.monotonic()
        entries = [
            {
                "ticker":    t,
                "price":     v["price"],
                "age_s":     round(now - v["ts"], 1),
                "fresh":     _is_fresh(v),
                "source":    v.get("source", "yfinance"),
            }
            for t, v in _cache.items()
        ]
        entries.sort(key=lambda x: x["age_s"])

        return {
            "hits":         _stats["hits"],
            "misses":       _stats["misses"],
            "hit_rate":     hit_rate,
            "size":         len(_cache),
            "batch_calls":  _stats["batch_calls"],
            "single_calls": _stats["single_calls"],
            "errors":       _stats["errors"],
            "ttl_seconds":  _ttl,
            "entries":      entries,
        }


def reset_stats() -> None:
    """Resetea los contadores de estadísticas (no borra la caché)."""
    with _lock:
        for k in _stats:            _stats[k] = 0


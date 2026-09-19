"""
modules/fx.py
─────────────────────────────────────────────────────────────────────────────
Motor de tipos de cambio para WealthView.

Funciones principales:
  get_base_currency()          → divisa base del usuario (EUR por defecto)
  get_ticker_currency(ticker)  → divisa de cotización de un ticker
  get_fx_rate(from, to)        → tipo de cambio (cuántas 'to' por 1 'from')
  convert_to_base(amount, ccy) → convierte a divisa base
  fx_table(currencies)         → DataFrame con tasas para mostrar en UI

Fuente de datos: yfinance (tickers XXXYYY=X). Caché de 1 hora.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from modules.logger import get_logger

_log = get_logger(__name__)

# ── Divisas soportadas ────────────────────────────────────────────────────────
SUPPORTED_CURRENCIES = ["EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD", "SEK", "NOK", "DKK"]
BASE_CURRENCY_DEFAULT = "EUR"
_CONFIG_PATH = os.path.join("data", "user_config.json")


# ── Configuración de usuario ──────────────────────────────────────────────────

def get_base_currency() -> str:
    """Lee la divisa base del usuario desde data/user_config.json."""
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ccy = cfg.get("base_currency", BASE_CURRENCY_DEFAULT)
            return str(ccy).upper()
    except Exception:
        pass
    return BASE_CURRENCY_DEFAULT


def set_base_currency(ccy: str):
    """Guarda la divisa base en data/user_config.json."""
    ccy = str(ccy).upper()
    os.makedirs("data", exist_ok=True)
    cfg = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg["base_currency"] = ccy
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Tipo de cambio ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rate(from_ccy: str, to_ccy: str) -> float:
    """
    Devuelve el tipo de cambio: cuántas unidades de 'to_ccy' por 1 'from_ccy'.
    Ejemplo: get_fx_rate("USD", "EUR") → 0.92  (1 USD = 0.92 EUR)

    Estrategia:
      1. Ticker directo USDEUR=X
      2. Ticker inverso EURUSD=X → 1/rate
      3. Triangulación via USD (from→USD→to)
      4. Fallback: 1.0 (sin conversión)
    """
    from_ccy = from_ccy.upper().strip()
    to_ccy   = to_ccy.upper().strip()

    if from_ccy == to_ccy:
        return 1.0

    # 1. Ticker directo
    try:
        ticker_direct = f"{from_ccy}{to_ccy}=X"
        price = yf.Ticker(ticker_direct).fast_info.last_price
        if price and float(price) > 0:
            _log.debug("FX directo %s→%s = %.6f", from_ccy, to_ccy, float(price))
            return round(float(price), 8)
    except Exception as e:
        _log.debug("FX directo %s fallido: %s", ticker_direct, e)

    # 2. Ticker inverso
    try:
        ticker_inv = f"{to_ccy}{from_ccy}=X"
        price = yf.Ticker(ticker_inv).fast_info.last_price
        if price and float(price) > 0:
            _log.debug("FX inverso %s→%s = %.6f", from_ccy, to_ccy, 1.0 / float(price))
            return round(1.0 / float(price), 8)
    except Exception as e:
        _log.debug("FX inverso %s fallido: %s", ticker_inv, e)

    # 3. Triangulación via USD
    if from_ccy != "USD" and to_ccy != "USD":
        try:
            rate_from_usd = get_fx_rate(from_ccy, "USD")
            rate_usd_to   = get_fx_rate("USD", to_ccy)
            combined = rate_from_usd * rate_usd_to
            if combined > 0:
                _log.debug("FX triangulación %s→USD→%s = %.6f", from_ccy, to_ccy, combined)
                return round(combined, 8)
        except Exception as e:
            _log.warning("FX triangulación %s→%s fallida: %s", from_ccy, to_ccy, e)

    # 4. Fallback: sin conversión
    _log.warning("FX fallback 1.0 para %s→%s (sin datos disponibles)", from_ccy, to_ccy)
    return 1.0


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_currency(ticker: str) -> str:
    """
    Devuelve la divisa de cotización de un ticker (ej: AAPL → USD, SAP.DE → EUR).
    Usa yfinance fast_info.currency. Fallback: USD.
    """
    try:
        info = yf.Ticker(ticker.strip().upper()).fast_info
        ccy = getattr(info, "currency", None)
        if ccy and isinstance(ccy, str) and len(ccy) == 3:
            return ccy.upper()
    except Exception as e:
        _log.warning("No se pudo obtener divisa para %s: %s — usando USD", ticker, e)
    return "USD"


# ── Conversión ────────────────────────────────────────────────────────────────

def convert_to_base(amount: float, from_ccy: str, base_ccy: str | None = None) -> float:
    """
    Convierte 'amount' desde 'from_ccy' a la divisa base del usuario.
    Si base_ccy es None, usa get_base_currency().
    """
    if base_ccy is None:
        base_ccy = get_base_currency()
    from_ccy = from_ccy.upper()
    base_ccy = base_ccy.upper()
    if from_ccy == base_ccy:
        return amount
    rate = get_fx_rate(from_ccy, base_ccy)
    return amount * rate


# ── Tabla de tipos de cambio para UI ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fx_table(currencies_tuple: tuple) -> pd.DataFrame:
    """
    Devuelve un DataFrame con tipos de cambio de todas las divisas del portfolio
    a la divisa base del usuario.
    Columnas: Currency, Rate (a base), Base
    """
    base = get_base_currency()
    rows = []
    for ccy in currencies_tuple:
        if ccy == base:
            rows.append({"Divisa": ccy, f"Tipo ({base})": 1.0, "Fuente": "—"})
        else:
            rate = get_fx_rate(ccy, base)
            rows.append({"Divisa": ccy, f"Tipo ({base})": round(rate, 6), "Fuente": "yfinance"})
    return pd.DataFrame(rows)



# ── Helpers de display ────────────────────────────────────────────────────────

def ccy_symbol(ccy: str) -> str:
    """Devuelve el simbolo de la divisa."""
    _MAP = {
        "EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥",
        "CHF": "Fr", "CAD": "CA$", "AUD": "A$",
        "SEK": "kr", "NOK": "kr", "DKK": "kr",
    }
    return _MAP.get(str(ccy).upper(), str(ccy).upper())


def format_value(amount: float, ccy: str, decimals: int = 0) -> str:
    """Formatea un valor con el simbolo de divisa correcto."""
    sym = ccy_symbol(ccy)
    if decimals == 0:
        return f"{sym}{amount:,.0f}"
    return f"{sym}{amount:,.{decimals}f}"

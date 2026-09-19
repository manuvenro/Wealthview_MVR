"""
modules/fmp.py — Financial Modeling Prep data layer
=====================================================
Fuente primaria para datos fundamentales profundos:
  • Cash Flow Statements  (FCF fiable)
  • Income Statements     (Revenue, EBITDA, EPS)
  • Company Profile       (Shares, Market Cap, Sector)
  • Key Metrics           (P/FCF, EV/EBITDA)
  • Ratios TTM            (Forward P/E, ROE, Márgenes)

Plan Free: 250 llamadas/día → TTL mínimo 6 horas en todas las llamadas.
Todas las funciones devuelven None en caso de error (el caller hace fallback a yfinance).

Uso:
    from modules.fmp import get_financials_fmp, get_profile_fmp

    data = get_financials_fmp("AAPL")   # dict con fcf, revenue, ebitda…
    prof = get_profile_fmp("AAPL")      # dict con shares, price, beta…
"""

import os
import sqlite3
import logging
import datetime
import requests
import streamlit as st

log = logging.getLogger(__name__)

# ── API Key ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Lee la API key desde variables de entorno o session_state."""
    key = os.getenv("FMP_API_KEY") or st.session_state.get("fmp_api_key")
    return key or None


_BASE = "https://financialmodelingprep.com/api"

# ── HTTP helper ───────────────────────────────────────────────────────────────

# ── API Usage counter (SQLite, daily reset) ──────────────────────────────────

_DAILY_LIMIT = 250

def _get_db_path() -> str:
    """Same DB as auth.py — wealthview.db at project root."""
    import modules.auth as _auth
    return _auth.DB_PATH


def _ensure_fmp_counter_table():
    """Create api_call_log table if not exists."""
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fmp_api_calls (
                date TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _increment_call_counter() -> int:
    """Increment today's call count and return the new total."""
    today = datetime.date.today().isoformat()
    try:
        conn = sqlite3.connect(_get_db_path())
        conn.execute(
            "INSERT INTO fmp_api_calls (date, count) VALUES (?, 1) "
            "ON CONFLICT(date) DO UPDATE SET count = count + 1",
            (today,)
        )
        conn.commit()
        cur = conn.execute("SELECT count FROM fmp_api_calls WHERE date=?", (today,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 1
    except Exception:
        return 0


def get_daily_call_count() -> int:
    """Return today's FMP API call count."""
    today = datetime.date.today().isoformat()
    try:
        _ensure_fmp_counter_table()
        conn = sqlite3.connect(_get_db_path())
        cur = conn.execute("SELECT count FROM fmp_api_calls WHERE date=?", (today,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _get(path: str, params: dict | None = None, timeout: int = 10) -> list | dict | None:
    """
    Hace GET a FMP API. Devuelve datos parseados o None.
    • Cuenta cada llamada real (no cacheada) contra el límite diario.
    • Avisa cuando se acerca al límite (>200/250).
    • Nunca levanta excepción al caller.
    """
    key = _get_api_key()
    if not key:
        log.warning("FMP_API_KEY no configurada. Saltando llamada a FMP.")
        return None

    # Soft rate-limit guard
    _ensure_fmp_counter_table()
    current_count = get_daily_call_count()
    if current_count >= _DAILY_LIMIT:
        log.warning("FMP: límite diario de %d llamadas alcanzado. Abortando.", _DAILY_LIMIT)
        if "fmp_limit_warned" not in st.session_state:
            st.session_state["fmp_limit_warned"] = True
            st.warning(
                f"⚠ WealthView ha alcanzado el límite diario de {_DAILY_LIMIT} llamadas "
                f"a FMP API. Los datos fundamentales usarán la caché local hasta mañana."
            )
        return None

    url = f"{_BASE}{path}"
    p = {"apikey": key, **(params or {})}
    try:
        r = requests.get(url, params=p, timeout=timeout)

        if r.status_code == 401:
            log.error("FMP: API key inválida o sin acceso al endpoint %s", path)
            return None
        if r.status_code == 429:
            log.warning("FMP: Límite de llamadas diarias alcanzado (HTTP 429).")
            return None
        if not r.ok:
            log.debug("FMP %s → HTTP %d", path, r.status_code)
            return None

        data = r.json()
        if isinstance(data, list) and len(data) == 0:
            return None
        if isinstance(data, dict) and data.get("Error Message"):
            log.debug("FMP error: %s", data["Error Message"])
            return None

        # Count successful call
        new_count = _increment_call_counter()
        if new_count >= 200 and "fmp_warn_200" not in st.session_state:
            st.session_state["fmp_warn_200"] = True
            log.info("FMP: %d/%d llamadas usadas hoy.", new_count, _DAILY_LIMIT)

        return data

    except requests.Timeout:
        log.debug("FMP timeout en %s", path)
        return None
    except Exception as exc:
        log.debug("FMP error inesperado en %s: %s", path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Funciones públicas  (todas cached 6h para respetar 250 llamadas/día)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def get_profile_fmp(ticker: str) -> dict | None:
    """
    Perfil de la empresa: shares outstanding, mktcap, precio, beta, sector…
    Endpoint: /v3/profile/{symbol}

    Campos devueltos (compatibles con yfinance .info):
      sharesOutstanding, marketCap, currentPrice, beta,
      sector, industry, longName, country, website,
      longBusinessSummary, fullTimeEmployees
    """
    ticker = ticker.upper()
    data = _get(f"/v3/profile/{ticker}")
    if not data:
        return None

    p = data[0] if isinstance(data, list) else data
    try:
        return {
            "sharesOutstanding":   _safe_float(p.get("sharesOutstanding")),
            "marketCap":           _safe_float(p.get("mktCap")),
            "currentPrice":        _safe_float(p.get("price")),
            "beta":                _safe_float(p.get("beta")),
            "sector":              p.get("sector"),
            "industry":            p.get("industry"),
            "longName":            p.get("companyName") or ticker,
            "shortName":           p.get("companyName") or ticker,
            "country":             p.get("country"),
            "website":             p.get("website"),
            "longBusinessSummary": p.get("description"),
            "fullTimeEmployees":   _safe_float(p.get("fullTimeEmployees")),
            "dividendYield":       _safe_float(p.get("lastDiv")),  # absoluto, no pct
            "ipoDate":             p.get("ipoDate"),
            "exchange":            p.get("exchangeShortName"),
            "image":               p.get("image"),
            "_fmp_source":         True,
        }
    except Exception as exc:
        log.debug("FMP profile parse error para %s: %s", ticker, exc)
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def get_cash_flow_fmp(ticker: str, limit: int = 4) -> list[dict] | None:
    """
    Estado de flujos de caja (anual, más reciente primero).
    Endpoint: /v3/cash-flow-statement/{symbol}?limit=N&period=annual

    Campos clave por período:
      date, freeCashFlow, operatingCashFlow, capitalExpenditure,
      netIncome, dividendsPaid, stockBasedCompensation
    """
    ticker = ticker.upper()
    data = _get(f"/v3/cash-flow-statement/{ticker}",
                params={"limit": limit, "period": "annual"})
    if not data:
        return None

    result = []
    for item in (data if isinstance(data, list) else [data]):
        result.append({
            "date":                    item.get("date"),
            "freeCashFlow":            _safe_float(item.get("freeCashFlow")),
            "operatingCashFlow":       _safe_float(item.get("operatingCashFlow")),
            "capitalExpenditure":      _safe_float(item.get("capitalExpenditure")),
            "netIncome":               _safe_float(item.get("netIncome")),
            "dividendsPaid":           _safe_float(item.get("dividendsPaid")),
            "stockBasedCompensation":  _safe_float(item.get("stockBasedCompensation")),
            "netChangeInCash":         _safe_float(item.get("netChangeInCash")),
        })
    return result or None


@st.cache_data(ttl=21600, show_spinner=False)
def get_income_statement_fmp(ticker: str, limit: int = 4) -> list[dict] | None:
    """
    Estado de resultados (anual, más reciente primero).
    Endpoint: /v3/income-statement/{symbol}?limit=N&period=annual

    Campos clave por período:
      date, revenue, grossProfit, ebitda, netIncome, eps,
      grossProfitRatio, operatingIncomeRatio, netIncomeRatio
    """
    ticker = ticker.upper()
    data = _get(f"/v3/income-statement/{ticker}",
                params={"limit": limit, "period": "annual"})
    if not data:
        return None

    result = []
    for item in (data if isinstance(data, list) else [data]):
        result.append({
            "date":                  item.get("date"),
            "revenue":               _safe_float(item.get("revenue")),
            "grossProfit":           _safe_float(item.get("grossProfit")),
            "ebitda":                _safe_float(item.get("ebitda")),
            "operatingIncome":       _safe_float(item.get("operatingIncome")),
            "netIncome":             _safe_float(item.get("netIncome")),
            "eps":                   _safe_float(item.get("eps")),
            "epsDiluted":            _safe_float(item.get("epsDiluted")),
            "grossProfitRatio":      _safe_float(item.get("grossProfitRatio")),
            "operatingIncomeRatio":  _safe_float(item.get("operatingIncomeRatio")),
            "netIncomeRatio":        _safe_float(item.get("netIncomeRatio")),
            "weightedAverageShsOut": _safe_float(item.get("weightedAverageShsOut")),
            "weightedAverageShsOutDil": _safe_float(item.get("weightedAverageShsOutDil")),
        })
    return result or None


@st.cache_data(ttl=21600, show_spinner=False)
def get_key_metrics_fmp(ticker: str, limit: int = 1) -> dict | None:
    """
    Key metrics TTM (trailing twelve months).
    Endpoint: /v3/key-metrics-ttm/{symbol}

    Campos clave:
      revenuePerShareTTM, peRatioTTM, pbRatioTTM, priceToFreeCashFlowsRatioTTM,
      enterpriseValueOverEBITDATTM, evToFreeCashFlowTTM, debtToEquityTTM,
      returnOnEquityTTM, returnOnAssetsTTM, currentRatioTTM,
      earningsYieldTTM, freeCashFlowYieldTTM, netDebtToEBITDATTM
    """
    ticker = ticker.upper()
    data = _get(f"/v3/key-metrics-ttm/{ticker}")
    if not data:
        return None

    item = data[0] if isinstance(data, list) else data
    try:
        return {
            "peRatioTTM":                          _safe_float(item.get("peRatioTTM")),
            "pbRatioTTM":                          _safe_float(item.get("pbRatioTTM")),
            "psRatioTTM":                          _safe_float(item.get("priceToSalesRatioTTM")),
            "pfcfRatioTTM":                        _safe_float(item.get("priceToFreeCashFlowsRatioTTM")),
            "evEbitdaTTM":                         _safe_float(item.get("enterpriseValueOverEBITDATTM")),
            "evFcfTTM":                            _safe_float(item.get("evToFreeCashFlowTTM")),
            "debtToEquityTTM":                     _safe_float(item.get("debtToEquityTTM")),
            "returnOnEquityTTM":                   _safe_float(item.get("returnOnEquityTTM")),
            "returnOnAssetsTTM":                   _safe_float(item.get("returnOnAssetsTTM")),
            "currentRatioTTM":                     _safe_float(item.get("currentRatioTTM")),
            "earningsYieldTTM":                    _safe_float(item.get("earningsYieldTTM")),
            "freeCashFlowYieldTTM":                _safe_float(item.get("freeCashFlowYieldTTM")),
            "netDebtToEBITDATTM":                  _safe_float(item.get("netDebtToEBITDATTM")),
            "dividendYieldTTM":                    _safe_float(item.get("dividendYieldTTM")),
            "payoutRatioTTM":                      _safe_float(item.get("payoutRatioTTM")),
            "revenuePerShareTTM":                  _safe_float(item.get("revenuePerShareTTM")),
            "freeCashFlowPerShareTTM":             _safe_float(item.get("freeCashFlowPerShareTTM")),
            "bookValuePerShareTTM":                _safe_float(item.get("bookValuePerShareTTM")),
            "tangibleBookValuePerShareTTM":        _safe_float(item.get("tangibleBookValuePerShareTTM")),
            "marketCapTTM":                        _safe_float(item.get("marketCapTTM")),
            "enterpriseValueTTM":                  _safe_float(item.get("enterpriseValueTTM")),
        }
    except Exception as exc:
        log.debug("FMP key-metrics parse error para %s: %s", ticker, exc)
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def get_ratios_fmp(ticker: str) -> dict | None:
    """
    Ratios TTM: forward PE, márgenes, liquidez, rentabilidad.
    Endpoint: /v3/ratios-ttm/{symbol}
    """
    ticker = ticker.upper()
    data = _get(f"/v3/ratios-ttm/{ticker}")
    if not data:
        return None

    item = data[0] if isinstance(data, list) else data
    try:
        return {
            "grossProfitMarginTTM":      _safe_float(item.get("grossProfitMarginTTM")),
            "operatingProfitMarginTTM":  _safe_float(item.get("operatingProfitMarginTTM")),
            "netProfitMarginTTM":        _safe_float(item.get("netProfitMarginTTM")),
            "returnOnEquityTTM":         _safe_float(item.get("returnOnEquityTTM")),
            "returnOnAssetsTTM":         _safe_float(item.get("returnOnAssetsTTM")),
            "currentRatioTTM":           _safe_float(item.get("currentRatioTTM")),
            "quickRatioTTM":             _safe_float(item.get("quickRatioTTM")),
            "debtRatioTTM":              _safe_float(item.get("debtRatioTTM")),
            "debtEquityRatioTTM":        _safe_float(item.get("debtEquityRatioTTM")),
            "dividendYieldTTM":          _safe_float(item.get("dividendYielTTM")),
            "payoutRatioTTM":            _safe_float(item.get("payoutRatioTTM")),
            "peRatioTTM":                _safe_float(item.get("peRatioTTM")),
            "priceToBookRatioTTM":       _safe_float(item.get("priceToBookRatioTTM")),
            "priceToSalesRatioTTM":      _safe_float(item.get("priceToSalesRatioTTM")),
            "priceToFreeCashFlowsRatioTTM": _safe_float(item.get("priceToFreeCashFlowsRatioTTM")),
        }
    except Exception as exc:
        log.debug("FMP ratios parse error para %s: %s", ticker, exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_quote_fmp(ticker: str) -> dict | None:
    """
    Quote en tiempo real (o ligeramente retrasado).
    Endpoint: /v3/quote/{symbol}

    Campos: price, changesPercentage, pe, eps, earningsAnnouncement,
            sharesOutstanding, marketCap, volume, avgVolume
    """
    ticker = ticker.upper()
    data = _get(f"/v3/quote/{ticker}")
    if not data:
        return None

    item = data[0] if isinstance(data, list) else data
    try:
        return {
            "currentPrice":        _safe_float(item.get("price")),
            "changePercent":       _safe_float(item.get("changesPercentage")),
            "trailingPE":          _safe_float(item.get("pe")),
            "eps":                 _safe_float(item.get("eps")),
            "sharesOutstanding":   _safe_float(item.get("sharesOutstanding")),
            "marketCap":           _safe_float(item.get("marketCap")),
            "volume":              _safe_float(item.get("volume")),
            "avgVolume":           _safe_float(item.get("avgVolume")),
            "open":                _safe_float(item.get("open")),
            "previousClose":       _safe_float(item.get("previousClose")),
            "fiftyTwoWeekHigh":    _safe_float(item.get("yearHigh")),
            "fiftyTwoWeekLow":     _safe_float(item.get("yearLow")),
            "earningsAnnouncement": item.get("earningsAnnouncement"),
            "name":                item.get("name"),
            "exchange":            item.get("exchange"),
        }
    except Exception as exc:
        log.debug("FMP quote parse error para %s: %s", ticker, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Función agregada principal: todo lo que necesita el DCF en una sola llamada
# ─────────────────────────────────────────────────────────────────────────────

def get_dcf_data_fmp(ticker: str) -> dict:
    """
    Agrega todos los datos necesarios para el DCF de WealthView.
    Consume ~3 llamadas API (cash_flow + income + key_metrics) + profile si es necesario.

    Devuelve dict con claves compatibles con yfinance .info donde sea posible:
      freeCashflow, operatingCashflow, capitalExpenditures,
      totalRevenue, ebitda, netIncome, eps,
      sharesOutstanding, marketCap, currentPrice,
      forwardPE (= peRatioTTM de key-metrics),
      enterpriseToEbitda, priceToFreeCashFlow,
      grossMargins, operatingMargins, profitMargins,
      sector, industry, longName, beta,
      _fmp_source, _fmp_fcf_year, _fmp_rev_year
    """
    ticker = ticker.upper()
    result: dict = {"_fmp_source": False}

    # ── Cash flow (fuente más fiable para FCF) ────────────────────────────────
    cf_list = get_cash_flow_fmp(ticker, limit=4)
    if cf_list:
        cf = cf_list[0]  # año más reciente
        result["freeCashflow"]        = cf.get("freeCashFlow")
        result["operatingCashflow"]   = cf.get("operatingCashFlow")
        result["capitalExpenditures"] = cf.get("capitalExpenditure")
        result["netIncomeFromCF"]     = cf.get("netIncome")
        result["_fmp_fcf_year"]       = cf.get("date", "")[:4]
        result["_fmp_source"]         = True

        # Trailing 3y avg FCF (más robusto que 1 año)
        fcf_vals = [x.get("freeCashFlow") for x in cf_list if x.get("freeCashFlow") is not None]
        if len(fcf_vals) >= 2:
            result["freeCashflow_3y_avg"] = sum(fcf_vals[:3]) / len(fcf_vals[:3])

    # ── Income Statement ──────────────────────────────────────────────────────
    inc_list = get_income_statement_fmp(ticker, limit=4)
    if inc_list:
        inc = inc_list[0]
        result["totalRevenue"]  = inc.get("revenue")
        result["ebitda"]        = inc.get("ebitda")
        result["netIncome"]     = inc.get("netIncome")
        result["eps"]           = inc.get("epsDiluted") or inc.get("eps")
        result["grossMargins"]  = inc.get("grossProfitRatio")
        result["operatingMargins"] = inc.get("operatingIncomeRatio")
        result["profitMargins"] = inc.get("netIncomeRatio")
        result["_fmp_rev_year"] = inc.get("date", "")[:4]
        result["_fmp_source"]   = True

        # Calcular revenue growth YoY si hay 2 años
        if len(inc_list) >= 2:
            rev_now  = inc_list[0].get("revenue")
            rev_prev = inc_list[1].get("revenue")
            if rev_now and rev_prev and rev_prev != 0:
                result["revenueGrowth"] = (rev_now - rev_prev) / abs(rev_prev)

    # ── Key Metrics TTM ───────────────────────────────────────────────────────
    km = get_key_metrics_fmp(ticker)
    if km:
        result["forwardPE"]             = km.get("peRatioTTM")   # mejor estimador disponible
        result["trailingPE"]            = km.get("peRatioTTM")
        result["enterpriseToEbitda"]    = km.get("evEbitdaTTM")
        result["priceToFreeCashFlow"]   = km.get("pfcfRatioTTM")
        result["priceToBook"]           = km.get("pbRatioTTM")
        result["priceToSalesTrailing12Months"] = km.get("psRatioTTM")
        result["returnOnEquity"]        = km.get("returnOnEquityTTM")
        result["returnOnAssets"]        = km.get("returnOnAssetsTTM")
        result["currentRatio"]          = km.get("currentRatioTTM")
        result["freeCashFlowYield"]     = km.get("freeCashFlowYieldTTM")
        result["enterpriseValue"]       = km.get("enterpriseValueTTM")
        result["marketCapTTM"]          = km.get("marketCapTTM")
        result["_fmp_source"]           = True

    # ── Profile (shares, precio, beta, sector) ────────────────────────────────
    prof = get_profile_fmp(ticker)
    if prof:
        result["sharesOutstanding"]     = prof.get("sharesOutstanding")
        result["currentPrice"]          = prof.get("currentPrice")
        result["marketCap"]             = prof.get("marketCap")
        result["beta"]                  = prof.get("beta")
        result["sector"]                = prof.get("sector")
        result["industry"]              = prof.get("industry")
        result["longName"]              = prof.get("longName")
        result["shortName"]             = prof.get("shortName")
        result["country"]               = prof.get("country")
        result["longBusinessSummary"]   = prof.get("longBusinessSummary")
        result["fullTimeEmployees"]     = prof.get("fullTimeEmployees")
        result["_fmp_source"]           = True

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Función para enriquecer el dict de info existente (yfinance/Finviz → + FMP)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_info_with_fmp(info: dict, ticker: str) -> dict:
    """
    Toma un dict de info (yfinance/Finviz) y lo enriquece con datos FMP
    en los campos donde yfinance/Finviz suelen fallar:
      - freeCashflow  (lo más importante)
      - sharesOutstanding
      - operatingCashflow / capitalExpenditures
      - forwardPE
      - ebitda, totalRevenue, grossMargins, profitMargins

    No sobreescribe campos que ya tienen valor.
    Devuelve el dict enriquecido.
    """
    fmp_data = get_dcf_data_fmp(ticker)
    if not fmp_data.get("_fmp_source"):
        return info  # FMP no respondió, no hay nada que enriquecer

    enriched = dict(info)  # copia

    _FMP_FIELD_MAP = {
        # yfinance key          → FMP key en get_dcf_data_fmp
        "freeCashflow":                      "freeCashflow",
        "operatingCashflow":                 "operatingCashflow",
        "capitalExpenditures":               "capitalExpenditures",
        "totalRevenue":                      "totalRevenue",
        "ebitda":                            "ebitda",
        "netIncome":                         "netIncome",
        "eps":                               "eps",
        "sharesOutstanding":                 "sharesOutstanding",
        "marketCap":                         "marketCap",
        "currentPrice":                      "currentPrice",
        "forwardPE":                         "forwardPE",
        "trailingPE":                        "trailingPE",
        "enterpriseToEbitda":                "enterpriseToEbitda",
        "priceToBook":                       "priceToBook",
        "priceToSalesTrailing12Months":      "priceToSalesTrailing12Months",
        "returnOnEquity":                    "returnOnEquity",
        "returnOnAssets":                    "returnOnAssets",
        "currentRatio":                      "currentRatio",
        "grossMargins":                      "grossMargins",
        "operatingMargins":                  "operatingMargins",
        "profitMargins":                     "profitMargins",
        "beta":                              "beta",
        "sector":                            "sector",
        "industry":                          "industry",
        "longName":                          "longName",
        "shortName":                         "shortName",
        "revenueGrowth":                     "revenueGrowth",
    }

    for yf_key, fmp_key in _FMP_FIELD_MAP.items():
        # Sólo rellenar si está vacío en yfinance
        existing = enriched.get(yf_key)
        if existing in (None, "N/A", "", "None") and fmp_data.get(fmp_key) is not None:
            enriched[yf_key] = fmp_data[fmp_key]

    # Campos extra de FMP que no existen en yfinance
    enriched["_fmp_source"]       = True
    enriched["_fmp_fcf_year"]     = fmp_data.get("_fmp_fcf_year")
    enriched["_fmp_rev_year"]     = fmp_data.get("_fmp_rev_year")
    enriched["freeCashflow_3y_avg"] = fmp_data.get("freeCashflow_3y_avg")
    enriched["freeCashFlowYield"] = fmp_data.get("freeCashFlowYield")

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Utils internos
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    """Convierte a float, devuelve None si falla o es 0 en algunos contextos."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if not (f != f) else None  # NaN check
    except (TypeError, ValueError):
        return None


def api_status() -> dict:
    """
    Verifica el estado de la API key FMP.
    Devuelve {
      'ok': bool, 'key_set': bool, 'error': str | None,
      'calls_today': int, 'calls_limit': int, 'calls_pct': float
    }
    """
    _ensure_fmp_counter_table()
    calls_today = get_daily_call_count()
    base = {
        "calls_today": calls_today,
        "calls_limit": _DAILY_LIMIT,
        "calls_pct":   calls_today / _DAILY_LIMIT * 100,
    }

    key = _get_api_key()
    if not key:
        return {**base, "ok": False, "key_set": False, "error": "FMP_API_KEY no configurada"}

    # Test call (will be counted in _get)
    data = _get("/v3/profile/AAPL")
    if data:
        return {**base, "ok": True, "key_set": True, "error": None}
    return {**base, "ok": False, "key_set": True, "error": "API key inválida o sin conexión"}

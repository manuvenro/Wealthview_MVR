"""
data_provider.py — Unified data layer for WealthView
Price source:     Polygon.io (real-time/15min delay) → yfinance fallback
Fundamentals:     Finviz (via finvizfinance) → yfinance fallback

All public functions return the same key schema as yfinance .info so
consuming modules need only swap the call — no logic changes needed elsewhere.

New functions:
  get_live_price(ticker)          → float | None   (Polygon → yfinance)
  get_live_prices_batch(tickers)  → {ticker: float} (1 API call via Polygon)
  get_price_source()              → 'polygon' | 'yfinance'
"""

import logging
from datetime import datetime
from typing import Optional
import streamlit as st

log = logging.getLogger(__name__)

# ── Check if finvizfinance is available ───────────────────────────────────────
try:
    from finvizfinance.quote import finvizfinance as _FVQ
    _FINVIZ_AVAILABLE = True
except ImportError:
    _FINVIZ_AVAILABLE = False
    log.info("finvizfinance not installed — using yfinance only. "
             "Run: pip install finvizfinance")

# ── Polygon.io availability (lazy import to avoid circular deps) ──────────────
def _polygon():
    """Lazy-import polygon_client to avoid circular import at module load."""
    try:
        import modules.polygon_client as pc
        return pc if pc.api_key_set() else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# String parsers  (Finviz returns everything as formatted strings)
# ─────────────────────────────────────────────────────────────────────────────

def _pf(s) -> Optional[float]:
    """Parse plain float: '28.45', '-', '' → float or None."""
    if s is None:
        return None
    s = str(s).strip()
    if s in ("-", "", "N/A", "None", "nan"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _pp(s) -> Optional[float]:
    """Parse percentage string: '65.00%' → 0.65,  '-5.23%' → -0.0523."""
    if s is None:
        return None
    s = str(s).strip().rstrip("%")
    if s in ("-", "", "N/A", "None", "nan"):
        return None
    try:
        return float(s.replace(",", "")) / 100.0
    except (ValueError, TypeError):
        return None


def _pn(s) -> Optional[float]:
    """Parse large number: '182.45B' → 1.82e11,  '1.23T' → 1.23e12, etc."""
    if s is None:
        return None
    s = str(s).strip()
    if s in ("-", "", "N/A", "None", "nan"):
        return None
    mult = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    if s and s[-1] in mult:
        try:
            return float(s[:-1].replace(",", "")) * mult[s[-1]]
        except (ValueError, TypeError):
            return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _p52w(s) -> tuple:
    """Parse '52W Range': '124.17 - 199.62' → (low, high) or (None, None)."""
    if not s or str(s).strip() in ("-", "", "N/A"):
        return None, None
    parts = str(s).split(" - ")
    if len(parts) != 2:
        return None, None
    try:
        lo = float(parts[0].replace(",", ""))
        hi = float(parts[1].replace(",", ""))
        return lo, hi
    except (ValueError, TypeError):
        return None, None


def _prec(s) -> Optional[str]:
    """
    Convert Finviz numeric recommendation to yfinance-style string.
    Finviz:   1.0 = Strong Buy … 5.0 = Sell
    Returns:  'strong_buy' | 'buy' | 'hold' | 'underperform' | 'sell'
    """
    v = _pf(s)
    if v is None:
        return None
    if v <= 1.5:
        return "strong_buy"
    if v <= 2.5:
        return "buy"
    if v <= 3.5:
        return "hold"
    if v <= 4.5:
        return "underperform"
    return "sell"


# ─────────────────────────────────────────────────────────────────────────────
# Finviz → yfinance field mapping
# ─────────────────────────────────────────────────────────────────────────────

def _finviz_raw(ticker: str) -> Optional[dict]:
    """Fetch raw fundament dict from Finviz. Returns None on any failure."""
    if not _FINVIZ_AVAILABLE:
        return None
    try:
        return _FVQ(ticker).ticker_fundament()
    except Exception as exc:
        log.debug("Finviz fundament error for %s: %s", ticker, exc)
        return None


def _finviz_to_info(f: dict, ticker: str) -> dict:
    """
    Map a Finviz fundament dict → yfinance .info-compatible dict.
    Every yfinance-standard key is present; fields Finviz can't provide → None.
    """
    price   = _pf(f.get("Price"))
    mktcap  = _pn(f.get("Market Cap"))
    shs_out = _pn(f.get("Shs Outstand"))
    pfcf    = _pf(f.get("P/FCF"))
    cash_sh = _pf(f.get("Cash/sh"))

    low52, high52 = _p52w(f.get("52W Range"))

    # Free cash flow  =  MarketCap / P/FCF
    fcf = (mktcap / pfcf) if (mktcap and pfcf and pfcf != 0) else None

    # Total cash  =  Cash/sh × Shares outstanding
    total_cash = (cash_sh * shs_out) if (cash_sh is not None and shs_out) else None

    # Debt/Equity: Finviz gives ratio (1.45); yfinance gives ×100 (145.0).
    # All downstream code expects the yfinance format and does its own /100.
    de_raw = _pf(f.get("Debt/Eq"))
    de_yf  = de_raw * 100 if de_raw is not None else None

    return {
        # ── Price ─────────────────────────────────────────────────────────
        "currentPrice":                  price,
        "regularMarketPrice":            price,
        "previousClose":                 _pf(f.get("Prev Close")),

        # ── Market data ───────────────────────────────────────────────────
        "marketCap":                     mktcap,
        "enterpriseValue":               None,
        "sharesOutstanding":             shs_out,
        "beta":                          _pf(f.get("Beta")),
        "fiftyTwoWeekHigh":              high52,
        "fiftyTwoWeekLow":               low52,
        "averageVolume":                 _pn(f.get("Avg Volume")),
        "volume":                        _pn(f.get("Volume")),

        # ── Valuation multiples ───────────────────────────────────────────
        "trailingPE":                    _pf(f.get("P/E")),
        "forwardPE":                     _pf(f.get("Forward P/E")),
        "priceToSalesTrailing12Months":  _pf(f.get("P/S")),
        "priceToBook":                   _pf(f.get("P/B")),
        "enterpriseToEbitda":            _pf(f.get("EV/EBITDA")),
        "pegRatio":                      _pf(f.get("PEG")),

        # ── Income / margins ──────────────────────────────────────────────
        "totalRevenue":                  _pn(f.get("Sales")),
        "grossMargins":                  _pp(f.get("Gross Margin")),
        "operatingMargins":              _pp(f.get("Oper. Margin")),
        "profitMargins":                 _pp(f.get("Profit Margin")),

        # ── Balance sheet ─────────────────────────────────────────────────
        "debtToEquity":                  de_yf,
        "currentRatio":                  _pf(f.get("Current Ratio")),
        "quickRatio":                    _pf(f.get("Quick Ratio")),
        "freeCashflow":                  fcf,
        "totalCash":                     total_cash,
        "totalDebt":                     None,

        # ── Returns ───────────────────────────────────────────────────────
        "returnOnEquity":                _pp(f.get("ROE")),
        "returnOnAssets":                _pp(f.get("ROA")),

        # ── Growth ────────────────────────────────────────────────────────
        "revenueGrowth":                 _pp(f.get("Sales Q/Q")),
        "earningsGrowth":                _pp(f.get("EPS Q/Q")),

        # ── Dividends ─────────────────────────────────────────────────────
        "dividendYield":                 _pp(f.get("Dividend %")),

        # ── Short interest ────────────────────────────────────────────────
        "shortPercentOfFloat":           _pp(f.get("Short Float")),
        "shortRatio":                    _pf(f.get("Short Ratio")),
        "sharesShort":                   None,

        # ── Analyst consensus ─────────────────────────────────────────────
        "targetMeanPrice":               _pf(f.get("Target Price")),
        "targetHighPrice":               None,
        "targetLowPrice":                None,
        "recommendationKey":             _prec(f.get("Recom.")),
        "numberOfAnalystOpinions":       None,

        # ── Company metadata ──────────────────────────────────────────────
        "shortName":                     f.get("Company") or ticker,
        "longName":                      f.get("Company") or ticker,
        "sector":                        f.get("Sector"),
        "industry":                      f.get("Industry"),
        "country":                       f.get("Country"),
        "website":                       None,
        "longBusinessSummary":           f.get("Description"),
        "fullTimeEmployees":             _pn(f.get("Employees")),

        # ── Finviz-only bonus fields (prefixed _ to avoid collisions) ─────
        "_fv_rsi14":                     _pf(f.get("RSI (14)")),
        "_fv_sma200_pct":                _pp(f.get("SMA200")),
        "_fv_sma50_pct":                 _pp(f.get("SMA50")),
        "_fv_eps_next_y":                _pp(f.get("EPS next Y")),
        "_fv_source":                    True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public data functions
# ─────────────────────────────────────────────────────────────────────────────

def _is_finviz_data_reliable(info: dict) -> bool:
    """
    Validate that Finviz data is reliable enough to use.
    Returns False if critical fields are missing — triggers yfinance fallback.
    Common failure cases: pre-revenue biotechs, penny stocks, recent IPOs,
    tickers with recent reverse splits, or companies with no earnings.
    """
    price = info.get("currentPrice")
    mktcap = info.get("marketCap")
    name = info.get("shortName") or info.get("longName")
    # Price must be a positive number
    if not price or not isinstance(price, (int, float)) or price <= 0:
        return False
    # Market cap must be present and positive
    if not mktcap or not isinstance(mktcap, (int, float)) or mktcap <= 0:
        return False
    # Company name must exist
    if not name or name == ticker:
        return False
    return True


def fetch_fundamentals_raw(ticker: str) -> dict:
    """
    Fundamentals — Finviz primary, yfinance fallback.
    Price is overlaid with Polygon live price if available.
    NOT cached: safe to call from threads (peers.py, screener.py).
    Returns a yfinance .info-compatible dict.
    """
    import yfinance as yf

    info = None
    f = _finviz_raw(ticker)
    if f:
        candidate = _finviz_to_info(f, ticker)
        if _is_finviz_data_reliable(candidate):
            info = candidate
            log.debug("Finviz data OK for %s (price=%.2f)", ticker,
                      candidate.get("currentPrice", 0))
        else:
            log.info("Finviz data unreliable for %s — falling back to yfinance", ticker)

    if info is None:
        # Fallback: yfinance
        try:
            yf_info = yf.Ticker(ticker).info or {}
            yf_info.setdefault("_fv_source", False)
            yf_info["_yf_fallback"] = True
            info = yf_info
        except Exception as exc:
            log.warning("yfinance fallback failed for %s: %s", ticker, exc)
            info = {"_fv_source": False, "currentPrice": None,
                    "shortName": ticker, "longName": ticker}

    # ── Overlay yfinance price as secondary check when price seems stale ──────
    # For volatile tickers (biotech, penny stocks), cross-check with yfinance
    try:
        yf_price = getattr(yf.Ticker(ticker).fast_info, "last_price", None)
        current  = info.get("currentPrice")
        if yf_price and current and abs(yf_price - current) / current > 0.10:
            # Prices differ by more than 10% — prefer yfinance price as sanity check
            log.info("Price mismatch for %s: Finviz=%.2f yf=%.2f — using yfinance",
                     ticker, current, yf_price)
            info["currentPrice"]       = float(yf_price)
            info["regularMarketPrice"] = float(yf_price)
            info["_price_corrected"]   = True
        elif yf_price and not current:
            info["currentPrice"]       = float(yf_price)
            info["regularMarketPrice"] = float(yf_price)
    except Exception as exc:
        log.debug("yfinance price cross-check failed for %s: %s", ticker, exc)

    # ── Overlay Polygon live price (more accurate / fresher) ─────────────────
    pc = _polygon()
    if pc:
        try:
            snap = pc.get_snapshot(ticker)
            if snap and snap.get("currentPrice"):
                info["currentPrice"]     = snap["currentPrice"]
                info["regularMarketPrice"] = snap["currentPrice"]
                info["_polygon_price"]   = True
                info["_polygon_realtime"] = pc.detect_plan() != "free"
                if snap.get("previousClose"):
                    info["previousClose"] = snap["previousClose"]
                if snap.get("dayHigh"):
                    info["dayHigh"]      = snap["dayHigh"]
                if snap.get("dayLow"):
                    info["dayLow"]       = snap["dayLow"]
                if snap.get("volume"):
                    info["volume"]       = snap["volume"]
        except Exception as exc:
            log.debug("Polygon price overlay failed for %s: %s", ticker, exc)

    return info


@st.cache_data(ttl=900, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """
    Cached (15 min) fundamentals — Finviz primary, yfinance fallback.
    Use from the Streamlit main thread (deep_dive.py, portfolio.py, etc.).
    """
    return fetch_fundamentals_raw(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker: str) -> list:
    """
    Cached (30 min) news — Finviz primary, yfinance fallback.
    Returns list of dicts compatible with yfinance .news format:
      {title, publisher, link, providerPublishTime}
    """
    if _FINVIZ_AVAILABLE:
        try:
            df = _FVQ(ticker).ticker_news()
            if df is not None and not df.empty:
                result = []
                for _, row in df.iterrows():
                    ts = None
                    date_val = row.get("Date") or row.get("date")
                    if date_val is not None:
                        try:
                            if isinstance(date_val, str):
                                dt = datetime.strptime(date_val, "%b-%d-%Y %I:%M%p")
                            elif hasattr(date_val, "timestamp"):
                                dt = date_val
                            else:
                                dt = None
                            ts = int(dt.timestamp()) if dt else None
                        except Exception:
                            ts = None
                    result.append({
                        "title":               str(row.get("Title") or row.get("title") or ""),
                        "publisher":           str(row.get("Source") or row.get("source") or "Finviz"),
                        "link":                str(row.get("Link") or row.get("Url") or row.get("url") or ""),
                        "providerPublishTime": ts or 0,
                    })
                if result:
                    return result
        except Exception as exc:
            log.debug("Finviz news error for %s: %s", ticker, exc)

    # Fallback: yfinance
    try:
        import yfinance as yf
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


def get_insider_trades_df(ticker: str):
    """
    Insider trades DataFrame — Finviz primary, yfinance fallback.
    Returns a DataFrame compatible with _render_yf_insiders() in edgar.py, or None.
    """
    if _FINVIZ_AVAILABLE:
        try:
            df = _FVQ(ticker).ticker_inside_trader()
            if df is not None and not df.empty:
                # Rename to match what _render_yf_insiders() looks for via substring matching:
                # it looks for col.lower() containing 'date', 'insider', 'title',
                # 'transaction', 'shares', 'value', 'ownership'
                rename = {
                    "Date":            "Date",
                    "Owner":           "Insider",
                    "Relationship":    "Title",
                    "Transaction":     "Transaction",
                    "#Shares":         "Shares",
                    "Value ($)":       "Value",
                    "Insider Trading": "Ownership",
                }
                existing = {k: v for k, v in rename.items() if k in df.columns}
                return df.rename(columns=existing)
        except Exception as exc:
            log.debug("Finviz insider trades error for %s: %s", ticker, exc)

    # Fallback: yfinance
    try:
        import yfinance as yf
        return yf.Ticker(ticker).insider_transactions
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Live price functions (Polygon → yfinance fallback)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> Optional[float]:
    """
    Precio en tiempo real (Polygon) o diferido 15min (yfinance).
    Polygon es la fuente primaria si hay API key configurada.
    """
    pc = _polygon()
    if pc:
        snap = pc.get_snapshot(ticker)
        if snap and snap.get("currentPrice"):
            return float(snap["currentPrice"])

    # Fallback: yfinance fast_info
    try:
        import yfinance as yf
        p = getattr(yf.Ticker(ticker).fast_info, "last_price", None)
        return float(p) if p else None
    except Exception:
        return None


def get_live_prices_batch(tickers: list | tuple) -> dict:
    """
    Precios de múltiples tickers.
    Con Polygon: 1 sola llamada API para todos (hasta 250 tickers).
    Sin Polygon: yfinance llamada por llamada.

    Devuelve {ticker: float}.
    """
    if not tickers:
        return {}

    pc = _polygon()
    if pc:
        snaps = pc.get_snapshots(tuple(tickers))
        prices = {t: float(s["currentPrice"])
                  for t, s in snaps.items()
                  if s.get("currentPrice")}
        # Fill missing with yfinance
        missing = [t for t in tickers if t not in prices]
        if missing:
            prices.update(_yf_prices_fallback(missing))
        return prices

    return _yf_prices_fallback(list(tickers))


def _yf_prices_fallback(tickers: list) -> dict:
    """yfinance precio por precio — fallback cuando no hay Polygon key."""
    result = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                p = getattr(yf.Ticker(t).fast_info, "last_price", None)
                if p:
                    result[t] = float(p)
            except Exception:
                pass
    except Exception:
        pass
    return result


def get_price_source() -> str:
    """Devuelve 'polygon' si hay key configurada, 'yfinance' si no."""
    return "polygon" if _polygon() else "yfinance"


def get_market_status() -> dict:
    """Estado del mercado USA (abierto/cerrado/extended hours)."""
    pc = _polygon()
    if pc:
        try:
            return pc.get_market_status()
        except Exception:
            pass
    return {"market": "unknown", "serverTime": None}

"""
peers.py — Dynamic peer median computation for WealthView
Fetches live fundamentals for sector representative companies and computes
real-time medians to use as benchmarks in scoring. Cached 24h.
"""

import streamlit as st
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.data_provider import fetch_fundamentals_raw

# ── Representative tickers per GICS sector (25-30 per sector) ────────────────
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "QCOM",
        "TXN", "INTC", "NOW", "AMAT", "ADBE", "MU", "LRCX", "KLAC",
        "SNPS", "CDNS", "ADI", "MCHP", "FTNT", "PANW", "ANSS", "TEL", "APH"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "PFE", "AMGN", "SYK", "GILD", "ISRG", "VRTX", "REGN", "ZTS", "BSX",
        "MDT", "ELV", "CI", "HUM", "IQV", "BDX", "IDXX"
    ],
    "Financial Services": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
        "AXP", "SPGI", "MCO", "CB", "PGR", "AON", "MMC", "ICE", "CME",
        "TFC", "USB", "PNC", "FIS", "PYPL", "COF"
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
        "CMG", "MAR", "GM", "F", "ORLY", "AZO", "YUM", "DRI", "HLT",
        "APTV", "BBY", "ROST", "DHI", "LEN", "PHM", "NVR"
    ],
    "Consumer Defensive": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL",
        "KMB", "GIS", "SYY", "KHC", "HSY", "MKC", "CHD", "CLX", "K",
        "CAG", "HRL", "SJM", "CPB", "TAP", "BF-B", "TSN"
    ],
    "Industrials": [
        "GE", "HON", "UPS", "CAT", "DE", "RTX", "BA", "LMT", "NOC", "GD",
        "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "XYL", "IR", "FDX",
        "PCAR", "CTAS", "NSC", "UNP", "CSX", "WAB"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD",
        "OXY", "DVN", "HES", "HAL", "BKR", "FANG", "MRO", "APA", "CTRA",
        "PR", "OVV", "SM", "CNX", "RRC", "AR", "EQT"
    ],
    "Basic Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "STLD",
        "ALB", "PPG", "EMN", "CE", "CF", "MOS", "FMC", "IFF", "RPM",
        "WRK", "IP", "PKG", "SEE", "SON", "BALL", "ATI"
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "DLR", "SPG",
        "EXR", "AVB", "EQR", "MAA", "UDR", "NNN", "VICI", "MPW", "SBA",
        "ARE", "BXP", "VTR", "PEAK", "HST", "REG", "KIM"
    ],
    "Utilities": [
        "NEE", "SO", "DUK", "AEP", "XEL", "SRE", "PCG", "ED", "EXC",
        "WEC", "ES", "ETR", "AWK", "FE", "CMS", "NI", "AES", "CNP",
        "LNT", "EVRG", "PPL", "OGE", "PNW", "NWE", "AVA"
    ],
    "Communication Services": [
        "META", "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "ATVI", "EA", "TTWO", "LYV", "WBD", "FOXA", "PARA",
        "OMC", "IPG", "ZETA", "SNAP", "PINS", "MTCH", "IAC", "ZG", "ANGI"
    ],
}

# Alias mapping from yfinance sector names to our keys
SECTOR_ALIAS: dict[str, str] = {
    "technology":              "Technology",
    "information technology":  "Technology",
    "healthcare":              "Healthcare",
    "health care":             "Healthcare",
    "financial services":      "Financial Services",
    "financials":              "Financial Services",
    "finance":                 "Financial Services",
    "consumer cyclical":       "Consumer Cyclical",
    "consumer discretionary":  "Consumer Cyclical",
    "consumer defensive":      "Consumer Defensive",
    "consumer staples":        "Consumer Defensive",
    "industrials":             "Industrials",
    "industrial conglomerates":"Industrials",
    "energy":                  "Energy",
    "basic materials":         "Basic Materials",
    "materials":               "Basic Materials",
    "real estate":             "Real Estate",
    "utilities":               "Utilities",
    "communication services":  "Communication Services",
    "communication":           "Communication Services",
    "telecommunications":      "Communication Services",
}

# Safe median ignoring None / NaN
def _safe_median(values: list) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0]
    return float(np.median(clean)) if clean else None

def _fetch_peer_metrics(ticker: str) -> dict | None:
    """Fetch fundamental metrics for a single peer ticker."""
    try:
        info = fetch_fundamentals_raw(ticker)
        if not info:
            return None
        pe = info.get("trailingPE") or info.get("forwardPE")
        ps = info.get("priceToSalesTrailing12Months")
        ev_ebitda = info.get("enterpriseToEbitda")
        gross_m = info.get("grossMargins")
        net_m = info.get("profitMargins")
        op_m = info.get("operatingMargins")
        roe = info.get("returnOnEquity")
        de_raw = info.get("debtToEquity")
        de = de_raw / 100 if de_raw is not None else None
        fwd_pe = info.get("forwardPE")
        fwd_eps_growth = info.get("earningsGrowth")
        mktcap    = info.get("marketCap")
        fcf       = info.get("freeCashflow")
        pfcf      = (mktcap / fcf) if (mktcap and fcf and fcf > 0) else None

        return {
            "ticker": ticker,
            "pe": pe if pe and 0 < pe < 500 else None,
            "ps": ps if ps and 0 < ps < 200 else None,
            "ev_ebitda": ev_ebitda if ev_ebitda and 0 < ev_ebitda < 200 else None,
            "gross_m": gross_m if gross_m is not None else None,
            "net_m": net_m if net_m is not None else None,
            "op_m": op_m if op_m is not None else None,
            "roe": roe if roe is not None else None,
            "de": de if de is not None else None,
            "fwd_pe": fwd_pe if fwd_pe and 0 < fwd_pe < 500 else None,
            "eps_growth": fwd_eps_growth if fwd_eps_growth is not None else None,
            "pfcf":      pfcf if pfcf and 0 < pfcf < 200 else None,
        }
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)   # 24-hour cache
def get_sector_medians(sector_key: str) -> dict:
    """
    Compute live median fundamentals for the given sector key.
    Returns a dict with keys: pe, ps, ev_ebitda, gross_m, net_m, op_m, roe, de, fwd_pe, eps_growth.
    Falls back to hardcoded defaults if fetching fails.
    """
    peers = SECTOR_PEERS.get(sector_key, [])
    if not peers:
        return _SECTOR_DEFAULT

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_peer_metrics, t): t for t in peers}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)

    if len(results) < 5:
        # Not enough data — return defaults
        return _SECTOR_DEFAULT

    medians = {
        "pe":        _safe_median([r["pe"] for r in results]),
        "ps":        _safe_median([r["ps"] for r in results]),
        "ev_ebitda": _safe_median([r["ev_ebitda"] for r in results]),
        "gross_m":   _safe_median([r["gross_m"] for r in results]),
        "net_m":     _safe_median([r["net_m"] for r in results]),
        "op_m":      _safe_median([r["op_m"] for r in results]),
        "roe":       _safe_median([r["roe"] for r in results]),
        "de":        _safe_median([r["de"] for r in results]),
        "fwd_pe":    _safe_median([r["fwd_pe"] for r in results]),
        "eps_growth":_safe_median([r["eps_growth"] for r in results]),
        "pfcf":      _safe_median([r["pfcf"]       for r in results]),
        "_n_peers":  len(results),
        "_live":     True,
    }
    # Fill any None values from defaults
    defaults = _SECTOR_DEFAULT.copy()
    for k, v in medians.items():
        if v is None and k in defaults:
            medians[k] = defaults[k]

    return medians


def normalize_sector(raw_sector: str) -> str:
    """Map yfinance sector string to our canonical sector key."""
    if not raw_sector:
        return "Technology"
    return SECTOR_ALIAS.get(raw_sector.lower().strip(), raw_sector)


# ── Hardcoded fallback defaults ───────────────────────────────────────────────
_SECTOR_DEFAULT: dict = {
    "pe": 20.0, "ps": 2.5, "ev_ebitda": 14.0,
    "gross_m": 0.40, "net_m": 0.10, "op_m": 0.13,
    "roe": 0.14, "de": 0.80,
    "fwd_pe": 18.0, "eps_growth": 0.10, "pfcf": 20.0,
    "_live": False, "_n_peers": 0,
}

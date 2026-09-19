"""
screener.py — Sector screener for WealthView
Scans all peer tickers in a sector, computes a lightweight WealthView score
for each, and ranks them in a sortable table with key metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from modules.data_provider import fetch_fundamentals_raw

from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK
)
from modules.peers import SECTOR_PEERS, normalize_sector, get_sector_medians, _SECTOR_DEFAULT


# ── Lightweight per-ticker scoring ────────────────────────────────────────────

def _score_ticker(ticker: str, sec_ref: dict) -> dict | None:
    """
    Fetch key metrics for one ticker and compute a fast WealthView score.
    Returns None on failure.
    """
    try:
        info = fetch_fundamentals_raw(ticker)
        if not info or not (info.get("regularMarketPrice") or info.get("currentPrice")):
            return None

        def _g(key, fallback=None):
            v = info.get(key)
            return v if v is not None else fallback

        price     = _g("currentPrice") or _g("regularMarketPrice")
        mktcap    = _g("marketCap")
        pe        = _g("trailingPE")
        fwd_pe    = _g("forwardPE")
        ps        = _g("priceToSalesTrailing12Months")
        ev_ebitda = _g("enterpriseToEbitda")
        gross_m   = _g("grossMargins")
        net_m     = _g("profitMargins")
        roe       = _g("returnOnEquity")
        de_raw    = _g("debtToEquity")
        de        = de_raw / 100 if de_raw is not None else None
        rev_growth= _g("revenueGrowth")
        eps_growth= _g("earningsGrowth")
        target    = _g("targetMeanPrice")
        rec       = _g("recommendationKey", "")
        name      = _g("shortName") or _g("longName") or ticker

        up_pct = ((target - price) / price * 100) if (target and price and price > 0) else None

        # ── Quick score (0-10) ────────────────────────────────────────────────
        scores = []

        # Valuation vs sector
        pe_ref = sec_ref.get("pe") or _SECTOR_DEFAULT["pe"]
        nm_ref = sec_ref.get("net_m") or _SECTOR_DEFAULT["net_m"]
        roe_ref = sec_ref.get("roe") or _SECTOR_DEFAULT["roe"]

        if pe and pe > 0 and pe_ref:
            r = pe / pe_ref
            scores.append(max(1, min(10, 10 - (r - 1) * 4)))
        if fwd_pe and fwd_pe > 0:
            fpe_ref = sec_ref.get("fwd_pe") or 18.0
            r = fwd_pe / fpe_ref
            scores.append(max(1, min(10, 10 - (r - 1) * 4)))
        if up_pct is not None:
            scores.append(9 if up_pct > 30 else 7 if up_pct > 15 else 5 if up_pct > 5 else 3)

        # Profitability
        if net_m is not None and nm_ref:
            if net_m < 0:
                scores.append(max(1, int(2 + net_m * 10)))
            else:
                r = net_m / nm_ref
                scores.append(max(1, min(10, 5 + (r - 1) * 3)))
        if roe is not None and roe_ref:
            if roe > 0:
                r = roe / roe_ref
                scores.append(max(1, min(10, 5 + (r - 1) * 3)))
            else:
                scores.append(2)

        # Growth
        if rev_growth is not None:
            scores.append(9 if rev_growth > 0.20 else 7 if rev_growth > 0.10 else 5 if rev_growth > 0 else 2)
        if eps_growth is not None:
            scores.append(9 if eps_growth > 0.20 else 7 if eps_growth > 0.10 else 5 if eps_growth > 0 else 2)

        # Analyst consensus
        rec_up = rec.upper()
        if "STRONG_BUY" in rec_up or "STRONG BUY" in rec_up:
            scores.append(9)
        elif "BUY" in rec_up:
            scores.append(7)
        elif "HOLD" in rec_up:
            scores.append(5)
        elif "SELL" in rec_up:
            scores.append(2)

        score = round(float(np.mean(scores)), 1) if scores else 5.0

        # Recommendation label
        if score >= 7.5:   wv_rec = "COMPRA FUERTE"
        elif score >= 6.2: wv_rec = "COMPRA"
        elif score >= 4.5: wv_rec = "MANTENER"
        elif score >= 3.0: wv_rec = "VENDER"
        else:              wv_rec = "VENTA FUERTE"

        def _fmt_mktcap(v):
            if not v: return "—"
            if v >= 1e12: return f"${v/1e12:.1f}T"
            if v >= 1e9:  return f"${v/1e9:.1f}B"
            return f"${v/1e6:.0f}M"

        return {
            "Ticker":       ticker,
            "Empresa":      name[:28],
            "Precio":       f"${price:,.2f}" if price else "—",
            "Cap.":         _fmt_mktcap(mktcap),
            "P/E":          f"{pe:.1f}x" if pe else "—",
            "Fwd P/E":      f"{fwd_pe:.1f}x" if fwd_pe else "—",
            "P/S":          f"{ps:.1f}x" if ps else "—",
            "Mg. Neto":     f"{net_m*100:.1f}%" if net_m is not None else "—",
            "ROE":          f"{roe*100:.1f}%" if roe is not None else "—",
            "Crec. Rev.":   f"{rev_growth*100:+.1f}%" if rev_growth is not None else "—",
            "Upside":       f"{up_pct:+.1f}%" if up_pct is not None else "—",
            "Consenso":     rec.replace("_", " ").title() if rec else "—",
            "WV Score":     score,
            "Recomendación":wv_rec,
            "_score_raw":   score,
            "_rec":         wv_rec,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def scan_sector(sector_key: str) -> list[dict]:
    """Scan all tickers in a sector and return scored list."""
    tickers = SECTOR_PEERS.get(sector_key, [])
    if not tickers:
        return []

    sec_ref = get_sector_medians(sector_key)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_score_ticker, t, sec_ref): t for t in tickers}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["_score_raw"], reverse=True)
    return results


# ── Render ────────────────────────────────────────────────────────────────────

def render_screener():
    """Main screener page render."""
    from modules.styles import page_header
    page_header("Sector Screener", "Rankea todas las empresas de un sector por score WealthView")

    col_sec, col_run = st.columns([3, 1])
    with col_sec:
        sector = st.selectbox(
            "Sector",
            options=list(SECTOR_PEERS.keys()),
            key="screener_sector",
        )
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔍 Escanear sector", type="primary",
                         use_container_width=True, key="screener_run")

    if not run and "screener_results" not in st.session_state:
        st.markdown(
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:32px; text-align:center; color:{TEXT_MUTED}; margin-top:16px;'>"
            f"<div style='font-size:32px; margin-bottom:12px;'>🔍</div>"
            f"<div style='font-size:14px; font-weight:600; color:{TEXT_SECONDARY}; margin-bottom:8px;'>"
            f"Screener de Sector</div>"
            f"<div style='font-size:12px;'>Selecciona un sector y haz clic en Escanear para rankear "
            f"todas las empresas representativas por score WealthView.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    if run:
        n = len(SECTOR_PEERS.get(sector, []))
        with st.spinner(f"Escaneando {n} empresas del sector {sector}... (~20-30 seg)"):
            results = scan_sector(sector)
        st.session_state["screener_results"] = results
        st.session_state["screener_sector_name"] = sector
    else:
        results = st.session_state.get("screener_results", [])
        sector  = st.session_state.get("screener_sector_name", sector)

    if not results:
        st.error("No se pudieron obtener datos para este sector. Inténtalo de nuevo.")
        return

    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin:8px 0 16px;'>"
        f"Escaneadas {len(results)} empresas del sector <b>{sector}</b> · "
        f"Actualizado: {datetime.now().strftime('%H:%M')}</p>",
        unsafe_allow_html=True,
    )

    # ── KPI summary ───────────────────────────────────────────────────────────
    scores = [r["_score_raw"] for r in results]
    buys   = sum(1 for r in results if r["_rec"] in ("COMPRA", "COMPRA FUERTE"))
    sells  = sum(1 for r in results if r["_rec"] in ("VENDER", "VENTA FUERTE"))

    c1, c2, c3, c4 = st.columns(4)
    def _card(title, val, color=GOLD, sub=""):
        return (
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:14px 10px; text-align:center;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:5px;'>{title}</div>"
            f"<div style='font-size:20px; font-weight:700; color:{color};'>{val}</div>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:3px;'>{sub}</div>"
            f"</div>"
        )

    with c1:
        st.markdown(_card("Score promedio", f"{np.mean(scores):.1f}/10",
                           GOLD, f"de {len(results)} empresas"), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Score máximo", f"{max(scores):.1f}",
                           POSITIVE, results[0]["Ticker"]), unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Señales COMPRA", str(buys),
                           POSITIVE, f"{buys/len(results)*100:.0f}% del sector"), unsafe_allow_html=True)
    with c4:
        st.markdown(_card("Señales VENDER", str(sells),
                           NEGATIVE, f"{sells/len(results)*100:.0f}% del sector"), unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    display_cols = [
        "Ticker", "Empresa", "Precio", "Cap.", "P/E", "Fwd P/E",
        "Mg. Neto", "ROE", "Crec. Rev.", "Upside", "WV Score", "Recomendación"
    ]
    df = pd.DataFrame(results)[display_cols]

    def _color_rec(val):
        if "FUERTE" in str(val) and "COMPRA" in str(val):
            return f"background-color: rgba(74,143,90,0.25); color: {POSITIVE}; font-weight:700"
        if "COMPRA" in str(val):
            return f"color: {POSITIVE}; font-weight:600"
        if "FUERTE" in str(val) and "VENTA" in str(val):
            return f"background-color: rgba(155,77,77,0.25); color: {NEGATIVE}; font-weight:700"
        if "VENDER" in str(val):
            return f"color: {NEGATIVE}; font-weight:600"
        return f"color: {TEXT_MUTED}"

    def _color_score(val):
        try:
            v = float(val)
            if v >= 7.5:   return f"color: {POSITIVE}; font-weight:700"
            if v >= 6.2:   return f"color: {GOLD}; font-weight:600"
            if v >= 4.5:   return f"color: {TEXT_MUTED}"
            return f"color: {NEGATIVE}"
        except Exception:
            return ""

    styled = (
        df.style
        .map(_color_rec,   subset=["Recomendación"])
        .map(_color_score, subset=["WV Score"])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=580)

    # ── Deep Dive quick links ─────────────────────────────────────────────────
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    st.caption("Haz clic en un ticker para abrir su Deep Dive:")
    cols = st.columns(min(len(results), 8))
    for i, (col, r) in enumerate(zip(cols, results[:8])):
        with col:
            if st.button(r["Ticker"], key=f"screener_dd_{r['Ticker']}", use_container_width=True):
                st.session_state["deep_dive_ticker"] = r["Ticker"]
                st.session_state["page"] = "Deep Dive"
                st.rerun()

    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:16px;'>"
        "⚠ Scores calculados con datos de yfinance en tiempo real. El screener es un punto de partida — "
        "abre el Deep Dive de cada empresa para el análisis completo. "
        "Caché: 1 hora. Costes de transacción y liquidez no considerados.</p>",
        unsafe_allow_html=True,
    )

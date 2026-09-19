"""
edgar.py — SEC EDGAR integration for WealthView
Fetches insider trading (Form 4) and institutional ownership (13F)
via the free SEC EDGAR REST API. No API key required.
"""

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK
)

EDGAR_HEADERS = {"User-Agent": "WealthView research@wealthview.app"}
EDGAR_BASE    = "https://data.sec.gov"
EDGAR_SEARCH  = "https://efts.sec.gov/LATEST/search-index"


# ── CIK lookup ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _get_cik(ticker: str) -> str | None:
    """Resolve ticker → CIK (10-digit padded string)."""
    try:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=&dateb=&owner=include&count=1&search_text=&output=atom"
        # Use the company tickers JSON instead — more reliable
        url2 = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url2, headers=EDGAR_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        ticker_up = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_up:
                cik = str(entry["cik_str"]).zfill(10)
                return cik
        return None
    except Exception:
        return None


# ── Form 4: Insider trading ───────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_insider_trades(ticker: str, max_filings: int = 20) -> list[dict]:
    """
    Fetch recent Form 4 filings (insider buys/sells) for a ticker.
    Returns list of dicts with: date, insider_name, title, transaction_type,
    shares, price_per_share, value, shares_owned_after.
    """
    cik = _get_cik(ticker)
    if not cik:
        return []

    try:
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        recent = data.get("filings", {}).get("recent", {})
        forms   = recent.get("form", [])
        dates   = recent.get("filingDate", [])
        accnums = recent.get("accessionNumber", [])

        # Filter Form 4 filings
        form4_indices = [i for i, f in enumerate(forms) if f == "4"][:max_filings]
        if not form4_indices:
            return []

        trades = []
        for idx in form4_indices[:10]:   # limit to 10 most recent
            acc = accnums[idx].replace("-", "")
            filing_date = dates[idx]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{accnums[idx]}-index.htm"

            # Parse the XML directly for transaction data
            xml_url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc}/"
            try:
                idx_r = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=8)
                # Find the form4 XML file
                xml_file = None
                for line in idx_r.text.split('\n'):
                    if '.xml' in line.lower() and 'form4' not in line.lower() and '<a href' in line.lower():
                        import re
                        match = re.search(r'href="([^"]*\.xml)"', line, re.IGNORECASE)
                        if match:
                            xml_file = match.group(1)
                            break
                if not xml_file:
                    # Try index json
                    idx_json = requests.get(
                        f"{EDGAR_BASE}/submissions/CIK{cik}.json",
                        headers=EDGAR_HEADERS, timeout=8
                    )
                    continue

                xml_r = requests.get(
                    f"https://www.sec.gov{xml_file}" if xml_file.startswith('/') else
                    f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc}/{xml_file}",
                    headers=EDGAR_HEADERS, timeout=8
                )
                xml_text = xml_r.text

                # Extract key fields with regex
                import re
                def _xml_val(tag, text):
                    m = re.search(rf'<{tag}[^>]*>([^<]+)</{tag}>', text, re.IGNORECASE)
                    return m.group(1).strip() if m else None

                name  = _xml_val("rptOwnerName", xml_text) or "Unknown"
                title = _xml_val("officerTitle", xml_text) or _xml_val("relationship", xml_text) or ""
                tx_type  = _xml_val("transactionCode", xml_text)
                shares   = _xml_val("transactionShares", xml_text)
                price    = _xml_val("transactionPricePerShare", xml_text)
                owned    = _xml_val("sharesOwnedFollowingTransaction", xml_text)

                tx_label = {
                    "P": "Compra", "S": "Venta", "A": "Award",
                    "D": "Disposición", "M": "Ejercicio opción",
                    "F": "Retención fiscal", "G": "Gift",
                }.get(tx_type or "", tx_type or "—")

                shares_n = float(shares) if shares else None
                price_n  = float(price)  if price  else None
                value    = shares_n * price_n if (shares_n and price_n) else None

                trades.append({
                    "fecha":         filing_date,
                    "insider":       name,
                    "cargo":         title[:40] if title else "—",
                    "tipo":          tx_label,
                    "codigo":        tx_type or "—",
                    "acciones":      shares_n,
                    "precio":        price_n,
                    "valor":         value,
                    "total_despues": float(owned) if owned else None,
                    "url":           doc_url,
                })
            except Exception:
                continue

        return trades

    except Exception:
        return []


# ── 13F: Institutional ownership ──────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_institutional_ownership(ticker: str) -> list[dict]:
    """
    Fetch recent 13F filings to show institutional ownership.
    Uses yfinance institutionalHolders as primary (simpler + more reliable),
    with EDGAR as supplement for recent changes.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        inst_df = tk.institutional_holders
        if inst_df is not None and not inst_df.empty:
            records = []
            for _, row in inst_df.head(15).iterrows():
                records.append({
                    "institucion": str(row.get("Holder", "—")),
                    "acciones":    row.get("Shares"),
                    "valor":       row.get("Value"),
                    "pct_out":     row.get("% Out"),
                    "fecha":       str(row.get("Date Reported", "—")),
                })
            return records
    except Exception:
        pass
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_institutional_changes(ticker: str) -> dict:
    """
    Fetch recent changes in institutional ownership (new/sold positions).
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        mut_df  = tk.mutualfund_holders
        inst_df = tk.institutional_holders

        total_inst_pct = None
        if inst_df is not None and not inst_df.empty and "% Out" in inst_df.columns:
            total_inst_pct = inst_df["% Out"].sum()

        return {
            "total_institutional_pct": total_inst_pct,
            "n_institutional":         len(inst_df) if inst_df is not None else 0,
            "n_mutual_funds":          len(mut_df)  if mut_df  is not None else 0,
        }
    except Exception:
        return {}


# ── Render functions ──────────────────────────────────────────────────────────

def _section(title: str):
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:18px 0 10px 0;'>"
        f"{title}</p>",
        unsafe_allow_html=True,
    )


def _card(title, value, sub="", color=None):
    c = color or GOLD
    return (
        f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
        f"border-radius:8px; padding:14px 12px; text-align:center;'>"
        f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:6px;'>{title}</div>"
        f"<div style='font-size:20px; font-weight:700; color:{c};'>{value}</div>"
        f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:4px;'>{sub}</div>"
        f"</div>"
    )


def render_edgar_tab(ticker: str):
    """Main render for the EDGAR / Insiders tab in Deep Dive."""

    tabs = st.tabs(["🧑‍💼 Insiders (Form 4)", "🏛️ Institucionales (13F)"])

    # ── Tab A: Insider Trading ────────────────────────────────────────────────
    with tabs[0]:
        _section(f"Operaciones recientes de insiders — {ticker}")
        st.caption(
            "Fuente: SEC EDGAR Form 4 (obligatorio declarar en 2 días hábiles). "
            "Las compras de insiders son una de las señales más fiables en finanzas conductuales."
        )

        with st.spinner("Consultando SEC EDGAR..."):
            trades = fetch_insider_trades(ticker)

        if not trades:
            # Fallback: Finviz insider trades (via data_provider)
            try:
                from modules.data_provider import get_insider_trades_df
                ins_df = get_insider_trades_df(ticker)
                if ins_df is not None and not ins_df.empty:
                    _render_yf_insiders(ins_df)
                else:
                    st.info("No se encontraron operaciones de insiders recientes en SEC EDGAR.")
            except Exception:
                st.info("No se encontraron operaciones de insiders recientes en SEC EDGAR.")
        else:
            _render_edgar_trades(trades)

    # ── Tab B: Institutional Ownership ───────────────────────────────────────
    with tabs[1]:
        _section(f"Tenencias institucionales — {ticker}")
        st.caption(
            "Fuente: SEC 13F + yfinance. Los fondos con > $100M en activos deben "
            "declarar sus posiciones trimestralmente."
        )

        with st.spinner("Cargando datos institucionales..."):
            institutions = fetch_institutional_ownership(ticker)
            changes      = fetch_institutional_changes(ticker)

        # KPI row
        c1, c2, c3 = st.columns(3)
        with c1:
            pct = changes.get("total_institutional_pct")
            st.markdown(_card(
                "% en manos institucionales",
                f"{pct*100:.1f}%" if pct else "N/D",
                "> 70% = alto interés institucional",
                POSITIVE if pct and pct > 0.5 else TEXT_MUTED,
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(_card(
                "Nº instituciones",
                str(changes.get("n_institutional", "—")),
                "declaradas en último 13F",
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(_card(
                "Fondos de inversión",
                str(changes.get("n_mutual_funds", "—")),
                "mutual funds con posición",
            ), unsafe_allow_html=True)

        if institutions:
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            _section("Top 15 accionistas institucionales")

            rows = []
            for inst in institutions:
                shares = inst.get("acciones")
                value  = inst.get("valor")
                pct_o  = inst.get("pct_out")
                rows.append({
                    "Institución":   inst.get("institucion", "—"),
                    "Acciones":      f"{int(shares):,}" if shares else "—",
                    "Valor ($)":     f"${value/1e6:.1f}M" if value else "—",
                    "% del float":   f"{pct_o*100:.2f}%" if pct_o else "—",
                    "Actualización": inst.get("fecha", "—"),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de tenencias institucionales disponibles.")

        st.markdown(
            f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:12px;'>"
            "⚠ Datos 13F con retraso de hasta 45 días (fecha límite de declaración). "
            "Las posiciones actuales pueden diferir de las mostradas.</p>",
            unsafe_allow_html=True,
        )


def _render_yf_insiders(df: pd.DataFrame):
    """Render yfinance insider_transactions fallback."""
    try:
        display = df.copy()
        # Rename common columns
        rename_map = {}
        for col in display.columns:
            cl = col.lower()
            if "date" in cl:        rename_map[col] = "Fecha"
            elif "insider" in cl:   rename_map[col] = "Insider"
            elif "title" in cl:     rename_map[col] = "Cargo"
            elif "transaction" in cl and "shares" not in cl: rename_map[col] = "Tipo"
            elif "shares" in cl:    rename_map[col] = "Acciones"
            elif "value" in cl:     rename_map[col] = "Valor ($)"
            elif "ownership" in cl: rename_map[col] = "Propiedad"
        display = display.rename(columns=rename_map)

        # Deduplicar columnas y resetear índice para que Styler no falle
        seen_cols: dict = {}
        new_cols = []
        for c in display.columns:
            if c in seen_cols:
                seen_cols[c] += 1
                new_cols.append(f"{c}.{seen_cols[c]}")
            else:
                seen_cols[c] = 0
                new_cols.append(c)
        display.columns = new_cols
        display = display.reset_index(drop=True)

        # Color buy vs sell
        def _style_type(val):
            val_s = str(val).lower()
            if any(k in val_s for k in ["buy", "purchase", "compra", "acquisition"]):
                return f"color: {POSITIVE}; font-weight: 700"
            elif any(k in val_s for k in ["sell", "sale", "venta"]):
                return f"color: {NEGATIVE}; font-weight: 700"
            return ""

        if "Tipo" in display.columns:
            styled = display.style.map(_style_type, subset=["Tipo"])
        else:
            styled = display.style

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Insider signal summary
        if "Tipo" in display.columns:
            buys  = display["Tipo"].str.lower().str.contains("buy|purchase|compra", na=False).sum()
            sells = display["Tipo"].str.lower().str.contains("sell|sale|venta", na=False).sum()
            if buys > sells * 2:
                st.success(f"✅ Señal insider ALCISTA: {buys} compras vs {sells} ventas en el período.")
            elif sells > buys * 2:
                st.warning(f"⚠️ Señal insider BAJISTA: {sells} ventas vs {buys} compras en el período.")
            else:
                st.info(f"Actividad insider mixta: {buys} compras / {sells} ventas.")
    except Exception as e:
        st.error(f"Error procesando datos de insiders: {e}")


def _render_edgar_trades(trades: list[dict]):
    """Render EDGAR Form 4 trades."""
    buys  = [t for t in trades if t.get("codigo") == "P"]
    sells = [t for t in trades if t.get("codigo") == "S"]

    c1, c2 = st.columns(2)
    with c1:
        buy_val = sum(t["valor"] for t in buys if t.get("valor"))
        st.markdown(_card(
            "Compras (últimas 10 declaraciones)",
            str(len(buys)),
            f"${buy_val:,.0f} total" if buy_val else "",
            POSITIVE,
        ), unsafe_allow_html=True)
    with c2:
        sell_val = sum(t["valor"] for t in sells if t.get("valor"))
        st.markdown(_card(
            "Ventas (últimas 10 declaraciones)",
            str(len(sells)),
            f"${sell_val:,.0f} total" if sell_val else "",
            NEGATIVE,
        ), unsafe_allow_html=True)

    rows = []
    for t in trades:
        rows.append({
            "Fecha":     t["fecha"],
            "Insider":   t["insider"],
            "Cargo":     t["cargo"],
            "Tipo":      t["tipo"],
            "Acciones":  f"{t['acciones']:,.0f}" if t.get("acciones") else "—",
            "Precio":    f"${t['precio']:.2f}" if t.get("precio") else "—",
            "Valor":     f"${t['valor']:,.0f}" if t.get("valor") else "—",
        })

    if rows:
        df = pd.DataFrame(rows)
        def _style_tipo(val):
            if "Compra" in str(val): return f"color:{POSITIVE}; font-weight:700"
            if "Venta"  in str(val): return f"color:{NEGATIVE}; font-weight:700"
            return ""
        st.dataframe(
            df.style.map(_style_tipo, subset=["Tipo"]),
            use_container_width=True, hide_index=True,
        )

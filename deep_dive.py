"""
modules/deep_dive.py
────────────────────
Página "Deep Dive" — análisis completo de un ticker individual.
Tabs: Overview (fundamentales) · Técnico (RSI/MACD/SMAs) · Tesis (escenarios + EV) · Exportar
"""

import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import modules.auth as auth
import modules.llm_analyst as llm_analyst
import modules.edgar as edgar
import modules.quality_scores as quality_scores
import modules.backtesting as backtesting
from modules.peers import get_sector_medians, normalize_sector
from modules.styles import (
    PLOTLY_DARK, GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG,
    page_header, section_label, kpi_card, gold_divider, plotly_layout
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt(v, prefix="", suffix="", decimals=2, na="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    if abs(v) >= 1e9:
        return f"{prefix}{v/1e9:.{decimals}f}B{suffix}"
    if abs(v) >= 1e6:
        return f"{prefix}{v/1e6:.{decimals}f}M{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _pct(v, na="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return f"{v*100:.2f}%"


def _safe(d, key, default=None):
    v = d.get(key, default)
    return default if v in (None, "N/A", "", "None") else v


# ── data fetching ─────────────────────────────────────────────────────────────

import modules.data_provider as _dp  # Finviz-primary data layer
from modules.fmp import enrich_info_with_fmp  # FMP enrichment layer

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_info(ticker: str):
    try:
        base = _dp.get_fundamentals(ticker)
        # Enriquecer con FMP donde yfinance/Finviz fallan (FCF, shares, forward PE…)
        return enrich_info_with_fmp(base, ticker)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_history(ticker: str, period: str = "1y"):
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw
    except Exception:
        return pd.DataFrame()


# ── financial history ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_financials(ticker: str):
    """Return (income_df, cashflow_df) annual statements, columns sorted newest→oldest."""
    try:
        t = yf.Ticker(ticker)
        fin = t.financials   # rows = metrics, cols = year dates
        cf  = t.cashflow
        if fin is not None and not fin.empty:
            fin = fin.reindex(sorted(fin.columns, reverse=True), axis=1)
        if cf is not None and not cf.empty:
            cf = cf.reindex(sorted(cf.columns, reverse=True), axis=1)
        return fin, cf
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ── real-time news ───────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_news(ticker: str):
    """
    Return list of recent news dicts: {title, publisher, url, ts, sentiment}.
    Sentiment: +1 positive, -1 negative, 0 neutral (keyword-based heuristic).
    """
    try:
        raw_news = _dp.get_news(ticker)
    except Exception:
        return []

    POS_KW = {
        "beat", "beats", "surge", "surges", "rally", "rallies", "record",
        "profit", "upgrade", "upgraded", "bullish", "strong", "exceed",
        "exceeds", "raised", "raises", "outperform", "growth", "gains",
        "acquisition", "partnership", "deal", "buy", "overweight", "upside",
        "expansion", "innovation", "milestone", "breakthrough", "approval",
        "approved", "wins", "win", "soars", "soar", "jumps", "jump",
    }
    NEG_KW = {
        "miss", "misses", "fall", "falls", "decline", "declines", "loss",
        "losses", "downgrade", "downgraded", "bearish", "weak", "cut",
        "cuts", "lawsuit", "regulatory", "fraud", "bankruptcy", "sell",
        "underperform", "underweight", "concern", "concerns", "risk",
        "risks", "warning", "probe", "investigation", "recall", "drops",
        "drop", "plunges", "plunge", "slump", "slumps", "disappoints",
        "disappointing", "layoffs", "layoff", "restructuring",
    }

    results = []
    for item in raw_news[:12]:
        title = item.get("title", "")
        words = set(re.findall(r"[a-z]+", title.lower()))
        pos   = len(words & POS_KW)
        neg   = len(words & NEG_KW)
        if pos > neg:   sent = 1
        elif neg > pos: sent = -1
        else:           sent = 0

        ts = item.get("providerPublishTime", 0)
        results.append({
            "title":     title,
            "publisher": item.get("publisher", ""),
            "url":       item.get("link", ""),
            "ts":        ts,
            "sentiment": sent,
        })
    return results


# ── Sector median reference values (used for relative scoring) ───────────────
# Source: historical S&P 500 sector medians — updated periodically
SECTOR_MEDIANS = {
    "Technology": {
        "pe": 28, "ps": 6.0, "ev_ebitda": 22, "gross_m": 0.60,
        "net_m": 0.18, "op_m": 0.22, "roe": 0.22, "de": 0.50,
    },
    "Healthcare": {
        "pe": 22, "ps": 4.0, "ev_ebitda": 16, "gross_m": 0.55,
        "net_m": 0.12, "op_m": 0.15, "roe": 0.15, "de": 0.60,
    },
    "Financial Services": {
        "pe": 13, "ps": 2.5, "ev_ebitda": 12, "gross_m": 0.75,
        "net_m": 0.20, "op_m": 0.30, "roe": 0.12, "de": 3.00,
    },
    "Consumer Cyclical": {
        "pe": 18, "ps": 1.2, "ev_ebitda": 11, "gross_m": 0.35,
        "net_m": 0.06, "op_m": 0.08, "roe": 0.15, "de": 0.90,
    },
    "Consumer Defensive": {
        "pe": 20, "ps": 1.2, "ev_ebitda": 14, "gross_m": 0.38,
        "net_m": 0.07, "op_m": 0.10, "roe": 0.18, "de": 0.80,
    },
    "Energy": {
        "pe": 11, "ps": 1.0, "ev_ebitda": 7, "gross_m": 0.28,
        "net_m": 0.08, "op_m": 0.12, "roe": 0.12, "de": 0.50,
    },
    "Industrials": {
        "pe": 20, "ps": 1.5, "ev_ebitda": 14, "gross_m": 0.33,
        "net_m": 0.08, "op_m": 0.11, "roe": 0.15, "de": 0.80,
    },
    "Basic Materials": {
        "pe": 13, "ps": 1.2, "ev_ebitda": 8, "gross_m": 0.28,
        "net_m": 0.09, "op_m": 0.12, "roe": 0.12, "de": 0.50,
    },
    "Real Estate": {
        "pe": 38, "ps": 6.0, "ev_ebitda": 20, "gross_m": 0.50,
        "net_m": 0.15, "op_m": 0.25, "roe": 0.05, "de": 1.50,
    },
    "Utilities": {
        "pe": 18, "ps": 2.0, "ev_ebitda": 12, "gross_m": 0.40,
        "net_m": 0.12, "op_m": 0.18, "roe": 0.10, "de": 1.40,
    },
    "Communication Services": {
        "pe": 21, "ps": 2.8, "ev_ebitda": 13, "gross_m": 0.52,
        "net_m": 0.14, "op_m": 0.18, "roe": 0.16, "de": 0.70,
    },
}
_SECTOR_DEFAULT = {
    "pe": 20, "ps": 2.5, "ev_ebitda": 14, "gross_m": 0.40,
    "net_m": 0.10, "op_m": 0.13, "roe": 0.14, "de": 0.80,
}

# ── technical indicators ──────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


# ── thesis persistence ────────────────────────────────────────────────────────

def _thesis_key(ticker: str) -> str:
    return f"thesis_{ticker.upper()}"


def _load_thesis(username: str, ticker: str) -> dict:
    raw = auth.load_user_kv(username, _thesis_key(ticker))
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "scenarios": [
            {"Escenario": "Negativo",  "Precio objetivo ($)": 0.0, "Probabilidad (%)": 10},
            {"Escenario": "Moderado",  "Precio objetivo ($)": 0.0, "Probabilidad (%)": 40},
            {"Escenario": "Re-rating", "Precio objetivo ($)": 0.0, "Probabilidad (%)": 30},
            {"Escenario": "Fuerte",    "Precio objetivo ($)": 0.0, "Probabilidad (%)": 15},
            {"Escenario": "Outlier",   "Precio objetivo ($)": 0.0, "Probabilidad (%)": 5},
        ],
        "notes": "",
        "milestones": [],
        "entry_price": 0.0,
    }


def _save_thesis(username: str, ticker: str, data: dict):
    auth.save_user_kv(username, _thesis_key(ticker), json.dumps(data))


# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────

def _render_news_keyword(news_items: list):
    """Fallback news render using keyword sentiment (no LLM)."""
    for item in news_items[:8]:
        sent = item.get("sentiment", 0)
        icon  = "▲" if sent == 1 else "▼" if sent == -1 else "●"
        color = POSITIVE if sent == 1 else NEGATIVE if sent == -1 else TEXT_MUTED
        ts = item.get("ts")
        date_str = ""
        if ts:
            from datetime import datetime as _dt
            try: date_str = _dt.fromtimestamp(ts).strftime("%d %b")
            except: pass
        st.markdown(
            f"<div style='padding:7px 0; border-bottom:1px solid {BORDER_SOFT};'>"
            f"<span style='color:{color}; font-weight:700;'>{icon}</span> "
            f"<span style='font-size:12px; color:{TEXT_PRIMARY};'>{item['title']}</span> "
            f"<span style='font-size:10px; color:{TEXT_MUTED};'>· {item.get('publisher','')} {date_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )



def _render_overview(info: dict, ticker: str):
    name = _safe(info, "longName") or _safe(info, "shortName") or ticker
    sector = _safe(info, "sector", "—")
    industry = _safe(info, "industry", "—")
    country = _safe(info, "country", "—")
    website = _safe(info, "website", "")
    summary = _safe(info, "longBusinessSummary", "")

    # Header
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <div style="font-family:'Playfair Display',serif; font-size:22px;
                    font-weight:700; color:{TEXT_PRIMARY};">{name}</div>
        <div style="color:{TEXT_MUTED}; font-size:12px; margin-top:4px;">
            <span style="color:{GOLD}; font-weight:600;">{ticker.upper()}</span>
            &nbsp;·&nbsp;{sector}&nbsp;·&nbsp;{industry}&nbsp;·&nbsp;{country}
            {'&nbsp;·&nbsp;<a href="' + website + '" target="_blank" style="color:' + TEXT_MUTED + ';">🔗 Web</a>' if website else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs fila 1: valoración ───────────────────────────────────────────────
    section_label("Valoración")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")
    mktcap = _safe(info, "marketCap")
    ev = _safe(info, "enterpriseValue")
    pe_ttm = _safe(info, "trailingPE")
    pe_fwd = _safe(info, "forwardPE")
    ps = _safe(info, "priceToSalesTrailing12Months")
    pb = _safe(info, "priceToBook")

    cards = [
        ("Precio", _fmt(price, "$", decimals=2) if price else "—"),
        ("Market Cap", _fmt(mktcap, "$") if mktcap else "—"),
        ("EV", _fmt(ev, "$") if ev else "—"),
        ("P/E (TTM)", f"{pe_ttm:.1f}x" if pe_ttm else "—"),
        ("P/E (Fwd)", f"{pe_fwd:.1f}x" if pe_fwd else "—"),
        ("P/S", f"{ps:.2f}x" if ps else "—"),
    ]
    for col, (lbl, val) in zip([c1, c2, c3, c4, c5, c6], cards):
        col.markdown(kpi_card(lbl, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPIs fila 2: finanzas ─────────────────────────────────────────────────
    section_label("Finanzas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    rev = _safe(info, "totalRevenue")
    gross_m = _safe(info, "grossMargins")
    op_m = _safe(info, "operatingMargins")
    net_m = _safe(info, "profitMargins")
    de = _safe(info, "debtToEquity")
    cr = _safe(info, "currentRatio")

    cards2 = [
        ("Revenue TTM", _fmt(rev, "$") if rev else "—"),
        ("Gross Margin", _pct(gross_m) if gross_m else "—"),
        ("Op. Margin", _pct(op_m) if op_m else "—"),
        ("Net Margin", _pct(net_m) if net_m else "—"),
        ("Debt / Equity", f"{de/100:.2f}" if de else "—"),
        ("Current Ratio", f"{cr:.2f}" if cr else "—"),
    ]
    for col, (lbl, val) in zip([c1, c2, c3, c4, c5, c6], cards2):
        col.markdown(kpi_card(lbl, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPIs fila 3: cortos + técnico ─────────────────────────────────────────
    section_label("Short Interest · Técnico · Analistas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    short_fl = _safe(info, "shortPercentOfFloat")
    short_rt = _safe(info, "shortRatio")
    shares_sh = _safe(info, "sharesShort")
    beta = _safe(info, "beta")
    high52 = _safe(info, "fiftyTwoWeekHigh")
    low52 = _safe(info, "fiftyTwoWeekLow")

    cards3 = [
        ("Short Float", _pct(short_fl) if short_fl else "—"),
        ("Short Ratio", f"{short_rt:.1f}" if short_rt else "—"),
        ("Shares Short", _fmt(shares_sh) if shares_sh else "—"),
        ("Beta", f"{beta:.2f}" if beta else "—"),
        ("52W High", _fmt(high52, "$", decimals=2) if high52 else "—"),
        ("52W Low",  _fmt(low52,  "$", decimals=2) if low52 else "—"),
    ]
    for col, (lbl, val) in zip([c1, c2, c3, c4, c5, c6], cards3):
        col.markdown(kpi_card(lbl, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Consenso analistas ────────────────────────────────────────────────────
    target_mean = _safe(info, "targetMeanPrice")
    target_hi   = _safe(info, "targetHighPrice")
    target_lo   = _safe(info, "targetLowPrice")
    _raw_rec    = (_safe(info, "recommendationKey") or "").upper().replace("_", " ").strip()
    rec         = _raw_rec if _raw_rec and _raw_rec not in ("NONE", "N/A", "-") else ""
    n_analysts  = _safe(info, "numberOfAnalystOpinions")

    if target_mean and price:
        section_label("Consenso de analistas")
        upside = (target_mean / price - 1) * 100
        upside_color = POSITIVE if upside >= 0 else NEGATIVE

        ca, cb, cc, cd = st.columns(4)
        ca.markdown(kpi_card("Precio objetivo medio", f"${target_mean:.2f}",
                              sub=f"Upside: <span style='color:{upside_color};font-weight:600;'>{upside:+.1f}%</span>"),
                    unsafe_allow_html=True)
        cb.markdown(kpi_card("Objetivo alto",   f"${target_hi:.2f}"  if target_hi else "—"), unsafe_allow_html=True)
        cc.markdown(kpi_card("Objetivo bajo",   f"${target_lo:.2f}"  if target_lo else "—"), unsafe_allow_html=True)
        cd.markdown(kpi_card("Recomendación",   rec or "—",
                              sub=f"{n_analysts} analistas" if n_analysts else ""),
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Descripción ───────────────────────────────────────────────────────────
    if summary:
        with st.expander("📋 Descripción del negocio"):
            st.write(summary)

    # ── Noticias recientes (GPT-4o si disponible, keyword si no) ─────────────
    section_label("Noticias recientes")
    _raw_news = _fetch_news(ticker)
    if _raw_news:
        from modules.llm_analyst import analyze_news_sentiment, llm_available
        import json as _json

        if llm_available():
            _headlines_payload = _json.dumps([
                {"title": n["title"], "publisher": n.get("publisher",""), "ts": n.get("ts",0)}
                for n in _raw_news[:10]
            ])
            with st.spinner("Analizando noticias con IA..."):
                _ai_news = analyze_news_sentiment(ticker, _headlines_payload)

            if not _ai_news.get("error") and _ai_news.get("items"):
                # Summary banner
                overall = _ai_news.get("overall","neutro")
                ov_color = POSITIVE if overall=="positivo" else NEGATIVE if overall=="negativo" else TEXT_MUTED
                ov_icon  = "✅" if overall=="positivo" else "⚠️" if overall=="negativo" else "ℹ️"
                st.markdown(
                    f"<div style='background:{SURFACE_2}; border-left:4px solid {ov_color}; "
                    f"border-radius:0 8px 8px 0; padding:10px 16px; margin-bottom:12px;'>"
                    f"<span style='font-size:12px; color:{TEXT_PRIMARY};'>"
                    f"{ov_icon} <b>Sentimiento general: {overall.upper()}</b> — {_ai_news.get('summary','')}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
                # Individual items
                ai_items = {item["idx"]: item for item in _ai_news["items"]}
                for i, news_item in enumerate(_raw_news[:10], 1):
                    ai = ai_items.get(i, {})
                    sent = ai.get("sentiment","neutro")
                    impact = ai.get("impact","bajo")
                    insight = ai.get("insight","")
                    s_color = POSITIVE if sent=="positivo" else NEGATIVE if sent=="negativo" else TEXT_MUTED
                    s_icon  = "▲" if sent=="positivo" else "▼" if sent=="negativo" else "●"
                    imp_badge = (f"<span style='background:rgba(201,168,76,0.15);color:{GOLD};"
                                 f"font-size:9px;padding:2px 6px;border-radius:4px;'>ALTO</span> "
                                 if impact=="alto" else "")
                    ts = news_item.get("ts")
                    date_str = ""
                    if ts:
                        from datetime import datetime as _dt
                        try: date_str = _dt.fromtimestamp(ts).strftime("%d %b")
                        except: pass
                    st.markdown(
                        f"<div style='padding:8px 0; border-bottom:1px solid {BORDER_SOFT};'>"
                        f"<div style='display:flex; align-items:flex-start; gap:10px;'>"
                        f"<span style='color:{s_color}; font-weight:700; font-size:14px; min-width:16px;'>{s_icon}</span>"
                        f"<div>"
                        f"<div style='font-size:12px; color:{TEXT_PRIMARY};'>{imp_badge}{news_item['title']}</div>"
                        f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:3px;'>"
                        f"{news_item.get('publisher','')} · {date_str}"
                        + (f" — <i>{insight}</i>" if insight else "")
                        + f"</div></div></div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                # Fallback to keyword rendering
                _render_news_keyword(_raw_news)
        else:
            _render_news_keyword(_raw_news)
    else:
        st.info("No hay noticias recientes disponibles.")


# ── TAB 1b: HISTORICAL FINANCIALS ───────────────────────────────────────────

def _render_historicos(ticker: str):
    """Annual revenue, gross profit, operating income, net income + FCF."""
    fin, cf = _fetch_financials(ticker)
    if fin is None or fin.empty:
        st.info("No hay datos de históricos financieros disponibles para este ticker.")
        return

    rows = {}
    for label, key in [
        ("Ingresos", "Total Revenue"),
        ("Beneficio bruto", "Gross Profit"),
        ("EBIT", "EBIT"),
        ("Beneficio neto", "Net Income"),
    ]:
        if key in fin.index:
            rows[label] = fin.loc[key]

    if not cf.empty:
        ocf_key = next((k for k in ["Operating Cash Flow", "Total Cash From Operating Activities"] if k in cf.index), None)
        capex_key = next((k for k in ["Capital Expenditure", "Capital Expenditures"] if k in cf.index), None)
        if ocf_key:
            ocf = cf.loc[ocf_key]
            rows["FCF (aprox.)"] = ocf if capex_key is None else ocf + cf.loc[capex_key]

    if not rows:
        st.info("Datos insuficientes para mostrar históricos.")
        return

    df = pd.DataFrame(rows).T
    df.columns = [str(c)[:4] for c in df.columns]  # year labels
    df = df / 1e9  # convert to billions

    fig = go.Figure()
    bar_metrics = ["Ingresos", "Beneficio bruto", "EBIT", "Beneficio neto"]
    colors_map = {
        "Ingresos":        "#c9a84c",
        "Beneficio bruto": "#5a8f6e",
        "EBIT":            "#4a7fa5",
        "Beneficio neto":  "#7a6eaa",
        "FCF (aprox.)":    "#e8c97a",
    }
    for metric in bar_metrics:
        if metric in df.index:
            vals = df.loc[metric].values
            fig.add_trace(go.Bar(
                name=metric,
                x=df.columns.tolist(),
                y=vals,
                marker_color=colors_map.get(metric, "#888"),
                opacity=0.85,
            ))
    if "FCF (aprox.)" in df.index:
        vals = df.loc["FCF (aprox.)"].values
        fig.add_trace(go.Scatter(
            name="FCF",
            x=df.columns.tolist(),
            y=vals,
            mode="lines+markers",
            line=dict(color=colors_map["FCF (aprox.)"], width=2, dash="dot"),
            marker=dict(size=7),
        ))

    layout = dict(**PLOTLY_DARK)
    layout["margin"] = dict(l=40, r=20, t=40, b=40)
    layout.update(
        title=None,
        barmode="group",
        yaxis_title="Miles de millones ($)",
        legend=dict(orientation="h", y=-0.15),
        height=340,
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display_df = df.copy()
    display_df.index.name = "Métrica"
    st.dataframe(
        display_df.map(lambda x: f"${x:.2f}B" if pd.notna(x) else "—"),
        use_container_width=True,
    )


# ── TAB 2: TÉCNICO ───────────────────────────────────────────────────────────

def _render_tecnico(ticker: str):
    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1A": "1y", "2A": "2y", "5A": "5y"}
    period_lbl = st.radio("Periodo", list(period_map.keys()), index=3, horizontal=True, key="dd_period")
    yf_period = period_map[period_lbl]

    with st.spinner("Cargando datos técnicos..."):
        df = _fetch_history(ticker, yf_period)

    if df.empty or "Close" not in df.columns:
        st.warning("No hay datos históricos disponibles para este ticker.")
        return

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else None

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi    = _calc_rsi(close)
    macd_line, macd_sig, macd_hist = _calc_macd(close)

    # ── Gráfico precio ────────────────────────────────────────────────────────
    row_heights = [0.55, 0.15, 0.15, 0.15] if volume is not None else [0.6, 0.2, 0.2]
    n_rows = 4 if volume is not None else 3
    row_specs = [[{}]] * n_rows

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
        specs=row_specs,
    )

    # Precio + vela (usamos línea para rendimiento; vela sería más pesado)
    change_total = (close.iloc[-1] / close.iloc[0] - 1) * 100
    line_color = POSITIVE if change_total >= 0 else NEGATIVE

    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, name="Precio",
        line=dict(color=line_color, width=1.8),
        hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
    ), row=1, col=1)

    for sma, label, color in [(sma20, "SMA 20", "#4a6fa5"), (sma50, "SMA 50", GOLD), (sma200, "SMA 200", "#9b4d4d")]:
        fig.add_trace(go.Scatter(
            x=sma.index, y=sma.values, name=label,
            line=dict(color=color, width=1.2, dash="dot"),
            hovertemplate=f"{label}: $%{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)

    # Volumen
    if volume is not None:
        vol_colors = [POSITIVE if c >= o else NEGATIVE
                      for c, o in zip(close.values, close.shift(1).values)]
        fig.add_trace(go.Bar(
            x=volume.index, y=volume.values, name="Volumen",
            marker_color=vol_colors, opacity=0.5,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # RSI
    rsi_row = 3 if volume is not None else 2
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi.values, name="RSI(14)",
        line=dict(color="#c9a84c", width=1.5),
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ), row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=NEGATIVE,  line_width=0.8, row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=POSITIVE,  line_width=0.8, row=rsi_row, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color=BORDER,    line_width=0.6, row=rsi_row, col=1)

    # MACD
    macd_row = 4 if volume is not None else 3
    hist_colors = [POSITIVE if v >= 0 else NEGATIVE for v in macd_hist.values]
    fig.add_trace(go.Bar(
        x=macd_hist.index, y=macd_hist.values, name="Histograma",
        marker_color=hist_colors, opacity=0.7,
        hovertemplate="Hist: %{y:.4f}<extra></extra>",
    ), row=macd_row, col=1)
    fig.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values, name="MACD",
        line=dict(color="#4a6fa5", width=1.4),
        hovertemplate="MACD: %{y:.4f}<extra></extra>",
    ), row=macd_row, col=1)
    fig.add_trace(go.Scatter(
        x=macd_sig.index, y=macd_sig.values, name="Señal",
        line=dict(color=GOLD, width=1.2, dash="dot"),
        hovertemplate="Señal: %{y:.4f}<extra></extra>",
    ), row=macd_row, col=1)

    # Layout
    layout = {k: v for k, v in PLOTLY_DARK.items()}
    layout["margin"] = dict(l=0, r=0, t=10, b=0)
    layout["height"] = 680
    layout["hovermode"] = "x unified"
    layout["xaxis_rangeslider_visible"] = False

    # Axis labels
    for i in range(1, n_rows + 1):
        layout[f"yaxis{i if i > 1 else ''}"] = dict(
            gridcolor=BORDER, tickfont=dict(size=9, color=TEXT_MUTED),
            showline=False, zeroline=False,
        )
        layout[f"xaxis{i if i > 1 else ''}"] = dict(
            gridcolor=BORDER, tickfont=dict(size=9, color=TEXT_MUTED),
            showline=False, zeroline=False, rangeslider_visible=False,
        )

    # Axis labels custom
    layout["yaxis_title"] = dict(text="Precio ($)", font=dict(size=10, color=TEXT_MUTED))
    if volume is not None:
        layout["yaxis2_title"] = dict(text="Vol", font=dict(size=9, color=TEXT_MUTED))
    layout[f"yaxis{rsi_row}_title"] = dict(text="RSI", font=dict(size=9, color=TEXT_MUTED))
    layout[f"yaxis{macd_row}_title"] = dict(text="MACD", font=dict(size=9, color=TEXT_MUTED))

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Stats rápidas ─────────────────────────────────────────────────────────
    last_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else None
    last_macd = macd_line.dropna().iloc[-1] if not macd_line.dropna().empty else None
    last_sig  = macd_sig.dropna().iloc[-1]  if not macd_sig.dropna().empty  else None
    last_close = close.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    # RSI lectura
    if last_rsi:
        rsi_label = "Sobrecompra" if last_rsi > 70 else ("Sobreventa" if last_rsi < 30 else "Neutral")
        rsi_color = NEGATIVE if last_rsi > 70 else (POSITIVE if last_rsi < 30 else TEXT_SECONDARY)
        col1.markdown(kpi_card("RSI (14)",
                               f"{last_rsi:.1f}",
                               sub=f"<span style='color:{rsi_color};'>{rsi_label}</span>"),
                      unsafe_allow_html=True)
    # MACD señal
    if last_macd and last_sig:
        macd_bullish = last_macd > last_sig
        col2.markdown(kpi_card("MACD señal",
                               "Alcista ▲" if macd_bullish else "Bajista ▼",
                               sub=f"MACD: {last_macd:.4f}"),
                      unsafe_allow_html=True)
    # vs SMAs
    above_sma50 = last_close > sma50.iloc[-1] if not sma50.dropna().empty else None
    if above_sma50 is not None:
        col3.markdown(kpi_card("vs SMA 50",
                               f"${sma50.iloc[-1]:,.2f}",
                               sub=f"<span style='color:{POSITIVE if above_sma50 else NEGATIVE};'>{'Por encima ▲' if above_sma50 else 'Por debajo ▼'}</span>"),
                      unsafe_allow_html=True)
    above_sma200 = last_close > sma200.iloc[-1] if not sma200.dropna().empty else None
    if above_sma200 is not None:
        col4.markdown(kpi_card("vs SMA 200",
                               f"${sma200.iloc[-1]:,.2f}",
                               sub=f"<span style='color:{POSITIVE if above_sma200 else NEGATIVE};'>{'Por encima ▲' if above_sma200 else 'Por debajo ▼'}</span>"),
                      unsafe_allow_html=True)


# ── TAB 3: TESIS ─────────────────────────────────────────────────────────────

def _render_tesis(ticker: str, current_price: float, username: str):
    thesis = _load_thesis(username, ticker)

    st.markdown(f"""
    <div style="background:{SURFACE}; border:1px solid {GOLD_BORDER}; border-radius:8px;
                padding:16px 20px; margin-bottom:20px;">
        <div style="font-size:10px; font-weight:700; color:{TEXT_MUTED};
                    text-transform:uppercase; letter-spacing:1.2px; margin-bottom:6px;">
            Constructor de Tesis · {ticker.upper()}
        </div>
        <div style="font-size:12px; color:{TEXT_SECONDARY}; line-height:1.6;">
            Define tus escenarios con precio objetivo y probabilidad.
            WealthView calcula automáticamente el <b style='color:{GOLD};'>Valor Esperado (EV)</b>
            y la simulación Monte Carlo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Precio de entrada ─────────────────────────────────────────────────────
    col_p, col_s = st.columns([1, 3])
    with col_p:
        entry = st.number_input(
            "Precio de entrada ($)",
            value=float(thesis.get("entry_price") or current_price or 0.0),
            min_value=0.0, step=0.01, format="%.2f", key="dd_entry"
        )
    thesis["entry_price"] = entry
    ref_price = entry if entry > 0 else (current_price or 1.0)

    # ── Tabla de escenarios ───────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:11px; color:{TEXT_MUTED}; margin-bottom:6px;'>Edita los escenarios directamente en la tabla:</p>", unsafe_allow_html=True)

    scen_df = pd.DataFrame(thesis["scenarios"])
    edited = st.data_editor(
        scen_df,
        num_rows="dynamic",
        use_container_width=True,
        key="dd_scenarios",
        column_config={
            "Escenario": st.column_config.TextColumn("Escenario", width="medium"),
            "Precio objetivo ($)": st.column_config.NumberColumn(
                "Precio objetivo ($)", min_value=0.0, step=0.01, format="$%.2f"),
            "Probabilidad (%)": st.column_config.NumberColumn(
                "Probabilidad (%)", min_value=0, max_value=100, step=1, format="%d%%"),
        },
        hide_index=True,
    )
    thesis["scenarios"] = edited.to_dict("records")

    # Validación
    total_prob = sum(r.get("Probabilidad (%)", 0) for r in thesis["scenarios"])
    if abs(total_prob - 100) > 0.5:
        st.warning(f"⚠️ Las probabilidades suman {total_prob:.0f}%. Deben sumar 100%.")
    else:
        st.caption("✅ Probabilidades: 100%")

    # ── Cálculo EV ───────────────────────────────────────────────────────────
    st.markdown("---")
    section_label("Valor Esperado (EV)")

    rows_ev = []
    ev_total = 0.0
    valid = total_prob > 0

    for r in thesis["scenarios"]:
        name  = r.get("Escenario", "")
        price = r.get("Precio objetivo ($)", 0.0) or 0.0
        prob  = (r.get("Probabilidad (%)", 0) or 0) / 100.0
        ret   = (price / ref_price - 1) if ref_price > 0 and price > 0 else 0.0
        contrib = prob * ret
        ev_total += contrib
        rows_ev.append({
            "Escenario": name,
            "P. Objetivo": f"${price:.2f}",
            "Prob.": f"{prob*100:.0f}%",
            "Retorno": f"{ret*100:+.1f}%",
            "Contribución": f"{contrib*100:+.1f}%",
        })

    if rows_ev:
        df_ev = pd.DataFrame(rows_ev)
        st.dataframe(df_ev, use_container_width=True, hide_index=True)

        ev_color = POSITIVE if ev_total >= 0 else NEGATIVE
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card("EV Total",
                             f"{ev_total*100:+.1f}%",
                             sub="vs precio de entrada",
                             color=ev_color), unsafe_allow_html=True)
        c2.markdown(kpi_card("Precio medio ponderado",
                             f"${ref_price * (1 + ev_total):,.2f}",
                             sub="escenario esperado"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Entrada de referencia",
                             f"${ref_price:,.2f}",
                             sub="precio base"), unsafe_allow_html=True)

        # ── Gráfico EV vs precio de entrada ──────────────────────────────────
        prices_range = np.linspace(ref_price * 0.3, ref_price * 3.5, 200)
        evs = []
        for p_in in prices_range:
            ev_p = 0.0
            for r in thesis["scenarios"]:
                prob  = (r.get("Probabilidad (%)", 0) or 0) / 100.0
                price = r.get("Precio objetivo ($)", 0.0) or 0.0
                ev_p += prob * ((price / p_in - 1) if p_in > 0 and price > 0 else 0.0)
            evs.append(ev_p)

        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(
            x=prices_range, y=[v * 100 for v in evs],
            mode="lines", line=dict(color=GOLD, width=2),
            fill="tozeroy",
            fillcolor=f"rgba(201,168,76,0.08)",
            hovertemplate="Entrada: $%{x:.2f}<br>EV: %{y:.1f}%<extra></extra>",
            name="EV",
        ))
        fig_ev.add_vline(x=ref_price, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)
        fig_ev.add_hline(y=0, line_color=BORDER, line_width=1)

        ev_layout = {k: v for k, v in PLOTLY_DARK.items()}
        ev_layout["margin"] = dict(l=0, r=0, t=20, b=0)
        ev_layout["height"] = 260
        ev_layout["xaxis"] = dict(title="Precio de entrada ($)", gridcolor=BORDER,
                                  tickfont=dict(size=10, color=TEXT_MUTED))
        ev_layout["yaxis"] = dict(title="EV (%)", gridcolor=BORDER,
                                  tickfont=dict(size=10, color=TEXT_MUTED),
                                  ticksuffix="%")
        fig_ev.update_layout(**ev_layout)
        st.plotly_chart(fig_ev, use_container_width=True)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    if valid and any(r.get("Precio objetivo ($)", 0) > 0 for r in thesis["scenarios"]):
        st.markdown("---")
        section_label("Simulación Monte Carlo (10.000 iteraciones)")

        scenarios_valid = [r for r in thesis["scenarios"] if r.get("Precio objetivo ($)", 0) > 0]
        probs  = np.array([(r.get("Probabilidad (%)", 0) or 0) / 100.0 for r in scenarios_valid])
        prices_scen = np.array([r.get("Precio objetivo ($)", 0.0) for r in scenarios_valid])
        names_scen  = [r.get("Escenario", "") for r in scenarios_valid]

        # Normalizar probs
        if probs.sum() > 0:
            probs = probs / probs.sum()

        n_sim = 10_000
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(scenarios_valid), size=n_sim, p=probs)
        sim_prices = prices_scen[chosen]

        mc_mean = sim_prices.mean()
        mc_median = np.median(sim_prices)
        pct_upside = (sim_prices > ref_price).mean() * 100

        cm1, cm2, cm3 = st.columns(3)
        cm1.markdown(kpi_card("Media simulada", f"${mc_mean:,.2f}",
                              color=POSITIVE if mc_mean > ref_price else NEGATIVE), unsafe_allow_html=True)
        cm2.markdown(kpi_card("Mediana simulada", f"${mc_median:,.2f}"), unsafe_allow_html=True)
        cm3.markdown(kpi_card("% simulaciones con upside", f"{pct_upside:.1f}%",
                              color=POSITIVE if pct_upside > 50 else NEGATIVE), unsafe_allow_html=True)

        # Histograma de distribución
        freq = {n: int((chosen == i).sum()) for i, n in enumerate(names_scen)}
        fig_mc = go.Figure()
        bar_colors = []
        for i, n in enumerate(names_scen):
            p_obj = scenarios_valid[i].get("Precio objetivo ($)", ref_price)
            bar_colors.append(POSITIVE if p_obj >= ref_price else NEGATIVE)

        fig_mc.add_trace(go.Bar(
            x=names_scen,
            y=[freq.get(n, 0) for n in names_scen],
            marker_color=bar_colors, opacity=0.85,
            text=[f"{freq.get(n, 0)/n_sim*100:.1f}%" for n in names_scen],
            textposition="outside",
            hovertemplate="%{x}: %{y} iteraciones<extra></extra>",
            name="Frecuencia",
        ))

        mc_layout = {k: v for k, v in PLOTLY_DARK.items()}
        mc_layout["margin"] = dict(l=0, r=0, t=30, b=0)
        mc_layout["height"] = 280
        mc_layout["yaxis"] = dict(title="Iteraciones", gridcolor=BORDER,
                                  tickfont=dict(size=10, color=TEXT_MUTED))
        mc_layout["xaxis"] = dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=TEXT_SECONDARY))
        fig_mc.update_layout(**mc_layout)
        st.plotly_chart(fig_mc, use_container_width=True)

    # ── Notas e hitos ─────────────────────────────────────────────────────────
    st.markdown("---")
    section_label("Notas de la tesis e hitos")

    notes = st.text_area(
        "Tesis, tesis de invalidación, hitos clave...",
        value=thesis.get("notes", ""),
        height=160,
        key="dd_notes",
        placeholder="Ejemplo: Tesis principal: expansión de indicaciones + adopción > 100 centros. Invalidador: pérdida sostenida de soporte + guidance negativo en Q3...",
    )
    thesis["notes"] = notes

    # ── Guardar ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Guardar tesis", type="primary", key="dd_save_thesis"):
        _save_thesis(username, ticker, thesis)
        st.success("✅ Tesis guardada.")


# ── TAB 4: EXPORTAR ──────────────────────────────────────────────────────────

def _render_exportar(ticker: str, info: dict, username: str):
    thesis = _load_thesis(username, ticker)
    price  = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice") or 0.0
    name   = _safe(info, "longName") or _safe(info, "shortName") or ticker

    st.markdown(f"""
    <div style="background:{SURFACE}; border:1px solid {GOLD_BORDER}; border-radius:8px;
                padding:20px; margin-bottom:20px;">
        <div style="font-size:14px; color:{TEXT_SECONDARY}; line-height:1.7;">
            Genera un informe PDF profesional con <b style="color:{GOLD};">todo el análisis</b>:
            valoración fundamental, indicadores técnicos, posicionamiento en cortos, consenso de
            analistas y tu tesis de inversión personal.<br>
            <span style="font-size:12px; color:{TEXT_MUTED};">
            Cada métrica incluye su explicación para que sea comprensible sin conocimientos previos.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    has_thesis = any(r.get("Precio objetivo ($)", 0) > 0 for r in thesis.get("scenarios", []))
    if not has_thesis:
        st.info("💡 Si defines escenarios en la pestaña **🎯 Tesis**, el PDF incluirá el análisis EV y Monte Carlo.")

    col_pdf, col_txt = st.columns([2, 1])
    with col_pdf:
        if st.button("📄 Generar PDF del análisis", type="primary", use_container_width=True, key="dd_gen_pdf"):
            with st.spinner("Generando informe PDF..."):
                try:
                    df_hist = _fetch_history(ticker, "1y")
                    pdf_bytes = _generate_deep_dive_pdf(ticker, info, thesis, df_hist)
                    fname = f"DeepDive_{ticker.upper()}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        label="⬇️ Descargar PDF",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        type="primary",
                        key="dd_dl_pdf",
                    )
                    st.success("✅ PDF generado correctamente.")
                except Exception as e:
                    st.error(f"Error generando PDF: {e}")

    # Resumen .txt como alternativa rápida
    with col_txt:
        with st.expander("⬇️ Exportar resumen .txt"):
            entry = thesis.get("entry_price", 0.0)
            ref_price = entry if entry and entry > 0 else (price or 1.0)
            ev_total = 0.0
            scen_lines = []
            for r in thesis.get("scenarios", []):
                p_obj = r.get("Precio objetivo ($)", 0.0) or 0.0
                prob  = (r.get("Probabilidad (%)", 0) or 0) / 100.0
                ret   = (p_obj / ref_price - 1) if ref_price > 0 and p_obj > 0 else 0.0
                ev_total += prob * ret
                scen_lines.append(f"  {r.get('Escenario',''):12s} | ${p_obj:6.2f} | {prob*100:4.0f}% | {ret*100:+6.1f}%")
            price_str = f"${price:.2f}" if price else "—"
            summary_text = (
                f"DEEP DIVE — {name} ({ticker.upper()})\n"
                + "="*55 + "\n"
                + f"Precio actual:   {price_str}\n"
                + f"Precio entrada:  ${ref_price:.2f}\n"
                + f"EV calculado:    {ev_total*100:+.1f}%\n\n"
                + "ESCENARIOS\n" + "-"*55 + "\n"
                + ("Escenario     | P.Obj   | Prob  | Retorno\n" + "\n".join(scen_lines) if scen_lines else "(sin escenarios)") + "\n\n"
                + "NOTAS\n" + "-"*55 + "\n"
                + (thesis.get("notes", "") or "(sin notas)") + "\n\n"
                + f"Generado por WealthView · {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
            )
            st.download_button(
                "⬇️ Descargar .txt",
                data=summary_text,
                file_name=f"DeepDive_{ticker.upper()}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )

# ── PDF GENERATION ────────────────────────────────────────────────────────────

def _generate_deep_dive_pdf(ticker: str, info: dict, thesis: dict, df_hist) -> bytes:
    """Generate a professional Deep Dive PDF with WealthView styling."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib import colors
    from reportlab.platypus import Image as RLImage

    # ── Colors ────────────────────────────────────────────────────────────────
    C_NAVY    = colors.HexColor("#0d1b2a")
    C_BG      = colors.HexColor("#0e1117")
    C_SURFACE = colors.HexColor("#161b24")
    C_BORDER  = colors.HexColor("#232b3a")
    C_GOLD    = colors.HexColor("#c9a84c")
    C_GOLD_L  = colors.HexColor("#e8c97a")
    C_TEXT    = colors.HexColor("#e8e0d5")
    C_MUTED   = colors.HexColor("#7a8799")
    C_SEC     = colors.HexColor("#b8c5d0")
    C_POS     = colors.HexColor("#5a8f6e")
    C_NEG     = colors.HexColor("#9b4d4d")
    C_WHITE   = colors.white

    W, H = A4
    LMAR = RMAR = 18 * mm
    TMAR = 28 * mm
    BMAR = 20 * mm
    PW   = W - LMAR - RMAR

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=LMAR, rightMargin=RMAR,
        topMargin=TMAR, bottomMargin=BMAR,
    )

    # ── Canvas callbacks ──────────────────────────────────────────────────────
    company_name = (_safe(info, "longName") or _safe(info, "shortName") or ticker).upper()
    gen_date     = datetime.now().strftime("%d %b %Y")

    def _on_page(canvas, doc):
        canvas.saveState()
        # Navy header bar
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, H - 18 * mm, W, 18 * mm, fill=1, stroke=0)
        # Gold left accent
        canvas.setFillColor(C_GOLD)
        canvas.rect(0, 0, 3 * mm, H, fill=1, stroke=0)
        # Header text: WealthView brand
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(C_GOLD)
        canvas.drawString(LMAR, H - 11 * mm, "WEALTHVIEW")
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_TEXT)
        canvas.drawString(LMAR + 52, H - 11 * mm, f"| DEEP DIVE — {ticker.upper()}")
        # Page number + date on right
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_MUTED)
        page_str = f"{company_name}  ·  {gen_date}  ·  Pág. {doc.page}"
        canvas.drawRightString(W - RMAR, H - 11 * mm, page_str)
        # Gold bottom line
        canvas.setStrokeColor(C_GOLD)
        canvas.setLineWidth(0.6)
        canvas.line(LMAR, BMAR - 6 * mm, W - RMAR, BMAR - 6 * mm)
        # Footer
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(C_MUTED)
        canvas.drawString(LMAR, BMAR - 10 * mm,
            "Documento generado por WealthView · Solo informativo, no constituye asesoramiento de inversión.")
        canvas.restoreState()

    # ── Styles ────────────────────────────────────────────────────────────────
    def sty(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=9, textColor=C_SEC,
                        leading=13, spaceAfter=4)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    S_H1   = sty("H1", fontName="Helvetica-Bold", fontSize=16, textColor=C_TEXT,
                 spaceAfter=4, leading=20)
    S_H2   = sty("H2", fontName="Helvetica-Bold", fontSize=10, textColor=C_GOLD,
                 spaceBefore=12, spaceAfter=6, leading=14,
                 borderPad=4, borderColor=C_GOLD, borderWidth=0,
                 leftIndent=0)
    S_H3   = sty("H3", fontName="Helvetica-Bold", fontSize=8, textColor=C_MUTED,
                 spaceBefore=8, spaceAfter=4, leading=10)
    S_BODY = sty("BODY", fontSize=8.5, textColor=C_SEC, leading=13, spaceAfter=6)
    S_NOTE = sty("NOTE", fontSize=8, textColor=C_MUTED, leading=12,
                 leftIndent=10, spaceAfter=4,
                 fontName="Helvetica-Oblique")
    S_CAP  = sty("CAP", fontSize=7, textColor=C_MUTED, leading=10, spaceAfter=2)
    S_META = sty("META", fontSize=8, textColor=C_MUTED, leading=11)

    def hr(color=C_BORDER, thickness=0.5):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceAfter=8, spaceBefore=4)

    def gold_hr():
        return HRFlowable(width="100%", thickness=1, color=C_GOLD,
                          spaceAfter=10, spaceBefore=2)

    def section(title, explanation=""):
        elems = [
            Spacer(1, 4 * mm),
            gold_hr(),
            Paragraph(title.upper(), S_H2),
        ]
        if explanation:
            elems.append(Paragraph(explanation, S_NOTE))
        return elems

    def kv_table(rows, col_widths=None):
        """rows = list of (label, value, explanation) tuples"""
        if col_widths is None:
            col_widths = [PW * 0.28, PW * 0.22, PW * 0.50]
        data = [["Métrica", "Valor", "Qué significa"]]
        for label, value, explanation in rows:
            data.append([
                Paragraph(str(label), sty("KL", fontName="Helvetica-Bold",
                          fontSize=7.5, textColor=C_GOLD_L, leading=11)),
                Paragraph(str(value), sty("KV", fontName="Helvetica-Bold",
                          fontSize=8.5, textColor=C_TEXT, leading=12)),
                Paragraph(str(explanation), sty("KE", fontSize=7.5,
                          textColor=C_SEC, leading=11)),
            ])
        tbl = Table(data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            # Header row
            ("BACKGROUND",  (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), C_MUTED),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 7),
            ("ALIGN",       (0, 0), (-1, 0), "LEFT"),
            ("TOPPADDING",  (0, 0), (-1, 0), 4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            # Data rows
            ("BACKGROUND",  (0, 1), (-1, -1), C_BG),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_SURFACE]),
            ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",  (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID",        (0, 0), (-1, -1), 0.3, C_BORDER),
            ("LINEBELOW",   (0, 0), (-1, 0), 1, C_GOLD),
        ]))
        return tbl

    def scenario_table(scenarios, ref_price):
        headers = ["Escenario", "P. Objetivo", "Probabilidad", "Retorno esperado", "Contribución al EV"]
        data = [headers]
        ev_total = 0.0
        for r in scenarios:
            name   = r.get("Escenario", "")
            price  = r.get("Precio objetivo ($)", 0.0) or 0.0
            prob   = (r.get("Probabilidad (%)", 0) or 0) / 100.0
            ret    = (price / ref_price - 1) if ref_price > 0 and price > 0 else 0.0
            contrib = prob * ret
            ev_total += contrib
            ret_color  = "#5a8f6e" if ret >= 0 else "#9b4d4d"
            data.append([
                Paragraph(name, sty("SN", fontName="Helvetica-Bold", fontSize=8,
                          textColor=C_TEXT, leading=11)),
                Paragraph(f"${price:.2f}", sty("SP", fontName="Helvetica-Bold",
                          fontSize=8.5, textColor=C_GOLD_L, leading=11)),
                Paragraph(f"{prob*100:.0f}%", sty("SPR", fontSize=8,
                          textColor=C_SEC, leading=11)),
                Paragraph(f"{ret*100:+.1f}%",
                          sty("SR", fontName="Helvetica-Bold", fontSize=8.5,
                              textColor=colors.HexColor(ret_color), leading=11)),
                Paragraph(f"{contrib*100:+.1f}%",
                          sty("SC", fontSize=8, textColor=C_SEC, leading=11)),
            ])
        cw = [PW*0.22, PW*0.16, PW*0.17, PW*0.22, PW*0.23]
        tbl = Table(data, colWidths=cw)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_MUTED),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 7),
            ("TOPPADDING",    (0, 0), (-1, 0), 4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("LINEBELOW",     (0, 0), (-1, 0), 1, C_GOLD),
            ("BACKGROUND",    (0, 1), (-1, -1), C_BG),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_BG, C_SURFACE]),
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ]))
        return tbl, ev_total

    # ── matplotlib chart (price + RSI) ────────────────────────────────────────
    def _price_chart_png():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.ticker import FuncFormatter

            if df_hist is None or df_hist.empty or "Close" not in df_hist.columns:
                return None

            close = df_hist["Close"].squeeze()
            sma50  = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            rsi    = _calc_rsi(close)

            fig = plt.figure(figsize=(7.5, 4.2), facecolor="#0e1117")
            gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            change = (close.iloc[-1] / close.iloc[0] - 1) * 100
            line_c = "#5a8f6e" if change >= 0 else "#9b4d4d"

            ax1.plot(close.index, close.values, color=line_c, linewidth=1.5, label="Precio")
            ax1.fill_between(close.index, close.values, close.min(),
                             color=line_c, alpha=0.06)
            if not sma50.dropna().empty:
                ax1.plot(sma50.index, sma50.values, color="#c9a84c",
                         linewidth=1.0, linestyle="--", label="SMA 50")
            if not sma200.dropna().empty:
                ax1.plot(sma200.index, sma200.values, color="#9b4d4d",
                         linewidth=1.0, linestyle=":", label="SMA 200")

            ax1.set_facecolor("#0e1117")
            ax1.tick_params(colors="#7a8799", labelsize=7)
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax1.legend(fontsize=7, facecolor="#161b24", edgecolor="#232b3a",
                       labelcolor="#b8c5d0", loc="upper left")
            ax1.spines[:].set_color("#232b3a")
            ax1.grid(color="#232b3a", linewidth=0.4)
            plt.setp(ax1.get_xticklabels(), visible=False)

            # RSI
            rsi_vals = rsi.dropna()
            ax2.plot(rsi_vals.index, rsi_vals.values, color="#c9a84c", linewidth=1.2)
            ax2.axhline(70, color="#9b4d4d", linewidth=0.6, linestyle="--")
            ax2.axhline(30, color="#5a8f6e", linewidth=0.6, linestyle="--")
            ax2.axhline(50, color="#232b3a", linewidth=0.5)
            ax2.fill_between(rsi_vals.index, rsi_vals.values, 50,
                             where=(rsi_vals.values > 50), alpha=0.15, color="#5a8f6e")
            ax2.fill_between(rsi_vals.index, rsi_vals.values, 50,
                             where=(rsi_vals.values < 50), alpha=0.15, color="#9b4d4d")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI", color="#7a8799", fontsize=7)
            ax2.set_facecolor("#0e1117")
            ax2.tick_params(colors="#7a8799", labelsize=7)
            ax2.spines[:].set_color("#232b3a")
            ax2.grid(color="#232b3a", linewidth=0.4)

            fig.patch.set_facecolor("#0e1117")
            plt.tight_layout(pad=0.5)

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=140, bbox_inches="tight",
                        facecolor="#0e1117")
            plt.close(fig)
            img_buf.seek(0)
            return img_buf
        except Exception:
            return None


    # ── Dynamic contextual explanations ──────────────────────────────────────

    def _expl_mktcap(v):
        if v is None: return "Dato no disponible."
        if v >= 200e9: return f"Mega Cap (${v/1e9:.0f}B). Empresa gigante con alta liquidez. Menos riesgo pero también menos crecimiento."
        if v >= 10e9:  return f"Large Cap (${v/1e9:.1f}B). Empresa grande y consolidada. Más estable y menos volátil que empresas pequeñas."
        if v >= 2e9:   return f"Mid Cap (${v/1e9:.1f}B). Equilibrio entre crecimiento y estabilidad."
        return f"Small Cap (${v/1e6:.0f}M). Mayor potencial de subida pero más riesgo y volatilidad."

    def _expl_pe(v, label="P/E"):
        if v is None or (isinstance(v, float) and v != v): return "No disponible."
        if v <= 0:   return f"{label} negativo: la empresa tiene pérdidas. Este múltiplo no aplica aquí."
        if v < 10:   return f"{label} bajo ({v:.1f}x) — barato respecto a sus beneficios. Puede ser oportunidad o señal de desconfianza."
        if v < 18:   return f"{label} razonable ({v:.1f}x) — por debajo de la media del mercado (~20x). Valoración justa."
        if v < 28:   return f"{label} normal ({v:.1f}x) — en línea con el mercado. El precio refleja las expectativas actuales."
        if v < 50:   return f"{label} alto ({v:.1f}x) — el mercado espera mucho crecimiento. Si los resultados decepcionan, puede corregir con fuerza."
        return f"{label} muy alto ({v:.1f}x) — valoración de empresa de crecimiento puro. Muy sensible a noticias negativas."

    def _expl_ps(v):
        if v is None: return "Dato no disponible."
        if v < 1:    return f"P/S muy bajo ({v:.2f}x) — el mercado paga menos que sus ingresos anuales. Puede ser barata o tener problemas de margen."
        if v < 3:    return f"P/S razonable ({v:.2f}x) — valoración normal para empresas con márgenes moderados."
        if v < 8:    return f"P/S alto ({v:.2f}x) — prima por crecimiento de ingresos. Necesita seguir escalando para justificarlo."
        if v < 15:   return f"P/S muy alto ({v:.2f}x) — típico de SaaS. Requiere más del 30% de crecimiento anual para justificarse."
        return f"P/S extremo ({v:.2f}x) — el mercado descuenta crecimiento excepcional. Muy sensible a cualquier decepción."

    def _expl_pb(v):
        if v is None: return "Dato no disponible."
        if v < 0:    return f"P/B negativo ({v:.2f}x) — tiene más deuda que activos. Señal de alerta importante."
        if v < 1:    return f"P/B menor que 1 ({v:.2f}x) — cotiza por debajo de su valor en libros. Puede ser muy barata."
        if v < 2:    return f"P/B bajo ({v:.2f}x) — valoración conservadora. Habitual en banca e industria pesada."
        if v < 5:    return f"P/B moderado ({v:.2f}x) — razonable para empresas con marca o tecnología valiosa."
        return f"P/B alto ({v:.2f}x) — el mercado confía mucho en la rentabilidad futura del capital."

    def _expl_evebitda(v):
        if v is None: return "Dato no disponible."
        if v < 0:    return f"EV/EBITDA negativo ({v:.1f}x) — la empresa todavía no es rentable a nivel operativo."
        if v < 8:    return f"EV/EBITDA bajo ({v:.1f}x) — barato. Por debajo de 8x suele ser atractivo en sectores maduros."
        if v < 15:   return f"EV/EBITDA normal ({v:.1f}x) — valoración en línea con el mercado."
        if v < 25:   return f"EV/EBITDA alto ({v:.1f}x) — prima de crecimiento. Justificado si el negocio crece con buenos márgenes."
        return f"EV/EBITDA muy alto ({v:.1f}x) — caro. Muy sensible a cualquier revisión a la baja del EBITDA."

    def _expl_margin(v, tipo):
        if v is None: return "Dato no disponible."
        pct = v * 100
        if tipo == "gross":
            if pct < 20:   return f"Margen bruto bajo ({pct:.1f}%) — queda poco tras costes directos. Normal en manufactura o retail."
            if pct < 40:   return f"Margen bruto moderado ({pct:.1f}%) — la empresa retiene un porcentaje razonable de los ingresos."
            if pct < 60:   return f"Margen bruto sólido ({pct:.1f}%) — buen poder de precios. Habitual en tecnología o farma."
            return f"Margen bruto excelente ({pct:.1f}%) — típico de software puro. Mucho margen para cubrir gastos y ganar dinero."
        if tipo == "op":
            if pct < 0:    return f"Margen operativo negativo ({pct:.1f}%) — gasta más de lo que ingresa. Vigilar el nivel de caja."
            if pct < 5:    return f"Margen operativo muy bajo ({pct:.1f}%) — poca eficiencia. Normal en sectores de bajo margen."
            if pct < 15:   return f"Margen operativo moderado ({pct:.1f}%) — gana dinero operativamente con margen de mejora."
            if pct < 25:   return f"Margen operativo bueno ({pct:.1f}%) — controla bien sus costes fijos."
            return f"Margen operativo excelente ({pct:.1f}%) — negocio muy eficiente. Típico de software o plataformas digitales."
        if tipo == "net":
            if pct < 0:    return f"Margen neto negativo ({pct:.1f}%) — tiene pérdidas netas. Vigilar cuánto tiempo puede aguantar sin más financiación."
            if pct < 3:    return f"Margen neto muy bajo ({pct:.1f}%) — queda muy poco beneficio de cada euro ingresado."
            if pct < 10:   return f"Margen neto moderado ({pct:.1f}%) — empresa rentable que genera valor para el accionista."
            if pct < 20:   return f"Margen neto bueno ({pct:.1f}%) — retiene una parte importante de los ingresos como beneficio."
            return f"Margen neto excelente ({pct:.1f}%) — muy pocas empresas logran este nivel de forma sostenida."
        return ""

    def _expl_de(v):
        if v is None: return "Dato no disponible."
        ratio = v / 100.0
        if ratio < 0:    return f"D/E negativo ({ratio:.2f}) — más deuda que capital propio. Señal de alerta."
        if ratio < 0.3:  return f"D/E muy bajo ({ratio:.2f}) — prácticamente sin deuda. Máxima solidez financiera."
        if ratio < 0.8:  return f"D/E bajo ({ratio:.2f}) — apalancamiento conservador. Bajo riesgo financiero."
        if ratio < 1.5:  return f"D/E moderado ({ratio:.2f}) — deuda manejable. Amplifica retornos pero también el riesgo."
        if ratio < 2.5:  return f"D/E elevado ({ratio:.2f}) — deuda significativa. Los intereses pueden presionar los beneficios."
        return f"D/E muy alto ({ratio:.2f}) — empresa muy endeudada. Alta sensibilidad a subidas de tipos o caídas de ingresos."

    def _expl_cr(v, tipo="current"):
        if v is None: return "Dato no disponible."
        nombre = "Current Ratio" if tipo == "current" else "Quick Ratio"
        if v < 0.8:  return f"{nombre} bajo ({v:.2f}) — puede tener problemas para pagar sus deudas a corto plazo."
        if v < 1.2:  return f"{nombre} ajustado ({v:.2f}) — liquidez justa. Cumple obligaciones pero sin mucho margen."
        if v < 2.0:  return f"{nombre} bueno ({v:.2f}) — puede pagar sus deudas a corto con comodidad."
        if v < 3.5:  return f"{nombre} excelente ({v:.2f}) — muy buena liquidez. Sin tensiones financieras a corto."
        return f"{nombre} muy alto ({v:.2f}) — liquidez extrema, aunque puede indicar capital infrautilizado."

    def _expl_roe(v):
        if v is None: return "Dato no disponible."
        pct = v * 100
        if pct < 0:   return f"ROE negativo ({pct:.1f}%) — destruye valor sobre el capital invertido."
        if pct < 5:   return f"ROE bajo ({pct:.1f}%) — rentabilidad pobre. Por debajo del coste del capital en casi cualquier mercado."
        if pct < 15:  return f"ROE moderado ({pct:.1f}%) — razonable. Genera retornos sobre el capital con margen de mejora."
        if pct < 25:  return f"ROE bueno ({pct:.1f}%) — usa eficientemente el capital de los accionistas."
        return f"ROE excelente ({pct:.1f}%) — clase mundial. Indica ventaja competitiva duradera."

    def _expl_short_float(v):
        if v is None: return "Dato no disponible."
        pct = v * 100
        if pct < 3:   return f"Interés corto muy bajo ({pct:.1f}%) — el mercado no apuesta en su contra. Sin presión bajista."
        if pct < 10:  return f"Interés corto normal ({pct:.1f}%) — nivel habitual sin señales especiales."
        if pct < 20:  return f"Interés corto elevado ({pct:.1f}%) — bastante gente apuesta en contra. Si el precio sube, puede haber short squeeze."
        if pct < 30:  return f"Interés corto alto ({pct:.1f}%) — alta posición bajista. Gran potencial de squeeze con buenas noticias."
        return f"Interés corto extremo ({pct:.1f}%) — riesgo máximo en ambas direcciones. Un catalizador positivo puede disparar el precio."

    def _expl_short_ratio(v):
        if v is None: return "Dato no disponible."
        if v < 2:    return f"Short ratio bajo ({v:.1f} días) — los bajistas cubrirían rápido. Riesgo de squeeze limitado."
        if v < 5:    return f"Short ratio moderado ({v:.1f} días) — varios días para cubrir. Presión compradora si el precio sube."
        if v < 10:   return f"Short ratio alto ({v:.1f} días) — más de una semana para cubrir. Mayor riesgo de short squeeze."
        return f"Short ratio muy alto ({v:.1f} días) — tardarían más de {v:.0f} días en cubrir. Mercado muy polarizado."

    def _expl_rsi(v):
        if v is None: return "Dato no disponible."
        if v < 20:   return f"RSI en sobreventa extrema ({v:.1f}) — ha caído muy rápido. Zona de rebote habitual, aunque puede seguir bajando."
        if v < 30:   return f"RSI en sobreventa ({v:.1f}) — el precio ha corregido en exceso. Muchos lo ven como señal de compra táctica."
        if v < 45:   return f"RSI bajo ({v:.1f}) — momentum débil. El precio no tiene impulso claro."
        if v < 55:   return f"RSI neutral ({v:.1f}) — sin señales extremas. El precio está en equilibrio."
        if v < 70:   return f"RSI alto ({v:.1f}) — tendencia alcista sin llegar a sobrecompra. Señal positiva."
        if v < 80:   return f"RSI en sobrecompra ({v:.1f}) — ha subido mucho rápido. Mayor riesgo de corrección a corto."
        return f"RSI en sobrecompra extrema ({v:.1f}) — movimiento parabólico. Alta probabilidad de corrección técnica."

    def _expl_vs_sma(price_val, sma_val, periodo):
        if sma_val is None or price_val is None: return "Dato no disponible."
        diff = (price_val / sma_val - 1) * 100
        sma_name = f"SMA{periodo}"
        plazo = "medio plazo" if periodo == 50 else "largo plazo"
        if diff > 20:  return f"+{diff:.1f}% sobre la {sma_name} (${sma_val:,.2f}) — tendencia {plazo} alcista pero muy extendido. Posible corrección hacia la media."
        if diff > 0:   return f"+{diff:.1f}% sobre la {sma_name} (${sma_val:,.2f}) — tendencia {plazo} alcista. La media actúa como soporte dinámico."
        if diff < -20: return f"{diff:.1f}% bajo la {sma_name} (${sma_val:,.2f}) — tendencia {plazo} bajista marcada."
        return f"{diff:.1f}% bajo la {sma_name} (${sma_val:,.2f}) — tendencia {plazo} bajista."

    def _expl_52w(price_val, extreme_val, tipo):
        if extreme_val is None or price_val is None: return "Dato no disponible."
        diff = (price_val / extreme_val - 1) * 100
        if tipo == "high":
            if diff > -5:   return f"Cerca del máximo anual (${extreme_val:,.2f}). Zona de resistencia. Una ruptura con volumen sería muy alcista."
            if diff > -20:  return f"A {abs(diff):.1f}% del máximo anual (${extreme_val:,.2f}). Ha corregido desde máximos pero sigue en zona alta."
            return f"A {abs(diff):.1f}% del máximo anual (${extreme_val:,.2f}). Lejos de máximos: debilidad o posible oportunidad de entrada."
        else:
            if diff < 10:   return f"Cerca del mínimo anual (${extreme_val:,.2f}). Zona de soporte. Un rebote aquí sería señal positiva."
            if diff < 40:   return f"+{diff:.1f}% desde el mínimo anual (${extreme_val:,.2f}). Cierta recuperación desde los mínimos."
            return f"+{diff:.1f}% desde el mínimo anual (${extreme_val:,.2f}). El precio muestra fortaleza relativa en el año."

    def _expl_target(target_val, current_val, n_analysts):
        if target_val is None or current_val is None: return "Dato no disponible."
        upside = (target_val / current_val - 1) * 100
        analistas = f"{n_analysts} analistas" if n_analysts else "los analistas"
        if upside > 50:   return f"Potencial de +{upside:.1f}% según {analistas}. Consenso muy alcista."
        if upside > 20:   return f"Potencial de +{upside:.1f}% según {analistas}. El mercado profesional cree que está infravalorada."
        if upside > 5:    return f"Potencial de +{upside:.1f}% según {analistas}. Margen de subida moderado."
        if upside > -5:   return f"Potencial de {upside:+.1f}% según {analistas}. El precio está cerca del valor justo según el consenso."
        if upside > -20:  return f"El precio supera el objetivo en {abs(upside):.1f}%. Los analistas ven la empresa cara."
        return f"El precio supera el objetivo en {abs(upside):.1f}%. El consenso la considera claramente sobrevalorada."

    def _expl_rec(rec_str):
        r = (rec_str or "").upper().strip()
        if "STRONG BUY" in r: return "Compra fuerte — la gran mayoría de analistas recomienda comprar con convicción."
        if "BUY" in r:        return "Compra — los analistas ven valor a precios actuales."
        if "HOLD" in r:       return "Mantener — el consenso no ve motivos claros para comprar ni vender ahora."
        if "SELL" in r:       return "Vender — los analistas creen que está sobrevalorada o que el negocio empeorará."
        return "Sin recomendación de consenso disponible."


    # ── Build story ───────────────────────────────────────────────────────────
    story = []

    # ── PORTADA / HEADER ──────────────────────────────────────────────────────
    sector   = _safe(info, "sector",   "—")
    industry = _safe(info, "industry", "—")
    country  = _safe(info, "country",  "—")

    # Fetch real-time news (cached 15 min)
    _news_items = _fetch_news(ticker)
    price    = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice") or 0.0
    prev     = _safe(info, "previousClose") or price
    change   = ((price - prev) / prev * 100) if prev else 0.0
    mktcap   = _safe(info, "marketCap")
    ev_val   = _safe(info, "enterpriseValue")

    chg_color_hex = "#5a8f6e" if change >= 0 else "#9b4d4d"

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(company_name, S_H1))
    story.append(Paragraph(
        f'<font color="#c9a84c"><b>{ticker.upper()}</b></font>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;{sector}&nbsp;&nbsp;·&nbsp;&nbsp;{industry}'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{country}',
        S_META
    ))
    story.append(Spacer(1, 3 * mm))

    # Price badge row
    sign = "+" if change >= 0 else ""
    story.append(Paragraph(
        f'<font name="Helvetica-Bold" size="18" color="#e8e0d5">${price:,.2f}</font>'
        f'&nbsp;&nbsp;&nbsp;'
        f'<font name="Helvetica-Bold" size="11" color="{chg_color_hex}">'
        f'{sign}{change:.2f}% hoy</font>'
        f'&nbsp;&nbsp;&nbsp;&nbsp;'
        f'<font size="8" color="#7a8799">Generado el {gen_date}</font>',
        sty("PRICE", leading=22, spaceAfter=6)
    ))
    story.append(gold_hr())
    story.append(Paragraph(
        "Este informe ha sido generado por WealthView e incluye un análisis completo del activo: "
        "valoración fundamental, indicadores técnicos, posicionamiento en cortos, consenso de "
        "analistas y tesis de inversión personalizada. Cada métrica incluye una explicación "
        "para facilitar su interpretación. <b>Documento informativo — no constituye asesoramiento financiero.</b>",
        S_NOTE
    ))

    # ── SECCIÓN 1: VALORACIÓN FUNDAMENTAL ────────────────────────────────────
    story += section(
        "1. Valoración Fundamental",
        "La valoración fundamental analiza si una empresa cotiza cara o barata en relación a sus "
        "fundamentales. Los múltiplos se comparan con el sector y con el histórico de la compañía."
    )

    pe_ttm  = _safe(info, "trailingPE")
    pe_fwd  = _safe(info, "forwardPE")
    ps      = _safe(info, "priceToSalesTrailing12Months")
    pb      = _safe(info, "priceToBook")
    ev_ebit = _safe(info, "enterpriseToEbitda")

    val_rows = [
        ("Market Cap",
         _fmt(mktcap, "$") if mktcap else "—",
         _expl_mktcap(mktcap) if mktcap else "Dato no disponible."),
        ("Enterprise Value (EV)",
         _fmt(ev_val, "$") if ev_val else "—",
         (f"EV de {_fmt(ev_val, '$')} frente a Market Cap de {_fmt(mktcap, '$')}. " +
          ("EV < Market Cap: la empresa tiene más caja que deuda (posición neta de caja positiva). Señal de solidez financiera." if ev_val and mktcap and ev_val < mktcap else
           "EV > Market Cap: la empresa tiene deuda neta. El comprador real pagaría el EV, no solo el Market Cap."))
         if ev_val and mktcap else "Valor total incluyendo deuda y descontando caja. Dato no disponible."),
        ("P/E Trailing (TTM)",
         f"{pe_ttm:.1f}x" if pe_ttm else "—",
         _expl_pe(pe_ttm, "P/E TTM") if pe_ttm is not None else "Sin beneficios reportados — empresa en pérdidas o dato no publicado."),
        ("P/E Forward",
         f"{pe_fwd:.1f}x" if pe_fwd else "—",
         _expl_pe(pe_fwd, "P/E Forward") if pe_fwd is not None else "Sin estimación de beneficios futuros disponible."),
        ("P/S (Price/Sales)",
         f"{ps:.2f}x" if ps else "—",
         _expl_ps(ps) if ps is not None else "Dato no disponible."),
        ("P/B (Price/Book)",
         f"{pb:.2f}x" if pb else "—",
         _expl_pb(pb) if pb is not None else "Dato no disponible."),
        ("EV/EBITDA",
         f"{ev_ebit:.1f}x" if ev_ebit else "—",
         _expl_evebitda(ev_ebit) if ev_ebit is not None else "EBITDA negativo o dato no disponible — empresa sin rentabilidad operativa aún."),
    ]
    story.append(kv_table(val_rows))

    # ── SECCIÓN 2: SALUD FINANCIERA ───────────────────────────────────────────
    story += section(
        "2. Salud Financiera",
        "Mide la capacidad de la empresa para generar ingresos, controlar costes y hacer frente a sus obligaciones. "
        "Una empresa financieramente sana puede invertir en crecimiento incluso en mercados adversos."
    )

    rev     = _safe(info, "totalRevenue")
    gross_m = _safe(info, "grossMargins")
    op_m    = _safe(info, "operatingMargins")
    net_m   = _safe(info, "profitMargins")
    de      = _safe(info, "debtToEquity")
    cr      = _safe(info, "currentRatio")
    qr      = _safe(info, "quickRatio")
    roe     = _safe(info, "returnOnEquity")
    roa     = _safe(info, "returnOnAssets")
    fcf     = _safe(info, "freeCashflow")
    cash    = _safe(info, "totalCash")
    debt    = _safe(info, "totalDebt")
    empl    = _safe(info, "fullTimeEmployees")

    _rev_per_emp = (rev / empl) if (rev and empl and empl > 0) else None
    _fcf_yield   = (fcf / mktcap * 100) if (fcf and mktcap and mktcap > 0) else None
    _net_cash    = (cash - debt) if (cash is not None and debt is not None) else None

    fin_rows = [
        ("Revenue TTM",
         _fmt(rev, "$") if rev else "—",
         (f"Ingresos de {_fmt(rev, '$')} en los últimos 12 meses. " +
          (f"Ingresos por empleado: {_fmt(_rev_per_emp, '$')} — " +
           ("muy alto, indica alta productividad por persona." if _rev_per_emp and _rev_per_emp > 1e6 else
            "razonable para el tamaño de la empresa.") if _rev_per_emp else ""))
         if rev else "Dato no disponible."),
        ("Gross Margin",
         _pct(gross_m) if gross_m else "—",
         _expl_margin(gross_m, "gross") if gross_m is not None else "Dato no disponible."),
        ("Operating Margin",
         _pct(op_m) if op_m else "—",
         _expl_margin(op_m, "op") if op_m is not None else "Dato no disponible."),
        ("Net Margin",
         _pct(net_m) if net_m else "—",
         _expl_margin(net_m, "net") if net_m is not None else "Dato no disponible."),
        ("Free Cash Flow",
         _fmt(fcf, "$") if fcf else "—",
         (f"FCF de {_fmt(fcf, '$')}. " +
          (f"FCF yield sobre Market Cap: {_fcf_yield:.1f}% — " +
           ("muy atractivo, el mercado paga poco por cada euro de caja generada." if _fcf_yield and _fcf_yield > 5 else
            "moderado." if _fcf_yield and _fcf_yield > 2 else
            "bajo, la empresa vale mucho respecto a su generación de caja." if _fcf_yield and _fcf_yield > 0 else
            "FCF negativo: la empresa consume caja, necesita financiación externa para operar.") if _fcf_yield is not None else
           ("FCF negativo: la empresa no genera caja libre aún. Vigilar la caja disponible y el burn rate." if fcf and fcf < 0 else "")))
         if fcf is not None else "FCF no disponible."),
        ("Posición neta de caja",
         (f"${_net_cash/1e9:.2f}B" if _net_cash and abs(_net_cash) >= 1e9 else
          f"${_net_cash/1e6:.0f}M" if _net_cash else "—"),
         (f"Caja ({_fmt(cash, '$')}) minus deuda ({_fmt(debt, '$')}): " +
          ("posición neta de CAJA positiva. La empresa tiene más caja que deuda — máxima solidez financiera." if _net_cash and _net_cash > 0 else
           "posición neta de DEUDA. La empresa debe más de lo que tiene en caja. Hay que evaluar si el negocio genera suficiente FCF para servir la deuda."))
         if _net_cash is not None else "Datos de caja/deuda no disponibles."),
        ("Debt/Equity",
         f"{de/100:.2f}" if de else "—",
         _expl_de(de) if de is not None else "Dato no disponible."),
        ("Current Ratio",
         f"{cr:.2f}" if cr else "—",
         _expl_cr(cr, "current") if cr is not None else "Dato no disponible."),
        ("Quick Ratio",
         f"{qr:.2f}" if qr else "—",
         _expl_cr(qr, "quick") if qr is not None else "Dato no disponible."),
        ("ROE",
         _pct(roe) if roe else "—",
         _expl_roe(roe) if roe is not None else "Dato no disponible."),
        ("ROA",
         _pct(roa) if roa else "—",
         (f"Retorno sobre activos del {roa*100:.1f}%. " +
          ("Muy bueno: la empresa genera alta rentabilidad con todos sus activos." if roa and roa > 0.10 else
           "Razonable: la empresa rentabiliza sus activos de forma aceptable." if roa and roa > 0.04 else
           "Bajo: los activos generan poca rentabilidad. Puede ser intensiva en capital o con problemas de eficiencia." if roa and roa >= 0 else
           "Negativo: activos que consumen valor neto. Empresa en pérdidas."))
         if roa is not None else "Dato no disponible."),
        ("Empleados",
         f"{int(empl):,}" if empl else "—",
         (f"{int(empl):,} empleados. " +
          (f"Ingresos por empleado: {_fmt(_rev_per_emp, '$')} — " +
           ("excepcional productividad (empresas de software suelen superar $1M/empleado)." if _rev_per_emp and _rev_per_emp > 800000 else
            "buena productividad." if _rev_per_emp and _rev_per_emp > 300000 else
            "moderada, habitual en sectores intensivos en mano de obra.") if _rev_per_emp else ""))
         if empl else "Dato no disponible."),
    ]
    story.append(kv_table(fin_rows))

    # ── SECCIÓN 3: POSICIONAMIENTO EN CORTOS ─────────────────────────────────
    short_fl = _safe(info, "shortPercentOfFloat")
    short_rt = _safe(info, "shortRatio")
    shares_sh = _safe(info, "sharesShort")
    if short_fl or short_rt or shares_sh:
        story += section(
            "3. Posicionamiento en Cortos (Short Interest)",
            "El short interest indica cuantos inversores estan apostando a la baja en la accion. "
            "Un short float elevado (>20%) puede provocar un 'short squeeze' si el precio sube: "
            "los cortos se ven forzados a comprar para cerrar posiciones, amplificando la subida."
        )
        short_rows = [
            ("Short Float",
             _pct(short_fl) if short_fl else "—",
             _expl_short_float(short_fl) if short_fl is not None else "Dato no disponible."),
            ("Short Ratio (días)",
             f"{short_rt:.1f} días" if short_rt else "—",
             _expl_short_ratio(short_rt) if short_rt is not None else "Dato no disponible."),
            ("Acciones en corto",
             _fmt(shares_sh) if shares_sh else "—",
             (f"{_fmt(shares_sh)} acciones en corto. " +
              (f"Representa el {short_fl*100:.1f}% del free float. " if short_fl else "") +
              ("Un número tan elevado implica una tesis bajista muy extendida entre inversores institucionales." if shares_sh and shares_sh > 50e6 else
               "Volumen de cortos relevante pero manejable." if shares_sh and shares_sh > 10e6 else
               "Posición bajista contenida en términos absolutos."))
             if shares_sh is not None else "Dato no disponible."),
        ]
        story.append(kv_table(short_rows))

    # ── SECCIÓN 4: ANALISIS TECNICO ───────────────────────────────────────────
    story += section(
        "4. Analisis Tecnico e Indicadores",
        "El analisis tecnico estudia el comportamiento historico del precio y el volumen para "
        "identificar tendencias, soportes, resistencias y momentum. Se usa para determinar momentos "
        "de entrada/salida, no para valorar la empresa en si."
    )

    # Chart
    chart_png = _price_chart_png()
    if chart_png:
        story.append(RLImage(chart_png, width=PW, height=PW * 0.42 * 1.3))
        story.append(Paragraph(
            "Grafico de precio (1 ano) con SMA 50 (dorada) y SMA 200 (roja). "
            "Panel inferior: RSI(14) — linea roja superior = sobrecompra (70), linea verde inferior = sobreventa (30).",
            S_CAP
        ))
        story.append(Spacer(1, 4 * mm))

    # Compute indicators for table
    tech_rows = []
    try:
        if df_hist is not None and not df_hist.empty and "Close" in df_hist.columns:
            close = df_hist["Close"].squeeze()
            sma20  = close.rolling(20).mean()
            sma50  = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            rsi_s  = _calc_rsi(close)
            macd_l, macd_sg, macd_h = _calc_macd(close)
            last_c   = close.iloc[-1]
            last_rsi = rsi_s.dropna().iloc[-1] if not rsi_s.dropna().empty else None
            last_macd = macd_l.dropna().iloc[-1]  if not macd_l.dropna().empty  else None
            last_sig  = macd_sg.dropna().iloc[-1] if not macd_sg.dropna().empty else None
            last_s50  = sma50.dropna().iloc[-1]   if not sma50.dropna().empty   else None
            last_s200 = sma200.dropna().iloc[-1]  if not sma200.dropna().empty  else None

            if last_rsi:
                tech_rows.append(("RSI (14)",
                                  f"{last_rsi:.1f}",
                                  _expl_rsi(last_rsi)))

            if last_macd and last_sig:
                hist_val = last_macd - last_sig
                macd_dir = "Alcista" if last_macd > last_sig else "Bajista"
                if last_macd > last_sig:
                    _m_detail = ("Momentum positivo fuerte — el diferencial está creciendo."
                                 if abs(hist_val) > abs(last_macd)*0.3 else
                                 "Momentum positivo moderado. Vigilar si el histograma sigue creciendo para confirmar el movimiento.")
                    macd_interp = (
                        f"Cruce alcista: MACD ({last_macd:.4f}) por encima de señal ({last_sig:.4f}). "
                        f"Histograma: +{abs(hist_val):.4f}. " + _m_detail
                    )
                else:
                    _m_detail = ("Momentum negativo fuerte — el diferencial bajista está aumentando."
                                 if abs(hist_val) > abs(last_macd)*0.3 else
                                 "Momentum negativo moderado. Puede ser corrección temporal o inicio de tendencia bajista.")
                    macd_interp = (
                        f"Cruce bajista: MACD ({last_macd:.4f}) por debajo de señal ({last_sig:.4f}). "
                        f"Histograma: -{abs(hist_val):.4f}. " + _m_detail
                    )
                tech_rows.append(("MACD (12,26,9)",
                                  macd_dir,
                                  macd_interp))

            if last_s50:
                vs50 = (last_c / last_s50 - 1) * 100
                tech_rows.append(("vs SMA 50",
                                  f"${last_s50:,.2f} ({vs50:+.1f}%)",
                                  _expl_vs_sma(last_c, last_s50, 50)))
            if last_s200:
                vs200 = (last_c / last_s200 - 1) * 100
                tech_rows.append(("vs SMA 200",
                                  f"${last_s200:,.2f} ({vs200:+.1f}%)",
                                  _expl_vs_sma(last_c, last_s200, 200)))

            high52 = _safe(info, "fiftyTwoWeekHigh")
            low52  = _safe(info, "fiftyTwoWeekLow")
            if high52 and low52:
                pct_from_high = (last_c / high52 - 1) * 100
                pct_from_low  = (last_c / low52  - 1) * 100
                tech_rows.append(("Máximo 52 semanas",
                                  f"${high52:,.2f} ({pct_from_high:+.1f}%)",
                                  _expl_52w(last_c, high52, "high")))
                tech_rows.append(("Mínimo 52 semanas",
                                  f"${low52:,.2f} ({pct_from_low:+.1f}%)",
                                  _expl_52w(last_c, low52, "low")))
    except Exception:
        pass

    if tech_rows:
        story.append(kv_table(tech_rows))
    else:
        story.append(Paragraph("Datos tecnicos no disponibles.", S_NOTE))

    target_mean = _safe(info, "targetMeanPrice")
    target_hi   = _safe(info, "targetHighPrice")
    target_lo   = _safe(info, "targetLowPrice")
    rec         = (_safe(info, "recommendationKey") or "").upper().replace("_", " ")
    n_analysts  = _safe(info, "numberOfAnalystOpinions")

    # Safely retrieve tech variables (they live inside a try block above)
    try:
        _lrsi  = last_rsi
        _ls50  = last_s50
        _ls200 = last_s200
        _lc    = last_c
    except NameError:
        _lrsi = _ls50 = _ls200 = _lc = None

    # Scoring helper
    def _avg(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals) / len(vals) if vals else 5.0

    # ── Reliability indicator ─────────────────────────────────────────────────
    _key_vars = {
        "Precio":         price,
        "Cap. bursatil":  mktcap,
        "P/E trailing":   pe_ttm,
        "P/E forward":    pe_fwd,
        "P/E forward live": _safe(info, "forwardPE"),
        "EPS growth":     _safe(info, "earningsGrowth"),
        "Rev growth":     _safe(info, "revenueGrowth"),
        "P/Ventas":       ps,
        "P/Libro":        pb,
        "EV/EBITDA":      ev_ebit,
        "Margen bruto":   gross_m,
        "Margen operativo": op_m,
        "Margen neto":    net_m,
        "ROE":            roe,
        "D/E":            de,
        "Current Ratio":  cr,
        "Caja neta":      _net_cash,
        "FCF":            fcf,
        "Short Float":    short_fl,
        "Precio objetivo": target_mean,
        "RSI":            _lrsi,
        "SMA50":          _ls50,
        "SMA200":         _ls200,
    }
    _available = sum(1 for v in _key_vars.values() if v is not None)
    _total_vars = len(_key_vars)
    _reliability_pct = _available / _total_vars

    if _reliability_pct >= 0.70:
        _conf_label = "ALTA"
        _conf_col   = "#5a8f6e"
        _conf_note  = f"{_available}/{_total_vars} variables disponibles"
    elif _reliability_pct >= 0.45:
        _conf_label = "MEDIA"
        _conf_col   = "#a07820"
        _conf_note  = f"{_available}/{_total_vars} variables disponibles — interpretar con cautela"
    else:
        _conf_label = "BAJA"
        _conf_col   = "#9b4d4d"
        _conf_note  = f"Solo {_available}/{_total_vars} variables disponibles — analisis muy limitado"

    # ── Sector reference medians (live peer-computed, 24h cached) ────────────
    _sector_key = normalize_sector(sector)
    _sec_ref = get_sector_medians(_sector_key)
    _pe_ref      = _sec_ref.get("pe")        or _SECTOR_DEFAULT["pe"]
    _ps_ref      = _sec_ref.get("ps")        or _SECTOR_DEFAULT["ps"]
    _ev_ref      = _sec_ref.get("ev_ebitda") or _SECTOR_DEFAULT["ev_ebitda"]
    _gm_ref      = _sec_ref.get("gross_m")   or _SECTOR_DEFAULT["gross_m"]
    _nm_ref      = _sec_ref.get("net_m")     or _SECTOR_DEFAULT["net_m"]
    _om_ref      = _sec_ref.get("op_m")      or _SECTOR_DEFAULT["op_m"]
    _roe_ref     = _sec_ref.get("roe")       or _SECTOR_DEFAULT["roe"]
    _de_ref      = _sec_ref.get("de")        or _SECTOR_DEFAULT["de"]
    _fwd_pe_ref  = _sec_ref.get("fwd_pe")    or 18.0
    _peers_live  = _sec_ref.get("_live",     False)
    _n_peers     = _sec_ref.get("_n_peers",  0)

    # ── Scoring helper ────────────────────────────────────────────────────────
    def _avg(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals) / len(vals) if vals else 5.0

    def _rel_score(value, reference, higher_is_better=True):
        """Score 1-10 based on ratio to sector reference median."""
        if not value or not reference or reference == 0:
            return None
        ratio = value / reference
        if higher_is_better:
            if ratio >= 2.0:  return 10
            if ratio >= 1.5:  return 9
            if ratio >= 1.2:  return 8
            if ratio >= 0.9:  return 6
            if ratio >= 0.7:  return 5
            if ratio >= 0.5:  return 3
            return 2
        else:  # lower is better (P/E, D/E, EV/EBITDA)
            if ratio <= 0.5:  return 10
            if ratio <= 0.7:  return 9
            if ratio <= 0.85: return 8
            if ratio <= 1.0:  return 6
            if ratio <= 1.2:  return 5
            if ratio <= 1.5:  return 3
            return 2

    # 1. Valuation score — sector-relative ────────────────────────────────────
    _sv = []
    if pe_ttm and pe_ttm > 0:
        _sv.append(_rel_score(pe_ttm, _pe_ref, higher_is_better=False))
    elif pe_ttm and pe_ttm <= 0:
        _sv.append(3)  # losses penalised but not harshly
    if ps:
        _sv.append(_rel_score(ps, _ps_ref, higher_is_better=False))
    if ev_ebit and ev_ebit > 0:
        _sv.append(_rel_score(ev_ebit, _ev_ref, higher_is_better=False))
    # Forward PE vs sector median
    _fwd_pe_val = _safe(info, "forwardPE")
    if _fwd_pe_val and _fwd_pe_val > 0:
        _sv.append(_rel_score(_fwd_pe_val, _fwd_pe_ref, higher_is_better=False))
    # PEG ratio (fwdPE / eps_growth*100): < 1 = very attractive, > 2.5 = expensive
    _eps_growth = _safe(info, "earningsGrowth")
    _rev_growth = _safe(info, "revenueGrowth")
    _peg = (_fwd_pe_val / (_eps_growth * 100)) if (_fwd_pe_val and _eps_growth and _eps_growth > 0.01) else None
    if _peg:
        if _peg < 0.75:   _sv.append(10)
        elif _peg < 1.0:  _sv.append(9)
        elif _peg < 1.5:  _sv.append(7)
        elif _peg < 2.0:  _sv.append(5)
        elif _peg < 2.5:  _sv.append(3)
        else:             _sv.append(1)
    # Upside to consensus price target
    _up_pct = (target_mean / price - 1) * 100 if (target_mean and price and price > 0) else None
    if _up_pct is not None:
        if _up_pct > 40:    _sv.append(9)
        elif _up_pct > 20:  _sv.append(7)
        elif _up_pct > 5:   _sv.append(5)
        elif _up_pct > -10: _sv.append(3)
        else:               _sv.append(1)
    sc_val = _avg([s for s in _sv if s is not None])

    # 1b. Growth score — sector-relative ──────────────────────────────────────
    _sg = []
    if _rev_growth is not None:
        _sg.append(_rel_score(_rev_growth, 0.10, higher_is_better=True))
    if _eps_growth is not None:
        _sg.append(_rel_score(_eps_growth, 0.10, higher_is_better=True))
    sc_growth = _avg(_sg) if _sg else 5.0

    # 2. Profitability score — sector-relative ────────────────────────────────
    _sp = []
    if net_m is not None:
        if net_m < 0:
            _sp.append(max(1, int(2 + net_m * 10)))  # penalise losses proportionally
        else:
            _sp.append(_rel_score(net_m, _nm_ref, higher_is_better=True))
    if op_m is not None:
        if op_m < 0:
            _sp.append(max(1, int(2 + op_m * 8)))
        else:
            _sp.append(_rel_score(op_m, _om_ref, higher_is_better=True))
    if roe is not None:
        if roe < 0:
            _sp.append(2)
        else:
            _sp.append(_rel_score(roe, _roe_ref, higher_is_better=True))
    if gross_m is not None and gross_m > 0:
        _sp.append(_rel_score(gross_m, _gm_ref, higher_is_better=True))
    if _fcf_yield:
        if _fcf_yield > 8:    _sp.append(10)
        elif _fcf_yield > 4:  _sp.append(8)
        elif _fcf_yield > 1:  _sp.append(6)
        elif _fcf_yield >= 0: _sp.append(4)
        else:                 _sp.append(2)
    sc_prof = _avg([s for s in _sp if s is not None])

    # 3. Financial health score ────────────────────────────────────────────────
    _sh = []
    if de is not None:
        _der = de / 100
        _sh.append(_rel_score(_der, _de_ref, higher_is_better=False))
    if cr:
        if cr < 0.8:   _sh.append(2)
        elif cr < 1.2: _sh.append(5)
        elif cr < 2.5: _sh.append(9)
        else:          _sh.append(7)
    if _net_cash is not None:
        _sh.append(9 if _net_cash > 0 else 4)
    sc_health = _avg([s for s in _sh if s is not None])

    # 4. Technical score ──────────────────────────────────────────────────────
    _st = []
    if _lrsi:
        if _lrsi < 20:   _st.append(8)
        elif _lrsi < 30: _st.append(7)
        elif _lrsi < 45: _st.append(5)
        elif _lrsi < 55: _st.append(6)
        elif _lrsi < 70: _st.append(7)
        elif _lrsi < 80: _st.append(4)
        else:            _st.append(2)
    if _ls50 and _lc:
        _st.append(7 if _lc > _ls50 else 4)
    if _ls200 and _lc:
        _st.append(8 if _lc > _ls200 else 3)
    sc_tech = _avg(_st)

    # 5. Sentiment score ──────────────────────────────────────────────────────
    _ss = []
    _rec_up = rec.upper() if rec else ""
    if "STRONG BUY" in _rec_up:  _ss.append(10)
    elif "BUY" in _rec_up:       _ss.append(8)
    elif "HOLD" in _rec_up:      _ss.append(5)
    elif "SELL" in _rec_up:      _ss.append(2)
    if short_fl:
        _sf = short_fl * 100
        if _sf < 3:    _ss.append(6)
        elif _sf < 10: _ss.append(5)
        elif _sf < 20: _ss.append(4)
        elif _sf < 30: _ss.append(3)
        else:          _ss.append(4)
    sc_sent = _avg(_ss)

    # Weighted composite
    sc_total = sc_val*0.22 + sc_growth*0.13 + sc_prof*0.25 + sc_health*0.17 + sc_tech*0.13 + sc_sent*0.10

    # ── Penalise low-confidence analyses ─────────────────────────────────────
    if _reliability_pct < 0.45:
        sc_total = min(sc_total, 5.5)  # cap at MANTENER when data is sparse

    # Map score to recommendation
    if sc_total >= 7.8:
        _wv_rec = "COMPRA FUERTE"
        _wv_col = C_POS
        _wv_bias = "claramente infravalorada respecto a su sector"
        _wv_verb = "Iniciamos COMPRA FUERTE"
    elif sc_total >= 6.3:
        _wv_rec = "COMPRA"
        _wv_col = C_POS
        _wv_bias = "atractiva a precios actuales vs. sector"
        _wv_verb = "Iniciamos COMPRA"
    elif sc_total >= 4.7:
        _wv_rec = "MANTENER"
        _wv_col = colors.HexColor("#a07820")
        _wv_bias = "correctamente valorada en el contexto de su sector"
        _wv_verb = "Recomendamos MANTENER"
    elif sc_total >= 3.2:
        _wv_rec = "VENDER"
        _wv_col = C_NEG
        _wv_bias = "sobrevalorada respecto a comparables del sector"
        _wv_verb = "Rebajamos a VENDER"
    else:
        _wv_rec = "VENTA FUERTE"
        _wv_col = C_NEG
        _wv_bias = "con fundamentales por debajo del sector y valoracion exigente"
        _wv_verb = "Rebajamos a VENTA FUERTE"

    _tgt_str = f"${target_mean:.2f}" if target_mean else "N/D"
    _up_str  = (f"{_up_pct:+.1f}%" if _up_pct is not None else "N/D")
    _cap_str = _fmt(mktcap, "$") if mktcap else "N/D"
    # ── Reasons driving the recommendation (sector-relative) ────────────────
    _reasons_buy  = []
    _reasons_sell = []

    # Valuation reasons vs. sector median
    if pe_ttm and pe_ttm > 0 and _pe_ref:
        _pe_ratio = pe_ttm / _pe_ref
        if _pe_ratio < 0.75:
            _reasons_buy.append(
                f"P/E de {pe_ttm:.1f}x un {(1-_pe_ratio)*100:.0f}% por debajo "
                f"de la mediana del sector {sector} ({_pe_ref}x)"
            )
        elif _pe_ratio > 1.4:
            _reasons_sell.append(
                f"P/E de {pe_ttm:.1f}x un {(_pe_ratio-1)*100:.0f}% por encima "
                f"de la mediana sectorial ({_pe_ref}x) — prima de valoracion elevada"
            )
    if ev_ebit and ev_ebit > 0 and _ev_ref:
        _ev_ratio = ev_ebit / _ev_ref
        if _ev_ratio < 0.75:
            _reasons_buy.append(
                f"EV/EBITDA de {ev_ebit:.1f}x vs. mediana sectorial de {_ev_ref}x "
                f"— descuento del {(1-_ev_ratio)*100:.0f}%"
            )
        elif _ev_ratio > 1.4:
            _reasons_sell.append(
                f"EV/EBITDA de {ev_ebit:.1f}x un {(_ev_ratio-1)*100:.0f}% sobre "
                f"el sector ({_ev_ref}x)"
            )
    if _up_pct and _up_pct > 20:
        _reasons_buy.append(f"potencial del {_up_str} hasta el objetivo de consenso ({_tgt_str})")
    if _up_pct and _up_pct < -5:
        _reasons_sell.append(f"cotizacion supera el objetivo de consenso ({_tgt_str}) en {abs(_up_pct):.0f}%")

    # Profitability vs. sector
    if net_m is not None and _nm_ref:
        _nm_ratio = net_m / _nm_ref if _nm_ref else 0
        if net_m < 0:
            _reasons_sell.append(
                f"margen neto negativo ({net_m*100:.1f}%) vs. mediana sectorial del {_nm_ref*100:.0f}%"
            )
        elif _nm_ratio > 1.3:
            _reasons_buy.append(
                f"margen neto del {net_m*100:.1f}% — un {(_nm_ratio-1)*100:.0f}% "
                f"superior a la mediana del sector ({_nm_ref*100:.0f}%)"
            )
        elif _nm_ratio < 0.6:
            _reasons_sell.append(
                f"margen neto del {net_m*100:.1f}% por debajo de la mediana sectorial "
                f"({_nm_ref*100:.0f}%)"
            )
    if roe is not None and _roe_ref:
        _roe_ratio = roe / _roe_ref if _roe_ref else 0
        if roe > 0 and _roe_ratio > 1.5:
            _reasons_buy.append(
                f"ROE del {roe*100:.1f}% — {(_roe_ratio-1)*100:.0f}% por encima "
                f"de la mediana del sector ({_roe_ref*100:.0f}%) — ventaja competitiva"
            )
        elif roe < 0:
            _reasons_sell.append(f"ROE negativo ({roe*100:.1f}%): destruccion de valor para el accionista")

    # Balance sheet reasons
    if _net_cash and _net_cash > 0:
        _cash_pct_r = (_net_cash / mktcap * 100) if mktcap else 0
        _reasons_buy.append(
            f"caja neta de {_fmt(_net_cash, '$')}"
            + (f" ({_cash_pct_r:.0f}% de la capitalizacion)" if _cash_pct_r > 8 else "")
            + " actua como suelo de valoracion"
        )
    if de is not None and _de_ref:
        _de_ratio = (de / 100) / _de_ref if _de_ref else 0
        if _de_ratio > 2.0:
            _reasons_sell.append(
                f"D/E de {de/100:.1f}x — el doble de la mediana sectorial ({_de_ref}x)"
            )
    if _fcf_yield and _fcf_yield > 5:
        _reasons_buy.append(f"FCF yield del {_fcf_yield:.1f}% — generacion de caja superior al mercado")

    # Technical reasons
    if _lrsi and _lrsi < 35:
        _reasons_buy.append(f"RSI de {_lrsi:.0f} en sobreventa — punto de entrada tecnico favorable")
    if _lrsi and _lrsi > 72:
        _reasons_sell.append(f"RSI de {_lrsi:.0f} en sobrecompra — momentum sobreextendido")
    if _lc and _ls200 and _lc > _ls200:
        _reasons_buy.append(f"precio sobre SMA200 (${_ls200:,.2f}) — tendencia estructural alcista")
    if _lc and _ls200 and _lc < _ls200:
        _reasons_sell.append(f"precio bajo SMA200 (${_ls200:,.2f}) — tendencia de fondo comprometida")
    if gross_m and _gm_ref and gross_m > _gm_ref * 1.2:
        _reasons_buy.append(
            f"margen bruto del {gross_m*100:.0f}% vs. {_gm_ref*100:.0f}% sectorial — pricing power"
        )
    _active_reasons = _reasons_buy if _wv_rec in ("COMPRA FUERTE", "COMPRA") else _reasons_sell
    if _active_reasons:
        _wv_expl = (
            f"WealthView recomienda <b>{_wv_rec}</b> (score {sc_total:.1f}/10) "
            f"basado en: {'; '.join(_active_reasons[:3])}."
        )
    else:
        _wv_expl = (
            f"WealthView recomienda <b>{_wv_rec}</b> (score {sc_total:.1f}/10). "
            f"Valoracion: {sc_val:.1f} · Rentabilidad: {sc_prof:.1f} · "
            f"Salud: {sc_health:.1f} · Tecnico: {sc_tech:.1f}."
        )

    # Persist scoring for AI Analyst tab
    st.session_state[f"_wv_score_{ticker}"] = sc_total
    st.session_state[f"_wv_rec_{ticker}"]   = _wv_rec
    st.session_state[f"_wv_rb_{ticker}"]    = list(_reasons_buy)
    st.session_state[f"_wv_rs_{ticker}"]    = list(_reasons_sell)

    # ── SECCIÓN 5: CONSENSO ANALISTAS ─────────────────────────────────────────
    if target_mean:
        story += section(
            "5. Consenso de Analistas",
            "El consenso recoge los precios objetivo y recomendaciones de los analistas institucionales "
            "que cubren el valor. No son garantia de rentabilidad, pero son una referencia util del "
            "precio justo segun el mercado profesional."
        )
        upside = (target_mean / price - 1) * 100 if price else 0.0
        _spread = (target_hi - target_lo) if (target_hi and target_lo) else None
        if _spread and target_mean:
            _disp_str = f"Dispersion de ${_spread:.2f} ({_spread/target_mean*100:.0f}% del objetivo medio). "
            if _spread/target_mean > 0.4:
                _disp_str += "Alta dispersion: analistas muy divididos, mayor incertidumbre."
            elif _spread/target_mean > 0.2:
                _disp_str += "Dispersion moderada: escenarios distintos pero cierto consenso."
            else:
                _disp_str += "Baja dispersion: los analistas tienen vision similar del valor justo."
        else:
            _disp_str = ""
        _range_expl = (
            f"Rango de consenso: ${target_lo:.2f} — ${target_hi:.2f}. {_disp_str}"
            if (target_hi and target_lo) else "Rango no disponible."
        )
        analyst_rows = [
            ("Precio objetivo medio",
             f"${target_mean:.2f}",
             _expl_target(target_mean, price, n_analysts)),
            ("Objetivo alto / bajo",
             f"${target_hi:.2f} / ${target_lo:.2f}" if (target_hi and target_lo) else "—",
             _range_expl),
            ("Recomendacion WealthView",
             _wv_rec,
             _wv_expl),
        ]
        story.append(kv_table(analyst_rows, col_widths=[PW*0.27, PW*0.18, PW*0.55]))

    # ── ANÁLISIS WEALTHVIEW — RECOMENDACIÓN DE INVERSIÓN ───────────────────────
    # Scores, _wv_rec, _reasons_buy/_sell, _wv_expl already computed above.

    # ══ NARRATIVAS DE CALIDAD INVESTMENT BANKING ══════════════════════════════

    # -- News sentiment aggregation ----------------------------------------
    _news_pos   = sum(1 for n in _news_items if n["sentiment"] ==  1)
    _news_neg   = sum(1 for n in _news_items if n["sentiment"] == -1)
    _news_total = len(_news_items)
    _news_bias  = (
        "positivo" if _news_pos > _news_neg * 1.3
        else "negativo" if _news_neg > _news_pos * 1.3
        else "mixto"
    )
    if _news_total >= 3:
        if _news_bias == "positivo":   sc_sent = min(10, sc_sent + 0.5)
        elif _news_bias == "negativo": sc_sent = max(0,  sc_sent - 0.5)
        sc_total = sc_val*0.22 + sc_growth*0.13 + sc_prof*0.25 + sc_health*0.17 + sc_tech*0.13 + sc_sent*0.10

    # -- Dimension scores for narrative ------------------------------------
    _dim_scores = {
        "valoracion":        sc_val,
        "rentabilidad":      sc_prof,
        "salud financiera":  sc_health,
        "analisis tecnico":  sc_tech,
        "sentimiento":       sc_sent,
    }
    _top_dims = sorted(_dim_scores, key=_dim_scores.get, reverse=True)[:2]
    _bot_dims = sorted(_dim_scores, key=_dim_scores.get)[:2]

    # -- Augment reasons with news signal ----------------------------------
    if _news_total >= 3:
        if _news_bias == "positivo":
            _reasons_buy.append(f"flujo de noticias positivo ({_news_pos}/{_news_total} favorables)")
        elif _news_bias == "negativo":
            _reasons_sell.append(f"flujo de noticias negativo ({_news_neg}/{_news_total} desfavorables)")

    # -- Executive Summary -------------------------------------------------
    _mktcap_tier = (
        "mega-cap" if mktcap and mktcap > 200e9 else
        "large-cap" if mktcap and mktcap > 10e9 else
        "mid-cap" if mktcap and mktcap > 2e9 else
        "small-cap" if mktcap and mktcap > 300e6 else
        "micro-cap" if mktcap else "empresa"
    )
    _exec_parts = [
        f"{company_name} ({ticker.upper()}) es una compañía {_mktcap_tier} del sector {sector} "
        f"({industry}), con una capitalización bursátil de {_cap_str}."
    ]

    # Core thesis: WHY — based on the metrics that drive the score
    if sc_total >= 7.8:
        _top_r = _reasons_buy[:3]
        _why = (
            f"Iniciamos cobertura con COMPRA FUERTE (score {sc_total:.1f}/10). "
            f"La tesis se sustenta en: "
            + (", ".join(_top_r) + "." if _top_r else
               f"la confluencia de {_top_dims[0]} ({_dim_scores[_top_dims[0]]:.1f}/10) "
               f"y {_top_dims[1]} ({_dim_scores[_top_dims[1]]:.1f}/10) significativamente por encima del umbral de compra.")
        )
    elif sc_total >= 6.3:
        _top_r = _reasons_buy[:2]
        _why = (
            f"Recomendamos COMPRA (score {sc_total:.1f}/10). "
            f"Los principales argumentos son: "
            + (", ".join(_top_r) + "." if _top_r else
               f"una puntuación destacada en {_top_dims[0]} ({_dim_scores[_top_dims[0]]:.1f}/10) "
               f"que genera un perfil riesgo/retorno favorable en el horizonte de 12 meses.")
        )
    elif sc_total >= 4.7:
        _why = (
            f"Mantenemos una visión NEUTRAL (score {sc_total:.1f}/10). "
            f"El valor no ofrece suficiente margen de seguridad para iniciar posición: "
            f"{'la ' + _bot_dims[0] + ' (' + str(round(_dim_scores[_bot_dims[0]], 1)) + '/10) limita el potencial alcista' if _bot_dims else 'el balance entre fortalezas y debilidades no es favorable'}. "
            f"Aguardamos catalizadores o un punto de entrada más atractivo."
        )
    elif sc_total >= 3.2:
        _top_r = _reasons_sell[:2]
        _why = (
            f"Recomendamos VENDER (score {sc_total:.1f}/10). "
            f"Las señales de alerta son: "
            + (", ".join(_top_r) + "." if _top_r else
               f"deterioro en {_bot_dims[0]} ({_dim_scores[_bot_dims[0]]:.1f}/10) "
               f"que supera los méritos del valor a cotizaciones actuales.")
        )
    else:
        _top_r = _reasons_sell[:3]
        _why = (
            f"Emitimos VENTA FUERTE (score {sc_total:.1f}/10). "
            f"La convergencia de señales negativas — "
            + (", ".join(_top_r) + " — nos lleva a desaconsejar cualquier exposición." if _top_r else
               f"deterioro estructural en {_bot_dims[0]} y {_bot_dims[1]} — desaconseja mantener exposición.")
        )

    _exec_parts.append(_why)

    # News context
    if _news_total >= 3:
        if _news_bias == "positivo":
            _exec_parts.append(
                f"El flujo de noticias en tiempo real es favorable ({_news_pos} de {_news_total} noticias recientes con sesgo positivo), "
                f"lo que añade un catalizador de corto plazo a la tesis de inversión."
            )
        elif _news_bias == "negativo":
            _exec_parts.append(
                f"El flujo de noticias reciente es un factor de cautela ({_news_neg} de {_news_total} noticias con sesgo negativo), "
                f"lo que podría presionar la cotización en el corto plazo independientemente de los fundamentales."
            )

    _exec = " ".join(_exec_parts)

    # -- Valuation Analysis -------------------------------------------------
    _vp = []

    # Contexto de valoración absoluta vs. sector
    _val_intro = (
        f"Desde una perspectiva de valoración, {company_name} cotiza "
    )
    _val_metrics = []
    if pe_ttm and pe_ttm > 0:
        _ref_pe = "por debajo" if pe_ttm < 20 else "en línea con" if pe_ttm < 28 else "por encima"
        _val_metrics.append(f"P/E trailing de {pe_ttm:.1f}x ({_ref_pe} de la media de mercado de ~20x)")
    if pe_fwd and pe_fwd > 0:
        _fwd_disc = pe_ttm - pe_fwd if (pe_ttm and pe_ttm > 0) else None
        if _fwd_disc and _fwd_disc > 3:
            _val_metrics.append(f"P/E forward de {pe_fwd:.1f}x, implicando expansión de márgenes "
                                f"de {_fwd_disc:.1f} puntos de P/E en los próximos 12 meses")
        elif pe_fwd:
            _val_metrics.append(f"P/E forward de {pe_fwd:.1f}x")
    if ps:
        _val_metrics.append(f"P/Ventas de {ps:.2f}x")
    if ev_ebit and ev_ebit > 0:
        _val_metrics.append(f"EV/EBITDA de {ev_ebit:.1f}x")

    if _val_metrics:
        _vp.append(_val_intro + ", ".join(_val_metrics) + ".")

    # Interpretación del múltiplo de valoración dominante
    if pe_ttm and pe_ttm > 0:
        if pe_ttm < 12:
            _vp.append(
                f"Un P/E de {pe_ttm:.1f}x es notablemente bajo y sugiere que el mercado "
                f"descuenta un escenario de declive de beneficios o algún riesgo idiosincrático "
                f"no reflejado en los estados financieros. Si dicho riesgo no se materializa, "
                f"el potencial de re-rating es considerable."
            )
        elif pe_ttm < 18:
            _vp.append(
                f"El P/E de {pe_ttm:.1f}x es atractivo en términos históricos. Empresas de "
                f"calidad comparable han cotizado a múltiplos de 18-25x durante periodos de "
                f"crecimiento sostenido; la diferencia actual representa upside de múltiplo "
                f"independiente del crecimiento orgánico de beneficios."
            )
        elif pe_ttm < 30:
            _vp.append(
                f"El P/E de {pe_ttm:.1f}x es consistente con una empresa de crecimiento "
                f"moderado-alto. El mercado descuenta mejora de resultados; cualquier "
                f"decepción en guías podría desencadenar una contracción de múltiplos "
                f"de 3-5 puntos, lo que representa un riesgo de caída del 10-20%."
            )
        else:
            _vp.append(
                f"El P/E de {pe_ttm:.1f}x solo se justifica con tasas de crecimiento de "
                f"beneficios superiores al 20-25% CAGR durante los próximos 3-5 años. "
                f"A estos múltiplos, la acción no tiene margen para decepcionar; "
                f"recomendamos extrema selectividad."
            )
    elif pe_ttm and pe_ttm <= 0:
        _vp.append(
            f"La empresa registra pérdidas en el ejercicio más reciente. En ausencia de "
            f"P/E significativo, la valoración debe anclarse en el P/S ({ps:.2f}x si ps else 'N/D') "
            f"y la trayectoria hacia el breakeven de EBITDA. La clave es si el modelo de "
            f"negocio tiene poder estructural para alcanzar márgenes positivos a escala."
        )

    # EV/EBITDA contextualización
    if ev_ebit and ev_ebit > 0:
        if ev_ebit < 10:
            _vp.append(
                f"El EV/EBITDA de {ev_ebit:.1f}x sitúa a la empresa en territorio de "
                f"potencial objetivo de adquisición: por debajo de 10x, los activos "
                f"generadores de caja resultan atractivos para operaciones de M&A, "
                f"lo que añade una opción de valor adicional para el accionista."
            )
        elif ev_ebit > 30:
            _vp.append(
                f"El EV/EBITDA de {ev_ebit:.1f}x implica que el mercado está pagando "
                f"una prima significativa por crecimiento futuro. Este nivel solo se "
                f"justifica si la empresa puede mantener tasas de crecimiento de EBITDA "
                f"superiores al 30% durante los próximos años."
            )

    val_para = " ".join(_vp) if _vp else (
        "La información pública disponible no permite construir un marco de valoración "
        "robusto. Recomendamos consultar los estados financieros auditados más recientes "
        "antes de tomar decisiones de inversión."
    )

    # -- Fundamentals Analysis ----------------------------------------------
    _fp = []

    # Calidad del P&L
    _margin_quality = []
    if gross_m:
        if gross_m > 0.60:
            _margin_quality.append(
                f"El margen bruto del {gross_m*100:.1f}% es excepcional y revela "
                f"un modelo de negocio con pricing power estructural — una característica "
                f"típica de empresas con fosos competitivos (moats) sólidos, ya sea "
                f"por propiedad intelectual, efectos de red o costes de cambio elevados."
            )
        elif gross_m > 0.40:
            _margin_quality.append(
                f"El margen bruto del {gross_m*100:.1f}% refleja un negocio con "
                f"diferenciación moderada. La evolución de este indicador en los "
                f"próximos trimestres será clave para validar la capacidad de "
                f"trasladar la inflación de costes al cliente final."
            )
        elif gross_m > 0.20:
            _margin_quality.append(
                f"El margen bruto del {gross_m*100:.1f}% es característico de sectores "
                f"con alta competencia o modelos asset-heavy. La eficiencia operativa "
                f"y el control de SG&A serán determinantes para la rentabilidad final."
            )
        else:
            _margin_quality.append(
                f"El margen bruto del {gross_m*100:.1f}% es reducido, lo que limita "
                f"estructuralmente la capacidad de generar beneficios netos. "
                f"El modelo de negocio depende críticamente del volumen."
            )
    if _margin_quality:
        _fp.extend(_margin_quality)

    # Rentabilidad neta y calidad de beneficios
    if net_m:
        if net_m > 0.20:
            _fp.append(
                f"La rentabilidad neta del {net_m*100:.1f}% es de primer nivel: "
                f"pocos negocios en el mundo consiguen convertir más de 1 de cada 5 "
                f"dólares de ingresos en beneficio neto de forma sostenida. "
                f"Este perfil apunta a una franquicia de alta calidad con barreras de "
                f"entrada sólidas."
            )
        elif net_m > 0.08:
            _fp.append(
                f"Los márgenes netos del {net_m*100:.1f}% son saludables y consistentes "
                f"con un negocio bien gestionado. La prioridad debería ser proteger "
                f"este nivel frente a presiones competitivas y de costes."
            )
        elif net_m > 0:
            _fp.append(
                f"Los márgenes netos del {net_m*100:.1f}% son ajustados y dejan "
                f"poco colchón ante shocks de demanda o de costes. La expansión "
                f"de margen es el principal catalizador de re-rating que monitorizamos."
            )
        elif net_m < 0:
            _fp.append(
                f"La compañía registra pérdidas netas del {abs(net_m)*100:.1f}% sobre "
                f"ingresos. En este contexto, la validez de la tesis de inversión depende "
                f"enteramente de la velocidad de ejecución hacia la rentabilidad y de la "
                f"solidez del balance para financiar el camino hasta el breakeven."
            )

    # Balance y generación de caja
    if _net_cash is not None:
        if _net_cash > 0:
            _cash_pct = (_net_cash / mktcap * 100) if mktcap else None
            _cash_msg = (
                f" — equivalente al {_cash_pct:.0f}% de la capitalización bursátil, "
                f"lo que actúa como suelo implícito de valoración"
                if _cash_pct and _cash_pct > 10 else ""
            )
            _fp.append(
                f"El balance muestra una posición de caja neta de {_fmt(_net_cash, '$')}"
                f"{_cash_msg}. Esta solidez financiera otorga a la dirección libertad "
                f"total para invertir en crecimiento, recomprar acciones o distribuir "
                f"dividendos sin depender de los mercados de capital."
            )
        else:
            _net_d_ratio = abs(_net_cash) / (mktcap or 1)
            if _net_d_ratio > 0.5:
                _fp.append(
                    f"La deuda neta de {_fmt(abs(_net_cash), '$')} representa el "
                    f"{_net_d_ratio*100:.0f}% de la capitalización: un nivel de "
                    f"apalancamiento que merece atención, especialmente en un entorno "
                    f"de tipos de interés elevados. La generación de FCF y los "
                    f"vencimientos de deuda son variables críticas a vigilar."
                )
            else:
                _fp.append(
                    f"La empresa opera con deuda neta de {_fmt(abs(_net_cash), '$')}, "
                    f"manejable en relación a su tamaño y generación de caja."
                )

    # ROE y ROIC
    if roe:
        if roe > 0.25:
            _fp.append(
                f"El ROE del {roe*100:.1f}% es excepcional y sugiere que la empresa "
                f"cuenta con una ventaja competitiva duradera (competitive moat). "
                f"Solo los negocios con retornos sobre el capital tan elevados pueden "
                f"crear valor para el accionista de forma consistente a lo largo del ciclo."
            )
        elif roe > 0.12:
            _fp.append(
                f"El ROE del {roe*100:.1f}% está por encima del coste de capital "
                f"estimado (~10%), lo que confirma que la empresa crea valor para "
                f"el accionista a tasas razonables."
            )
        elif roe < 0:
            _fp.append(
                f"El ROE negativo del {roe*100:.1f}% refleja el deterioro actual "
                f"de la rentabilidad; la recuperación de este indicador será un "
                f"catalizador clave para cualquier re-rating del múltiplo."
            )

    if not _fp:
        _fp.append(
            "Los datos fundamentales disponibles son insuficientes para construir "
            "un análisis de calidad. Recomendamos acceder directamente a los últimos "
            "informes anuales (10-K / Annual Report) antes de tomar decisiones."
        )
    fund_para = " ".join(_fp)

    # -- Technical Analysis -------------------------------------------------
    _tp = []

    # RSI con contexto
    if _lrsi:
        if _lrsi < 25:
            _tp.append(
                f"El RSI(14) de {_lrsi:.0f} se adentra en territorio de sobreventa "
                f"extrema — estadísticamente, lecturas por debajo de 25 han precedido "
                f"rebotes técnicos significativos en más del 70% de los casos históricos. "
                f"Desde una perspectiva de timing, el momento presente ofrece un ratio "
                f"riesgo/retorno favorable para iniciar o incrementar posición."
            )
        elif _lrsi < 35:
            _tp.append(
                f"El RSI(14) de {_lrsi:.0f} refleja presión vendedora y posible "
                f"sobreventa a corto plazo. Si los fundamentales sostienen la tesis, "
                f"esta debilidad técnica representa una oportunidad de entrada más "
                f"atractiva que la media."
            )
        elif _lrsi < 55:
            _tp.append(
                f"El RSI(14) de {_lrsi:.0f} se mueve en zona neutral, sin presión "
                f"extrema en ninguna dirección. El precio busca dirección y es "
                f"especialmente sensible a noticias o catalizadores fundamentales."
            )
        elif _lrsi < 70:
            _tp.append(
                f"El RSI(14) de {_lrsi:.0f} denota momentum positivo y tendencia "
                f"alcista activa. El precio sigue siendo técnicamente comprable aunque "
                f"el margen de seguridad es menor que en meses anteriores."
            )
        else:
            _tp.append(
                f"El RSI(14) de {_lrsi:.0f} señala sobrecompra técnica. Los inversores "
                f"que no tienen posición deberían esperar una consolidación antes de "
                f"entrar; los que ya tienen posición pueden valorar reducir riesgo táctico "
                f"y recomprar a niveles más atractivos."
            )

    # Medias móviles y tendencia estructural
    if _ls50 and _ls200 and _lc:
        _pct_vs_200 = (_lc / _ls200 - 1) * 100
        _pct_vs_50  = (_lc / _ls50  - 1) * 100

        if _lc > _ls50 and _lc > _ls200:
            _tp.append(
                f"La estructura técnica es constructiva: el precio (${_lc:,.2f}) cotiza "
                f"un {abs(_pct_vs_50):.1f}% por encima de la SMA50 y un {abs(_pct_vs_200):.1f}% "
                f"sobre la SMA200 (${_ls200:,.2f}). Ambas medias actúan como soporte dinámico "
                f"y la tendencia de fondo sigue siendo alcista. Esta configuración favorece "
                f"estrategias de compra en debilidades."
            )
        elif _lc < _ls50 and _lc < _ls200:
            _tp.append(
                f"El precio (${_lc:,.2f}) opera un {abs(_pct_vs_50):.1f}% por debajo de la "
                f"SMA50 y un {abs(_pct_vs_200):.1f}% bajo la SMA200 (${_ls200:,.2f}) — patrón "
                f"de debilidad estructural. La recuperación de la SMA50 sería la primera "
                f"señal técnica de estabilización; hasta entonces, el sesgo es bajista."
            )
        elif _lc > _ls200:
            _tp.append(
                f"El precio (${_lc:,.2f}) mantiene la tendencia de largo plazo intacta "
                f"(SMA200: ${_ls200:,.2f}) aunque muestra debilidad táctica al situarse "
                f"bajo la SMA50 (${_ls50:,.2f}). La SMA200 representa el nivel clave "
                f"de defensa: su pérdida cambiaría el sesgo estructural a bajista."
            )
        else:
            _tp.append(
                f"El precio (${_lc:,.2f}) ha recuperado la SMA50 (${_ls50:,.2f}) pero "
                f"sigue bajo la SMA200 (${_ls200:,.2f}). Esta configuración de recuperación "
                f"incipiente requiere confirmación mediante volumen y sostenimiento sobre "
                f"la media de 200 sesiones para validar el giro de tendencia."
            )
    elif not (_lrsi or _ls50 or _ls200):
        _tp.append(
            "No se dispone de datos técnicos suficientes para el periodo analizado. "
            "Recomendamos consultar el gráfico directamente en plataformas como "
            "Bloomberg, Reuters o TradingView para valorar el timing de entrada."
        )

    tech_para = " ".join(_tp)

    # Bull/Bear cases
    _bulls = []
    _bears = []
    if _up_pct and _up_pct > 20:
        _bulls.append(f"Re-rating alcista: consenso ve {_up_str} de potencial hasta {_tgt_str}")
    if _net_cash and _net_cash > 0:
        _bulls.append(f"Caja neta positiva ({_fmt(_net_cash, '$')}): sin riesgo de dilución")
    if short_fl and short_fl > 0.20:
        _bulls.append(f"Short float del {short_fl*100:.0f}%: potencial de short squeeze ante catalizador")
    if net_m and net_m > 0.12:
        _bulls.append(f"Márgenes sólidos ({net_m*100:.0f}% neto) con expansión posible vía escala")
    if _lrsi and _lrsi < 35:
        _bulls.append(f"Sobreventa técnica (RSI {_lrsi:.0f}): precio puede descontar el peor escenario")
    if roe and roe > 0.20:
        _bulls.append(f"ROE del {roe*100:.0f}%: ventaja competitiva defensible")
    if gross_m and gross_m > 0.60:
        _bulls.append(f"Margen bruto del {gross_m*100:.0f}%: alto poder de fijación de precios")

    if net_m and net_m < 0:
        _bears.append(f"Empresa en pérdidas (margen neto {net_m*100:.1f}%): incertidumbre sobre el breakeven")
    if de and de / 100 > 1.5:
        _bears.append(f"D/E de {de/100:.1f}x: carga financiera elevada en entorno de tipos altos")
    if pe_ttm and pe_ttm > 50:
        _bears.append(f"P/E de {pe_ttm:.0f}x: múltiplo exigente con poco margen para decepciones")
    if _lc and _ls200 and _lc < _ls200:
        _bears.append(f"Precio bajo SMA200 (${_ls200:,.2f}): tendencia de largo plazo bajista")
    if cr and cr < 1:
        _bears.append(f"Current Ratio de {cr:.2f}: liquidez ajustada, posible necesidad de financiación")
    if _up_pct is not None and _up_pct < -10:
        _bears.append(f"Precio supera el objetivo de consenso en {abs(_up_pct):.0f}%: riesgo de revisión a la baja")
    if short_fl and short_fl < 0.05 and (_lrsi and _lrsi > 65):
        _bears.append("Momentum overbought sin posiciones cortas que amortigüen caídas")

    _bulls = _bulls[:4]
    _bears = _bears[:4]

    # ── BUILD PDF ELEMENTS ────────────────────────────────────────────────────
    story += section(
        "Análisis WealthView — Recomendación de Inversión",
        f"Análisis propio con scoring relativo al sector ({sector}): valoración (25%), "
        "rentabilidad (20%), salud financiera (20%), técnico (20%) y sentimiento (15%). "
        "No constituye asesoramiento de inversión regulado."
    )

    # ── Reliability badge ─────────────────────────────────────────────────────
    _missing_vars = [k for k, v in _key_vars.items() if v is None]
    _conf_detail  = (
        f"Variables no disponibles: {', '.join(_missing_vars[:6])}"
        + ("..." if len(_missing_vars) > 6 else "")
        if _missing_vars else "Todos los indicadores clave disponibles."
    )
    _conf_tbl_data = [[
        Paragraph(
            f'<font name="Helvetica-Bold" size="9" color="{_conf_col}">● Fiabilidad del análisis: {_conf_label}</font>'
            f'<br/><font size="7" color="#7a8799">{_conf_note}</font>',
            sty("CF1", leading=13, spaceAfter=0)
        ),
        Paragraph(
            f'<font size="7" color="#7a8799">{_conf_detail}</font>',
            sty("CF2", fontSize=7, leading=11, spaceAfter=0)
        ),
    ]]
    _conf_tbl = Table(_conf_tbl_data, colWidths=[PW*0.38, PW*0.62])
    _conf_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SURFACE),
        ("LINEBEFORE",    (0, 0), (0, -1), 2, colors.HexColor(_conf_col)),
        ("LINEABOVE",     (0, 0), (-1, 0), 0.3, C_BORDER),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(_conf_tbl)
    story.append(Spacer(1, 4 * mm))

    # Sector reference note
    story.append(Paragraph(
        f'<font size="7" color="#7a8799">Scoring relativo al sector <b>{sector}</b> — '
        f'referencias: P/E {_pe_ref}x · P/S {_ps_ref}x · EV/EBITDA {_ev_ref}x · '
        f'Margen neto {_nm_ref*100:.0f}% · ROE {_roe_ref*100:.0f}%</font>',
        sty("SECREF", fontSize=7, leading=10, spaceAfter=0,
            textColor=colors.HexColor("#7a8799"))
    ))
    story.append(Spacer(1, 4 * mm))

    # Recommendation box
    _rec_color_hex = "#5a8f6e" if _wv_rec in ("COMPRA FUERTE", "COMPRA") else ("#a07820" if _wv_rec == "MANTENER" else "#9b4d4d")
    _box_data = [[
        Paragraph(
            f'<font name="Helvetica-Bold" size="14" color="#e8e0d5">{_wv_rec}</font>'
            f'<br/><font name="Helvetica" size="9" color="#7a8799">WealthView Rating • {ticker.upper()}</font>',
            sty("RB1", leading=20, spaceAfter=0, alignment=1)
        ),
        Paragraph(
            f'<font name="Helvetica-Bold" size="12" color="#c9a84c">{_tgt_str}</font>'
            f'<br/><font name="Helvetica" size="8" color="#7a8799">Precio Objetivo Consenso</font>',
            sty("RB2", leading=18, spaceAfter=0, alignment=1)
        ),
        Paragraph(
            f'<font name="Helvetica-Bold" size="12" color="{_rec_color_hex}">{_up_str}</font>'
            f'<br/><font name="Helvetica" size="8" color="#7a8799">Potencial</font>',
            sty("RB3", leading=18, spaceAfter=0, alignment=1)
        ),
        Paragraph(
            f'<font name="Helvetica-Bold" size="12" color="#c9a84c">{sc_total:.1f}/10</font>'
            f'<br/><font name="Helvetica" size="8" color="#7a8799">Score Compuesto</font>',
            sty("RB4", leading=18, spaceAfter=0, alignment=1)
        ),
    ]]
    _box_tbl = Table(_box_data, colWidths=[PW*0.30, PW*0.23, PW*0.22, PW*0.25])
    _box_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("LINEABOVE",     (0, 0), (-1, 0), 2.5, _wv_col),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, C_BORDER),
        ("LINEBEFORE",    (0, 0), (0, -1), 2.5, _wv_col),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEAFTER",     (0, 0), (2, -1), 0.3, C_BORDER),
    ]))
    story.append(_box_tbl)
    story.append(Spacer(1, 3 * mm))

    # Dimension scores bar
    _dim_data = [[
        Paragraph(f'<font size="7" color="#7a8799">Valoración</font><br/>'
                  f'<font name="Helvetica-Bold" size="9" color="#c9a84c">{sc_val:.1f}</font>',
                  sty("DS1", alignment=1, spaceAfter=0, leading=13)),
        Paragraph(f'<font size="7" color="#7a8799">Rentabilidad</font><br/>'
                  f'<font name="Helvetica-Bold" size="9" color="#c9a84c">{sc_prof:.1f}</font>',
                  sty("DS2", alignment=1, spaceAfter=0, leading=13)),
        Paragraph(f'<font size="7" color="#7a8799">Salud Financiera</font><br/>'
                  f'<font name="Helvetica-Bold" size="9" color="#c9a84c">{sc_health:.1f}</font>',
                  sty("DS3", alignment=1, spaceAfter=0, leading=13)),
        Paragraph(f'<font size="7" color="#7a8799">Técnico</font><br/>'
                  f'<font name="Helvetica-Bold" size="9" color="#c9a84c">{sc_tech:.1f}</font>',
                  sty("DS4", alignment=1, spaceAfter=0, leading=13)),
        Paragraph(f'<font size="7" color="#7a8799">Sentimiento</font><br/>'
                  f'<font name="Helvetica-Bold" size="9" color="#c9a84c">{sc_sent:.1f}</font>',
                  sty("DS5", alignment=1, spaceAfter=0, leading=13)),
    ]]
    _dim_tbl = Table(_dim_data, colWidths=[PW/5]*5)
    _dim_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SURFACE),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(_dim_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── Noticias recientes ────────────────────────────────────────────────────
    if _news_items:
        from datetime import datetime as _dt
        story.append(Paragraph("<b>Noticias recientes</b> <font size='7' color='#7a8799'>(tiempo real)</font>", S_H3))
        _n_shown = 0
        for _ni in _news_items[:6]:
            if not _ni["title"]: continue
            _icon = "▲" if _ni["sentiment"] == 1 else ("▼" if _ni["sentiment"] == -1 else "●")
            _icon_col = "#5a8f6e" if _ni["sentiment"] == 1 else ("#9b4d4d" if _ni["sentiment"] == -1 else "#7a8799")
            try:
                _ts_str = _dt.utcfromtimestamp(_ni["ts"]).strftime("%-d %b %Y") if _ni["ts"] else ""
            except Exception:
                _ts_str = ""
            _pub = f" — {_ni['publisher']}" if _ni["publisher"] else ""
            _date_pub = f"<font size='7' color='#7a8799'>{_ts_str}{_pub}</font>" if (_ts_str or _pub) else ""
            story.append(Paragraph(
                f'<font color="{_icon_col}" size="9"><b>{_icon}</b></font> '
                f'<font size="8" color="#b8c5d0">{_ni["title"]}</font> {_date_pub}',
                sty(f"NI{_n_shown}", fontSize=8, leading=12, spaceBefore=2, spaceAfter=1,
                    leftIndent=4)
            ))
            _n_shown += 1
        # News sentiment summary line
        if _news_total >= 3:
            _sent_color = "#5a8f6e" if _news_bias=="positivo" else ("#9b4d4d" if _news_bias=="negativo" else "#7a8799")
            story.append(Paragraph(
                f'<font color="{_sent_color}" size="7">Sentimiento de noticias: '
                f'<b>{_news_bias.upper()}</b> — {_news_pos} positivas · {_news_neg} negativas · '
                f'{_news_total - _news_pos - _news_neg} neutras ({_news_total} analizadas)</font>',
                sty("NSummary", fontSize=7, leading=10, spaceBefore=3, spaceAfter=0,
                    textColor=colors.HexColor(_sent_color))
            ))
        story.append(Spacer(1, 4 * mm))

    # Narrative paragraphs
    story.append(Paragraph("<b>Resumen ejecutivo</b>", S_H3))
    story.append(Paragraph(_exec, S_BODY))
    story.append(Paragraph("<b>Valoración</b>", S_H3))
    story.append(Paragraph(val_para, S_BODY))
    story.append(Paragraph("<b>Fundamentales y salud financiera</b>", S_H3))
    story.append(Paragraph(fund_para, S_BODY))
    story.append(Paragraph("<b>Análisis técnico</b>", S_H3))
    story.append(Paragraph(tech_para, S_BODY))

    # Bull / Bear table
    if _bulls or _bears:
        story.append(Spacer(1, 3 * mm))
        _bb_rows = [[
            Paragraph('▲ CATALIZADORES (CASO ALCISTA)',
                      sty("BBH1", fontName="Helvetica-Bold", fontSize=7,
                          textColor=C_POS, leading=10)),
            Paragraph('▼ RIESGOS (CASO BAJISTA)',
                      sty("BBH2", fontName="Helvetica-Bold", fontSize=7,
                          textColor=C_NEG, leading=10)),
        ]]
        _max_bb = max(len(_bulls), len(_bears))
        for _i in range(_max_bb):
            _b = _bulls[_i] if _i < len(_bulls) else ""
            _r = _bears[_i]  if _i < len(_bears)  else ""
            _bb_rows.append([
                Paragraph(f"• {_b}" if _b else "",
                          sty(f"B{_i}", fontSize=7.5, textColor=C_SEC, leading=11)),
                Paragraph(f"• {_r}" if _r else "",
                          sty(f"R{_i}", fontSize=7.5, textColor=C_SEC, leading=11)),
            ])
        _bb_tbl = Table(_bb_rows, colWidths=[PW*0.5, PW*0.5])
        _bb_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), C_NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_SURFACE]),
            ("LINEABOVE",      (0, 0), (0, 0), 1.5, C_POS),
            ("LINEABOVE",      (1, 0), (1, 0), 1.5, C_NEG),
            ("LINEAFTER",      (0, 0), (0, -1), 0.5, C_BORDER),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_BORDER),
        ]))
        story.append(_bb_tbl)

    # Final recommendation line
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        f'<font name="Helvetica-Bold" color="{_rec_color_hex}">● WealthView Research: {_wv_rec}</font>'
        + (f'&nbsp;&nbsp;|&nbsp;&nbsp;<font color="#b8c5d0">Objetivo consenso: {_tgt_str} ({_up_str} de potencial)</font>'
           if target_mean else "")
        + f'&nbsp;&nbsp;|&nbsp;&nbsp;<font color="#7a8799">Score: {sc_total:.1f}/10</font>',
        sty("FREC", fontSize=9, fontName="Helvetica-Bold", leading=14,
            borderPad=6, borderColor=colors.HexColor(_rec_color_hex),
            borderWidth=0.5, borderRadius=2)
    ))
    story.append(hr())

    # ── SECCIÓN 6: TESIS DE INVERSIÓN ────────────────────────────────────────
    scenarios = thesis.get("scenarios", [])
    entry_price = thesis.get("entry_price", 0.0) or price or 1.0
    notes = thesis.get("notes", "").strip()

    if scenarios and any(r.get("Precio objetivo ($)", 0) > 0 for r in scenarios):
        story += section(
            "6. Tesis de Inversion por Escenarios",
            "El modelo de escenarios asigna una probabilidad a cada posible desenlace y calcula el "
            "Valor Esperado (EV) ponderado. El EV positivo indica que, en terminos estadisticos, "
            "la operacion tiene sentido. No predice el futuro: solo cuantifica el riesgo/recompensa "
            "segun tus estimaciones."
        )
        story.append(Paragraph(
            f"<b>Precio de entrada de referencia:</b> ${entry_price:.2f}",
            sty("EP", fontName="Helvetica-Bold", fontSize=9, textColor=C_GOLD_L, spaceAfter=8)
        ))

        tbl_sc, ev_total = scenario_table(scenarios, entry_price)
        story.append(tbl_sc)
        story.append(Spacer(1, 4 * mm))

        ev_color_hex = "#5a8f6e" if ev_total >= 0 else "#9b4d4d"
        ev_price = entry_price * (1 + ev_total)
        story.append(Paragraph(
            f'<font name="Helvetica-Bold" color="{ev_color_hex}">EV Total: {ev_total*100:+.1f}%</font>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<font color="#b8c5d0">Precio esperado ponderado: ${ev_price:,.2f}</font>',
            sty("EV", fontSize=10, leading=14, spaceAfter=6)
        ))
        story.append(Paragraph(
            "Interpretacion: el EV representa el retorno medio esperado si el modelo de escenarios "
            "se repitiera muchas veces. Un EV positivo no garantiza rentabilidad en ninguna operacion "
            "individual — es una guia de decision bajo incertidumbre.",
            S_NOTE
        ))

    # ── SECCIÓN 7: NOTAS DEL INVERSOR ────────────────────────────────────────
    if notes:
        story += section(
            "7. Notas del Inversor",
            "Observaciones personales, hitos a seguir e invalidadores de la tesis."
        )
        # Split by newlines
        for line in notes.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, S_BODY))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(gold_hr())
    story.append(Paragraph(
        "<b>AVISO LEGAL:</b> Este documento ha sido generado automaticamente por WealthView con fines "
        "exclusivamente informativos. No constituye asesoramiento de inversion, recomendacion de compra "
        "o venta, ni analisis de inversion regulado. Las metricas provienen de fuentes publicas y pueden "
        "contener errores o desactualizaciones. Invertir conlleva riesgo de perdida de capital. "
        "Consulte siempre con un asesor financiero regulado antes de tomar decisiones de inversion.",
        sty("DISC", fontSize=7, textColor=C_MUTED, leading=10, spaceAfter=4,
            fontName="Helvetica-Oblique")
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.read()



# ── TAB: COMPARABLES ─────────────────────────────────────────────────────────

def _render_comparables(ticker: str, info: dict):
    """Peer comparison table with key valuation multiples."""
    sector   = _safe(info, "sector", "")
    industry = _safe(info, "industry", "")

    from modules.peers import SECTOR_PEERS

    default_peers = [p for p in SECTOR_PEERS.get(sector, []) if p != ticker][:5]

    st.markdown("#### Selecciona los comparables")
    peers_input = st.text_input(
        "Tickers de comparables (separados por coma)",
        value=", ".join(default_peers),
        help="Introduce los tickers de las empresas con las que quieres comparar",
        key="dd_peers_input",
    )

    peer_list = [t.strip().upper() for t in peers_input.split(",") if t.strip()]
    all_tickers = [ticker.upper()] + peer_list

    if not peer_list:
        st.info("Introduce tickers de comparables para ver la tabla.")
        return

    metrics_keys = {
        "P/E": "trailingPE",
        "P/S": "priceToSalesTrailing12Months",
        "P/B": "priceToBook",
        "EV/EBITDA": "enterpriseToEbitda",
        "Margen neto %": "profitMargins",
        "Margen op. %": "operatingMargins",
        "ROE %": "returnOnEquity",
        "D/E": "debtToEquity",
        "Cap. mercado": "marketCap",
    }

    with st.spinner("Cargando datos de comparables..."):
        rows = []
        for t in all_tickers:
            try:
                inf = _fetch_info(t)
                row = {"Empresa": _safe(inf, "shortName") or t, "Ticker": t}
                for col, key in metrics_keys.items():
                    v = _safe(inf, key)
                    if v is None:
                        row[col] = "—"
                    elif col == "Cap. mercado":
                        if v >= 1e12:    row[col] = f"${v/1e12:.2f}T"
                        elif v >= 1e9:   row[col] = f"${v/1e9:.1f}B"
                        else:            row[col] = f"${v/1e6:.0f}M"
                    elif col in ("Margen neto %", "Margen op. %", "ROE %"):
                        row[col] = f"{v*100:.1f}%"
                    elif col == "D/E":
                        row[col] = f"{v/100:.2f}" if v else "—"
                    else:
                        row[col] = f"{v:.1f}x"
                rows.append(row)
            except Exception:
                rows.append({"Empresa": t, "Ticker": t})

    if not rows:
        st.warning("No se pudieron cargar datos.")
        return

    df_comp = pd.DataFrame(rows).set_index("Ticker")

    def highlight_focus(s):
        return ["background-color: rgba(201,168,76,0.15); font-weight:600;" if s.name == ticker.upper()
                else "" for _ in s]

    st.dataframe(
        df_comp.style.apply(highlight_focus, axis=1),
        use_container_width=True,
        height=min(60 + len(df_comp) * 38, 400),
    )

    if sector or industry:
        st.caption(f"Sector: {sector} · Industria: {industry}")


# ── TAB: DCF ─────────────────────────────────────────────────────────────────

def _render_dcf(ticker: str, info: dict, current_price: float):
    """Blended intrinsic value model: P/FCF (50%) + Fwd P/E (20%) + EV/EBITDA (20%) + DCF (10%)."""

    from modules.peers import get_sector_medians, normalize_sector

    GOLD        = "#c9a84c"
    POSITIVE    = "#4a8f5a"
    NEGATIVE    = "#9b4d4d"
    TEXT_MUTED  = "#6b7a99"

    # ── Sector medians ────────────────────────────────────────────────────────
    sector_raw  = _safe(info, "sector") or ""
    sector_key  = normalize_sector(sector_raw)
    with st.spinner("Cargando medianas sectoriales..."):
        pm = get_sector_medians(sector_key)

    pfcf_sector    = pm.get("pfcf")    or 20.0
    fwd_pe_sector  = pm.get("fwd_pe")  or 18.0
    ev_ebitda_sect = pm.get("ev_ebitda") or 14.0

    # ── FMP badge ─────────────────────────────────────────────────────────────
    _fmp_active = info.get("_fmp_source", False)
    if _fmp_active:
        _fmp_fcf_yr = info.get("_fmp_fcf_year", "")
        _fmp_rev_yr = info.get("_fmp_rev_year", "")
        _yr_label   = f" · datos {_fmp_fcf_yr}" if _fmp_fcf_yr else ""
        st.markdown(
            f"<div style='font-size:10px; color:#4a8f5a; background:#0a1a0f; border:1px solid #1c3a26; "
            f"border-radius:6px; padding:5px 10px; display:inline-block; margin-bottom:12px;'>"
            f"✓ Fuente: Financial Modeling Prep (FMP){_yr_label}</div>",
            unsafe_allow_html=True,
        )

    # ── Base data with fallbacks ──────────────────────────────────────────────
    fcf    = _safe(info, "freeCashflow")
    _fcf_src = "FMP reportado" if (_fmp_active and fcf is not None) else "reportado"
    if fcf is None:
        _ocf   = _safe(info, "operatingCashflow")
        _capex = _safe(info, "capitalExpenditures")   # negativo en yfinance
        if _ocf is not None and _capex is not None:
            fcf      = _ocf + _capex
            _fcf_src = f"estimado (OCF {_ocf/1e6:+.0f}M + Capex {_capex/1e6:+.0f}M)"
        else:
            _fcf_src = "no disponible"

    # FCF promedio 3 años de FMP (más estable para el DCF)
    _fcf_3y = info.get("freeCashflow_3y_avg")

    mktcap           = _safe(info, "marketCap") or 0.0
    shares = _safe(info, "sharesOutstanding") or _safe(info, "impliedSharesOutstanding")
    _shr_src = "reportado"
    if not shares:
        current_price_ref = current_price or 0.0
        if mktcap and current_price_ref > 0:
            shares   = mktcap / current_price_ref
            _shr_src = "estimado (MarketCap / Precio)"
        else:
            _shr_src = "no disponible"

    cash   = _safe(info, "totalCash")   or 0.0
    debt   = _safe(info, "totalDebt")   or 0.0
    net_cash = cash - debt

    revenue          = _safe(info, "totalRevenue")
    fwd_pe_ticker    = _safe(info, "forwardPE")
    ev               = _safe(info, "enterpriseValue")
    ev_ebitda_ticker = _safe(info, "enterpriseToEbitda")

    # ── Method 1: P/FCF (50%) ─────────────────────────────────────────────────
    m1 = None
    _m1_reason = ""
    if fcf is not None and shares and shares > 0:
        if fcf > 0:
            m1 = (fcf / shares) * pfcf_sector
        else:
            _m1_reason = f"FCF negativo ({fcf/1e6:.0f}M) — P/FCF no aplicable"
    elif fcf is None:
        _m1_reason = "FCF no disponible en el API"
    elif not shares:
        _m1_reason = "Número de acciones no disponible"

    # ── Method 2: Forward P/E (20%) ───────────────────────────────────────────
    m2 = None
    _m2_reason = ""
    if fwd_pe_ticker and fwd_pe_ticker > 0 and current_price and current_price > 0:
        fwd_eps = current_price / fwd_pe_ticker
        m2 = fwd_eps * fwd_pe_sector
    elif not fwd_pe_ticker:
        _m2_reason = "Forward P/E no disponible (empresa sin estimados de beneficios)"
    elif fwd_pe_ticker <= 0:
        _m2_reason = f"Forward P/E negativo ({fwd_pe_ticker:.1f}) — empresa en pérdidas"

    # ── Method 3: EV/EBITDA (20%) ────────────────────────────────────────────
    m3 = None
    _m3_reason = ""
    if ev and ev_ebitda_ticker and ev_ebitda_ticker > 0 and shares and shares > 0:
        ebitda   = ev / ev_ebitda_ticker
        fair_ev  = ebitda * ev_ebitda_sect
        m3       = (fair_ev + net_cash) / shares
    elif mktcap and ev_ebitda_ticker and ev_ebitda_ticker > 0 and shares and shares > 0:
        est_ev  = mktcap + debt - cash
        ebitda  = est_ev / ev_ebitda_ticker
        fair_ev = ebitda * ev_ebitda_sect
        m3      = (fair_ev + net_cash) / shares
    elif not ev_ebitda_ticker or ev_ebitda_ticker <= 0:
        _m3_reason = "EV/EBITDA no disponible o negativo (empresa sin beneficios operativos)"
    elif not shares:
        _m3_reason = "Número de acciones no disponible"

    # ── Method 4: DCF — sliders first so result feeds blended ────────────────
    st.markdown("#### Supuestos DCF (Método 4 — 10% del precio objetivo)")
    col1, col2, col3 = st.columns(3)
    with col1:
        g1 = st.slider("Crec. FCF años 1-3 (%)", -20, 80, 15, 1, key="dcf_g1",
                       help="Tasa de crecimiento anual del FCF en los primeros 3 años") / 100.0
        g2 = st.slider("Crec. FCF años 4-5 (%)", -20, 50, 8, 1, key="dcf_g2") / 100.0
    with col2:
        wacc   = st.slider("WACC (%)", 5, 25, 10, 1, key="dcf_wacc",
                           help="Coste medio ponderado del capital.") / 100.0
        g_term = st.slider("Crec. terminal (%)", 0, 6, 3, 1, key="dcf_gterm") / 100.0
    with col3:
        mos    = st.slider("Margen de seguridad (%)", 0, 40, 20, 5, key="dcf_mos") / 100.0
        fcf_adj = st.slider("Ajuste FCF base (%)", -50, 100, 0, 5, key="dcf_adj",
                            help="Ajusta si el FCF actual es atípico.") / 100.0

    # Opción de usar FCF promedio 3 años (disponible cuando FMP está activo)
    _use_3y_fcf = False
    if _fcf_3y is not None and fcf is not None and abs(_fcf_3y - fcf) / max(abs(fcf), 1) > 0.1:
        _use_3y_fcf = st.checkbox(
            f"Usar FCF promedio 3 años (${_fcf_3y/1e9:.2f}B) en lugar del FCF anual (${fcf/1e9:.2f}B)",
            value=False, key="dcf_use_3y",
            help="El promedio de 3 años suaviza años atípicos y da una base más conservadora para el DCF."
        )

    m4 = None
    _m4_reason = ""
    fcf_base = (_fcf_3y if _use_3y_fcf else fcf)
    pv_list, projected = [], []
    pv_tv = 0.0
    if fcf_base is not None and shares and shares > 0 and wacc > g_term:
        fcf0 = fcf_base * (1 + fcf_adj)
        cf   = fcf0
        for yr in range(1, 6):
            g  = g1 if yr <= 3 else g2
            cf = cf * (1 + g)
            pv_list.append(cf / ((1 + wacc) ** yr))
            projected.append(cf)
        tv    = projected[-1] * (1 + g_term) / (wacc - g_term)
        pv_tv = tv / ((1 + wacc) ** 5)
        m4    = (sum(pv_list) + pv_tv + net_cash) / shares
    else:
        if fcf_base is None:
            _m4_reason = "FCF no disponible — " + _fcf_src
        elif not shares:
            _m4_reason = "Número de acciones no disponible"
        elif wacc <= g_term:
            _m4_reason = f"WACC ({int(wacc*100)}%) debe ser mayor que crecimiento terminal ({int(g_term*100)}%)"

    # ── Method 5: P/S (fallback para empresas sin FCF ni beneficios) ─────────
    ps_sector = pm.get("ps") or 3.0
    m5 = None
    _m5_reason = ""
    if revenue and shares and shares > 0:
        m5 = (revenue / shares) * ps_sector
    elif not revenue:
        _m5_reason = "Revenue no disponible"

    st.markdown("---")

    # FCF source note
    if _fcf_src != "reportado":
        st.caption(f"ℹ FCF {_fcf_src}. Ajusta con el slider 'Ajuste FCF base' si es necesario.")

    # ── Weighted blended price ────────────────────────────────────────────────
    RAW_W = {1: 0.50, 2: 0.20, 3: 0.20, 4: 0.10, 5: 0.0}
    vals  = {1: m1,   2: m2,   3: m3,   4: m4,   5: m5}

    # Si < 2 métodos principales disponibles, activar P/S como rescate
    _main_avail = sum(1 for k in [1, 2, 3, 4] if vals[k] is not None and vals[k] > 0)
    if _main_avail < 2 and m5 is not None and m5 > 0:
        RAW_W[5] = 0.40

    # Redistribuir pesos de métodos no disponibles
    avail_w  = {k: w for k, w in RAW_W.items() if vals[k] is not None and vals[k] > 0}
    total_aw = sum(avail_w.values())
    if total_aw > 0:
        norm_w  = {k: w / total_aw for k, w in avail_w.items()}
        blended = sum(norm_w[k] * vals[k] for k in norm_w)
    else:
        blended = None

    # ── UI: Blended target ────────────────────────────────────────────────────
    upside = ((blended / current_price) - 1) * 100 if (blended and current_price and current_price > 0) else None
    up_color = POSITIVE if (upside or 0) >= 0 else NEGATIVE

    if blended:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#0d1117,#131929); "
            f"border:1px solid {GOLD}44; border-radius:12px; padding:24px 28px; margin-bottom:20px;'>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1.5px; margin-bottom:8px;'>Precio Objetivo Intrinseco — WealthView</div>"
            f"<div style='font-size:42px; font-weight:800; color:{GOLD}; line-height:1;'>"
            f"${blended:,.2f}</div>"
            f"<div style='font-size:14px; color:{up_color}; margin-top:8px; font-weight:600;'>"
            f"{upside:+.1f}% vs precio actual (${current_price:,.2f})</div>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:6px;'>"
            f"Media ponderada de {len(avail_w)} métodos · P/FCF {int(norm_w.get(1,0)*100)}% · "
            f"Fwd P/E {int(norm_w.get(2,0)*100)}% · EV/EBITDA {int(norm_w.get(3,0)*100)}% · "
            f"DCF {int(norm_w.get(4,0)*100)}%"
            + (f" · P/S {int(norm_w.get(5,0)*100)}%" if norm_w.get(5, 0) > 0 else "")
            + "</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Datos insuficientes para calcular el precio objetivo — no hay métodos de valoración disponibles.")
        with st.expander("¿Por qué no se pudo calcular?"):
            reasons = {
                "P/FCF (M1)":     _m1_reason or "FCF negativo o no disponible",
                "Fwd P/E (M2)":   _m2_reason or "No disponible",
                "EV/EBITDA (M3)": _m3_reason or "No disponible",
                "DCF (M4)":       _m4_reason or "FCF no disponible",
                "P/S (M5)":       _m5_reason or "Revenue no disponible",
            }
            for method, reason in reasons.items():
                st.markdown(f"**{method}**: {reason}")

    # ── UI: Method cards ──────────────────────────────────────────────────────
    def _method_card(label, method_id, price, weight_raw, weight_eff, ref_label, ref_val):
        available = price is not None and price > 0
        up = ((price / current_price) - 1) * 100 if (available and current_price and current_price > 0) else None
        up_c = POSITIVE if (up or 0) >= 0 else NEGATIVE
        price_str = f"${price:,.2f}" if available else "N/D"
        up_str    = f"{up:+.1f}%" if up is not None else ""
        w_str     = f"{int(weight_eff*100)}% efectivo" if available else "excluido"
        return (
            f"<div style='background:#0d1117; border:1px solid {'#c9a84c44' if available else '#1c2333'}; "
            f"border-radius:10px; padding:16px 14px; height:100%;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:4px;'>Método {method_id} · {w_str}</div>"
            f"<div style='font-size:11px; color:{GOLD}; font-weight:700; margin-bottom:8px;'>{label}</div>"
            f"<div style='font-size:24px; font-weight:800; color:{'#e8e8e8' if available else TEXT_MUTED};'>{price_str}</div>"
            f"<div style='font-size:11px; color:{up_c}; margin-top:4px;'>{up_str}</div>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:8px; border-top:1px solid #1c2333; padding-top:6px;'>"
            f"{ref_label}: <b style='color:#e8e8e8'>{ref_val}</b></div>"
            f"</div>"
        )

    eff_w = norm_w if total_aw > 0 else {}
    _reasons_map = {1: _m1_reason, 2: _m2_reason, 3: _m3_reason, 4: _m4_reason, 5: _m5_reason}

    # Extend _method_card to show reason when N/D
    def _method_card_ext(label, method_id, price, weight_raw, weight_eff, ref_label, ref_val, reason=""):
        available = price is not None and price > 0
        up = ((price / current_price) - 1) * 100 if (available and current_price and current_price > 0) else None
        up_c = POSITIVE if (up or 0) >= 0 else NEGATIVE
        price_str = f"${price:,.2f}" if available else "N/D"
        up_str    = f"{up:+.1f}%" if up is not None else ""
        w_str     = f"{int(weight_eff*100)}% efectivo" if available else "excluido"
        reason_html = (f"<div style='font-size:9px; color:#9b4d4d; margin-top:4px; "
                       f"line-height:1.3;'>{reason}</div>") if reason and not available else ""
        return (
            f"<div style='background:#0d1117; border:1px solid {'#c9a84c44' if available else '#1c2333'}; "
            f"border-radius:10px; padding:16px 14px; height:100%;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:4px;'>Método {method_id} · {w_str}</div>"
            f"<div style='font-size:11px; color:{GOLD}; font-weight:700; margin-bottom:8px;'>{label}</div>"
            f"<div style='font-size:24px; font-weight:800; color:{'#e8e8e8' if available else TEXT_MUTED};'>{price_str}</div>"
            f"<div style='font-size:11px; color:{up_c}; margin-top:4px;'>{up_str}</div>"
            f"{reason_html}"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:8px; border-top:1px solid #1c2333; padding-top:6px;'>"
            f"{ref_label}: <b style='color:#e8e8e8'>{ref_val}</b></div>"
            f"</div>"
        )

    _show_ps = m5 is not None or _main_avail < 2
    if _show_ps:
        c1, c2, c3, c4, c5 = st.columns(5)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c5 = None
    with c1:
        st.markdown(_method_card_ext(
            "P/FCF", 1, m1, RAW_W[1], eff_w.get(1, 0),
            f"P/FCF sector ({sector_key})", f"{pfcf_sector:.1f}x", _m1_reason
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_method_card_ext(
            "P/E Forward", 2, m2, RAW_W[2], eff_w.get(2, 0),
            f"P/E fwd sector", f"{fwd_pe_sector:.1f}x", _m2_reason
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_method_card_ext(
            "EV/EBITDA", 3, m3, RAW_W[3], eff_w.get(3, 0),
            f"EV/EBITDA sector", f"{ev_ebitda_sect:.1f}x", _m3_reason
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(_method_card_ext(
            "DCF (5 años)", 4, m4, RAW_W[4], eff_w.get(4, 0),
            "WACC / Crec. term.", f"{int(wacc*100)}% / {int(g_term*100)}%", _m4_reason
        ), unsafe_allow_html=True)
    if c5 is not None:
        with c5:
            st.markdown(_method_card_ext(
                "P/S (rescate)", 5, m5, RAW_W[5], eff_w.get(5, 0),
                f"P/S sector", f"{ps_sector:.1f}x", _m5_reason
            ), unsafe_allow_html=True)

    # ── UI: Horizontal comparison chart ───────────────────────────────────────
    if blended and current_price and current_price > 0:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        method_names = []
        method_prices = []
        method_colors = []
        labels_map = {1: "P/FCF (50%)", 2: "Fwd P/E (20%)", 3: "EV/EBITDA (20%)", 4: "DCF (10%)"}
        for k in [1, 2, 3, 4]:
            v = vals[k]
            if v and v > 0:
                method_names.append(labels_map[k])
                method_prices.append(v)
                method_colors.append(GOLD)
        method_names.append("Precio actual")
        method_prices.append(current_price)
        method_colors.append("#4a6080")
        method_names.append("Objetivo blended")
        method_prices.append(blended)
        method_colors.append("#5a8f5a" if blended >= current_price else "#9b4d4d")

        fig_bar = go.Figure(go.Bar(
            x=method_prices,
            y=method_names,
            orientation="h",
            marker_color=method_colors,
            text=[f"${v:,.2f}" for v in method_prices],
            textposition="outside",
            textfont=dict(color="#e8e8e8", size=11),
        ))
        layout_bar = dict(**PLOTLY_DARK)
        layout_bar["margin"] = dict(l=10, r=80, t=10, b=30)
        layout_bar.update(height=220, showlegend=False,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        fig_bar.update_layout(**layout_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── UI: DCF waterfall ─────────────────────────────────────────────────────
    if pv_list and m4 and shares and shares > 0:
        st.markdown("##### Detalle DCF — Flujos descontados")
        labels_wf = [f"Año {i+1}" for i in range(5)] + ["Valor Terminal", "Caja Neta", "Valor Intrínseco"]
        values_wf = pv_list + [pv_tv, net_cash, 0]
        measure   = ["relative"] * 7 + ["total"]
        fig_wf = go.Figure(go.Waterfall(
            orientation="v", measure=measure,
            x=labels_wf, y=values_wf,
            connector=dict(line=dict(color="rgba(201,168,76,0.3)", width=1)),
            increasing=dict(marker_color=POSITIVE),
            decreasing=dict(marker_color=NEGATIVE),
            totals=dict(marker_color=GOLD),
            text=[_fmt(v, prefix="$") for v in values_wf],
            textposition="outside",
        ))
        layout_wf = dict(**PLOTLY_DARK)
        layout_wf["margin"] = dict(l=40, r=20, t=20, b=40)
        layout_wf.update(yaxis_title="USD", height=320, showlegend=False)
        fig_wf.update_layout(**layout_wf)
        st.plotly_chart(fig_wf, use_container_width=True)

    st.caption(
        "⚠ Precio objetivo calculado con datos publicos. No constituye asesoramiento de inversion. "
        "El modelo es sensible a los supuestos de crecimiento, WACC y a la calidad del FCF reportado. "
        f"Medianas sectoriales: P/FCF {pfcf_sector:.1f}x · Fwd P/E {fwd_pe_sector:.1f}x · "
        f"EV/EBITDA {ev_ebitda_sect:.1f}x — sector: {sector_key}"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlc(ticker: str, period: str = "1y"):
    """Fetch OHLC history for a ticker via yfinance. Returns DataFrame or None."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        return hist if not hist.empty else None
    except Exception:
        return None


# ── TAB: AI ANALYST ──────────────────────────────────────────────────────────

def _render_ai_analyst(ticker: str, info: dict, name: str):
    """LLM-powered qualitative investment analysis tab."""
    from modules.peers import get_sector_medians, normalize_sector
    from modules.llm_analyst import generate_analysis, build_snapshot, llm_available, active_provider

    if not llm_available():
        st.warning(
            "No hay API key configurada para el Analyst IA. "
            "Añade ANTHROPIC_API_KEY o OPENAI_API_KEY a tu archivo .env y reinicia la app."
        )
        return

    # Gather all the data we need for the snapshot
    sector_raw = _safe(info, "sector") or ""
    sector_key = normalize_sector(sector_raw)

    with st.spinner("Calculando métricas y cargando medianas del sector..."):
        peer_medians = get_sector_medians(sector_key)

    # Technical indicators
    hist = _fetch_ohlc(ticker)
    rsi_val = None
    sma200_above = None
    last_price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")
    if hist is not None and not hist.empty:
        close = hist["Close"].squeeze()
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_series = 100 - 100 / (1 + rs)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        sma200 = close.rolling(200).mean()
        if not sma200.empty and last_price:
            sma200_above = bool(last_price > float(sma200.iloc[-1]))

    # FCF yield
    fcf_yield = None
    try:
        cf = _fetch_cashflow(ticker)
        mktcap = _safe(info, "marketCap")
        if cf is not None and not cf.empty and mktcap and mktcap > 0:
            ocf_row = next((r for r in ["Operating Cash Flow", "Total Cash From Operating Activities"] if r in cf.index), None)
            capex_row = next((r for r in ["Capital Expenditure", "Capital Expenditures"] if r in cf.index), None)
            if ocf_row and capex_row:
                ocf = float(cf.loc[ocf_row].iloc[0])
                capex = float(cf.loc[capex_row].iloc[0])
                fcf = ocf + capex
                fcf_yield = fcf / mktcap * 100
    except Exception:
        pass

    # Price target / upside
    target_price = _safe(info, "targetMeanPrice")
    up_pct = None
    if target_price and last_price and last_price > 0:
        up_pct = (target_price - last_price) / last_price * 100

    # News
    news = _fetch_news(ticker)
    headlines = [n["title"] for n in news]

    # WV score (quick recompute — lightweight version for display)
    wv_score = st.session_state.get(f"_wv_score_{ticker}")
    wv_rec = st.session_state.get(f"_wv_rec_{ticker}", "")
    reasons_buy = st.session_state.get(f"_wv_rb_{ticker}", [])
    reasons_sell = st.session_state.get(f"_wv_rs_{ticker}", [])

    snap_json = build_snapshot(
        ticker=ticker, name=name, sector=sector_key, info=info,
        financials={}, wv_score=wv_score, wv_rec=wv_rec,
        reasons_buy=reasons_buy, reasons_sell=reasons_sell,
        news_headlines=headlines, peer_medians=peer_medians,
        rsi=rsi_val, sma200_above=sma200_above,
        price=last_price, fcf_yield=fcf_yield,
        up_pct=up_pct, target_price=target_price,
    )

    provider = active_provider()
    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-bottom:10px;'>"
        f"Análisis generado por <b>{provider}</b> · "
        f"Sector de referencia: <b>{sector_key}</b> "
        f"({peer_medians.get('_n_peers', 0)} empresas comparables)</p>",
        unsafe_allow_html=True,
    )

    run_col, _ = st.columns([2, 4])
    with run_col:
        run_btn = st.button("🤖 Generar análisis AI", type="primary",
                             use_container_width=True, key=f"ai_run_{ticker}")

    if run_btn or st.session_state.get(f"_ai_result_{ticker}"):
        if run_btn:
            # Force regeneration
            st.session_state[f"_ai_result_{ticker}"] = None
            with st.spinner(f"Analizando {name} con {provider}..."):
                result = generate_analysis(snap_json)
            st.session_state[f"_ai_result_{ticker}"] = result
        else:
            result = st.session_state[f"_ai_result_{ticker}"]

        if result and result.get("error"):
            st.error(f"Error al generar análisis: {result['error']}")
        elif result and result.get("text"):
            st.markdown(
                f"<div style='background:{SURFACE_2}; border:1px solid {GOLD_BORDER}; "
                f"border-left:4px solid {GOLD}; border-radius:8px; padding:24px 28px; "
                f"margin-top:12px; line-height:1.8; color:{TEXT_PRIMARY}; font-size:13px;'>"
                + result["text"].replace("\n", "<br>").replace("\n\n", "<br><br>")
                + f"<br><br><span style='font-size:10px; color:{TEXT_MUTED};'>"
                f"Generado por {result.get('provider','IA')} — "
                "Solo con fines informativos. No constituye asesoramiento financiero.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Sin resultado disponible. Pulsa el botón para generar el análisis.")
    else:
        st.markdown(
            f"<div style='background:{SURFACE_2}; border:1px solid {GOLD_BORDER}; "
            f"border-radius:8px; padding:32px; text-align:center; color:{TEXT_MUTED};'>"
            f"<div style='font-size:32px; margin-bottom:12px;'>🤖</div>"
            f"<div style='font-size:14px; font-weight:600; color:{TEXT_SECONDARY}; margin-bottom:8px;'>"
            f"Análisis cualitativo con IA</div>"
            f"<div style='font-size:12px;'>Pulsa el botón para que {provider} genere "
            f"una tesis de inversión de calidad institucional basada en los datos actuales de {ticker}.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── ENTRY POINT ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns_for_jb(ticker: str):
    """Descarga retornos diarios de 2 años para el test JB."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None
        return hist["Close"].squeeze().pct_change().dropna()
    except Exception:
        return None


def _render_normalidad(ticker: str):
    """Tab de análisis de normalidad — 3 tests: JB, Ljung-Box, ARCH."""
    from modules.utils import jarque_bera_test, ljung_box_test, arch_lm_test
    from modules.styles import (
        GOLD, SURFACE, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK, plotly_layout
)

    st.markdown(
        "<div style='margin-bottom:6px; font-size:10px; font-weight:700; "
        "color:#7a8799; text-transform:uppercase; letter-spacing:1.2px;'>"
        "Diagnóstico de Normalidad · JB · Ljung-Box · ARCH</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Tres tests complementarios: **JB** evalúa la forma de la distribución · "
        "**Ljung-Box** detecta dependencia serial (autocorrelación) · "
        "**ARCH** detecta volatility clustering. "
        "Cuando alguno falla, ciertas métricas estándar pierden validez estadística."
    )
    st.markdown("---")

    with st.spinner(f"Calculando tests de normalidad para {ticker}..."):
        returns = _fetch_returns_for_jb(ticker)

    if returns is None or len(returns) < 30:
        st.warning(f"Datos insuficientes para {ticker} (mínimo 30 observaciones).")
        return

    jb_res   = jarque_bera_test(returns)
    lb_res   = ljung_box_test(returns)
    arch_res = arch_lm_test(returns)
    _rho     = lb_res["rho"] if lb_res else {}

    if jb_res is None:
        st.error("No se pudo calcular el test.")
        return

    # ── Overall verdict ───────────────────────────────────────────────────────
    _s_map   = {"normal": 0, "dudosa": 1, "no_normal": 2}
    _all_res = [r for r in [jb_res, lb_res, arch_res] if r]
    _score   = sum(_s_map.get(r["status"], 0) for r in _all_res)
    _n_res   = len(_all_res)
    _n_fail  = sum(1 for r in _all_res if r["status"] != "normal")

    if _score == 0:
        _overall = "normal"
    elif _score <= _n_res * 0.75:
        _overall = "dudosa"
    else:
        _overall = "no_normal"

    def _nt_color(s): return {"normal": POSITIVE, "dudosa": GOLD, "no_normal": NEGATIVE}.get(s, "#6b7280")
    def _nt_dot(s):   return {"normal": "🟢", "dudosa": "🟡", "no_normal": "🔴"}.get(s, "⚪")
    def _nt_lbl(s):   return {"normal": "Normal", "dudosa": "Dudosa", "no_normal": "No normal"}.get(s, "—")

    # ── Mini-cell helper ──────────────────────────────────────────────────────
    def _mm(label, value):
        return (
            f"<div style='background:#0a0f1a; border-radius:6px; padding:8px 10px;'>"
            f"<div style='color:#6b7280; font-size:9px; text-transform:uppercase; "
            f"letter-spacing:0.5px; margin-bottom:3px;'>{label}</div>"
            f"<div style='color:#e2e8f0; font-size:13px; font-weight:700; "
            f"font-family:monospace;'>{value}</div></div>"
        )

    # ── Test card helper ──────────────────────────────────────────────────────
    def _test_card(title, subtitle, st_val, cells):
        c = _nt_color(st_val)
        return (
            f"<div style='background:#0d1117; border:1px solid {c}44; "
            f"border-left:3px solid {c}; border-radius:10px; padding:16px 18px;'>"
            f"<div style='display:flex; align-items:flex-start; gap:8px; margin-bottom:12px;'>"
            f"<span style='font-size:18px; line-height:1;'>{_nt_dot(st_val)}</span>"
            f"<div><div style='color:{c}; font-size:12px; font-weight:700; "
            f"line-height:1.2;'>{title} — {_nt_lbl(st_val)}</div>"
            f"<div style='color:#6b7280; font-size:10px; margin-top:2px;'>{subtitle}</div>"
            f"</div></div>"
            f"<div style='display:grid; grid-template-columns:1fr 1fr; gap:6px;'>"
            f"{''.join(cells)}</div></div>"
        )

    # ── Three test cards ──────────────────────────────────────────────────────
    _tc1, _tc2, _tc3 = st.columns(3)

    _tc1.markdown(_test_card(
        "Jarque-Bera", "Forma de la distribución",
        jb_res["status"],
        [_mm("JB Stat", f"{jb_res['jb']:.2f}"),
         _mm("p-valor", f"{jb_res['p_value']:.4f}"),
         _mm("Skewness", f"{jb_res['skewness']:+.3f}"),
         _mm("Exc. Curtosis", f"{jb_res['excess_kurtosis']:+.3f}")]
    ), unsafe_allow_html=True)

    if lb_res:
        _tc2.markdown(_test_card(
            "Ljung-Box", "Dependencia serial — lags 1-3",
            lb_res["status"],
            [_mm("Q stat", f"{lb_res['Q']:.2f}"),
             _mm("p-valor", f"{lb_res['p_value']:.4f}"),
             _mm("ρ₁ / ρ₂", f"{_rho.get(1,0):+.3f} / {_rho.get(2,0):+.3f}"),
             _mm("ρ₃", f"{_rho.get(3,0):+.3f}")]
        ), unsafe_allow_html=True)

    if arch_res:
        _tc3.markdown(_test_card(
            "ARCH — Engle", "Volatility clustering",
            arch_res["status"],
            [_mm("LM stat", f"{arch_res['LM']:.2f}"),
             _mm("p-valor", f"{arch_res['p_value']:.4f}"),
             _mm("R² OLS", f"{arch_res['R2']:.5f}"),
             _mm("Lags", f"{arch_res['lags']}")]
        ), unsafe_allow_html=True)

    # ── Overall verdict banner ────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _ov_c = _nt_color(_overall)
    if _overall == "normal":
        _ov_title = "Diagnóstico: Distribución Normal"
        _ov_msg   = f"Los {_n_res} tests no rechazan normalidad para {ticker}. Las métricas estándar tienen plena validez."
    elif _overall == "dudosa":
        _ov_title = "Diagnóstico: Distribución Dudosa"
        _ov_msg   = f"{_n_fail} de {_n_res} tests detectan desviaciones para {ticker}. Interpreta Sharpe y VaR con precaución."
    else:
        _ov_title = "Diagnóstico: Distribución Problemática"
        _ov_msg   = (f"{_n_fail} de {_n_res} tests rechazan los supuestos para {ticker}. "
                     f"Sharpe, Beta, VaR paramétrico y Alpha de Jensen tienen validez reducida.")

    st.markdown(
        f"<div style='background:#0d1117; border:1px solid {_ov_c}44; "
        f"border-left:4px solid {_ov_c}; border-radius:10px; padding:16px 20px; margin-top:4px;'>"
        f"<div style='display:flex; align-items:center; gap:10px;'>"
        f"<span style='font-size:22px;'>{_nt_dot(_overall)}</span>"
        f"<div><div style='color:{_ov_c}; font-size:13px; font-weight:700;'>{_ov_title}</div>"
        f"<div style='color:#9ca3af; font-size:12px; margin-top:3px; line-height:1.5;'>{_ov_msg}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── Panel comparativo de métricas robustas ─────────────────────────────────
    if _overall != "normal":
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _ret_arr    = returns.values
        _ann_ret    = float(np.mean(_ret_arr)) * 252
        _ann_vol    = float(np.std(_ret_arr)) * np.sqrt(252)
        _rf_dd      = st.session_state.get("risk_free_rate", 4.0) / 100
        _sharpe_dd  = (_ann_ret - _rf_dd) / _ann_vol if _ann_vol > 0 else 0.0
        _dr         = returns[returns < 0]
        _down_vol   = float(_dr.std()) * np.sqrt(252) if len(_dr) > 1 else _ann_vol
        _sortino_dd = (_ann_ret - _rf_dd) / _down_vol if _down_vol > 0 else 0.0
        _var95_t    = float(np.percentile(_ret_arr, 5))
        _cvar95     = float(returns[returns <= _var95_t].mean())
        _gains_s    = returns[returns > 0].sum()
        _loss_s     = abs(returns[returns < 0].sum())
        _omega_dd   = float(_gains_s / _loss_s) if _loss_s > 0 else 99.9

        _R_BG  = "rgba(155,77,77,0.06)";  _R_BD  = "rgba(155,77,77,0.28)"
        _G_BG  = "rgba(201,168,76,0.07)"; _G_BD  = "rgba(201,168,76,0.32)"

        def _dd_card(label, value, desc, bg, bd, text_c, badge, badge_c):
            return (
                f"<div style='background:{bg}; border:1px solid {bd}; border-radius:10px; "
                f"padding:14px 16px; text-align:center;'>"
                f"<div style='color:{badge_c}; font-size:10px; font-weight:700; "
                f"text-transform:uppercase; letter-spacing:0.7px; margin-bottom:8px;'>{badge}</div>"
                f"<div style='color:#9ca3af; font-size:12px; font-weight:600; margin-bottom:6px;'>{label}</div>"
                f"<div style='color:{text_c}; font-size:22px; font-weight:700; font-family:monospace; "
                f"margin-bottom:4px;'>{value}</div>"
                f"<div style='color:#6b7280; font-size:10px; line-height:1.4;'>{desc}</div></div>"
            )

        def _dd_arrow():
            return "<div style='text-align:center; color:#4b5563; font-size:20px; padding-top:26px;'>→</div>"

        _dd_pairs = [
            ("Sharpe Ratio",        f"{_sharpe_dd:.2f}", "Asume dist. normal",
             "Sortino Ratio",       f"{_sortino_dd:.2f}", "Solo penaliza pérdidas"),
            ("VaR 95%",             f"{_var95_t*100:.2f}%", "Cuantil de pérdida",
             "CVaR / E. Shortfall", f"{_cvar95*100:.2f}%", "Pérdida media en peor 5%"),
            ("Volatilidad total",   f"{_ann_vol*100:.1f}%", "Penaliza subidas y bajadas",
             "Desv. bajista",       f"{_down_vol*100:.1f}%", "Solo mide la vol. negativa"),
        ]

        st.markdown(
            f"<div style='margin:4px 0 14px 0; padding:10px 16px; "
            f"background:rgba(201,168,76,0.05); border:1px solid rgba(201,168,76,0.2); "
            f"border-radius:8px;'>"
            f"<span style='color:#c9a84c; font-size:11px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:1px;'>"
            f"⚖ Con no-normalidad detectada — métricas que debes priorizar</span></div>",
            unsafe_allow_html=True,
        )

        for _i in range(0, len(_dd_pairs), 2):
            _chunk = _dd_pairs[_i:_i + 2]
            if len(_chunk) == 2:
                _p1, _p2 = _chunk
                _dc1, _dar1, _dc2, _dsp, _dc3, _dar2, _dc4 = st.columns([4, 0.55, 4, 0.4, 4, 0.55, 4])
                _dc1.markdown(_dd_card(_p1[0],_p1[1],_p1[2],_R_BG,_R_BD,"#e2e8f0","⚠ Con reservas","#9b4d4d"), unsafe_allow_html=True)
                _dar1.markdown(_dd_arrow(), unsafe_allow_html=True)
                _dc2.markdown(_dd_card(_p1[3],_p1[4],_p1[5],_G_BG,_G_BD,"#c9a84c","✓ Priorizar","#c9a84c"), unsafe_allow_html=True)
                _dsp.markdown("", unsafe_allow_html=True)
                _dc3.markdown(_dd_card(_p2[0],_p2[1],_p2[2],_R_BG,_R_BD,"#e2e8f0","⚠ Con reservas","#9b4d4d"), unsafe_allow_html=True)
                _dar2.markdown(_dd_arrow(), unsafe_allow_html=True)
                _dc4.markdown(_dd_card(_p2[3],_p2[4],_p2[5],_G_BG,_G_BD,"#c9a84c","✓ Priorizar","#c9a84c"), unsafe_allow_html=True)
            else:
                _p1 = _chunk[0]
                _dc1, _dar1, _dc2, _ = st.columns([4, 0.55, 4, 4.9])
                _dc1.markdown(_dd_card(_p1[0],_p1[1],_p1[2],_R_BG,_R_BD,"#e2e8f0","⚠ Con reservas","#9b4d4d"), unsafe_allow_html=True)
                _dar1.markdown(_dd_arrow(), unsafe_allow_html=True)
                _dc2.markdown(_dd_card(_p1[3],_p1[4],_p1[5],_G_BG,_G_BD,"#c9a84c","✓ Priorizar","#c9a84c"), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Histograma de retornos vs curva normal ────────────────────────────────
    import plotly.graph_objects as go

    ret_vals = returns.values
    mu, sigma = float(np.mean(ret_vals)), float(np.std(ret_vals))
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)

    from scipy.stats import norm as _norm
    pdf_normal = _norm.pdf(x_range, mu, sigma)

    _hist_color = _nt_color(_overall)
    def _hex_rgba(h, a=0.55):
        h = h.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ret_vals,
        histnorm="probability density",
        nbinsx=60,
        name="Retornos reales",
        marker_color=_hex_rgba(_hist_color, 0.55),
        marker_line=dict(color=_hist_color, width=0.5),
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_normal,
        mode="lines",
        name="Distribución normal teórica",
        line=dict(color=GOLD, width=2, dash="dot"),
    ))
    layout = dict(**PLOTLY_DARK)
    layout["margin"] = dict(l=20, r=20, t=40, b=30)
    layout.update(
        title=f"Distribución de retornos diarios — {ticker}",
        xaxis_title="Retorno diario",
        yaxis_title="Densidad",
        height=340,
        legend=dict(orientation="h", y=1.08, x=0),
        bargap=0.02,
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Interpretación ────────────────────────────────────────────────────────
    with st.expander("¿Cómo interpretar estos resultados?"):
        st.markdown(f"""
**Jarque-Bera** contrasta la normalidad usando skewness (S) y exceso de curtosis (K):
- **Skewness = {jb_res["skewness"]:+.4f}** — negativa = caídas más extremas que subidas equivalentes
- **Exc. curtosis = {jb_res["excess_kurtosis"]:+.4f}** — positivo = fat tails (eventos extremos más frecuentes de lo que predice la normal)
- JB = {jb_res["jb"]:.3f} · p-valor = {jb_res["p_value"]:.4f} · umbrales χ²(2): 4.61 (90%), 5.99 (95%), 9.21 (99%)

**Ljung-Box** detecta si los retornos son serialmente independientes:
- {"ρ₁=" + str(_rho.get(1,"—")) + " · ρ₂=" + str(_rho.get(2,"—")) + " · ρ₃=" + str(_rho.get(3,"—")) if lb_res else "No calculado"}
- Si hay autocorrelación significativa, los retornos pasados predicen parcialmente los futuros — violan la hipótesis de mercado eficiente

**ARCH de Engle** detecta volatility clustering:
- {"LM=" + str(arch_res["LM"]) + " · p=" + str(arch_res["p_value"]) if arch_res else "No calculado"}
- Si el test rechaza H₀, la varianza cambia en el tiempo (periodos volátiles se agrupan) — el VaR con σ fija es impreciso

**Con no-normalidad detectada, usa:** Omega Ratio, Sortino, Calmar, CVaR histórico y Downside Beta como métricas principales.
        """)


def render_deep_dive():
    from modules.styles import page_header, plotly_layout

    # ── Selector de ticker ────────────────────────────────────────────────────
    ticker_init = st.session_state.get("deep_dive_ticker", "")

    col_search, col_btn, col_refresh = st.columns([3, 1, 1])
    with col_search:
        ticker_input = st.text_input(
            "Ticker",
            value=ticker_init,
            placeholder="Ej: AAPL, IOVA, MSFT, BTC-USD...",
            label_visibility="collapsed",
            key="dd_ticker_input",
        )
    with col_btn:
        go_btn = st.button("🔍 Analizar", type="primary", use_container_width=True, key="dd_go")
    with col_refresh:
        refresh_btn = st.button("🔄 Actualizar", use_container_width=True, key="dd_refresh",
                                help="Fuerza la descarga de datos frescos, ignorando la caché")

    ticker = (ticker_input.strip().upper() if ticker_input else ticker_init)

    if go_btn and ticker:
        st.session_state.deep_dive_ticker = ticker

    # ── Forzar refresco de caché ──────────────────────────────────────────────
    if refresh_btn and ticker:
        _fetch_info.clear()
        _fetch_history.clear()
        _fetch_financials.clear()
        _fetch_news.clear()
        _dp.get_fundamentals.clear()  # limpia también la caché de data_provider
        st.session_state.deep_dive_ticker = ticker
        st.rerun()

    ticker = st.session_state.get("deep_dive_ticker", ticker)

    if not ticker:
        st.markdown(f"""
        <div style="text-align:center; padding:80px 20px; color:{TEXT_MUTED};">
            <div style="font-size:40px; margin-bottom:16px;">🔍</div>
            <div style="font-size:16px; font-weight:600; color:{TEXT_SECONDARY}; margin-bottom:8px;">
                Busca cualquier activo del mundo
            </div>
            <div style="font-size:13px;">
                Acciones · ETFs · Criptos · Divisas · Fondos (ISIN)
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Cargar datos ──────────────────────────────────────────────────────────
    page_header(f"Deep Dive — {ticker}", "Análisis fundamental · técnico · tesis de inversión")

    with st.spinner(f"Cargando {ticker}..."):
        info = _fetch_info(ticker)

    if not info:
        st.error(f"No se encontraron datos para '{ticker}'. Verifica el ticker.")
        return

    current_price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice") or 0.0
    name = _safe(info, "longName") or _safe(info, "shortName") or ticker

    # ── Precio en tiempo real — siempre fresco, sin caché ────────────────────
    try:
        live = _dp.get_live_price(ticker)
        if live and live > 0:
            current_price = live
    except Exception:
        pass

    # Badge de precio actual
    prev = _safe(info, "previousClose") or current_price
    change_pct = ((current_price - prev) / prev * 100) if prev else 0.0
    chg_color = POSITIVE if change_pct >= 0 else NEGATIVE
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:16px; margin-bottom:20px;'>"
        f"<span style='font-family:DM Mono,monospace; font-size:28px; font-weight:600;"
        f" color:{TEXT_PRIMARY};'>${current_price:,.2f}</span>"
        f"<span style='background:{'rgba(90,143,110,0.15)' if change_pct >= 0 else 'rgba(155,77,77,0.15)'};"
        f" color:{chg_color}; border-radius:4px; padding:4px 12px;"
        f" font-size:13px; font-weight:600;'>{change_pct:+.2f}% hoy</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    username = st.session_state.get("username", "")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Overview", "Técnico", "Comparables", "DCF",
        "AI Analyst", "Backtesting", "Normalidad JB",
        "Calidad (F/Z)", "Insiders & Inst.",
        "Tesis", "Exportar"
    ])

    with tab1:
        _render_overview(info, ticker)
        gold_divider()
        _render_historicos(ticker)

    with tab2:
        _render_tecnico(ticker)

    with tab3:
        _render_comparables(ticker, info)

    with tab4:
        _render_dcf(ticker, info, current_price)

    with tab5:
        _render_ai_analyst(ticker, info, name)

    with tab6:
        backtesting.render_backtest(ticker)

    with tab7:
        _render_normalidad(ticker)

    with tab8:
        quality_scores.render_quality_scores(ticker, info)

    with tab9:
        edgar.render_edgar_tab(ticker)

    with tab10:
        _render_tesis(ticker, current_price, username)

    with tab11:
        _render_exportar(ticker, info, username)

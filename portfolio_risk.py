"""
portfolio_risk.py — Advanced portfolio management tools for WealthView
Provides:
  - Kelly criterion & volatility-based position sizing
  - ATR-based stop-loss suggestions
  - Concentration heat map
  - Rebalancing drift alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK
)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns(ticker: str, days: int = 252) -> pd.Series | None:
    try:
        end = datetime.now()
        start = end - timedelta(days=days + 30)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True,
                         threads=False)
        if df.empty:
            return None
        close = df["Close"].squeeze()
        return close.pct_change().dropna().tail(days)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_atr(ticker: str, period: int = 14, lookback_days: int = 60) -> float | None:
    """Average True Range over last `lookback_days`."""
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 10)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True,
                         threads=False)
        if df.empty or len(df) < period + 1:
            return None
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        close = df["Close"].squeeze()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_last_price(ticker: str) -> float | None:
    try:
        p = yf.Ticker(ticker).fast_info.last_price
        return float(p) if p else None
    except Exception:
        return None


# ── Position sizing ───────────────────────────────────────────────────────────

def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion: f* = W/L - (1-W)/W_ratio"""
    if avg_loss == 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, min(kelly, 1.0))


def _vol_position_size(
    capital: float,
    ticker_vol: float,
    portfolio_vol_target: float = 0.15,
    n_positions: int = 10,
) -> float:
    """
    Volatility-based sizing: weight each position so its contribution
    to portfolio volatility equals (target_vol / n_positions).
    """
    if ticker_vol <= 0:
        return 0.0
    contribution_target = portfolio_vol_target / n_positions
    weight = contribution_target / ticker_vol
    return min(weight, 0.40)   # cap at 40% per position


def render_position_sizing(ticker: str):
    """Render position sizing calculator for a single ticker."""
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;'>"
        f"Calculadora de Sizing — {ticker}</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        capital = st.number_input("Capital total ($)", min_value=1_000, value=100_000,
                                   step=1_000, key=f"ps_capital_{ticker}")
    with col2:
        n_pos = st.number_input("Nº posiciones en cartera", min_value=1, max_value=50,
                                 value=10, key=f"ps_npos_{ticker}")
    with col3:
        vol_target = st.slider("Volatilidad objetivo cartera (%)", 5, 30, 15,
                                key=f"ps_voltgt_{ticker}") / 100

    with st.spinner("Calculando volatilidad..."):
        ret = _fetch_returns(ticker)
        price = _fetch_last_price(ticker)
        atr = _fetch_atr(ticker)

    if ret is None or len(ret) < 30:
        st.warning(f"No hay suficientes datos históricos para {ticker}.")
        return

    ann_vol = float(ret.std() * np.sqrt(252))
    daily_vol = float(ret.std())

    # Kelly requires historical win/loss — we approximate from daily returns
    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    win_rate = len(wins) / len(ret) if len(ret) > 0 else 0.5
    avg_win_daily = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss_daily = float(losses.mean()) if len(losses) > 0 else -0.01

    kelly_f = _kelly_fraction(win_rate, avg_win_daily, abs(avg_loss_daily))
    half_kelly = kelly_f / 2   # half-Kelly is standard practice

    vol_weight = _vol_position_size(capital, ann_vol, vol_target, n_pos)

    # Conservative: min of half-Kelly and vol-based, capped at 1/n_pos * 1.5
    conservative = min(half_kelly, vol_weight, 1.5 / n_pos)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    def _card(title, val, sub="", color=GOLD):
        return (
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:14px 10px; text-align:center;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:5px;'>{title}</div>"
            f"<div style='font-size:20px; font-weight:700; color:{color};'>{val}</div>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:4px;'>{sub}</div>"
            f"</div>"
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card(
            "Volatilidad anual", f"{ann_vol*100:.1f}%",
            "histórica 252d"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_card(
            "Kelly completo", f"{kelly_f*100:.1f}%",
            f"≈ ${capital*kelly_f:,.0f}",
            GOLD_LIGHT
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_card(
            "½ Kelly (recomendado)", f"{half_kelly*100:.1f}%",
            f"≈ ${capital*half_kelly:,.0f}",
            GOLD
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(_card(
            "Sizing conservador", f"{conservative*100:.1f}%",
            f"≈ ${capital*conservative:,.0f}",
            POSITIVE
        ), unsafe_allow_html=True)

    # ATR stop-loss
    if atr and price:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:1px;'>Stop-Loss basado en ATR(14)</p>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        for col, mult, label in zip([c1, c2, c3], [1.5, 2.0, 3.0], ["Ajustado", "Estándar", "Amplio"]):
            stop = price - mult * atr
            stop_pct = (stop - price) / price * 100
            with col:
                st.markdown(_card(
                    f"Stop {label} ({mult}×ATR)",
                    f"${stop:.2f}",
                    f"{stop_pct:+.1f}% desde precio actual",
                    NEGATIVE,
                ), unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:8px;'>"
            f"ATR(14) actual de {ticker}: <b>${atr:.2f}</b> · Precio: <b>${price:.2f}</b>. "
            f"El stop amplio (3×ATR) reduce el ruido en acciones volátiles.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:14px;'>"
        f"⚠ Kelly se basa en rendimientos diarios históricos ({len(ret)} sesiones) "
        f"y asume distribución estacionaria. En activos con alta asimetría o fat tails, "
        f"usar ½ Kelly es prudente. El sizing conservador es el mínimo entre ½ Kelly, "
        f"sizing por volatilidad y 1.5× la posición equi-ponderada.</p>",
        unsafe_allow_html=True,
    )


# ── Portfolio concentration heat map ─────────────────────────────────────────

def render_concentration_heatmap(portfolio_df: pd.DataFrame):
    """
    Render a concentration heat map for the portfolio.
    portfolio_df must have columns: Ticker, Value (market value per position).
    """
    if portfolio_df.empty or "Value" not in portfolio_df.columns:
        st.info("Carga y actualiza tu portfolio para ver el análisis de concentración.")
        return

    df = portfolio_df[["Ticker", "Value"]].dropna()
    df = df[df["Value"] > 0].copy()
    if df.empty:
        st.info("No hay posiciones con valor > 0.")
        return

    total = df["Value"].sum()
    df["Weight"] = df["Value"] / total * 100
    df = df.sort_values("Weight", ascending=False).reset_index(drop=True)

    # Top 5 concentration
    top5 = df.head(5)["Weight"].sum()
    herfindahl = (df["Weight"] / 100).pow(2).sum()   # HHI
    effective_n = 1 / herfindahl if herfindahl > 0 else len(df)

    c1, c2, c3 = st.columns(3)
    def _card(title, val, color=GOLD, sub=""):
        return (
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:12px; text-align:center;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:4px;'>{title}</div>"
            f"<div style='font-size:18px; font-weight:700; color:{color};'>{val}</div>"
            f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:3px;'>{sub}</div>"
            f"</div>"
        )

    with c1:
        color = NEGATIVE if top5 > 60 else (GOLD if top5 > 40 else POSITIVE)
        st.markdown(_card("Concentración top 5", f"{top5:.1f}%",
                           color, "recomendado < 40%"), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("HHI (índice Herfindahl)", f"{herfindahl:.3f}",
                           NEGATIVE if herfindahl > 0.25 else POSITIVE,
                           "< 0.15 = diversificado"), unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Posiciones efectivas", f"{effective_n:.1f}",
                           POSITIVE if effective_n > 8 else NEGATIVE,
                           f"de {len(df)} totales"), unsafe_allow_html=True)

    # Treemap
    fig = px.treemap(
        df, path=["Ticker"], values="Weight",
        color="Weight",
        color_continuous_scale=[[0, "#1e293b"], [0.4, "#d97706"], [1.0, "#ef4444"]],
        custom_data=["Value"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:.1f}%",
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<br>Valor: $%{customdata[0]:,.0f}<extra></extra>",
        textfont_size=13,
    )
    layout = {k: v for k, v in PLOTLY_DARK.items()}
    layout.update(dict(
        height=360,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=False,
        title=dict(text="Mapa de concentración de cartera", font=dict(size=13, color=TEXT_MUTED)),
    ))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # Weight table
    with st.expander("Ver pesos detallados"):
        display_df = df[["Ticker", "Value", "Weight"]].copy()
        display_df.columns = ["Ticker", "Valor ($)", "Peso (%)"]
        display_df["Valor ($)"] = display_df["Valor ($)"].apply(lambda x: f"${x:,.0f}")
        display_df["Peso (%)"] = display_df["Peso (%)"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Rebalancing drift alerts ──────────────────────────────────────────────────

def render_drift_alerts(portfolio_df: pd.DataFrame, drift_threshold: float = 0.20):
    """
    Show alerts when a position has drifted more than drift_threshold from target.
    Target = equal weight (1 / n_positions).
    portfolio_df must have: Ticker, Value
    """
    if portfolio_df.empty or "Value" not in portfolio_df.columns:
        return

    df = portfolio_df[["Ticker", "Value"]].dropna()
    df = df[df["Value"] > 0].copy()
    if len(df) < 2:
        return

    n = len(df)
    total = df["Value"].sum()
    df["Weight"] = df["Value"] / total
    df["Target"] = 1.0 / n
    df["Drift"] = df["Weight"] - df["Target"]
    df["Drift_abs"] = df["Drift"].abs()

    alerts = df[df["Drift_abs"] > drift_threshold].sort_values("Drift_abs", ascending=False)

    if alerts.empty:
        st.success(
            f"✅ Todas las posiciones están dentro del ±{drift_threshold*100:.0f}% "
            f"del peso objetivo ({100/n:.1f}% equi-ponderado)."
        )
        return

    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{NEGATIVE}; "
        f"text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>"
        f"⚠ {len(alerts)} posición(es) con drift > {drift_threshold*100:.0f}%</p>",
        unsafe_allow_html=True,
    )

    for _, row in alerts.iterrows():
        action = "REDUCIR" if row["Drift"] > 0 else "AUMENTAR"
        color = NEGATIVE if row["Drift"] > 0 else POSITIVE
        target_val = total * row["Target"]
        current_val = row["Value"]
        delta_val = target_val - current_val

        st.markdown(
            f"<div style='background:{SURFACE_2}; border-left:3px solid {color}; "
            f"border-radius:0 6px 6px 0; padding:10px 14px; margin-bottom:8px; "
            f"display:flex; justify-content:space-between; align-items:center;'>"
            f"<div>"
            f"  <span style='font-weight:700; color:{TEXT_PRIMARY};'>{row['Ticker']}</span> "
            f"  <span style='color:{TEXT_MUTED}; font-size:11px;'>— peso actual: "
            f"<b style='color:{TEXT_PRIMARY}'>{row['Weight']*100:.1f}%</b> "
            f"vs objetivo: {row['Target']*100:.1f}%</span>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"  <span style='font-weight:700; color:{color}; font-size:13px;'>{action}</span> "
            f"  <span style='color:{TEXT_MUTED}; font-size:11px;'>${abs(delta_val):,.0f}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Rebalancing cost estimate
    rebal_amount = sum(abs(total * row["Target"] - row["Value"]) for _, row in df.iterrows()) / 2
    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:10px;'>"
        f"Coste estimado de rebalanceo (0.1% tx): "
        f"<b>${rebal_amount * 0.001:,.0f}</b> sobre ${rebal_amount:,.0f} en operaciones.</p>",
        unsafe_allow_html=True,
    )


# ── Main render (called from app.py / portfolio.py) ──────────────────────────

def render_portfolio_risk_tab(portfolio_df: pd.DataFrame):
    """
    Master render function for the Portfolio Risk & Sizing tab.
    portfolio_df: the user's portfolio DataFrame with Ticker + Value columns.
    Lazy-loads on demand to avoid SEGV from eager yf.download on page render.
    """
    # ── Lazy load guard ───────────────────────────────────────────────────────
    if not st.session_state.get("risk_tab_loaded", False):
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.info("El análisis de riesgo se calcula bajo demanda para no ralentizar la carga.")
            if st.button("📊 Cargar Análisis de Riesgo", type="primary", use_container_width=True):
                st.session_state.risk_tab_loaded = True
                st.rerun()
        return

    # ── Render completo ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Concentración",
        "⚖️ Sizing & Stop-Loss",
        "🔔 Alertas de Drift",
    ])

    with tabs[0]:
        render_concentration_heatmap(portfolio_df)

    with tabs[1]:
        equities = []
        if not portfolio_df.empty and "Ticker" in portfolio_df.columns:
            equities = [
                t for t in portfolio_df["Ticker"].dropna().unique()
                if str(t).strip()
            ]

        if not equities:
            st.info("Añade posiciones al portfolio para usar la calculadora de sizing.")
        else:
            selected = st.selectbox(
                "Selecciona ticker para calcular sizing",
                options=equities,
                key="ps_ticker_select",
            )
            total_capital = portfolio_df["Value"].sum() if "Value" in portfolio_df.columns else 100_000
            if pd.isna(total_capital) or total_capital <= 0:
                total_capital = 100_000
            render_position_sizing(selected)

    with tabs[2]:
        threshold = st.slider("Umbral de alerta de drift (%)", 5, 40, 20,
                               key="drift_threshold") / 100
        render_drift_alerts(portfolio_df, drift_threshold=threshold)

"""
backtesting.py — Signal-based backtesting engine for WealthView
Simulates buy/sell/hold decisions based on technical + fundamental signals
over a historical window and compares results vs. SPY benchmark.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK, plotly_layout
)


# ── Signal computation ────────────────────────────────────────────────────────

def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and generate a composite signal column.
    Signal: +1 = BUY, -1 = SELL, 0 = HOLD (position maintained)
    """
    close = df["Close"].squeeze()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal_line"]

    # SMA 50 / 200
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    # Volume MA (20)
    if "Volume" in df.columns:
        df["Vol_MA"] = df["Volume"].rolling(20).mean()
    else:
        df["Vol_MA"] = np.nan

    # ── Composite signal logic ────────────────────────────────────────────────
    # BUY conditions (need >= 2 out of 4):
    #   1. RSI < 40 (oversold / entering)
    #   2. MACD crossover above signal line (histogram turns positive)
    #   3. Price crossed above SMA50
    #   4. Price near BB lower band (< BB_lower * 1.02)
    # SELL conditions (need >= 2 out of 4):
    #   1. RSI > 65
    #   2. MACD crosses below signal line
    #   3. Price crossed below SMA50
    #   4. Price near BB upper band (> BB_upper * 0.98)

    rsi = df["RSI"]
    macd_hist = df["MACD_hist"]
    macd_hist_prev = macd_hist.shift(1)
    sma50 = df["SMA50"]
    sma50_prev = sma50.shift(1)
    close_prev = close.shift(1)

    buy_1 = (rsi < 40).astype(int)
    buy_2 = ((macd_hist > 0) & (macd_hist_prev <= 0)).astype(int)
    buy_3 = ((close > sma50) & (close_prev <= sma50_prev)).astype(int)
    buy_4 = (close < df["BB_lower"] * 1.02).astype(int)

    sell_1 = (rsi > 65).astype(int)
    sell_2 = ((macd_hist < 0) & (macd_hist_prev >= 0)).astype(int)
    sell_3 = ((close < sma50) & (close_prev >= sma50_prev)).astype(int)
    sell_4 = (close > df["BB_upper"] * 0.98).astype(int)

    buy_score = buy_1 + buy_2 + buy_3 + buy_4
    sell_score = sell_1 + sell_2 + sell_3 + sell_4

    signal = pd.Series(0, index=df.index)
    signal[buy_score >= 2] = 1
    signal[sell_score >= 2] = -1
    # When both trigger, net wins
    signal[(buy_score >= 2) & (sell_score >= 2)] = 0
    df["Signal"] = signal

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(ticker: str, years: int = 10) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df


def _simulate(df: pd.DataFrame,
              tx_cost: float = 0.001,
              slippage: float = 0.001) -> tuple:
    """
    Simulate a long-only strategy with realistic costs:
    - Enter (buy) when Signal == +1, exit (sell) when Signal == -1
    - tx_cost: commission as fraction of trade value (default 0.1%)
    - slippage: market impact / bid-ask spread (default 0.1%)
    - Starting capital: 10,000
    """
    capital = 10_000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = []
    total_cost = tx_cost + slippage   # effective round-trip cost per leg

    close = df["Close"].squeeze()
    signals = df["Signal"]
    in_market = False

    for i, (dt, row) in enumerate(df.iterrows()):
        price = float(close.iloc[i])
        sig = int(signals.iloc[i])

        if not in_market and sig == 1 and capital > 0:
            # BUY: pay tx_cost + slippage on entry
            effective_buy = price * (1 + total_cost)
            shares = capital / effective_buy
            position = shares
            entry_price = effective_buy
            capital = 0.0
            in_market = True
            trades.append({
                "date": dt, "action": "BUY", "price": price,
                "effective_price": effective_buy, "shares": shares,
                "cost_pct": total_cost * 100,
            })

        elif in_market and sig == -1:
            # SELL: pay tx_cost + slippage on exit
            effective_sell = price * (1 - total_cost)
            proceeds = position * effective_sell
            pnl_pct = (effective_sell - entry_price) / entry_price * 100
            pnl_abs = proceeds - (position * entry_price)
            trades.append({
                "date": dt, "action": "SELL", "price": price,
                "effective_price": effective_sell, "shares": position,
                "pnl_pct": pnl_pct, "pnl_abs": pnl_abs,
                "cost_pct": total_cost * 100,
            })
            capital = proceeds
            position = 0.0
            in_market = False

        # Mark-to-market
        if in_market:
            equity.append(capital + position * price)
        else:
            equity.append(capital)

    # Close any open position at last price
    if in_market and len(equity) > 0:
        last_price = float(close.iloc[-1])
        effective_close = last_price * (1 - total_cost)
        equity[-1] = position * effective_close

    df = df.copy()
    df["Equity"] = equity
    return df, trades


def _compute_stats(df: pd.DataFrame, equity_col: str,
                   start_capital: float = 10_000,
                   risk_free: float = 0.04) -> dict:
    """
    Compute comprehensive performance statistics.
    Metrics: CAGR, Sharpe, Sortino, Calmar, Omega, Max DD, Avg DD, Ulcer Index,
             Volatility, Win Rate, Profit Factor.
    """
    eq = df[equity_col].dropna()
    if len(eq) < 2:
        return {}

    returns = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] - start_capital) / start_capital
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25

    # Basic
    cagr = (eq.iloc[-1] / start_capital) ** (1 / max(n_years, 0.1)) - 1
    vol  = float(returns.std() * np.sqrt(252))

    # Sharpe (annualized, daily rf)
    rf_daily = (1 + risk_free) ** (1/252) - 1
    excess   = returns - rf_daily
    sharpe   = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Sortino (downside deviation only)
    down_ret = returns[returns < rf_daily]
    down_std = float(down_ret.std() * np.sqrt(252)) if len(down_ret) > 0 else vol
    sortino  = float(excess.mean() / (down_std / np.sqrt(252)) * np.sqrt(252)) if down_std > 0 else 0.0

    # Drawdown series
    rolling_max = eq.expanding().max()
    drawdown    = (eq - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())
    avg_dd      = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    # Calmar = CAGR / |Max DD|
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Ulcer Index = RMS of drawdown percentages (penalizes depth AND duration)
    ulcer = float(np.sqrt(np.mean(drawdown ** 2))) * 100

    # Omega ratio (threshold = risk_free daily)
    gains  = (returns[returns > rf_daily] - rf_daily).sum()
    losses = (rf_daily - returns[returns <= rf_daily]).sum()
    omega  = float(gains / losses) if losses > 0 else float("inf")

    # Max drawdown duration
    in_dd = False
    dd_start = None
    max_dd_days = 0
    for i, (dt, val) in enumerate(drawdown.items()):
        if val < 0 and not in_dd:
            in_dd = True
            dd_start = dt
        elif val >= 0 and in_dd:
            dur = (dt - dd_start).days
            max_dd_days = max(max_dd_days, dur)
            in_dd = False
    if in_dd and dd_start is not None:
        max_dd_days = max(max_dd_days, (drawdown.index[-1] - dd_start).days)

    return {
        "total_return":  total_return,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "omega":         omega,
        "max_drawdown":  max_dd,
        "avg_drawdown":  avg_dd,
        "max_dd_days":   max_dd_days,
        "ulcer_index":   ulcer,
        "volatility":    vol,
        "n_years":       n_years,
        "drawdown_series": drawdown,
    }


# ── Main render function ──────────────────────────────────────────────────────

def render_backtest(ticker: str):
    from modules.styles import GOLD, SURFACE, SURFACE_2, TEXT_PRIMARY, TEXT_MUTED, POSITIVE, NEGATIVE, plotly_layout

    st.markdown(
        f"<p style='font-size:11px; color:{TEXT_MUTED}; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:12px;'>Backtesting · {ticker}</p>",
        unsafe_allow_html=True,
    )

    col_years, col_cost, col_slip, col_run = st.columns([2, 2, 2, 2])
    with col_years:
        years = st.selectbox("Horizonte histórico",
                             [1, 3, 5, 7, 10], index=4, key="bt_years",
                             help="10 años es el estándar para evaluar ciclos completos de mercado.")
    with col_cost:
        tx_pct = st.number_input("Comisión por operación (%)", min_value=0.0, max_value=1.0,
                                  value=0.10, step=0.05, key="bt_tx",
                                  help="Broker online: 0.05-0.15%. Broker tradicional: 0.20-0.50%.") / 100
    with col_slip:
        slip_pct = st.number_input("Slippage (%)", min_value=0.0, max_value=0.5,
                                    value=0.05, step=0.025, key="bt_slip",
                                    help="Impacto de mercado y spread bid-ask. Típico: 0.02-0.10% en líquidos.") / 100
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶  Ejecutar backtest", use_container_width=True, key="bt_run")

    if not run:
        st.markdown(
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:24px; text-align:center; color:{TEXT_MUTED}; margin-top:12px;'>"
            f"Configura los parámetros y haz clic en <b>Ejecutar backtest</b> para simular la estrategia técnica.</div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner(f"Descargando {years} años de datos para {ticker} y SPY..."):
        df_raw = _fetch_history(ticker, years)
        df_spy = _fetch_history("SPY", years)

    if df_raw.empty:
        st.error(f"No se pudieron obtener datos históricos para {ticker}.")
        return

    # Align dates
    start = max(df_raw.index[0], df_spy.index[0]) if not df_spy.empty else df_raw.index[0]
    df_raw = df_raw[df_raw.index >= start]
    if not df_spy.empty:
        df_spy = df_spy[df_spy.index >= start]

    with st.spinner("Calculando señales y simulando cartera..."):
        df = _compute_signals(df_raw.copy())
        df, trades = _simulate(df, tx_cost=tx_pct, slippage=slip_pct)

        # SPY buy-and-hold
        if not df_spy.empty:
            spy_close = df_spy["Close"].squeeze().reindex(df.index, method="ffill")
            df["SPY_bh"] = 10_000 * (spy_close / spy_close.iloc[0])
        else:
            df["SPY_bh"] = None

        # Ticker buy-and-hold baseline
        ticker_close = df["Close"].squeeze()
        df["Ticker_bh"] = 10_000 * (ticker_close / ticker_close.iloc[0])

    stats_strat = _compute_stats(df, "Equity")
    stats_bh = _compute_stats(df, "Ticker_bh")
    stats_spy = _compute_stats(df, "SPY_bh") if "SPY_bh" in df.columns and df["SPY_bh"].notna().any() else {}

    # ── KPI cards ─────────────────────────────────────────────────────────────
    def _card(label, value, color=None):
        col_color = color or TEXT_PRIMARY
        return (
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-radius:8px; padding:14px 10px; text-align:center;'>"
            f"<div style='font-size:9px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:6px;'>{label}</div>"
            f"<div style='font-size:18px; font-weight:700; color:{col_color};'>{value}</div>"
            f"</div>"
        )

    def _pct_str(v):
        if v is None:
            return "N/D"
        return f"{v*100:+.1f}%"

    def _color(v):
        if v is None:
            return TEXT_MUTED
        return POSITIVE if v >= 0 else NEGATIVE

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Row 1: primary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_r1 = [
        ("Retorno total", _pct_str(stats_strat.get("total_return")),                _color(stats_strat.get("total_return"))),
        ("CAGR",          _pct_str(stats_strat.get("cagr")),                        _color(stats_strat.get("cagr"))),
        ("Sharpe",        f"{stats_strat.get('sharpe', 0):.2f}",                    GOLD if stats_strat.get('sharpe',0)>=1 else TEXT_MUTED),
        ("Máx. Drawdown", _pct_str(stats_strat.get("max_drawdown")),                NEGATIVE),
        ("Volatilidad",   _pct_str(stats_strat.get("volatility")),                  TEXT_MUTED),
    ]
    for col, (lbl, val, clr) in zip([c1, c2, c3, c4, c5], metrics_r1):
        with col:
            st.markdown(_card(lbl, val, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row 2: advanced risk metrics
    d1, d2, d3, d4, d5 = st.columns(5)
    omega_val = stats_strat.get("omega", 0)
    omega_str = f"{omega_val:.2f}" if omega_val < 99 else "∞"
    dd_days = stats_strat.get("max_dd_days", 0)
    metrics_r2 = [
        ("Sortino",        f"{stats_strat.get('sortino', 0):.2f}",                  GOLD if stats_strat.get('sortino',0)>=1 else TEXT_MUTED),
        ("Calmar",         f"{stats_strat.get('calmar', 0):.2f}",                   POSITIVE if stats_strat.get('calmar',0)>0.5 else NEGATIVE),
        ("Omega ratio",    omega_str,                                                POSITIVE if omega_val > 1 else NEGATIVE),
        ("Ulcer Index",    f"{stats_strat.get('ulcer_index', 0):.1f}",              NEGATIVE if stats_strat.get('ulcer_index',0)>10 else TEXT_MUTED),
        ("DD más largo",   f"{dd_days}d",                                           TEXT_MUTED),
    ]
    for col, (lbl, val, clr) in zip([d1, d2, d3, d4, d5], metrics_r2):
        with col:
            st.markdown(_card(lbl, val, clr), unsafe_allow_html=True)

    # ── Equity curve chart ────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity"],
        name=f"Estrategia WealthView ({ticker})",
        line=dict(color=GOLD, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Ticker_bh"],
        name=f"Buy & Hold {ticker}",
        line=dict(color="#60a5fa", width=1.5, dash="dot"),
    ))
    if "SPY_bh" in df.columns and df["SPY_bh"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SPY_bh"],
            name="SPY (benchmark)",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
        ))

    # Mark buy/sell trades
    buy_dates = [t["date"] for t in trades if t["action"] == "BUY"]
    sell_dates = [t["date"] for t in trades if t["action"] == "SELL"]
    if buy_dates:
        buy_eq = df.loc[df.index.isin(buy_dates), "Equity"]
        fig.add_trace(go.Scatter(
            x=buy_eq.index, y=buy_eq.values,
            mode="markers", name="Compra",
            marker=dict(symbol="triangle-up", size=10, color=POSITIVE),
        ))
    if sell_dates:
        sell_eq = df.loc[df.index.isin(sell_dates), "Equity"]
        fig.add_trace(go.Scatter(
            x=sell_eq.index, y=sell_eq.values,
            mode="markers", name="Venta",
            marker=dict(symbol="triangle-down", size=10, color=NEGATIVE),
        ))

    layout = {k: v for k, v in PLOTLY_DARK.items()}
    layout.update(dict(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Valor de la cartera ($)", tickprefix="$"),
        xaxis=dict(title=""),
        hovermode="x unified",
        title=dict(text=f"Curva de capital — {years} años", font=dict(size=13, color=TEXT_MUTED)),
    ))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Drawdown chart ────────────────────────────────────────────────────────
    if "drawdown_series" in stats_strat:
        dd_series = stats_strat["drawdown_series"] * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values,
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.15)",
            line=dict(color=NEGATIVE, width=1),
            name="Drawdown estrategia",
            hovertemplate="%{x|%d %b %Y}<br>DD: %{y:.1f}%<extra></extra>",
        ))
        if "drawdown_series" in stats_spy:
            spy_dd = stats_spy["drawdown_series"] * 100
            fig_dd.add_trace(go.Scatter(
                x=spy_dd.index, y=spy_dd.values,
                line=dict(color="#6b7280", width=1, dash="dot"),
                name="Drawdown SPY",
                hovertemplate="%{x|%d %b %Y}<br>SPY DD: %{y:.1f}%<extra></extra>",
            ))
        fig_dd.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_dd.update_layout(**plotly_layout(
            height=180,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
            title=dict(text="Underwater chart (drawdown)", font=dict(size=11, color=TEXT_MUTED)),
            legend=dict(orientation="h", y=1.1, font=dict(size=9, color=TEXT_MUTED)),
        ))
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    comparison = []
    for label, s in [
        (f"Estrategia ({ticker})", stats_strat),
        (f"Buy & Hold {ticker}", stats_bh),
        ("SPY Buy & Hold", stats_spy),
    ]:
        if s:
            omega_v = s.get("omega", 0)
            comparison.append({
                "Estrategia":   label,
                "Retorno":      f"{s.get('total_return', 0)*100:+.1f}%",
                "CAGR":         f"{s.get('cagr', 0)*100:+.1f}%",
                "Sharpe":       f"{s.get('sharpe', 0):.2f}",
                "Sortino":      f"{s.get('sortino', 0):.2f}",
                "Calmar":       f"{s.get('calmar', 0):.2f}",
                "Omega":        f"{omega_v:.2f}" if omega_v < 99 else "∞",
                "Máx. DD":      f"{s.get('max_drawdown', 0)*100:.1f}%",
                "Ulcer":        f"{s.get('ulcer_index', 0):.1f}",
                "Volat.":       f"{s.get('volatility', 0)*100:.1f}%",
            })

    if comparison:
        cdf = pd.DataFrame(comparison).set_index("Estrategia")
        st.dataframe(cdf, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    if trades:
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        n_trades = len(sell_trades)
        if n_trades > 0:
            wins = sum(1 for t in sell_trades if t.get("pnl_pct", 0) > 0)
            win_rate = wins / n_trades * 100
            avg_win = np.mean([t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) > 0] or [0])
            avg_loss = np.mean([t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) <= 0] or [0])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(_card("Operaciones cerradas", str(n_trades)), unsafe_allow_html=True)
            with c2:
                st.markdown(_card("Win rate", f"{win_rate:.0f}%",
                                   POSITIVE if win_rate > 50 else NEGATIVE), unsafe_allow_html=True)
            with c3:
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
                st.markdown(_card("Profit factor",
                                   f"{profit_factor:.2f}" if profit_factor < 99 else "∞",
                                   POSITIVE if profit_factor > 1 else NEGATIVE), unsafe_allow_html=True)

        with st.expander(f"Ver registro de operaciones ({len(trades)} señales)"):
            tdf = pd.DataFrame(trades)
            if "date" in tdf.columns:
                tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d")
            if "pnl_pct" in tdf.columns:
                tdf["pnl_pct"] = tdf["pnl_pct"].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
                )
            tdf.columns = [c.replace("_", " ").title() for c in tdf.columns]
            st.dataframe(tdf, use_container_width=True)

    # Disclaimer
    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-top:12px;'>"
        "⚠️ Rendimientos pasados no garantizan resultados futuros. "
        "Esta simulación es puramente técnica (RSI, MACD, SMA50, Bollinger Bands) "
        "y no incorpora datos fundamentales históricos. Coste por operación aplicado: "
        f"{tx_pct*100:.2f}%.</p>",
        unsafe_allow_html=True,
    )
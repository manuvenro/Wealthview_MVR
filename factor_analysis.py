"""
modules/factor_analysis.py — Factor Analysis & Advanced Backtesting
====================================================================
Análisis cuantitativo avanzado del portfolio:

1. Fama-French 3 factores (MKT-RF, SMB, HML) via OLS
   → alpha, betas, R², explicación del retorno
2. Rolling metrics (ventana 12 meses)
   → Sharpe, volatilidad, beta vs SPY, correlación
3. Underwater drawdown chart
   → períodos bajo el máximo histórico
4. Heatmap de retornos mensuales
   → matriz Año × Mes con colores rojo/verde
5. Calendar year returns
   → retorno anual vs benchmark
6. Estadísticas de distribución de retornos
   → percentiles, VaR, CVaR histórico

Fuente de factores F-F: Kenneth French Data Library (Dartmouth)
Fallback CAPM si descarga falla.
"""

import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.figure_factory as ff_fig
from scipy import stats as sp_stats

from modules.styles import (
    GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG,
    PLOTLY_DARK, section_label, gold_divider, page_header, plotly_layout
)
from modules.utils import ensure_portfolio_data


# ─────────────────────────────────────────────────────────────────────────────
# Fama-French data
# ─────────────────────────────────────────────────────────────────────────────

_FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

@st.cache_data(ttl=86400, show_spinner=False)  # 24h cache
def _fetch_ff_factors() -> pd.DataFrame | None:
    """
    Descarga factores Fama-French 3F diarios desde Dartmouth.
    Columnas: Mkt-RF, SMB, HML, RF  (en decimales, no %)
    Índice: DatetimeIndex
    Devuelve None si la descarga falla.
    """
    try:
        resp = requests.get(_FF_URL, timeout=15)
        if not resp.ok:
            return None
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        name = [n for n in z.namelist() if n.endswith(".CSV")][0]
        raw  = z.read(name).decode("utf-8", errors="ignore")

        lines = raw.splitlines()
        # Find the header row (contains "Mkt-RF")
        header_idx = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
        # Find the footer (blank line or "Annual" block)
        data_lines = []
        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                break
            data_lines.append(stripped)

        from io import StringIO
        df = pd.read_csv(
            StringIO("Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(data_lines)),
            parse_dates=["Date"],
            date_format="%Y%m%d",
        )
        df = df.dropna().set_index("Date")
        df = df.apply(pd.to_numeric, errors="coerce") / 100.0  # % → decimal
        return df.sort_index()

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Price & returns fetch
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(tickers: tuple, period: str = "10y") -> pd.DataFrame:
    """Adjusted close prices for a tuple of tickers."""
    raw = yf.download(list(tickers), period=period,
                      auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    return prices.dropna(how="all")


def _portfolio_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Weighted portfolio daily log-returns.
    weights: {ticker: float} (sum should be 1)
    """
    w = np.array([weights.get(t, 0) for t in prices.columns])
    w = w / w.sum() if w.sum() > 0 else w
    log_rets = np.log(prices / prices.shift(1)).dropna()
    port_ret  = log_rets @ w
    port_ret.name = "Portfolio"
    return port_ret


# ─────────────────────────────────────────────────────────────────────────────
# Factor regression (OLS)
# ─────────────────────────────────────────────────────────────────────────────

def run_factor_regression(portfolio_rets: pd.Series,
                          ff_factors: pd.DataFrame | None) -> dict:
    """
    Regresiona retornos del portfolio sobre factores F-F.
    Si ff_factors es None, usa CAPM (1 factor) con SPY como proxy.

    Devuelve dict con:
      alpha_ann, alpha_pval, mkt_beta, smb_beta, hml_beta,
      r_squared, adj_r_squared, te, ir, model ('FF3' | 'CAPM')
    """
    from scipy import stats as sp_stats

    r = portfolio_rets.dropna()

    if ff_factors is not None:
        # Alinear con factores F-F
        aligned = pd.concat([r, ff_factors], axis=1).dropna()
        if len(aligned) < 60:
            ff_factors = None  # insuficientes datos solapados
        else:
            y = (aligned["Portfolio"] - aligned["RF"]).values
            X = aligned[["Mkt-RF", "SMB", "HML"]].values
            X_const = np.column_stack([np.ones(len(y)), X])
            # Near-singular check via condition number
            cond_num = np.linalg.cond(X_const)
            if cond_num > 1e10:
                ff_factors = None  # treat as unavailable — fall back to CAPM
            else:
                pass
            beta, residuals, rank, sv = np.linalg.lstsq(X_const, y, rcond=None)
            y_hat = X_const @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            n, k = len(y), X_const.shape[1] - 1
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            # t-stats for alpha
            mse = ss_res / (n - k - 1)
            XtX_inv = np.linalg.pinv(X_const.T @ X_const)
            se = np.sqrt(np.diag(XtX_inv * mse))
            t_alpha = beta[0] / se[0] if se[0] > 0 else 0
            p_alpha = 2 * (1 - sp_stats.t.cdf(abs(t_alpha), df=n - k - 1))
            te = float(np.std(y - y_hat)) * np.sqrt(252)
            ir = float(beta[0]) * 252 / te if te > 0 else 0.0

            return {
                "model":       "FF3",
                "alpha_daily": float(beta[0]),
                "alpha_ann":   float(beta[0]) * 252,
                "alpha_pval":  float(p_alpha),
                "mkt_beta":    float(beta[1]),
                "smb_beta":    float(beta[2]),
                "hml_beta":    float(beta[3]),
                "r_squared":   float(r2),
                "adj_r2":      float(adj_r2),
                "te":          te,
                "ir":          ir,
                "n_obs":       n,
            }

    # ── Fallback: CAPM con SPY ─────────────────────────────────────────────
    try:
        spy_raw = yf.download("SPY", period="max", auto_adjust=True,
                              progress=False, threads=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_prices = spy_raw["Close"]["SPY"]
        else:
            spy_prices = spy_raw["Close"]
        spy_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()
        aligned = pd.concat([r, spy_ret], axis=1).dropna()
        aligned.columns = ["port", "spy"]
        y = aligned["port"].values
        x = aligned["spy"].values
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
        y_hat = slope * x + intercept
        te = float(np.std(y - y_hat)) * np.sqrt(252)
        ir = intercept * 252 / te if te > 0 else 0.0
        return {
            "model":       "CAPM",
            "alpha_daily": float(intercept),
            "alpha_ann":   float(intercept) * 252,
            "alpha_pval":  float(p_val),
            "mkt_beta":    float(slope),
            "smb_beta":    None,
            "hml_beta":    None,
            "r_squared":   float(r_val ** 2),
            "adj_r2":      float(r_val ** 2),
            "te":          te,
            "ir":          ir,
            "n_obs":       len(y),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Rolling metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling(portfolio_rets: pd.Series,
                    benchmark_rets: pd.Series | None = None,
                    window: int = 252) -> pd.DataFrame:
    """
    Rolling Sharpe, volatility, beta, correlation — all annualized.
    window: días (252 = 1 año)
    """
    r = portfolio_rets.dropna()
    rolling = pd.DataFrame(index=r.index)
    rolling["vol"]    = r.rolling(window).std() * np.sqrt(252) * 100
    rolling["sharpe"] = (r.rolling(window).mean() / r.rolling(window).std()
                         ) * np.sqrt(252)
    if benchmark_rets is not None:
        b = benchmark_rets.reindex(r.index).dropna()
        aligned = pd.concat([r, b], axis=1).dropna()
        aligned.columns = ["port", "bench"]
        roll_cov  = aligned["port"].rolling(window).cov(aligned["bench"])
        roll_var  = aligned["bench"].rolling(window).var()
        rolling["beta"] = roll_cov / roll_var.replace(0, np.nan)
        rolling["corr"] = aligned["port"].rolling(window).corr(aligned["bench"])
    return rolling.dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown series
# ─────────────────────────────────────────────────────────────────────────────

def compute_drawdown(portfolio_rets: pd.Series) -> pd.Series:
    """Return drawdown series (0 to -1) from log returns."""
    equity = np.exp(portfolio_rets.cumsum())
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return dd


# ─────────────────────────────────────────────────────────────────────────────
# Monthly returns pivot
# ─────────────────────────────────────────────────────────────────────────────

def monthly_returns_table(portfolio_rets: pd.Series) -> pd.DataFrame:
    """Return pivot table: rows=Year, cols=Jan..Dec, values=monthly return %"""
    monthly = portfolio_rets.resample("ME").sum() * 100
    tbl = monthly.to_frame("ret")
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    month_names = ["Ene","Feb","Mar","Abr","May","Jun",
                   "Jul","Ago","Sep","Oct","Nov","Dic"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    pivot.index.name = "Año"
    # Annual total
    pivot["Total"] = pivot.sum(axis=1, skipna=False)
    return pivot.sort_index(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stat_card(label: str, value: str, color: str = TEXT_PRIMARY,
               sub: str = "", border_color: str = BORDER_SOFT) -> str:
    sub_html = (f"<div style='font-size:10px;color:{TEXT_MUTED};"
                f"margin-top:3px;line-height:1.4'>{sub}</div>") if sub else ""
    return (
        f"<div style='background:{SURFACE};border:1px solid {border_color};"
        f"border-radius:10px;padding:14px 16px;height:100%;'>"
        f"<div style='font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:6px;'>{label}</div>"
        f"<div style='font-size:20px;font-weight:700;color:{color};'>{value}</div>"
        f"{sub_html}</div>"
    )


def _significance_badge(p: float) -> str:
    if p < 0.01:   return f"<span style='color:{POSITIVE};font-weight:700'>★★★ p&lt;0.01</span>"
    if p < 0.05:   return f"<span style='color:{POSITIVE}'>★★ p&lt;0.05</span>"
    if p < 0.10:   return f"<span style='color:{GOLD}'>★ p&lt;0.10</span>"
    return f"<span style='color:{TEXT_MUTED}'>no significativo (p={p:.2f})</span>"


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_factor_analysis():
    page_header("Análisis de Factores", "Fama-French · Rolling metrics · Drawdown · Heatmap")

    # ── Portfolio data ────────────────────────────────────────────────────────
    portfolio_data = ensure_portfolio_data()
    if portfolio_data is None or portfolio_data.empty:
        from modules.utils import no_portfolio_warning
        no_portfolio_warning()
        return

    # Asegurar que la columna 'Asset Type' existe (puede faltar si viene del dashboard)
    if "Asset Type" not in portfolio_data.columns:
        portfolio_data = portfolio_data.copy()
        portfolio_data["Asset Type"] = "equity"
    # Solo equity/ETF/crypto
    df_eq = portfolio_data[
        portfolio_data["Asset Type"].fillna("equity").str.lower().isin(
            ["equity", "etf", "crypto", "fund", ""]
        )
    ].copy()

    tickers = [t for t in df_eq["Ticker"].unique() if t]
    if not tickers:
        st.warning("No hay activos con historial de precios en el portfolio.")
        return

    # ── Parámetros ────────────────────────────────────────────────────────────
    col_p, col_r, col_b = st.columns(3)
    with col_p:
        period = st.selectbox("Período histórico", ["3y", "5y", "10y", "15y", "20y", "max"],
                              index=2, key="fa_period")
    with col_r:
        roll_window = st.selectbox("Ventana rolling", ["3 meses (63d)", "6 meses (126d)",
                                                        "12 meses (252d)", "24 meses (504d)"],
                                   index=2, key="fa_roll")
        win_map = {"3 meses (63d)": 63, "6 meses (126d)": 126,
                   "12 meses (252d)": 252, "24 meses (504d)": 504}
        window = win_map[roll_window]
    with col_b:
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "EEM", "AGG"],
                                 key="fa_bench")

    # ── Fetch prices ──────────────────────────────────────────────────────────
    fetch_tickers = tuple(sorted(set(tickers + [benchmark])))
    with st.spinner(f"Descargando {len(fetch_tickers)} series — {period}..."):
        prices = _fetch_prices(fetch_tickers, period=period)

    if prices.empty:
        st.error("No se pudieron obtener datos de precios.")
        return

    available = [t for t in tickers if t in prices.columns]
    if not available:
        st.error("Ningún ticker del portfolio tiene datos en el período seleccionado.")
        return

    dropped = [t for t in tickers if t not in available]
    if dropped:
        st.caption(f"⚠ Sin datos suficientes (excluidos): {', '.join(dropped)}")

    # Pesos del portfolio
    total_val = df_eq.groupby("Ticker")["Total Value ($)"].sum()
    total_sum  = total_val.reindex(available).fillna(0).sum()
    weights    = {t: float(total_val.get(t, 0)) / total_sum for t in available} \
                 if total_sum > 0 else {t: 1 / len(available) for t in available}

    port_rets  = _portfolio_returns(prices[available], weights)
    bench_rets = (np.log(prices[benchmark] / prices[benchmark].shift(1)).dropna()
                  if benchmark in prices.columns else None)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ff, tab_rolling, tab_dd, tab_heatmap = st.tabs([
        "Factores Fama-French", "Métricas Rolling", "Drawdown", "Heatmap Retornos"
    ])

    # ── Tab 1: Fama-French ────────────────────────────────────────────────────
    with tab_ff:
        section_label("Modelo de Factores — Fama-French 3F")

        with st.spinner("Descargando factores Fama-French..."):
            ff = _fetch_ff_factors()

        if ff is None:
            st.caption("⚠ No se pudieron descargar los factores F-F (red). Usando CAPM con SPY.")

        with st.spinner("Ejecutando regresión OLS..."):
            reg = run_factor_regression(port_rets, ff)

        if not reg:
            st.error("Error en la regresión. Datos insuficientes.")
        else:
            model_label = "Fama-French 3 Factores" if reg["model"] == "FF3" else "CAPM (1 factor)"
            st.markdown(
                f"<div style='font-size:10px;color:{TEXT_MUTED};margin-bottom:16px;'>"
                f"Modelo: <b style='color:{GOLD}'>{model_label}</b> · "
                f"N = {reg['n_obs']:,} días · "
                f"R² = {reg['r_squared']:.3f} · "
                f"R² ajustado = {reg['adj_r2']:.3f}</div>",
                unsafe_allow_html=True,
            )

            # ── Alpha ─────────────────────────────────────────────────────────
            alpha_sign = "+" if reg["alpha_ann"] >= 0 else ""
            alpha_color = POSITIVE if reg["alpha_ann"] >= 0 else NEGATIVE
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(_stat_card(
                    "Alpha anualizado",
                    f"{alpha_sign}{reg['alpha_ann']*100:.2f}%",
                    color=alpha_color,
                    sub=f"Significancia: {_significance_badge(reg['alpha_pval'])}",
                    border_color=alpha_color + "66",
                ), unsafe_allow_html=True)
            with c2:
                beta_color = GOLD if 0.6 <= reg["mkt_beta"] <= 1.2 else TEXT_PRIMARY
                st.markdown(_stat_card(
                    "Beta mercado (MKT)",
                    f"{reg['mkt_beta']:.3f}",
                    color=beta_color,
                    sub="&lt;1 más defensivo · &gt;1 más agresivo",
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(_stat_card(
                    "Tracking Error",
                    f"{reg['te']*100:.2f}%",
                    color=TEXT_PRIMARY,
                    sub="Desv. anual vs. benchmark",
                ), unsafe_allow_html=True)
            with c4:
                ir_color = POSITIVE if reg["ir"] > 0.5 else NEGATIVE if reg["ir"] < 0 else TEXT_MUTED
                st.markdown(_stat_card(
                    "Information Ratio",
                    f"{reg['ir']:.3f}",
                    color=ir_color,
                    sub="Alpha / Tracking Error",
                ), unsafe_allow_html=True)

            # ── SMB / HML (solo FF3) ──────────────────────────────────────────
            if reg["model"] == "FF3" and reg["smb_beta"] is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    smb_color = POSITIVE if reg["smb_beta"] > 0.1 else NEGATIVE if reg["smb_beta"] < -0.1 else TEXT_MUTED
                    st.markdown(_stat_card(
                        "Beta tamaño (SMB)",
                        f"{reg['smb_beta']:+.3f}",
                        color=smb_color,
                        sub=">0 sesgo small-cap · <0 sesgo large-cap",
                    ), unsafe_allow_html=True)
                with c2:
                    hml_color = POSITIVE if reg["hml_beta"] > 0.1 else NEGATIVE if reg["hml_beta"] < -0.1 else TEXT_MUTED
                    st.markdown(_stat_card(
                        "Beta valor (HML)",
                        f"{reg['hml_beta']:+.3f}",
                        color=hml_color,
                        sub=">0 sesgo value · <0 sesgo growth",
                    ), unsafe_allow_html=True)
                with c3:
                    r2_color = GOLD if reg["r_squared"] > 0.8 else TEXT_MUTED
                    st.markdown(_stat_card(
                        "R² del modelo",
                        f"{reg['r_squared']*100:.1f}%",
                        color=r2_color,
                        sub=f"% retorno explicado por los {reg['model']} factores",
                    ), unsafe_allow_html=True)

                # Factor exposure bar
                factor_names  = ["MKT-RF", "SMB (tamaño)", "HML (valor)"]
                factor_betas  = [reg["mkt_beta"], reg["smb_beta"], reg["hml_beta"]]
                factor_colors = [
                    POSITIVE if b > 0 else NEGATIVE for b in factor_betas
                ]
                fig_b = go.Figure(go.Bar(
                    x=factor_names, y=factor_betas,
                    marker_color=factor_colors,
                    text=[f"{b:+.3f}" for b in factor_betas],
                    textposition="outside",
                    textfont=dict(size=12, color=TEXT_PRIMARY),
                ))
                layout_b = dict(**PLOTLY_DARK)
                layout_b["height"] = 240
                layout_b["margin"] = dict(l=20, r=20, t=30, b=40)
                layout_b["yaxis"]  = dict(showgrid=True, gridcolor=BORDER_SOFT,
                                          zeroline=True, zerolinecolor=BORDER)
                layout_b["title"]  = dict(text="Exposición a Factores (betas)",
                                          font=dict(size=12, color=TEXT_MUTED))
                fig_b.update_layout(**layout_b)
                st.plotly_chart(fig_b, use_container_width=True)

            # ── Interpretación ────────────────────────────────────────────────
            with st.expander("¿Qué significan estos resultados?"):
                st.markdown(f"""
**Alpha ({alpha_sign}{reg['alpha_ann']*100:.2f}% anual):** El retorno del portfolio que no se explica
por la exposición a factores de mercado. Un alpha positivo significa que el portfolio genera valor
más allá del riesgo asumido. {'Alpha estadísticamente significativo.' if reg['alpha_pval'] < 0.05 else 'Alpha no estadísticamente distinguible de cero — podría ser ruido.'}

**Beta de mercado ({reg['mkt_beta']:.3f}):** Por cada 1% que sube el mercado, el portfolio mueve
{reg['mkt_beta']*100:.0f}%. {'Portfolio más volátil que el mercado.' if reg['mkt_beta'] > 1 else 'Portfolio más defensivo que el mercado.' if reg['mkt_beta'] < 1 else 'Portfolio alineado con el mercado.'}
""" + (f"""
**SMB ({reg.get('smb_beta', 0):+.3f}):** {'Sesgo hacia empresas pequeñas (small-caps).' if reg.get('smb_beta', 0) > 0.1 else 'Sesgo hacia empresas grandes (large-caps).' if reg.get('smb_beta', 0) < -0.1 else 'Sin sesgo significativo de tamaño.'}

**HML ({reg.get('hml_beta', 0):+.3f}):** {'Sesgo hacia acciones de valor (value).' if reg.get('hml_beta', 0) > 0.1 else 'Sesgo hacia acciones de crecimiento (growth).' if reg.get('hml_beta', 0) < -0.1 else 'Sin sesgo significativo de value/growth.'}

**R² ({reg['r_squared']*100:.1f}%):** El {reg['r_squared']*100:.0f}% de la variación del portfolio se explica por los factores. El restante {(1-reg['r_squared'])*100:.0f}% es idiosincrático (selección de valores).
""" if reg["model"] == "FF3" else ""))

    # ── Tab 2: Rolling Metrics ────────────────────────────────────────────────
    with tab_rolling:
        section_label(f"Métricas Rolling — ventana {roll_window}")

        rolling = compute_rolling(port_rets, bench_rets, window=window)

        if rolling.empty:
            st.warning(f"Datos insuficientes para ventana de {window} días.")
        else:
            fig_roll = go.Figure()

            # Sharpe rolling
            fig_roll.add_trace(go.Scatter(
                x=rolling.index, y=rolling["sharpe"],
                name="Sharpe ratio", line=dict(color=GOLD, width=1.5),
                yaxis="y1",
            ))
            # Vol rolling
            fig_roll.add_trace(go.Scatter(
                x=rolling.index, y=rolling["vol"],
                name="Volatilidad (%)", line=dict(color=NEGATIVE, width=1.5, dash="dot"),
                yaxis="y2",
            ))
            # Zero line for Sharpe
            fig_roll.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"), yref="y1")
            fig_roll.add_hline(y=1, line=dict(color="rgba(90,143,110,0.27)", width=1, dash="dash"), yref="y1")

            layout_r = dict(**PLOTLY_DARK)
            layout_r["height"] = 320
            layout_r["margin"] = dict(l=60, r=60, t=30, b=40)
            layout_r["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
            layout_r["yaxis"]  = dict(title="Sharpe ratio", showgrid=True,
                                      gridcolor=BORDER_SOFT, side="left")
            layout_r["yaxis2"] = dict(title="Volatilidad (%)", overlaying="y",
                                      side="right", showgrid=False, ticksuffix="%")
            layout_r["title"]  = dict(text="Sharpe y Volatilidad Anualizados (rolling)",
                                      font=dict(size=12, color=TEXT_MUTED))
            fig_roll.update_layout(**layout_r)
            st.plotly_chart(fig_roll, use_container_width=True)

            # Beta & correlation
            if "beta" in rolling.columns and "corr" in rolling.columns:
                fig_bc = go.Figure()
                fig_bc.add_trace(go.Scatter(
                    x=rolling.index, y=rolling["beta"],
                    name=f"Beta vs {benchmark}", line=dict(color="#6699cc", width=1.5),
                    yaxis="y1",
                ))
                fig_bc.add_trace(go.Scatter(
                    x=rolling.index, y=rolling["corr"],
                    name=f"Correlación vs {benchmark}",
                    line=dict(color=GOLD_DIM, width=1.5, dash="dot"),
                    yaxis="y2",
                ))
                fig_bc.add_hline(y=1, line=dict(color=BORDER, width=1, dash="dot"), yref="y1")
                layout_bc = dict(**PLOTLY_DARK)
                layout_bc["height"] = 280
                layout_bc["margin"] = dict(l=60, r=60, t=30, b=40)
                layout_bc["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
                layout_bc["yaxis"]  = dict(title=f"Beta vs {benchmark}", showgrid=True,
                                           gridcolor=BORDER_SOFT, side="left")
                layout_bc["yaxis2"] = dict(title="Correlación", overlaying="y",
                                           side="right", showgrid=False, range=[-1, 1])
                layout_bc["title"]  = dict(text=f"Beta y Correlación vs {benchmark} (rolling)",
                                           font=dict(size=12, color=TEXT_MUTED))
                fig_bc.update_layout(**layout_bc)
                st.plotly_chart(fig_bc, use_container_width=True)

        # Rolling Alpha (only if FF regression succeeded)
        if "reg" in dir() and reg and reg.get("model") == "FF3":
            with st.spinner("Calculando alpha rolling..."):
                try:
                    ff_local = ff  # captured in enclosing scope
                    aligned_all = pd.concat([port_rets, ff_local], axis=1).dropna() if ff_local is not None else None
                    if aligned_all is not None and len(aligned_all) >= window + 10:
                        roll_alpha = []
                        from scipy import stats as sp_stats
                        for i in range(window, len(aligned_all)):
                            seg = aligned_all.iloc[i - window: i]
                            y_r = (seg["Portfolio"] - seg["RF"]).values
                            X_r = np.column_stack([np.ones(len(y_r)), seg[["Mkt-RF", "SMB", "HML"]].values])
                            try:
                                b, _, _, _ = np.linalg.lstsq(X_r, y_r, rcond=None)
                                roll_alpha.append((aligned_all.index[i], float(b[0]) * 252 * 100))
                            except Exception:
                                roll_alpha.append((aligned_all.index[i], np.nan))

                        if roll_alpha:
                            ra_df = pd.DataFrame(roll_alpha, columns=["date", "alpha_ann_pct"]).set_index("date")
                            ra_df = ra_df.dropna()
                            if not ra_df.empty:
                                fig_ra = go.Figure()
                                fig_ra.add_trace(go.Scatter(
                                    x=ra_df.index,
                                    y=ra_df["alpha_ann_pct"],
                                    mode="lines",
                                    line=dict(color=GOLD, width=1.5),
                                    name="Alpha rolling anualizado",
                                    fill="tozeroy",
                                    fillcolor="rgba(201,168,76,0.13)",
                                    hovertemplate="%{x|%b %Y}: %{y:+.2f}%<extra></extra>",
                                ))
                                fig_ra.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
                                layout_ra = dict(**PLOTLY_DARK)
                                layout_ra["height"] = 260
                                layout_ra["margin"] = dict(l=60, r=20, t=30, b=40)
                                layout_ra["yaxis"]  = dict(title="Alpha anualizado (%)",
                                                           ticksuffix="%", showgrid=True,
                                                           gridcolor=BORDER_SOFT,
                                                           zeroline=True, zerolinecolor=BORDER)
                                layout_ra["title"]  = dict(text=f"Alpha FF3 rolling ({roll_window})",
                                                            font=dict(size=12, color=TEXT_MUTED))
                                fig_ra.update_layout(**layout_ra)
                                st.plotly_chart(fig_ra, use_container_width=True)
                except Exception:
                    pass  # Rolling alpha is optional — silently skip on failure

    # ── Tab 3: Drawdown ───────────────────────────────────────────────────────
    with tab_dd:
        section_label("Análisis de Drawdown")

        dd = compute_drawdown(port_rets)
        equity = np.exp(port_rets.cumsum()) * 100  # base 100

        # Max drawdown stats
        max_dd = float(dd.min())
        max_dd_date = dd.idxmin()
        # Recovery date (first day after max_dd where dd >= 0)
        post = dd[dd.index > max_dd_date]
        recovered = post[post >= -0.001]
        recovery_date = recovered.index[0] if not recovered.empty else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_stat_card(
                "Máximo Drawdown",
                f"{max_dd*100:.2f}%",
                color=NEGATIVE,
                sub=f"Fecha: {max_dd_date.strftime('%b %Y')}",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(_stat_card(
                "Drawdown actual",
                f"{float(dd.iloc[-1])*100:.2f}%",
                color=NEGATIVE if dd.iloc[-1] < -0.01 else POSITIVE,
                sub="0% = en máximos históricos",
            ), unsafe_allow_html=True)
        with c3:
            rec_str = recovery_date.strftime("%b %Y") if recovery_date else "No recuperado aún"
            days_under = int((dd < -0.001).sum())
            st.markdown(_stat_card(
                "Recuperación máx. DD",
                rec_str,
                color=POSITIVE if recovery_date else TEXT_MUTED,
                sub=f"{days_under} días bajo máximo histórico",
            ), unsafe_allow_html=True)

        # Underwater chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            fill="tozeroy",
            fillcolor="rgba(155,77,77,0.20)",
            line=dict(color=NEGATIVE, width=1),
            name="Drawdown",
            hovertemplate="%{x|%d %b %Y}<br>DD: %{y:.2f}%<extra></extra>",
        ))
        # Mark the max drawdown
        fig_dd.add_trace(go.Scatter(
            x=[max_dd_date], y=[max_dd * 100],
            mode="markers+text",
            marker=dict(color=NEGATIVE, size=10, symbol="x"),
            text=[f"Máx: {max_dd*100:.1f}%"],
            textposition="top right",
            textfont=dict(color=NEGATIVE, size=10),
            name="Máximo DD",
        ))
        layout_dd = dict(**PLOTLY_DARK)
        layout_dd["height"] = 300
        layout_dd["margin"] = dict(l=60, r=20, t=30, b=40)
        layout_dd["yaxis"]  = dict(title="Drawdown (%)", ticksuffix="%",
                                   showgrid=True, gridcolor=BORDER_SOFT)
        layout_dd["title"]  = dict(text="Underwater Chart — % por debajo del máximo histórico",
                                   font=dict(size=12, color=TEXT_MUTED))
        fig_dd.update_layout(**layout_dd)
        st.plotly_chart(fig_dd, use_container_width=True)

        # Equity curve alongside benchmark
        if bench_rets is not None:
            bench_eq = np.exp(bench_rets.reindex(port_rets.index).fillna(0).cumsum()) * 100
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity.index, y=equity.values,
                name="Portfolio", line=dict(color=GOLD, width=2),
            ))
            fig_eq.add_trace(go.Scatter(
                x=bench_eq.index, y=bench_eq.values,
                name=benchmark, line=dict(color="#4a6080", width=1.5, dash="dot"),
            ))
            layout_eq = dict(**PLOTLY_DARK)
            layout_eq["height"] = 260
            layout_eq["margin"] = dict(l=60, r=20, t=30, b=40)
            layout_eq["yaxis"]  = dict(title="Valor (base 100)", showgrid=True,
                                       gridcolor=BORDER_SOFT)
            layout_eq["title"]  = dict(text="Curva de capital vs benchmark (base 100)",
                                       font=dict(size=12, color=TEXT_MUTED))
            layout_eq["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
            fig_eq.update_layout(**layout_eq)
            st.plotly_chart(fig_eq, use_container_width=True)

    # ── Tab 4: Heatmap de retornos mensuales ──────────────────────────────────
    with tab_heatmap:
        section_label("Heatmap de Retornos Mensuales")

        monthly = monthly_returns_table(port_rets)

        if monthly.empty:
            st.warning("Datos insuficientes para heatmap mensual.")
        else:
            # Build Plotly heatmap
            years  = monthly.index.tolist()
            months = [c for c in monthly.columns if c != "Total"]
            z_data = monthly[months].values.tolist()

            # Custom colorscale: red → white → green
            colorscale = [
                [0.0,  "#8b1a1a"],
                [0.35, "#c0392b"],
                [0.5,  "#1a1a2e"],
                [0.65, "#27ae60"],
                [1.0,  "#1a6b3a"],
            ]

            # Text annotations
            text_data = []
            for row in monthly[months].values:
                text_row = []
                for v in row:
                    text_row.append(f"{v:+.1f}%" if not np.isnan(v) else "—")
                text_data.append(text_row)

            valid_vals = monthly[months].values[~np.isnan(monthly[months].values)]
            if len(valid_vals) == 0:
                st.warning("Sin datos suficientes para heatmap.")
            else:
                vmax = max(abs(valid_vals.max()), abs(valid_vals.min()), 0.01)  # guard /0

                fig_hm = go.Figure(go.Heatmap(
                z=z_data,
                x=months,
                y=[str(y) for y in years],
                text=text_data,
                texttemplate="%{text}",
                textfont=dict(size=10, color="white"),
                colorscale=colorscale,
                zmid=0,
                zmin=-vmax,
                zmax=vmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text="%", side="right"),
                    ticksuffix="%",
                    len=0.8,
                ),
                hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            ))
            layout_hm = dict(**PLOTLY_DARK)
            layout_hm["height"] = max(280, 35 * len(years) + 80)
            layout_hm["margin"] = dict(l=60, r=60, t=30, b=40)
            layout_hm["xaxis"]  = dict(side="top")
            layout_hm["title"]  = dict(text="Retornos mensuales del portfolio (%)",
                                       font=dict(size=12, color=TEXT_MUTED))
            fig_hm.update_layout(**layout_hm)
            st.plotly_chart(fig_hm, use_container_width=True)

            # Annual total column as bar chart
            annual = monthly["Total"].dropna()
            if not annual.empty:
                fig_ann = go.Figure(go.Bar(
                    x=[str(y) for y in annual.index],
                    y=annual.values,
                    marker_color=[POSITIVE if v >= 0 else NEGATIVE for v in annual.values],
                    text=[f"{v:+.1f}%" for v in annual.values],
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_PRIMARY),
                ))
                layout_ann = dict(**PLOTLY_DARK)
                layout_ann["height"] = 220
                layout_ann["margin"] = dict(l=20, r=20, t=20, b=40)
                layout_ann["yaxis"]  = dict(ticksuffix="%", showgrid=True,
                                            gridcolor=BORDER_SOFT,
                                            zeroline=True, zerolinecolor=BORDER)
                layout_ann["title"]  = dict(text="Retorno total anual (%)",
                                            font=dict(size=12, color=TEXT_MUTED))
                fig_ann.update_layout(**layout_ann)
                st.plotly_chart(fig_ann, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# OLS Beta — Full diagnostics per ticker
# ─────────────────────────────────────────────────────────────────────────────

def _ols_full(y: np.ndarray, x: np.ndarray) -> dict:
    """
    OLS regression y = alpha + beta*x with full diagnostic suite.
    All computations via numpy + scipy — no statsmodels required.

    Returns dict with:
        alpha, beta, alpha_se, beta_se, alpha_pval, beta_pval,
        r2, adj_r2, beta_ci_low, beta_ci_high (95%),
        residuals, fitted, n,
        durbin_watson,
        bp_stat, bp_pval  (Breusch-Pagan heteroskedasticity),
        cook_d            (Cook's distances array),
        outlier_mask      (bool array: Cook's D > 4/n threshold),
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])       # design matrix [1, x]
    p = X.shape[1]                              # 2 parameters

    # ── OLS coefficients ─────────────────────────────────────────────────────
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y               # [alpha, beta]
    fitted   = X @ beta_hat
    residuals = y - fitted

    # ── Variance & standard errors ────────────────────────────────────────────
    sse    = float(residuals @ residuals)
    mse    = sse / (n - p)
    se     = np.sqrt(np.diag(XtX_inv) * mse)  # [se_alpha, se_beta]

    # ── t-stats & p-values (2-tailed) ─────────────────────────────────────────
    t_stats = beta_hat / np.where(se > 0, se, np.inf)
    p_vals  = 2 * sp_stats.t.sf(np.abs(t_stats), df=n - p)

    # ── R² & adjusted R² ──────────────────────────────────────────────────────
    sst   = float(np.sum((y - y.mean()) ** 2))
    r2    = 1 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else 0.0

    # ── 95% Confidence intervals for beta ────────────────────────────────────
    t_crit = sp_stats.t.ppf(0.975, df=n - p)
    beta_ci_low  = float(beta_hat[1] - t_crit * se[1])
    beta_ci_high = float(beta_hat[1] + t_crit * se[1])

    # ── Durbin-Watson ─────────────────────────────────────────────────────────
    diff_res = np.diff(residuals)
    dw = float(np.sum(diff_res ** 2) / sse) if sse > 0 else 2.0

    # ── Breusch-Pagan (test for heteroskedasticity) ───────────────────────────
    # Auxiliary regression: e² ~ X
    e2 = residuals ** 2
    XtX_inv2 = np.linalg.pinv(X.T @ X)
    beta2    = XtX_inv2 @ X.T @ e2
    fitted2  = X @ beta2
    resid2   = e2 - fitted2
    sst2     = float(np.sum((e2 - e2.mean()) ** 2))
    sse2     = float(resid2 @ resid2)
    r2_aux   = 1 - sse2 / sst2 if sst2 > 0 else 0.0
    bp_stat  = float(n * r2_aux)           # LM statistic ~ chi2(k-1)
    bp_pval  = float(sp_stats.chi2.sf(bp_stat, df=p - 1))

    # ── Cook's Distance ───────────────────────────────────────────────────────
    # D_i = (e_i / (p * MSE))² * (h_ii / (1 - h_ii)²)
    H = X @ XtX_inv @ X.T                  # hat matrix (n×n)
    h = np.diag(H)                          # leverage
    h = np.clip(h, 0, 0.9999)
    cook_d = (residuals ** 2 / (p * mse)) * (h / (1 - h) ** 2) if mse > 0 else np.zeros(n)
    outlier_threshold = 4.0 / n
    outlier_mask = cook_d > outlier_threshold

    return {
        "alpha":       float(beta_hat[0]),
        "beta":        float(beta_hat[1]),
        "alpha_se":    float(se[0]),
        "beta_se":     float(se[1]),
        "alpha_pval":  float(p_vals[0]),
        "beta_pval":   float(p_vals[1]),
        "r2":          float(r2),
        "adj_r2":      float(adj_r2),
        "beta_ci_low":  beta_ci_low,
        "beta_ci_high": beta_ci_high,
        "residuals":   residuals,
        "fitted":      fitted,
        "n":           n,
        "durbin_watson": dw,
        "bp_stat":     bp_stat,
        "bp_pval":     bp_pval,
        "cook_d":      cook_d,
        "outlier_mask": outlier_mask,
    }


def _dw_interpretation(dw: float) -> tuple[str, str]:
    """Returns (label, color) for Durbin-Watson statistic."""
    if dw < 1.5:
        return "Autocorrelación positiva", "#e74c3c"
    elif dw > 2.5:
        return "Autocorrelación negativa", "#e74c3c"
    elif 1.8 <= dw <= 2.2:
        return "Sin autocorrelación", "#5a8f6e"
    else:
        return "Leve autocorrelación", "#c9a84c"


def _bp_interpretation(bp_pval: float) -> tuple[str, str]:
    """Returns (label, color) for Breusch-Pagan p-value."""
    if bp_pval < 0.01:
        return "Heterocedasticidad alta (p<0.01)", "#e74c3c"
    elif bp_pval < 0.05:
        return "Heterocedasticidad moderada (p<0.05)", "#c9a84c"
    else:
        return "Homocedasticidad (p>{:.2f})".format(round(bp_pval, 2)), "#5a8f6e"


def _sig_stars(pval: float) -> str:
    if pval < 0.001: return "★★★ p<0.001"
    if pval < 0.01:  return "★★ p<0.01"
    if pval < 0.05:  return "★ p<0.05"
    return "n.s."


def _render_ols_tab(available: list[str], returns: pd.DataFrame,
                    benchmark: str, portfolio_data):
    """Renderiza el tab de OLS Beta + diagnósticos."""
    section_label("Regresión OLS Beta — Diagnósticos Estadísticos por Activo")
    st.caption(
        "Regresión OLS: r_i = α + β·r_benchmark + ε. "
        "Diagnósticos: Durbin-Watson (autocorrelación), Breusch-Pagan (heterocedasticidad), "
        "Cook's Distance (outliers influyentes)."
    )

    if benchmark not in returns.columns:
        st.warning(f"No hay datos para el benchmark {benchmark}.")
        return

    bench_rets = returns[benchmark].dropna().values

    # ── Selector de ticker ────────────────────────────────────────────────────
    ticker_sel = st.selectbox(
        "Seleccionar activo para análisis detallado",
        options=available,
        key="ols_ticker_sel"
    )

    gold_divider()

    # ── Resumen de todos los tickers ──────────────────────────────────────────
    section_label("Tabla Resumen — Todos los Activos")
    summary_rows = []
    for tk in available:
        if tk not in returns.columns or tk == benchmark:
            continue
        tk_rets = returns[tk].dropna()
        common_idx = tk_rets.index.intersection(returns[benchmark].dropna().index)
        if len(common_idx) < 30:
            continue
        y = tk_rets.loc[common_idx].values
        x = returns[benchmark].loc[common_idx].values
        res = _ols_full(y, x)
        dw_label, _ = _dw_interpretation(res["durbin_watson"])
        bp_label, _ = _bp_interpretation(res["bp_pval"])
        n_outliers = int(res["outlier_mask"].sum())
        summary_rows.append({
            "Ticker":     tk,
            "Beta":       round(res["beta"], 3),
            "IC 95% Beta": f"[{res['beta_ci_low']:.3f}, {res['beta_ci_high']:.3f}]",
            "Alpha ann. (%)": round(res["alpha"] * 252 * 100, 2),
            "R²":         round(res["r2"], 3),
            "R² adj.":    round(res["adj_r2"], 3),
            "Durbin-Watson": round(res["durbin_watson"], 3),
            "DW Diagnóstico": dw_label,
            "BP p-val":   round(res["bp_pval"], 4),
            "BP Diagnóstico": bp_label,
            "Outliers (Cook's D)": n_outliers,
            "N obs.":     res["n"],
        })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)

        def _style_summary(row):
            beta = row.get("Beta", 1.0)
            alpha = row.get("Alpha ann. (%)", 0.0)
            styles_list = []
            for col in sum_df.columns:
                if col == "Beta":
                    s = f"color:{POSITIVE}" if 0 < beta < 1 else f"color:{NEGATIVE}" if beta > 1.5 else f"color:{GOLD}"
                elif col == "Alpha ann. (%)":
                    s = f"color:{POSITIVE}" if alpha > 0 else f"color:{NEGATIVE}"
                elif col == "DW Diagnóstico":
                    _, clr = _dw_interpretation(row.get("Durbin-Watson", 2.0))
                    s = f"color:{clr}"
                elif col == "BP Diagnóstico":
                    _, clr = _bp_interpretation(row.get("BP p-val", 1.0))
                    s = f"color:{clr}"
                else:
                    s = ""
                styles_list.append(s)
            return styles_list

        st.dataframe(
            sum_df.reset_index(drop=True).style.apply(_style_summary, axis=1),
            use_container_width=True, hide_index=True
        )

    gold_divider()

    # ── Análisis detallado del ticker seleccionado ────────────────────────────
    section_label(f"Análisis Detallado — {ticker_sel}")

    if ticker_sel not in returns.columns:
        st.warning("No hay datos para este ticker en el período seleccionado.")
        return

    tk_rets = returns[ticker_sel].dropna()
    common_idx = tk_rets.index.intersection(returns[benchmark].dropna().index)
    if len(common_idx) < 30:
        st.warning("Datos insuficientes (mínimo 30 observaciones).")
        return

    y = tk_rets.loc[common_idx].values
    x = returns[benchmark].loc[common_idx].values
    dates = common_idx
    res = _ols_full(y, x)

    # ── KPIs principales ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    beta_color = POSITIVE if 0 < res["beta"] < 1 else NEGATIVE if res["beta"] > 1.5 else GOLD
    alpha_color = POSITIVE if res["alpha"] > 0 else NEGATIVE
    dw_label, dw_color = _dw_interpretation(res["durbin_watson"])
    bp_label,  bp_color = _bp_interpretation(res["bp_pval"])

    with c1:
        st.markdown(f"""<div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
            border-radius:8px;padding:12px;text-align:center;'>
            <div style='color:{TEXT_MUTED};font-size:0.75rem;text-transform:uppercase;'>Beta</div>
            <div style='color:{beta_color};font-size:1.6rem;font-weight:700;'>{res["beta"]:.3f}</div>
            <div style='color:{TEXT_MUTED};font-size:0.72rem;'>IC [{res["beta_ci_low"]:.3f}, {res["beta_ci_high"]:.3f}]</div>
            <div style='color:{TEXT_MUTED};font-size:0.72rem;'>{_sig_stars(res["beta_pval"])}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        alpha_ann = res["alpha"] * 252 * 100
        st.markdown(f"""<div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
            border-radius:8px;padding:12px;text-align:center;'>
            <div style='color:{TEXT_MUTED};font-size:0.75rem;text-transform:uppercase;'>Alpha (anual)</div>
            <div style='color:{alpha_color};font-size:1.6rem;font-weight:700;'>{alpha_ann:+.2f}%</div>
            <div style='color:{TEXT_MUTED};font-size:0.72rem;'>{_sig_stars(res["alpha_pval"])}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
            border-radius:8px;padding:12px;text-align:center;'>
            <div style='color:{TEXT_MUTED};font-size:0.75rem;text-transform:uppercase;'>R² / R² adj.</div>
            <div style='color:{GOLD};font-size:1.6rem;font-weight:700;'>{res["r2"]:.3f}</div>
            <div style='color:{TEXT_MUTED};font-size:0.72rem;'>adj. {res["adj_r2"]:.3f} · N={res["n"]}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
            border-radius:8px;padding:12px;text-align:center;'>
            <div style='color:{TEXT_MUTED};font-size:0.75rem;text-transform:uppercase;'>Durbin-Watson</div>
            <div style='color:{dw_color};font-size:1.6rem;font-weight:700;'>{res["durbin_watson"]:.3f}</div>
            <div style='color:{dw_color};font-size:0.72rem;'>{dw_label}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        n_out = int(res["outlier_mask"].sum())
        out_color = NEGATIVE if n_out > 5 else GOLD if n_out > 2 else POSITIVE
        st.markdown(f"""<div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
            border-radius:8px;padding:12px;text-align:center;'>
            <div style='color:{TEXT_MUTED};font-size:0.75rem;text-transform:uppercase;'>Outliers (Cook)</div>
            <div style='color:{out_color};font-size:1.6rem;font-weight:700;'>{n_out}</div>
            <div style='color:{TEXT_MUTED};font-size:0.72rem;'>umbral 4/N={4/res["n"]:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Breusch-Pagan result ──────────────────────────────────────────────────
    bp_icon = "✅" if res["bp_pval"] >= 0.05 else "⚠️"
    st.markdown(
        f"**Breusch-Pagan:** {bp_icon} {bp_label} &nbsp;·&nbsp; "
        f"Estadístico: {res['bp_stat']:.3f} &nbsp;·&nbsp; p-valor: {res['bp_pval']:.4f}",
        unsafe_allow_html=True
    )

    gold_divider()

    # ── Gráfico 1: Scatter retornos + recta OLS ───────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        section_label("Scatter: Retornos vs Benchmark")
        outlier_mask = res["outlier_mask"]
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = res["alpha"] + res["beta"] * x_line

        fig_sc = go.Figure()
        # Normal points
        fig_sc.add_trace(go.Scatter(
            x=x[~outlier_mask], y=y[~outlier_mask],
            mode='markers',
            marker=dict(color=GOLD, opacity=0.5, size=4),
            name="Observaciones",
            hovertemplate=f"{benchmark}: %{{x:.3f}}<br>{ticker_sel}: %{{y:.3f}}<extra></extra>",
        ))
        # Outliers (Cook's D > 4/n)
        if outlier_mask.sum() > 0:
            fig_sc.add_trace(go.Scatter(
                x=x[outlier_mask], y=y[outlier_mask],
                mode='markers',
                marker=dict(color=NEGATIVE, size=7, symbol='x'),
                name=f"Outliers (Cook's D>4/N)",
                hovertemplate=f"{benchmark}: %{{x:.3f}}<br>{ticker_sel}: %{{y:.3f}}<extra>Outlier</extra>",
            ))
        # Regression line
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color=POSITIVE, width=2),
            name=f"OLS: β={res['beta']:.3f}",
        ))
        # 95% CI band
        se_fit = np.sqrt(mse_val := (res["residuals"] @ res["residuals"]) / (res["n"] - 2))
        x_std = np.std(x)
        ci_band = sp_stats.t.ppf(0.975, df=res["n"]-2) * se_fit * np.sqrt(
            1/res["n"] + (x_line - x.mean())**2 / ((res["n"]-1) * x_std**2 + 1e-12)
        )
        fig_sc.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_line + ci_band, (y_line - ci_band)[::-1]]),
            fill='toself',
            fillcolor='rgba(90,143,110,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='IC 95%', showlegend=True,
        ))
        fig_sc.update_layout(**plotly_layout(
            height=350,
            xaxis_title=f"r_{benchmark}",
            yaxis_title=f"r_{ticker_sel}",
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=40, r=20, t=30, b=40),
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_g2:
        section_label("Cook's Distance")
        cook_threshold = 4.0 / res["n"]
        cook_colors = [NEGATIVE if v > cook_threshold else SURFACE_2 for v in res["cook_d"]]
        fig_cook = go.Figure()
        fig_cook.add_trace(go.Bar(
            x=list(range(len(res["cook_d"]))),
            y=res["cook_d"],
            marker_color=cook_colors,
            hovertemplate="Obs %{x}: D=%{y:.4f}<extra></extra>",
        ))
        fig_cook.add_hline(
            y=cook_threshold,
            line=dict(color=NEGATIVE, width=1.5, dash="dash"),
            annotation_text=f"4/N={cook_threshold:.4f}"
        )
        fig_cook.update_layout(**plotly_layout(
            height=350,
            xaxis_title="Observación",
            yaxis_title="Cook's D",
            margin=dict(l=40, r=20, t=30, b=40),
        ))
        st.plotly_chart(fig_cook, use_container_width=True)

    # ── Gráfico 2: Residuos ───────────────────────────────────────────────────
    gold_divider()
    section_label("Análisis de Residuos")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        # Residuos en el tiempo
        fig_res = go.Figure()
        fig_res.add_hline(y=0, line=dict(color=BORDER, width=1))
        fig_res.add_trace(go.Scatter(
            x=list(dates), y=res["residuals"],
            mode='lines', line=dict(color=GOLD, width=1),
            hovertemplate="%{x|%d %b %Y}: %{y:.4f}<extra>Residuo</extra>",
        ))
        # Highlight outliers
        if outlier_mask.sum() > 0:
            fig_res.add_trace(go.Scatter(
                x=[dates[i] for i in range(len(dates)) if outlier_mask[i]],
                y=res["residuals"][outlier_mask],
                mode='markers',
                marker=dict(color=NEGATIVE, size=6),
                name="Outliers",
            ))
        fig_res.update_layout(**plotly_layout(
            height=280,
            title=dict(text="Residuos en el tiempo", font=dict(size=12, color=TEXT_SECONDARY)),
            margin=dict(l=40, r=20, t=40, b=30),
        ))
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        # Q-Q plot de residuos (normalidad)
        sorted_res = np.sort(res["residuals"])
        n_pts = len(sorted_res)
        theoretical_q = sp_stats.norm.ppf(np.linspace(0.01, 0.99, n_pts))
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sorted_res,
            mode='markers',
            marker=dict(color=GOLD, size=4, opacity=0.6),
            name="Residuos",
            hovertemplate="Q teórico: %{x:.3f}<br>Residuo: %{y:.3f}<extra></extra>",
        ))
        # Reference line
        min_q, max_q = theoretical_q.min(), theoretical_q.max()
        ref_y1 = res["residuals"].mean() + res["residuals"].std() * min_q
        ref_y2 = res["residuals"].mean() + res["residuals"].std() * max_q
        fig_qq.add_trace(go.Scatter(
            x=[min_q, max_q], y=[ref_y1, ref_y2],
            mode='lines', line=dict(color=POSITIVE, width=1.5, dash='dash'),
            name="Normal teórica",
        ))
        fig_qq.update_layout(**plotly_layout(
            height=280,
            title=dict(text="Q-Q Plot residuos (normalidad)", font=dict(size=12, color=TEXT_SECONDARY)),
            xaxis_title="Cuantiles teóricos (Normal)",
            yaxis_title="Cuantiles empíricos",
            margin=dict(l=40, r=20, t=40, b=30),
        ))
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Interpretación ────────────────────────────────────────────────────────
    with st.expander("📖 Cómo interpretar los diagnósticos"):
        st.markdown(f"""
**Beta ({res['beta']:.3f}):** Por cada 1% que se mueve {benchmark}, {ticker_sel} se mueve {res['beta']:.2f}%.
{"Beta > 1 → más volátil que el mercado." if res["beta"] > 1 else "Beta < 1 → menos volátil que el mercado."}
El intervalo de confianza al 95% es [{res['beta_ci_low']:.3f}, {res['beta_ci_high']:.3f}].

**Alpha ({res['alpha']*252*100:+.2f}% anual):** Retorno no explicado por el mercado.
{"Alpha positivo y significativo → el activo genera valor más allá del riesgo de mercado." if res["alpha"] > 0 else "Alpha negativo → el activo destruye valor ajustado por riesgo de mercado."}

**R² ({res['r2']:.3f}):** El {res['r2']*100:.1f}% de la varianza de {ticker_sel} se explica por {benchmark}.
R² ajustado ({res['adj_r2']:.3f}) penaliza por el número de parámetros.

**Durbin-Watson ({res['durbin_watson']:.3f}):** Detecta autocorrelación en residuos.
Rango 0-4; valor ~2 = sin autocorrelación. < 1.5 = autocorrelación positiva (los residuos se parecen a los anteriores).

**Breusch-Pagan (p={res['bp_pval']:.4f}):** Detecta heterocedasticidad (varianza no constante de los errores).
{"p > 0.05 → homocedasticidad, los supuestos OLS se cumplen." if res["bp_pval"] >= 0.05 else "p < 0.05 → heterocedasticidad, la varianza de los errores no es constante. Los errores estándar pueden estar sesgados."}

**Cook's Distance:** Identifica observaciones influyentes (outliers que distorsionan la regresión).
Umbral convencional: 4/N = {4/res['n']:.4f}. Se encontraron **{int(res['outlier_mask'].sum())} observaciones** por encima del umbral.
        """)

    # ── Tab 5: OLS Beta + Diagnósticos ───────────────────────────────────────
    with tab_ols:
        _render_ols_tab(available, returns, benchmark, portfolio_data)

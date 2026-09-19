"""
modules/optimizer.py — Portfolio Optimization Engine
=====================================================
Implementa tres estrategias clásicas de optimización:

1. Frontera Eficiente de Markowitz (Mean-Variance)
2. Máximo Sharpe Ratio (tangency portfolio)
3. Risk Parity (equal risk contribution)

Basado en scipy.optimize — sin dependencias externas adicionales.
Solo activos con datos históricos disponibles (equity, ETF, crypto).
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from scipy.optimize import minimize
from modules.styles import (
    GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG,
    PLOTLY_DARK, section_label, gold_divider, page_header,
)
from modules.utils import ensure_portfolio_data


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns(tickers: tuple, period: str = "3y") -> pd.DataFrame:
    """
    Descarga retornos diarios para la lista de tickers.
    Devuelve DataFrame de retornos log-diarios, columnas = tickers.
    Elimina tickers sin datos suficientes (<252 días).
    """
    raw = yf.download(
        list(tickers), period=period,
        auto_adjust=True, progress=False, threads=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # Retornos log-diarios
    rets = np.log(prices / prices.shift(1)).dropna(how="all")

    # Eliminar tickers con menos de 252 observaciones válidas
    valid = [c for c in rets.columns if rets[c].notna().sum() >= 252]
    return rets[valid].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf: float = 0.05, ann: int = 252) -> tuple:
    """Return (annualized_return, annualized_vol, sharpe)."""
    w = np.array(weights)
    ret = float(w @ mu) * ann
    vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ann)
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _neg_sharpe(weights, mu, cov, rf=0.05, ann=252):
    r, v, s = _portfolio_stats(weights, mu, cov, rf, ann)
    return -s


def _portfolio_vol(weights, cov, ann=252):
    return float(np.sqrt(weights @ cov @ weights)) * np.sqrt(ann)


def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution × weight for each asset."""
    sigma = np.sqrt(weights @ cov @ weights)
    mrc = cov @ weights / sigma
    return weights * mrc


def _risk_parity_objective(weights, cov):
    """Sum of squared differences of risk contributions (target = equal)."""
    rc = _risk_contribution(weights, cov)
    target = np.full(len(weights), rc.sum() / len(weights))
    return float(np.sum((rc - target) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────

def _multi_start_minimize(objective, bounds, constraints, args=(),
                          n_starts: int = 8, options=None) -> np.ndarray:
    """
    Run SLSQP from multiple random starting points.
    Returns the solution with the best (lowest) objective value.
    Guarantees convergence for non-convex objectives like neg-Sharpe.
    """
    n = len(bounds)
    rng = np.random.default_rng(42)
    best_result = None
    best_val = np.inf

    # Always try equal weights first
    start_points = [np.ones(n) / n]
    for _ in range(n_starts - 1):
        w = rng.dirichlet(np.ones(n))
        # Clip to bounds
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w = w / w.sum()
        start_points.append(w)

    for w0 in start_points:
        try:
            res = minimize(
                objective, w0, args=args, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options=options or {"ftol": 1e-12, "maxiter": 1000}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception:
            continue

    return best_result.x if best_result is not None else np.ones(n) / n


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray,
                        rf: float = 0.05, ann: int = 252) -> np.ndarray:
    """Weights that maximize Sharpe ratio (long-only), multi-start."""
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    return _multi_start_minimize(_neg_sharpe, bounds, constraints,
                                  args=(mu, cov, rf, ann), n_starts=8)


def optimize_min_vol(mu: np.ndarray, cov: np.ndarray, ann: int = 252) -> np.ndarray:
    """Weights that minimize portfolio volatility (long-only), multi-start."""
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    return _multi_start_minimize(_portfolio_vol, bounds, constraints,
                                  args=(cov, ann), n_starts=6)


def optimize_risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution (Risk Parity) weights, multi-start."""
    n = cov.shape[0]
    bounds = [(1e-6, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w = _multi_start_minimize(_risk_parity_objective, bounds, constraints,
                               args=(cov,), n_starts=10,
                               options={"ftol": 1e-15, "maxiter": 2000})
    return w / w.sum()


def compute_efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                                rf: float = 0.05, ann: int = 252,
                                n_points: int = 60) -> pd.DataFrame:
    """
    Compute efficient frontier by sweeping target returns.
    Returns DataFrame with columns: ret, vol, sharpe, weights...
    """
    n = len(mu)
    ret_min = float(mu.min()) * ann
    ret_max = float(mu.max()) * ann
    target_returns = np.linspace(ret_min * 0.8, ret_max * 1.05, n_points)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) * ann - t},
        ]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n
        res = minimize(_portfolio_vol, w0, args=(cov, ann),
                       method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            r, v, s = _portfolio_stats(res.x, mu, cov, rf, ann)
            frontier.append({"ret": r, "vol": v, "sharpe": s, **{f"w_{i}": res.x[i] for i in range(n)}})

    return pd.DataFrame(frontier)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, color: str = TEXT_PRIMARY, sub: str = "") -> str:
    sub_html = f"<div style='font-size:10px;color:{TEXT_MUTED};margin-top:3px'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER_SOFT};border-radius:10px;"
        f"padding:14px 16px;'>"
        f"<div style='font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:6px;'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{color};'>{value}</div>"
        f"{sub_html}</div>"
    )


def _weights_bar(weights: dict, color: str = GOLD) -> go.Figure:
    tickers = list(weights.keys())
    vals    = [weights[t] * 100 for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=vals,
        marker_color=color,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
    ))
    layout = dict(**PLOTLY_DARK)
    layout["margin"] = dict(l=10, r=10, t=10, b=40)
    layout["height"] = 220
    layout["yaxis"] = dict(showgrid=True, gridcolor=BORDER_SOFT, ticksuffix="%",
                           range=[0, max(vals) * 1.25])
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_optimizer():
    page_header("Optimización de Portfolio", "Frontera eficiente · Máximo Sharpe · Risk Parity")

    # ── Datos del portfolio ───────────────────────────────────────────────────
    portfolio_data = ensure_portfolio_data()
    if portfolio_data is None or portfolio_data.empty:
        from modules.utils import no_portfolio_warning
        no_portfolio_warning()
        return

    # Asegurar que la columna 'Asset Type' existe (puede faltar si viene del dashboard)
    if "Asset Type" not in portfolio_data.columns:
        portfolio_data = portfolio_data.copy()
        portfolio_data["Asset Type"] = "equity"
    # Solo activos equity/ETF con precio disponible
    VALID_TYPES = ("equity", "etf", "crypto", "fund", None, "")
    df_eq = portfolio_data[
        portfolio_data["Asset Type"].fillna("equity").str.lower().isin(
            ["equity", "etf", "crypto", "fund", ""]
        )
    ].copy()

    tickers = [t for t in df_eq["Ticker"].unique() if t]
    if len(tickers) < 2:
        st.warning("Necesitas al menos 2 activos en el portfolio para optimizar.")
        return

    # ── Parámetros ────────────────────────────────────────────────────────────
    with st.expander("Parámetros de optimización", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            period = st.selectbox("Historial de datos", ["1y", "2y", "3y", "5y"], index=2,
                                  help="Período de datos históricos para estimar retornos y covarianza")
        with col2:
            rf = st.slider("Tasa libre de riesgo (%)", 0.0, 8.0, 4.5, 0.25,
                           help="Usada para calcular el Sharpe ratio") / 100.0
        with col3:
            max_weight = st.slider("Peso máx. por activo (%)", 10, 100, 60, 5,
                                   help="Límite de concentración por activo (long-only)") / 100.0

    # ── Fetch & compute ───────────────────────────────────────────────────────
    with st.spinner(f"Descargando {len(tickers)} activos — {period} de historial..."):
        rets = _fetch_returns(tuple(tickers), period=period)

    if rets.empty or rets.shape[1] < 2:
        st.error("No se pudo obtener suficiente historial para los activos del portfolio.")
        return

    available = list(rets.columns)
    dropped = [t for t in tickers if t not in available]
    if dropped:
        st.caption(f"⚠ Sin historial suficiente (excluidos): {', '.join(dropped)}")

    mu  = rets.mean().values          # media diaria
    cov = rets.cov().values           # covarianza diaria
    n   = len(available)
    ann = 252

    # Pesos actuales del portfolio
    total_val = df_eq.groupby("Ticker")["Total Value ($)"].sum()
    total_sum = total_val.reindex(available).fillna(0).sum()
    w_current = (total_val.reindex(available).fillna(0) / total_sum).values if total_sum > 0 \
                else np.ones(n) / n

    # ── Optimizaciones ────────────────────────────────────────────────────────
    bounds_custom = [(0.0, max_weight)] * n
    constraints   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Feasibility check
    if max_weight * n < 1.0:
        st.error(
            f"⚠ Con {n} activos y peso máximo {max_weight*100:.0f}%, "
            f"la suma máxima de pesos es {max_weight*n*100:.0f}% < 100%. "
            f"Aumenta el peso máximo por activo a ≥ {100/n:.0f}%."
        )
        return

    def _opt_max_sharpe_custom():
        return _multi_start_minimize(_neg_sharpe, bounds_custom, constraints,
                                     args=(mu, cov, rf, ann), n_starts=8)

    def _opt_min_vol_custom():
        return _multi_start_minimize(_portfolio_vol, bounds_custom, constraints,
                                     args=(cov, ann), n_starts=6)

    def _opt_risk_parity_custom():
        bounds_rp = [(1e-6, max_weight)] * n
        w = _multi_start_minimize(_risk_parity_objective, bounds_rp, constraints,
                                   args=(cov,), n_starts=10,
                                   options={"ftol": 1e-15, "maxiter": 2000})
        return w / w.sum()

    with st.spinner("Calculando frontera eficiente..."):
        w_sharpe = _opt_max_sharpe_custom()
        w_minvol = _opt_min_vol_custom()
        w_rp     = _opt_risk_parity_custom()
        frontier = compute_efficient_frontier(mu, cov, rf=rf, ann=ann, n_points=50)

    r_cur, v_cur, s_cur = _portfolio_stats(w_current, mu, cov, rf, ann)
    r_sh,  v_sh,  s_sh  = _portfolio_stats(w_sharpe,  mu, cov, rf, ann)
    r_mv,  v_mv,  s_mv  = _portfolio_stats(w_minvol,  mu, cov, rf, ann)
    r_rp,  v_rp,  s_rp  = _portfolio_stats(w_rp,      mu, cov, rf, ann)

    # ── Tabs de resultado ─────────────────────────────────────────────────────
    tab_frontier, tab_sharpe, tab_minvol, tab_rp = st.tabs([
        "Frontera Eficiente", "Máximo Sharpe", "Mínima Volatilidad", "Risk Parity"
    ])

    # ── Tab 1: Frontera Eficiente ─────────────────────────────────────────────
    with tab_frontier:
        section_label("Frontera Eficiente de Markowitz")

        if not frontier.empty:
            fig = go.Figure()

            # Frontera
            fig.add_trace(go.Scatter(
                x=frontier["vol"] * 100, y=frontier["ret"] * 100,
                mode="lines",
                line=dict(color=GOLD, width=2),
                name="Frontera eficiente",
                hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
            ))

            # Puntos especiales
            special = [
                ("Portfolio actual", v_cur, r_cur, "#4a6080", "circle", 14),
                ("Máx. Sharpe",      v_sh,  r_sh,  POSITIVE,  "star",   16),
                ("Mín. Volatilidad", v_mv,  r_mv,  "#6699cc", "diamond", 14),
                ("Risk Parity",      v_rp,  r_rp,  "#cc9966", "square",  12),
            ]
            for name, vx, ry, color, symbol, size in special:
                fig.add_trace(go.Scatter(
                    x=[vx * 100], y=[ry * 100],
                    mode="markers+text",
                    marker=dict(color=color, size=size, symbol=symbol,
                                line=dict(color="#0d1117", width=2)),
                    text=[name], textposition="top right",
                    textfont=dict(size=10, color=color),
                    name=name,
                    hovertemplate=f"{name}<br>Vol: {vx*100:.2f}%<br>Ret: {ry*100:.2f}%<br>Sharpe: {(ry-rf)/vx:.2f}<extra></extra>",
                ))

            layout = dict(**PLOTLY_DARK)
            layout["height"] = 460
            layout["margin"] = dict(l=60, r=30, t=40, b=60)
            layout["xaxis"] = dict(title="Volatilidad anualizada (%)", showgrid=True,
                                   gridcolor=BORDER_SOFT, ticksuffix="%")
            layout["yaxis"] = dict(title="Retorno anualizado (%)", showgrid=True,
                                   gridcolor=BORDER_SOFT, ticksuffix="%")
            layout["legend"] = dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)")
            layout["title"] = dict(text="Frontera Eficiente — Riesgo vs. Retorno",
                                   font=dict(size=13, color=TEXT_MUTED))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        # Tabla comparativa
        st.markdown("<br>", unsafe_allow_html=True)
        comp = pd.DataFrame([
            {"Estrategia": "Portfolio actual",    "Retorno":  f"{r_cur*100:.2f}%", "Volatilidad": f"{v_cur*100:.2f}%", "Sharpe": f"{s_cur:.3f}"},
            {"Estrategia": "Máximo Sharpe",       "Retorno":  f"{r_sh*100:.2f}%",  "Volatilidad": f"{v_sh*100:.2f}%",  "Sharpe": f"{s_sh:.3f}"},
            {"Estrategia": "Mínima Volatilidad",  "Retorno":  f"{r_mv*100:.2f}%",  "Volatilidad": f"{v_mv*100:.2f}%",  "Sharpe": f"{s_mv:.3f}"},
            {"Estrategia": "Risk Parity",         "Retorno":  f"{r_rp*100:.2f}%",  "Volatilidad": f"{v_rp*100:.2f}%",  "Sharpe": f"{s_rp:.3f}"},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)

        # Correlation matrix heatmap
        gold_divider()
        section_label("Matriz de Correlación")
        st.caption("Correlación entre activos basada en retornos diarios históricos.")
        corr_matrix = rets.corr()
        corr_vals = corr_matrix.values
        ticker_labels = list(corr_matrix.columns)

        text_corr = []
        for row in corr_vals:
            text_corr.append([f"{v:.2f}" for v in row])

        corr_colorscale = [
            [0.0, "#8b1a1a"],   # strong negative → dark red
            [0.5, "#1a1a2e"],   # zero correlation → dark bg
            [1.0, "#1a6b3a"],   # strong positive → dark green
        ]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_vals.tolist(),
            x=ticker_labels,
            y=ticker_labels,
            text=text_corr,
            texttemplate="%{text}",
            textfont=dict(size=9 if len(ticker_labels) > 8 else 11, color="white"),
            colorscale=corr_colorscale,
            zmin=-1, zmax=1, zmid=0,
            showscale=True,
            colorbar=dict(title=dict(text="r", side="right"), len=0.8),
            hovertemplate="%{y} / %{x}: %{text}<extra></extra>",
        ))
        layout_corr = dict(**PLOTLY_DARK)
        layout_corr["height"] = max(320, 40 * len(ticker_labels) + 80)
        layout_corr["margin"] = dict(l=80, r=60, t=30, b=80)
        layout_corr["xaxis"]  = dict(tickangle=-45)
        layout_corr["title"]  = dict(text="Matriz de Correlación (retornos diarios)",
                                      font=dict(size=12, color=TEXT_MUTED))
        fig_corr.update_layout(**layout_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Diversification insight
        n_pairs = len(ticker_labels) * (len(ticker_labels) - 1) // 2
        if n_pairs > 0:
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            ).stack()
            avg_corr = float(upper_tri.mean())
            high_corr = int((upper_tri > 0.7).sum())
            low_corr  = int((upper_tri < 0.3).sum())
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                c = POSITIVE if avg_corr < 0.5 else NEGATIVE
                st.markdown(
                    f"<div style='text-align:center;padding:12px;background:{SURFACE};"
                    f"border-radius:8px;border:1px solid {BORDER_SOFT};'>"
                    f"<div style='font-size:10px;color:{TEXT_MUTED};'>Correlación media</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{c};'>{avg_corr:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_d2:
                c = NEGATIVE if high_corr > n_pairs * 0.3 else TEXT_MUTED
                st.markdown(
                    f"<div style='text-align:center;padding:12px;background:{SURFACE};"
                    f"border-radius:8px;border:1px solid {BORDER_SOFT};'>"
                    f"<div style='font-size:10px;color:{TEXT_MUTED};'>Pares altamente correlados (r&gt;0.7)</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{c};'>{high_corr} / {n_pairs}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_d3:
                c = POSITIVE if low_corr > n_pairs * 0.3 else TEXT_MUTED
                st.markdown(
                    f"<div style='text-align:center;padding:12px;background:{SURFACE};"
                    f"border-radius:8px;border:1px solid {BORDER_SOFT};'>"
                    f"<div style='font-size:10px;color:{TEXT_MUTED};'>Pares poco correlados (r&lt;0.3)</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{c};'>{low_corr} / {n_pairs}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── Tab 2: Máximo Sharpe ──────────────────────────────────────────────────
    with tab_sharpe:
        section_label("Portfolio de Máximo Sharpe Ratio")
        st.caption("Maximiza el retorno por unidad de riesgo. El punto óptimo en la línea de capital.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_kpi_card("Retorno anualizado", f"{r_sh*100:.2f}%",
                                  POSITIVE if r_sh > r_cur else NEGATIVE), unsafe_allow_html=True)
        with c2:
            st.markdown(_kpi_card("Volatilidad", f"{v_sh*100:.2f}%",
                                  POSITIVE if v_sh < v_cur else NEGATIVE,
                                  sub=f"Actual: {v_cur*100:.2f}%"), unsafe_allow_html=True)
        with c3:
            st.markdown(_kpi_card("Sharpe Ratio", f"{s_sh:.3f}", GOLD,
                                  sub=f"Actual: {s_cur:.3f}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        w_dict = {available[i]: w_sharpe[i] for i in range(n) if w_sharpe[i] > 0.001}
        st.plotly_chart(_weights_bar(w_dict, POSITIVE), use_container_width=True)
        _render_weights_table(available, w_current, w_sharpe, "Máx. Sharpe")

    # ── Tab 3: Mínima Volatilidad ─────────────────────────────────────────────
    with tab_minvol:
        section_label("Portfolio de Mínima Volatilidad")
        st.caption("Minimiza el riesgo total. Ideal para perfiles conservadores o mercados volátiles.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_kpi_card("Retorno anualizado", f"{r_mv*100:.2f}%",
                                  POSITIVE if r_mv > 0 else NEGATIVE), unsafe_allow_html=True)
        with c2:
            st.markdown(_kpi_card("Volatilidad", f"{v_mv*100:.2f}%", "#6699cc",
                                  sub=f"Actual: {v_cur*100:.2f}%"), unsafe_allow_html=True)
        with c3:
            st.markdown(_kpi_card("Sharpe Ratio", f"{s_mv:.3f}", GOLD,
                                  sub=f"Actual: {s_cur:.3f}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        w_dict = {available[i]: w_minvol[i] for i in range(n) if w_minvol[i] > 0.001}
        st.plotly_chart(_weights_bar(w_dict, "#6699cc"), use_container_width=True)
        _render_weights_table(available, w_current, w_minvol, "Mín. Volatilidad")

    # ── Tab 4: Risk Parity ────────────────────────────────────────────────────
    with tab_rp:
        section_label("Risk Parity — Contribución Igual al Riesgo")
        st.caption("Cada activo aporta la misma cantidad de riesgo al portfolio. "
                   "Diversificación real — no por capital sino por riesgo.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_kpi_card("Retorno anualizado", f"{r_rp*100:.2f}%",
                                  POSITIVE if r_rp > 0 else NEGATIVE), unsafe_allow_html=True)
        with c2:
            st.markdown(_kpi_card("Volatilidad", f"{v_rp*100:.2f}%", "#cc9966",
                                  sub=f"Actual: {v_cur*100:.2f}%"), unsafe_allow_html=True)
        with c3:
            st.markdown(_kpi_card("Sharpe Ratio", f"{s_rp:.3f}", GOLD,
                                  sub=f"Actual: {s_cur:.3f}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk contribution chart
        rc_current = _risk_contribution(w_current, cov)
        rc_rp      = _risk_contribution(w_rp, cov)
        rc_cur_pct = rc_current / rc_current.sum() * 100 if rc_current.sum() > 0 else rc_current
        rc_rp_pct  = rc_rp      / rc_rp.sum()      * 100 if rc_rp.sum()      > 0 else rc_rp

        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            name="Contribución actual", x=available,
            y=rc_cur_pct.tolist(),
            marker_color="#4a6080",
            text=[f"{v:.1f}%" for v in rc_cur_pct], textposition="outside",
        ))
        fig_rc.add_trace(go.Bar(
            name="Risk Parity (objetivo)", x=available,
            y=rc_rp_pct.tolist(),
            marker_color="#cc9966",
            text=[f"{v:.1f}%" for v in rc_rp_pct], textposition="outside",
        ))
        layout_rc = dict(**PLOTLY_DARK)
        layout_rc["height"]    = 260
        layout_rc["margin"]    = dict(l=10, r=10, t=40, b=40)
        layout_rc["barmode"]   = "group"
        layout_rc["yaxis"]     = dict(ticksuffix="%", showgrid=True, gridcolor=BORDER_SOFT)
        layout_rc["title"]     = dict(text="Contribución al riesgo por activo",
                                      font=dict(size=12, color=TEXT_MUTED))
        layout_rc["legend"]    = dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)")
        fig_rc.update_layout(**layout_rc)
        st.plotly_chart(fig_rc, use_container_width=True)

        _render_weights_table(available, w_current, w_rp, "Risk Parity")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: weights comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _render_weights_table(tickers: list, w_current: np.ndarray,
                          w_target: np.ndarray, label: str):
    gold_divider()
    st.markdown(f"##### Cambios sugeridos — {label} vs. Portfolio actual")

    rows = []
    for i, t in enumerate(tickers):
        cur = w_current[i] * 100
        tgt = w_target[i] * 100
        delta = tgt - cur
        rows.append({
            "Ticker":     t,
            "Actual (%)": round(cur, 2),
            f"{label} (%)": round(tgt, 2),
            "Δ (pp)":     round(delta, 2),
            "Acción":     ("AUMENTAR" if delta > 1 else "REDUCIR" if delta < -1 else "mantener"),
        })
    tbl = pd.DataFrame(rows).sort_values("Δ (pp)", ascending=False)

    def _style_delta(val):
        if isinstance(val, (int, float)):
            return f"color:{POSITIVE};font-weight:600" if val > 1 else \
                   f"color:{NEGATIVE};font-weight:600" if val < -1 else f"color:{TEXT_MUTED}"
        return ""

    def _style_action(val):
        if val == "AUMENTAR": return f"color:{POSITIVE};font-weight:700"
        if val == "REDUCIR":  return f"color:{NEGATIVE};font-weight:700"
        return f"color:{TEXT_MUTED}"

    st.dataframe(
        tbl.style
           .map(_style_delta,  subset=["Δ (pp)"])
           .map(_style_action, subset=["Acción"])
           .format({"Actual (%)": "{:.2f}%", f"{label} (%)": "{:.2f}%",
                    "Δ (pp)": "{:+.2f}pp"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("pp = puntos porcentuales de peso en el portfolio · Solo activos con historial disponible")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from modules.utils import ensure_portfolio_data, no_portfolio_warning, jarque_bera_test, ljung_box_test, arch_lm_test, safe_df_display
from modules.styles import PLOTLY_DARK, GOLD, SURFACE, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, plotly_layout
from modules.i18n import t


# ── Funciones de renta fija ───────────────────────────────────────────────────

def bond_price(face: float, coupon_rate: float, ytm: float,
               periods: int, frequency: int = 1) -> float:
    """Calcula el precio teórico de un bono dado su YTM."""
    c = face * coupon_rate / frequency
    r = ytm / frequency
    if r == 0:
        return c * periods + face
    price = sum(c / (1 + r) ** i for i in range(1, periods + 1))
    price += face / (1 + r) ** periods
    return price


def macaulay_duration(face: float, coupon_rate: float, ytm: float,
                      periods: int, frequency: int = 1) -> float:
    """Duración de Macaulay en años."""
    c = face * coupon_rate / frequency
    r = ytm / frequency
    price = bond_price(face, coupon_rate, ytm, periods, frequency)
    if price == 0:
        return 0
    weighted = sum((i / frequency) * c / (1 + r) ** i for i in range(1, periods + 1))
    weighted += (periods / frequency) * face / (1 + r) ** periods
    return weighted / price


def modified_duration(mac_dur: float, ytm: float, frequency: int = 1) -> float:
    return mac_dur / (1 + ytm / frequency)


def ytm_from_price(face: float, coupon_rate: float, price: float,
                   periods: int, frequency: int = 1) -> float:
    """Estima YTM por bisección numérica."""
    lo, hi = 0.0001, 0.50
    for _ in range(100):
        mid = (lo + hi) / 2
        p   = bond_price(face, coupon_rate, mid, periods, frequency)
        if abs(p - price) < 0.01:
            return mid
        if p > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def periods_to_maturity(maturity_str: str, frequency: int = 1) -> int:
    """Calcula el número de periodos hasta el vencimiento."""
    try:
        mat = datetime.strptime(maturity_str, "%Y-%m-%d")
        years = max((mat - datetime.today()).days / 365.25, 0)
        return max(int(round(years * frequency)), 1)
    except Exception:
        return 10  # fallback 10 años


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_data(tickers_tuple):
    tickers = list(tickers_tuple)
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw['Close']
    else:
        df = raw[['Close']].copy()
        df.columns = tickers
        return df



def _render_robust_panel(sharpe, sortino, beta, downside_beta,
                         var_95_dollar, cvar_95_dollar, alpha, omega, is_es):
    """Panel comparativo: métrica paramétrica vs alternativa robusta cuando hay no-normalidad."""
    RED_BG  = "rgba(155,77,77,0.06)"
    RED_BD  = "rgba(155,77,77,0.28)"
    GOLD_BG = "rgba(201,168,76,0.07)"
    GOLD_BD = "rgba(201,168,76,0.32)"

    badge_warn = "⚠ Con reservas"  if is_es else "⚠ Use with caution"
    badge_ok   = "✓ Priorizar"     if is_es else "✓ Prioritize"
    title = ("Con no-normalidad detectada — métricas que debes priorizar"
             if is_es else
             "Non-normal returns detected — prioritize these metrics")

    def _card(label, value, desc, bg, bd, text_c, badge, badge_c):
        return (
            f"<div style='background:{bg}; border:1px solid {bd}; border-radius:10px; "
            f"padding:14px 16px; text-align:center;'>"
            f"<div style='color:{badge_c}; font-size:10px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:0.7px; margin-bottom:8px;'>{badge}</div>"
            f"<div style='color:#9ca3af; font-size:12px; font-weight:600; margin-bottom:6px;'>{label}</div>"
            f"<div style='color:{text_c}; font-size:22px; font-weight:700; font-family:monospace; "
            f"margin-bottom:4px;'>{value}</div>"
            f"<div style='color:#6b7280; font-size:10px; line-height:1.4;'>{desc}</div>"
            f"</div>"
        )

    def _arrow():
        return ("<div style='text-align:center; color:#4b5563; "
                "font-size:20px; padding-top:26px;'>→</div>")

    sharpe_desc  = "Asume dist. normal"          if is_es else "Assumes normal dist."
    sortino_desc = "Solo penaliza pérdidas"       if is_es else "Only penalizes downside"
    beta_desc    = "Beta en todos los días"       if is_es else "Beta across all days"
    dbeta_desc   = "Beta solo en días bajistas"   if is_es else "Beta on down-market days"
    var_desc     = "Cuantil de pérdida"           if is_es else "Loss quantile"
    cvar_desc    = "Pérdida media en peor 5%"     if is_es else "Avg loss in worst 5%"
    alpha_desc   = "Basado en CAPM"               if is_es else "CAPM-based"
    omega_desc   = "Ganancias/pérdidas reales"    if is_es else "Actual gains/losses ratio"

    pairs = [
        ("Sharpe Ratio",        f"{sharpe:.2f}",           sharpe_desc,
         "Sortino Ratio",       f"{sortino:.2f}",           sortino_desc),
        ("Beta (CAPM)",         f"{beta:.2f}",              beta_desc,
         "Downside Beta",       f"{downside_beta:.2f}",     dbeta_desc),
        ("VaR histórico 95%",   f"${var_95_dollar:,.0f}",  var_desc,
         "CVaR / E. Shortfall", f"${cvar_95_dollar:,.0f}", cvar_desc),
        ("Alpha de Jensen",     f"{alpha*100:.2f}%",        alpha_desc,
         "Omega Ratio",         f"{min(omega, 99.9):.2f}", omega_desc),
    ]

    st.markdown(
        f"<div style='margin:4px 0 16px 0; padding:10px 16px; "
        f"background:rgba(201,168,76,0.05); border:1px solid rgba(201,168,76,0.2); "
        f"border-radius:8px;'>"
        f"<span style='color:#c9a84c; font-size:11px; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:1px;'>⚖ {title}</span></div>",
        unsafe_allow_html=True,
    )

    for i in range(0, 4, 2):
        p1, p2 = pairs[i], pairs[i + 1]
        c1, ar1, c2, _sp, c3, ar2, c4 = st.columns([4, 0.55, 4, 0.4, 4, 0.55, 4])
        c1.markdown(_card(p1[0], p1[1], p1[2], RED_BG, RED_BD, "#e2e8f0", badge_warn, "#9b4d4d"),
                    unsafe_allow_html=True)
        ar1.markdown(_arrow(), unsafe_allow_html=True)
        c2.markdown(_card(p1[3], p1[4], p1[5], GOLD_BG, GOLD_BD, "#c9a84c", badge_ok, "#c9a84c"),
                    unsafe_allow_html=True)
        _sp.markdown("", unsafe_allow_html=True)
        c3.markdown(_card(p2[0], p2[1], p2[2], RED_BG, RED_BD, "#e2e8f0", badge_warn, "#9b4d4d"),
                    unsafe_allow_html=True)
        ar2.markdown(_arrow(), unsafe_allow_html=True)
        c4.markdown(_card(p2[3], p2[4], p2[5], GOLD_BG, GOLD_BD, "#c9a84c", badge_ok, "#c9a84c"),
                    unsafe_allow_html=True)
        if i == 0:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def render_risk_analytics():
    st.title(t("risk.title"))

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        no_portfolio_warning()
        return

    total_aum = portfolio_df['Total Value ($)'].sum()
    if total_aum <= 0:
        st.error("El valor total del portfolio debe ser mayor que cero.")
        return

    st.markdown("### Composición Actual del Portfolio")
    display_cols = ['Ticker']
    if 'Name' in portfolio_df.columns:
        display_cols.append('Name')
    display_cols += ['Shares', 'Current Price ($)', 'Total Value ($)']
    display_cols = [c for c in display_cols if c in portfolio_df.columns]
    st.dataframe(safe_df_display(portfolio_df[display_cols]), use_container_width=True, hide_index=True)

    portfolio_df = portfolio_df.copy()
    portfolio_df['Peso'] = portfolio_df['Total Value ($)'] / total_aum
    tickers = portfolio_df['Ticker'].tolist()
    weights = portfolio_df['Peso'].values

    if st.button(t("risk.btn_run"), type="primary", use_container_width=True):
        with st.spinner("Descargando datos históricos y ejecutando modelos estocásticos..."):
            try:
                # 1. Descargar histórico 1 año + SPY como benchmark
                hist_tickers = list(set(tickers + ['SPY']))
                data = get_historical_data(tuple(sorted(hist_tickers)))

                # 2. Extraer precios de cierre
                if isinstance(data.columns, pd.MultiIndex):
                    close_prices = data['Close']
                else:
                    close_prices = data

                close_prices = close_prices.ffill().dropna()

                if close_prices.empty:
                    st.error("No hay suficientes datos históricos para estos activos.")
                    return

                # 3. Calcular retornos diarios
                returns = close_prices.pct_change().dropna()
                returns = returns.clip(lower=-0.5, upper=0.5)

                port_returns = returns[[tk for tk in tickers if tk in returns.columns]]
                # Dedup columns (ticker could appear twice if portfolio has duplicate rows)
                port_returns = port_returns.loc[:, ~port_returns.columns.duplicated()]
                bench_returns = returns['SPY'] if 'SPY' in returns.columns else None

                valid_tickers = port_returns.columns.tolist()
                # Agrupar pesos por Ticker para evitar mismatch si hay filas duplicadas
                _w_series = portfolio_df.groupby('Ticker')['Peso'].sum()
                valid_weights = np.array([float(_w_series.get(t, 0.0)) for t in valid_tickers])

                if len(valid_weights) == 0 or np.sum(valid_weights) == 0:
                    st.error("No se pudo calcular la distribución de pesos.")
                    return

                valid_weights = valid_weights / np.sum(valid_weights)

                # 4. Métricas de riesgo
                mean_returns = port_returns.mean()
                cov_matrix = port_returns.cov()

                port_daily_return = np.sum(mean_returns * valid_weights)
                port_annual_return = port_daily_return * 252

                variance = np.dot(valid_weights.T, np.dot(cov_matrix, valid_weights))
                if variance < 0 or np.isnan(variance) or np.isinf(variance):
                    st.error("La varianza calculada no es válida. Revisa los símbolos del portfolio.")
                    return

                port_daily_vol = np.sqrt(variance)
                port_annual_vol = port_daily_vol * np.sqrt(252)

                risk_free_rate = st.session_state.get('risk_free_rate', 4.0) / 100
                sharpe_ratio = (port_annual_return - risk_free_rate) / port_annual_vol if port_annual_vol > 0 else 0

                hist_port_returns = port_returns.dot(valid_weights)

                if bench_returns is not None:
                    cov_port_bench = np.cov(hist_port_returns, bench_returns)[0, 1]
                    var_bench = np.var(bench_returns)
                    beta = cov_port_bench / var_bench if var_bench > 0 else 1.0
                    # Jensen's Alpha
                    bench_annual = bench_returns.mean() * 252
                    alpha = port_annual_return - (risk_free_rate + beta * (bench_annual - risk_free_rate))
                    # Treynor Ratio
                    treynor = (port_annual_return - risk_free_rate) / beta if beta != 0 else 0
                else:
                    beta, alpha, treynor = 1.0, 0.0, 0.0

                # VaR histórico
                var_95 = np.percentile(hist_port_returns, 5)
                var_99 = np.percentile(hist_port_returns, 1)
                var_95_dollar = abs(var_95) * total_aum
                var_99_dollar = abs(var_99) * total_aum

                # Max Drawdown
                cumulative = (1 + hist_port_returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()

                # ── Panel de métricas ─────────────────────────────────────────
                st.markdown("---")
                st.markdown(f"### {t('risk.metrics_title')}")

                # Fila 1: métricas base
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(t("risk.annual_return"), f"{port_annual_return * 100:.2f}%")
                col2.metric(t("risk.volatility"), f"{port_annual_vol * 100:.2f}%")
                col3.metric(t("risk.sharpe"), f"{sharpe_ratio:.2f}",
                            help="Retorno excedente por unidad de riesgo total. >1 es bueno, >2 es excelente.")
                col4.metric(t("risk.beta"), f"{beta:.2f}",
                            help="Sensibilidad al mercado. Beta>1 = más volátil que el mercado.")

                # Fila 2: métricas avanzadas
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                col5, col6, col7, col8 = st.columns(4)
                col5.metric(t("risk.alpha"),
                            f"{alpha * 100:.2f}%",
                            help="Retorno generado por encima del benchmark ajustado por riesgo. Alpha>0 indica outperformance.",
                            delta=f"{'outperformance' if alpha > 0 else 'underperformance'}",
                            delta_color="normal" if alpha > 0 else "inverse")
                col6.metric(t("risk.treynor"),
                            f"{treynor * 100:.2f}%",
                            help="Como el Sharpe pero usando Beta en lugar de volatilidad total.")
                col7.metric(t("risk.var95"),
                            f"${var_95_dollar:,.0f}",
                            help=f"Pérdida máxima esperada en 1 día con 95% de confianza ({var_95*100:.2f}%)")
                col8.metric(t("risk.max_drawdown"),
                            f"{max_drawdown * 100:.1f}%",
                            delta=f"{max_drawdown * 100:.1f}%",
                            delta_color="inverse")

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                col9, col10, _, __ = st.columns(4)
                col9.metric(t("risk.var99"),
                            f"${var_99_dollar:,.0f}",
                            help=f"Pérdida máxima esperada en 1 día con 99% de confianza ({var_99*100:.2f}%)")
                col10.metric(t("risk.var99m"),
                             f"${var_99_dollar * np.sqrt(21):,.0f}",
                             help="VaR diario escalado a 1 mes (√21 días de trading).")

                # ── Fila 3: métricas avanzadas adicionales ────────────────────
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                # Sortino Ratio
                downside_returns = hist_port_returns[hist_port_returns < 0]
                downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else port_annual_vol
                sortino = (port_annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

                # Calmar Ratio
                calmar = abs(port_annual_return / max_drawdown) if max_drawdown < 0 else 0

                # Information Ratio (vs SPY)
                if bench_returns is not None:
                    active_ret = hist_port_returns - bench_returns
                    tracking_error = active_ret.std() * np.sqrt(252)
                    info_ratio = (active_ret.mean() * 252) / tracking_error if tracking_error > 0 else 0
                else:
                    info_ratio = 0.0
                    tracking_error = 0.0

                # Omega Ratio (threshold = 0)
                gains = hist_port_returns[hist_port_returns > 0].sum()
                losses = abs(hist_port_returns[hist_port_returns < 0].sum())
                omega = gains / losses if losses > 0 else float('inf')

                # CVaR / Expected Shortfall al 95%
                _cvar_thresh = np.percentile(hist_port_returns, 5)
                _tail_rets = hist_port_returns[hist_port_returns <= _cvar_thresh]
                cvar_95 = float(_tail_rets.mean()) if len(_tail_rets) > 0 else var_95
                cvar_95_dollar = abs(cvar_95) * total_aum

                # Downside Beta (solo días bajistas del benchmark)
                downside_beta = beta
                if bench_returns is not None:
                    _aligned_db = pd.concat([hist_port_returns, bench_returns], axis=1).dropna()
                    _aligned_db.columns = ['_port', '_bench']
                    _down_days = _aligned_db[_aligned_db['_bench'] < 0]
                    if len(_down_days) >= 10:
                        _cov_db = np.cov(_down_days['_port'], _down_days['_bench'])[0, 1]
                        _var_db = np.var(_down_days['_bench'])
                        downside_beta = float(_cov_db / _var_db) if _var_db > 0 else beta

                is_es = t('general.loading') == 'Cargando...'

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric(
                    "Sortino Ratio" if is_es else "Sortino Ratio",
                    f"{sortino:.2f}",
                    help="Como el Sharpe pero solo penaliza la volatilidad bajista (downside). Mejor que el Sharpe para evaluar portfolios con sesgo alcista." if is_es else
                    "Like Sharpe but only penalizes downside volatility. Better for assessing upside-skewed portfolios."
                )
                col_b.metric(
                    "Calmar Ratio" if is_es else "Calmar Ratio",
                    f"{calmar:.2f}",
                    help="Retorno anual dividido entre el Max Drawdown. Mide cuánto retorno se obtiene por cada unidad de máxima pérdida histórica." if is_es else
                    "Annual return divided by Max Drawdown. Measures return per unit of historical peak loss."
                )
                col_c.metric(
                    "Information Ratio" if is_es else "Information Ratio",
                    f"{info_ratio:.2f}",
                    help=f"Retorno activo sobre el tracking error vs SPY. TE: {tracking_error*100:.2f}%. Mide consistencia del outperformance." if is_es else
                    f"Active return over tracking error vs SPY. TE: {tracking_error*100:.2f}%. Measures consistency of outperformance."
                )
                col_d.metric(
                    "Omega Ratio" if is_es else "Omega Ratio",
                    f"{min(omega, 99.9):.2f}",
                    help="Ratio ganancias/pérdidas reales sin asumir normalidad. >1 indica que las ganancias superan las pérdidas históricamente." if is_es else
                    "Gains/losses ratio without assuming normality. >1 means gains historically exceed losses."
                )

                # ── Matriz de correlaciones ───────────────────────────────────
                if len(valid_tickers) > 1:
                    st.markdown("---")
                    st.markdown(
                        f"### {'Matriz de Correlaciones' if is_es else 'Correlation Matrix'}",
                    )
                    st.caption(
                        "Correlación de Pearson entre retornos diarios. Cercano a 1 = movimiento conjunto; cercano a -1 = movimiento inverso; 0 = independientes."
                        if is_es else
                        "Pearson correlation between daily returns. Near 1 = move together; near -1 = move inversely; 0 = independent."
                    )
                    corr_matrix = port_returns.corr().round(2)

                    # Renderizar como heatmap con colores
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns.tolist(),
                        y=corr_matrix.index.tolist(),
                        colorscale=[
                            [0.0, '#9b4d4d'],
                            [0.5, '#1c2333'],
                            [1.0, '#5a8f6e'],
                        ],
                        zmin=-1, zmax=1,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont=dict(size=10, color='#e2d9cc'),
                        hovertemplate='%{y} — %{x}<br>Correlación: %{z:.3f}<extra></extra>',
                        showscale=True,
                        colorbar=dict(
                            title='r',
                            thickness=12,
                            tickfont=dict(color='#6b7280', size=9),
                        )
                    ))
                    # Excluir xaxis/yaxis de PLOTLY_DARK para evitar duplicados
                    corr_layout = {k: v for k, v in PLOTLY_DARK.items()
                                   if k not in ('xaxis', 'yaxis')}
                    corr_layout['margin'] = dict(l=0, r=60, t=20, b=0)
                    corr_layout['height'] = max(280, len(valid_tickers) * 40 + 80)
                    corr_layout['xaxis'] = dict(
                        tickfont=dict(size=10, color='#9ca3af'),
                        gridcolor='#2d3748',
                    )
                    corr_layout['yaxis'] = dict(
                        tickfont=dict(size=10, color='#9ca3af'),
                        gridcolor='#2d3748',
                    )
                    fig_corr.update_layout(**corr_layout)
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Tabla de correlaciones como DataFrame
                    with st.expander("📋 " + ("Ver tabla de correlaciones" if is_es else "View correlation table")):
                        def _color_corr(val):
                            # Red for negative, green for positive, white near 0
                            try:
                                v = float(val)
                            except (TypeError, ValueError):
                                return ""
                            if v > 0:
                                intensity = int(v * 180)
                                return f"background-color: rgba(74,{130+intensity//3},{90+intensity//3},0.6); color: #e8e0d5"
                            else:
                                intensity = int(abs(v) * 180)
                                return f"background-color: rgba({130+intensity//3},74,74,0.6); color: #e8e0d5"
                        st.dataframe(
                            corr_matrix.style.map(_color_corr).format("{:.2f}"),
                            use_container_width=True
                        )


                # ── Diagnóstico de Normalidad — 3 Tests ──────────────────────────
                st.markdown("---")
                st.markdown(
                    f"### {'Diagnóstico de Normalidad' if is_es else 'Normality Diagnostics'}"
                )
                st.caption(
                    "Tres tests complementarios: JB evalúa la forma, Ljung-Box la independencia serial y ARCH la estabilidad de la varianza."
                    if is_es else
                    "Three complementary tests: JB evaluates shape, Ljung-Box tests serial independence, ARCH tests variance stability."
                )

                jb_port   = jarque_bera_test(hist_port_returns)
                lb_port   = ljung_box_test(hist_port_returns)
                arch_port = arch_lm_test(hist_port_returns)

                # ── Overall verdict ───────────────────────────────────────────────────
                _s_map = {"normal": 0, "dudosa": 1, "no_normal": 2}
                _all_results = [r for r in [jb_port, lb_port, arch_port] if r]
                _score_sum = sum(_s_map.get(r["status"], 0) for r in _all_results)
                _n_tests   = len(_all_results)
                _n_fail    = sum(1 for r in _all_results if r["status"] != "normal")
                if _score_sum == 0:
                    _overall_st = "normal"
                elif _score_sum <= _n_tests * 0.75:
                    _overall_st = "dudosa"
                else:
                    _overall_st = "no_normal"

                def _nt_color(s): return {"normal": "#5a8f6e", "dudosa": "#c9a84c", "no_normal": "#9b4d4d"}.get(s, "#6b7280")
                def _nt_dot(s):   return {"normal": "🟢", "dudosa": "🟡", "no_normal": "🔴"}.get(s, "⚪")
                def _nt_lbl(s):
                    es = {"normal": "Normal", "dudosa": "Dudosa", "no_normal": "No normal"}
                    en = {"normal": "Normal", "dudosa": "Uncertain", "no_normal": "Non-normal"}
                    return (es if is_es else en).get(s, "—")

                # ── Mini-metric cell ──────────────────────────────────────────────────
                def _mm(label, value):
                    return (
                        f"<div style='background:#0a0f1a; border-radius:6px; padding:8px 10px;'>"
                        f"<div style='color:#6b7280; font-size:9px; text-transform:uppercase; "
                        f"letter-spacing:0.5px; margin-bottom:3px;'>{label}</div>"
                        f"<div style='color:#e2e8f0; font-size:13px; font-weight:700; "
                        f"font-family:monospace;'>{value}</div></div>"
                    )

                # ── Test card ─────────────────────────────────────────────────────────
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

                _tc1, _tc2, _tc3 = st.columns(3)

                if jb_port:
                    _tc1.markdown(_test_card(
                        "Jarque-Bera",
                        "Forma de la distribución" if is_es else "Distribution shape",
                        jb_port["status"],
                        [_mm("JB Stat", f"{jb_port['jb']:.2f}"),
                         _mm("p-valor", f"{jb_port['p_value']:.4f}"),
                         _mm("Skewness", f"{jb_port['skewness']:+.3f}"),
                         _mm("Exc. Curtosis", f"{jb_port['excess_kurtosis']:+.3f}")]
                    ), unsafe_allow_html=True)

                if lb_port:
                    _rho = lb_port["rho"]
                    _tc2.markdown(_test_card(
                        "Ljung-Box",
                        "Dependencia serial — lags 1-3" if is_es else "Serial dependence — lags 1-3",
                        lb_port["status"],
                        [_mm("Q stat", f"{lb_port['Q']:.2f}"),
                         _mm("p-valor", f"{lb_port['p_value']:.4f}"),
                         _mm("ρ₁ / ρ₂", f"{_rho.get(1,0):+.3f} / {_rho.get(2,0):+.3f}"),
                         _mm("ρ₃", f"{_rho.get(3,0):+.3f}")]
                    ), unsafe_allow_html=True)

                if arch_port:
                    _tc3.markdown(_test_card(
                        "ARCH — Engle",
                        "Volatility clustering",
                        arch_port["status"],
                        [_mm("LM stat", f"{arch_port['LM']:.2f}"),
                         _mm("p-valor", f"{arch_port['p_value']:.4f}"),
                         _mm("R² OLS", f"{arch_port['R2']:.5f}"),
                         _mm("Lags", f"{arch_port['lags']}")]
                    ), unsafe_allow_html=True)

                # ── Overall verdict banner ────────────────────────────────────────────
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                _ov_c = _nt_color(_overall_st)
                if _overall_st == "normal":
                    _ov_title = "Diagnóstico: Distribución Normal" if is_es else "Diagnosis: Normal Distribution"
                    _ov_msg   = ("Los tres tests no rechazan normalidad. Las métricas estándar tienen plena validez estadística."
                                 if is_es else
                                 "All three tests fail to reject normality. Standard metrics have full statistical validity.")
                elif _overall_st == "dudosa":
                    _ov_title = "Diagnóstico: Distribución Dudosa" if is_es else "Diagnosis: Uncertain Distribution"
                    _ov_msg   = (f"{_n_fail} de {_n_tests} tests detectan desviaciones. Interpreta las métricas estándar con precaución."
                                 if is_es else
                                 f"{_n_fail} of {_n_tests} tests detect deviations. Interpret standard metrics with caution.")
                else:
                    _ov_title = "Diagnóstico: Distribución Problemática" if is_es else "Diagnosis: Non-Normal Distribution"
                    _ov_msg   = (f"{_n_fail} de {_n_tests} tests rechazan los supuestos. "
                                 f"Sharpe, Beta, VaR paramétrico y Alpha de Jensen tienen validez reducida."
                                 if is_es else
                                 f"{_n_fail} of {_n_tests} tests reject assumptions. "
                                 f"Sharpe, Beta, parametric VaR and Jensen's Alpha have reduced validity.")

                st.markdown(
                    f"<div style='background:#0d1117; border:1px solid {_ov_c}44; "
                    f"border-left:4px solid {_ov_c}; border-radius:10px; padding:16px 20px; margin-top:4px;'>"
                    f"<div style='display:flex; align-items:center; gap:10px;'>"
                    f"<span style='font-size:22px;'>{_nt_dot(_overall_st)}</span>"
                    f"<div><div style='color:{_ov_c}; font-size:13px; font-weight:700;'>{_ov_title}</div>"
                    f"<div style='color:#9ca3af; font-size:12px; margin-top:3px; line-height:1.5;'>{_ov_msg}</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

                # ── Panel comparativo (si no-normal) ──────────────────────────────────
                if _overall_st != "normal":
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    _render_robust_panel(
                        sharpe_ratio, sortino, beta, downside_beta,
                        var_95_dollar, cvar_95_dollar, alpha, omega, is_es
                    )

                # ── Tests por activo ─────────────────────────────────────────────
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"#### {'Tests por activo (últimos 252 días)' if is_es else 'Per-asset tests (last 252 days)'}"
                )
                st.caption(
                    "JB = Jarque-Bera (forma)  ·  LB = Ljung-Box (dependencia)  ·  ARCH = volatility clustering"
                    if is_es else
                    "JB = Jarque-Bera (shape)  ·  LB = Ljung-Box (dependence)  ·  ARCH = volatility clustering"
                )
                _tkr_rows = []
                for _tk in valid_tickers:
                    _r_jb   = jarque_bera_test(port_returns[_tk])
                    _r_lb   = ljung_box_test(port_returns[_tk])
                    _r_arch = arch_lm_test(port_returns[_tk])
                    def _dot(r): return ("🟢" if r and r["status"] == "normal"
                                         else ("🟡" if r and r["status"] == "dudosa" else "🔴"))
                    _scores_tk = [{"normal": 0, "dudosa": 1, "no_normal": 2}.get(r["status"], 0)
                                  for r in [_r_jb, _r_lb, _r_arch] if r]
                    _sum_tk = sum(_scores_tk)
                    _n_tk   = len(_scores_tk)
                    if _sum_tk == 0:
                        _fiable_tk = "Sí" if is_es else "Yes"
                    elif _sum_tk <= _n_tk * 0.75:
                        _fiable_tk = "Con reservas" if is_es else "Caution"
                    else:
                        _fiable_tk = "No"
                    _fiable_col = "Métricas fiables" if is_es else "Metrics reliable"
                    _tkr_rows.append({
                        "Ticker": _tk,
                        "JB":     _dot(_r_jb),
                        "LB":     _dot(_r_lb),
                        "ARCH":   _dot(_r_arch),
                        _fiable_col: _fiable_tk,
                        "JB stat":  _r_jb["jb"]          if _r_jb   else "—",
                        "ρ₁":      _r_lb["rho"].get(1, "—") if _r_lb else "—",
                        "LM stat":  _r_arch["LM"]         if _r_arch else "—",
                    })

                if _tkr_rows:
                    import pandas as _pd2
                    _df_tkr = _pd2.DataFrame(_tkr_rows)
                    _fiable_col_key = "Métricas fiables" if is_es else "Metrics reliable"
                    def _color_tkr(val):
                        v = str(val)
                        if "🔴" in v or v.strip() == "No":
                            return "color: #9b4d4d; font-weight:600"
                        if "🟡" in v or v.strip() in ("Con reservas", "Caution"):
                            return "color: #c9a84c; font-weight:600"
                        if "🟢" in v or v.strip() in ("Sí", "Yes"):
                            return "color: #5a8f6e; font-weight:600"
                        return ""
                    st.dataframe(
                        _df_tkr.reset_index(drop=True).style.map(_color_tkr, subset=["JB", "LB", "ARCH", _fiable_col_key]),
                        use_container_width=True,
                        hide_index=True,
                    )

# ── Glosario de métricas ─────────────────────────────────────
                is_es = t('general.loading') == 'Cargando...'
                with st.expander("📖 " + ("¿Qué significa cada métrica?" if is_es else "What does each metric mean?")):
                    metrics_explained = [
                        ("Retorno Anual Esperado" if is_es else "Expected Annual Return",
                         "Es la ganancia media que esperamos obtener en un año, basada en el comportamiento histórico del portfolio. Un 10% significa que, de media, el portfolio habría crecido un 10% al año." if is_es else
                         "The average annual gain expected based on historical portfolio behavior."),
                        ("Volatilidad Anual" if is_es else "Annual Volatility",
                         "Mide cuánto fluctúa el valor del portfolio. Una volatilidad alta significa que los movimientos de precio son grandes e impredecibles. Una volatilidad baja indica un portfolio más estable." if is_es else
                         "Measures how much the portfolio value fluctuates. High volatility means large, unpredictable price swings."),
                        ("Ratio de Sharpe" if is_es else "Sharpe Ratio",
                         "Mide el retorno obtenido por cada unidad de riesgo asumido. Un Sharpe >1 es bueno, >2 es excelente. Si es negativo, el portfolio no compensa el riesgo que toma." if is_es else
                         "Measures return per unit of risk. Above 1 is good, above 2 is excellent."),
                        ("Beta" if is_es else "Beta",
                         "Mide la sensibilidad del portfolio frente al mercado (S&P 500). Beta=1 significa que se mueve igual que el mercado. Beta=1.5 significa que si el mercado sube 10%, el portfolio sube 15% (y viceversa en caídas)." if is_es else
                         "Sensitivity to the market (S&P 500). Beta=1.5 means if the market moves 10%, the portfolio moves 15%."),
                        ("Alpha (Jensen)" if is_es else "Jensen's Alpha",
                         "Es el retorno adicional generado por encima de lo que cabría esperar dado el riesgo asumido (Beta). Alpha positivo = el gestor está añadiendo valor. Alpha negativo = el portfolio rinde peor de lo esperado para su nivel de riesgo." if is_es else
                         "Extra return above what is expected given the risk taken. Positive alpha means value is being added."),
                        ("Ratio de Treynor" if is_es else "Treynor Ratio",
                         "Similar al Sharpe, pero usa Beta en lugar de volatilidad total. Es más útil cuando el portfolio está bien diversificado, ya que solo tiene en cuenta el riesgo sistemático (de mercado)." if is_es else
                         "Like Sharpe, but uses Beta instead of total volatility. More useful for well-diversified portfolios."),
                        ("VaR 95% / 99% (1 día)" if is_es else "VaR 95% / 99% (1 day)",
                         "Value at Risk: es la pérdida máxima que podría sufrir el portfolio en un día normal, con un 95% o 99% de confianza. Si el VaR 95% es $1.000, significa que en condiciones normales, el portfolio no perdería más de $1.000 en un día el 95% de las veces." if is_es else
                         "Maximum loss expected in one day under normal conditions, with 95% or 99% confidence."),
                        ("Max Drawdown" if is_es else "Max Drawdown",
                         "Es la mayor caída desde un máximo hasta el mínimo posterior en el periodo analizado. Si el Max Drawdown es -30%, significa que en algún momento el portfolio cayó un 30% desde su punto más alto antes de recuperarse." if is_es else
                         "The largest peak-to-trough decline in the analyzed period."),
                    ]
                    for name, explanation in metrics_explained:
                        st.markdown(f"""
                        <div style='padding:10px 0; border-bottom:1px solid #1c2333;'>
                            <span style='font-weight:700; color:#c9a84c; font-size:13px;'>{name}</span>
                            <p style='color:#7c8694; font-size:12px; margin:4px 0 0 0; line-height:1.6;'>{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Monte Carlo ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown(f"### {t('risk.mc_title')}")
                st.caption(t("risk.mc_subtitle"))

                days = 252
                simulations = 500

                simulation_df = np.zeros((days, simulations))
                simulation_df[0] = total_aum

                for step in range(1, days):
                    z = np.random.normal(0, 1, simulations)
                    growth_factor = np.exp(
                        (port_daily_return - 0.5 * port_daily_vol ** 2) + port_daily_vol * z
                    )
                    growth_factor = np.clip(growth_factor, 0.5, 2.0)
                    simulation_df[step] = simulation_df[step - 1] * growth_factor

                sim_df = pd.DataFrame(simulation_df)
                final_values = simulation_df[-1]   # valores finales de las simulaciones
                p5  = sim_df.quantile(0.05, axis=1)
                p50 = sim_df.quantile(0.50, axis=1)
                p95 = sim_df.quantile(0.95, axis=1)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(days)), y=p95,
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(days)), y=p5,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0, 168, 107, 0.15)',
                    name='Rango 90% de confianza'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(days)), y=p50,
                    mode='lines', name='Trayectoria esperada (Mediana)',
                    line=dict(color='#00a86b', width=2, dash='dot')
                ))
                layout = {**PLOTLY_DARK}
                layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                        bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280'))
                fig.update_layout(
                    **layout,
                    title=dict(text="Monte Carlo — 500 trayectorias (1 año)", font=dict(color='#9ca3af', size=13)),
                    xaxis_title="Días de trading",
                    yaxis_title="Valor del Portfolio ($)",
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Distribución final ───────────────────────────────────────
                st.markdown(f"#### {t('risk.distribution')}")
                scol1, scol2, scol3 = st.columns(3)
                scol1.metric(t("risk.median"),      f"${np.median(final_values):,.0f}")
                scol2.metric(t("risk.pessimistic"),  f"${np.percentile(final_values, 5):,.0f}")
                scol3.metric(t("risk.optimistic"),   f"${np.percentile(final_values, 95):,.0f}")

                # ── Drawdown histórico ───────────────────────────────────────
                st.markdown("---")
                st.markdown(f"### {t('risk.drawdown_title')}")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown.values * 100,
                    mode='lines', fill='tozeroy',
                    fillcolor='rgba(155,77,77,0.15)',
                    line=dict(color='#9b4d4d', width=1.5),
                    hovertemplate='%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>',
                    name='Drawdown'
                ))
                fig_dd.add_hline(y=max_drawdown * 100, line_dash="dot",
                                 line_color="#c9a84c", line_width=1,
                                 annotation_text=f"Max DD: {max_drawdown*100:.1f}%",
                                 annotation_font_color="#c9a84c")
                fig_dd.update_layout(**plotly_layout(
                    yaxis_title="Drawdown (%)",
                    height=220,
                    showlegend=False
                ))
                st.plotly_chart(fig_dd, use_container_width=True)

            except Exception as e:
                st.error(f"Error en el motor cuantitativo: {e}")

    # ── Análisis de Renta Fija ────────────────────────────────────────────────
    if 'portfolio' in st.session_state and not st.session_state.portfolio.empty:
        port_raw = st.session_state.portfolio
        bonds = port_raw[port_raw['Asset Type'] == 'bond'] \
            if 'Asset Type' in port_raw.columns else pd.DataFrame()

        if not bonds.empty:
            st.markdown("---")
            is_es_rf = t('general.loading') == 'Cargando...'
            st.markdown(f"### {'Análisis de Renta Fija' if is_es_rf else 'Fixed Income Analytics'}")
            st.caption("Duración, YTM y sensibilidad a variaciones de tipos de interés."
                       if is_es_rf else "Duration, YTM and interest rate sensitivity analysis.")

            rows = []
            port_dur_weighted = 0.0
            total_bond_value  = 0.0

            for _, b in bonds.iterrows():
                face     = float(b.get('Face Value') or 1000)
                coupon   = float(b.get('Coupon %') or 0) / 100
                maturity = str(b.get('Maturity') or '')
                qty      = float(b.get('Shares') or 1)
                freq     = 1
                periods  = periods_to_maturity(maturity, freq)
                price    = face
                ytm      = ytm_from_price(face, coupon, price, periods, freq)
                mac_dur  = macaulay_duration(face, coupon, ytm, periods, freq)
                mod_dur  = modified_duration(mac_dur, ytm, freq)
                value    = face * qty
                price_up   = bond_price(face, coupon, ytm + 0.01, periods, freq)
                price_down = bond_price(face, coupon, ytm - 0.01, periods, freq)
                sens_up    = (price_up - price) * qty
                sens_down  = (price_down - price) * qty
                rows.append({
                    'Bono' if is_es_rf else 'Bond': b.get('Name', b['Ticker']),
                    'Títulos': int(qty),
                    'Valor Nominal ($)': f"${value:,.0f}",
                    'Cupón (%)': f"{coupon*100:.2f}%",
                    'Vcto.': maturity[:10] if maturity else '—',
                    'YTM (%)': f"{ytm*100:.2f}%",
                    'Dur. Mac. (años)': f"{mac_dur:.2f}",
                    'Dur. Mod.': f"{mod_dur:.2f}",
                    '+100pb ($)': f"${sens_up:,.0f}",
                    '-100pb ($)': f"${sens_down:,.0f}",
                })
                port_dur_weighted += mod_dur * value
                total_bond_value  += value

            if rows:
                st.dataframe(safe_df_display(pd.DataFrame(rows)), use_container_width=True, hide_index=True)

            if total_bond_value > 0:
                port_mod_dur   = port_dur_weighted / total_bond_value
                sens_port_up   = -port_mod_dur * total_bond_value * 0.01
                sens_port_down =  port_mod_dur * total_bond_value * 0.01
                st.markdown(f"#### {'Resumen de Riesgo de Tipos' if is_es_rf else 'Interest Rate Risk Summary'}")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Duración Modificada Portfolio" if is_es_rf else "Portfolio Modified Duration",
                    f"{port_mod_dur:.2f} años",
                )
                c2.metric(
                    "Impacto +100pb (pérdida)" if is_es_rf else "Impact +100bps (loss)",
                    f"${abs(sens_port_up):,.0f}",
                    delta=f"{(sens_port_up/total_bond_value)*100:.2f}%",
                    delta_color="inverse"
                )
                c3.metric(
                    "Impacto -100pb (ganancia)" if is_es_rf else "Impact -100bps (gain)",
                    f"${sens_port_down:,.0f}",
                    delta=f"+{(sens_port_down/total_bond_value)*100:.2f}%"
                )

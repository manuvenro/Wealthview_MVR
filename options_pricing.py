"""
options_pricing.py — Binomial CRR + CAPM ex-ante / ex-post
===========================================================
1. Árbol Binomial CRR (Cox-Ross-Rubinstein)
   - Call / Put, European / American
   - Cualquier subyacente (acción, ETF, índice)
   - Greeks por diferencias finitas: Delta, Gamma, Theta, Vega, Rho
   - Visualización del árbol (hasta N=6 para legibilidad)
   - Comparativa con Black-Scholes (European only)

2. CAPM ex-ante vs ex-post
   - Ex-ante: retorno esperado = r_f + β * (E[Rm] - r_f)
   - Ex-post: retorno realizado vs retorno CAPM-implícito
   - Por activo individual y para el portfolio completo
   - Alpha de Jensen: retorno realizado - retorno CAPM esperado
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy import stats as sp_stats
from datetime import date, datetime
from math import log, sqrt, exp

from modules.styles import (
    inject_global_css, page_header, section_label, gold_divider,
    GOLD, GOLD_DIM, GOLD_BORDER, SURFACE, SURFACE_2, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG, PLOTLY_DARK,
    kpi_card, badge, positive_color, plotly_layout
)
from modules.utils import ensure_portfolio_data, no_portfolio_warning

# ─────────────────────────────────────────────────────────────────────────────
# CRR Binomial Tree
# ─────────────────────────────────────────────────────────────────────────────

def crr_price(
    S: float,        # Precio spot del subyacente
    K: float,        # Strike
    T: float,        # Tiempo a vencimiento (años)
    r: float,        # Tasa libre de riesgo (anual, decimal)
    sigma: float,    # Volatilidad implícita (anual, decimal)
    N: int,          # Número de pasos del árbol
    option_type: str = "call",   # "call" | "put"
    style: str = "european",     # "european" | "american"
    q: float = 0.0,  # Dividend yield continuo
) -> dict:
    """
    Precio CRR y árbol completo.
    Devuelve: price, stock_tree, option_tree, u, d, p, dt
    """
    dt   = T / N
    u    = exp(sigma * sqrt(dt))
    d    = 1 / u
    disc = exp(-r * dt)
    p    = (exp((r - q) * dt) - d) / (u - d)   # prob. riesgo-neutral
    p    = max(0.0, min(1.0, p))                 # clip por seguridad

    # ── Árbol de precios del subyacente ───────────────────────────────────────
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # ── Árbol de valores de la opción ─────────────────────────────────────────
    option_tree = np.zeros((N + 1, N + 1))

    # Valores en vencimiento
    for j in range(N + 1):
        St = stock_tree[j, N]
        if option_type == "call":
            option_tree[j, N] = max(0.0, St - K)
        else:
            option_tree[j, N] = max(0.0, K - St)

    # Inducción hacia atrás
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if style == "american":
                St = stock_tree[j, i]
                intrinsic = max(0.0, St - K) if option_type == "call" else max(0.0, K - St)
                option_tree[j, i] = max(continuation, intrinsic)
            else:
                option_tree[j, i] = continuation

    price = option_tree[0, 0]
    return {
        "price": price,
        "stock_tree": stock_tree,
        "option_tree": option_tree,
        "u": u, "d": d, "p": p, "dt": dt,
        "N": N, "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        "option_type": option_type, "style": style,
    }


def crr_greeks(
    S, K, T, r, sigma, N, option_type="call", style="european", q=0.0
) -> dict:
    """
    Greeks por diferencias finitas sobre el árbol CRR.
    Delta, Gamma, Theta: extraídos directamente del árbol (nodos 1 y 2).
    Vega, Rho: diferencias finitas externas.
    """
    res0 = crr_price(S, K, T, r, sigma, N, option_type, style, q)
    tree_opt  = res0["option_tree"]
    tree_stk  = res0["stock_tree"]
    price     = res0["price"]
    dt        = res0["dt"]

    # Delta: (f_u - f_d) / (S_u - S_d) en nodo 0
    f_u  = tree_opt[0, 1]
    f_d  = tree_opt[1, 1]
    S_u  = tree_stk[0, 1]
    S_d  = tree_stk[1, 1]
    delta = (f_u - f_d) / (S_u - S_d) if (S_u - S_d) != 0 else 0.0

    # Gamma: (f_uu - f_ud - f_du + f_dd) / ((S_uu - S_dd)/2)²  at step 2
    if N >= 2:
        f_uu = tree_opt[0, 2]; S_uu = tree_stk[0, 2]
        f_ud = tree_opt[1, 2]; S_ud = tree_stk[1, 2]
        f_dd = tree_opt[2, 2]; S_dd = tree_stk[2, 2]
        delta_u = (f_uu - f_ud) / (S_uu - S_ud) if (S_uu - S_ud) != 0 else 0.0
        delta_d = (f_ud - f_dd) / (S_ud - S_dd) if (S_ud - S_dd) != 0 else 0.0
        S_mid   = (S_uu + S_dd) / 2
        gamma   = (delta_u - delta_d) / (S_mid - S_dd) if (S_mid - S_dd) != 0 else 0.0
    else:
        gamma = 0.0

    # Theta: (f_ud - f_0) / (2*dt)  — precio en nodo central después de 2 pasos
    if N >= 2:
        f_mid = tree_opt[1, 2]   # nodo central en t=2dt
        theta = (f_mid - price) / (2 * dt) / 365   # por día calendario
    else:
        theta = 0.0

    # Vega: df/dσ (bump σ ± 1%)
    dv = 0.01
    if sigma + dv < 5.0:
        p_up = crr_price(S, K, T, r, sigma + dv, N, option_type, style, q)["price"]
        p_dn = crr_price(S, K, T, r, sigma - dv, N, option_type, style, q)["price"]
        vega = (p_up - p_dn) / (2 * dv) / 100   # por 1% de σ
    else:
        vega = 0.0

    # Rho: df/dr (bump r ± 0.01%)
    dr = 0.0001
    p_rho_up = crr_price(S, K, T, r + dr, sigma, N, option_type, style, q)["price"]
    p_rho_dn = crr_price(S, K, T, r - dr, sigma, N, option_type, style, q)["price"]
    rho = (p_rho_up - p_rho_dn) / (2 * dr) / 100   # por 1% de r

    return {
        "price": price, "delta": delta, "gamma": gamma,
        "theta": theta, "vega": vega,   "rho": rho,
    }


def black_scholes(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """Precio Black-Scholes para opción europea (referencia)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return (S * exp(-q * T) * sp_stats.norm.cdf(d1)
                - K * exp(-r * T) * sp_stats.norm.cdf(d2))
    else:
        return (K * exp(-r * T) * sp_stats.norm.cdf(-d2)
                - S * exp(-q * T) * sp_stats.norm.cdf(-d1))


# ─────────────────────────────────────────────────────────────────────────────
# Volatilidad implícita del mercado — cadena de opciones via yfinance
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_expiries(ticker: str) -> list[str]:
    """
    Devuelve la lista de fechas de vencimiento disponibles para un ticker.
    yfinance.Ticker.options → tuple de strings 'YYYY-MM-DD'.
    TTL: 5 min (los vencimientos no cambian intraday).
    """
    try:
        exps = yf.Ticker(ticker.strip().upper()).options
        return list(exps) if exps else []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_chain(ticker: str, expiry: str, opt_type: str) -> pd.DataFrame:
    """
    Descarga la cadena de opciones para un ticker, vencimiento y tipo.
    Devuelve DataFrame con columnas estandarizadas:
      strike, lastPrice, bid, ask, impliedVolatility, inTheMoney, volume, openInterest
    La IV de yfinance ya está en decimal (0.25 = 25%).
    TTL: 5 min.
    """
    _KEY_COLS = ['strike', 'lastPrice', 'bid', 'ask',
                 'impliedVolatility', 'inTheMoney', 'volume', 'openInterest']
    try:
        chain = yf.Ticker(ticker.strip().upper()).option_chain(expiry)
        df = chain.calls if opt_type == "call" else chain.puts
        # Solo columnas que existen
        available = [c for c in _KEY_COLS if c in df.columns]
        out = df[available].copy()
        # Limpiar IV: yfinance puede devolver NaN o 0 para contratos sin actividad
        if 'impliedVolatility' in out.columns:
            out = out[out['impliedVolatility'].notna() & (out['impliedVolatility'] > 0)]
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _iv_for_strike(chain_df: pd.DataFrame, strike: float) -> float | None:
    """Devuelve la IV más cercana al strike pedido desde la cadena."""
    if chain_df.empty or 'impliedVolatility' not in chain_df.columns:
        return None
    idx = (chain_df['strike'] - strike).abs().idxmin()
    iv = float(chain_df.at[idx, 'impliedVolatility'])
    return iv if iv > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# CAPM ex-ante / ex-post
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_capm_data(
    tickers_tuple: tuple,
    benchmark: str,
    period: str,
    rf_annual: float,
) -> pd.DataFrame:
    """
    Para cada ticker: beta (OLS vs benchmark), retorno realizado,
    CAPM ex-ante (r_f + β*(E[Rm]-r_f)), CAPM ex-post (r_f + β*r_m_realizado),
    Alpha de Jensen (realizado - CAPM ex-post).
    """
    all_tickers = list(tickers_tuple) + [benchmark]
    try:
        raw = yf.download(all_tickers, period=period, auto_adjust=True, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        else:
            prices = raw
    except Exception:
        return pd.DataFrame()

    if benchmark not in prices.columns:
        return pd.DataFrame()

    rets = prices.pct_change().dropna()
    bench_rets = rets[benchmark].values
    bench_total = float((1 + rets[benchmark]).prod() - 1)  # retorno total realizado

    rf_period = rf_annual  # aproximamos como si fuera el período completo

    rows = []
    for tk in tickers_tuple:
        if tk not in rets.columns or tk == benchmark:
            continue
        tk_rets = rets[tk].dropna()
        common  = rets[benchmark].reindex(tk_rets.index).dropna()
        y = tk_rets.reindex(common.index).dropna().values
        x = common.reindex(pd.Index(common.index)).values[:len(y)]
        if len(y) < 30:
            continue

        # OLS beta
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
        beta = float(slope)
        r2   = float(r_val ** 2)

        # Retorno anualizado realizado
        n_years = len(y) / 252
        ret_total = float((1 + pd.Series(y)).prod() - 1)
        ret_ann   = float((1 + ret_total) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # Benchmark retorno anualizado realizado
        bench_ann = float((1 + bench_total) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # CAPM ex-ante: usa prima de mercado esperada histórica ~6% (ERP)
        erp_expected = 0.06  # equity risk premium largo plazo (Damodaran)
        capm_exante  = rf_annual + beta * erp_expected

        # CAPM ex-post: usa retorno de mercado REALIZADO
        capm_expost  = rf_annual + beta * (bench_ann - rf_annual)

        # Alpha de Jensen = retorno realizado - CAPM ex-post
        alpha_jensen = ret_ann - capm_expost

        rows.append({
            "Ticker":            tk,
            "Beta":              round(beta, 3),
            "R²":                round(r2, 3),
            "Retorno real. (%)": round(ret_ann * 100, 2),
            "Benchmark real. (%)": round(bench_ann * 100, 2),
            "CAPM ex-ante (%)":  round(capm_exante * 100, 2),
            "CAPM ex-post (%)":  round(capm_expost * 100, 2),
            "Alpha Jensen (%)":  round(alpha_jensen * 100, 2),
            "N obs.":            len(y),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render_options_pricing():
    inject_global_css()
    page_header(
        "Pricing de Opciones & CAPM",
        "Árbol binomial CRR · Greeks · CAPM ex-ante vs ex-post · Alpha de Jensen"
    )

    tab_binomial, tab_capm = st.tabs([
        "🌳 Árbol Binomial CRR",
        "📐 CAPM ex-ante / ex-post"
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: ÁRBOL BINOMIAL CRR
    # ═════════════════════════════════════════════════════════════════════════
    with tab_binomial:
        # Aplicar valores pendientes ANTES de renderizar los widgets
        # (Streamlit no permite modificar session_state de un widget ya renderizado)
        for _key in ("crr_S", "crr_K", "crr_sigma", "crr_expiry"):
            _pending = f"_pending_{_key}"
            if _pending in st.session_state:
                st.session_state[_key] = st.session_state.pop(_pending)

        section_label("Parámetros de la Opción")

        col1, col2, col3 = st.columns(3)
        with col1:
            S = st.number_input("Precio del subyacente (S)", min_value=0.01,
                                value=100.0, step=1.0, format="%.2f", key="crr_S")
            K = st.number_input("Strike (K)", min_value=0.01,
                                value=100.0, step=1.0, format="%.2f", key="crr_K")
            option_type = st.radio("Tipo", ["call", "put"], horizontal=True, key="crr_type")

        with col2:
            expiry_date = st.date_input("Fecha de vencimiento",
                                         value=date.today().replace(month=date.today().month % 12 + 1, day=1),
                                         min_value=date.today(), key="crr_expiry")
            T = max((expiry_date - date.today()).days / 365.0, 1/365)
            st.caption(f"T = {T:.4f} años ({(expiry_date - date.today()).days} días)")
            style = st.radio("Estilo", ["european", "american"], horizontal=True, key="crr_style")

        with col3:
            sigma = st.number_input("Volatilidad implícita (σ %)", min_value=0.1,
                                    value=25.0, step=0.5, format="%.1f",
                                    key="crr_sigma") / 100
            r = st.slider("Tasa libre de riesgo (r %)", 0.0, 15.0, 4.5,
                          step=0.1, key="crr_r") / 100
            q = st.slider("Dividend yield (q %)", 0.0, 10.0, 0.0,
                          step=0.1, key="crr_q") / 100
            N = st.slider("Pasos del árbol (N)", 2, 500, 100, step=1, key="crr_N")

        # Fetch live price si hay ticker
        st.markdown("---")
        col_tk, col_imp = st.columns([2, 3])
        with col_tk:
            live_ticker = st.text_input("Ticker del subyacente (opcional — rellena S y σ)",
                                        placeholder="AAPL, TSLA...", key="crr_live_ticker")
            if live_ticker and st.button("📡 Obtener precio y vol. histórica", key="crr_fetch_btn"):
                with st.spinner(f"Descargando {live_ticker.upper()}..."):
                    try:
                        tk_obj  = yf.Ticker(live_ticker.strip().upper())
                        live_px = tk_obj.fast_info.last_price
                        hist    = tk_obj.history(period="1y")["Close"].pct_change().dropna()
                        hist_vol = float(hist.std() * np.sqrt(252)) if len(hist) > 10 else None
                        if live_px:
                            st.session_state["_pending_crr_S"] = round(float(live_px), 2)
                            st.session_state["_pending_crr_K"] = round(float(live_px), 2)
                            st.success(f"S = ${live_px:.2f}")
                        if hist_vol:
                            st.session_state["_pending_crr_sigma"] = round(hist_vol * 100, 1)
                            st.success(f"σ histórica = {hist_vol*100:.1f}%")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_imp:
            # ── Volatilidad implícita del mercado ─────────────────────────────
            if live_ticker and live_ticker.strip():
                tk_up = live_ticker.strip().upper()
                with st.expander("📊 Cadena de opciones del mercado — IV real", expanded=True):
                    _col_exp, _col_typ = st.columns(2)
                    with _col_exp:
                        expiries = fetch_option_expiries(tk_up)
                        if expiries:
                            sel_expiry = st.selectbox(
                                "Vencimiento disponible",
                                expiries,
                                key="crr_mkt_expiry"
                            )
                        else:
                            st.warning(f"No hay opciones listadas para {tk_up}.")
                            sel_expiry = None
                    with _col_typ:
                        mkt_opt_type = st.radio(
                            "Tipo de cadena",
                            ["call", "put"],
                            horizontal=True,
                            key="crr_mkt_type",
                            index=0 if option_type == "call" else 1
                        )

                    if sel_expiry:
                        with st.spinner(f"Descargando cadena {tk_up} {sel_expiry}..."):
                            chain_df = fetch_option_chain(tk_up, sel_expiry, mkt_opt_type)

                        if chain_df.empty:
                            st.warning("Cadena vacía o sin datos de IV para este vencimiento.")
                        else:
                            # Tabla formateada
                            display_df = chain_df.copy()
                            if 'impliedVolatility' in display_df.columns:
                                display_df['IV (%)'] = (display_df['impliedVolatility'] * 100).round(2)
                                display_df = display_df.drop(columns=['impliedVolatility'])
                            if 'inTheMoney' in display_df.columns:
                                display_df['ITM'] = display_df['inTheMoney'].map(
                                    lambda x: "✅" if x else "⬜")
                                display_df = display_df.drop(columns=['inTheMoney'])
                            col_rename = {
                                'strike': 'Strike ($)',
                                'lastPrice': 'Último',
                                'bid': 'Bid',
                                'ask': 'Ask',
                                'volume': 'Vol.',
                                'openInterest': 'OI',
                            }
                            display_df = display_df.rename(columns=col_rename)
                            if 'Strike ($)' in display_df.columns:
                                display_df = display_df.sort_values('Strike ($)')
                            st.dataframe(display_df, use_container_width=True,
                                         hide_index=True, height=210)

                            # Selector de strike → aplicar IV
                            st.markdown("**Selecciona un contrato para usar su IV en el modelo:**")
                            strikes_available = sorted(chain_df['strike'].tolist())
                            atm_strike = min(strikes_available, key=lambda x: abs(x - S))
                            atm_idx = strikes_available.index(atm_strike)
                            sel_strike = st.selectbox(
                                "Strike",
                                strikes_available,
                                index=atm_idx,
                                format_func=lambda x: f"${x:.2f}",
                                key="crr_mkt_strike"
                            )
                            mkt_iv = _iv_for_strike(chain_df, sel_strike)

                            if mkt_iv is not None:
                                itm_flag = (
                                    (mkt_opt_type == "call" and S > sel_strike) or
                                    (mkt_opt_type == "put"  and S < sel_strike)
                                )
                                st.markdown(
                                    f"<div style='background:{SURFACE};border:1px solid {GOLD_BORDER};"
                                    f"border-radius:8px;padding:12px 16px;margin-top:8px;'>"
                                    f"<div style='color:{TEXT_MUTED};font-size:10px;"
                                    f"text-transform:uppercase;letter-spacing:1px;'>IV de mercado · ${sel_strike:.2f}</div>"
                                    f"<div style='color:{GOLD};font-size:22px;font-weight:700;"
                                    f"font-family:Georgia,serif;'>{mkt_iv*100:.2f}%"
                                    f"<span style='color:{TEXT_MUTED};font-size:12px;"
                                    f"font-weight:400;margin-left:10px;font-family:Inter,sans-serif;'>"
                                    f"{'✅ ITM' if itm_flag else '⬜ OTM/ATM'}</span></div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                                if st.button(
                                    f"⬆️ Usar IV {mkt_iv*100:.1f}% · K ${sel_strike:.2f} en el modelo",
                                    type="primary",
                                    key="crr_use_mkt_iv",
                                    use_container_width=True
                                ):
                                    st.session_state["_pending_crr_sigma"] = round(mkt_iv * 100, 2)
                                    st.session_state["_pending_crr_K"]     = float(sel_strike)
                                    try:
                                        from datetime import datetime as _dt
                                        _exp_date = _dt.strptime(sel_expiry, "%Y-%m-%d").date()
                                        st.session_state["_pending_crr_expiry"] = _exp_date
                                    except Exception:
                                        pass
                                    st.success(
                                        f"✅ Cargado: σ = {mkt_iv*100:.2f}%, "
                                        f"K = ${sel_strike:.2f}, vto. {sel_expiry}"
                                    )
                                    st.rerun()
            else:
                st.info(
                    "💡 Introduce un ticker en la izquierda para cargar "
                    "la cadena de opciones real del mercado y usar la IV implícita."
                )

        # ── Calcular ──────────────────────────────────────────────────────────
        with st.spinner("Calculando árbol..."):
            result  = crr_price(S, K, T, r, sigma, N, option_type, style, q)
            greeks  = crr_greeks(S, K, T, r, sigma, min(N, 50), option_type, style, q)
            bs_price = black_scholes(S, K, T, r, sigma, option_type, q) if style == "european" else None

        gold_divider()
        section_label("Resultados")

        # ── KPIs precio y greeks ──────────────────────────────────────────────
        moneyness = S / K
        itm = (option_type == "call" and S > K) or (option_type == "put" and S < K)
        intrinsic = max(0.0, S - K if option_type == "call" else K - S)
        time_value = result["price"] - intrinsic

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card(
                f"Precio CRR ({option_type.upper()})",
                f"${result['price']:.4f}",
                f"{'ITM' if itm else 'OTM/ATM'} · Intrínseco: ${intrinsic:.4f}",
                color=POSITIVE if itm else GOLD
            ), unsafe_allow_html=True)
        with c2:
            if bs_price is not None:
                diff = result["price"] - bs_price
                st.markdown(kpi_card(
                    "Black-Scholes (ref.)",
                    f"${bs_price:.4f}",
                    f"Diferencia: ${diff:+.5f}",
                    color=POSITIVE if abs(diff) < 0.01 else GOLD
                ), unsafe_allow_html=True)
            else:
                st.markdown(kpi_card(
                    "Valor tiempo",
                    f"${time_value:.4f}",
                    f"Moneyness: {moneyness:.3f}"
                ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card(
                "u / d (factores)",
                f"{result['u']:.5f} / {result['d']:.5f}",
                f"p* = {result['p']:.4f} · dt={result['dt']:.6f}a"
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("N pasos", str(N), f"T={T:.4f}a · σ={sigma*100:.1f}%"), unsafe_allow_html=True)

        # ── Greeks ────────────────────────────────────────────────────────────
        gold_divider()
        section_label("Greeks (diferencias finitas)")

        g1, g2, g3, g4, g5 = st.columns(5)
        greek_data = [
            ("Delta", greeks["delta"], "Sensibilidad al precio del subyacente (∂V/∂S)", "neutral"),
            ("Gamma", greeks["gamma"], "Cambio del Delta (∂²V/∂S²)", "neutral"),
            ("Theta", greeks["theta"], "Decaimiento temporal (∂V/∂t por día)", "negative"),
            ("Vega",  greeks["vega"],  "Sensibilidad a la volatilidad por 1%", "positive"),
            ("Rho",   greeks["rho"],   "Sensibilidad a la tasa de interés por 1%", "neutral"),
        ]
        greek_cols = [g1, g2, g3, g4, g5]
        for col, (name, val, desc, tone) in zip(greek_cols, greek_data):
            color = POSITIVE if tone == "positive" else NEGATIVE if tone == "negative" else GOLD
            with col:
                st.markdown(f"""
                <div style='background:{SURFACE_2};border:1px solid {GOLD_BORDER};
                    border-radius:8px;padding:10px;text-align:center;'>
                    <div style='color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;'>{name}</div>
                    <div style='color:{color};font-size:1.4rem;font-weight:700;'>{val:+.4f}</div>
                    <div style='color:{TEXT_MUTED};font-size:0.7rem;margin-top:2px;'>{desc[:40]}</div>
                </div>""", unsafe_allow_html=True)

        # ── Visualización del árbol (N_vis pasos) ─────────────────────────────
        gold_divider()
        N_vis = min(N, 6)
        section_label(f"Árbol de Precios (primeros {N_vis} pasos de {N})")

        vis = crr_price(S, K, T, r, sigma, N_vis, option_type, style, q)
        st_tree = vis["stock_tree"]
        op_tree = vis["option_tree"]

        fig_tree = go.Figure()
        node_x, node_y, node_text, node_color = [], [], [], []
        edge_x, edge_y = [], []

        for step in range(N_vis + 1):
            for j in range(step + 1):
                sx = float(stk_val := st_tree[j, step])
                ox = float(op_tree[j, step])
                is_itm = (option_type == "call" and sx > K) or (option_type == "put" and sx < K)
                node_x.append(step)
                node_y.append(step / 2 - j)
                node_text.append(f"S={stk_val:.2f}<br>V={ox:.4f}")
                node_color.append(POSITIVE if is_itm else SURFACE_2)
                if step > 0:
                    # Up edge (from j, step-1)
                    if j < step:
                        edge_x += [step - 1, step, None]
                        edge_y += [((step-1)/2 - j), (step/2 - j), None]
                    # Down edge (from j-1, step-1)
                    if j > 0:
                        edge_x += [step - 1, step, None]
                        edge_y += [((step-1)/2 - (j-1)), (step/2 - j), None]

        fig_tree.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color=BORDER, width=1),
            hoverinfo='skip', showlegend=False,
        ))
        fig_tree.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=28, color=node_color,
                       line=dict(color=GOLD_BORDER, width=1)),
            text=[t.split('<br>')[0].replace('S=', '') for t in node_text],
            textposition='middle center',
            textfont=dict(size=8, color=TEXT_PRIMARY),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False,
        ))
        fig_tree.update_layout(**plotly_layout(
            height=max(320, 60 * (N_vis + 2)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(x=step, y=(step/2 + 0.55), text=f"t={step*vis['dt']:.3f}a",
                     showarrow=False,
                     font=dict(size=9, color=TEXT_MUTED), xanchor='center')
                for step in range(N_vis + 1)
            ]
        ))
        st.plotly_chart(fig_tree, use_container_width=True)
        st.caption(f"Verde = ITM. Nodos: precio del subyacente (grande) / valor opción (tooltip). "
                   f"Árbol completo tiene {N} pasos; se muestran {N_vis} para legibilidad.")

        # ── Sensibilidad precio a σ y S ───────────────────────────────────────
        gold_divider()
        section_label("Análisis de Sensibilidad")

        col_sv1, col_sv2 = st.columns(2)
        with col_sv1:
            sigmas  = np.linspace(max(0.01, sigma * 0.3), sigma * 2.5, 40)
            prices_sv = [crr_price(S, K, T, r, s, min(N, 50), option_type, style, q)["price"]
                         for s in sigmas]
            bs_sv = [black_scholes(S, K, T, r, s, option_type, q) for s in sigmas] if style == "european" else None
            fig_sv = go.Figure()
            fig_sv.add_trace(go.Scatter(
                x=sigmas * 100, y=prices_sv, mode='lines',
                line=dict(color=GOLD, width=2), name="CRR",
                hovertemplate="σ=%{x:.1f}%<br>V=%{y:.4f}<extra></extra>"
            ))
            if bs_sv:
                fig_sv.add_trace(go.Scatter(
                    x=sigmas * 100, y=bs_sv, mode='lines',
                    line=dict(color=POSITIVE, width=1.5, dash='dash'), name="Black-Scholes",
                ))
            fig_sv.add_vline(x=sigma * 100, line=dict(color=BORDER, dash='dot'))
            fig_sv.update_layout(**plotly_layout(height=280,
                                  xaxis_title="Volatilidad (%)",
                                  yaxis_title="Precio opción ($)",
                                  margin=dict(l=40, r=20, t=20, b=40)))
            st.plotly_chart(fig_sv, use_container_width=True)

        with col_sv2:
            spots = np.linspace(S * 0.5, S * 1.5, 40)
            prices_sp = [crr_price(s, K, T, r, sigma, min(N, 50), option_type, style, q)["price"]
                         for s in spots]
            intrinsics = [max(0.0, s - K if option_type == "call" else K - s) for s in spots]
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=spots, y=prices_sp, mode='lines',
                line=dict(color=GOLD, width=2), name="CRR",
                hovertemplate="S=%{x:.2f}<br>V=%{y:.4f}<extra></extra>"
            ))
            fig_sp.add_trace(go.Scatter(
                x=spots, y=intrinsics, mode='lines',
                line=dict(color=NEGATIVE, width=1.5, dash='dash'), name="Valor intrínseco",
            ))
            fig_sp.add_vline(x=S, line=dict(color=BORDER, dash='dot'),
                             annotation_text=f"S={S:.2f}")
            fig_sp.add_vline(x=K, line=dict(color=GOLD_BORDER, dash='dot'),
                             annotation_text=f"K={K:.2f}")
            fig_sp.update_layout(**plotly_layout(height=280,
                                  xaxis_title="Precio subyacente ($)",
                                  yaxis_title="Precio opción ($)",
                                  margin=dict(l=40, r=20, t=20, b=40)))
            st.plotly_chart(fig_sp, use_container_width=True)

        # ── Smile de volatilidad del mercado ──────────────────────────────────
        if live_ticker and live_ticker.strip():
            tk_up = live_ticker.strip().upper()
            expiries_smile = fetch_option_expiries(tk_up)
            if expiries_smile:
                gold_divider()
                section_label(f"Smile de Volatilidad — {tk_up}")
                st.caption(
                    "Volatilidad implícita extraída de la cadena de opciones real del mercado. "
                    "Una curva plana indica pocos skews; una curva inclinada revela "
                    "expectativas de cola (put skew) o momentum alcista (call skew)."
                )

                smile_col1, smile_col2 = st.columns([1, 3])
                with smile_col1:
                    smile_expiry = st.selectbox(
                        "Vencimiento",
                        expiries_smile,
                        key="crr_smile_expiry",
                        help="Selecciona el vencimiento para visualizar el smile."
                    )
                    show_calls = st.checkbox("Calls", value=True, key="crr_smile_calls")
                    show_puts  = st.checkbox("Puts",  value=True, key="crr_smile_puts")

                with smile_col2:
                    fig_smile = go.Figure()

                    if show_calls:
                        with st.spinner("Cargando cadena calls..."):
                            c_chain = fetch_option_chain(tk_up, smile_expiry, "call")
                        if not c_chain.empty and 'impliedVolatility' in c_chain.columns:
                            fig_smile.add_trace(go.Scatter(
                                x=c_chain['strike'],
                                y=c_chain['impliedVolatility'] * 100,
                                mode='lines+markers',
                                name='Call IV',
                                line=dict(color=POSITIVE, width=2),
                                marker=dict(size=5),
                                hovertemplate="Strike: $%{x:.2f}<br>IV (call): %{y:.2f}%<extra></extra>"
                            ))

                    if show_puts:
                        with st.spinner("Cargando cadena puts..."):
                            p_chain = fetch_option_chain(tk_up, smile_expiry, "put")
                        if not p_chain.empty and 'impliedVolatility' in p_chain.columns:
                            fig_smile.add_trace(go.Scatter(
                                x=p_chain['strike'],
                                y=p_chain['impliedVolatility'] * 100,
                                mode='lines+markers',
                                name='Put IV',
                                line=dict(color=NEGATIVE, width=2),
                                marker=dict(size=5),
                                hovertemplate="Strike: $%{x:.2f}<br>IV (put): %{y:.2f}%<extra></extra>"
                            ))

                    if S > 0:
                        fig_smile.add_vline(
                            x=S,
                            line=dict(color=GOLD, dash='dot', width=1.5),
                            annotation_text=f"S={S:.2f}",
                            annotation_font=dict(color=GOLD, size=10)
                        )
                    fig_smile.add_hline(
                        y=sigma * 100,
                        line=dict(color=GOLD_BORDER, dash='dash', width=1),
                        annotation_text=f"sigma modelo={sigma*100:.1f}%",
                        annotation_font=dict(color=TEXT_MUTED, size=9)
                    )
                    fig_smile.update_layout(**plotly_layout(
                        height=320,
                        xaxis_title="Strike ($)",
                        yaxis_title="Volatilidad implicita (%)",
                        legend=dict(orientation="h", y=1.05, x=0,
                                    font=dict(size=11, color=TEXT_SECONDARY)),
                        margin=dict(l=40, r=20, t=30, b=40),
                        hovermode="x unified",
                    ))
                    if fig_smile.data:
                        st.plotly_chart(fig_smile, use_container_width=True)
                    else:
                        st.info("No hay datos de IV disponibles para este vencimiento.")

    # =====================================================================
    # TAB 2: CAPM EX-ANTE / EX-POST
    # =====================================================================
    with tab_capm:
        section_label("CAPM ex-ante vs ex-post — Alpha de Jensen")
        st.caption(
            "Ex-ante: retorno esperado = rx + beta x ERP. "
            "Ex-post: compara el retorno realizado con lo que el CAPM predecia dado el beta y el mercado real."
        )

        portfolio_data = ensure_portfolio_data()
        if portfolio_data is None or portfolio_data.empty:
            no_portfolio_warning()
        else:
            if "Asset Type" not in portfolio_data.columns:
                portfolio_data = portfolio_data.copy()
                portfolio_data["Asset Type"] = "equity"

            df_eq = portfolio_data[
                portfolio_data["Asset Type"].fillna("equity").str.lower().isin(
                    ["equity", "etf", "crypto", "fund", ""]
                )
            ].copy()
            tickers = tuple(t for t in df_eq["Ticker"].unique() if t)

            if not tickers:
                st.warning("No hay activos equity en el portfolio.")
            else:
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                with col_cfg1:
                    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "EEM"],
                                             key="capm_bench")
                with col_cfg2:
                    period = st.selectbox("Periodo historico",
                                          ["1y", "2y", "3y", "5y", "10y"],
                                          index=2, key="capm_period")
                with col_cfg3:
                    rf_pct = st.slider("Tasa libre de riesgo anual (%)", 0.0, 10.0, 4.5,
                                       step=0.1, key="capm_rf")
                    rf = rf_pct / 100

                with st.spinner("Calculando CAPM..."):
                    capm_df = _compute_capm_data(tickers, benchmark, period, rf)

                if capm_df.empty:
                    st.warning("No se pudieron obtener datos suficientes.")
                else:
                    gold_divider()
                    avg_alpha = capm_df["Alpha Jensen (%)"].mean()
                    n_pos_alpha = (capm_df["Alpha Jensen (%)"] > 0).sum()
                    avg_beta    = capm_df["Beta"].mean()

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(kpi_card(
                            "Alpha Jensen medio", f"{avg_alpha:+.2f}%",
                            "retorno realizado vs CAPM ex-post",
                            color=POSITIVE if avg_alpha > 0 else NEGATIVE
                        ), unsafe_allow_html=True)
                    with c2:
                        st.markdown(kpi_card(
                            "Activos con alpha > 0", f"{n_pos_alpha}/{len(capm_df)}",
                            "superan al CAPM ex-post",
                            color=POSITIVE if n_pos_alpha > len(capm_df) / 2 else NEGATIVE
                        ), unsafe_allow_html=True)
                    with c3:
                        st.markdown(kpi_card(
                            "Beta medio cartera", f"{avg_beta:.3f}",
                            f"vs {benchmark}", color=GOLD
                        ), unsafe_allow_html=True)
                    with c4:
                        erp = 0.06
                        capm_portf = rf + avg_beta * erp
                        st.markdown(kpi_card(
                            "CAPM ex-ante cartera", f"{capm_portf*100:.2f}%",
                            f"rx={rf*100:.1f}% + beta x ERP({erp*100:.0f}%)"
                        ), unsafe_allow_html=True)

                    gold_divider()
                    section_label("Security Market Line (SML)")

                    betas_range = np.linspace(0, capm_df["Beta"].max() * 1.3 + 0.3, 50)
                    erp_assumed = 0.06
                    sml_returns = rf * 100 + betas_range * erp_assumed * 100

                    fig_sml = go.Figure()
                    fig_sml.add_trace(go.Scatter(
                        x=betas_range, y=sml_returns, mode='lines',
                        line=dict(color=GOLD, width=2, dash='dash'),
                        name=f"SML (ERP={erp_assumed*100:.0f}%)",
                    ))
                    for _, row in capm_df.iterrows():
                        above = row["Retorno real. (%)"] > row["CAPM ex-post (%)"]
                        fig_sml.add_trace(go.Scatter(
                            x=[row["Beta"]], y=[row["Retorno real. (%)"]],
                            mode='markers+text',
                            marker=dict(size=12,
                                        color=POSITIVE if above else NEGATIVE,
                                        symbol='circle',
                                        line=dict(color=GOLD_BORDER, width=1)),
                            text=[row["Ticker"]],
                            textposition='top center',
                            textfont=dict(size=10, color=TEXT_PRIMARY),
                            name=row["Ticker"],
                            hovertemplate=(
                                f"<b>{row['Ticker']}</b><br>"
                                f"Beta: {row['Beta']:.3f}<br>"
                                f"Retorno real.: {row['Retorno real. (%)']:.2f}%<br>"
                                f"CAPM ex-post: {row['CAPM ex-post (%)']:.2f}%<br>"
                                f"Alpha Jensen: {row['Alpha Jensen (%)']:+.2f}%"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        ))
                    fig_sml.add_hline(y=rf * 100, line=dict(color=BORDER, dash='dot'),
                                      annotation_text=f"rx={rf*100:.1f}%")
                    fig_sml.update_layout(**plotly_layout(
                        height=400,
                        xaxis_title="Beta",
                        yaxis_title="Retorno anualizado (%)",
                        margin=dict(l=50, r=20, t=30, b=50),
                    ))
                    st.plotly_chart(fig_sml, use_container_width=True)

                    gold_divider()
                    section_label("Tabla detallada por activo")

                    def _style_capm(row):
                        styles_list = []
                        for col in capm_df.columns:
                            if col == "Alpha Jensen (%)":
                                v = row[col]
                                s = f"color:{POSITIVE};font-weight:600" if v > 0 else f"color:{NEGATIVE};font-weight:600"
                            elif col == "Beta":
                                v = row[col]
                                s = f"color:{POSITIVE}" if 0 < v < 1 else f"color:{NEGATIVE}" if v > 1.5 else f"color:{GOLD}"
                            elif col == "Retorno real. (%)":
                                v = row[col]
                                s = f"color:{POSITIVE}" if v > 0 else f"color:{NEGATIVE}"
                            else:
                                s = ""
                            styles_list.append(s)
                        return styles_list

                    st.dataframe(
                        capm_df.reset_index(drop=True).style.apply(_style_capm, axis=1),
                        use_container_width=True, hide_index=True
                    )

                    gold_divider()
                    section_label("Alpha de Jensen por activo")

                    sorted_capm = capm_df.sort_values("Alpha Jensen (%)", ascending=True)
                    bar_colors  = [POSITIVE if v > 0 else NEGATIVE
                                   for v in sorted_capm["Alpha Jensen (%)"]]
                    fig_alpha = go.Figure(go.Bar(
                        x=sorted_capm["Alpha Jensen (%)"],
                        y=sorted_capm["Ticker"],
                        orientation='h',
                        marker_color=bar_colors,
                        hovertemplate="%{y}: alpha=%{x:+.2f}%<extra></extra>",
                    ))
                    fig_alpha.add_vline(x=0, line=dict(color=BORDER, width=1))
                    fig_alpha.update_layout(**plotly_layout(
                        height=max(250, len(capm_df) * 45),
                        xaxis_title="Alpha Jensen (%)",
                        margin=dict(l=80, r=20, t=20, b=40),
                    ))
                    st.plotly_chart(fig_alpha, use_container_width=True)

                    with st.expander("Como interpretar el CAPM ex-ante / ex-post"):
                        st.markdown(f"""
**CAPM ex-ante** = rx + beta x ERP esperado
Responde a: que retorno deberia exigirle a este activo dado su riesgo?
Usamos ERP historico largo plazo = 6% (Damodaran). rx actual = {rf*100:.1f}%.

**CAPM ex-post** = rx + beta x (r_mercado_realizado - rx)
Responde a: cuanto deberia haber ganado dado el beta y lo que hizo el mercado?

**Alpha de Jensen** = Retorno realizado - CAPM ex-post
- alpha > 0: el activo genero mas retorno del que su beta justifica.
- alpha < 0: el activo destruyo valor ajustado por riesgo.

**SML (Security Market Line)**: los activos sobre la linea superan las expectativas del CAPM.
                        """)

"""
Stress Test & Horizonte Temporal
- Escenarios macro con impacto diferenciado por clase de activo
- Bonos: impacto por duración modificada
- Equity: impacto por beta histórica
- Derivados: impacto por delta estimada
- Proyección de recuperación Monte Carlo a 10 años
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from modules.utils import ensure_portfolio_data, no_portfolio_warning, safe_df_display
from modules.styles import (
    PLOTLY_DARK, GOLD, SURFACE, SURFACE_2, BORDER, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, plotly_layout
)
from modules.i18n import t, get_lang

def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convierte color hex (#rrggbb) a rgba() para Plotly fillcolor."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


DERIV_TYPES = ('option_call', 'option_put', 'future', 'warrant')


def _es():
    return get_lang() == 'es'


# ── Escenarios con parámetros por clase de activo ─────────────────────────────

SCENARIOS = {
    "🦠 Covid-19 — Crash Global (Feb-Mar 2020)": {
        "desc_es": "Caída brusca y recuperación en V. Máximo impacto en equity.",
        "desc_en": "Sharp crash and V-shaped recovery. Maximum impact on equity.",
        "equity_shock":   -0.34,
        "bond_dur_shock": -0.005,   # tipos bajaron → bonos subieron
        "bond_etf_shock": +0.04,
        "fund_shock":     -0.28,
        "deriv_call_mult": 0.1,
        "deriv_put_mult":  2.5,
        "future_shock":   -0.30,
        "stress_days":     126,
        "stress_mu_mult":  3.0,
        "stress_sigma":    2.0,
        "recovery_mu":     1.2,
        "category": "market_crash"
    },
    "💥 Crisis Financiera Global (2008)": {
        "desc_es": "Crisis de crédito sistémica. Caída prolongada en todos los activos.",
        "desc_en": "Systemic credit crisis. Prolonged decline across all assets.",
        "equity_shock":   -0.56,
        "bond_dur_shock": -0.010,
        "bond_etf_shock": +0.08,
        "fund_shock":     -0.45,
        "deriv_call_mult": 0.05,
        "deriv_put_mult":  4.0,
        "future_shock":   -0.50,
        "stress_days":     504,
        "stress_mu_mult": -0.5,
        "stress_sigma":    1.5,
        "recovery_mu":     0.8,
        "category": "market_crash"
    },
    "💻 Burbuja Dot-Com (2000-2002)": {
        "desc_es": "Colapso de valuaciones tech. Recuperación muy lenta.",
        "desc_en": "Tech valuation collapse. Very slow recovery.",
        "equity_shock":   -0.49,
        "bond_dur_shock": -0.008,
        "bond_etf_shock": +0.06,
        "fund_shock":     -0.40,
        "deriv_call_mult": 0.05,
        "deriv_put_mult":  3.5,
        "future_shock":   -0.45,
        "stress_days":     756,
        "stress_mu_mult":  0.1,
        "stress_sigma":    1.3,
        "recovery_mu":     1.0,
        "category": "market_crash"
    },
    "⬛ Lunes Negro (1987)": {
        "desc_es": "Flash crash de un día. Recuperación relativamente rápida.",
        "desc_en": "One-day flash crash. Relatively fast recovery.",
        "equity_shock":   -0.22,
        "bond_dur_shock": -0.003,
        "bond_etf_shock": +0.02,
        "fund_shock":     -0.20,
        "deriv_call_mult": 0.2,
        "deriv_put_mult":  2.0,
        "future_shock":   -0.20,
        "stress_days":     60,
        "stress_mu_mult":  2.0,
        "stress_sigma":    1.8,
        "recovery_mu":     1.0,
        "category": "market_crash"
    },
    "📈 Subida de Tipos +200 pb (Fed / BCE)": {
        "desc_es": "Subida brusca de tipos. Fuerte impacto en bonos de larga duración. Equity tech presionado.",
        "desc_en": "Sharp rate hike. Strong impact on long-duration bonds. Tech equity under pressure.",
        "equity_shock":   -0.15,
        "bond_dur_shock": +0.020,   # +200pb → precio bono cae
        "bond_etf_shock": -0.12,
        "fund_shock":     -0.10,
        "deriv_call_mult": 0.6,
        "deriv_put_mult":  1.5,
        "future_shock":   -0.12,
        "stress_days":     252,
        "stress_mu_mult": -0.2,
        "stress_sigma":    1.4,
        "recovery_mu":     1.1,
        "category": "rates"
    },
    "🔥 Stagflación (Tipos Altos + Crecimiento Bajo)": {
        "desc_es": "Inflación persistente >4% con estancamiento económico. Los bonos y la renta variable sufren simultáneamente.",
        "desc_en": "Persistent inflation >4% with economic stagnation. Bonds and equity suffer simultaneously.",
        "equity_shock":   -0.20,
        "bond_dur_shock": +0.025,
        "bond_etf_shock": -0.18,
        "fund_shock":     -0.15,
        "deriv_call_mult": 0.5,
        "deriv_put_mult":  1.8,
        "future_shock":   -0.15,
        "stress_days":     756,
        "stress_mu_mult":  0.2,
        "stress_sigma":    1.6,
        "recovery_mu":     0.9,
        "category": "macro"
    },
    "💳 Crisis de Crédito — Spreads +300 pb": {
        "desc_es": "Expansión masiva de spreads de crédito. Impacto severo en HY y IG corporativo.",
        "desc_en": "Massive credit spread widening. Severe impact on HY and IG corporate bonds.",
        "equity_shock":   -0.25,
        "bond_dur_shock": +0.030,
        "bond_etf_shock": -0.20,
        "fund_shock":     -0.18,
        "deriv_call_mult": 0.3,
        "deriv_put_mult":  2.5,
        "future_shock":   -0.22,
        "stress_days":     378,
        "stress_mu_mult": -0.3,
        "stress_sigma":    1.5,
        "recovery_mu":     0.9,
        "category": "credit"
    },
    "⚡ Flash Crash / Dislocación de Liquidez": {
        "desc_es": "Evento de liquidez intradiario. Impacto breve pero muy intenso.",
        "desc_en": "Intraday liquidity event. Brief but very intense impact.",
        "equity_shock":   -0.12,
        "bond_dur_shock": +0.005,
        "bond_etf_shock": -0.05,
        "fund_shock":     -0.10,
        "deriv_call_mult": 0.4,
        "deriv_put_mult":  1.8,
        "future_shock":   -0.10,
        "stress_days":     30,
        "stress_mu_mult":  2.5,
        "stress_sigma":    2.2,
        "recovery_mu":     1.1,
        "category": "market_crash"
    },
    "📉 Recesión Técnica (2 trimestres negativos)": {
        "desc_es": "Contracción económica moderada. Equity y high yield afectados.",
        "desc_en": "Moderate economic contraction. Equity and high yield affected.",
        "equity_shock":   -0.18,
        "bond_dur_shock": -0.005,
        "bond_etf_shock": +0.03,
        "fund_shock":     -0.14,
        "deriv_call_mult": 0.5,
        "deriv_put_mult":  1.6,
        "future_shock":   -0.16,
        "stress_days":     504,
        "stress_mu_mult":  0.3,
        "stress_sigma":    1.3,
        "recovery_mu":     1.0,
        "category": "macro"
    },
    "🌍 Shock Geopolítico / Energético": {
        "desc_es": "Crisis de suministro energético o conflicto geopolítico mayor.",
        "desc_en": "Energy supply crisis or major geopolitical conflict.",
        "equity_shock":   -0.12,
        "bond_dur_shock": +0.008,
        "bond_etf_shock": -0.06,
        "fund_shock":     -0.10,
        "deriv_call_mult": 0.5,
        "deriv_put_mult":  1.8,
        "future_shock":   +0.05,    # materias primas suben
        "stress_days":     180,
        "stress_mu_mult":  0.5,
        "stress_sigma":    1.6,
        "recovery_mu":     1.0,
        "category": "geopolitical"
    },
    "🚀 Rally Alcista / Recuperación en V": {
        "desc_es": "Escenario positivo. Fuerte rally de riesgo, yield curve steepening.",
        "desc_en": "Positive scenario. Strong risk rally, yield curve steepening.",
        "equity_shock":   +0.25,
        "bond_dur_shock": +0.010,
        "bond_etf_shock": -0.05,
        "fund_shock":     +0.20,
        "deriv_call_mult": 2.0,
        "deriv_put_mult":  0.2,
        "future_shock":   +0.22,
        "stress_days":     252,
        "stress_mu_mult":  1.5,
        "stress_sigma":    0.8,
        "recovery_mu":     1.0,
        "category": "positive"
    },
}

CATEGORY_COLORS = {
    "market_crash": "#c05a5a",
    "rates":        "#e8a020",
    "macro":        "#5a7abf",
    "credit":       "#8b5abf",
    "geopolitical": "#5a8b7a",
    "positive":     "#5a8f6e",
}


def _get_bond_duration(row) -> float:
    """Estima duración modificada de un bono."""
    try:
        from modules.risk import periods_to_maturity, ytm_from_price, bond_price, macaulay_duration, modified_duration
        face   = float(row.get('Face Value') or 1000)
        coupon = float(row.get('Coupon %') or 0) / 100
        mat    = str(row.get('Maturity') or '')
        periods = periods_to_maturity(mat)
        if periods <= 0:
            return 3.0
        ytm    = ytm_from_price(face, coupon, face, periods)
        mac_d  = macaulay_duration(face, coupon, ytm, periods)
        mod_d  = modified_duration(mac_d, ytm)
        return min(mod_d, 30.0)
    except Exception:
        return 3.0


def _compute_position_impact(row: pd.Series, scenario: dict, total_aum: float) -> dict:
    """
    Calcula el impacto del escenario sobre una posición individual
    teniendo en cuenta el tipo de activo.
    """
    ticker  = str(row.get('Ticker', ''))
    atype   = str(row.get('Asset Type', 'equity'))
    value   = float(row.get('Total Value ($)', 0))
    name    = str(row.get('Name', ticker))

    if atype == 'equity' or atype == 'fund' or atype == 'bond_etf':
        if atype == 'equity':
            shock = scenario['equity_shock']
        elif atype == 'fund':
            shock = scenario['fund_shock']
        else:  # bond_etf
            shock = scenario['bond_etf_shock']
        impact_pct  = shock
        impact_usd  = value * shock
        methodology = f"Equity beta shock: {shock*100:+.1f}%"

    elif atype == 'bond':
        # Impacto por duración: ΔP ≈ -Mod.Duration × ΔYield × Price
        mod_dur = _get_bond_duration(row)
        delta_yield = scenario['bond_dur_shock']
        impact_pct  = -mod_dur * delta_yield
        impact_usd  = value * impact_pct
        methodology = f"Duration {mod_dur:.1f}y × Δyield {delta_yield*100:+.0f}pb = {impact_pct*100:+.1f}%"

    elif atype == 'option_call':
        mult       = float(row.get('Multiplier') or 100)
        contracts  = float(row.get('Shares') or 0)
        premium    = float(row.get('Premium') or 0)
        cost_basis = premium * contracts * mult
        impact_pct  = scenario['deriv_call_mult'] - 1.0
        impact_usd  = cost_basis * impact_pct
        methodology = f"Call ×{scenario['deriv_call_mult']:.1f} sobre prima pagada"

    elif atype == 'option_put':
        mult       = float(row.get('Multiplier') or 100)
        contracts  = float(row.get('Shares') or 0)
        premium    = float(row.get('Premium') or 0)
        cost_basis = premium * contracts * mult
        impact_pct  = scenario['deriv_put_mult'] - 1.0
        impact_usd  = cost_basis * impact_pct
        methodology = f"Put ×{scenario['deriv_put_mult']:.1f} sobre prima pagada (cobertura)"

    elif atype == 'future':
        shock       = scenario['future_shock']
        impact_pct  = shock
        impact_usd  = value * shock
        methodology = f"Futures notional shock: {shock*100:+.1f}%"

    elif atype == 'warrant':
        mult       = float(row.get('Multiplier') or 100)
        contracts  = float(row.get('Shares') or 0)
        premium    = float(row.get('Premium') or 0)
        cost_basis = premium * contracts * mult
        impact_pct  = scenario['deriv_call_mult'] - 1.0
        impact_usd  = cost_basis * impact_pct
        methodology = f"Warrant ×{scenario['deriv_call_mult']:.1f} sobre prima"
    else:
        impact_pct = scenario['equity_shock']
        impact_usd = value * impact_pct
        methodology = f"Equity shock: {impact_pct*100:+.1f}%"

    peso = value / total_aum if total_aum > 0 else 0
    return {
        'Ticker':       ticker,
        'Nombre':       name,
        'Tipo':         atype.replace('_',' ').title(),
        'Valor ($)':    value,
        'Peso (%)':     peso * 100,
        'Impacto (%)':  impact_pct * 100,
        'Impacto ($)':  impact_usd,
        'Metodología':  methodology,
    }


def render_stress_test():
    es = _es()
    st.title(t("stress.title"))

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        no_portfolio_warning()
        return

    tab1, tab2 = st.tabs([t("stress.tab_stress"), t("stress.tab_horizon")])

    with tab1:
        _render_stress(portfolio_df)

    with tab2:
        _render_time_horizon(portfolio_df)


def _render_stress(portfolio_df):
    es = _es()
    total_aum = portfolio_df['Total Value ($)'].sum()
    if total_aum <= 0:
        st.error("El valor total del portfolio debe ser mayor que cero." if es else
                 "Portfolio total value must be greater than zero.")
        return

    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:12px; margin-bottom:20px;'>"
        f"{t('stress.subtitle')}</p>",
        unsafe_allow_html=True
    )

    # ── Selector de escenario ─────────────────────────────────────────────────
    scen_name = st.selectbox(t("stress.select"), list(SCENARIOS.keys()))
    scen = SCENARIOS[scen_name]
    desc = scen['desc_es'] if es else scen['desc_en']
    cat_color = CATEGORY_COLORS.get(scen['category'], '#6b7280')

    st.markdown(f"""
    <div style='background:{SURFACE}; border-left:3px solid {cat_color}; border-radius:0 8px 8px 0;
                padding:12px 18px; margin:10px 0 20px 0;'>
        <span style='color:{cat_color}; font-size:11px; font-weight:700;
                     text-transform:uppercase; letter-spacing:1px;'>
            {scen['category'].replace('_',' ').upper()}
        </span>
        <span style='color:{TEXT_MUTED}; font-size:12px; margin-left:10px;'>{desc}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Calcular impacto por posición ─────────────────────────────────────────
    impacts = []
    for _, row in portfolio_df.iterrows():
        impacts.append(_compute_position_impact(row, scen, total_aum))

    impact_df = pd.DataFrame(impacts)
    total_impact_usd = impact_df['Impacto ($)'].sum()
    total_impact_pct = total_impact_usd / total_aum if total_aum > 0 else 0
    post_shock_value = total_aum + total_impact_usd

    # ── KPIs principales ──────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("stress.current_value"),
        f"${total_aum:,.0f}"
    )
    c2.metric(
        t("stress.post_shock"),
        f"${post_shock_value:,.0f}",
        delta=f"{total_impact_pct*100:+.1f}%",
        delta_color="normal" if total_impact_usd >= 0 else "inverse"
    )
    c3.metric(
        t("stress.loss") if total_impact_usd < 0 else t("stress.gain"),
        f"${abs(total_impact_usd):,.0f}",
        delta=f"{total_impact_pct*100:+.1f}%",
        delta_color="normal" if total_impact_usd >= 0 else "inverse"
    )

    # Indicar qué posiciones actúan como cobertura
    hedges = impact_df[impact_df['Impacto ($)'] > 0]
    losses = impact_df[impact_df['Impacto ($)'] < 0]
    hedge_total = hedges['Impacto ($)'].sum()
    c4.metric(
        "Coberturas naturales" if es else "Natural hedges",
        f"${hedge_total:,.0f}",
        delta=f"{len(hedges)} posiciones" if es else f"{len(hedges)} positions",
        delta_color="normal" if hedge_total > 0 else "off"
    )

    # ── Tabla de impacto por posición ────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;'>"
        f"{'IMPACTO POR POSICIÓN' if es else 'IMPACT PER POSITION'}</p>",
        unsafe_allow_html=True
    )

    display_df = impact_df.copy()
    display_df['Valor ($)']   = display_df['Valor ($)'].apply(lambda x: f"${x:,.0f}")
    display_df['Peso (%)']    = display_df['Peso (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['Impacto (%)'] = display_df['Impacto (%)'].apply(lambda x: f"{x:+.1f}%")
    display_df['Impacto ($)'] = display_df['Impacto ($)'].apply(
        lambda x: f"▲ ${x:,.0f}" if x >= 0 else f"▼ -${abs(x):,.0f}"
    )
    cols_show = ['Ticker', 'Nombre', 'Tipo', 'Valor ($)', 'Peso (%)',
                 'Impacto (%)', 'Impacto ($)', 'Metodología']
    if not es:
        display_df = display_df.rename(columns={
            'Nombre': 'Name', 'Tipo': 'Type', 'Peso (%)': 'Weight (%)',
            'Impacto (%)': 'Impact (%)', 'Impacto ($)': 'Impact ($)',
            'Metodología': 'Methodology'
        })
        cols_show = ['Ticker', 'Name', 'Type', 'Valor ($)', 'Weight (%)',
                     'Impact (%)', 'Impact ($)', 'Methodology']
    st.dataframe(safe_df_display(display_df[cols_show]), use_container_width=True, hide_index=True)

    # ── Gráfico de impacto por posición ──────────────────────────────────────
    fig_bar = go.Figure()
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in impact_df['Impacto ($)']]
    fig_bar.add_trace(go.Bar(
        x=impact_df['Ticker'],
        y=impact_df['Impacto ($)'],
        marker_color=colors,
        text=[f"${v:+,.0f}" for v in impact_df['Impacto ($)']],
        textposition='outside',
        textfont=dict(size=11, color='#9ca3af'),
        hovertemplate=(
            "<b>%{x}</b><br>"
            + ("Impacto: $%{y:,.0f}" if es else "Impact: $%{y:,.0f}")
            + "<extra></extra>"
        )
    ))
    fig_bar.update_layout(**plotly_layout(
        title=dict(
            text=f"{'Impacto por posición — ' if es else 'Impact per position — '}{scen_name}",
            font=dict(color='#9ca3af', size=12)
        ),
        xaxis_title="Ticker",
        yaxis_title="P&L ($)",
        height=320,
        showlegend=False,
    ))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Desglose por clase de activo ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;'>"
        f"{'IMPACTO POR CLASE DE ACTIVO' if es else 'IMPACT BY ASSET CLASS'}</p>",
        unsafe_allow_html=True
    )
    by_type = (
        impact_df.groupby('Tipo')[['Valor ($)', 'Impacto ($)']].sum().reset_index()
    )
    by_type['Impacto (%)'] = by_type['Impacto ($)'] / by_type['Valor ($)'] * 100
    by_type_disp = by_type.copy()
    by_type_disp['Valor ($)']   = by_type_disp['Valor ($)'].apply(lambda x: f"${x:,.0f}")
    by_type_disp['Impacto ($)'] = by_type_disp['Impacto ($)'].apply(
        lambda x: f"▲ ${x:,.0f}" if x >= 0 else f"▼ -${abs(x):,.0f}"
    )
    by_type_disp['Impacto (%)'] = by_type_disp['Impacto (%)'].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(safe_df_display(by_type_disp), use_container_width=True, hide_index=True)

    # ── Nota metodológica ─────────────────────────────────────────────────────
    bond_note = (
        "Bonos individuales: impacto calculado via Duración Modificada × ΔYield."
        if es else
        "Individual bonds: impact calculated via Modified Duration × ΔYield."
    )
    deriv_note = (
        "Opciones: impacto sobre la prima pagada (coste). Puts actúan como cobertura."
        if es else
        "Options: impact on premium paid (cost basis). Puts act as hedges."
    )
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:11px; margin-top:10px;'>"
        f"💡 {bond_note} {deriv_note}</p>",
        unsafe_allow_html=True
    )

    # ── Trayectoria histórica + proyección de recuperación ───────────────────
    st.markdown("---")
    st.markdown(f"### {t('stress.chart_title')}")

    with st.spinner("Calculando trayectoria de 20 años..." if es else "Calculating 20-year trajectory..."):
        try:
            # Filtrar solo equity/ETF/fund para datos históricos
            hist_mask = portfolio_df['Asset Type'].isin(['equity', 'bond_etf', 'fund']) \
                if 'Asset Type' in portfolio_df.columns \
                else pd.Series([True] * len(portfolio_df))
            hist_df = portfolio_df[hist_mask]

            if hist_df.empty:
                hist_df = portfolio_df

            tickers    = hist_df['Ticker'].tolist()
            start_date = datetime.today() - timedelta(days=365 * 10)

            data_raw = yf.download(tickers, start=start_date, end=datetime.today(),
                                   auto_adjust=True, progress=False, threads=False)
            if isinstance(data_raw.columns, pd.MultiIndex):
                data = data_raw['Close']
            else:
                data = data_raw[['Close']].copy()
                if len(tickers) == 1:
                    data.columns = tickers

            data = data.ffill().bfill().dropna()
            if data.empty:
                st.warning("No hay suficientes datos históricos para el gráfico de trayectoria."
                           if es else "Not enough historical data for trajectory chart.")
            else:
                hist_total = hist_df['Total Value ($)'].sum()
                weights    = (hist_df['Total Value ($)'] / hist_total).values if hist_total > 0 \
                             else np.ones(len(tickers)) / len(tickers)

                valid_tickers = [t for t in tickers if t in data.columns]
                if valid_tickers:
                    valid_weights = (hist_df[hist_df['Ticker'].isin(valid_tickers)]['Total Value ($)']
                                     / hist_total).values
                    returns_hist  = data[valid_tickers].pct_change().dropna()
                    port_ret      = (returns_hist * valid_weights).sum(axis=1)
                    cumulative    = (1 + port_ret).cumprod()
                    scale         = total_aum / cumulative.iloc[-1]
                    hist_curve    = cumulative * scale

                    # Parámetros para la proyección
                    mu_d    = port_ret.mean()
                    sigma_d = port_ret.std()
                    days    = 252 * 10
                    n_sims  = 300

                    sims = np.zeros((days, n_sims))
                    sims[0] = post_shock_value
                    stress_d = scen['stress_days']
                    stress_mu = mu_d * scen['stress_mu_mult']
                    stress_s  = sigma_d * scen['stress_sigma']
                    rec_mu    = mu_d * scen['recovery_mu']

                    for i in range(1, days):
                        cur_mu    = stress_mu if i < stress_d else rec_mu
                        cur_sigma = stress_s  if i < stress_d else sigma_d
                        sims[i]   = sims[i-1] * (1 + np.random.normal(cur_mu, cur_sigma, n_sims))

                    p10 = np.percentile(sims, 10, axis=1)
                    p50 = np.percentile(sims, 50, axis=1)
                    p90 = np.percentile(sims, 90, axis=1)

                    last_date    = hist_curve.index[-1]
                    future_dates = [last_date + timedelta(days=int(i / 252 * 365))
                                    for i in range(days)]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_curve.index, y=hist_curve.values,
                        mode='lines', name='Histórico (10 años)' if es else 'Historical (10Y)',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[last_date, last_date],
                        y=[total_aum, post_shock_value],
                        mode='lines+markers',
                        name=f"Shock: {scen_name[:30]}",
                        line=dict(color=cat_color, width=3, dash='dash'),
                        marker=dict(size=8, color=cat_color)
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=p90, mode='lines',
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=p10, mode='lines',
                        line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(0,168,107,0.12)',
                        name='Rango 80% MC' if es else '80% MC Range'
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=p50, mode='lines',
                        name='Recuperación esperada' if es else 'Expected recovery',
                        line=dict(color='#22c55e', width=2, dash='dot')
                    ))
                    lyt = {**PLOTLY_DARK}
                    lyt['legend'] = dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="right", x=1,
                                         bgcolor='rgba(0,0,0,0)',
                                         font=dict(color='#6b7280'))
                    fig.update_layout(
                        **lyt,
                        title=dict(
                            text=f"{'Portfolio · 10 años histórico → shock → 10 años proyectados' if es else 'Portfolio · 10Y history → shock → 10Y projection'}",
                            font=dict(color='#9ca3af', size=12)
                        ),
                        xaxis_title="Fecha" if es else "Date",
                        yaxis_title="Valor ($)" if es else "Value ($)",
                        height=480,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # KPIs de proyección
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric(t("stress.invested_10y"), f"${hist_curve.iloc[0]:,.0f}")
                    c_b.metric(t("stress.return_10y"),
                               f"+{((total_aum / hist_curve.iloc[0]) - 1)*100:.1f}%")
                    c_c.metric(t("stress.future_return"),
                               f"{((p50[-1] / post_shock_value) - 1)*100:+.1f}%")

        except Exception as e:
            st.error(f"Error al calcular trayectoria: {e}" if es else f"Error calculating trajectory: {e}")


def _render_time_horizon(portfolio_df):
    es = _es()
    st.caption(t("horizon.subtitle"))

    total_aum = portfolio_df['Total Value ($)'].sum()
    if total_aum <= 0:
        st.error("El valor total del portfolio debe ser mayor que cero.")
        return

    tab_back, tab_proj = st.tabs([t("horizon.tab_backtest"), t("horizon.tab_proj")])

    with tab_back:
        st.caption(t("horizon.backtest_sub"))
        horizon_years = st.selectbox(t("horizon.period"), [1, 3, 5, 10, 20, 30], index=3)

        if st.button(t("horizon.btn_backtest"), type="primary", key="th_backtest"):
            with st.spinner("Descargando datos históricos..." if es else "Downloading historical data..."):
                # Usar solo tickers equity para el backtest
                hist_mask = portfolio_df['Asset Type'].isin(['equity', 'bond_etf', 'fund']) \
                    if 'Asset Type' in portfolio_df.columns \
                    else pd.Series([True] * len(portfolio_df))
                hist_df = portfolio_df[hist_mask] if hist_mask.any() else portfolio_df
                tickers = hist_df['Ticker'].tolist()
                hist_total = hist_df['Total Value ($)'].sum()

                try:
                    raw = yf.download(
                        tickers,
                        start=datetime.today() - timedelta(days=365 * horizon_years),
                        end=datetime.today(),
                        auto_adjust=True, progress=False, threads=False
                    )
                    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) \
                        else raw[['Close']].rename(columns={'Close': tickers[0]})
                    data = data.ffill().dropna()

                    if data.empty:
                        st.error("No hay datos suficientes para este periodo." if es else
                                 "Not enough data for this period.")
                        return

                    valid  = [t for t in tickers if t in data.columns]
                    vw     = (hist_df[hist_df['Ticker'].isin(valid)]['Total Value ($)'] / hist_total).values
                    ret    = data[valid].pct_change().dropna()
                    pret   = (ret * vw).sum(axis=1)
                    cumul  = (1 + pret).cumprod() * 10000
                    final  = cumul.iloc[-1]
                    cagr   = (final / 10000) ** (1 / horizon_years) - 1
                    vol    = pret.std() * np.sqrt(252)
                    sharpe = (pret.mean() * 252 - 0.04) / (vol if vol > 0 else 1)

                    color = '#22c55e' if final >= 10000 else '#ef4444'
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumul.index, y=cumul.values, mode='lines',
                        fill='tozeroy', fillcolor=_hex_to_rgba(color),
                        line=dict(color=color, width=2),
                        hovertemplate='%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>'
                    ))
                    fig.update_layout(**plotly_layout(
                        title=dict(
                            text=f"{'Crecimiento de $10.000 —' if es else 'Growth of $10,000 —'} {horizon_years} {'años' if es else 'years'}",
                            font=dict(color='#9ca3af', size=12)
                        ),
                        yaxis_title="Valor ($)" if es else "Value ($)",
                        height=340,
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(t("horizon.initial"), "$10,000")
                    c2.metric(t("horizon.final"), f"${final:,.0f}")
                    c3.metric(t("horizon.cagr"), f"{cagr*100:.2f}%")
                    c4.metric("Sharpe", f"{sharpe:.2f}")

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_proj:
        st.caption(t("horizon.proj_sub"))
        proj_years = st.slider(
            "Horizonte de proyección (años)" if es else "Projection horizon (years)",
            min_value=1, max_value=20, value=10
        )
        regime = st.selectbox(
            "Régimen de mercado" if es else "Market regime",
            options=["neutral", "bull", "bear"],
            format_func=lambda x: {
                "neutral": "⚖️ Neutral / Histórico",
                "bull":    "🐂 Bull Market (+25% retorno anual)",
                "bear":    "🐻 Bear Market (-15% retorno anual)"
            }[x] if es else {
                "neutral": "⚖️ Neutral / Historical",
                "bull":    "🐂 Bull Market (+25% annual return)",
                "bear":    "🐻 Bear Market (-15% annual return)"
            }[x]
        )

        if st.button(t("horizon.btn_proj"), type="primary", key="th_proj"):
            with st.spinner("Simulando trayectorias..." if es else "Simulating trajectories..."):
                hist_mask = portfolio_df['Asset Type'].isin(['equity', 'bond_etf', 'fund']) \
                    if 'Asset Type' in portfolio_df.columns \
                    else pd.Series([True] * len(portfolio_df))
                hist_df = portfolio_df[hist_mask] if hist_mask.any() else portfolio_df
                tickers   = hist_df['Ticker'].tolist()
                hist_total = hist_df['Total Value ($)'].sum()

                try:
                    raw = yf.download(tickers, period='3y', auto_adjust=True, progress=False, threads=False)
                    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) \
                        else raw[['Close']].rename(columns={'Close': tickers[0]})
                    data = data.ffill().dropna()

                    valid = [t for t in tickers if t in data.columns]
                    vw    = (hist_df[hist_df['Ticker'].isin(valid)]['Total Value ($)'] / hist_total).values
                    ret   = data[valid].pct_change().dropna()
                    pret  = (ret * vw).sum(axis=1)
                    mu_d  = pret.mean()
                    sig_d = pret.std()

                    # Ajuste por régimen
                    regime_adj = {'neutral': 0, 'bull': 0.25 / 252, 'bear': -0.15 / 252}
                    mu_d += regime_adj.get(regime, 0)

                    days  = 252 * proj_years
                    n_sim = 1000
                    sims  = np.zeros((days, n_sim))
                    sims[0] = total_aum
                    for i in range(1, days):
                        sims[i] = sims[i-1] * (1 + np.random.normal(mu_d, sig_d, n_sim))

                    years_x = np.linspace(0, proj_years, days)
                    p5  = np.percentile(sims, 5,  axis=1)
                    p25 = np.percentile(sims, 25, axis=1)
                    p50 = np.percentile(sims, 50, axis=1)
                    p75 = np.percentile(sims, 75, axis=1)
                    p95 = np.percentile(sims, 95, axis=1)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years_x, y=p95, mode='lines',
                                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=years_x, y=p5, mode='lines',
                                             line=dict(width=0),
                                             fill='tonexty', fillcolor='rgba(59,130,246,0.05)',
                                             name='P5–P95'))
                    fig.add_trace(go.Scatter(x=years_x, y=p75, mode='lines',
                                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=years_x, y=p25, mode='lines',
                                             line=dict(width=0),
                                             fill='tonexty', fillcolor='rgba(59,130,246,0.12)',
                                             name='P25–P75'))
                    fig.add_trace(go.Scatter(x=years_x, y=p50, mode='lines',
                                             line=dict(color='#3b82f6', width=2.5),
                                             name='Mediana' if es else 'Median'))
                    fig.add_hline(y=total_aum, line_dash='dot', line_color=GOLD,
                                  annotation_text=f"AUM actual: ${total_aum:,.0f}" if es
                                  else f"Current AUM: ${total_aum:,.0f}",
                                  annotation_font_color=GOLD)
                    fig.update_layout(**plotly_layout(
                        title=dict(
                            text=f"{'Proyección Monte Carlo — ' if es else 'Monte Carlo Projection — '}{proj_years} {'años · ' if es else 'years · '}{n_sim} {'trayectorias' if es else 'paths'}",
                            font=dict(color='#9ca3af', size=12)
                        ),
                        xaxis_title="Años" if es else "Years",
                        yaxis_title="Valor del Portfolio ($)" if es else "Portfolio Value ($)",
                        height=400,
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric(t("horizon.p10"),  f"${p5[-1]:,.0f}")
                    c2.metric("Percentil 25" if es else "25th Percentile", f"${p25[-1]:,.0f}")
                    c3.metric(t("horizon.median"), f"${p50[-1]:,.0f}")
                    c4.metric("Percentil 75" if es else "75th Percentile", f"${p75[-1]:,.0f}")
                    c5.metric(t("horizon.p90"),  f"${p95[-1]:,.0f}")

                    # CAGR mediano
                    cagr_med = (p50[-1] / total_aum) ** (1 / proj_years) - 1
                    st.markdown(
                        f"<p style='color:{TEXT_MUTED}; font-size:12px; margin-top:8px;'>"
                        f"CAGR mediano esperado: <b style='color:#f3f4f6;'>{cagr_med*100:.2f}%</b>"
                        f"</p>",
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

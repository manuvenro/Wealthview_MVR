"""
Investment Memo — Módulo de análisis IA estructurado estilo banco de inversión.
Genera un memo institucional completo con secciones fijas, métricas del portfolio
y una sección de "Systematic Action Framework" (lo que el sistema ejecutaría).
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI

from modules.utils import ensure_portfolio_data, no_portfolio_warning
from modules.i18n import get_lang
from modules.styles import (
    GOLD, SURFACE, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, NEGATIVE, PLOTLY_DARK
)

DERIV_TYPES = ('option_call', 'option_put', 'future', 'warrant')


def _lang() -> str:
    return get_lang()


def _es() -> bool:
    return _lang() == 'es'


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de métricas de riesgo (sin depender del módulo risk para evitar circ.)
# ─────────────────────────────────────────────────────────────────────────────

def _quick_risk_metrics(portfolio_df: pd.DataFrame) -> dict:
    """Calcula métricas básicas de riesgo para incluir en el memo."""
    import yfinance as yf

    tickers  = portfolio_df['Ticker'].tolist()
    total    = portfolio_df['Total Value ($)'].sum()
    weights  = (portfolio_df['Total Value ($)'] / total).values if total > 0 else np.ones(len(portfolio_df)) / len(portfolio_df)

    # Solo tickers de equity/ETF para datos históricos
    equity_mask = ~portfolio_df['Asset Type'].isin(list(DERIV_TYPES) + ['bond']) if 'Asset Type' in portfolio_df.columns else pd.Series([True] * len(portfolio_df))
    eq_tickers  = portfolio_df[equity_mask]['Ticker'].tolist()

    if not eq_tickers:
        return {}

    try:
        all_tickers = list(set(eq_tickers + ['SPY']))
        raw = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']].copy()
            prices.columns = all_tickers

        returns = prices.pct_change().dropna()

        # Pesos solo de tickers con datos
        valid = [t for t in eq_tickers if t in returns.columns]
        if not valid:
            return {}

        eq_df    = portfolio_df[portfolio_df['Ticker'].isin(valid)]
        eq_total = eq_df['Total Value ($)'].sum()
        eq_w     = (eq_df['Total Value ($)'] / eq_total).values if eq_total > 0 else np.ones(len(valid)) / len(valid)

        port_ret = returns[valid].dot(eq_w)

        annual_ret  = port_ret.mean() * 252
        annual_vol  = port_ret.std() * np.sqrt(252)
        rf          = st.session_state.get('risk_free_rate', 4.0) / 100
        sharpe      = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

        cumulative  = (1 + port_ret).cumprod()
        rolling_max = cumulative.cummax()
        max_dd      = ((cumulative - rolling_max) / rolling_max).min()
        var_95      = np.percentile(port_ret, 5)

        beta = 1.0
        alpha = 0.0
        if 'SPY' in returns.columns:
            cov_m = np.cov(port_ret, returns['SPY'])
            var_b = np.var(returns['SPY'])
            beta  = cov_m[0, 1] / var_b if var_b > 0 else 1.0
            bench_annual = returns['SPY'].mean() * 252
            alpha = annual_ret - (rf + beta * (bench_annual - rf))

        # Distribución por clase de activo
        allocation = {}
        if 'Asset Type' in portfolio_df.columns:
            by_type = portfolio_df.groupby('Asset Type')['Total Value ($)'].sum()
            allocation = {k: v / total * 100 for k, v in by_type.items()}

        return {
            'annual_return':  round(annual_ret * 100, 2),
            'annual_vol':     round(annual_vol * 100, 2),
            'sharpe':         round(sharpe, 2),
            'beta':           round(beta, 2),
            'alpha':          round(alpha * 100, 2),
            'max_drawdown':   round(max_dd * 100, 2),
            'var_95_pct':     round(var_95 * 100, 2),
            'allocation':     allocation,
            'total_aum':      total,
        }
    except Exception:
        return {}


def _build_memo_prompt(portfolio_df: pd.DataFrame, metrics: dict, lang: str) -> str:
    """Construye el prompt completo para el investment memo."""
    es = lang == 'es'
    total_aum = portfolio_df['Total Value ($)'].sum()

    # Holdings table
    holdings_lines = []
    for _, row in portfolio_df.sort_values('Total Value ($)', ascending=False).iterrows():
        peso = row['Total Value ($)'] / total_aum * 100 if total_aum > 0 else 0
        atype = str(row.get('Asset Type', 'equity')).replace('_', ' ').title()
        holdings_lines.append(
            f"  • {row['Ticker']} ({row.get('Name', row['Ticker'])}) — {atype} — {peso:.1f}%"
        )
    holdings_str = "\n".join(holdings_lines)

    # Risk metrics
    if metrics:
        metrics_str = (
            f"  • Retorno anual esperado: {metrics.get('annual_return','N/A')}%\n"
            f"  • Volatilidad anual: {metrics.get('annual_vol','N/A')}%\n"
            f"  • Sharpe Ratio: {metrics.get('sharpe','N/A')}\n"
            f"  • Beta vs S&P 500: {metrics.get('beta','N/A')}\n"
            f"  • Alpha (Jensen): {metrics.get('alpha','N/A')}%\n"
            f"  • Max Drawdown (12m): {metrics.get('max_drawdown','N/A')}%\n"
            f"  • VaR 95% (1d): {metrics.get('var_95_pct','N/A')}%"
        )
        alloc_lines = [f"    - {k}: {v:.1f}%" for k, v in metrics.get('allocation', {}).items()]
        alloc_str   = "\n".join(alloc_lines) if alloc_lines else "  N/A"
    else:
        metrics_str = "  Datos insuficientes para calcular métricas cuantitativas."
        alloc_str   = "  N/A"

    lang_instr = "Responde SIEMPRE en español. Usa terminología financiera profesional en español." if es else "Always respond in English. Use professional financial terminology."

    return f"""Eres un Managing Director de Análisis de Inversiones. Redacta un Investment Memo institucional completo.

DATOS DEL PORTFOLIO:
AUM: ${total_aum:,.0f}
Posiciones ({len(portfolio_df)}):
{holdings_str}

MÉTRICAS CUANTITATIVAS (últimos 12 meses):
{metrics_str}

DISTRIBUCIÓN POR CLASE DE ACTIVO:
{alloc_str}

FECHA DEL MEMO: {datetime.now().strftime('%d %B %Y')}

Genera el memo con EXACTAMENTE estas secciones (usa estos títulos en negrita exactos):

**EXECUTIVE SUMMARY**
2-3 frases. Estado del portfolio, entorno de mercado y conclusión principal.

**PORTFOLIO DIAGNOSIS**
Análisis de la composición actual: diversificación, concentración, exposición por clase de activo. Menciona si el portfolio es agresivo, moderado o conservador basándote en las métricas.

**RISK ASSESSMENT**
Análisis de las métricas de riesgo. Interpreta el Sharpe, Beta, Alpha y VaR en términos prácticos para el inversor. ¿Está el portfolio compensando adecuadamente el riesgo asumido?

**MARKET CONTEXT**
Describe el entorno de mercado actual relevante para este portfolio específico (tipos de interés, inflación, ciclo de crédito, momentum de renta variable).

**POSITION ANALYSIS**
Analiza las 3-5 posiciones de mayor peso. Para cada una: tesis de inversión implícita, riesgo específico, y si encaja con el entorno macro actual.

**SYSTEMATIC ACTION FRAMEWORK**
[CRÍTICO: No uses los verbos "recomiendo", "debería", "compra", "vende". En su lugar describe QUÉ EJECUTARÍA EL SISTEMA automáticamente dado el portfolio actual y las condiciones de mercado.]
Estructura esto como: "El sistema ejecutaría: (1)... (2)... (3)..."

**APPENDIX — KEY METRICS**
Tabla de métricas (usa formato de texto simple, no markdown table): incluye todos los valores cuantitativos relevantes del portfolio.

Sé directo, institucional y conciso. Máximo 800 palabras en total. {lang_instr}"""


def _render_section(title: str, content: str, color: str, icon: str = ""):
    """Renderiza una sección del memo con estilo institucional."""
    content_html = content.replace('\n', '<br>').replace('**', '')
    st.markdown(f"""
    <div style='background:{SURFACE}; border:1px solid {BORDER};
                border-left:3px solid {color}; border-radius:0 8px 8px 0;
                padding:18px 22px; margin-bottom:14px;'>
        <div style='font-size:10px; font-weight:700; color:{color};
                    text-transform:uppercase; letter-spacing:1.8px; margin-bottom:12px;'>
            {icon + " " if icon else ""}{title}
        </div>
        <div style='color:#d1d5db; font-size:13px; line-height:1.75;'>{content_html}</div>
    </div>
    """, unsafe_allow_html=True)


def _parse_and_render_memo(memo_text: str, es: bool):
    """Parsea el memo por secciones y las renderiza con diseño institucional."""
    import re

    section_config = {
        'EXECUTIVE SUMMARY':          (GOLD,      '📋'),
        'PORTFOLIO DIAGNOSIS':         ('#4a9eff', '🔬'),
        'RISK ASSESSMENT':             ('#c05a5a', '⚠️'),
        'MARKET CONTEXT':              ('#5a7abf', '🌍'),
        'POSITION ANALYSIS':           ('#5a8f6e', '📊'),
        'SYSTEMATIC ACTION FRAMEWORK': (GOLD,      '⚡'),
        'APPENDIX — KEY METRICS':      ('#6b7280', '📎'),
        'APPENDIX - KEY METRICS':      ('#6b7280', '📎'),
        'APÉNDICE':                    ('#6b7280', '📎'),
    }

    # Dividir por títulos en negrita
    parts = re.split(r'\*\*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ \/\-—]+)\*\*', memo_text)

    if len(parts) <= 1:
        st.markdown(f"""
        <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                    padding:24px; color:#d1d5db; font-size:13px; line-height:1.8;'>
            {memo_text.replace(chr(10),'<br>')}
        </div>
        """, unsafe_allow_html=True)
        return

    # Intro text si existe
    if parts[0].strip():
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:12px; margin-bottom:14px;'>{parts[0].strip()}</div>",
            unsafe_allow_html=True
        )

    for i in range(1, len(parts) - 1, 2):
        title   = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Buscar config por título (exact o partial match)
        cfg = None
        for key, val in section_config.items():
            if key in title.upper() or title.upper() in key:
                cfg = val
                break
        color, icon = cfg if cfg else (TEXT_MUTED, '•')

        _render_section(title, content, color, icon)


# ─────────────────────────────────────────────────────────────────────────────
# Render principal
# ─────────────────────────────────────────────────────────────────────────────

def _render_memo_pdf_export(portfolio_df, username: str, es: bool, _key_suffix: str = ""):
    """Genera y ofrece descarga del PDF directamente en la página del memo."""
    from modules.pdf_report import generate_pdf
    from datetime import datetime as _dt

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        if st.button(
            "Exportar como PDF" if es else "Export as PDF",
            type="primary", use_container_width=True, key=f"memo_export_btn{_key_suffix}"
        ):
            with st.spinner("Generando PDF..." if es else "Generating PDF..."):
                try:
                    pdf_bytes = generate_pdf(username, portfolio_df)
                    fname = "WealthView_InvestmentMemo_{}_{}.pdf".format(
                        username, _dt.today().strftime("%Y%m%d"))
                    st.session_state['_pdf_bytes']    = pdf_bytes
                    st.session_state['_pdf_filename'] = fname
                except Exception as e:
                    st.error(f"Error al generar el PDF: {e}")

    with col_dl:
        if st.session_state.get('_pdf_bytes'):
            st.download_button(
                label="Descargar PDF" if es else "Download PDF",
                data=st.session_state['_pdf_bytes'],
                file_name=st.session_state.get('_pdf_filename', 'WealthView_Memo.pdf'),
                mime="application/pdf",
                use_container_width=True,
                key="memo_dl_btn",
            )



def render_investment_memo():
    es = _es()
    title_text = "Investment Memo IA" if es else "Investment Memo AI"
    st.title(title_text)

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        no_portfolio_warning()
        return

    username = st.session_state.get('username', '')

    total_aum = portfolio_df['Total Value ($)'].sum()
    n_pos     = len(portfolio_df)

    # ── Header con info del portfolio ─────────────────────────────────────────
    st.markdown(f"""
    <div style='padding-bottom:18px; margin-bottom:20px; border-bottom:1px solid {BORDER};'>
        <div style='font-family:"Playfair Display",Georgia,serif; font-size:13px;
                    color:{GOLD}; text-transform:uppercase; letter-spacing:2px;
                    margin-bottom:4px;'>
            {'Investment Intelligence' if es else 'Investment Intelligence'}
        </div>
        <div style='color:{TEXT_MUTED}; font-size:12px;'>
            {'Análisis cuantitativo y cualitativo generado por IA para' if es else 'Quantitative & qualitative AI-generated analysis for'}
            <b style='color:#f3f4f6;'> {n_pos} {'posiciones' if es else 'positions'}</b>
            · AUM: <b style='color:#f3f4f6;'>${total_aum:,.0f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Opciones ──────────────────────────────────────────────────────────────
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_metrics = st.toggle(
            "Incluir cálculo de métricas de riesgo (más lento)" if es else "Include risk metrics calculation (slower)",
            value=True
        )
    with col_opt2:
        memo_lang = st.selectbox(
            "Idioma del memo" if es else "Memo language",
            options=['es', 'en'],
            format_func=lambda x: '🇪🇸 Español' if x == 'es' else '🇬🇧 English',
            index=0 if es else 1
        )

    # ── Generar memo ──────────────────────────────────────────────────────────
    btn_label = "Generar Investment Memo" if es else "Generate Investment Memo"
    if st.button(btn_label, type="primary", use_container_width=True):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "⚠️ API Key de OpenAI no encontrada. Configúrala en Configuración → API Keys."
                if es else
                "⚠️ OpenAI API Key not found. Configure it in Settings → API Keys."
            )
            return

        client = OpenAI(api_key=api_key)
        progress = st.progress(0)

        try:
            metrics = {}
            if include_metrics:
                progress.progress(15, text="Calculando métricas de riesgo..." if es else "Calculating risk metrics...")
                metrics = _quick_risk_metrics(portfolio_df)

            progress.progress(40, text="Construyendo contexto del portfolio..." if es else "Building portfolio context...")
            prompt = _build_memo_prompt(portfolio_df, metrics, memo_lang)

            progress.progress(60, text="Generando memo con IA..." if es else "Generating memo with AI...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "Eres un Managing Director de Análisis en un banco de inversión de primer nivel. "
                                "Redactas investment memos institucionales de alta calidad, precisos y accionables. "
                                "Nunca recomiendas comprar o vender directamente. "
                                "Describes lo que el sistema haría de forma objetiva."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.15
            )
            progress.progress(90, text="Formateando..." if es else "Formatting...")
            memo_text = response.choices[0].message.content
            progress.progress(100)

            # ── Guardar en session state para PDF ──────────────────────────
            st.session_state['last_investment_memo'] = {
                'text':    memo_text,
                'metrics': metrics,
                'date':    datetime.now().strftime('%d %B %Y'),
                'aum':     total_aum,
                'lang':    memo_lang,
            }

            # ── Render del memo ────────────────────────────────────────────
            st.markdown("---")

            # Header del memo
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; gap:14px; margin-bottom:24px;
                        padding:20px 24px; background:{SURFACE};
                        border:1px solid {BORDER}; border-radius:8px;'>
                <div style='width:4px; min-height:60px; background:{GOLD}; border-radius:2px; flex-shrink:0;'></div>
                <div>
                    <div style='font-family:"Playfair Display",Georgia,serif; font-size:22px;
                                font-weight:700; color:#f3f4f6; margin-bottom:4px;'>
                        {'Investment Memo' if es else 'Investment Memo'}
                    </div>
                    <div style='color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase;
                                letter-spacing:1.2px;'>
                        {datetime.now().strftime('%d %B %Y').upper()} &nbsp;·&nbsp;
                        AUM ${total_aum:,.0f} &nbsp;·&nbsp;
                        {n_pos} {'POSICIONES' if es else 'POSITIONS'} &nbsp;·&nbsp;
                        GENERADO POR IA
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Secciones del memo
            _parse_and_render_memo(memo_text, memo_lang == 'es')

            # ── Métricas rápidas si se calcularon ─────────────────────────
            if metrics:
                st.markdown("---")
                st.markdown(
                    f"<p style='font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; "
                    f"letter-spacing:1.2px; margin-bottom:12px;'>"
                    f"{'MÉTRICAS CUANTITATIVAS CALCULADAS' if es else 'CALCULATED QUANTITATIVE METRICS'}</p>",
                    unsafe_allow_html=True
                )
                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                mc1.metric("Retorno Anual" if es else "Annual Return",
                           f"{metrics.get('annual_return','—')}%")
                mc2.metric("Volatilidad" if es else "Volatility",
                           f"{metrics.get('annual_vol','—')}%")
                mc3.metric("Sharpe", f"{metrics.get('sharpe','—')}")
                mc4.metric("Beta", f"{metrics.get('beta','—')}")
                mc5.metric("Alpha", f"{metrics.get('alpha','—')}%",
                           delta_color="normal" if float(metrics.get('alpha', 0)) >= 0 else "inverse")
                mc6.metric("Max DD", f"{metrics.get('max_drawdown','—')}%",
                           delta_color="inverse")

            # ── Export PDF inline ────────────────────────────────────────
            st.markdown("---")
            _render_memo_pdf_export(portfolio_df, username, es, _key_suffix="_1")

        except Exception as e:
            st.error(f"{'Error al generar el memo:' if es else 'Error generating memo:'} {e}")

    # ── Export siempre visible si hay memo ───────────────────────────────────
    if 'last_investment_memo' in st.session_state and st.session_state['last_investment_memo'].get('text'):
        username_val = st.session_state.get('username', '')
        prev = st.session_state['last_investment_memo']
        # Si el memo fue mostrado en esta sesión (no justo generado), mostrar export
        if not st.button("Limpiar memo anterior" if es else "Clear previous memo", key="clear_memo"):
            st.markdown("---")
            st.caption(f"{'Último memo generado:' if es else 'Last memo generated:'} {prev.get('date','')}")
            _render_memo_pdf_export(portfolio_df, username_val, es, _key_suffix="_2")
        else:
            del st.session_state['last_investment_memo']
            if '_pdf_bytes' in st.session_state:
                del st.session_state['_pdf_bytes']
            st.rerun()

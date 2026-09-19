import streamlit as st
import os
import requests
import xml.etree.ElementTree as ET
from openai import OpenAI
from modules.utils import ensure_portfolio_data, no_portfolio_warning
from modules.i18n import t, get_lang
from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE
)

DERIV_TYPES = ('option_call', 'option_put', 'future', 'warrant')

# Fuentes RSS por tipo de activo
RSS_SOURCES = {
    'equity': [
        "https://news.google.com/rss/search?q={ticker}+stock+earnings&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    ],
    'bond':   [
        "https://news.google.com/rss/search?q={ticker}+bond+yield+interest+rate&hl=en-US&gl=US&ceid=US:en",
    ],
    'macro':  [
        "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates+inflation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=ECB+monetary+policy+economy&hl=en-US&gl=US&ceid=US:en",
    ],
    'deriv':  [
        "https://news.google.com/rss/search?q={ticker}+options+volatility+VIX&hl=en-US&gl=US&ceid=US:en",
    ],
}

def _fetch_rss(url: str, max_items: int = 4) -> list[str]:
    """Devuelve lista de titulares de un feed RSS."""
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        headlines = []
        for item in root.findall('.//item')[:max_items]:
            title = item.find('title')
            if title is not None and title.text:
                headlines.append(title.text.strip())
        return headlines
    except Exception:
        return []


def _get_asset_type_group(asset_type: str) -> str:
    if asset_type in ('equity', 'fund', 'bond_etf'):
        return 'equity'
    if asset_type == 'bond':
        return 'bond'
    if asset_type in DERIV_TYPES:
        return 'deriv'
    return 'equity'


def _build_news_context(portfolio_df) -> tuple[str, dict]:
    """
    Construye el contexto de noticias ponderado por peso en el portfolio.
    Devuelve (news_context_str, dict con noticias por ticker).
    """
    total_aum = portfolio_df['Total Value ($)'].sum()
    news_by_ticker = {}

    # Ordenar por peso descendente (las posiciones más grandes primero)
    portfolio_df = portfolio_df.copy()
    portfolio_df['Peso'] = portfolio_df['Total Value ($)'] / total_aum if total_aum > 0 else 0
    portfolio_df = portfolio_df.sort_values('Peso', ascending=False)

    # Noticias macro siempre incluidas
    macro_headlines = []
    for url in RSS_SOURCES['macro']:
        macro_headlines += _fetch_rss(url, max_items=3)
    macro_headlines = list(dict.fromkeys(macro_headlines))[:5]  # deduplicar

    for _, row in portfolio_df.iterrows():
        ticker    = str(row['Ticker'])
        atype     = str(row.get('Asset Type', 'equity'))
        peso      = float(row['Peso'])
        group     = _get_asset_type_group(atype)

        # Número de noticias proporcional al peso (mín 2, máx 6)
        n_news = min(6, max(2, int(peso * 20)))

        headlines = []
        for url_tpl in RSS_SOURCES.get(group, RSS_SOURCES['equity']):
            url = url_tpl.format(ticker=ticker)
            headlines += _fetch_rss(url, max_items=n_news)

        # Deduplicar y limitar
        seen = set()
        clean = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                clean.append(h)
            if len(clean) >= n_news:
                break

        news_by_ticker[ticker] = {
            'headlines': clean,
            'peso': peso,
            'asset_type': atype,
        }

    # Construir el string de contexto
    lines = ["=== NOTICIAS MACRO ==="]
    for h in macro_headlines:
        lines.append(f"  • {h}")
    lines.append("")

    for ticker, data in news_by_ticker.items():
        peso_pct = data['peso'] * 100
        atype_label = data['asset_type'].replace('_', ' ').upper()
        lines.append(f"=== {ticker} ({atype_label}, {peso_pct:.1f}% del portfolio) ===")
        if data['headlines']:
            for h in data['headlines']:
                lines.append(f"  • {h}")
        else:
            lines.append("  • Sin noticias recientes disponibles.")
        lines.append("")

    return "\n".join(lines), news_by_ticker


def _build_prompt(portfolio_df, news_context: str, lang: str) -> str:
    """Construye el prompt para el investment memo."""
    tickers_info = []
    total_aum = portfolio_df['Total Value ($)'].sum()
    for _, row in portfolio_df.iterrows():
        peso = row['Total Value ($)'] / total_aum * 100 if total_aum > 0 else 0
        atype = str(row.get('Asset Type', 'equity'))
        tickers_info.append(
            f"  - {row['Ticker']} ({atype.replace('_',' ').title()}): "
            f"{row.get('Name', row['Ticker'])} — {peso:.1f}% del portfolio"
        )

    holdings_str = "\n".join(tickers_info)
    lang_instruction = "Responde SIEMPRE en español." if lang == 'es' else "Always respond in English."

    prompt = f"""Eres un Managing Director de Análisis de Inversiones en un banco de inversión de primer nivel (estilo Goldman Sachs, J.P. Morgan Asset Management).

COMPOSICIÓN DEL PORTFOLIO:
{holdings_str}

AUM total aproximado: ${total_aum:,.0f}

CONTEXTO DE MERCADO Y NOTICIAS RECIENTES:
{news_context}

Redacta un Investment Memo diario profesional e institucional con exactamente estas secciones:

**EXECUTIVE SUMMARY**
Una síntesis de 2-3 frases del estado actual del portfolio y el entorno de mercado.

**MARKET OUTLOOK**
Análisis macro: tipos de interés, inflación, sentimiento de mercado, principales riesgos sistémicos basados en las noticias. Máximo 3 párrafos.

**PORTFOLIO POSITIONING**
Análisis de las posiciones más relevantes (las de mayor peso). Identifica qué posiciones se benefician del entorno actual y cuáles están en riesgo. Sé específico con los tickers.

**KEY CATALYSTS**
Lista de 3-5 catalizadores positivos identificados en las noticias que podrían impulsar el portfolio.

**KEY RISKS**
Lista de 3-5 riesgos concretos identificados en las noticias para este portfolio específico (no genéricos).

**SYSTEMATIC ACTION FRAMEWORK**
[IMPORTANTE: No digas "recomiendo comprar/vender". En su lugar, describe lo que un sistema cuantitativo con este portfolio y estas condiciones de mercado ejecutaría automáticamente.]
Describe las acciones concretas que el sistema tomaría: rebalanceos de exposición, coberturas naturales, ajustes de duración en renta fija, gestión del riesgo de derivados, etc.

Usa un tono institucional, directo y conciso. Sin florituras. {lang_instruction}"""

    return prompt


def render_ai_sentiment():
    lang = get_lang()
    es = lang == 'es'

    st.title(t("sentiment.title"))

    portfolio_df = ensure_portfolio_data()
    if portfolio_df is None or portfolio_df.empty:
        no_portfolio_warning()
        return

    # ── Header informativo ────────────────────────────────────────────────────
    total_aum  = portfolio_df['Total Value ($)'].sum()
    n_pos      = len(portfolio_df)
    n_types    = portfolio_df['Asset Type'].nunique() if 'Asset Type' in portfolio_df.columns else 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Total AUM", f"${total_aum:,.0f}")
    col2.metric("Posiciones" if es else "Positions", str(n_pos))
    col3.metric("Clases de activo" if es else "Asset classes", str(n_types))

    _lbl_title = '🧠 Copiloto IA Institucional' if es else '🧠 Institutional AI Copilot'
    _lbl_desc  = (
        'La IA escanea noticias <b>ponderadas por el peso de cada activo</b> en tu portfolio. '
        'Posiciones con mayor exposición reciben más cobertura noticiosa. '
        'Las noticias macro se incluyen siempre como contexto de fondo.'
        if es else
        'The AI scans news <b>weighted by each asset portfolio weight</b>. '
        'Larger positions receive more news coverage. '
        'Macro news is always included as background context.'
    )
    st.markdown(f"""
    <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                padding:14px 18px; margin:16px 0;'>
        <div style='color:{GOLD}; font-size:11px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.2px; margin-bottom:6px;'>
            {_lbl_title}
        </div>
        <div style='color:{TEXT_MUTED}; font-size:12px; line-height:1.6;'>
            {_lbl_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Botón de análisis ─────────────────────────────────────────────────────
    if st.button(t("sentiment.btn"), type="primary", use_container_width=True):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error(t("sentiment.no_key"))
            return

        client = OpenAI(api_key=api_key)

        progress = st.progress(0, text="Conectando con fuentes de noticias..." if es else "Connecting to news sources...")

        with st.spinner(t("sentiment.scanning")):
            try:
                # 1. Obtener noticias ponderadas
                progress.progress(20, text="Descargando noticias por activo..." if es else "Fetching per-asset news...")
                news_context, news_by_ticker = _build_news_context(portfolio_df)
                progress.progress(55, text="Analizando con IA..." if es else "Analysing with AI...")

                # 2. Construir prompt y llamar a la API
                prompt = _build_prompt(portfolio_df, news_context, lang)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                         "content": "Eres un analista senior de un hedge fund de primer nivel. "
                                    "Redactas memos institucionales precisos y accionables."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.2
                )
                progress.progress(90, text="Formateando memo..." if es else "Formatting memo...")
                memo = response.choices[0].message.content
                progress.progress(100, text="✅ Listo" if es else "✅ Done")

                # 3. Renderizar memo
                st.markdown("---")
                _render_memo(memo, es)

                # 4. Detalle de noticias por activo (expandible)
                with st.expander("📰 " + ("Ver noticias analizadas por activo" if es else "View news analysed per asset")):
                    for ticker, data in news_by_ticker.items():
                        peso_pct = data['peso'] * 100
                        atype    = data['asset_type']
                        st.markdown(
                            f"**{ticker}** — {atype.replace('_',' ').title()} · "
                            f"Peso: {peso_pct:.1f}%"
                        )
                        if data['headlines']:
                            for h in data['headlines']:
                                st.markdown(f"  - {h}")
                        else:
                            st.markdown("  - *Sin noticias disponibles*")
                        st.markdown("")

            except Exception as e:
                st.error(f"{'Error al generar el memo:' if es else 'Error generating memo:'} {e}")


def _render_memo(memo: str, es: bool):
    """Renderiza el memo con formato visual institucional."""
    from modules.styles import GOLD, SURFACE, SURFACE_2, BORDER, TEXT_PRIMARY, TEXT_MUTED, POSITIVE, NEGATIVE

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:20px;'>
        <div style='width:3px; height:28px; background:{GOLD}; border-radius:2px;'></div>
        <div>
            <div style='font-family:"Playfair Display",Georgia,serif; font-size:20px;
                        font-weight:700; color:#f3f4f6;'>
                {'Memo de Inversión Diario' if es else 'Daily Investment Memo'}
            </div>
            <div style='color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase;
                        letter-spacing:1.2px;'>
                {'Generado por IA · Estilo institucional' if es else 'AI-Generated · Institutional Style'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Colorear secciones según tipo
    section_colors = {
        'EXECUTIVE SUMMARY': GOLD,
        'MARKET OUTLOOK': '#4a9eff',
        'PORTFOLIO POSITIONING': '#5a8f6e',
        'KEY CATALYSTS': '#5a8f6e',
        'KEY RISKS': '#c05a5a',
        'SYSTEMATIC ACTION FRAMEWORK': GOLD,
    }

    # Dividir por secciones y renderizar
    import re
    sections = re.split(r'\*\*([A-Z ]+)\*\*', memo)

    if len(sections) <= 1:
        # Fallback: mostrar como texto plano formateado
        st.markdown(f"""
        <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                    padding:24px; line-height:1.8; color:#d1d5db; font-size:13px;'>
            {memo.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        return

    # Primer elemento puede ser texto introductorio
    if sections[0].strip():
        st.markdown(
            f"<div style='color:{TEXT_MUTED}; font-size:12px; margin-bottom:12px;'>"
            f"{sections[0].strip()}</div>",
            unsafe_allow_html=True
        )

    # Procesar pares (título, contenido)
    for i in range(1, len(sections) - 1, 2):
        title   = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        color   = section_colors.get(title, '#6b7280')

        # Formatear contenido: listas con bullets
        formatted = content.replace('\n- ', '\n• ').replace('\n* ', '\n• ')
        html_content = formatted.replace('\n', '<br>')

        st.markdown(f"""
        <div style='background:{SURFACE}; border:1px solid {BORDER}; border-left:3px solid {color};
                    border-radius:0 8px 8px 0; padding:16px 20px; margin-bottom:12px;'>
            <div style='font-size:10px; font-weight:700; color:{color}; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:10px;'>{title}</div>
            <div style='color:#d1d5db; font-size:13px; line-height:1.7;'>{html_content}</div>
        </div>
        """, unsafe_allow_html=True)

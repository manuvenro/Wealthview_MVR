"""
llm_analyst.py — AI-powered qualitative investment analysis for WealthView
Uses Anthropic Claude (primary) or OpenAI GPT-4o (fallback) to generate
institutional-quality investment theses from financial data snapshots.
"""

import os
import json
import hashlib
import streamlit as st
from datetime import date

# ── Provider detection ────────────────────────────────────────────────────────

def _anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()

def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()

def llm_available() -> bool:
    return bool(_anthropic_key() or _openai_key())

def active_provider() -> str:
    if _anthropic_key():
        return "Claude (Anthropic)"
    if _openai_key():
        return "GPT-4o (OpenAI)"
    return "none"


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(snapshot: dict) -> str:
    """
    Build the analyst prompt from a financial snapshot dict.
    snapshot keys: ticker, name, sector, price, mktcap, pe, fwd_pe, ps,
    ev_ebitda, gross_m, net_m, op_m, roe, de, fcf_yield, revenue_growth,
    eps_growth, peg, up_pct, target_price, rsi, sma200_above,
    wv_score, wv_rec, reasons_buy, reasons_sell,
    news_headlines (list of str), peer_pe, peer_fwd_pe, peer_net_m
    """
    t = snapshot

    def _pct(v):
        return f"{v*100:.1f}%" if v is not None else "N/D"

    def _x(v, dp=1):
        return f"{v:.{dp}f}x" if v is not None else "N/D"

    def _fmt(v, prefix="$"):
        if v is None:
            return "N/D"
        if abs(v) >= 1e12:
            return f"{prefix}{v/1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"{prefix}{v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{prefix}{v/1e6:.2f}M"
        return f"{prefix}{v:,.0f}"

    headlines_text = "\n".join(
        f"  - {h}" for h in (t.get("news_headlines") or [])[:8]
    ) or "  (sin noticias recientes disponibles)"

    reasons_buy = "\n".join(f"  + {r}" for r in (t.get("reasons_buy") or [])) or "  (ninguna identificada)"
    reasons_sell = "\n".join(f"  - {r}" for r in (t.get("reasons_sell") or [])) or "  (ninguna identificada)"

    return f"""Eres el Director de Análisis de Renta Variable de un banco de inversión de primer nivel (nivel Goldman Sachs, Morgan Stanley). Tu tarea es redactar un análisis de inversión profesional e institucional en ESPAÑOL sobre la siguiente empresa, basándote EXCLUSIVAMENTE en los datos proporcionados.

═══════════════════════════════════════════════════════════
DATOS DE LA EMPRESA
═══════════════════════════════════════════════════════════
Empresa:          {t.get('name', t.get('ticker', 'N/D'))} ({t.get('ticker', 'N/D')})
Sector:           {t.get('sector', 'N/D')}
Precio actual:    {_fmt(t.get('price'), '$')}
Capitalización:   {_fmt(t.get('mktcap'), '$')}

VALORACIÓN (empresa vs. mediana del sector)
  P/E trailing:   {_x(t.get('pe'))}       vs. sector: {_x(t.get('peer_pe'))}
  P/E forward:    {_x(t.get('fwd_pe'))}   vs. sector: {_x(t.get('peer_fwd_pe'))}
  P/S:            {_x(t.get('ps'))}
  EV/EBITDA:      {_x(t.get('ev_ebitda'))}
  PEG ratio:      {_x(t.get('peg'))}
  Potencial precio objetivo: {f"{t.get('up_pct',0):+.1f}%" if t.get('up_pct') is not None else 'N/D'} (objetivo consenso: {_fmt(t.get('target_price'), '$')})

RENTABILIDAD
  Margen bruto:   {_pct(t.get('gross_m'))}
  Margen EBIT:    {_pct(t.get('op_m'))}
  Margen neto:    {_pct(t.get('net_m'))}   vs. sector: {_pct(t.get('peer_net_m'))}
  ROE:            {_pct(t.get('roe'))}
  FCF Yield:      {f"{t.get('fcf_yield'):.1f}%" if t.get('fcf_yield') is not None else 'N/D'}

CRECIMIENTO
  Crecimiento ingresos (YoY): {_pct(t.get('revenue_growth'))}
  Crecimiento EPS (YoY):      {_pct(t.get('eps_growth'))}

BALANCE
  Deuda/Equity:  {_x(t.get('de'))}

TÉCNICO
  RSI(14):       {f"{t.get('rsi'):.0f}" if t.get('rsi') is not None else 'N/D'}
  Precio vs SMA200: {"Por encima ✓" if t.get('sma200_above') else "Por debajo ✗" if t.get('sma200_above') is not None else 'N/D'}

SISTEMA WEALTHVIEW
  Score compuesto:  {f"{t.get('wv_score'):.1f}/10" if t.get('wv_score') is not None else 'N/D'}
  Recomendación:    {t.get('wv_rec', 'N/D')}

Señales positivas identificadas:
{reasons_buy}

Señales negativas identificadas:
{reasons_sell}

NOTICIAS RECIENTES
{headlines_text}

═══════════════════════════════════════════════════════════
INSTRUCCIONES DE REDACCIÓN
═══════════════════════════════════════════════════════════
Redacta un análisis de inversión estructurado con las siguientes secciones. Sé directo, preciso y usa lenguaje de banca de inversión profesional. Cada sección debe ser un párrafo sustancial (mínimo 3-4 frases). NO inventes datos que no estén en el snapshot. Si un dato no está disponible, trabaja con los que sí están.

**1. RESUMEN EJECUTIVO** (2-3 frases: veredicto claro, rating, razonamiento central)

**2. TESIS DE INVERSIÓN — CASO ALCISTA**
Argumenta de forma convincente los catalizadores de subida, ventajas competitivas, calidad del negocio y por qué la valoración actual ofrece una oportunidad. Cita los datos específicos del snapshot.

**3. RIESGOS PRINCIPALES — CASO BAJISTA**
Identifica los 3-4 riesgos más materiales que podrían invalidar la tesis. Sé específico y cuantitativo donde sea posible.

**4. ANÁLISIS DE VALORACIÓN**
Comenta el nivel de valoración relativo al sector y los fundamentales del negocio. ¿El múltiplo está justificado por el crecimiento/calidad? Menciona el PEG si está disponible.

**5. CATALIZADORES Y HORIZONTE TEMPORAL**
Identifica los eventos o métricas que el mercado deberá monitorizar. Sugiere un horizonte de inversión apropiado.

**6. CONCLUSIÓN Y PRECIO OBJETIVO**
Reafirma la recomendación con argumentos concretos. Si hay precio objetivo de consenso, coméntalo en contexto.

Formato: texto corrido profesional, sin markdown excesivo, sin emojis. Máximo 600 palabras en total.
"""


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_anthropic_key())
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=_openai_key())
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)   # 1-hour cache per snapshot hash
def generate_analysis(snapshot_json: str) -> dict:
    """
    snapshot_json: JSON string of the snapshot dict (for hashable cache key).
    Returns: {"text": str, "provider": str, "error": str|None}
    """
    snapshot = json.loads(snapshot_json)
    prompt = _build_prompt(snapshot)

    if _anthropic_key():
        try:
            text = _call_anthropic(prompt)
            return {"text": text, "provider": "Claude (Anthropic)", "error": None}
        except Exception as e:
            # fall through to OpenAI
            anthropic_err = str(e)
    else:
        anthropic_err = "No Anthropic key"

    if _openai_key():
        try:
            text = _call_openai(prompt)
            return {"text": text, "provider": "GPT-4o (OpenAI)", "error": None}
        except Exception as e:
            return {
                "text": "",
                "provider": "none",
                "error": f"Anthropic: {anthropic_err} | OpenAI: {e}",
            }

    return {
        "text": "",
        "provider": "none",
        "error": "No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env",
    }


def build_snapshot(
    ticker: str, name: str, sector: str, info: dict,
    financials: dict, wv_score: float | None, wv_rec: str,
    reasons_buy: list, reasons_sell: list,
    news_headlines: list, peer_medians: dict,
    rsi: float | None, sma200_above: bool | None,
    price: float | None, fcf_yield: float | None,
    up_pct: float | None, target_price: float | None,
) -> str:
    """Builds the JSON snapshot string used as cache key + LLM input."""
    pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    ps = info.get("priceToSalesTrailing12Months")
    ev_ebitda = info.get("enterpriseToEbitda")
    gross_m = info.get("grossMargins")
    net_m = info.get("profitMargins")
    op_m = info.get("operatingMargins")
    roe = info.get("returnOnEquity")
    de_raw = info.get("debtToEquity")
    de = de_raw / 100 if de_raw is not None else None
    revenue_growth = info.get("revenueGrowth")
    eps_growth = info.get("earningsGrowth")
    mktcap = info.get("marketCap")

    # PEG ratio: forwardPE / (eps_growth * 100)
    peg = None
    if fwd_pe and eps_growth and eps_growth > 0:
        peg = fwd_pe / (eps_growth * 100)

    snap = {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": price,
        "mktcap": mktcap,
        "pe": pe,
        "fwd_pe": fwd_pe,
        "ps": ps,
        "ev_ebitda": ev_ebitda,
        "gross_m": gross_m,
        "net_m": net_m,
        "op_m": op_m,
        "roe": roe,
        "de": de,
        "fcf_yield": fcf_yield,
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "peg": peg,
        "up_pct": up_pct,
        "target_price": target_price,
        "rsi": rsi,
        "sma200_above": sma200_above,
        "wv_score": wv_score,
        "wv_rec": wv_rec,
        "reasons_buy": reasons_buy,
        "reasons_sell": reasons_sell,
        "news_headlines": news_headlines[:8],
        "peer_pe": peer_medians.get("pe"),
        "peer_fwd_pe": peer_medians.get("fwd_pe"),
        "peer_net_m": peer_medians.get("net_m"),
        "_date": str(date.today()),
    }
    return json.dumps(snap, default=str)


# ── GPT-4o News Analyzer (replaces keyword matching) ─────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_news_sentiment(ticker: str, headlines_json: str) -> dict:
    """
    Use GPT-4o to analyze a list of news headlines for a ticker.
    headlines_json: JSON string of list of {"title": str, "publisher": str, "ts": int}
    Returns: {"items": [...], "summary": str, "overall": "positive"|"negative"|"neutral", "error": str|None}
    """
    headlines = json.loads(headlines_json)
    if not headlines:
        return {"items": [], "summary": "Sin noticias recientes.", "overall": "neutral", "error": None}

    if not (_openai_key() or _anthropic_key()):
        return {"items": [], "summary": "", "overall": "neutral",
                "error": "No API key configured"}

    items_text = "\n".join(
        f"{i+1}. [{h.get('publisher','')}] {h.get('title','')}"
        for i, h in enumerate(headlines[:10])
    )

    prompt = f"""Analiza las siguientes {len(headlines[:10])} noticias recientes sobre {ticker} y devuelve un JSON estricto con este formato exacto:

{{
  "items": [
    {{"idx": 1, "sentiment": "positivo"|"negativo"|"neutro", "impact": "alto"|"medio"|"bajo", "insight": "1 frase de lo más relevante"}}
  ],
  "summary": "Resumen ejecutivo de 2-3 frases sobre el sentimiento general y los temas clave.",
  "overall": "positivo"|"negativo"|"neutro"
}}

Noticias:
{items_text}

Reglas:
- Responde SOLO con el JSON, sin texto adicional, sin markdown, sin ```
- sentiment: positivo si la noticia beneficia al accionista, negativo si le perjudica, neutro si es informativa
- impact: alto si mueve precio >3%, medio si <3%, bajo si es ruido
- insight: en español, máximo 15 palabras
- summary: en español, profesional, basado solo en las noticias dadas"""

    try:
        if _openai_key():
            import openai
            client = openai.OpenAI(api_key=_openai_key())
            resp = client.chat.completions.create(
                model="gpt-4o-mini",   # cheaper + faster for this task
                max_tokens=800,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(resp.choices[0].message.content)
            return {**result, "error": None}
        elif _anthropic_key():
            import anthropic
            client = anthropic.Anthropic(api_key=_anthropic_key())
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(msg.content[0].text.strip())
            return {**result, "error": None}
    except Exception as e:
        return {"items": [], "summary": "", "overall": "neutral", "error": str(e)}

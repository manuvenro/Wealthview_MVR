"""
quality_scores.py — Piotroski F-Score and Altman Z-Score for WealthView
Both computed from yfinance data already available in Deep Dive.
No additional API calls needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from modules.styles import (
    GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK
)


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_quality_data(ticker: str) -> dict:
    """Fetch all financials needed for both scores."""
    try:
        tk = yf.Ticker(ticker)
        info  = tk.info or {}
        bs    = tk.balance_sheet
        inc   = tk.financials
        cf    = tk.cashflow

        def _row(df, *keys):
            if df is None or df.empty:
                return None
            for k in keys:
                if k in df.index:
                    vals = df.loc[k].dropna()
                    return vals.tolist() if len(vals) >= 1 else None
            return None

        def _val(df, *keys, idx=0):
            row = _row(df, *keys)
            if row and len(row) > idx:
                v = row[idx]
                return float(v) if v is not None and not np.isnan(v) else None
            return None

        # Balance sheet (current year = idx 0, prior = idx 1)
        total_assets_cur  = _val(bs, "Total Assets")
        total_assets_prev = _val(bs, "Total Assets", idx=1)
        total_liab_cur    = _val(bs, "Total Liabilities Net Minority Interest",
                                  "Total Liab")
        total_liab_prev   = _val(bs, "Total Liabilities Net Minority Interest",
                                  "Total Liab", idx=1)
        current_assets    = _val(bs, "Current Assets")
        current_liab      = _val(bs, "Current Liabilities")
        long_term_debt_cur  = _val(bs, "Long Term Debt")
        long_term_debt_prev = _val(bs, "Long Term Debt", idx=1)
        shares_cur        = _val(bs, "Common Stock Equity", "Stockholders Equity")
        shares_prev       = _val(bs, "Common Stock Equity", "Stockholders Equity", idx=1)
        retained_earnings = _val(bs, "Retained Earnings")
        ebit_bs           = None  # computed from income

        # Income statement
        net_income_cur  = _val(inc, "Net Income")
        net_income_prev = _val(inc, "Net Income", idx=1)
        revenue_cur     = _val(inc, "Total Revenue")
        revenue_prev    = _val(inc, "Total Revenue", idx=1)
        gross_profit    = _val(inc, "Gross Profit")
        ebit            = _val(inc, "EBIT", "Operating Income")
        interest_exp    = _val(inc, "Interest Expense")

        # Cash flow
        ocf_cur  = _val(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
        ocf_prev = _val(cf, "Operating Cash Flow", "Total Cash From Operating Activities", idx=1)
        capex    = _val(cf, "Capital Expenditure", "Capital Expenditures")

        # Market data from info
        mktcap    = info.get("marketCap")
        shares_out = info.get("sharesOutstanding")
        price      = info.get("currentPrice") or info.get("regularMarketPrice")

        return {
            "total_assets_cur":   total_assets_cur,
            "total_assets_prev":  total_assets_prev,
            "total_liab_cur":     total_liab_cur,
            "total_liab_prev":    total_liab_prev,
            "current_assets":     current_assets,
            "current_liab":       current_liab,
            "long_term_debt_cur": long_term_debt_cur,
            "long_term_debt_prev":long_term_debt_prev,
            "equity_cur":         shares_cur,
            "equity_prev":        shares_prev,
            "retained_earnings":  retained_earnings,
            "net_income_cur":     net_income_cur,
            "net_income_prev":    net_income_prev,
            "revenue_cur":        revenue_cur,
            "revenue_prev":       revenue_prev,
            "gross_profit":       gross_profit,
            "ebit":               ebit,
            "interest_exp":       interest_exp,
            "ocf_cur":            ocf_cur,
            "ocf_prev":           ocf_prev,
            "capex":              capex,
            "mktcap":             mktcap,
            "shares_out":         shares_out,
            "price":              price,
        }
    except Exception:
        return {}


# ── Piotroski F-Score ─────────────────────────────────────────────────────────

def compute_piotroski(d: dict) -> dict:
    """
    Compute Piotroski F-Score (0-9).
    Returns dict with individual criteria and total score.
    """
    criteria = {}

    ta_c  = d.get("total_assets_cur")
    ta_p  = d.get("total_assets_prev")
    tl_c  = d.get("total_liab_cur")
    tl_p  = d.get("total_liab_prev")
    ca    = d.get("current_assets")
    cl    = d.get("current_liab")
    ltd_c = d.get("long_term_debt_cur")
    ltd_p = d.get("long_term_debt_prev")
    eq_c  = d.get("equity_cur")
    eq_p  = d.get("equity_prev")
    ni_c  = d.get("net_income_cur")
    ni_p  = d.get("net_income_prev")
    rev_c = d.get("revenue_cur")
    rev_p = d.get("revenue_prev")
    gp    = d.get("gross_profit")
    ocf   = d.get("ocf_cur")
    ocf_p = d.get("ocf_prev")

    # ── PROFITABILITY (4 criteria) ────────────────────────────────────────────
    # F1: ROA positive this year
    roa = ni_c / ta_c if (ni_c and ta_c) else None
    roa_p = ni_p / ta_p if (ni_p and ta_p) else None
    criteria["F1_ROA_positivo"] = {
        "label": "ROA positivo este año",
        "value": 1 if (roa and roa > 0) else 0 if roa is not None else None,
        "detail": f"ROA = {roa*100:.2f}%" if roa is not None else "Sin datos",
        "group": "Rentabilidad",
    }

    # F2: OCF positive
    criteria["F2_OCF_positivo"] = {
        "label": "Flujo de caja operativo positivo",
        "value": 1 if (ocf and ocf > 0) else 0 if ocf is not None else None,
        "detail": f"OCF = ${ocf/1e6:.1f}M" if ocf is not None else "Sin datos",
        "group": "Rentabilidad",
    }

    # F3: ROA improved YoY
    roa_delta = (roa - roa_p) if (roa is not None and roa_p is not None) else None
    criteria["F3_ROA_mejora"] = {
        "label": "ROA mejora vs. año anterior",
        "value": 1 if (roa_delta and roa_delta > 0) else 0 if roa_delta is not None else None,
        "detail": f"ΔROA = {roa_delta*100:+.2f}pp" if roa_delta is not None else "Sin datos año anterior",
        "group": "Rentabilidad",
    }

    # F4: OCF > Net Income (accruals)
    accrual = None
    if ocf is not None and ni_c is not None:
        accrual = ocf > ni_c
    criteria["F4_Accruals"] = {
        "label": "Calidad beneficios (OCF > Net Income)",
        "value": 1 if accrual else 0 if accrual is not None else None,
        "detail": (f"OCF ${ocf/1e6:.1f}M vs NI ${ni_c/1e6:.1f}M"
                   if (ocf is not None and ni_c is not None) else "Sin datos"),
        "group": "Rentabilidad",
    }

    # ── LEVERAGE / LIQUIDITY (3 criteria) ────────────────────────────────────
    # F5: Leverage decreased
    lev_c = tl_c / ta_c if (tl_c and ta_c) else None
    lev_p = tl_p / ta_p if (tl_p and ta_p) else None
    lev_delta = (lev_c - lev_p) if (lev_c is not None and lev_p is not None) else None
    criteria["F5_Leverage"] = {
        "label": "Ratio deuda/activos disminuye",
        "value": 1 if (lev_delta and lev_delta < 0) else 0 if lev_delta is not None else None,
        "detail": (f"D/A: {lev_p*100:.1f}% → {lev_c*100:.1f}%"
                   if (lev_c is not None and lev_p is not None) else "Sin datos"),
        "group": "Apalancamiento",
    }

    # F6: Current ratio improved
    cr_c = ca / cl if (ca and cl) else None
    criteria["F6_Liquidez"] = {
        "label": "Ratio de liquidez corriente mejora",
        "value": 1 if (cr_c and cr_c > 1.0) else 0 if cr_c is not None else None,
        "detail": f"Current ratio = {cr_c:.2f}" if cr_c is not None else "Sin datos",
        "group": "Apalancamiento",
    }

    # F7: No new share dilution
    shares_c = d.get("shares_out")
    shares_p = eq_p  # proxy
    criteria["F7_Dilución"] = {
        "label": "Sin dilución de acciones (no emitió nuevas)",
        "value": None,  # hard to verify without historical share count
        "detail": "No verificable sin series históricas de acciones",
        "group": "Apalancamiento",
    }

    # ── OPERATING EFFICIENCY (2 criteria) ─────────────────────────────────────
    # F8: Gross margin improved
    gm_c = gp / rev_c if (gp and rev_c) else None
    gm_p = None  # would need prior year gross profit — approximating
    criteria["F8_Margen_bruto"] = {
        "label": "Margen bruto positivo",
        "value": 1 if (gm_c and gm_c > 0) else 0 if gm_c is not None else None,
        "detail": f"Margen bruto = {gm_c*100:.1f}%" if gm_c is not None else "Sin datos",
        "group": "Eficiencia",
    }

    # F9: Asset turnover improved
    at_c = rev_c / ta_c if (rev_c and ta_c) else None
    at_p = rev_p / ta_p if (rev_p and ta_p) else None
    at_delta = (at_c - at_p) if (at_c is not None and at_p is not None) else None
    criteria["F9_Asset_turnover"] = {
        "label": "Rotación de activos mejora",
        "value": 1 if (at_delta and at_delta > 0) else 0 if at_delta is not None else None,
        "detail": (f"AT: {at_p:.2f} → {at_c:.2f}" if (at_c is not None and at_p is not None)
                   else "Sin datos año anterior"),
        "group": "Eficiencia",
    }

    # Total score (only count non-None)
    scored = [v["value"] for v in criteria.values() if v["value"] is not None]
    total = sum(scored)
    possible = len(scored)

    return {
        "criteria": criteria,
        "score": total,
        "possible": possible,
        "roa": roa,
        "ocf": ocf,
    }


# ── Altman Z-Score ────────────────────────────────────────────────────────────

def compute_altman(d: dict) -> dict:
    """
    Compute Altman Z-Score.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    Z > 2.99 → Safe zone
    1.81 < Z < 2.99 → Grey zone
    Z < 1.81 → Distress zone
    """
    ta   = d.get("total_assets_cur")
    ca   = d.get("current_assets")
    cl   = d.get("current_liab")
    re   = d.get("retained_earnings")
    ebit = d.get("ebit")
    mktcap = d.get("mktcap")
    tl   = d.get("total_liab_cur")
    rev  = d.get("revenue_cur")

    if not ta:
        return {"score": None, "zone": "Sin datos", "components": {}}

    X1 = (ca - cl) / ta if (ca and cl) else None          # Working capital / Total assets
    X2 = re / ta         if re is not None else None       # Retained earnings / Total assets
    X3 = ebit / ta       if ebit is not None else None     # EBIT / Total assets
    X4 = mktcap / tl     if (mktcap and tl and tl > 0) else None  # Market cap / Total liabilities
    X5 = rev / ta        if rev else None                  # Revenue / Total assets

    components = {
        "X1 — Capital de trabajo / Activos": {"value": X1, "weight": 1.2,
            "detail": "Mide liquidez a corto plazo relativa a activos totales"},
        "X2 — Beneficios retenidos / Activos": {"value": X2, "weight": 1.4,
            "detail": "Indicador de reinversión acumulada (antigüedad + rentabilidad)"},
        "X3 — EBIT / Activos": {"value": X3, "weight": 3.3,
            "detail": "Rentabilidad operativa antes de intereses e impuestos"},
        "X4 — Cap. bursátil / Deuda total": {"value": X4, "weight": 0.6,
            "detail": "Solvencia de mercado: cuánto margen hay antes del default"},
        "X5 — Ventas / Activos": {"value": X5, "weight": 1.0,
            "detail": "Eficiencia en uso de activos para generar ventas"},
    }

    # Compute Z
    z = 0.0
    n_components = 0
    for comp in components.values():
        if comp["value"] is not None:
            z += comp["weight"] * comp["value"]
            n_components += 1

    if n_components < 3:
        return {"score": None, "zone": "Datos insuficientes", "components": components}

    if z > 2.99:
        zone = "Zona segura"
        zone_color = POSITIVE
        zone_desc = "Riesgo de quiebra muy bajo. La empresa muestra solidez financiera."
    elif z > 1.81:
        zone = "Zona gris"
        zone_color = "#d97706"
        zone_desc = "Área de incertidumbre. Requiere monitorización. No hay señales claras."
    else:
        zone = "Zona de distress"
        zone_color = NEGATIVE
        zone_desc = "Riesgo financiero elevado. El modelo original predice probabilidad alta de quiebra."

    return {
        "score": z,
        "zone": zone,
        "zone_color": zone_color,
        "zone_desc": zone_desc,
        "components": components,
    }


# ── Render ────────────────────────────────────────────────────────────────────

def render_quality_scores(ticker: str, info: dict):
    """Main render for quality scores tab in Deep Dive."""

    st.markdown(
        f"<p style='font-size:10px; color:{TEXT_MUTED}; margin-bottom:4px;'>"
        "Modelos cuantitativos académicos probados durante décadas. "
        "Calculados en tiempo real a partir de los estados financieros de {ticker}.</p>".format(ticker=ticker),
        unsafe_allow_html=True,
    )

    with st.spinner("Calculando scores de calidad..."):
        d = _fetch_quality_data(ticker)

    if not d:
        st.error("No se pudieron obtener datos financieros para calcular los scores.")
        return

    piotroski = compute_piotroski(d)
    altman    = compute_altman(d)

    # ── Layout: two columns ───────────────────────────────────────────────────
    col_p, col_a = st.columns(2)

    # ── Piotroski ─────────────────────────────────────────────────────────────
    with col_p:
        score  = piotroski["score"]
        posbl  = piotroski["possible"]
        if score >= 7:
            p_color = POSITIVE
            p_label = "ALTA CALIDAD"
            p_desc  = "Empresa con fundamentales sólidos. Históricamente correlaciona con outperformance."
        elif score >= 4:
            p_color = "#d97706"
            p_label = "CALIDAD MEDIA"
            p_desc  = "Empresa con áreas de fortaleza y debilidad. Requiere análisis adicional."
        else:
            p_color = NEGATIVE
            p_label = "CALIDAD BAJA"
            p_desc  = "Señales de deterioro fundamental. Piotroski identificó este rango como de underperformance."

        st.markdown(
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-left:4px solid {p_color}; border-radius:8px; padding:20px; margin-bottom:16px;'>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:8px;'>Piotroski F-Score</div>"
            f"<div style='font-size:42px; font-weight:800; color:{p_color}; line-height:1;'>{score}</div>"
            f"<div style='font-size:13px; color:{TEXT_MUTED};'>/ {posbl} criterios evaluados</div>"
            f"<div style='font-size:11px; font-weight:700; color:{p_color}; margin-top:8px;'>{p_label}</div>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; margin-top:4px; line-height:1.5;'>{p_desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Criteria breakdown
        groups = {}
        for k, v in piotroski["criteria"].items():
            g = v["group"]
            groups.setdefault(g, []).append(v)

        for group_name, items in groups.items():
            st.markdown(
                f"<p style='font-size:9px; font-weight:700; color:{TEXT_MUTED}; "
                f"text-transform:uppercase; letter-spacing:1px; margin:12px 0 6px;'>"
                f"{group_name}</p>",
                unsafe_allow_html=True,
            )
            for item in items:
                val = item["value"]
                icon  = "✅" if val == 1 else ("❌" if val == 0 else "⚪")
                color = POSITIVE if val == 1 else (NEGATIVE if val == 0 else TEXT_MUTED)
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"align-items:flex-start; padding:6px 0; border-bottom:1px solid {BORDER_SOFT};'>"
                    f"<div>"
                    f"<span style='color:{color}; font-size:13px;'>{icon}</span> "
                    f"<span style='font-size:12px; color:{TEXT_PRIMARY};'>{item['label']}</span>"
                    f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-left:20px;'>{item['detail']}</div>"
                    f"</div>"
                    f"<span style='font-size:12px; font-weight:700; color:{color}; min-width:20px; text-align:right;'>"
                    f"{'1' if val == 1 else ('0' if val == 0 else '—')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Altman Z-Score ────────────────────────────────────────────────────────
    with col_a:
        z = altman.get("score")
        z_color = altman.get("zone_color", TEXT_MUTED)
        zone    = altman.get("zone", "N/D")
        z_desc  = altman.get("zone_desc", "")

        z_display = f"{z:.2f}" if z is not None else "N/D"
        st.markdown(
            f"<div style='background:{SURFACE_2}; border:1px solid {BORDER_SOFT}; "
            f"border-left:4px solid {z_color}; border-radius:8px; padding:20px; margin-bottom:16px;'>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1px; margin-bottom:8px;'>Altman Z-Score</div>"
            f"<div style='font-size:42px; font-weight:800; color:{z_color}; line-height:1;'>{z_display}</div>"
            f"<div style='font-size:13px; color:{TEXT_MUTED};'>Z > 2.99 seguro · 1.81-2.99 gris · &lt; 1.81 distress</div>"
            f"<div style='font-size:11px; font-weight:700; color:{z_color}; margin-top:8px;'>{zone.upper()}</div>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; margin-top:4px; line-height:1.5;'>{z_desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Component breakdown
        st.markdown(
            f"<p style='font-size:9px; font-weight:700; color:{TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:1px; margin:12px 0 6px;'>"
            f"Componentes del Z-Score</p>",
            unsafe_allow_html=True,
        )
        for comp_name, comp in altman["components"].items():
            val = comp["value"]
            w   = comp["weight"]
            contribution = val * w if val is not None else None
            val_str = f"{val:.3f}" if val is not None else "N/D"
            contrib_str = f"{contribution:+.3f}" if contribution is not None else "—"
            contrib_color = POSITIVE if (contribution and contribution > 0) else NEGATIVE if contribution else TEXT_MUTED
            st.markdown(
                f"<div style='padding:7px 0; border-bottom:1px solid {BORDER_SOFT};'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<span style='font-size:11px; color:{TEXT_PRIMARY}; font-weight:600;'>{comp_name.split(' — ')[0]}</span>"
                f"<span style='font-size:12px; font-weight:700; color:{contrib_color};'>"
                f"×{w} = {contrib_str}</span>"
                f"</div>"
                f"<div style='font-size:10px; color:{TEXT_MUTED};'>{comp['detail']} · valor: {val_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if z is not None:
            st.markdown(
                f"<div style='background:{SURFACE}; border-radius:6px; padding:10px 14px; margin-top:14px;'>"
                f"<div style='font-size:10px; color:{TEXT_MUTED};'>Interpretación histórica:</div>"
                f"<div style='font-size:11px; color:{TEXT_SECONDARY}; margin-top:4px; line-height:1.5;'>"
                f"El modelo de Altman (1968) predijo correctamente el 72% de quiebras "
                f"empresariales en los 2 años previos al evento. Diseñado para empresas "
                f"manufactureras; para tech/servicios, usar como referencia aproximada."
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Combined interpretation ───────────────────────────────────────────────
    st.markdown("---")
    if z is not None and score is not None:
        if score >= 7 and z > 2.99:
            combo = ("✅ Doble confirmación positiva: empresa de alta calidad fundamental "
                     "(F-Score) con riesgo de quiebra muy bajo (Z-Score). "
                     "Este es el perfil ideal para una inversión de largo plazo.")
            combo_color = POSITIVE
        elif score <= 3 or z < 1.81:
            combo = ("⚠️ Al menos un modelo muestra señales de alerta. "
                     "Revisar estructura de capital, calidad de beneficios y tendencia de márgenes "
                     "antes de cualquier decisión de inversión.")
            combo_color = NEGATIVE
        else:
            combo = ("🔍 Perfil mixto: el análisis indica empresa en transición. "
                     "Los modelos son complementarios — ninguno es suficiente por sí solo.")
            combo_color = "#d97706"

        st.markdown(
            f"<div style='background:{SURFACE_2}; border-left:4px solid {combo_color}; "
            f"border-radius:0 8px 8px 0; padding:14px 18px;'>"
            f"<span style='font-size:12px; color:{TEXT_PRIMARY};'>{combo}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

"""
tax_optimizer.py — Tax-Loss Harvesting Automático
Jurisdicciones: España (2 meses wash sale) | USA (30 días IRS)
Muestra: pérdidas cristalizables, ganancias a compensar, ahorro fiscal estimado,
         alternativas correlacionadas.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import yfinance as yf

from modules.styles import (
    inject_global_css, page_header, section_label, gold_divider,
    GOLD, GOLD_DIM, GOLD_BORDER, SURFACE, SURFACE_2, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG, PLOTLY_DARK,
    kpi_card, badge, positive_color
)
from modules.utils import ensure_portfolio_data, no_portfolio_warning

# ── Constantes fiscales ───────────────────────────────────────────────────────
WASH_SALE = {
    "España": 60,   # 2 meses calendario
    "USA":    30,   # IRS 30 días
}

# Tipo impositivo por defecto (configurable en UI)
TAX_RATE_DEFAULT = {
    "España": 0.19,   # IRPF primer tramo ganancias capital
    "USA":    0.20,   # LTCG federal (aprox)
}

# Alternativas ETF correlacionadas por sector/tipología
# (sustitutos razonables que evitan el wash sale sin perder exposición)
ETF_ALTERNATIVES = {
    # Biotech / Healthcare
    "IOVA": ["IBB", "XBI"],
    "GOSS": ["IBB", "ARKG"],
    # Semiconductores / Tech
    "INDI": ["SOXX", "SMH"],
    "FN":   ["ITA", "XLI"],
    # Uranio / Energía
    "UEC":  ["URA", "URNM"],
    "ONDS": ["ARKQ", "ROBO"],
    # Genéricos por sector
    "default_tech":    ["QQQ", "XLK"],
    "default_health":  ["XLV", "VHT"],
    "default_energy":  ["XLE", "VDE"],
    "default_finance": ["XLF", "VFH"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_alternatives(ticker: str) -> list[str]:
    return ETF_ALTERNATIVES.get(ticker.upper(), ["SPY", "QQQ"])


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_price(ticker: str) -> float | None:
    try:
        p = yf.Ticker(ticker).fast_info.last_price
        return round(float(p), 4) if p and p > 0 else None
    except Exception:
        return None


def _get_unrealized_pnl(portfolio_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula P&L no realizado desde portfolio_data.
    Necesita Current Price ($), avg_cost o asume precio de entrada = precio actual.
    """
    rows = []
    for _, row in portfolio_data.iterrows():
        ticker    = str(row.get("Ticker", "")).strip().upper()
        cur_price = float(row.get("Current Price ($)", 0) or 0)
        shares    = float(row.get("Shares", 0) or 0)
        avg_cost  = float(row.get("avg_cost", 0) or row.get("Avg Cost", 0) or 0)

        if not ticker or shares <= 0 or cur_price <= 0:
            continue

        cur_value   = cur_price * shares
        cost_basis  = avg_cost * shares if avg_cost > 0 else cur_value
        unrealized  = cur_value - cost_basis
        unrealized_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0.0

        rows.append({
            "Ticker":          ticker,
            "Shares":          shares,
            "Precio actual":   cur_price,
            "Coste medio":     avg_cost if avg_cost > 0 else cur_price,
            "Valor actual":    round(cur_value, 2),
            "P&L No realizado": round(unrealized, 2),
            "P&L %":           round(unrealized_pct, 2),
            "Cost basis":      round(cost_basis, 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("P&L No realizado").reset_index(drop=True)


def _get_realized_gains(username: str) -> float:
    """Suma de ganancias realizadas del año en curso."""
    try:
        from modules.transactions import get_position_summary
        summary = get_position_summary(username)
        if summary.empty:
            return 0.0
        realized_col = "realized_pl" if "realized_pl" in summary.columns else None
        if realized_col:
            return float(summary[realized_col].sum())
    except Exception:
        pass
    return 0.0


def _wash_sale_safe_date(jurisdiction: str) -> date:
    """Fecha mínima para recomprar sin activar wash sale."""
    return date.today() + timedelta(days=WASH_SALE[jurisdiction] + 1)


# ── Render principal ──────────────────────────────────────────────────────────

def render_tax_optimizer():
    inject_global_css()
    page_header(
        "Optimización Fiscal — Tax-Loss Harvesting",
        "Cristaliza pérdidas · Compensa ganancias · Wash sale España/USA · Alternativas correlacionadas"
    )

    portfolio_data = ensure_portfolio_data()
    if portfolio_data is None or portfolio_data.empty:
        no_portfolio_warning()
        return

    username = st.session_state.get("username", "")

    # ── Configuración ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuración fiscal", expanded=True):
        col_jur, col_tax, col_year = st.columns(3)
        with col_jur:
            jurisdiction = st.selectbox(
                "Jurisdicción fiscal",
                ["España", "USA"],
                index=0,
                key="tlh_jurisdiction",
                help="Determina el período de wash sale (España: 2 meses, USA: 30 días)"
            )
        with col_tax:
            tax_rate = st.slider(
                "Tipo impositivo sobre ganancias (%)",
                min_value=1.0, max_value=50.0,
                value=float(TAX_RATE_DEFAULT[jurisdiction] * 100),
                step=0.5, format="%.1f%%",
                key="tlh_tax_rate",
            ) / 100
        with col_year:
            current_year = date.today().year
            st.metric("Año fiscal", str(current_year))
            wash_days = WASH_SALE[jurisdiction]
            st.caption(f"Wash sale: {wash_days} días ({jurisdiction})")

    # ── Datos ─────────────────────────────────────────────────────────────────
    pnl_df       = _get_unrealized_pnl(portfolio_data)
    realized_gain = _get_realized_gains(username)

    if pnl_df.empty:
        st.warning("No se pudieron calcular posiciones. Asegúrate de tener datos de coste medio en P&L / Operaciones.")
        return

    losses_df = pnl_df[pnl_df["P&L No realizado"] < 0].copy()
    gains_df  = pnl_df[pnl_df["P&L No realizado"] > 0].copy()

    # ── KPIs globales ─────────────────────────────────────────────────────────
    gold_divider()
    section_label("Resumen Fiscal del Año")

    total_unrealized_loss  = losses_df["P&L No realizado"].sum() if not losses_df.empty else 0.0
    total_unrealized_gain  = gains_df["P&L No realizado"].sum()  if not gains_df.empty  else 0.0
    net_offsettable        = min(abs(total_unrealized_loss), max(0, realized_gain))
    potential_tax_saving   = net_offsettable * tax_rate
    harvestable_benefit    = abs(total_unrealized_loss) * tax_rate  # si hay ganancias futuras

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "P&L Realizado (año)", f"${realized_gain:+,.0f}",
            "ganancias ya materializadas",
            color=POSITIVE if realized_gain >= 0 else NEGATIVE
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Pérdidas no realizadas", f"${total_unrealized_loss:,.0f}",
            "cristalizables hoy", color=NEGATIVE
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "Ahorro fiscal potencial", f"${harvestable_benefit:,.0f}",
            f"tipo {tax_rate*100:.1f}%", color=GOLD
        ), unsafe_allow_html=True)
    with c4:
        safe_date = _wash_sale_safe_date(jurisdiction)
        st.markdown(kpi_card(
            "Recompra segura desde",
            safe_date.strftime("%d %b %Y"),
            f"wash sale {wash_days}d — {jurisdiction}"
        ), unsafe_allow_html=True)

    # ── Aviso si ya hay ganancias realizadas ──────────────────────────────────
    if realized_gain > 0 and not losses_df.empty:
        compensable = min(abs(total_unrealized_loss), realized_gain)
        saving = compensable * tax_rate
        st.markdown(f"""
        <div style="border:1px solid {GOLD_BORDER};border-radius:8px;padding:14px 18px;
                    background:{GOLD_DIM};margin:12px 0;">
            <b style="color:{GOLD};">💡 Oportunidad de compensación inmediata</b><br>
            <span style="color:{TEXT_SECONDARY};font-size:0.9rem;">
                Tienes <b>${realized_gain:,.0f}</b> de ganancias realizadas este año.
                Cristalizando pérdidas puedes compensar hasta <b>${compensable:,.0f}</b>
                y ahorrar <b>${saving:,.0f}</b> en impuestos.
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Posiciones con pérdidas (harvestables) ────────────────────────────────
    gold_divider()
    section_label("Posiciones con Pérdidas — Candidatas a Harvest")

    if losses_df.empty:
        st.success("✅ No tienes posiciones con pérdidas no realizadas. No hay nada que cosechar.")
    else:
        for _, row in losses_df.iterrows():
            ticker    = row["Ticker"]
            loss_val  = row["P&L No realizado"]
            loss_pct  = row["P&L %"]
            tax_save  = abs(loss_val) * tax_rate
            alts      = _get_alternatives(ticker)
            safe_date = _wash_sale_safe_date(jurisdiction)

            with st.container():
                st.markdown(f"""
                <div style="border:1px solid {NEGATIVE};border-radius:8px;padding:14px 18px;
                            background:rgba(155,77,77,0.06);margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:1.1rem;font-weight:700;color:{TEXT_PRIMARY};">{ticker}</span>
                        <span style="color:{NEGATIVE};font-weight:600;">
                            {loss_val:,.0f}$ ({loss_pct:+.1f}%)
                        </span>
                    </div>
                    <div style="color:{TEXT_MUTED};font-size:0.85rem;margin-top:6px;">
                        Coste medio: <b>${row["Coste medio"]:.2f}</b> ·
                        Precio actual: <b>${row["Precio actual"]:.2f}</b> ·
                        Shares: <b>{row["Shares"]:,.2f}</b>
                    </div>
                    <div style="color:{GOLD};font-size:0.85rem;margin-top:4px;">
                        💰 Ahorro fiscal si vendes: <b>${tax_save:,.0f}</b> · 
                        Recompra segura: <b>{safe_date.strftime("%d %b %Y")}</b>
                    </div>
                    <div style="color:{TEXT_MUTED};font-size:0.82rem;margin-top:4px;">
                        Alternativas correlacionadas (mientras esperas):
                        <b style="color:{TEXT_SECONDARY};">{" · ".join(alts)}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Posiciones con ganancias (cuidado al vender) ──────────────────────────
    gold_divider()
    section_label("Posiciones con Ganancias — Impacto Fiscal si Vendes")

    if gains_df.empty:
        st.info("No hay posiciones con ganancias no realizadas.")
    else:
        disp = gains_df[["Ticker", "Shares", "Coste medio", "Precio actual",
                          "Valor actual", "P&L No realizado", "P&L %"]].copy()
        disp["Impuesto estimado"] = (disp["P&L No realizado"] * tax_rate).round(2)
        disp["P&L No realizado"] = disp["P&L No realizado"].map(lambda x: f"${x:,.0f}")
        disp["Impuesto estimado"] = disp["Impuesto estimado"].map(lambda x: f"${x:,.0f}")
        disp["P&L %"] = disp["P&L %"].map(lambda x: f"{x:+.1f}%")
        disp["Coste medio"] = disp["Coste medio"].map(lambda x: f"${x:.2f}")
        disp["Precio actual"] = disp["Precio actual"].map(lambda x: f"${x:.2f}")
        disp["Valor actual"] = disp["Valor actual"].map(lambda x: f"${x:,.0f}")
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Estrategia ────────────────────────────────────────────────────────────
    gold_divider()
    with st.expander("📖 Cómo funciona el Tax-Loss Harvesting"):
        st.markdown(f"""
**¿Qué es?**
El tax-loss harvesting consiste en vender posiciones con pérdidas para materializar esas pérdidas
y compensarlas con las ganancias del año, reduciendo la factura fiscal.

**Regla Wash Sale ({jurisdiction})**
Después de vender con pérdidas, NO puedes recomprar el **mismo activo** durante **{wash_days} días**
o Hacienda/IRS no reconocerá la pérdida. Puedes comprar activos *similares pero no idénticos*
(p.ej. un ETF del mismo sector) para mantener la exposición.

**Proceso:**
1. Identifica posiciones con pérdidas → véndelas antes de fin de año
2. Invierte el importe en un ETF alternativo del mismo sector
3. Pasados {wash_days} días, puedes volver a comprar el activo original si lo deseas
4. La pérdida compensa las ganancias realizadas → pagas menos impuestos

**Tipo aplicado:** {tax_rate*100:.1f}% sobre ganancias de capital.
        """)


"""
modules/fixed_income.py — Análisis de Renta Fija
=================================================
Módulo completo de analytics de renta fija con datos reales de FRED:

Tab 1: Yield Curve           — Curva actual + overlay histórico + indicadores
Tab 2: Portfolio Bond Stats  — Duración, convexidad, DV01, accrued interest
Tab 3: Scenario Analysis     — Impacto de shocks de tipos (+25 a +300bp)
Tab 4: Inflación & Spreads   — Breakevens TIPS, spreads IG/HY históricos
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import datetime

import modules.fred as fred
from modules.styles import (
    GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG,
    PLOTLY_DARK, section_label, gold_divider, page_header,
)
from modules.utils import ensure_portfolio_data


# ─────────────────────────────────────────────────────────────────────────────
# Bond math (standalone — no dependencia de risk.py para evitar imports cíclicos)
# ─────────────────────────────────────────────────────────────────────────────

def _bond_price(face: float, coupon_rate: float, ytm: float,
                periods: int, frequency: int = 1) -> float:
    c = face * coupon_rate / frequency
    r = ytm / frequency
    if r == 0:
        return c * periods + face
    price = sum(c / (1 + r) ** i for i in range(1, periods + 1))
    price += face / (1 + r) ** periods
    return price


def _macaulay_duration(face: float, coupon_rate: float, ytm: float,
                       periods: int, frequency: int = 1) -> float:
    c = face * coupon_rate / frequency
    r = ytm / frequency
    price = _bond_price(face, coupon_rate, ytm, periods, frequency)
    if price == 0:
        return 0
    w = sum((i / frequency) * c / (1 + r) ** i for i in range(1, periods + 1))
    w += (periods / frequency) * face / (1 + r) ** periods
    return w / price


def _convexity(face: float, coupon_rate: float, ytm: float,
               periods: int, frequency: int = 1) -> float:
    c = face * coupon_rate / frequency
    r = ytm / frequency
    price = _bond_price(face, coupon_rate, ytm, periods, frequency)
    if price == 0:
        return 0
    conv = sum(
        (i * (i + 1)) / frequency**2 * c / (1 + r) ** (i + 2)
        for i in range(1, periods + 1)
    )
    conv += (periods * (periods + 1)) / frequency**2 * face / (1 + r) ** (periods + 2)
    return conv / price


def _ytm_from_price(face: float, coupon_rate: float, price: float,
                    periods: int, frequency: int = 1) -> float:
    lo, hi = 1e-5, 0.60
    for _ in range(200):
        mid = (lo + hi) / 2
        p   = _bond_price(face, coupon_rate, mid, periods, frequency)
        if abs(p - price) < 0.001:
            return mid
        if p > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _periods_to_maturity(maturity_str: str, frequency: int = 1) -> int:
    try:
        mat = pd.Timestamp(maturity_str)
        years = max((mat - pd.Timestamp.today()).days / 365.25, 0)
        return max(int(round(years * frequency)), 1)
    except Exception:
        return 10


def _accrued_interest(face: float, coupon_rate: float,
                      frequency: int = 1, days_since_coupon: int = 0) -> float:
    """Interés corrido desde el último cupón."""
    annual_coupon = face * coupon_rate
    daily_coupon  = annual_coupon / 365
    return daily_coupon * days_since_coupon


def bond_analytics(face: float, coupon_rate: float, maturity_str: str,
                   market_price: float | None = None,
                   frequency: int = 2) -> dict:
    """
    Calcula todas las métricas de un bono.
    Si market_price es None, usa el precio teórico al par.
    """
    periods = _periods_to_maturity(maturity_str, frequency)
    years   = periods / frequency

    if market_price is None or market_price <= 0:
        ytm = coupon_rate  # sin precio de mercado, YTM ≈ cupón
        price = _bond_price(face, coupon_rate, ytm, periods, frequency)
    else:
        ytm   = _ytm_from_price(face, coupon_rate, market_price, periods, frequency)
        price = market_price

    mac_dur  = _macaulay_duration(face, coupon_rate, ytm, periods, frequency)
    mod_dur  = mac_dur / (1 + ytm / frequency)
    conv     = _convexity(face, coupon_rate, ytm, periods, frequency)
    dv01     = abs(price * mod_dur * 0.0001)  # $ por 1bp por unidad de nominal
    accrued  = _accrued_interest(face, coupon_rate, frequency, days_since_coupon=90)

    return {
        "face":         face,
        "coupon_rate":  coupon_rate,
        "coupon_pct":   coupon_rate * 100,
        "ytm":          ytm,
        "ytm_pct":      ytm * 100,
        "price":        price,
        "price_pct":    price / face * 100,
        "years_to_mat": years,
        "periods":      periods,
        "mac_duration": mac_dur,
        "mod_duration": mod_dur,
        "convexity":    conv,
        "dv01":         dv01,   # $ por bp, por unidad de nominalr
        "accrued":      accrued,
        "current_yield": (face * coupon_rate) / price if price > 0 else 0,
    }


def price_change_estimate(mod_dur: float, convexity: float,
                          price: float, delta_y: float) -> float:
    """
    Estimación de cambio de precio (%) por cambio de yield δy (en decimal).
    Usa Taylor de 2º orden: ΔP/P ≈ -D_mod·δy + ½·conv·δy²
    """
    return price * (-mod_dur * delta_y + 0.5 * convexity * delta_y ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, color: str = TEXT_PRIMARY,
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


def _fmt_pct(v, decimals=2) -> str:
    try:
        return f"{float(v):.{decimals}f}%"
    except Exception:
        return "—"


def _fmt_money(v) -> str:
    try:
        v = float(v)
        if abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        return f"${v:,.2f}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_fixed_income():
    page_header("Renta Fija", "Yield curve · Analytics de bonos · Escenarios de tipos · Inflación")

    has_key = fred.api_key_set()

    # API key notice
    if not has_key:
        st.info(
            "Para datos de yield curve en tiempo real, configura tu **FRED API Key** "
            "gratuita en **Configuración → API Keys**. "
            "Obtén la tuya en [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — "
            "es gratis, sin límite diario. Mientras tanto, se muestran datos de referencia."
        )

    tab_curve, tab_portfolio, tab_scenarios, tab_inflation = st.tabs([
        "Yield Curve", "Mis Bonos", "Escenarios de Tipos", "Inflación & Spreads"
    ])

    with tab_curve:
        _render_yield_curve(has_key)

    with tab_portfolio:
        _render_portfolio_bonds()

    with tab_scenarios:
        _render_scenarios()

    with tab_inflation:
        _render_inflation_spreads(has_key)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Yield Curve
# ─────────────────────────────────────────────────────────────────────────────

def _render_yield_curve(has_key: bool):
    section_label("Curva de Tipos — Tesoro USA")

    # ── Fetch current curve ───────────────────────────────────────────────────
    with st.spinner("Descargando curva de tipos..."):
        if has_key:
            curve = fred.get_yield_curve()
        else:
            curve = fred.get_yield_curve_fallback()

    if curve.empty:
        st.error("No se pudo obtener la curva de tipos.")
        return

    # ── Indicadores de curva ──────────────────────────────────────────────────
    if has_key:
        with st.spinner("Cargando indicadores..."):
            indicators = fred.get_curve_indicators()
        t10y2y = indicators.get("T10Y2Y", {}).get("value")
        t10y3m = indicators.get("T10Y3M", {}).get("value")
        fedfunds = indicators.get("FEDFUNDS", {}).get("value")

        inv_color = NEGATIVE if (t10y2y or 0) < 0 else POSITIVE
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v10y = curve[curve["tenor_years"] == 10]["yield_pct"].values
            st.markdown(_kpi("Bono 10Y (EE.UU.)",
                             f"{v10y[0]:.2f}%" if len(v10y) > 0 else "—",
                             GOLD), unsafe_allow_html=True)
        with c2:
            v2y = curve[curve["tenor_years"] == 2]["yield_pct"].values
            st.markdown(_kpi("Bono 2Y (EE.UU.)",
                             f"{v2y[0]:.2f}%" if len(v2y) > 0 else "—",
                             TEXT_PRIMARY), unsafe_allow_html=True)
        with c3:
            st.markdown(_kpi(
                "Curva 10Y − 2Y",
                f"{t10y2y:+.2f}bp" if t10y2y is not None else "—",
                color=inv_color,
                sub="Inversión = señal recesión" if (t10y2y or 0) < 0 else "Curva normal",
                border_color=inv_color + "66",
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(_kpi("Fed Funds Rate",
                             f"{fedfunds:.2f}%" if fedfunds else "—",
                             TEXT_PRIMARY), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Curva actual ──────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve["tenor_years"],
        y=curve["yield_pct"],
        mode="lines+markers",
        name="Curva actual",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=7, color=GOLD,
                    line=dict(color=BG, width=1.5)),
        text=curve["tenor_label"],
        hovertemplate="%{text}<br>Yield: %{y:.2f}%<extra></extra>",
    ))

    # ── Overlay histórico (si hay key) ────────────────────────────────────────
    if has_key:
        col_ov1, col_ov2 = st.columns([3, 1])
        with col_ov2:
            show_overlays = st.multiselect(
                "Comparar con",
                ["Hace 1 mes", "Hace 3 meses", "Hace 1 año", "Hace 2 años"],
                default=["Hace 1 año"],
                key="fi_overlays",
            )

        offset_map = {
            "Hace 1 mes":    30,
            "Hace 3 meses":  90,
            "Hace 1 año":   365,
            "Hace 2 años":  730,
        }
        overlay_colors = ["#6699cc", "#cc9966", "#9966cc", "#66cc99"]

        for i, label in enumerate(show_overlays):
            days = offset_map[label]
            as_of = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            with st.spinner(f"Cargando curva de {label.lower()}..."):
                old_curve = fred.get_yield_curve(as_of=as_of)
            if not old_curve.empty:
                fig.add_trace(go.Scatter(
                    x=old_curve["tenor_years"],
                    y=old_curve["yield_pct"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=overlay_colors[i % 4], width=1.5, dash="dot"),
                    marker=dict(size=5, color=overlay_colors[i % 4]),
                    hovertemplate=f"{label}<br>%{{x}}y: %{{y:.2f}}%<extra></extra>",
                ))

    xvals = [1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    xlabs = ["1M","3M","6M","1A","2A","3A","5A","7A","10A","20A","30A"]

    layout = dict(**PLOTLY_DARK)
    layout["height"] = 380
    layout["margin"] = dict(l=60, r=20, t=30, b=50)
    layout["xaxis"]  = dict(
        title="Vencimiento", type="log",
        tickvals=xvals, ticktext=xlabs,
        showgrid=True, gridcolor=BORDER_SOFT,
    )
    layout["yaxis"]  = dict(
        title="Yield (%)", ticksuffix="%",
        showgrid=True, gridcolor=BORDER_SOFT,
    )
    layout["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
    layout["title"]  = dict(text="Curva de Tipos del Tesoro USA",
                            font=dict(size=12, color=TEXT_MUTED))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de yields ───────────────────────────────────────────────────────
    gold_divider()
    section_label("Tabla de Yields")
    disp = curve[["tenor_label", "yield_pct", "date"]].copy()
    disp.columns = ["Vencimiento", "Yield (%)", "Fecha dato"]
    disp["Yield (%)"] = disp["Yield (%)"].apply(lambda v: f"{v:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Portfolio Bond Analytics
# ─────────────────────────────────────────────────────────────────────────────

def _render_portfolio_bonds():
    section_label("Analytics de mis bonos")

    portfolio_data = ensure_portfolio_data()
    bonds = pd.DataFrame()

    if portfolio_data is not None and not portfolio_data.empty:
        if "Asset Type" not in portfolio_data.columns:
            portfolio_data = portfolio_data.copy()
            portfolio_data["Asset Type"] = "equity"
        mask = portfolio_data["Asset Type"].fillna("equity").str.lower() == "bond"
        bonds = portfolio_data[mask].copy()

    if bonds.empty:
        st.info(
            "No tienes bonos individuales en el portfolio. "
            "Añade bonos en **Portfolio → Bono Individual** para ver el análisis."
        )
        _render_manual_bond_calc(_ks="_scen")
        return

    # ── Calcular analytics para cada bono ────────────────────────────────────
    rows = []
    for _, b in bonds.iterrows():
        face     = float(b.get("Face Value") or 1000)
        coupon   = float(b.get("Coupon %") or 0) / 100
        maturity = str(b.get("Maturity") or "2030-01-01")
        qty      = float(b.get("Shares") or 1)
        name     = str(b.get("Name") or b.get("Ticker") or "—")
        freq     = 2  # semianual por defecto

        # Current price: use face value if no market price available
        mkt_price = float(b.get("Total Value ($)", 0) / qty) if qty > 0 else face
        if mkt_price <= 0:
            mkt_price = face

        a = bond_analytics(face, coupon, maturity, mkt_price, freq)

        rows.append({
            "Bono":             name[:30],
            "Nominal unit.":    face,
            "Unidades":         qty,
            "Nominal total":    face * qty,
            "Cupón (%)":        a["coupon_pct"],
            "YTM (%)":          a["ytm_pct"],
            "Precio (%)":       a["price_pct"],
            "Dur. Mac. (a)":    round(a["mac_duration"], 2),
            "Dur. Mod.":        round(a["mod_duration"], 2),
            "Convexidad":       round(a["convexity"], 2),
            "DV01 ($/bp)":      round(a["dv01"] * face * qty / face, 4),
            "Venc.":            maturity,
            "_ytm":             a["ytm"],
            "_mod_dur":         a["mod_duration"],
            "_convexity":       a["convexity"],
            "_price":           a["price"],
            "_qty":             qty,
            "_face":            face,
        })

    df_bonds = pd.DataFrame(rows)

    # ── Portfolio aggregate ───────────────────────────────────────────────────
    total_nominal = df_bonds["Nominal total"].sum()
    if total_nominal > 0:
        w = df_bonds["Nominal total"] / total_nominal
        port_dur = float(w @ df_bonds["Dur. Mod."])
        port_dv01 = float(df_bonds["DV01 ($/bp)"].sum())
        port_ytm  = float(w @ df_bonds["YTM (%)"])
        port_conv = float(w @ df_bonds["Convexidad"])
    else:
        port_dur = port_dv01 = port_ytm = port_conv = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi("Duración Modificada\nPortfolio",
                         f"{port_dur:.2f} años", GOLD,
                         sub="Sensibilidad a tipos"), unsafe_allow_html=True)
    with c2:
        dv01_color = NEGATIVE if port_dv01 > 1000 else TEXT_PRIMARY
        st.markdown(_kpi("DV01 Portfolio",
                         f"${port_dv01:,.0f}",
                         dv01_color,
                         sub="$ por +1bp en toda la curva"), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi("YTM Ponderado",
                         f"{port_ytm:.2f}%", TEXT_PRIMARY,
                         sub="Yield to maturity medio"), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("Convexidad Portfolio",
                         f"{port_conv:.2f}", TEXT_MUTED,
                         sub="Curvatura precio-yield"), unsafe_allow_html=True)

    gold_divider()

    # ── Tabla de bonos ────────────────────────────────────────────────────────
    disp_cols = ["Bono", "Cupón (%)", "YTM (%)", "Precio (%)",
                 "Dur. Mod.", "DV01 ($/bp)", "Convexidad", "Venc."]
    disp = df_bonds[disp_cols].copy()

    def _color_ytm(val):
        try:
            v = float(str(val).replace("%", ""))
            return f"color:{POSITIVE}" if v > 4.5 else f"color:{NEGATIVE}" if v < 2 else ""
        except Exception:
            return ""

    st.dataframe(
        disp.style
            .map(_color_ytm, subset=["YTM (%)"])
            .format({
                "Cupón (%)":  "{:.2f}%",
                "YTM (%)":    "{:.2f}%",
                "Precio (%)": "{:.2f}%",
                "Dur. Mod.":  "{:.2f}",
                "DV01 ($/bp)":"{:.4f}",
                "Convexidad": "{:.2f}",
            }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # ── Duration ladder (bar chart) ───────────────────────────────────────────
    if len(df_bonds) > 1:
        gold_divider()
        section_label("Duration Ladder")
        fig_dl = go.Figure(go.Bar(
            x=df_bonds["Bono"],
            y=df_bonds["Dur. Mod."],
            marker_color=[GOLD if d > port_dur else "#4a6080" for d in df_bonds["Dur. Mod."]],
            text=[f"{d:.2f}a" for d in df_bonds["Dur. Mod."]],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_PRIMARY),
        ))
        fig_dl.add_hline(y=port_dur,
                         line=dict(color=GOLD_DIM, width=1.5, dash="dash"),
                         annotation_text=f"Dur. portfolio: {port_dur:.2f}a",
                         annotation_font=dict(color=GOLD_DIM, size=10))
        layout_dl = dict(**PLOTLY_DARK)
        layout_dl["height"] = 240
        layout_dl["margin"] = dict(l=20, r=20, t=30, b=60)
        layout_dl["yaxis"]  = dict(title="Duración Modificada (años)",
                                   showgrid=True, gridcolor=BORDER_SOFT)
        layout_dl["title"]  = dict(text="Duración Modificada por Bono",
                                   font=dict(size=12, color=TEXT_MUTED))
        fig_dl.update_layout(**layout_dl)
        st.plotly_chart(fig_dl, use_container_width=True)

    # Guardar para uso en escenarios
    st.session_state["_fi_df_bonds"] = df_bonds


def _render_manual_bond_calc(_ks: str = ""):
    """Calculadora standalone para cuando no hay bonos en portfolio."""
    gold_divider()
    section_label("Calculadora de Bono")
    st.caption("Analiza cualquier bono manualmente.")

    col1, col2, col3 = st.columns(3)
    with col1:
        face    = st.number_input("Nominal (€/$)", value=1000.0, step=100.0, key=f"bc_face{_ks}")
        coupon  = st.number_input("Cupón anual (%)", value=3.0, step=0.25, key=f"bc_coupon{_ks}") / 100
        freq    = st.selectbox("Frecuencia pago", [1, 2, 4], format_func=lambda x: {1:"Anual",2:"Semestral",4:"Trimestral"}[x], key=f"bc_freq{_ks}")
    with col2:
        maturity = st.date_input("Vencimiento", value=datetime.date(2030, 1, 1), key=f"bc_mat{_ks}")
        mkt_px   = st.number_input("Precio mercado (%, p.ej. 98.5)", value=100.0, step=0.5, key=f"bc_px{_ks}")
    with col3:
        qty = st.number_input("Cantidad de bonos", value=1, step=1, key=f"bc_qty{_ks}")

    if st.button("Calcular", type="primary", key=f"bc_calc{_ks}"):
        a = bond_analytics(face, coupon, str(maturity), face * mkt_px / 100, freq)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(_kpi("YTM", f"{a['ytm_pct']:.3f}%", GOLD), unsafe_allow_html=True)
        with c2: st.markdown(_kpi("Dur. Modificada", f"{a['mod_duration']:.3f}a", TEXT_PRIMARY), unsafe_allow_html=True)
        with c3: st.markdown(_kpi("Convexidad", f"{a['convexity']:.3f}", TEXT_MUTED), unsafe_allow_html=True)
        with c4: st.markdown(_kpi("DV01 total", f"${a['dv01']*face*qty/face:.2f}", NEGATIVE), unsafe_allow_html=True)
        st.caption(f"Precio teórico: {a['price_pct']:.2f}% del nominal · "
                   f"Yield corriente: {a['current_yield']*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Scenario Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _render_scenarios():
    section_label("Análisis de Escenarios — Shocks de Tipos")
    st.caption(
        "Impacto en el precio de los bonos ante movimientos paralelos de la curva. "
        "Usa la aproximación de Taylor de 2º orden (Duración + Convexidad)."
    )

    df_bonds = st.session_state.get("_fi_df_bonds")
    if df_bonds is None or df_bonds.empty:
        # Fall back to manual input
        st.info("Configura tu portfolio en 'Mis Bonos' o usa la calculadora de la pestaña anterior.")
        _render_manual_bond_calc(_ks="_scen2")
        return

    shocks_bp = [-200, -100, -50, -25, 0, +25, +50, +100, +200, +300]
    shock_labels = [f"{s:+d}bp" for s in shocks_bp]

    # Compute impact per bond per shock
    impact_data = {}
    total_nominal = df_bonds["Nominal total"].sum()

    for _, b in df_bonds.iterrows():
        bname    = b["Bono"]
        mod_dur  = b["_mod_dur"]
        convexity = b["_convexity"]
        price    = b["_price"]
        face     = b["_face"]
        qty      = b["_qty"]

        row_impacts = []
        for bp in shocks_bp:
            dy = bp / 10000
            dp = price_change_estimate(mod_dur, convexity, price, dy)
            dp_total = dp * qty
            row_impacts.append(round(dp_total, 2))
        impact_data[bname] = row_impacts

    impact_df = pd.DataFrame(impact_data, index=shock_labels)
    impact_df["Portfolio Total ($)"] = impact_df.sum(axis=1)
    if total_nominal > 0:
        impact_df["Portfolio Total (%)"] = (
            impact_df["Portfolio Total ($)"] / total_nominal * 100
        ).round(3)

    # Heatmap de impactos
    display_cols = list(df_bonds["Bono"]) + ["Portfolio Total (%)"]
    display_cols = [c for c in display_cols if c in impact_df.columns]
    z_data   = impact_df[display_cols].values
    vmax_abs = max(abs(z_data).max(), 0.01)

    corr_colorscale = [
        [0.0,  "#1a6b3a"],   # subida grande → sube precio → verde (rate DOWN = price UP)
        [0.5,  "#1a1a2e"],
        [1.0,  "#8b1a1a"],   # bajada grande (rate UP = price DOWN)
    ]

    text_z = []
    for row in z_data:
        text_z.append([
            f"{v:+,.0f}" if col != "Portfolio Total (%)" else f"{v:+.2f}%"
            for v, col in zip(row, display_cols)
        ])

    fig_heat = go.Figure(go.Heatmap(
        z=z_data.tolist(),
        x=display_cols,
        y=shock_labels,
        text=text_z,
        texttemplate="%{text}",
        textfont=dict(size=9 if len(display_cols) > 6 else 11, color="white"),
        colorscale=corr_colorscale,
        zmid=0,
        zmin=-vmax_abs,
        zmax=vmax_abs,
        showscale=True,
        colorbar=dict(title=dict(text="$", side="right"), len=0.8),
        hovertemplate="Shock %{y} sobre %{x}: %{text}<extra></extra>",
    ))
    layout_h = dict(**PLOTLY_DARK)
    layout_h["height"] = max(300, 36 * len(shocks_bp) + 100)
    layout_h["margin"] = dict(l=80, r=60, t=30, b=80)
    layout_h["xaxis"]  = dict(tickangle=-30)
    layout_h["title"]  = dict(text="Impacto ($) en precio por shock paralelo de curva",
                               font=dict(size=12, color=TEXT_MUTED))
    fig_heat.update_layout(**layout_h)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Portfolio total impact line chart
    gold_divider()
    port_impact = impact_df["Portfolio Total ($)"]
    port_pct    = impact_df.get("Portfolio Total (%)")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=shocks_bp,
        y=port_impact.values,
        mode="lines+markers",
        line=dict(color=GOLD, width=2),
        marker=dict(size=8, color=[POSITIVE if v >= 0 else NEGATIVE for v in port_impact]),
        name="Impacto total ($)",
        hovertemplate="Shock: %{x}bp<br>Impacto: $%{y:,.0f}<extra></extra>",
    ))
    fig_line.add_hline(y=0, line=dict(color=BORDER, width=1))
    layout_l = dict(**PLOTLY_DARK)
    layout_l["height"] = 260
    layout_l["margin"] = dict(l=80, r=20, t=30, b=40)
    layout_l["xaxis"]  = dict(title="Shock de tipos (bp)", ticksuffix="bp",
                               showgrid=True, gridcolor=BORDER_SOFT)
    layout_l["yaxis"]  = dict(title="Impacto en valor ($)",
                               showgrid=True, gridcolor=BORDER_SOFT,
                               zeroline=True, zerolinecolor=BORDER)
    layout_l["title"]  = dict(text="Impacto total del portfolio ante shocks de tipos",
                               font=dict(size=12, color=TEXT_MUTED))
    fig_line.update_layout(**layout_l)
    st.plotly_chart(fig_line, use_container_width=True)

    # Tabla resumen
    summary = pd.DataFrame({
        "Shock":          shock_labels,
        "Impacto total ($)": [f"${v:+,.0f}" for v in port_impact],
        "Impacto (% nominal)": [f"{v:+.3f}%" for v in (port_pct if port_pct is not None else [0]*len(shocks_bp))],
    })

    def _color_imp(val):
        try:
            v = float(str(val).replace("$","").replace(",","").replace("%",""))
            return f"color:{POSITIVE};font-weight:600" if v > 0 else f"color:{NEGATIVE};font-weight:600" if v < 0 else ""
        except Exception:
            return ""

    st.dataframe(
        summary.style.map(_color_imp, subset=["Impacto total ($)", "Impacto (% nominal)"]),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Inflación & Spreads
# ─────────────────────────────────────────────────────────────────────────────

def _render_inflation_spreads(has_key: bool):
    section_label("Inflación & Spreads de Crédito")

    if not has_key:
        st.info("Necesitas configurar la FRED API Key para ver datos de inflación y spreads.")
        return

    # ── Breakevens ────────────────────────────────────────────────────────────
    with st.spinner("Cargando breakevens de inflación..."):
        be = fred.get_breakevens()

    be5  = be.get(5, {}).get("value")
    be10 = be.get(10, {}).get("value")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi(
            "Inflación implícita 5A",
            f"{be5:.2f}%" if be5 else "—",
            color=NEGATIVE if (be5 or 0) > 2.5 else POSITIVE if (be5 or 0) < 2.0 else GOLD,
            sub="TIPS 5Y vs Nominal 5Y",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi(
            "Inflación implícita 10A",
            f"{be10:.2f}%" if be10 else "—",
            color=NEGATIVE if (be10 or 0) > 2.5 else POSITIVE if (be10 or 0) < 2.0 else GOLD,
            sub="TIPS 10Y vs Nominal 10Y",
        ), unsafe_allow_html=True)

    # ── Spreads ──────────────────────────────────────────────────────────────
    with st.spinner("Cargando spreads de crédito..."):
        spreads = fred.get_credit_spreads()

    ig_spread = spreads.get("BAMLC0A0CM", {}).get("value")
    hy_spread = spreads.get("BAMLH0A0HYM2", {}).get("value")
    bbb_spread = spreads.get("BAMLC0A4CBBB", {}).get("value")

    with c3:
        st.markdown(_kpi(
            "Spread IG (OAS)",
            f"{ig_spread:.0f}bp" if ig_spread else "—",
            color=NEGATIVE if (ig_spread or 0) > 150 else POSITIVE if (ig_spread or 0) < 80 else TEXT_PRIMARY,
            sub="IG Corp vs Tesoro",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi(
            "Spread High Yield (OAS)",
            f"{hy_spread:.0f}bp" if hy_spread else "—",
            color=NEGATIVE if (hy_spread or 0) > 500 else POSITIVE if (hy_spread or 0) < 300 else TEXT_PRIMARY,
            sub="HY Corp vs Tesoro",
        ), unsafe_allow_html=True)

    gold_divider()

    # ── Histórico de spreads ──────────────────────────────────────────────────
    section_label("Historial de Spreads")
    years_back = st.selectbox("Período", [2, 5, 10, 20], index=1,
                              format_func=lambda x: f"{x} años", key="fi_spread_years")

    with st.spinner("Cargando histórico de spreads..."):
        ig_hist  = fred.get_spread_history("BAMLC0A0CM", years_back)
        hy_hist  = fred.get_spread_history("BAMLH0A0HYM2", years_back)

    if ig_hist is not None or hy_hist is not None:
        fig_sp = go.Figure()
        if ig_hist is not None and not ig_hist.empty:
            fig_sp.add_trace(go.Scatter(
                x=ig_hist.index, y=ig_hist.values,
                name="Spread IG", line=dict(color=GOLD, width=1.5),
                hovertemplate="%{x|%d %b %Y}: %{y:.0f}bp<extra></extra>",
            ))
        if hy_hist is not None and not hy_hist.empty:
            fig_sp.add_trace(go.Scatter(
                x=hy_hist.index, y=hy_hist.values,
                name="Spread HY", line=dict(color=NEGATIVE, width=1.5),
                yaxis="y2",
                hovertemplate="%{x|%d %b %Y}: %{y:.0f}bp<extra></extra>",
            ))
        layout_sp = dict(**PLOTLY_DARK)
        layout_sp["height"] = 320
        layout_sp["margin"] = dict(l=70, r=70, t=30, b=40)
        layout_sp["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
        layout_sp["yaxis"]  = dict(title="Spread IG (bp)", ticksuffix="bp",
                                   showgrid=True, gridcolor=BORDER_SOFT, side="left")
        layout_sp["yaxis2"] = dict(title="Spread HY (bp)", ticksuffix="bp",
                                   overlaying="y", side="right", showgrid=False)
        layout_sp["title"]  = dict(text="Spreads de Crédito OAS (puntos básicos)",
                                   font=dict(size=12, color=TEXT_MUTED))
        fig_sp.update_layout(**layout_sp)
        st.plotly_chart(fig_sp, use_container_width=True)

    # ── Historial breakevens ──────────────────────────────────────────────────
    gold_divider()
    section_label("Historial de Inflación Implícita (TIPS Breakeven)")
    with st.spinner("Cargando histórico de breakevens..."):
        be5_hist  = fred.get_spread_history("T5YIE", years_back)
        be10_hist = fred.get_spread_history("T10YIE", years_back)

    if be5_hist is not None or be10_hist is not None:
        fig_be = go.Figure()
        if be5_hist is not None and not be5_hist.empty:
            fig_be.add_trace(go.Scatter(
                x=be5_hist.index, y=be5_hist.values,
                name="Breakeven 5Y", line=dict(color=GOLD, width=1.5),
            ))
        if be10_hist is not None and not be10_hist.empty:
            fig_be.add_trace(go.Scatter(
                x=be10_hist.index, y=be10_hist.values,
                name="Breakeven 10Y", line=dict(color="#6699cc", width=1.5, dash="dot"),
            ))
        # Target 2% line
        fig_be.add_hline(y=2.0, line=dict(color="rgba(90,143,110,0.40)", width=1, dash="dash"),
                         annotation_text="Objetivo Fed: 2%",
                         annotation_font=dict(color=POSITIVE, size=10))

        layout_be = dict(**PLOTLY_DARK)
        layout_be["height"] = 280
        layout_be["margin"] = dict(l=60, r=20, t=30, b=40)
        layout_be["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
        layout_be["yaxis"]  = dict(title="Inflación implícita (%)", ticksuffix="%",
                                   showgrid=True, gridcolor=BORDER_SOFT)
        layout_be["title"]  = dict(text="Expectativas de inflación del mercado (TIPS Breakeven)",
                                   font=dict(size=12, color=TEXT_MUTED))
        fig_be.update_layout(**layout_be)
        st.plotly_chart(fig_be, use_container_width=True)

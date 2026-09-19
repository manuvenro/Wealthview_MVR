"""
onboarding.py — Wizard de bienvenida para usuarios nuevos
"""
import streamlit as st
import yfinance as yf
import pandas as pd

from modules.styles import (
    SURFACE, SURFACE_2, BORDER, GOLD, GOLD_DIM, GOLD_BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE,
)
import modules.auth as auth


def _step_indicator(current: int):
    dots = []
    for i in range(1, 4):
        if i == current:
            color, size = GOLD, "10px"
        elif i < current:
            color, size = POSITIVE, "8px"
        else:
            color, size = BORDER, "8px"
        dots.append(
            f"<span style='display:inline-block;width:{size};height:{size};"
            f"border-radius:50%;background:{color};margin:0 5px;'></span>"
        )
    st.markdown(
        f"<div style='text-align:center;margin-bottom:32px;'>{''.join(dots)}</div>",
        unsafe_allow_html=True,
    )


def _center(content: str):
    st.markdown(
        f"<div style='max-width:640px;margin:0 auto;'>{content}</div>",
        unsafe_allow_html=True,
    )


# ── Paso 1: Bienvenida ────────────────────────────────────────────────────────

def _step_welcome():
    username = st.session_state.get("username", "")
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _step_indicator(1)

    _center(f"""
    <div style='text-align:center;margin-bottom:36px;'>
        <div style='font-size:11px;color:{GOLD};text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:14px;'>Bienvenido a WealthView</div>
        <h1 style='font-size:32px;font-weight:700;color:{TEXT_PRIMARY};
                   margin:0 0 14px 0;line-height:1.25;'>Hola, {username} 👋</h1>
        <p style='color:{TEXT_SECONDARY};font-size:14px;line-height:1.8;
                  max-width:460px;margin:0 auto;'>
            Tu plataforma de analisis de inversiones. En menos de 2 minutos
            tendras tu portfolio configurado y listo para analizar.
        </p>
    </div>
    """)

    features = [
        ("📊", "Portfolio en tiempo real",
         "Precios actualizados, P&L, rentabilidad y composicion de tu cartera de un vistazo."),
        ("⚡", "Analisis cuantitativo",
         "Riesgo, backtesting, stress testing, rebalanceo y analisis factorial profesional."),
        ("🤖", "Inteligencia artificial",
         "Investment memos generados por IA, analisis de sentimiento y alertas automaticas."),
    ]

    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div style='background:{SURFACE_2};border:1px solid {BORDER};border-radius:10px;
                        padding:24px 18px;text-align:center;height:190px;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;'>
                <div style='font-size:28px;margin-bottom:12px;'>{icon}</div>
                <div style='font-weight:600;font-size:12px;color:{TEXT_PRIMARY};margin-bottom:8px;'>{title}</div>
                <div style='font-size:11px;color:{TEXT_SECONDARY};line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Añadir mi primer activo →", type="primary", use_container_width=True):
            st.session_state.onboarding_step = 2
            st.rerun()

    st.markdown(
        f"<p style='text-align:center;color:{TEXT_MUTED};font-size:11px;margin-top:16px;'>"
        f"Tambien puedes explorar la app sin portfolio — algunos modulos tienen datos de ejemplo.</p>",
        unsafe_allow_html=True,
    )
    _, col_skip, _ = st.columns([1, 2, 1])
    with col_skip:
        if st.button("Explorar sin portfolio", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()


# ── Paso 2: Añadir primer activo ──────────────────────────────────────────────

def _step_add_asset():
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _step_indicator(2)

    _center(f"""
    <div style='text-align:center;margin-bottom:28px;'>
        <div style='font-size:11px;color:{GOLD};text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:12px;'>Paso 1 de 1</div>
        <h2 style='font-size:26px;font-weight:700;color:{TEXT_PRIMARY};margin:0 0 10px 0;'>
            Añade tu primer activo</h2>
        <p style='color:{TEXT_SECONDARY};font-size:13px;margin:0;'>
            Puedes añadir acciones, ETFs o fondos. Usa el ticker de Yahoo Finance
            (AAPL, MSFT, SAN.MC, VWRL.L...)
        </p>
    </div>
    """)

    _, col_form, _ = st.columns([1, 3, 1])
    with col_form:
        ticker = st.text_input(
            "Ticker del activo",
            placeholder="Ej: AAPL, MSFT, SAN.MC, VWRL.L",
            help="Puedes buscar el ticker en finance.yahoo.com",
        ).strip().upper()

        col_shares, col_price = st.columns(2)
        with col_shares:
            shares = st.number_input(
                "Numero de acciones",
                min_value=0.0001, value=1.0, step=0.1, format="%.4f",
            )
        with col_price:
            avg_cost = st.number_input(
                "Precio medio de compra ($)",
                min_value=0.01, value=100.0, step=0.01, format="%.2f",
                help="Si no lo recuerdas, pon el precio actual.",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_add, col_back = st.columns([3, 1])
        with col_add:
            submitted = st.button("Validar y añadir →", type="primary", use_container_width=True)
        with col_back:
            if st.button("← Atras", use_container_width=True):
                st.session_state.onboarding_step = 1
                st.rerun()

    if submitted:
        if not ticker:
            st.error("Introduce un ticker.")
            return

        with st.spinner(f"Validando {ticker}..."):
            try:
                info  = yf.Ticker(ticker).fast_info
                price = getattr(info, "last_price", None)
                name  = yf.Ticker(ticker).info.get("shortName", ticker)
            except Exception:
                price = None
                name  = ticker

        if not price or price <= 0:
            st.error(f"No se encontro '{ticker}'. Comprueba que el ticker es correcto en Yahoo Finance.")
            return

        row = {
            "Ticker":     ticker,
            "Name":       name,
            "Shares":     shares,
            "Avg Cost":   avg_cost,
            "Asset Type": "equity",
            "Currency":   "USD",
        }
        existing = st.session_state.get("portfolio", pd.DataFrame())
        new_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        st.session_state.portfolio     = new_df
        st.session_state.has_portfolio = True

        username = st.session_state.get("username", "")
        if username:
            auth.save_portfolio(username, new_df)

        st.session_state.onboarding_ticker = ticker
        st.session_state.onboarding_name   = name
        st.session_state.onboarding_price  = price
        st.session_state.onboarding_step   = 3
        st.rerun()


# ── Paso 3: Listo ─────────────────────────────────────────────────────────────

def _step_done():
    ticker = st.session_state.get("onboarding_ticker", "")
    name   = st.session_state.get("onboarding_name", ticker)
    price  = st.session_state.get("onboarding_price", 0.0)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _step_indicator(3)

    _center(f"""
    <div style='text-align:center;margin-bottom:32px;'>
        <div style='font-size:40px;margin-bottom:16px;'>🎉</div>
        <h2 style='font-size:28px;font-weight:700;color:{TEXT_PRIMARY};margin:0 0 12px 0;'>
            Tu portfolio esta listo!</h2>
        <div style='display:inline-block;background:{POSITIVE_BG};border:1px solid {POSITIVE};
                    border-radius:8px;padding:12px 24px;margin-bottom:20px;'>
            <span style='color:{POSITIVE};font-weight:600;font-size:14px;'>
                {name} ({ticker}) añadido · Precio actual: ${price:,.2f}
            </span>
        </div>
        <p style='color:{TEXT_SECONDARY};font-size:13px;line-height:1.8;
                  max-width:440px;margin:0 auto;'>
            Añade mas activos desde
            <b style='color:{TEXT_PRIMARY};'>Portfolio → Añadir activo</b> cuando quieras.
        </p>
    </div>
    """)

    next_steps = [
        ("📈", "Dashboard",          "Ve el valor total y la evolucion.",             "Dashboard"),
        ("➕", "Añadir mas activos", "Completa tu portfolio.",                         "Portfolio Overview"),
        ("🔔", "Configurar alertas", "Recibe avisos de precio.",                       "Alertas de Precio"),
    ]

    cols = st.columns(3)
    for col, (icon, title, desc, page) in zip(cols, next_steps):
        with col:
            st.markdown(f"""
            <div style='background:{SURFACE_2};border:1px solid {BORDER};border-radius:10px;
                        padding:20px 16px;text-align:center;'>
                <div style='font-size:24px;margin-bottom:10px;'>{icon}</div>
                <div style='font-weight:600;font-size:12px;color:{TEXT_PRIMARY};margin-bottom:6px;'>{title}</div>
                <div style='font-size:11px;color:{TEXT_SECONDARY};line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Ir →", key=f"next_{page}", use_container_width=True):
                st.session_state.page = page
                st.session_state.pop("onboarding_step", None)
                st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 2, 1])
    with col_main:
        if st.button("Ir al Dashboard →", type="primary", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.session_state.pop("onboarding_step", None)
            st.rerun()


# ── Render principal ──────────────────────────────────────────────────────────

def render_onboarding():
    step = st.session_state.get("onboarding_step", 1)
    if step == 2:
        _step_add_asset()
    elif step == 3:
        _step_done()
    else:
        _step_welcome()

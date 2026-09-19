import streamlit as st

st.set_page_config(
    page_title="WealthView",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Parche global: evita SEGV de pyarrow al serializar DataFrames con columnas object ──
# pyarrow.pandas_compat.convert_column crashea (SIGSEGV) cuando una columna es dtype=object
# y contiene mezcla de tipos Python. Este parche convierte columnas object a str antes
# de que lleguen a Arrow, sin afectar columnas numéricas ni objetos Styler.
import pandas as _pd
_orig_st_dataframe = st.dataframe
def _safe_st_dataframe(data=None, *args, **kwargs):
    if isinstance(data, _pd.DataFrame):
        data = data.copy()
        for _col in data.select_dtypes(include='object').columns:
            data[_col] = data[_col].fillna('—').astype(str)
    return _orig_st_dataframe(data, *args, **kwargs)
st.dataframe = _safe_st_dataframe

from dotenv import load_dotenv
load_dotenv()

import os
import json
import modules.styles as styles
from modules.i18n import t, language_selector
import modules.dashboard as dashboard
import modules.portfolio as portfolio
import modules.watchlist as watchlist
import modules.alerts as alerts
import modules.pdf_report as pdf_report
import modules.risk as risk
import modules.sentiment as sentiment
import modules.stresstest as stresstest
import modules.onboarding as onboarding
import modules.settings as settings
import modules.auth as auth
import modules.investment_memo as investment_memo
import modules.deep_dive as deep_dive
import modules.screener as screener
import modules.transactions as transactions
import modules.optimizer as optimizer
import modules.factor_analysis as factor_analysis
import modules.fixed_income as fixed_income
import modules.rebalancing as rebalancing
import modules.tax_optimizer as tax_optimizer
import modules.options_pricing as options_pricing
import modules.scheduler as scheduler
import modules.legal as legal
from modules.error_handler import safe_render

def save_persisted_session(username):
    # La persistencia en disco se eliminó: todos los visitantes comparten el servidor
    # y un archivo local_session.json haría que cualquier visita entrara como el último usuario.
    # La sesión vive únicamente en st.session_state (por pestaña/navegador).
    pass

def clear_persisted_session():
    pass

def login_screen():
    from modules.styles import GOLD, TEXT_PRIMARY, TEXT_MUTED

    col_lang = st.columns([4, 1])
    with col_lang[1]:
        language_selector()

    st.markdown(f"""
    <div style='max-width:400px; margin:40px auto 32px auto; text-align:center;'>
        <div style='font-family:"Playfair Display",Georgia,serif; font-size:11px;
                    color:{GOLD}; text-transform:uppercase; letter-spacing:3px;
                    margin-bottom:14px;'>Wealth Management</div>
        <div style='font-family:"Playfair Display",Georgia,serif; font-size:32px;
                    font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:-0.5px;
                    line-height:1.1; margin-bottom:8px;'>WealthView</div>
        <div style='color:{TEXT_MUTED}; font-size:12px; text-transform:uppercase;
                    letter-spacing:1.2px;'>{t("login.title")}</div>
    </div>
    """, unsafe_allow_html=True)

    auth.init_db()
    choice = st.radio("", [t("login.tab_login"), t("login.tab_register")], horizontal=True)
    username = st.text_input(t("login.username"))
    password = st.text_input(t("login.password"), type="password")

    if choice == t("login.tab_login"):
        if st.button(t("login.btn_login"), type="primary", use_container_width=True):
            if not username or not password:
                st.warning(t("login.err_empty"))
            else:
                import datetime as _dt
                ok, err_msg = auth.login_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.session_expires_at = (
                        _dt.datetime.utcnow() + _dt.timedelta(hours=auth._SESSION_TIMEOUT_HOURS)
                    )
                    st.rerun()
                else:
                    st.error(err_msg or t("login.err_credentials"))
    else:
        if st.button(t("login.btn_register"), type="primary", use_container_width=True):
            if not username or not password:
                st.warning(t("login.err_empty"))
            elif len(password) < 6:
                st.warning(t("login.err_password_short"))
            elif auth.create_user(username, password):
                st.success(t("login.success_register"))
            else:
                st.error(t("login.err_user_exists"))


def main():
    styles.inject_global_css()
    auth.init_db()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_screen()
        return

    # ── Expiración de sesión ──────────────────────────────────────────────────
    import datetime as _dt
    expires = st.session_state.get("session_expires_at")
    if expires and _dt.datetime.utcnow() > expires:
        st.session_state.logged_in = False
        st.session_state.portfolio_loaded = False
        st.session_state.page = "Onboarding"
        clear_persisted_session()
        st.warning("⏰ Tu sesión ha expirado. Por favor, inicia sesión de nuevo.")
        st.rerun()
        return

    if 'portfolio_loaded' not in st.session_state:
        saved = auth.load_portfolio(st.session_state.username)
        if not saved.empty:
            st.session_state.portfolio = saved
            st.session_state.has_portfolio = True
        else:
            st.session_state.has_portfolio = False
        st.session_state.portfolio_loaded = True

    # ── Scheduler de alertas en background ───────────────────────────────────
    # TEMPORALMENTE DESHABILITADO: el hilo de fondo causaba SEGV al llamar
    # yf.download() con threads internos de numpy/OpenBLAS.
    # TODO: reactivar cuando price_cache esté confirmado estable con threads=False
    if 'scheduler_started' not in st.session_state:
        st.session_state.scheduler_started = True

    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard" if st.session_state.has_portfolio else "Onboarding"

    # ── Sidebar ───────────────────────────────────────────────────────────────
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=160)
    else:
        st.sidebar.title("WealthView 📈")

    st.sidebar.markdown(
        f"<p style='color:#4b5563; font-size:12px; padding:0 8px;'>👤 {st.session_state.username}</p>",
        unsafe_allow_html=True
    )
    if st.sidebar.button(t("nav.logout"), use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.portfolio_loaded = False
        st.session_state.page = "Onboarding"
        clear_persisted_session()
        st.rerun()

    st.sidebar.markdown("---")
    with st.sidebar:
        language_selector()
    st.sidebar.markdown("---")

    # ── Búsqueda global de ticker ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar:
        st.markdown(
            f"<p style='color:#4b5563; font-size:10px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:1px; padding:0 8px; margin-bottom:4px;'>"
            f"🔍 Buscar ticker</p>",
            unsafe_allow_html=True
        )
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_ticker = st.text_input(
                "ticker", label_visibility="collapsed",
                placeholder="AAPL, IOVA...", key="global_ticker_search"
            )
        with search_col2:
            if st.button("→", key="global_search_btn", use_container_width=True):
                if search_ticker.strip():
                    st.session_state.deep_dive_ticker = search_ticker.strip().upper()
                    st.session_state.page = "Deep Dive"
                    st.rerun()
    st.sidebar.markdown("---")

    # ── Páginas ───────────────────────────────────────────────────────────────
    NAV_MEMO = "Investment Memo IA" if t("general.loading") == "Cargando..." else "Investment Memo AI"
    NAV_HORIZON = "Horizonte Temporal" if t("general.loading") == "Cargando..." else "Time Horizon"

    pages = [
        # 1. Dónde estás
        t("nav.dashboard"),
        t("nav.portfolio"),
        t("nav.watchlist"),
        # 2. Riesgo
        t("nav.risk"),
        t("nav.stress"),
        # 3. Análisis de activos
        "Deep Dive",
        "Screener",
        t("nav.sentiment"),
        NAV_MEMO,
        # 4. Optimización
        "Optimización",
        "Rebalanceo",
        "Análisis de Factores",
        # 5. Herramientas especializadas
        "Renta Fija",
        "Pricing & CAPM",
        "Optimización Fiscal",
        # 6. Gestión operativa
        "P&L / Operaciones",
        t("nav.alerts"),
        t("nav.pdf"),
        # 7. Configuración
        t("nav.settings"),
    ]

    current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
    selection = st.sidebar.radio("Navigation", pages, index=current_index)

    if selection != st.session_state.page:
        st.session_state.page = selection
        st.rerun()

    # CSS injection basado en selection (ya sincronizado con session_state)
    active_idx = pages.index(selection) if selection in pages else current_index
    st.sidebar.markdown(f"""
<style>
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child({active_idx + 1}) {{
    background: rgba(201,168,76,0.14) !important;
    color: #e8d070 !important;
    font-weight: 700 !important;
    border-left: 3px solid #c9a84c !important;
    padding-left: 13px !important;
}}
</style>
""", unsafe_allow_html=True)

    # ── Footer legal ──────────────────────────────────────────────────────────
    st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.sidebar.button("📄 Aviso Legal / Privacidad", use_container_width=True):
        st.session_state.page = "Legal"
        st.rerun()

    # ── Render ────────────────────────────────────────────────────────────────
    page = st.session_state.page

    if page == "Onboarding":
        safe_render(onboarding.render_onboarding, page_name="Onboarding")
    elif page == t("nav.dashboard"):
        safe_render(dashboard.render_dashboard, page_name="Dashboard")
    elif page == t("nav.portfolio"):
        safe_render(portfolio.render_portfolio, page_name="Portfolio")
    elif page == t("nav.watchlist"):
        safe_render(watchlist.render_watchlist, page_name="Watchlist")
    elif page == t("nav.alerts"):
        safe_render(alerts.render_alerts, page_name="Alertas")
        st.markdown("---")
        safe_render(alerts.render_event_calendar, page_name="Calendario de Eventos")
    elif page == t("nav.risk"):
        safe_render(risk.render_risk_analytics, page_name="Riesgo")
    elif page == t("nav.stress"):
        safe_render(stresstest.render_stress_test, page_name="Stress Test")
    elif page == NAV_HORIZON:
        from modules.time_horizon import render_time_horizon
        safe_render(render_time_horizon, page_name="Horizonte Temporal")
    elif page == t("nav.sentiment"):
        safe_render(sentiment.render_ai_sentiment, page_name="Sentimiento IA")
    elif page == NAV_MEMO:
        safe_render(investment_memo.render_investment_memo, page_name="Investment Memo")
    elif page == t("nav.pdf"):
        safe_render(pdf_report.render_pdf_export, page_name="Informe PDF")
    elif page == t("nav.settings"):
        safe_render(settings.render_settings, page_name="Configuración")
        safe_render(settings.render_automation_settings, page_name="Automatización")
    elif page == "Deep Dive":
        safe_render(deep_dive.render_deep_dive, page_name="Deep Dive")
    elif page == "Screener":
        safe_render(screener.render_screener, page_name="Screener")
    elif page == "P&L / Operaciones":
        safe_render(transactions.render_transactions, st.session_state.username,
                    page_name="P&L / Operaciones")
    elif page == "Optimización":
        safe_render(optimizer.render_optimizer, page_name="Optimización")
    elif page == "Análisis de Factores":
        safe_render(factor_analysis.render_factor_analysis, page_name="Análisis de Factores")
    elif page == "Renta Fija":
        safe_render(fixed_income.render_fixed_income, page_name="Renta Fija")
    elif page == "Rebalanceo":
        safe_render(rebalancing.render_rebalancing, page_name="Rebalanceo")
    elif page == "Optimización Fiscal":
        safe_render(tax_optimizer.render_tax_optimizer, page_name="Optimización Fiscal")
    elif page == "Pricing & CAPM":
        safe_render(options_pricing.render_options_pricing, page_name="Pricing & CAPM")
    elif page == "Legal":
        safe_render(legal.render_legal, page_name="Legal")


if __name__ == "__main__":
    main()

import streamlit as st
import os
from dotenv import load_dotenv, set_key, find_dotenv
import modules.auth as auth

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')


def render_settings():
    st.title("Configuración ⚙️")
    load_dotenv(ENV_PATH)

    tab1, tab2, tab3 = st.tabs(["🔑 API Keys & IA", "📐 Parámetros", "👤 Cuenta"])

    # ── Tab 1: API Keys ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### OpenAI API Key")
        st.caption("Necesaria para el módulo de Sentimiento IA. Tu clave se guarda localmente en el archivo .env y nunca se comparte.")

        current_key = os.getenv("OPENAI_API_KEY", "")
        masked = f"sk-...{current_key[-6:]}" if len(current_key) > 10 else "No configurada"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div style='background:#0d1117; border:1px solid #161d2b; border-radius:10px;
                        padding:14px 18px; margin-bottom:12px;'>
                <div style='color:#374151; font-size:11px; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:4px;'>Clave actual</div>
                <div style='color:#f9fafb; font-size:14px; font-weight:600; font-family:monospace;'>
                    {masked}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("api_key_form"):
            new_key = st.text_input("Nueva API Key", type="password",
                                    placeholder="sk-proj-...")
            if st.form_submit_button("💾 Guardar API Key", type="primary"):
                if new_key.startswith("sk-"):
                    set_key(ENV_PATH, "OPENAI_API_KEY", new_key)
                    os.environ["OPENAI_API_KEY"] = new_key
                    st.success("✅ API Key guardada correctamente.")
                else:
                    st.error("La clave debe empezar por 'sk-'.")

        st.markdown("---")
        st.caption("¿No tienes API Key? Consíguela en [platform.openai.com](https://platform.openai.com/api-keys)")

        st.markdown("### Anthropic API Key (Claude)")
        st.caption("Necesaria para el Analyst IA en Deep Dive (análisis cualitativo con Claude). Prioridad sobre OpenAI si ambas están configuradas.")

        current_ant = os.getenv("ANTHROPIC_API_KEY", "")
        masked_ant = f"sk-ant-...{current_ant[-6:]}" if len(current_ant) > 10 else "No configurada"

        st.markdown(f"""
        <div style='background:#0d1117; border:1px solid #161d2b; border-radius:10px;
                    padding:14px 18px; margin-bottom:12px;'>
            <div style='color:#374151; font-size:11px; text-transform:uppercase;
                        letter-spacing:0.5px; margin-bottom:4px;'>Clave actual</div>
            <div style='color:#f9fafb; font-size:14px; font-weight:600; font-family:monospace;'>
                {masked_ant}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("anthropic_key_form"):
            new_ant = st.text_input("Nueva Anthropic API Key", type="password",
                                    placeholder="sk-ant-...")
            if st.form_submit_button("💾 Guardar Anthropic Key", type="primary"):
                if new_ant.startswith("sk-ant-") or new_ant.startswith("sk-"):
                    set_key(ENV_PATH, "ANTHROPIC_API_KEY", new_ant)
                    os.environ["ANTHROPIC_API_KEY"] = new_ant
                    st.success("✅ Anthropic API Key guardada correctamente.")
                    st.info("Reinicia la app (Ctrl+C y vuelve a lanzarla) para que el Analyst IA use la nueva clave.")
                else:
                    st.error("La clave de Anthropic debe empezar por 'sk-ant-'.")

        st.markdown("---")
        from modules.llm_analyst import llm_available, active_provider
        if llm_available():
            st.success(f"✅ Analyst IA activo — proveedor: **{active_provider()}**")
        else:
            st.warning("⚠️ No hay API key de IA configurada. El Analyst IA no estará disponible.")
        st.caption("Obtén tu clave en [console.anthropic.com](https://console.anthropic.com/settings/keys)")

        st.markdown("---")
        st.markdown("### Financial Modeling Prep (FMP)")
        st.caption("Fuente primaria para datos fundamentales: FCF, EBITDA, shares outstanding, márgenes. Plan Free: 250 llamadas/día.")

        fmp_key_current = os.getenv("FMP_API_KEY", "")
        with st.form("fmp_key_form"):
            col_fmp1, col_fmp2 = st.columns([3, 1])
            with col_fmp1:
                new_fmp = st.text_input(
                    "FMP API Key",
                    value=fmp_key_current,
                    type="password",
                    placeholder="GIS...",
                )
            with col_fmp2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_fmp = st.form_submit_button("Guardar", type="primary", use_container_width=True)

            if submit_fmp:
                if new_fmp.strip():
                    set_key(ENV_PATH, "FMP_API_KEY", new_fmp.strip())
                    os.environ["FMP_API_KEY"] = new_fmp.strip()
                    st.success("✅ FMP API Key guardada. Reinicia la app para que surta efecto.")
                else:
                    st.warning("Introduce una API key válida.")

        # FMP usage counter
        try:
            from modules.fmp import get_daily_call_count, api_status, _DAILY_LIMIT
            calls_today = get_daily_call_count()
            pct = calls_today / _DAILY_LIMIT * 100
            color = "#e74c3c" if calls_today >= 200 else "#f39c12" if calls_today >= 150 else "#27ae60"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:16px;padding:12px 16px;"
                f"background:#0d1117;border-radius:8px;border:1px solid #161d2b;margin-top:8px;'>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;'>Llamadas FMP hoy</div>"
                f"<div style='font-size:20px;font-weight:700;color:{color};'>{calls_today} / {_DAILY_LIMIT}</div>"
                f"</div>"
                f"<div style='flex:2;'>"
                f"<div style='background:#1a1f2e;border-radius:4px;height:8px;overflow:hidden;'>"
                f"<div style='background:{color};height:100%;width:{min(pct,100):.1f}%;transition:width 0.3s;'></div>"
                f"</div>"
                f"<div style='font-size:10px;color:#374151;margin-top:4px;'>{pct:.1f}% del límite diario</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            if calls_today >= 200:
                st.warning(f"⚠ Se han usado {calls_today}/{_DAILY_LIMIT} llamadas hoy. "
                           "Los datos de FMP están cacheados 6 horas — las nuevas consultas pueden fallar.")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### FRED — Federal Reserve Economic Data")
        st.caption("Fuente gratuita y oficial para yield curve, datos macro e inflación. "
                   "API key gratuita en [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)")

        fred_key_current = os.getenv("FRED_API_KEY", "")
        with st.form("fred_key_form"):
            col_fred1, col_fred2 = st.columns([3, 1])
            with col_fred1:
                new_fred = st.text_input(
                    "FRED API Key",
                    value=fred_key_current,
                    type="password",
                    placeholder="abcdef1234567890...",
                )
            with col_fred2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_fred = st.form_submit_button("Guardar", type="primary", use_container_width=True)
            if submit_fred:
                if new_fred.strip():
                    set_key(ENV_PATH, "FRED_API_KEY", new_fred.strip())
                    os.environ["FRED_API_KEY"] = new_fred.strip()
                    st.success("✅ FRED API Key guardada. Reinicia la app para que surta efecto.")
                else:
                    st.warning("Introduce una API key válida.")

        # FRED connection test
        try:
            from modules.fred import api_status as fred_api_status
            status = fred_api_status()
            if status["ok"]:
                dgs10 = status.get("dgs10", "—")
                st.success(f"✅ FRED conectado — Bono 10Y USA: **{dgs10:.2f}%**")
            elif not status["key_set"]:
                st.caption("🔑 Sin FRED API Key — módulo de Renta Fija en modo referencia.")
            else:
                st.warning(f"⚠ FRED: {status['error']}")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### Polygon.io — Precios en Tiempo Real")
        st.caption(
            "Fuente de precios en tiempo real. Plan gratuito: datos con 15 min de retraso. "
            "Plan Starter ($29/mes): tiempo real + WebSocket. "
            "Consigue tu clave en [polygon.io](https://polygon.io)"
        )

        poly_key_current = os.getenv("POLYGON_API_KEY", "")
        masked_poly = f"...{poly_key_current[-6:]}" if len(poly_key_current) > 6 else "No configurada"

        col_poly_disp, _ = st.columns([2, 1])
        with col_poly_disp:
            st.markdown(f"""
            <div style='background:#0d1117; border:1px solid #161d2b; border-radius:10px;
                        padding:14px 18px; margin-bottom:12px;'>
                <div style='color:#374151; font-size:11px; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:4px;'>Clave actual</div>
                <div style='color:#f9fafb; font-size:14px; font-weight:600; font-family:monospace;'>
                    {masked_poly}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("polygon_key_form"):
            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                new_poly = st.text_input(
                    "Polygon.io API Key",
                    type="password",
                    placeholder="abcd1234EFGH...",
                )
            with col_p2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_poly = st.form_submit_button("Guardar", type="primary", use_container_width=True)
            if submit_poly:
                if new_poly.strip():
                    set_key(ENV_PATH, "POLYGON_API_KEY", new_poly.strip())
                    os.environ["POLYGON_API_KEY"] = new_poly.strip()
                    # Clear Polygon caches so next call picks up new key
                    try:
                        from modules import polygon_client as _pc
                        _pc.get_snapshot.clear()
                        _pc.get_snapshots.clear()
                        _pc.detect_plan.clear()
                    except Exception:
                        pass
                    st.success("✅ Polygon API Key guardada.")
                else:
                    st.warning("Introduce una API key válida.")
        # Polygon connection test
        try:
            from modules.polygon_client import api_status as poly_status
            ps = poly_status()
            if ps["ok"]:
                plan_label = ps["plan"].capitalize()
                realtime_badge = "🟢 Tiempo real" if ps["realtime"] else "🟡 15 min delay"
                price_str = f"AAPL: **${ps['aapl_price']:.2f}**" if ps.get("aapl_price") else ""
                st.success(f"✅ Polygon conectado — Plan: **{plan_label}** · {realtime_badge} {price_str}")
                if not ps["realtime"]:
                    st.info(
                        "Con el plan **Starter ($29/mes)** obtienes precios en tiempo real "
                        "y datos intraday. [Ver planes →](https://polygon.io/dashboard/billing)"
                    )
            elif not ps["key_set"]:
                st.caption(
                    "🔑 Sin Polygon API Key — los precios usan yfinance (15 min delay). "
                    "Regístrate gratis en [polygon.io](https://polygon.io)"
                )
            else:
                st.warning(f"⚠ Polygon: {ps['error']}")
        except Exception:
            pass

    # ── Tab 2: Parámetros ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Parámetros del Motor Cuantitativo")
        st.caption("Estos valores se usan en los cálculos de Análisis de Riesgo y Monte Carlo.")

        # Cargar divisa base desde fx.py (persiste en data/user_config.json)
        try:
            from modules.fx import get_base_currency as _get_base_ccy, set_base_currency as _set_base_ccy
            from modules.fx import SUPPORTED_CURRENCIES as _SUPP_CCY
            _persisted_ccy = _get_base_ccy()
        except Exception:
            _persisted_ccy = "EUR"
            _SUPP_CCY = ["EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD"]

        if 'risk_free_rate' not in st.session_state:
            st.session_state.risk_free_rate = 4.0
        if 'monte_carlo_sims' not in st.session_state:
            st.session_state.monte_carlo_sims = 500
        if 'base_currency' not in st.session_state:
            st.session_state.base_currency = _persisted_ccy

        # Sincronizar session_state con valor persistido si difieren
        if st.session_state.base_currency != _persisted_ccy:
            st.session_state.base_currency = _persisted_ccy

        with st.form("params_form"):
            col1, col2 = st.columns(2)
            with col1:
                rfr = st.number_input(
                    "Tasa Libre de Riesgo (%)",
                    min_value=0.0, max_value=20.0,
                    value=st.session_state.risk_free_rate,
                    step=0.25,
                    help="Se usa en el cálculo del Ratio de Sharpe. Valor actual del bono del tesoro a 10 años."
                )
                _ccy_options = _SUPP_CCY
                _ccy_idx = _ccy_options.index(st.session_state.base_currency) if st.session_state.base_currency in _ccy_options else 0
                currency = st.selectbox(
                    "Divisa base del portfolio",
                    _ccy_options,
                    index=_ccy_idx,
                    help="Todos los valores del portfolio se convertirán a esta divisa usando tipos de cambio en tiempo real."
                )
            with col2:
                sims = st.select_slider(
                    "Simulaciones Monte Carlo",
                    options=[100, 250, 500, 1000, 2000],
                    value=st.session_state.monte_carlo_sims,
                    help="Más simulaciones = más precisión, pero más lento."
                )

            if st.form_submit_button("💾 Guardar Parámetros", type="primary"):
                st.session_state.risk_free_rate = rfr
                st.session_state.monte_carlo_sims = sims
                st.session_state.base_currency = currency
                # Persistir en disco via fx.py
                try:
                    _set_base_ccy(currency)
                    # Limpiar caché de FX si cambia la divisa base
                    from modules.fx import get_fx_rate, get_ticker_currency, fx_table
                    get_fx_rate.clear()
                    get_ticker_currency.clear()
                    fx_table.clear()
                except Exception:
                    pass
                st.success(f"✅ Parámetros actualizados. Divisa base: **{currency}**.")

        st.markdown("---")
        st.markdown("### Valores Actuales")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tasa Libre de Riesgo", f"{st.session_state.risk_free_rate}%")
        c2.metric("Simulaciones MC", f"{st.session_state.monte_carlo_sims:,}")
        c3.metric("Divisa Base", st.session_state.base_currency)

    # ── Tab 3: Cuenta ────────────────────────────────────────────────
    with tab3:
        import datetime as _dt
        from modules.styles import GOLD, GOLD_DIM, SURFACE_2, TEXT_PRIMARY, TEXT_MUTED, POSITIVE, NEGATIVE

        username = st.session_state.get("username", "—")

        # ── Info de cuenta ───────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#0d1117; border:1px solid #161d2b; border-radius:10px;
                    padding:20px 24px; margin-bottom:20px;'>
            <div style='color:#374151; font-size:11px; text-transform:uppercase;
                        letter-spacing:0.5px; margin-bottom:8px;'>{_t("settings.active_user")}</div>
            <div style='color:#f9fafb; font-size:18px; font-weight:700;
                        font-family:monospace;'>{username}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Estado de seguridad ──────────────────────────────────────
        st.markdown(f"#### 🔒 {_t('settings.security_status')}")

        last_login   = auth.get_last_login(username)
        failed       = auth.get_failed_attempts(username)
        locked, until_str = auth._is_locked(username)
        expires      = st.session_state.get("session_expires_at")
        expires_str  = expires.strftime("%H:%M:%S UTC") if expires else "—"

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric(_t("settings.last_login"), last_login or "—")
        with col_s2:
            fail_color = NEGATIVE if failed >= 3 else (GOLD if failed > 0 else POSITIVE)
            st.metric(_t("settings.failed_attempts"), str(failed))
        with col_s3:
            lock_label = f"🔴 {_t('settings.locked_until')} {until_str}" if locked else f"🟢 {_t('settings.not_locked')}"
            st.metric(_t("settings.account_status"), lock_label)
        with col_s4:
            st.metric(_t("settings.session_expires"), expires_str)

        if locked:
            st.warning(f"⚠️ {_t('settings.account_locked_msg')} {until_str}")
            if st.button(f"🔓 {_t('settings.btn_unlock')}", key="unlock_user_btn"):
                auth.unlock_user(username)
                st.success(_t("settings.unlock_ok"))
                st.rerun()

        st.markdown("---")

        # ── Cambiar contraseña ───────────────────────────────────────
        st.markdown(f"#### 🔑 {_t('settings.change_password')}")
        with st.form("change_password_form"):
            current_pw = st.text_input(_t("settings.current_password"), type="password")
            new_pw     = st.text_input(_t("settings.new_password"), type="password")
            confirm_pw = st.text_input(_t("settings.confirm_password"), type="password")
            if st.form_submit_button(_t("settings.btn_change_pwd"), type="primary"):
                if not current_pw or not new_pw or not confirm_pw:
                    st.error("Rellena todos los campos.")
                elif new_pw != confirm_pw:
                    st.error(_t("settings.err_pwd_mismatch"))
                elif len(new_pw) < 6:
                    st.error(_t("settings.err_pwd_short"))
                else:
                    ok, msg = auth.change_password(username, current_pw, new_pw)
                    if ok:
                        st.success(_t("settings.ok_pwd"))
                    else:
                        st.error(msg or _t("settings.err_pwd_wrong"))

        st.markdown("---")

        # ── Timeout de sesión ────────────────────────────────────────
        st.markdown(f"#### ⏱️ {_t('settings.session_timeout')}")
        import json as _json
        _cfg_path = os.path.join("data", "user_config.json")
        _cfg = {}
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path) as _f:
                    _cfg = _json.load(_f)
            except Exception:
                pass
        current_timeout = int(_cfg.get("session_timeout_hours", auth._SESSION_TIMEOUT_HOURS))
        new_timeout = st.number_input(
            _t("settings.timeout_hours"), min_value=1, max_value=72,
            value=current_timeout, step=1, key="session_timeout_input"
        )
        if st.button(_t("settings.btn_save_timeout"), key="save_timeout_btn"):
            _cfg["session_timeout_hours"] = int(new_timeout)
            os.makedirs("data", exist_ok=True)
            with open(_cfg_path, "w") as _f:
                _json.dump(_cfg, _f, indent=2)
            # Actualizar sesión actual
            if st.session_state.get("session_expires_at"):
                st.session_state.session_expires_at = (
                    _dt.datetime.utcnow() + _dt.timedelta(hours=int(new_timeout))
                )
            st.success(f"✅ {_t('settings.timeout_saved')}")

        st.markdown("---")

        # ── Cerrar sesión ────────────────────────────────────────────
        st.markdown(f"#### 🚪 {_t('settings.logout_section')}")
        if st.button(_t("settings.btn_logout"), type="secondary", key="settings_logout_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATIZACIÓN — Informe mensual + SMTP + Rebalanceo programado
# ═══════════════════════════════════════════════════════════════════════════════

def render_automation_settings():
    """
    Sección de automatización en Configuración.
    - Informe mensual PDF: guardar en disco + envío SMTP opcional
    - Configuración SMTP
    - Trigger manual del informe
    """
    import os, json, smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email.mime.text import MIMEText
    from email import encoders
    from datetime import datetime as _dt
    from modules.styles import (
        section_label, gold_divider, GOLD, GOLD_DIM, GOLD_BORDER,
        TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, SURFACE_2
    )
    from modules.utils import ensure_portfolio_data

    DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
    REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
    SMTP_CFG_FILE = os.path.join(DB_PATH, 'smtp_config.json')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)

    def _load_smtp() -> dict:
        if os.path.exists(SMTP_CFG_FILE):
            try:
                with open(SMTP_CFG_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_smtp(cfg: dict):
        with open(SMTP_CFG_FILE, 'w') as f:
            json.dump(cfg, f)

    smtp_cfg = _load_smtp()
    username = st.session_state.get('username', '')

    gold_divider()

    # ── Panel de control del scheduler de alertas ───────────────────────────
    from modules.i18n import t as _t
    section_label(_t("auto.scheduler_title"))
    st.markdown(_t("auto.scheduler_desc"))

    try:
        from modules.scheduler import (
            get_scheduler_state, start_scheduler, stop_scheduler,
            update_interval, run_now
        )
        sched_state = get_scheduler_state()
        is_running  = sched_state.get("running", False)

        # Estado visual
        status_color = "#27ae60" if is_running else "#e74c3c"
        status_text  = _t("auto.scheduler_active") if is_running else _t("auto.scheduler_stopped")
        last_run     = sched_state.get("last_run", "—")
        last_result  = sched_state.get("last_result", "—")
        interval_min = sched_state.get("interval_minutes", 15)

        st.markdown(
            f"<div style='background:#161d2b;border:1px solid #2a3a52;"
            f"border-radius:8px;padding:16px 20px;margin-bottom:16px;'>"
            f"<span style='color:{status_color};font-size:11px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:1px;'>● {status_text}</span>"
            f"&ensp;<span style='color:#8b95a8;font-size:11px;'>{_t('auto.scheduler_interval').format(n=interval_min)}</span><br>"
            f"<span style='color:#8b95a8;font-size:11px;'>{_t('auto.scheduler_last_run')}: {last_run}</span><br>"
            f"<span style='color:#c9c9c9;font-size:11px;margin-top:4px;display:block;'>{last_result}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            new_interval = st.number_input(
                _t("auto.scheduler_interval_input"), min_value=5, max_value=60,
                value=int(interval_min), step=5, key="sched_interval_input"
            )
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            _btn_label = _t("auto.scheduler_btn_update") if is_running else _t("auto.scheduler_btn_start")
            if st.button(_btn_label, key="sched_start_btn", use_container_width=True):
                if is_running:
                    update_interval(username, new_interval)
                    st.success(_t("auto.scheduler_updated").format(n=new_interval))
                else:
                    start_scheduler(username, new_interval)
                    st.success(_t("auto.scheduler_started").format(n=new_interval))
                try:
                    _ucfg_path = os.path.join("data", "user_config.json")
                    _ucfg = {}
                    if os.path.exists(_ucfg_path):
                        with open(_ucfg_path) as _f: _ucfg = json.load(_f)
                    _ucfg["alert_interval_minutes"] = new_interval
                    with open(_ucfg_path, "w") as _f: json.dump(_ucfg, _f, indent=2)
                except Exception:
                    pass
                st.rerun()
        with col_s3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(_t("auto.scheduler_btn_stop"), key="sched_stop_btn",
                         use_container_width=True, disabled=not is_running):
                stop_scheduler()
                st.warning(_t("auto.scheduler_stopped_msg"))
                st.rerun()
        with col_s4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(_t("auto.scheduler_btn_runnow"), key="sched_runnow_btn",
                         use_container_width=True, type="primary"):
                with st.spinner(_t("general.loading")):
                    result = run_now(username)
                st.info(f"{_t('auto.scheduler_result')}: {result}")

        st.caption(_t("auto.scheduler_warning"))

    except ImportError:
        st.error(_t("auto.scheduler_no_apscheduler"))
    except Exception as _e:
        st.warning(f"Panel de scheduler no disponible: {_e}")

    # ── Panel de backup automático ──────────────────────────────────────────
    gold_divider()
    section_label(_t("auto.backup_title"))
    st.markdown(_t("auto.backup_desc"))

    try:
        portfolio_df = ensure_portfolio_data()

        # Config: máximo de backups a conservar
        _ucfg_path = os.path.join("data", "user_config.json")
        _ucfg = {}
        if os.path.exists(_ucfg_path):
            try:
                with open(_ucfg_path) as _f:
                    _ucfg = json.load(_f)
            except Exception:
                pass
        max_bk = int(_ucfg.get("max_backups", 10))

        col_bk1, col_bk2, col_bk3 = st.columns([2, 2, 2])
        with col_bk1:
            new_max_bk = st.number_input(
                _t("auto.backup_max_label"), min_value=3, max_value=50,
                value=max_bk, step=1, key="max_backups_input"
            )
        with col_bk2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(_t("auto.backup_btn_save"), key="save_max_bk_btn", use_container_width=True):
                _ucfg["max_backups"] = new_max_bk
                with open(_ucfg_path, "w") as _f:
                    json.dump(_ucfg, _f, indent=2)
                st.success(_t("auto.backup_saved").format(n=new_max_bk))
        with col_bk3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(_t("auto.backup_btn_now"), key="manual_backup_btn",
                         use_container_width=True, type="primary"):
                if portfolio_df is not None and not portfolio_df.empty:
                    path = auth.backup_portfolio(username, portfolio_df, max_backups=new_max_bk)
                    st.success(f"{_t('auto.backup_created')}: {os.path.basename(path)}")
                    st.rerun()
                else:
                    st.warning(_t("auto.backup_empty"))

        # Lista de backups disponibles
        backups = auth.list_backups(username)
        if not backups:
            st.info(_t("auto.backup_none"))
        else:
            st.markdown(
                f"<div style='color:#8b95a8;font-size:12px;margin-bottom:8px;'>"
                f"{_t('auto.backup_count').format(n=len(backups))}</div>",
                unsafe_allow_html=True
            )
            for bk in backups:
                ts_display = bk['timestamp_str'][:19].replace('T', ' ')
                tickers_preview = ', '.join(bk['tickers'][:5])
                if len(bk['tickers']) > 5:
                    tickers_preview += f" {_t('auto.backup_more_tickers').format(n=len(bk['tickers'])-5)}"

                col_info, col_restore, col_export = st.columns([4, 1, 1])
                with col_info:
                    st.markdown(
                        f"<div style='background:{SURFACE_2};border:1px solid #2a3a52;"
                        f"border-radius:6px;padding:10px 14px;margin-bottom:6px;'>"
                        f"<span style='font-size:12px;font-weight:700;color:#f9fafb;'>{ts_display}</span>"
                        f"&ensp;<span style='font-size:11px;color:#8b95a8;'>"
                        f"{bk['n_positions']} {_t('auto.backup_positions')} · {bk['size_kb']} KB</span><br>"
                        f"<span style='font-size:11px;color:#6b7280;'>{tickers_preview}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_restore:
                    if st.button(_t("auto.backup_btn_restore"), key=f"restore_{bk['filename']}",
                                 use_container_width=True):
                        restored_df = auth.restore_backup(username, bk['filename'])
                        if restored_df is not None and not restored_df.empty:
                            auth.save_portfolio(username, restored_df)
                            st.session_state.portfolio = restored_df
                            st.session_state.portfolio_loaded = False
                            st.success(_t("auto.backup_restored").format(ts=ts_display))
                            st.rerun()
                        else:
                            st.error(_t("auto.backup_restore_error"))
                with col_export:
                    try:
                        with open(bk['path'], 'r', encoding='utf-8') as _ef:
                            _json_bytes = _ef.read().encode('utf-8')
                        st.download_button(
                            _t("auto.backup_btn_export"),
                            data=_json_bytes,
                            file_name=bk['filename'],
                            mime="application/json",
                            key=f"dl_{bk['filename']}",
                            use_container_width=True,
                        )
                    except Exception:
                        pass

    except Exception as _be:
        st.warning(f"Panel de backups no disponible: {_be}")

    # ── Visor de logs ───────────────────────────────────────────────────────────
    gold_divider()
    section_label(_t("auto.logs_title"))

    try:
        from modules.logger import tail_log, log_file_path, log_file_size_kb, clear_log

        _log_path = log_file_path()
        _log_size = log_file_size_kb()

        col_linfo, col_lclear = st.columns([4, 1])
        with col_linfo:
            st.markdown(
                f"<div style='color:#8b95a8;font-size:12px;'>"
                f"📁 <code>{_log_path}</code> &ensp;·&ensp; {_log_size} KB</div>",
                unsafe_allow_html=True
            )
        with col_lclear:
            if st.button(_t("auto.logs_btn_clear"), key="clear_log_btn", use_container_width=True):
                if clear_log():
                    st.success(_t("auto.logs_cleared"))
                    st.rerun()

        n_lines = st.slider(_t("auto.logs_lines_slider"), 20, 500, 100, step=20, key="log_lines_slider")
        lines = tail_log(n_lines)

        if not lines:
            st.info(_t("auto.logs_empty"))
        else:
            # Colorear por nivel
            colored = []
            for line in reversed(lines):  # más reciente primero
                if "| ERROR" in line:
                    color = "#e74c3c"
                elif "| WARNING" in line:
                    color = "#f39c12"
                elif "| DEBUG" in line:
                    color = "#6b7280"
                else:
                    color = "#c9c9c9"
                safe = line.replace("<", "&lt;").replace(">", "&gt;")
                colored.append(
                    f"<div style='font-size:11px;font-family:monospace;"
                    f"color:{color};padding:1px 0;line-height:1.5;'>{safe}</div>"
                )
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid #1e2a3b;border-radius:8px;"
                f"overflow-y:auto;'>{''.join(colored)}</div>"
            , unsafe_allow_html=True)
    except Exception as _e:
        st.warning(f"No se pudo cargar el log: {_e}")

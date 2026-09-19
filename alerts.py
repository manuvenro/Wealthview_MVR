import streamlit as st
import yfinance as yf
import pandas as pd
import modules.auth as auth
from modules.i18n import t


def check_alerts(username):
    """
    Comprueba todas las alertas activas del usuario contra precios actuales.
    Devuelve lista de alertas disparadas: [{'ticker', 'direction', 'threshold', 'price', 'id'}]
    """
    alerts_df = auth.load_alerts(username)
    if alerts_df.empty:
        return []

    active = alerts_df[alerts_df['active'] == 1]
    triggered = []

    # Batch fetch — una sola petición HTTP para todos los tickers activos
    from modules.price_cache import get_prices as _get_prices
    tickers_active = [str(t).upper() for t in active['ticker'].unique().tolist()]
    prices_map = _get_prices(tickers_active) if tickers_active else {}

    for _, row in active.iterrows():
        try:
            price = prices_map.get(str(row['ticker']).upper(), 0.0) or 0.0
            if row['direction'] == 'above' and price >= row['threshold']:
                triggered.append({
                    'id': row['id'], 'ticker': row['ticker'],
                    'direction': 'por encima de', 'threshold': row['threshold'], 'price': price
                })
            elif row['direction'] == 'below' and price <= row['threshold']:
                triggered.append({
                    'id': row['id'], 'ticker': row['ticker'],
                    'direction': 'por debajo de', 'threshold': row['threshold'], 'price': price
                })
        except Exception:
            continue

    return triggered


def send_alert_email_standalone(alerts_triggered: list) -> tuple[bool, str]:
    """
    Envía un email de alerta usando la config SMTP guardada.
    Wrapper standalone para usar desde fuera del scheduler.
    Devuelve (ok, mensaje).
    """
    try:
        import json, os
        cfg_path = os.path.join("data", "smtp_config.json")
        if not os.path.exists(cfg_path):
            return False, "SMTP no configurado."
        with open(cfg_path) as f:
            smtp_cfg = json.load(f)
        if not smtp_cfg.get("enabled") or not smtp_cfg.get("user"):
            return False, "SMTP deshabilitado o sin configurar."

        from modules.scheduler import _send_alert_email
        ok = _send_alert_email(smtp_cfg, alerts_triggered)
        return ok, "Email enviado." if ok else "Error al enviar."
    except Exception as e:
        return False, str(e)


def render_alerts():
    st.title(t("alerts.title"))
    st.caption(t("alerts.subtitle"))

    username = st.session_state.username

    # ── Crear nueva alerta ───────────────────────────────────────────────────
    st.subheader(t("alerts.new"))

    with st.form("alert_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            ticker_input = st.text_input(t("alerts.ticker"))
        with col2:
            above_label = t("alerts.above")
            below_label = t("alerts.below")
            direction = st.selectbox(t("alerts.condition"), [above_label, below_label])
        with col3:
            threshold = st.number_input(t("alerts.threshold"), min_value=0.01, value=100.0, step=0.5)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(t("alerts.btn_create"), use_container_width=True)

        if submitted and ticker_input:
            ticker_clean = ticker_input.strip().upper()
            direction_key = 'above' if direction == above_label else 'below'
            try:
                price = yf.Ticker(ticker_clean).fast_info.last_price
                if price is None or price <= 0:
                    st.error(f"'{ticker_clean}' no es un ticker válido.")
                else:
                    auth.save_alert(username, ticker_clean, direction_key, threshold)
                    st.success(f"✅ Alerta creada: {ticker_clean} {direction.lower()} ${threshold:,.2f} (precio actual: ${price:,.2f})")
                    st.rerun()
            except Exception:
                st.error(f"No se pudo validar '{ticker_clean}'.")

    st.markdown("---")

    # ── Estado actual de alertas ─────────────────────────────────────────────
    st.subheader(t("alerts.my_alerts"))
    alerts_df = auth.load_alerts(username)

    if alerts_df.empty:
        st.info(t("alerts.empty"))
        return

    # Enriquecer con precio actual — batch fetch en una sola petición
    with st.spinner("Comprobando precios..."):
        from modules.price_cache import get_prices as _get_prices
        tickers_list = [str(t).upper() for t in alerts_df['ticker'].unique().tolist()]
        raw_prices = _get_prices(tickers_list) if tickers_list else {}
        prices = {t: round(raw_prices.get(t, 0.0), 2) for t in tickers_list}

    for _, row in alerts_df.iterrows():
        current_price = prices.get(row['ticker'], 0.0)
        direction_label = t('alerts.above') if row['direction'] == 'above' else t('alerts.below')
        is_active = row['active'] == 1

        # Comprobar si está disparada
        triggered = (
            (row['direction'] == 'above' and current_price >= row['threshold']) or
            (row['direction'] == 'below' and current_price <= row['threshold'])
        )

        if triggered and is_active:
            bg = "rgba(245,158,11,0.08)"
            border = "#f59e0b"
            status_label = t("alerts.triggered")
        elif is_active:
            bg = "rgba(34,197,94,0.06)"
            border = "#22c55e"
            status_label = t("alerts.active")
        else:
            bg = "rgba(255,255,255,0.03)"
            border = "#374151"
            status_label = t("alerts.inactive")

        col_card, col_btn = st.columns([5, 1])
        with col_card:
            st.markdown(f"""
            <div style='padding:14px 18px; background:{bg}; border-radius:10px;
                        border-left:3px solid {border}; margin-bottom:8px;
                        border-top:1px solid {border}33; border-right:1px solid {border}22; border-bottom:1px solid {border}22;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-weight:700; font-size:15px; color:#f9fafb;'>{row['ticker']}</span>
                        <span style='color:#6b7280; margin-left:10px; font-size:13px;'>{direction_label} <strong style="color:#e2e8f0;">${row['threshold']:,.2f}</strong></span>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:13px; color:#9ca3af;'>{t("alerts.current_price")}: <strong style="color:#f9fafb;">${current_price:,.2f}</strong></div>
                        <div style='font-size:11px; color:{border}; font-weight:600; margin-top:2px;'>{status_label}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_alert_{row['id']}", help="Eliminar alerta"):
                auth.delete_alert(row['id'])
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDARIO DE EVENTOS — Earnings, Dividendos, Vencimientos de Opciones
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_events_for_ticker(ticker: str) -> dict:
    """
    Obtiene earnings date, próximo dividendo y ex-dividend date para un ticker.
    Devuelve dict con claves: earnings_date, dividend_date, dividend_amount.
    """
    result = {"ticker": ticker, "earnings_date": None, "dividend_date": None,
              "dividend_amount": None, "ex_div_date": None}
    try:
        info = yf.Ticker(ticker).info or {}
        # Earnings
        earnings_ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if earnings_ts:
            from datetime import datetime
            result["earnings_date"] = datetime.utcfromtimestamp(earnings_ts).date()
        # Dividendo
        div_date_ts = info.get("exDividendDate")
        if div_date_ts:
            from datetime import datetime
            result["ex_div_date"] = datetime.utcfromtimestamp(div_date_ts).date()
        result["dividend_amount"] = info.get("dividendRate") or info.get("lastDividendValue")
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_events(tickers_tuple: tuple) -> list[dict]:
    """Obtiene eventos para todos los tickers del portfolio."""
    events = []
    for ticker in tickers_tuple:
        ev = _fetch_events_for_ticker(ticker)
        events.append(ev)
    return events


def _get_option_expiries(username: str) -> list[dict]:
    """Extrae fechas de vencimiento de opciones desde las transacciones del usuario."""
    try:
        from modules.transactions import get_transactions
        txs = get_transactions(username)
        if txs.empty:
            return []
        options = txs[txs["type"].isin(["option_call", "option_put"])] if "type" in txs.columns else pd.DataFrame()
        if options.empty:
            # Try asset_type column
            from modules.utils import ensure_portfolio_data
            portfolio = ensure_portfolio_data()
            if portfolio is not None and "Asset Type" in portfolio.columns:
                options_p = portfolio[portfolio["Asset Type"].isin(["option_call", "option_put"])]
                expiries = []
                for _, row in options_p.iterrows():
                    maturity = row.get("Maturity") or row.get("maturity")
                    if maturity:
                        expiries.append({
                            "ticker": row.get("Ticker", ""),
                            "type":   row.get("Asset Type", "option"),
                            "expiry": str(maturity),
                        })
                return expiries
        return []
    except Exception:
        return []


def render_event_calendar():
    """Renderiza el calendario de eventos del portfolio."""
    from datetime import date, timedelta
    from modules.styles import (
        inject_global_css, section_label, gold_divider,
        GOLD, GOLD_DIM, GOLD_BORDER, SURFACE_2, BORDER,
        TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
        POSITIVE, NEGATIVE, kpi_card
    )
    from modules.utils import ensure_portfolio_data

    inject_global_css()

    portfolio_data = ensure_portfolio_data()
    if portfolio_data is None or portfolio_data.empty:
        st.info(t("events.no_portfolio"))
        return

    username  = st.session_state.get("username", "")
    tickers   = tuple(portfolio_data["Ticker"].unique().tolist())
    today     = date.today()
    horizon   = st.slider(t("events.horizon_slider"), 7, 180, 90,
                          key="evt_horizon_slider")

    with st.spinner(t("events.loading")):
        raw_events = fetch_portfolio_events(tickers)
        option_expiries = _get_option_expiries(username)

    # ── Construir lista unificada ──────────────────────────────────────────
    events: list[dict] = []

    for ev in raw_events:
        ticker = ev["ticker"]
        # Earnings
        ed = ev.get("earnings_date")
        if ed and isinstance(ed, date) and today <= ed <= today + timedelta(days=horizon):
            days_to = (ed - today).days
            events.append({
                "Tipo": "📊 Earnings",
                "Ticker": ticker,
                "Fecha": ed.strftime("%d %b %Y"),
                "Días restantes": days_to,
                "Detalle": "Publicación de resultados trimestrales",
                "_date": ed,
                "_urgency": "high" if days_to <= 7 else "medium" if days_to <= 30 else "low",
            })
        # Dividendo ex-date
        xd = ev.get("ex_div_date")
        if xd and isinstance(xd, date) and today <= xd <= today + timedelta(days=horizon):
            days_to = (xd - today).days
            amt = ev.get("dividend_amount")
            events.append({
                "Tipo": "💵 Dividendo",
                "Ticker": ticker,
                "Fecha": xd.strftime("%d %b %Y"),
                "Días restantes": days_to,
                "Detalle": f"Ex-dividend{f': ${amt:.3f}/acc' if amt else ''}",
                "_date": xd,
                "_urgency": "high" if days_to <= 3 else "medium" if days_to <= 14 else "low",
            })

    # Vencimientos de opciones
    for opt in option_expiries:
        try:
            from datetime import datetime as _dt
            expiry_date = _dt.strptime(str(opt["expiry"])[:10], "%Y-%m-%d").date()
            if today <= expiry_date <= today + timedelta(days=horizon):
                days_to = (expiry_date - today).days
                events.append({
                    "Tipo": "⏰ Vencimiento opción",
                    "Ticker": opt["ticker"],
                    "Fecha": expiry_date.strftime("%d %b %Y"),
                    "Días restantes": days_to,
                    "Detalle": opt.get("type", "option").replace("_", " ").title(),
                    "_date": expiry_date,
                    "_urgency": "high" if days_to <= 7 else "medium",
                })
        except Exception:
            pass

    # Ordenar por fecha
    events.sort(key=lambda x: x["_date"])

    # ── KPIs ──────────────────────────────────────────────────────────────
    gold_divider()
    section_label(t("events.section_title").format(n=horizon))

    n_earnings = sum(1 for e in events if "Earnings" in e["Tipo"])
    n_divs     = sum(1 for e in events if "Dividendo" in e["Tipo"])
    n_opts     = sum(1 for e in events if "Vencimiento" in e["Tipo"])
    n_urgent   = sum(1 for e in events if e["_urgency"] == "high")

    _in_n_days = f"en {horizon} días" if t("general.loading") == "Cargando..." else f"in {horizon} days"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(t("events.earnings"), str(n_earnings), _in_n_days), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(t("events.dividends"), str(n_divs), _in_n_days), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(t("events.options"), str(n_opts), _in_n_days), unsafe_allow_html=True)
    with c4:
        color = NEGATIVE if n_urgent > 0 else POSITIVE
        st.markdown(kpi_card(t("events.urgent"), str(n_urgent), t("events.attention"), color=color), unsafe_allow_html=True)

    # ── Lista de eventos ───────────────────────────────────────────────────
    if not events:
        st.success(t("events.none").format(n=horizon))
        return

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    urgency_colors = {"high": NEGATIVE, "medium": GOLD, "low": TEXT_MUTED}
    urgency_labels = {
        "high":   t("events.label_urgent"),
        "medium": t("events.label_soon"),
        "low":    t("events.label_far"),
    }

    for ev in events:
        color = urgency_colors.get(ev["_urgency"], TEXT_MUTED)
        label = urgency_labels.get(ev["_urgency"], "")
        st.markdown(f"""
        <div style="border-left:3px solid {color};padding:14px 18px;margin-bottom:14px;
                    background:{SURFACE_2};border-radius:0 8px 8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.95rem;">
                    <b style="color:{TEXT_PRIMARY};">{ev['Tipo']}</b>
                    <span style="color:{GOLD};margin-left:8px;">{ev['Ticker']}</span>
                </span>
                <span style="font-size:0.85rem;color:{color};">{label} · {ev['Días restantes']}{t('events.days_left')}</span>
            </div>
            <div style="color:{TEXT_MUTED};font-size:0.82rem;margin-top:5px;">
                📅 {ev['Fecha']} · {ev['Detalle']}
            </div>
        </div>""", unsafe_allow_html=True)

    # Tabla exportable
    with st.expander(t("events.table_title")):
        disp = pd.DataFrame([{k: v for k, v in e.items() if not k.startswith("_")} for e in events])
        st.dataframe(disp, use_container_width=True, hide_index=True)

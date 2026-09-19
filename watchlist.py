import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import modules.auth as auth
from modules.i18n import t
from modules.utils import resolve_isin_name
from modules.styles import PLOTLY_DARK, BORDER, TEXT_SECONDARY, TEXT_MUTED


@st.cache_data(ttl=300, show_spinner=False)
def fetch_watchlist_data(tickers_tuple):
    tickers = list(tickers_tuple)
    results = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            price = info.last_price or 0.0
            prev = info.previous_close or price
            change_pct = ((price - prev) / prev * 100) if prev else 0.0
            high_52w = getattr(info, 'year_high', None)
            low_52w  = getattr(info, 'year_low', None)
            results.append({
                'Ticker': t,
                'Precio ($)': round(price, 2),
                'Var. Hoy (%)': round(change_pct, 2),
                'Máx. 52 sem.': round(high_52w, 2) if high_52w else '-',
                'Mín. 52 sem.': round(low_52w, 2) if low_52w else '-',
            })
        except Exception:
            results.append({
                'Ticker': t, 'Precio ($)': 0.0,
                'Var. Hoy (%)': 0.0, 'Máx. 52 sem.': '-', 'Mín. 52 sem.': '-'
            })
    return pd.DataFrame(results)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_history(ticker, days=30):
    end = datetime.today()
    start = end - timedelta(days=days + 5)  # margen para días sin mercado
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw['Close']
    return raw[['Close']].rename(columns={'Close': ticker})


def render_watchlist():
    st.title(t("watchlist.title"))
    st.caption(t("watchlist.subtitle"))

    username = st.session_state.username

    # ── Cargar watchlist y nombres ────────────────────────────────────────────
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = auth.load_watchlist(username)
    if 'watchlist_names' not in st.session_state:
        st.session_state.watchlist_names = auth.load_watchlist_with_names(username)

    # Resolver automáticamente nombres que faltan (ISIN sin nombre asignado)
    names_updated = False
    for ticker in st.session_state.watchlist:
        current = st.session_state.watchlist_names.get(ticker, ticker)
        if not current or current == ticker:
            resolved = resolve_isin_name(ticker)
            if resolved != ticker:
                st.session_state.watchlist_names[ticker] = resolved
                names_updated = True
    if names_updated:
        auth.save_watchlist(username, st.session_state.watchlist,
                            st.session_state.watchlist_names)

    # ── Añadir ticker ────────────────────────────────────────────────────────
    with st.form("watchlist_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_ticker = st.text_input(t("watchlist.add"))
        with col2:
            new_name = st.text_input("Nombre del activo (opcional, ej: Fondo Indexado SP500)")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(t("watchlist.btn_add"), use_container_width=True)

        if submitted and new_ticker:
            ticker_clean = new_ticker.strip().upper()
            if ticker_clean in st.session_state.watchlist:
                st.warning(f"'{ticker_clean}' ya está en tu watchlist.")
            else:
                with st.spinner(f"Validando {ticker_clean} y buscando nombre..."):
                    try:
                        price = yf.Ticker(ticker_clean).fast_info.last_price
                        if price is None or price <= 0:
                            st.error(f"'{ticker_clean}' no es un ticker válido.")
                        else:
                            # Si el usuario dejó el nombre en blanco, resolverlo automáticamente
                            if new_name.strip():
                                name = new_name.strip()
                            else:
                                name = resolve_isin_name(ticker_clean)
                            st.session_state.watchlist.append(ticker_clean)
                            st.session_state.watchlist_names[ticker_clean] = name
                            auth.save_watchlist(username, st.session_state.watchlist,
                                                st.session_state.watchlist_names)
                            st.rerun()
                    except Exception:
                        st.error(f"No se pudo validar '{ticker_clean}'.")

    # ── Lista vacía ──────────────────────────────────────────────────────────
    if not st.session_state.watchlist:
        st.info(t("watchlist.empty"))
        st.caption("Puedes seguir acciones (AAPL), ETFs (SPY), criptos (BTC-USD), divisas (EURUSD=X) o fondos por ISIN.")
        return

    # ── Tabla de precios ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("Actualizando precios..."):
        df = fetch_watchlist_data(tuple(st.session_state.watchlist))

    # Añadir columna Nombre como primera columna
    names_map = st.session_state.watchlist_names
    df.insert(0, 'Nombre', df['Ticker'].map(lambda tk: names_map.get(tk, tk)))
    df = df.rename(columns={'Ticker': 'ISIN / Ticker'})

    def color_change(val):
        if isinstance(val, float):
            color = '#5a8f6e' if val >= 0 else '#9b4d4d'
            return f'color: {color}; font-weight: 600'
        return ''

    st.dataframe(
        df.style.map(color_change, subset=['Var. Hoy (%)']),
        use_container_width=True,
        hide_index=True
    )

    # ── Acciones por ticker ──────────────────────────────────────────────────
    # Mostrar nombre en el selectbox
    display_options = {tk: f"{names_map.get(tk, tk)} ({tk})" for tk in st.session_state.watchlist}
    col_del1, col_del2, col_del3 = st.columns([4, 1, 1])
    with col_del1:
        options_list = ["—"] + list(display_options.values())
        selected_display = st.selectbox(t("watchlist.select"), options_list)
        to_act = next((tk for tk, v in display_options.items() if v == selected_display), None)
    with col_del2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t("watchlist.btn_remove"), use_container_width=True):
            if to_act:
                st.session_state.watchlist.remove(to_act)
                st.session_state.watchlist_names.pop(to_act, None)
                auth.save_watchlist(username, st.session_state.watchlist,
                                    st.session_state.watchlist_names)
                st.rerun()
    with col_del3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analizar", use_container_width=True, key="wl_deep_dive"):
            if to_act:
                st.session_state.deep_dive_ticker = to_act
                st.session_state.page = "Deep Dive"
                st.rerun()

    # ── Editar nombres ───────────────────────────────────────────────────────
    with st.expander("✏️ Editar nombres de activos"):
        with st.form("edit_names_form", clear_on_submit=False):
            for ticker in st.session_state.watchlist:
                current_name = names_map.get(ticker, ticker)
                new_val = st.text_input(
                    f"Nombre para {ticker}",
                    value=current_name if current_name != ticker else "",
                    placeholder=ticker,
                    key=f"name_edit_{ticker}"
                )
            if st.form_submit_button("💾 Guardar nombres", type="primary"):
                for ticker in st.session_state.watchlist:
                    val = st.session_state.get(f"name_edit_{ticker}", "").strip()
                    st.session_state.watchlist_names[ticker] = val if val else ticker
                auth.save_watchlist(username, st.session_state.watchlist,
                                    st.session_state.watchlist_names)
                st.success("✅ Nombres actualizados.")
                st.rerun()

    # ── Mini gráficos ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Evolución — Último mes")

    is_en = t("watchlist.period") == "Period"
    opts = {
        ("1W"  if is_en else "1 semana"):  7,
        ("1M"  if is_en else "1 mes"):     30,
        ("3M"  if is_en else "3 meses"):   90,
        ("6M"  if is_en else "6 meses"):   180,
        ("1Y"  if is_en else "1 año"):     365,
    }
    period = st.radio(t("watchlist.period"), list(opts.keys()), index=1, horizontal=True)
    days_map = opts
    days = days_map[period]

    cols = st.columns(min(len(st.session_state.watchlist), 3))
    for i, ticker in enumerate(st.session_state.watchlist):
        display_name = names_map.get(ticker, ticker)
        col = cols[i % 3]
        with col:
            try:
                hist = fetch_ticker_history(ticker, days=days)
                if hist.empty:
                    st.warning(f"Sin datos para {ticker}")
                    continue

                prices = hist.iloc[:, 0]
                start_price = prices.iloc[0]
                end_price = prices.iloc[-1]
                change = ((end_price - start_price) / start_price) * 100
                line_color = '#00a86b' if change >= 0 else '#e03131'

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices.index, y=prices.values,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=f'rgba(0,168,107,0.08)' if change >= 0 else 'rgba(224,49,49,0.08)',
                    line=dict(color=line_color, width=2),
                    hovertemplate='%{x|%d %b}<br>$%{y:,.2f}<extra></extra>'
                ))
                title_text = f"<b>{display_name}</b>  <span style='color:{line_color}'>{change:+.2f}%</span>"
                if display_name != ticker:
                    title_text = f"<b>{display_name}</b><br><span style='font-size:9px;color:#4b5563;'>{ticker}</span>  <span style='color:{line_color}'>{change:+.2f}%</span>"
                fig.update_layout(
                    title=dict(
                        text=title_text, x=0,
                        font=dict(color=TEXT_SECONDARY, size=11, family='Inter')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=200,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(
                        showgrid=False, showticklabels=False,
                        tickfont=dict(size=9, color=TEXT_MUTED),
                    ),
                    yaxis=dict(
                        showgrid=True, tickprefix='$',
                        gridcolor=BORDER,
                        tickfont=dict(size=9, color=TEXT_MUTED),
                        tickformat=',.0f',
                    ),
                    showlegend=False,
                    hoverlabel=dict(
                        bgcolor='#1c2333', bordercolor='#c9a84c',
                        font=dict(size=11, color='#e8e0d5', family='Inter'),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f'Error cargando {ticker}: {e}')

    st.caption(f'Precios actualizados cada 5 minutos · {datetime.now().strftime("%H:%M:%S")}')

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import html as html_module
from datetime import datetime, timedelta
from modules.alerts import check_alerts
from modules.i18n import t
from modules.styles import (
    PLOTLY_DARK, BG, SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG, BLUE,
    kpi_card, section_label, gold_divider, page_header, plotly_layout
)


@st.cache_data(ttl=60, show_spinner=False)   # 1 min — Polygon da precios frescos
def fetch_dashboard_data(tickers_tuple):
    # Deduplicate preservando orden — evita 4x filas en el merge si hay tickers repetidos
    seen = set()
    tickers = []
    for tk in tickers_tuple:
        if tk not in seen:
            seen.add(tk)
            tickers.append(tk)

    # ── Polygon batch (1 llamada para todos los tickers) ─────────────────────
    try:
        import modules.polygon_client as pc
        if pc.api_key_set():
            snaps = pc.get_snapshots(tuple(tickers))
            results = []
            missing_yf = []
            for t in tickers:
                s = snaps.get(t, {})
                price = s.get("currentPrice") or 0.0
                prev  = s.get("previousClose") or price
                chg   = s.get("changePercent") or (((price - prev) / prev * 100) if prev else 0.0)
                if price > 0:
                    results.append({'Ticker': t, 'Price': round(price, 2),
                                     'Prev Close': round(prev, 2), 'Change %': round(chg, 2),
                                     '_source': 'polygon'})
                else:
                    missing_yf.append(t)
            # yfinance fallback para tickers sin precio en Polygon (small caps, ISINs, etc.)
            for t in missing_yf:
                try:
                    fi = yf.Ticker(t).fast_info
                    price = float(fi.last_price or 0)
                    prev  = float(fi.previous_close or price)
                    chg   = ((price - prev) / prev * 100) if prev else 0.0
                    results.append({'Ticker': t, 'Price': round(price, 2),
                                     'Prev Close': round(prev, 2), 'Change %': round(chg, 2),
                                     '_source': 'yfinance_fallback'})
                except Exception:
                    results.append({'Ticker': t, 'Price': 0.0, 'Prev Close': 0.0,
                                     'Change %': 0.0, '_source': 'error'})
            if results:
                return pd.DataFrame(results)
    except Exception:
        pass

    # ── yfinance fallback ─────────────────────────────────────────────────────
    results = []
    for t in tickers:
        try:
            info  = yf.Ticker(t).fast_info
            price = info.last_price or 0.0
            prev  = info.previous_close or price
            chg   = ((price - prev) / prev * 100) if prev else 0.0
            results.append({'Ticker': t, 'Price': round(price, 2),
                             'Prev Close': round(prev, 2), 'Change %': round(chg, 2),
                             '_source': 'yfinance'})
        except Exception:
            results.append({'Ticker': t, 'Price': 0.0, 'Prev Close': 0.0, 'Change %': 0.0,
                            '_source': 'error'})
    return pd.DataFrame(results)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(tickers_tuple, days=365):
    tickers = list(tickers_tuple)
    end     = datetime.today()
    start   = end - timedelta(days=days)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw['Close']
    df = raw[['Close']].copy()
    df.columns = tickers
    return df



@st.cache_data(ttl=120, show_spinner=False)
def fetch_pl_summary_cached(username: str):
    """
    Cached wrapper around transactions.get_position_summary().
    Returns DataFrame with WAC/FIFO P&L per ticker, or empty DF on error.
    """
    try:
        from modules.transactions import get_position_summary
        return get_position_summary(username)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_map(tickers_tuple: tuple) -> dict:
    """Fetches sector for each ticker → {ticker: sector}."""
    try:
        from modules.data_provider import fetch_fundamentals_raw
        result = {}
        for t in tickers_tuple:
            try:
                info = fetch_fundamentals_raw(t)
                result[t] = info.get("sector") or "Other"
            except Exception:
                result[t] = "Other"
        return result
    except Exception:
        return {}


def render_dashboard():

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_d = st.columns([3, 1])
    with col_h:
        st.markdown(f"""
        <div style='margin-bottom:28px; padding-bottom:18px; border-bottom:1px solid {BORDER};'>
            <div style='font-family:"Playfair Display",Georgia,serif; font-size:28px;
                        font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:-0.3px;'>
                {t("dashboard.title")}</div>
            <div style='color:{TEXT_MUTED}; font-size:11px; margin-top:5px;
                        text-transform:uppercase; letter-spacing:1px;'>
                {t("dashboard.subtitle")}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        # Market status + price source badge
        try:
            import modules.polygon_client as _pc
            if _pc.api_key_set():
                mkt = _pc.get_market_status()
                mkt_label = mkt.get("market", "unknown")
                mkt_color = {"open": "#27ae60", "closed": "#95a5a6",
                             "extended-hours": "#f39c12"}.get(mkt_label, "#95a5a6")
                plan = _pc.detect_plan()
                realtime = plan not in ("free", "none", "unknown")
                src_label = "Real-time · Polygon" if realtime else "15 min · Polygon"
                src_color = "#27ae60" if realtime else "#f39c12"
            else:
                mkt_label, mkt_color = "—", "#95a5a6"
                src_label, src_color = "Delayed · yfinance", "#95a5a6"
        except Exception:
            mkt_label, mkt_color = "—", "#95a5a6"
            src_label, src_color = "Delayed · yfinance", "#95a5a6"

        st.markdown(
            f"<div style='text-align:right; color:{TEXT_MUTED}; font-size:11px; "
            f"text-transform:uppercase; letter-spacing:0.8px; margin-top:14px;'>"
            f"{datetime.today().strftime('%d %B %Y')}<br>"
            f"<span style='color:{GOLD}; font-size:13px; font-weight:600;'>"
            f"{datetime.now().strftime('%H:%M')} CET</span><br><br>"
            f"<span style='color:{mkt_color}; font-size:10px;'>● {mkt_label.upper()}</span>"
            f"&ensp;<span style='color:{src_color}; font-size:10px;'>● {src_label}</span>"
            f"</div>",
            unsafe_allow_html=True)

    # ── Alertas ───────────────────────────────────────────────────────────────
    try:
        triggered = check_alerts(st.session_state.username)
        if triggered:
            for a in triggered:
                st.warning(
                    f"🔔 **{a['ticker']}** — precio actual **${a['price']:,.2f}** "
                    f"está {a['direction']} **${a['threshold']:,.2f}**")
    except Exception:
        pass

    # ── Sin portfolio ─────────────────────────────────────────────────────────
    if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
        st.markdown(f"""
        <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                    padding:60px 40px; text-align:center; margin-top:40px;'>
            <div style='font-family:"Playfair Display",serif; font-size:22px;
                        color:{TEXT_PRIMARY}; margin-bottom:12px;'>Bienvenido a WealthView</div>
            <p style='color:{TEXT_MUTED}; font-size:13px; max-width:400px;
                      margin:0 auto 24px auto; line-height:1.7;'>
                Configure su portfolio para acceder al análisis completo.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("Configurar Portfolio →", type="primary"):
            st.session_state.page = "Portfolio Overview"
            st.rerun()
        return

    portfolio_df = st.session_state.portfolio.copy()
    tickers = portfolio_df['Ticker'].tolist()

    with st.spinner("Actualizando valoraciones..."):
        live_df = fetch_dashboard_data(tuple(tickers))

    merged = portfolio_df.merge(live_df, on='Ticker', how='left')

    # Sanitizar columnas numéricas — evita ValueError si hay NaN (ej: bonos sin precio de mercado)
    for col in ['Price', 'Prev Close', 'Change %', 'Shares']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)

    merged['Value']      = merged['Shares'] * merged['Price']
    merged['Prev Value'] = merged['Shares'] * merged['Prev Close']

    # ── Multi-divisa: convertir a divisa base ────────────────────────────────
    try:
        from modules.fx import get_ticker_currency, get_fx_rate, get_base_currency, ccy_symbol

        base_ccy = get_base_currency()
        base_sym = ccy_symbol(base_ccy)

        # Detectar divisa por fila (usar columna guardada si existe, si no detectar)
        def _row_ccy(row):
            stored = row.get('Currency') if 'Currency' in merged.columns else None
            if stored and isinstance(stored, str) and len(stored) == 3:
                return stored.upper()
            return get_ticker_currency(str(row['Ticker']))

        merged['_Currency'] = merged.apply(_row_ccy, axis=1)
        merged['_FX']       = merged['_Currency'].apply(lambda c: get_fx_rate(c, base_ccy))
        merged['Value_Base']      = merged['Value']      * merged['_FX']
        merged['Prev Value_Base'] = merged['Prev Value'] * merged['_FX']

        total_aum  = merged['Value_Base'].sum()
        total_prev = merged['Prev Value_Base'].sum()

        # Flag para saber si el portfolio es multi-divisa (más de 1 divisa distinta)
        unique_ccys = merged['_Currency'].unique()
        is_multiccy = len(unique_ccys) > 1 or (len(unique_ccys) == 1 and unique_ccys[0] != base_ccy)

    except Exception:
        # Fallback sin conversión si fx.py falla
        base_ccy   = "USD"
        base_sym   = "$"
        merged['Value_Base']      = merged['Value']
        merged['Prev Value_Base'] = merged['Prev Value']
        merged['_Currency']       = "USD"
        merged['_FX']             = 1.0
        total_aum  = merged['Value_Base'].sum()
        total_prev = merged['Prev Value_Base'].sum()
        is_multiccy = False

    daily_pnl  = total_aum - total_prev
    daily_pct  = (daily_pnl / total_prev * 100) if total_prev > 0 else 0.0

    # Guardar portfolio_data incluyendo Name, Currency y FX si existen
    data_cols = ['Ticker', 'Shares', 'Price', 'Value']
    if 'Name' in merged.columns:
        data_cols.insert(1, 'Name')
    _save = merged[data_cols].rename(
        columns={'Price': 'Current Price ($)', 'Value': 'Total Value ($)'})
    if '_Currency' in merged.columns:
        _save['Currency'] = merged['_Currency'].values
    if '_FX' in merged.columns:
        _save['FX Rate']  = merged['_FX'].values
    _save[f'Total Value ({base_ccy})'] = merged['Value_Base'].values
    st.session_state.portfolio_data = _save

    # ── KPIs ──────────────────────────────────────────────────────────────────
    pnl_color = POSITIVE if daily_pnl >= 0 else NEGATIVE
    pnl_sign  = "+" if daily_pnl >= 0 else "-"
    best      = merged.loc[merged['Change %'].idxmax()]
    worst     = merged.loc[merged['Change %'].idxmin()]

    # Símbolo para precio nativo de best/worst (pueden estar en USD aunque base sea EUR)
    best_sym  = ccy_symbol(best.get('_Currency', 'USD')) if '_Currency' in merged.columns else "$"
    worst_sym = ccy_symbol(worst.get('_Currency', 'USD')) if '_Currency' in merged.columns else "$"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        subtitle_aum = f"{len(tickers)} {t('dashboard.positions')}"
        if is_multiccy:
            subtitle_aum += f" · {', '.join(unique_ccys[:3])}"
        st.markdown(kpi_card(t("dashboard.aum"), f"{base_sym}{total_aum:,.0f}",
                             subtitle_aum), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(t("dashboard.pnl"),
                             f"{pnl_sign}{base_sym}{abs(daily_pnl):,.0f}",
                             f"{pnl_sign}{daily_pct:.2f}% {t('dashboard.vs_close')}",
                             pnl_color), unsafe_allow_html=True)
    with k3:
        bc = POSITIVE if best['Change %'] >= 0 else NEGATIVE
        st.markdown(kpi_card(t("dashboard.best"), best['Ticker'],
                             f"{best['Change %']:+.2f}%  ·  {best_sym}{best['Price']:,.2f}", bc),
                    unsafe_allow_html=True)
    with k4:
        wc = NEGATIVE if worst['Change %'] <= 0 else POSITIVE
        st.markdown(kpi_card(t("dashboard.worst"), worst['Ticker'],
                             f"{worst['Change %']:+.2f}%  ·  {worst_sym}{worst['Price']:,.2f}", wc),
                    unsafe_allow_html=True)

    # ── Badge multi-divisa ────────────────────────────────────────────────────
    if is_multiccy:
        from modules.fx import get_fx_rate as _fx
        fx_parts = []
        for ccy in unique_ccys:
            if ccy != base_ccy:
                r = _fx(ccy, base_ccy)
                fx_parts.append(f"<b>{ccy}/{base_ccy}</b> {r:.4f}")
        if fx_parts:
            st.markdown(
                f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-bottom:12px; "
                f"padding:6px 12px; background:{SURFACE}; border-radius:6px; "
                f"border:1px solid {BORDER}; display:inline-block;'>"
                f"💱 Tipos de cambio: {'  ·  '.join(fx_parts)}"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── P&L desde transacciones ──────────────────────────────────────────────
    pl_data = None
    try:
        pl_raw = fetch_pl_summary_cached(st.session_state.username)
        if pl_raw is not None and not pl_raw.empty:
            pl_data = pl_raw
    except Exception:
        pass

    if pl_data is not None and not pl_data.empty:
        # Compute portfolio-level aggregates
        total_unreal_wac = pl_data["unrealized_pnl_wac"].sum() if "unrealized_pnl_wac" in pl_data.columns else 0.0
        total_unreal_fifo = pl_data["unrealized_pnl_fifo"].sum() if "unrealized_pnl_fifo" in pl_data.columns else 0.0
        total_realized = pl_data["realized_pnl_wac"].sum() if "realized_pnl_wac" in pl_data.columns else 0.0
        total_invested = pl_data["cost_basis_wac"].sum() if "cost_basis_wac" in pl_data.columns else 0.0

        unreal = total_unreal_wac
        unreal_pct = (unreal / total_invested * 100) if total_invested > 0 else 0.0
        total_pl = unreal + total_realized

        uc = POSITIVE if unreal >= 0 else NEGATIVE
        us = "+" if unreal >= 0 else "-"
        rc = POSITIVE if total_realized >= 0 else NEGATIVE
        rs = "+" if total_realized >= 0 else "-"
        tc = POSITIVE if total_pl >= 0 else NEGATIVE

        st.markdown(
            f"<div style='display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap;'>"
            f"<div style='flex:1; min-width:160px; background:{SURFACE}; border:1px solid {BORDER}; "
            f"border-radius:8px; padding:16px 20px;'>"
            f"<div style='font-size:9px; font-weight:700; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1.2px; margin-bottom:6px;'>P&L No Realizado</div>"
            f"<div style='font-size:22px; font-weight:700; color:{uc}; font-family:Georgia,serif;'>"
            f"{us}${abs(unreal):,.0f}</div>"
            f"<div style='font-size:11px; color:{uc}; margin-top:3px;'>{us}{abs(unreal_pct):.2f}% s/coste</div>"
            f"</div>"
            f"<div style='flex:1; min-width:160px; background:{SURFACE}; border:1px solid {BORDER}; "
            f"border-radius:8px; padding:16px 20px;'>"
            f"<div style='font-size:9px; font-weight:700; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1.2px; margin-bottom:6px;'>P&L Realizado</div>"
            f"<div style='font-size:22px; font-weight:700; color:{rc}; font-family:Georgia,serif;'>"
            f"{rs}${abs(total_realized):,.0f}</div>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; margin-top:3px;'>operaciones cerradas</div>"
            f"</div>"
            f"<div style='flex:1; min-width:160px; background:{SURFACE}; border:1px solid {BORDER}; "
            f"border-left:3px solid {GOLD}; border-radius:8px; padding:16px 20px;'>"
            f"<div style='font-size:9px; font-weight:700; color:{TEXT_MUTED}; text-transform:uppercase; "
            f"letter-spacing:1.2px; margin-bottom:6px;'>P&L Total</div>"
            f"<div style='font-size:22px; font-weight:700; color:{tc}; font-family:Georgia,serif;'>"
            f"{'+'if total_pl>=0 else '-'}${abs(total_pl):,.0f}</div>"
            f"<div style='font-size:11px; color:{TEXT_MUTED}; margin-top:3px;'>no real. + realizado</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    gold_divider()

    # ── Gráficos ──────────────────────────────────────────────────────────────
    col_pie, col_perf = st.columns([1, 2])

    with col_pie:
        section_label(t("dashboard.allocation"))
        palette = [
            GOLD, '#5a8f6e', '#4a6fa5', '#8b6fa5',
            '#c05a5a', '#5a8b8b', '#b07d3a', '#6b5a8b',
            '#8b7a5a', '#5a7a8b',
        ]
        n = len(tickers)
        # Pie usa Value_Base (en divisa base) para que los pesos sean correctos en portfolios mixtos
        _pie_values = merged['Value_Base'] if 'Value_Base' in merged.columns else merged['Value']
        fig_pie = go.Figure(go.Pie(
            labels=merged['Ticker'],
            values=_pie_values,
            hole=0.54,
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(size=9, color='white', family='DM Mono, monospace'),
            insidetextorientation='horizontal',
            marker=dict(
                colors=palette[:n],
                line=dict(color='#0e1117', width=2),
            ),
            hovertemplate=(
                '<b>%{label}</b><br>'
                f'{base_sym}' + '%{value:,.0f}<br>'
                '%{percent}<br>'
                '<extra></extra>'
            ),
            direction='clockwise',
            sort=True,
        ))
        # Anotación central
        aum_fmt = f"{base_sym}{total_aum/1e6:.1f}M" if total_aum >= 1e6 else f"{base_sym}{total_aum/1e3:.0f}K"
        fig_pie.add_annotation(
            text=f'<b>{aum_fmt}</b>',
            x=0.5, y=0.54, showarrow=False,
            font=dict(size=17, color=TEXT_PRIMARY, family='DM Mono, monospace'),
        )
        fig_pie.add_annotation(
            text='AUM',
            x=0.5, y=0.41, showarrow=False,
            font=dict(size=9, color=TEXT_MUTED, family='Inter'),
        )
        layout_pie = {**PLOTLY_DARK}
        # Override margin en el dict ANTES de hacer ** spread para evitar kwarg duplicado
        layout_pie['margin'] = dict(l=10, r=120, t=10, b=10)
        layout_pie['legend'] = dict(
            orientation='v',
            x=1.03, y=0.5,
            xanchor='left', yanchor='middle',
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT_SECONDARY, size=10, family='Inter'),
            itemsizing='constant',
        )
        fig_pie.update_layout(
            **layout_pie,
            height=300,
            showlegend=True,
            uniformtext=dict(minsize=7, mode='hide'),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_perf:
        section_label(t("dashboard.performance"))
        try:
            # Period selector
            period_opts = {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "3A": 1095}
            sel_period = st.radio("", list(period_opts.keys()), index=3,
                                  horizontal=True, key="dash_period",
                                  label_visibility="collapsed")
            days_back = period_opts[sel_period]

            all_fetch = list(dict.fromkeys(tickers + ["SPY"]))  # dedup preservando orden
            hist = fetch_history(tuple(all_fetch), days=days_back).ffill()
            hist = hist.dropna(how="all").ffill().bfill()
            if not hist.empty:
                available = [t for t in tickers if t in hist.columns]
                # groupby evita duplicados en el índice (merge puede crear filas repetidas)
                weights = (merged.groupby('Ticker')['Value'].sum() / total_aum)
                w = np.array([float(weights.get(t, 0)) for t in available])
                w = w / w.sum() if w.sum() > 0 else w
                port_ret = hist[available].pct_change().dropna().dot(w)
                cum = (1 + port_ret).cumprod() * 100
                final_val  = cum.iloc[-1]
                line_color = POSITIVE if final_val >= 100 else NEGATIVE
                fill_color = POSITIVE_BG if final_val >= 100 else NEGATIVE_BG

                fig = go.Figure()
                # Portfolio line
                fig.add_trace(go.Scatter(
                    x=cum.index, y=cum.values, mode='lines',
                    name="Portfolio",
                    fill='tozeroy', fillcolor=fill_color,
                    line=dict(color=line_color, width=2),
                    hovertemplate='%{x|%d %b %Y}<br><b>Portfolio: %{y:.1f}</b><extra></extra>'
                ))
                # SPY benchmark
                if "SPY" in hist.columns:
                    spy_cum = (1 + hist["SPY"].pct_change().dropna()).cumprod() * 100
                    spy_final = spy_cum.iloc[-1]
                    fig.add_trace(go.Scatter(
                        x=spy_cum.index, y=spy_cum.values, mode='lines',
                        name="SPY",
                        line=dict(color=GOLD_DIM, width=1.5, dash='dot'),
                        hovertemplate='%{x|%d %b %Y}<br>SPY: %{y:.1f}<extra></extra>'
                    ))
                    # Outperformance annotation
                    outperf = final_val - spy_final
                    outperf_color = POSITIVE if outperf >= 0 else NEGATIVE
                    outperf_sign  = "+" if outperf >= 0 else ""
                    fig.add_annotation(
                        x=0.98, y=0.92, xref="paper", yref="paper",
                        text=f"vs SPY: <b>{outperf_sign}{outperf:.1f}pp</b>",
                        showarrow=False, align="right",
                        font=dict(size=11, color=outperf_color),
                        bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                        borderpad=4,
                    )

                fig.add_hline(y=100, line_dash="dot", line_color=BORDER, line_width=1)
                fig.update_layout(**plotly_layout(
                    height=280,
                    legend=dict(orientation="h", y=1.08, x=0,
                                font=dict(size=10, color=TEXT_MUTED),
                                bgcolor="rgba(0,0,0,0)"),
                ))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo cargar el histórico: {e}")

    gold_divider()

    # ── Sector breakdown ──────────────────────────────────────────────────────
    try:
        sector_map = fetch_sector_map(tuple(tickers))
        if sector_map:
            sector_df = merged[["Ticker", "Value"]].copy()
            sector_df["Sector"] = sector_df["Ticker"].map(sector_map).fillna("Other")
            sector_agg = sector_df.groupby("Sector")["Value"].sum().reset_index()
            sector_agg = sector_agg.sort_values("Value", ascending=False)

            if not sector_agg.empty:
                sector_colors = [
                    GOLD, '#5a8f6e', '#4a6fa5', '#8b6fa5',
                    '#c05a5a', '#5a8b8b', '#b07d3a', '#6b5a8b',
                    '#8b7a5a', '#5a7a8b', '#7a8b5a',
                ]
                col_sec1, col_sec2 = st.columns([1, 2])
                with col_sec1:
                    section_label("Sectores")
                    fig_sec = go.Figure(go.Pie(
                        labels=sector_agg["Sector"],
                        values=sector_agg["Value"],
                        hole=0.45,
                        textinfo='label+percent',
                        textfont=dict(size=9, color='white'),
                        marker=dict(
                            colors=sector_colors[:len(sector_agg)],
                            line=dict(color='#0e1117', width=2),
                        ),
                        hovertemplate='<b>%{label}</b><br>$%{value:,.0f} · %{percent}<extra></extra>',
                        sort=True,
                    ))
                    fig_sec.update_layout(**plotly_layout(height=260,
                                         margin=dict(l=0, r=0, t=0, b=0),
                                         showlegend=False))
                    st.plotly_chart(fig_sec, use_container_width=True)

                with col_sec2:
                    section_label("Distribución por sector")
                    for _, sr in sector_agg.iterrows():
                        pct_s = sr["Value"] / total_aum * 100 if total_aum > 0 else 0
                        bar_w = min(int(pct_s * 1.5), 100)
                        ticks_in_sector = sector_df[sector_df["Sector"] == sr["Sector"]]["Ticker"].tolist()
                        tick_str = ", ".join(ticks_in_sector[:6]) + ("…" if len(ticks_in_sector) > 6 else "")
                        st.markdown(
                            f"<div style='margin-bottom:10px;'>"
                            f"<div style='display:flex; justify-content:space-between; "
                            f"margin-bottom:4px; font-size:12px;'>"
                            f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>{sr['Sector']}</span>"
                            f"<span style='color:{GOLD}; font-family:monospace;'>${sr['Value']:,.0f} · {pct_s:.1f}%</span>"
                            f"</div>"
                            f"<div style='background:{BORDER}; border-radius:3px; height:4px; margin-bottom:3px;'>"
                            f"<div style='background:{GOLD}; height:4px; width:{bar_w}%; border-radius:3px;'></div>"
                            f"</div>"
                            f"<div style='font-size:10px; color:{TEXT_MUTED};'>{tick_str}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        gold_divider()
    except Exception:
        gold_divider()

    # ── Tabla de posiciones (div/flex — compatible con Streamlit) ────────────
    col_tbl, col_mv = st.columns([3, 1])

    with col_tbl:
        section_label(t("dashboard.holdings"))

        # Cabecera
        lbl_ticker = t("dashboard.col_ticker")
        lbl_shares = t("dashboard.col_shares")
        lbl_price  = t("dashboard.col_price")
        lbl_change = t("dashboard.col_change")
        lbl_value  = t("dashboard.col_value")
        lbl_weight = t("dashboard.col_weight")

        # Build P&L lookup from transactions (if available)
        pl_lookup = {}
        if pl_data is not None and not pl_data.empty:
            for _, pr in pl_data.iterrows():
                tk = str(pr.get("ticker", "")).upper()
                pl_lookup[tk] = {
                    "avg_cost":      pr.get("avg_cost_wac",        0.0) or 0.0,
                    "unrealized":    pr.get("unrealized_pnl_wac",  0.0) or 0.0,
                    "unrealized_pct":pr.get("unrealized_pct_wac",  0.0) or 0.0,
                }
        has_pl = bool(pl_lookup)

        if has_pl:
            col_styles = [
                "flex:1.8; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px;",
                "flex:0.8; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:0.9; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:0.9; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.0; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.1; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.1; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.0; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
            ]
            col_styles = [s.replace('{m}', TEXT_MUTED) for s in col_styles]
            labels = [lbl_ticker, lbl_shares, lbl_price, "Coste Medio", lbl_change, "P&L Unreal.", "P&L %", lbl_weight]
        else:
            col_styles = [
                "flex:1.8; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px;",
                "flex:1; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.2; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.2; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
                "flex:1.4; padding:8px 14px; font-size:9px; font-weight:700; color:{m}; text-transform:uppercase; letter-spacing:1.2px; text-align:right;",
            ]
            col_styles = [s.replace('{m}', TEXT_MUTED) for s in col_styles]
            labels = [lbl_ticker, lbl_shares, lbl_price, lbl_change, lbl_value, lbl_weight]

        header_style = (
            f"display:flex; align-items:center; "
            f"background:{SURFACE}; border:1px solid {BORDER}; "
            f"border-radius:8px 8px 0 0; gap:0;"
        )

        header_divs = "".join(
            f"<div style='{col_styles[i]}'>{labels[i]}</div>"
            for i in range(len(labels))
        )
        rows_html = (
            f"<div style='{header_style}'>{header_divs}</div>"
        )

        for _, r in merged.sort_values('Value', ascending=False).iterrows():
            try:
                tick_str = str(r['Ticker'])
                peso  = float(r['Value']) / total_aum * 100 if total_aum > 0 else 0.0
                cc    = POSITIVE if float(r['Change %']) >= 0 else NEGATIVE
                cc_bg = POSITIVE_BG if float(r['Change %']) >= 0 else NEGATIVE_BG
                arrow = "▲" if float(r['Change %']) >= 0 else "▼"
                bar   = min(int(peso * 2), 100)

                raw_name = r.get('Name', '') if 'Name' in r else ''
                if pd.notna(raw_name) and str(raw_name).strip() and str(raw_name) != tick_str:
                    sub_label = (
                        f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:1px; "
                        f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>"
                        f"{html_module.escape(str(raw_name)[:32])}</div>"
                    )
                else:
                    sub_label = ""

                ticker_cell = (
                    f"<div style='flex:1.8; padding:10px 14px 10px 14px; min-width:0;'>"
                    f"<div style='font-weight:600; color:{TEXT_PRIMARY}; font-size:13px; "
                    f"letter-spacing:0.3px;'>{html_module.escape(tick_str)}</div>"
                    f"{sub_label}</div>"
                )

                if has_pl and tick_str.upper() in pl_lookup:
                    pl_row = pl_lookup[tick_str.upper()]
                    avg_c = pl_row["avg_cost"]
                    unreal = pl_row["unrealized"]
                    unreal_pct = pl_row["unrealized_pct"]
                    pc = POSITIVE if unreal >= 0 else NEGATIVE
                    ps = "+" if unreal >= 0 else ""

                    shares_cell = (
                        f"<div style='flex:0.8; padding:10px 14px; color:{TEXT_MUTED}; "
                        f"font-family:monospace; font-size:12px; text-align:right; align-self:center;'>"
                        f"{float(r['Shares']):,.2f}</div>"
                    )
                    price_cell = (
                        f"<div style='flex:0.9; padding:10px 14px; color:{TEXT_SECONDARY}; "
                        f"font-family:monospace; font-size:13px; text-align:right; align-self:center;'>"
                        f"${float(r['Price']):,.2f}</div>"
                    )
                    cost_cell = (
                        f"<div style='flex:0.9; padding:10px 14px; color:{TEXT_MUTED}; "
                        f"font-family:monospace; font-size:12px; text-align:right; align-self:center;'>"
                        f"${avg_c:,.2f}</div>"
                    )
                    chg_cell = (
                        f"<div style='flex:1.0; padding:10px 14px; text-align:right; align-self:center;'>"
                        f"<span style='color:{cc}; background:{cc_bg}; font-family:monospace; font-size:11px; "
                        f"font-weight:500; padding:3px 8px; border-radius:4px;'>"
                        f"{arrow} {abs(float(r['Change %'])):.2f}%</span></div>"
                    )
                    pl_cell = (
                        f"<div style='flex:1.1; padding:10px 14px; color:{pc}; "
                        f"font-family:monospace; font-size:13px; font-weight:600; text-align:right; align-self:center;'>"
                        f"{ps}${abs(unreal):,.0f}</div>"
                    )
                    pl_pct_cell = (
                        f"<div style='flex:1.1; padding:10px 14px; text-align:right; align-self:center;'>"
                        f"<span style='color:{pc}; background:{'#1a3a1a' if unreal>=0 else '#3a1a1a'}; "
                        f"font-family:monospace; font-size:11px; font-weight:500; padding:3px 8px; border-radius:4px;'>"
                        f"{ps}{abs(unreal_pct):.2f}%</span></div>"
                    )
                    bar_cell = (
                        f"<div style='flex:1.0; padding:10px 14px; text-align:right; align-self:center;'>"
                        f"<div style='display:inline-flex; align-items:center; gap:6px; justify-content:flex-end;'>"
                        f"<div style='background:{BORDER}; border-radius:2px; width:50px; height:2px;'>"
                        f"<div style='background:{GOLD}; height:2px; width:{bar}%; border-radius:2px;'></div></div>"
                        f"<span style='color:{TEXT_MUTED}; font-family:monospace; font-size:11px;'>{peso:.1f}%</span>"
                        f"</div></div>"
                    )
                    row_style = f"display:flex; align-items:stretch; border-bottom:1px solid {BORDER_SOFT}; background:{BG};"
                    rows_html += (
                        f"<div style='{row_style}'>"
                        f"{ticker_cell}{shares_cell}{price_cell}{cost_cell}{chg_cell}{pl_cell}{pl_pct_cell}{bar_cell}"
                        f"</div>"
                    )
                else:
                    shares_cell = (
                        f"<div style='flex:1; padding:10px 14px; color:{TEXT_MUTED}; "
                        f"font-family:monospace; font-variant-numeric:tabular-nums; font-size:12px; text-align:right; align-self:center;'>"
                        f"{float(r['Shares']):,.2f}</div>"
                    )
                    price_cell = (
                        f"<div style='flex:1; padding:10px 14px; color:{TEXT_SECONDARY}; "
                        f"font-family:monospace; font-variant-numeric:tabular-nums; font-size:13px; text-align:right; align-self:center;'>"
                        f"${float(r['Price']):,.2f}</div>"
                    )
                    chg_cell = (
                        f"<div style='flex:1.2; padding:10px 14px; text-align:right; align-self:center;'>"
                        f"<span style='color:{cc}; background:{cc_bg}; font-family:monospace; font-variant-numeric:tabular-nums; font-size:11px; "
                        f"font-weight:500; padding:3px 9px; border-radius:4px;'>"
                        f"{arrow} {abs(float(r['Change %'])):.2f}%</span></div>"
                    )
                    val_cell = (
                        f"<div style='flex:1.2; padding:10px 14px; color:{TEXT_PRIMARY}; "
                        f"font-family:monospace; font-variant-numeric:tabular-nums; font-size:14px; font-weight:500; text-align:right; align-self:center;'>"
                        f"${float(r['Value']):,.0f}</div>"
                    )
                    bar_cell = (
                        f"<div style='flex:1.4; padding:10px 14px; text-align:right; align-self:center;'>"
                        f"<div style='display:inline-flex; align-items:center; gap:8px; justify-content:flex-end;'>"
                        f"<div style='background:{BORDER}; border-radius:2px; width:60px; height:2px; flex-shrink:0;'>"
                        f"<div style='background:{GOLD}; height:2px; width:{bar}%; border-radius:2px;'></div></div>"
                        f"<span style='color:{TEXT_MUTED}; font-family:monospace; font-variant-numeric:tabular-nums; font-size:11px; min-width:30px;'>{peso:.1f}%</span>"
                        f"</div></div>"
                    )
                    row_style = (
                        f"display:flex; align-items:stretch; "
                        f"border-bottom:1px solid {BORDER_SOFT}; "
                        f"background:{BG};"
                    )
                    rows_html += (
                        f"<div style='{row_style}'>"
                        f"{ticker_cell}{shares_cell}{price_cell}{chg_cell}{val_cell}{bar_cell}"
                        f"</div>"
                    )
            except Exception:
                pass

        # Footer con total
        footer_style = (
            f"display:flex; align-items:center; padding:10px 14px; "
            f"background:{SURFACE}; border-top:1px solid {BORDER}; "
            f"border-radius:0 0 8px 8px;"
        )
        total_label = t("dashboard.total")
        if has_pl:
            total_unreal_disp = pl_data["unrealized_pnl_wac"].sum() if pl_data is not None and "unrealized_pnl_wac" in pl_data.columns else 0
            tuc = POSITIVE if total_unreal_disp >= 0 else NEGATIVE
            rows_html += (
                f"<div style='{footer_style}'>"
                f"<div style='flex:1.8; font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
                f"text-transform:uppercase; letter-spacing:1px;'>{total_label}</div>"
                f"<div style='flex:0.8;'></div>"
                f"<div style='flex:0.9; color:{GOLD}; font-size:14px; font-weight:700; "
                f"text-align:right; font-family:Georgia,serif; padding:0 14px;'>${total_aum:,.0f}</div>"
                f"<div style='flex:0.9;'></div>"
                f"<div style='flex:1.0;'></div>"
                f"<div style='flex:1.1; color:{tuc}; font-size:13px; font-weight:700; "
                f"text-align:right; padding:0 14px;'>{'+'if total_unreal_disp>=0 else '-'}${abs(total_unreal_disp):,.0f}</div>"
                f"<div style='flex:1.1;'></div>"
                f"text-align:right; padding-right:14px;'>100%</div>"
                f"</div>"
            )
        else:
            rows_html += (
                f"<div style='{footer_style}'>"
                f"<div style='flex:1.8; font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
                f"text-transform:uppercase; letter-spacing:1px;'>{total_label}</div>"
                f"<div style='flex:1;'></div>"
                f"<div style='flex:1;'></div>"
                f"<div style='flex:1.2;'></div>"
                f"<div style='flex:1.2; color:{GOLD}; font-size:14px; font-weight:700; "
                f"text-align:right; font-family:Georgia,serif;'>${total_aum:,.0f}</div>"
                f"<div style='flex:1.4; color:{TEXT_MUTED}; font-size:11px; "
                f"text-align:right; padding-right:14px;'>100%</div>"
                f"</div>"
            )

        wrapper = (
            f"<div style='border:1px solid {BORDER}; border-radius:8px; "
            f"overflow:hidden; font-family:Inter,sans-serif;'>{rows_html}</div>"
        )
        st.markdown(wrapper, unsafe_allow_html=True)

    # ── Movers ────────────────────────────────────────────────────────────────────────────
    with col_mv:
        section_label(t("dashboard.movers"))
        cols_mv = ['Ticker', 'Change %', 'Price'] + (['Name'] if 'Name' in merged.columns else [])
        movers = merged[cols_mv].sort_values('Change %', ascending=False)
        movers_html = ""
        for _, r in movers.iterrows():
            c    = POSITIVE if r['Change %'] >= 0 else NEGATIVE
            bg   = POSITIVE_BG if r['Change %'] >= 0 else NEGATIVE_BG
            bd   = f"{c}44"
            ar   = "▲" if r['Change %'] >= 0 else "▼"
            name_mv = html_module.escape(str(r['Name'])) if 'Name' in r and pd.notna(r.get('Name')) and str(r.get('Name','')) != r['Ticker'] else str(r['Price'])
            movers_html += (
                f"<div style='display:flex; justify-content:space-between; align-items:center; "
                f"padding:12px 14px; background:{bg}; border:1px solid {bd}; "
                f"border-radius:6px; margin-bottom:6px;'>"
                f"<div><div style='font-weight:600; font-size:13px; color:{TEXT_PRIMARY}; "
                f"letter-spacing:0.3px;'>{r['Ticker']}</div>"
                f"<div style='font-size:10px; color:{TEXT_MUTED}; margin-top:1px;'>{name_mv}</div></div>"
                f"<div style='font-weight:500; font-size:13px; color:{c};'>{ar} {abs(r['Change %']):.2f}%</div>"
                f"</div>"
            )
        st.markdown(movers_html, unsafe_allow_html=True)

    try:
        import modules.polygon_client as _pc2
        price_src_note = "Polygon.io" if _pc2.api_key_set() else "yfinance"
    except Exception:
        price_src_note = "yfinance"
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:10px; text-align:center; "
        f"margin-top:24px; letter-spacing:0.5px; text-transform:uppercase;'>"
        f"Precios vía {price_src_note} · {datetime.now().strftime('%H:%M:%S')}</p>",
        unsafe_allow_html=True)

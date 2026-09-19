import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import modules.auth as auth
from modules.i18n import t, get_lang
from modules.styles import GOLD, GOLD_LIGHT, SURFACE, SURFACE_2, BORDER, BORDER_SOFT, \
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, POSITIVE, NEGATIVE, PLOTLY_DARK
from modules.utils import resolve_isin_name
import modules.portfolio_risk as portfolio_risk

# ── Tipos de activo ───────────────────────────────────────────────────────────
ASSET_TYPES = {
    "equity":       {"es": "Renta Variable / ETF",   "en": "Equity / ETF"},
    "bond_etf":     {"es": "ETF de Renta Fija",       "en": "Fixed Income ETF"},
    "bond":         {"es": "Bono Individual",          "en": "Individual Bond"},
    "fund":         {"es": "Fondo de Inversión",       "en": "Investment Fund"},
    "option_call":  {"es": "Opción Call",              "en": "Call Option"},
    "option_put":   {"es": "Opción Put",               "en": "Put Option"},
    "future":       {"es": "Contrato de Futuros",      "en": "Futures Contract"},
    "warrant":      {"es": "Warrant",                  "en": "Warrant"},
}

DERIV_TYPES = ('option_call', 'option_put', 'future', 'warrant')

def asset_label(key: str) -> str:
    lang = get_lang()
    return ASSET_TYPES.get(key, {}).get(lang, key)

def is_es() -> bool:
    return get_lang() == "es"


def get_price(ticker: str, asset_type: str, face_value=None) -> float:
    """Obtiene precio de mercado del subyacente para cualquier tipo de activo.
    Usa Polygon.io si hay API key configurada; price_cache (batch yfinance) como fallback.
    """
    if asset_type == 'bond' and face_value:
        return float(face_value)

    # ── Polygon primero ───────────────────────────────────────────────────────
    try:
        import modules.polygon_client as pc
        if pc.api_key_set():
            snap = pc.get_snapshot(ticker.strip().upper())
            if snap and snap.get("currentPrice"):
                return round(float(snap["currentPrice"]), 4)
    except Exception:
        pass

    # ── price_cache (batch-aware) ─────────────────────────────────────────────
    from modules.price_cache import get_price as _cached_price
    p = _cached_price(ticker.strip().upper())
    return round(p, 4) if p and p > 0 else 0.0


def _section(title: str):
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:16px 0 10px 0;'>{title}</p>",
        unsafe_allow_html=True
    )


def _init_portfolio():
    """Inicializa el DataFrame del portfolio con todas las columnas necesarias."""
    required_cols = [
        'Ticker', 'Name', 'Asset Type', 'Shares',
        'Face Value', 'Coupon %', 'Maturity',
        'Strike', 'Multiplier', 'Option Type', 'Premium',
        'Avg Cost', 'Currency',
    ]
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=required_cols)
    else:
        # Migrar DataFrames antiguos que no tengan las nuevas columnas
        for col in required_cols:
            if col not in st.session_state.portfolio.columns:
                st.session_state.portfolio[col] = None


def _deriv_label(asset_type: str) -> str:
    icons = {
        'option_call': '📈 Call',
        'option_put':  '📉 Put',
        'future':      '⚡ Futuro',
        'warrant':     '🎫 Warrant',
    }
    return icons.get(asset_type, asset_type)


# ── Add helpers ──────────────────────────────────────────────────────────────

def _fetch_company_name(ticker: str) -> str:
    """Fetch company name via data_provider (Finviz primary, yfinance fallback)."""
    try:
        from modules.data_provider import get_fundamentals
        info = get_fundamentals(ticker)
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker


def _add_asset(ticker: str, shares: float, asset_type: str, avg_cost: float | None = None,
               currency: str | None = None):
    """Add an equity / bond-ETF row to the portfolio, auto-fetching the company name and currency."""
    _init_portfolio()
    name = _fetch_company_name(ticker)

    # Auto-detect currency if not provided
    if not currency:
        try:
            from modules.fx import get_ticker_currency
            currency = get_ticker_currency(ticker)
        except Exception:
            currency = "USD"

    new_row = pd.DataFrame([{
        'Ticker':     ticker,
        'Name':       name,
        'Asset Type': asset_type,
        'Shares':     shares,
        'Face Value': None,
        'Coupon %':   None,
        'Maturity':   None,
        'Strike':     None,
        'Multiplier': None,
        'Option Type': None,
        'Premium':    None,
        'Avg Cost':   avg_cost,
        'Currency':   currency,
    }])
    st.session_state.portfolio = pd.concat(
        [st.session_state.portfolio, new_row], ignore_index=True
    )
    st.session_state.has_portfolio = True
    auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
    st.success(f"✅ {name} ({ticker}) añadido al portfolio.")


def _add_bond(ticker: str, name: str, quantity: float, nominal: float,
              coupon: float, maturity: str, frequency: str):
    """Add a direct bond row."""
    _init_portfolio()
    display_name = name.strip() if name and name.strip() else ticker
    if not display_name or display_name == ticker:
        display_name = resolve_isin_name(ticker)
    new_row = pd.DataFrame([{
        'Ticker':     ticker,
        'Name':       display_name,
        'Asset Type': 'bond',
        'Shares':     quantity,
        'Face Value': nominal,
        'Coupon %':   coupon,
        'Maturity':   maturity,
        'Strike':     None,
        'Multiplier': None,
        'Option Type': None,
        'Premium':    None,
    }])
    st.session_state.portfolio = pd.concat(
        [st.session_state.portfolio, new_row], ignore_index=True
    )
    st.session_state.has_portfolio = True
    auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
    st.success(f"✅ Bono {display_name} añadido.")


def _add_fund(ticker: str, fund_name: str, units: float, manual_nav=None):
    """Add a mutual fund row, auto-fetching name if none provided."""
    _init_portfolio()
    if fund_name and fund_name.strip():
        name = fund_name.strip()
    else:
        name = _fetch_company_name(ticker)
    new_row = pd.DataFrame([{
        'Ticker':     ticker,
        'Name':       name,
        'Asset Type': 'fund',
        'Shares':     units,
        'Face Value': manual_nav,
        'Coupon %':   None,
        'Maturity':   None,
        'Strike':     None,
        'Multiplier': None,
        'Option Type': None,
        'Premium':    None,
    }])
    st.session_state.portfolio = pd.concat(
        [st.session_state.portfolio, new_row], ignore_index=True
    )
    st.session_state.has_portfolio = True
    auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
    st.success(f"✅ Fondo {name} ({ticker}) añadido.")


# ── Render principal ──────────────────────────────────────────────────────────

def render_portfolio():
    st.title(t("portfolio.title"))
    _init_portfolio()

    # Resolver automáticamente nombres de fondos/bonos sin nombre
    df = st.session_state.portfolio
    if not df.empty and 'Asset Type' in df.columns and 'Name' in df.columns:
        needs_resolve = df[
            df['Asset Type'].isin(['fund', 'bond']) &
            (df['Name'].isna() | (df['Name'] == df['Ticker']))
        ]
        for idx, row in needs_resolve.iterrows():
            resolved = resolve_isin_name(row['Ticker'])
            if resolved != row['Ticker']:
                st.session_state.portfolio.at[idx, 'Name'] = resolved

    # ── Tabs ─────────────────────────────────────────────────────────────────
    label_pos    = "Posiciones"       if is_es() else "Positions"
    label_eq     = asset_label("equity")
    label_betetf = asset_label("bond_etf")
    label_bond   = asset_label("bond")
    label_fund   = asset_label("fund")
    label_deriv  = "Derivados"        if is_es() else "Derivatives"
    label_risk   = "Riesgo & Sizing"  if is_es() else "Risk & Sizing"

    tab_view, tab_equity, tab_bond_etf, tab_bond, tab_fund, tab_deriv, tab_risk = st.tabs([
        label_pos, label_eq, label_betetf, label_bond, label_fund, label_deriv, label_risk
    ])

    # ── Tab 1: Vista general ──────────────────────────────────────────────────
    with tab_view:
        _render_positions_tab()

    # ── Tab 2: Renta Variable / ETF ───────────────────────────────────────────
    with tab_equity:
        st.caption(
            "Acciones, ETFs y cualquier activo cotizado en bolsa."
            if is_es() else
            "Stocks, ETFs and any listed security."
        )
        # Modo de entrada de coste
        entry_mode_eq = st.radio(
            "Modo de entrada" if is_es() else "Entry mode",
            ["Solo títulos (precio auto)", "Títulos + precio de coste", "Importe total invertido"],
            horizontal=True, key="eq_entry_mode",
            help="'Solo títulos' usa el precio actual de mercado. "
                 "'Precio de coste' guarda tu precio de compra real para P&L preciso. "
                 "'Importe total' calcula los títulos automáticamente."
        )
        with st.form("form_equity", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            with c1:
                ticker = st.text_input("Ticker (ej: AAPL, MSFT, NVDA)")
            with c2:
                if entry_mode_eq == "Importe total invertido":
                    total_invested = st.number_input(
                        "Importe total invertido ($)" if is_es() else "Total invested ($)",
                        min_value=0.01, value=1000.0, step=10.0, format="%.2f",
                        key="eq_total_invested"
                    )
                    shares_eq = None  # se deriva
                else:
                    shares_eq = st.number_input(
                        "Nº de acciones / participaciones" if is_es() else "Number of shares",
                        min_value=0.0001, value=1.0, step=0.01, format="%.4f",
                        key="eq_shares"
                    )
                    total_invested = None
            with c3:
                if entry_mode_eq == "Títulos + precio de coste":
                    avg_cost_eq = st.number_input(
                        "Precio de coste por título ($)" if is_es() else "Cost price per share ($)",
                        min_value=0.0, value=0.0, step=0.01, format="%.4f",
                        key="eq_avg_cost",
                        help="Tu precio real de compra. Se usa para calcular P&L."
                    )
                elif entry_mode_eq == "Importe total invertido":
                    avg_cost_eq = st.number_input(
                        "Precio de coste por título ($)" if is_es() else "Cost price per share ($)",
                        min_value=0.0, value=0.0, step=0.01, format="%.4f",
                        key="eq_avg_cost_total",
                        help="Precio al que compraste cada título. Necesario para derivar el nº de títulos."
                    )
                else:
                    st.markdown(
                        "<small style='color:#8b95a8;'>Precio auto al añadir</small>" if is_es()
                        else "<small style='color:#8b95a8;'>Price fetched automatically</small>",
                        unsafe_allow_html=True
                    )
                    avg_cost_eq = None
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                sub = st.form_submit_button("➕ Añadir", use_container_width=True)

            if sub and ticker:
                from modules.error_handler import validate_ticker, show_validation_errors, validate_positive
                tk = ticker.strip().upper()

                # Validación básica antes de tocar la red
                _skip_shares_val = (entry_mode_eq == "Importe total invertido")
                if not _skip_shares_val:
                    _ok_shares, _err_shares = validate_positive(shares_eq, "El número de títulos")
                    if not _ok_shares:
                        show_validation_errors([_err_shares])
                        shares_eq = None  # fuerza salida
                if _skip_shares_val or shares_eq is not None:
                    if entry_mode_eq == "Importe total invertido":
                        if avg_cost_eq and avg_cost_eq > 0:
                            shares_eq = round(total_invested / avg_cost_eq, 6)
                            _add_asset(tk, shares_eq, 'equity', avg_cost=avg_cost_eq)
                        else:
                            # Derive shares from live price — validar ticker primero
                            with st.spinner(f"Verificando {tk}..."):
                                _tok, _tmsg, live_p = validate_ticker(tk)
                            if not _tok:
                                show_validation_errors([_tmsg])
                            elif live_p and live_p > 0:
                                shares_eq = round(total_invested / live_p, 6)
                                _add_asset(tk, shares_eq, 'equity', avg_cost=live_p)
                            else:
                                st.error("No se pudo obtener el precio actual. Indica el precio de coste manualmente.")
                    elif entry_mode_eq == "Títulos + precio de coste":
                        cost = avg_cost_eq if avg_cost_eq and avg_cost_eq > 0 else None
                        _add_asset(tk, shares_eq, 'equity', avg_cost=cost)
                    else:
                        _add_asset(tk, shares_eq, 'equity')

    # ── Tab 3: ETF Renta Fija ─────────────────────────────────────────────────
    with tab_bond_etf:
        st.caption(
            "ETFs de bonos soberanos, corporativos o de mercados emergentes (TLT, BND, AGG, HYG...)."
            if is_es() else
            "Government, corporate or EM bond ETFs (TLT, BND, AGG, HYG...)."
        )
        entry_mode_etf = st.radio(
            "Modo de entrada" if is_es() else "Entry mode",
            ["Solo títulos (precio auto)", "Títulos + precio de coste", "Importe total invertido"],
            horizontal=True, key="etf_entry_mode"
        )
        with st.form("form_bond_etf", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            with c1:
                ticker = st.text_input("Ticker del ETF (ej: TLT, BND, AGG)")
            with c2:
                if entry_mode_etf == "Importe total invertido":
                    total_inv_etf = st.number_input(
                        "Importe total ($)", min_value=0.01, value=1000.0, step=10.0,
                        format="%.2f", key="etf_total"
                    )
                    shares_etf = None
                else:
                    shares_etf = st.number_input(
                        "Nº de participaciones" if is_es() else "Number of units",
                        min_value=0.0001, value=1.0, step=0.01, format="%.4f", key="etf_shares"
                    )
                    total_inv_etf = None
            with c3:
                if entry_mode_etf in ("Títulos + precio de coste", "Importe total invertido"):
                    avg_cost_etf = st.number_input(
                        "Precio de coste ($)", min_value=0.0, value=0.0,
                        step=0.01, format="%.4f", key="etf_avg_cost"
                    )
                else:
                    st.markdown("<small style='color:#8b95a8;'>Precio auto</small>", unsafe_allow_html=True)
                    avg_cost_etf = None
            with c4:
                st.markdown("<br>", unsafe_allow_html=True)
                sub = st.form_submit_button("➕ Añadir", use_container_width=True)

            if sub and ticker:
                tk = ticker.strip().upper()
                if entry_mode_etf == "Importe total invertido":
                    cost = avg_cost_etf if avg_cost_etf and avg_cost_etf > 0 else None
                    if cost:
                        shares_etf = round(total_inv_etf / cost, 6)
                        _add_asset(tk, shares_etf, 'bond_etf', avg_cost=cost)
                    else:
                        live_p = get_price(tk, 'bond_etf')
                        if live_p and live_p > 0:
                            shares_etf = round(total_inv_etf / live_p, 6)
                            _add_asset(tk, shares_etf, 'bond_etf', avg_cost=live_p)
                        else:
                            st.error(t("portfolio.cost_hint"))
                elif entry_mode_etf == "Títulos + precio de coste":
                    cost = avg_cost_etf if avg_cost_etf and avg_cost_etf > 0 else None
                    _add_asset(tk, shares_etf, 'bond_etf', avg_cost=cost)
                else:
                    _add_asset(tk, shares_etf, 'bond_etf')

    # ── Tab 4: Bono Individual ────────────────────────────────────────────────
    with tab_bond:
        st.caption(
            "Bonos soberanos o corporativos con cupón y fecha de vencimiento conocidos."
            if is_es() else
            "Sovereign or corporate bonds with known coupon and maturity date."
        )
        with st.form("form_bond", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                ticker   = st.text_input("ISIN o nombre identificador (ej: ES0000012445)")
                name     = st.text_input("Nombre del bono (ej: Bono España 2.9% 2026)")
                nominal  = st.number_input("Valor nominal por título ($)", min_value=1.0, value=1000.0, step=100.0)
                quantity = st.number_input("Número de títulos", min_value=1, value=1, step=1)
            with c2:
                coupon   = st.number_input("Cupón anual (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.25)
                maturity = st.date_input("Fecha de vencimiento", value=date(2030, 1, 1), min_value=date.today())
                frequency = st.selectbox(t("portfolio.coupon_freq"),
                                         ["Anual", "Semestral", "Trimestral"])
            sub = st.form_submit_button(t("portfolio.btn_add_bond"), type="primary", use_container_width=True)
            if sub and ticker:
                _add_bond(ticker.strip().upper(), name, quantity, nominal, coupon, str(maturity), frequency)

    # ── Tab 5: Fondo de Inversión ─────────────────────────────────────────────
    with tab_fund:
        st.caption(
            "Fondos de inversión por ticker (VFIAX, FXAIX) o con NAV manual para fondos europeos."
            if is_es() else
            "Mutual funds by ticker (VFIAX, FXAIX) or with manual NAV for European funds."
        )
        with st.form("form_fund", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                ticker     = st.text_input("Ticker o ISIN del fondo")
                fund_name  = st.text_input("Nombre del fondo")
                units      = st.number_input("Número de participaciones", min_value=0.0001, value=1.0, step=0.01)
            with c2:
                manual_nav = st.number_input(
                    "NAV por participación ($) — dejar en 0 para obtener automáticamente",
                    min_value=0.0, value=0.0, step=0.01
                )
            sub = st.form_submit_button(t("portfolio.btn_add_fund"), type="primary", use_container_width=True)
            if sub and ticker:
                _add_fund(ticker.strip().upper(), fund_name, units, manual_nav if manual_nav > 0 else None)

    # ── Tab 6: Derivados ──────────────────────────────────────────────────────
    with tab_deriv:
        _render_derivatives_tab()

    with tab_risk:
        # Build a value-enriched df for portfolio_risk
        st.info("Análisis de riesgo disponible próximamente.")


# ── Portfolio helpers ────────────────────────────────────────────────────────

def _fetch_and_display():
    """Clear price cache and navigate to Dashboard to show updated prices."""
    if 'portfolio_data' in st.session_state:
        del st.session_state['portfolio_data']
    try:
        from modules.dashboard import fetch_dashboard_data, fetch_history
        fetch_dashboard_data.clear()
        fetch_history.clear()
    except Exception:
        pass
    st.rerun()


# ── Tab: Posiciones ───────────────────────────────────────────────────────────

def _render_positions_tab():
    portfolio = st.session_state.portfolio

    if portfolio.empty:
        st.markdown(f"""
        <div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px;
                    padding:40px; text-align:center; margin:20px 0;'>
            <div style='font-size:32px; margin-bottom:12px;'>💼</div>
            <div style='color:{TEXT_PRIMARY}; font-size:14px; font-weight:600; margin-bottom:6px;'>
                {'Portfolio vacío' if is_es() else 'Empty portfolio'}
            </div>
            <div style='color:{TEXT_MUTED}; font-size:12px;'>{t("portfolio.empty")}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = portfolio.copy()

        # Mostrar por clase de activo
        order = ['equity', 'bond_etf', 'bond', 'fund', 'option_call', 'option_put', 'future', 'warrant']
        present = [a for a in order if a in df['Asset Type'].values]

        for atype in present:
            sub = df[df['Asset Type'] == atype].copy()
            icon_map = {
                'equity': '📈', 'bond_etf': '🏦', 'bond': '📄',
                'fund': '🏛️', 'option_call': '🟢', 'option_put': '🔴',
                'future': '⚡', 'warrant': '🎫'
            }
            _section(f"{icon_map.get(atype,'•')} {asset_label(atype)} ({len(sub)})")

            if atype in DERIV_TYPES:
                # Vista enriquecida para derivados
                cols_map = {
                    'Name':        'Nombre' if is_es() else 'Name',
                    'Ticker':      'Subyacente' if is_es() else 'Underlying',
                    'Shares':      'Contratos' if is_es() else 'Contracts',
                    'Strike':      'Strike ($)',
                    'Maturity':    'Vencimiento' if is_es() else 'Expiry',
                    'Multiplier':  'Multiplicador' if is_es() else 'Multiplier',
                    'Premium':     'Prima/cto ($)' if is_es() else 'Premium/ctr ($)',
                }
                display = sub[list(cols_map.keys())].fillna('—').rename(columns=cols_map).astype(str)
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                cols_map = {
                    'Name':       'Nombre' if is_es() else 'Name',
                    'Ticker':     'Ticker',
                    'Shares':     'Títulos' if is_es() else 'Shares',
                    'Currency':   'Divisa' if is_es() else 'Currency',
                    'Face Value': 'V. Nominal' if is_es() else 'Face Value',
                    'Coupon %':   'Cupón (%)' if is_es() else 'Coupon (%)',
                    'Maturity':   'Vencimiento' if is_es() else 'Maturity',
                }
                # Solo incluir columnas que existen en el DF
                available_cols = [c for c in cols_map if c in sub.columns]
                display = sub[available_cols].fillna('—').rename(columns=cols_map).astype(str)
                st.dataframe(display, use_container_width=True, hide_index=True)

        # ── Editar nombres ────────────────────────────────────────────────────
        with st.expander("✏️ " + ("Editar nombres de activos" if is_es() else "Edit asset names")):
            with st.form("edit_portfolio_names", clear_on_submit=False):
                for idx, row in st.session_state.portfolio.iterrows():
                    ticker = row['Ticker']
                    current = str(row.get('Name', ticker)) if pd.notna(row.get('Name')) else ticker
                    st.text_input(
                        f"Nombre para {ticker}",
                        value=current if current != ticker else "",
                        placeholder=ticker,
                        key=f"pname_{idx}_{ticker}"
                    )
                if st.form_submit_button("💾 " + ("Guardar nombres" if is_es() else "Save names"), type="primary"):
                    for idx, row in st.session_state.portfolio.iterrows():
                        ticker = row['Ticker']
                        val = st.session_state.get(f"pname_{idx}_{ticker}", "").strip()
                        st.session_state.portfolio.at[idx, 'Name'] = val if val else ticker
                    st.success("✅ " + ("Nombres actualizados." if is_es() else "Names updated."))
                    st.rerun()

        # ── Eliminar posición ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"<p style='color:#e03131; font-size:12px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>"
            f"🗑️ {'Eliminar posición' if is_es() else 'Remove position'}</p>",
            unsafe_allow_html=True
        )
        pf_all = st.session_state.portfolio
        all_idx = pf_all.index.tolist()
        all_labels = [
            f"{pf_all.at[i, 'Ticker']} — {asset_label(str(pf_all.at[i, 'Asset Type']))} "
            f"({pf_all.at[i, 'Shares']:.2f})"
            for i in all_idx
        ]
        col_sel, col_btn = st.columns([4, 1])
        with col_sel:
            sel_del_label = st.selectbox(
                "Selecciona la posición a eliminar" if is_es() else "Select position to remove",
                all_labels, key="del_pos_select_main", label_visibility="collapsed"
            )
        with col_btn:
            if st.button("🗑️ " + ("Eliminar" if is_es() else "Remove"),
                         key="del_pos_btn_main", type="primary", use_container_width=True):
                sel_pos_idx = all_idx[all_labels.index(sel_del_label)]
                st.session_state.portfolio = (
                    st.session_state.portfolio
                    .drop(index=sel_pos_idx)
                    .reset_index(drop=True)
                )
                st.session_state.has_portfolio = not st.session_state.portfolio.empty
                auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
                st.success("✅ " + ("Posición eliminada." if is_es() else "Position removed."))
                st.rerun()

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(t("portfolio.btn_save"), type="primary", use_container_width=True):
            clean = st.session_state.portfolio.dropna(subset=['Ticker'])
            clean = clean[clean['Ticker'].str.strip() != ""]
            auth.save_portfolio(st.session_state.username, clean)
            if 'portfolio_data' in st.session_state:
                del st.session_state['portfolio_data']
            from modules.dashboard import fetch_dashboard_data, fetch_history
            fetch_dashboard_data.clear()
            fetch_history.clear()
            st.success("✅ " + ("Portfolio guardado." if is_es() else "Portfolio saved."))
            st.rerun()
    with col_b:
        if st.button(t("portfolio.btn_load"), use_container_width=True):
            loaded = auth.load_portfolio(st.session_state.username)
            if not loaded.empty:
                st.session_state.portfolio = loaded
                _init_portfolio()   # asegura columnas nuevas
                if 'portfolio_data' in st.session_state:
                    del st.session_state['portfolio_data']
                st.success("✅ " + ("Portfolio cargado." if is_es() else "Portfolio loaded."))
                st.rerun()
            else:
                st.warning("No hay portfolio guardado." if is_es() else "No saved portfolio found.")

    if not st.session_state.portfolio.empty:
        st.markdown("---")
        if st.button(t("portfolio.btn_fetch"), type="primary", use_container_width=True):
            _fetch_and_display()


# ── Tab: Derivados ────────────────────────────────────────────────────────────

def _render_derivatives_tab():
    lang_es = is_es()

    st.markdown(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px; "
        f"padding:16px 20px; margin-bottom:20px;'>"
        f"<div style='color:{GOLD}; font-size:11px; font-weight:700; text-transform:uppercase; "
        f"letter-spacing:1.2px; margin-bottom:6px;'>"
        "<b>⚡ Derivados Financieros</b>"
        "</div></div>",
        unsafe_allow_html=True
    )

    with st.form("form_option", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_ticker = st.text_input("Ticker subyacente (ej: AAPL)", key="opt_ticker")
            opt_type   = st.selectbox("Tipo", ["option_call", "option_put"],
                                      format_func=_deriv_label, key="opt_type")
        with c2:
            opt_contracts  = st.number_input("Nº contratos", min_value=1, value=1, step=1, key="opt_contracts")
            opt_strike     = st.number_input("Strike ($)", min_value=0.01, value=100.0, step=1.0, key="opt_strike")
        with c3:
            opt_multiplier = st.number_input("Multiplicador", min_value=1, value=100, step=1, key="opt_mult")
            opt_premium    = st.number_input("Prima por contrato ($)", min_value=0.0, value=5.0,
                                             step=0.01, format="%.2f", key="opt_prem")
        with c4:
            opt_expiry = st.date_input("Vencimiento", key="opt_expiry")
            st.markdown("<br>", unsafe_allow_html=True)
            sub_opt = st.form_submit_button("➕ Añadir", use_container_width=True)

        if sub_opt and opt_ticker:
            tk = opt_ticker.strip().upper()
            _init_portfolio()
            name = _fetch_company_name(tk)
            new_row = pd.DataFrame([{
                'Ticker':      tk,
                'Name':        name,
                'Asset Type':  opt_type,
                'Shares':      float(opt_contracts),
                'Face Value':  None,
                'Coupon %':    None,
                'Maturity':    str(opt_expiry),
                'Strike':      opt_strike,
                'Multiplier':  opt_multiplier,
                'Option Type': opt_type,
                'Premium':     opt_premium,
                'Avg Cost':    None,
                'Currency':    'USD',
            }])
            st.session_state.portfolio = pd.concat(
                [st.session_state.portfolio, new_row], ignore_index=True)
            st.session_state.has_portfolio = True
            auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
            st.success(f"✅ {_deriv_label(opt_type)} sobre {tk} añadida.")

    st.markdown("**Futuros**")
    with st.form("form_future", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            fut_ticker     = st.text_input("Ticker del futuro (ej: ES=F, CL=F)", key="fut_ticker")
            fut_contracts  = st.number_input("Nº contratos", min_value=1, value=1, step=1, key="fut_contracts")
        with c2:
            fut_multiplier = st.number_input("Multiplicador (nº de unidades/contrato)", min_value=1,
                                             value=50, step=1, key="fut_mult")
            fut_expiry     = st.date_input("Vencimiento", key="fut_expiry")
        with c3:
            fut_premium    = st.number_input("Precio de entrada ($)", min_value=0.0, value=0.0,
                                             step=0.01, format="%.2f", key="fut_prem")
            st.markdown("<br>", unsafe_allow_html=True)
            sub_fut = st.form_submit_button(t("portfolio.btn_add_future"), use_container_width=True)

        if sub_fut and fut_ticker:
            tk = fut_ticker.strip().upper()
            _init_portfolio()
            name = _fetch_company_name(tk)
            new_row = pd.DataFrame([{
                'Ticker':      tk,
                'Name':        name,
                'Asset Type':  'future',
                'Shares':      float(fut_contracts),
                'Face Value':  None,
                'Coupon %':    None,
                'Maturity':    str(fut_expiry),
                'Strike':      None,
                'Multiplier':  fut_multiplier,
                'Option Type': None,
                'Premium':     fut_premium,
                'Avg Cost':    None,
                'Currency':    'USD',
            }])
            st.session_state.portfolio = pd.concat(
                [st.session_state.portfolio, new_row], ignore_index=True)
            st.session_state.has_portfolio = True
            auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
            st.success(f"✅ Futuro {tk} añadido.")

    st.markdown("**Warrants**")
    with st.form("form_warrant", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            war_ticker     = st.text_input("Ticker subyacente", key="war_ticker")
            war_contracts  = st.number_input("Nº warrants", min_value=1, value=1, step=1, key="war_contracts")
        with c2:
            war_strike     = st.number_input("Strike ($)", min_value=0.01, value=100.0, step=1.0, key="war_strike")
            war_multiplier = st.number_input("Ratio (acciones/warrant)", min_value=1, value=1, step=1, key="war_mult")
        with c3:
            war_premium    = st.number_input("Prima pagada ($)", min_value=0.0, value=0.0,
                                             step=0.01, format="%.4f", key="war_prem")
            war_expiry     = st.date_input("Vencimiento", key="war_expiry")
            sub_war = st.form_submit_button(t("portfolio.btn_add_warrant"), use_container_width=True)

        if sub_war and war_ticker:
            tk = war_ticker.strip().upper()
            _init_portfolio()
            name = _fetch_company_name(tk)
            new_row = pd.DataFrame([{
                'Ticker':      tk,
                'Name':        name,
                'Asset Type':  'warrant',
                'Shares':      float(war_contracts),
                'Face Value':  None,
                'Coupon %':    None,
                'Maturity':    str(war_expiry),
                'Strike':      war_strike,
                'Multiplier':  war_multiplier,
                'Option Type': None,
                'Premium':     war_premium,
                'Avg Cost':    None,
                'Currency':    'USD',
            }])
            st.session_state.portfolio = pd.concat(
                [st.session_state.portfolio, new_row], ignore_index=True)
            st.session_state.has_portfolio = True
            auth.save_portfolio(st.session_state.username, st.session_state.portfolio)
            st.success(f"✅ Warrant sobre {tk} añadido.")

    # ── Posiciones de derivados existentes ────────────────────────────────────
    pf = st.session_state.portfolio
    deriv_mask = pf['Asset Type'].isin(DERIV_TYPES) if not pf.empty else pd.Series([], dtype=bool)
    if not pf.empty and deriv_mask.any():
        st.markdown("---")
        st.markdown("**Posiciones actuales de derivados**")
        sub = pf[deriv_mask].copy()
        cols_show = {
            'Name':       'Nombre',
            'Ticker':     'Subyacente',
            'Asset Type': 'Tipo',
            'Shares':     'Contratos',
            'Strike':     'Strike ($)',
            'Multiplier': 'Multiplicador',
            'Premium':    'Prima ($)',
            'Maturity':   'Vencimiento',
        }
        avail = [c for c in cols_show if c in sub.columns]
        disp = sub[avail].fillna('—').rename(columns=cols_show).astype(str)
        disp['Tipo'] = disp['Tipo'].map(lambda x: _deriv_label(str(x)))
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.caption("💡 " + ("Para eliminar esta posición, ve a la pestaña 'Portfolio Overview'." if lang_es else "To remove this position, go to the 'Portfolio Overview' tab."))
    else:
        st.info(t("portfolio.no_derivatives"))

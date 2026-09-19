"""
modules/transactions.py — Transaction Ledger & P&L Engine
==========================================================
Gestiona el historial de operaciones y calcula P&L con:
  • Coste Medio Ponderado (WAC)
  • FIFO (First In, First Out)

Tipos soportados: buy, sell, dividend, fee, split
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from collections import deque

import modules.auth as auth
import modules.csv_importer as csv_importer
import modules.pdf_importer as pdf_importer
from modules.styles import (
    GOLD, GOLD_LIGHT, GOLD_DIM, GOLD_BORDER,
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    POSITIVE, POSITIVE_BG, NEGATIVE, NEGATIVE_BG,
    section_label, gold_divider,
)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_transaction(username: str, ticker: str, date_str: str, tx_type: str,
                    quantity: float, price: float, commission: float = 0.0,
                    currency: str = "USD", notes: str = "") -> int:
    conn = auth.get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO transactions "
        "(username, ticker, date, type, quantity, price, commission, currency, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (username, ticker.upper(), date_str, tx_type,
         float(quantity), float(price), float(commission), currency, notes or "")
    )
    conn.commit()
    tx_id = c.lastrowid
    conn.close()
    return tx_id


def delete_transaction(tx_id: int) -> None:
    conn = auth.get_conn()
    conn.execute("DELETE FROM transactions WHERE id=?", (tx_id,))
    conn.commit()
    conn.close()


def get_transactions(username: str, ticker: str | None = None) -> pd.DataFrame:
    conn = auth.get_conn()
    if ticker:
        df = pd.read_sql_query(
            "SELECT * FROM transactions WHERE username=? AND ticker=? ORDER BY date ASC, id ASC",
            conn, params=(username, ticker.upper())
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM transactions WHERE username=? ORDER BY date ASC, id ASC",
            conn, params=(username,)
        )
    conn.close()
    # Ensure numeric columns are clean
    for col in ["quantity", "price", "commission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# P&L engine — Weighted Average Cost
# ─────────────────────────────────────────────────────────────────────────────

def calc_pl_wac(df: pd.DataFrame) -> dict:
    result: dict = {}
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values(["date", "id"])
        qty = 0.0
        avg_cost = 0.0
        realized = 0.0
        dividends = 0.0

        for _, row in g.iterrows():
            t    = str(row["type"])
            q    = max(0.0, float(row["quantity"]))
            p    = float(row["price"])
            comm = abs(float(row.get("commission", 0) or 0))

            if t == "buy":
                if q > 0:
                    total_cost = qty * avg_cost + (q * p + comm)
                    qty += q
                    avg_cost = total_cost / qty

            elif t == "sell":
                sell_qty = min(q, qty)
                if sell_qty > 0:
                    realized += (p - avg_cost) * sell_qty - comm
                    qty = max(0.0, qty - sell_qty)
                    if qty < 1e-9:
                        qty = 0.0
                        avg_cost = 0.0

            elif t == "dividend":
                # price = dividend per share; quantity = shares held
                dividends += q * p

            elif t == "split":
                # price = ratio (e.g. 4.0 = 4:1)
                if p > 0:
                    qty *= p
                    avg_cost = avg_cost / p if avg_cost > 0 else 0.0

            elif t == "fee":
                realized -= abs(p)

        result[ticker] = {
            "open_qty":           round(qty, 6),
            "avg_cost":           round(avg_cost, 6),
            "realized_pl":        round(realized, 4),
            "dividends_received": round(dividends, 4),
            "total_cost":         round(qty * avg_cost, 4),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# P&L engine — FIFO
# ─────────────────────────────────────────────────────────────────────────────

def calc_pl_fifo(df: pd.DataFrame) -> dict:
    result: dict = {}
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values(["date", "id"])
        lots: deque = deque()
        realized = 0.0
        dividends = 0.0

        for _, row in g.iterrows():
            t    = str(row["type"])
            q    = max(0.0, float(row["quantity"]))
            p    = float(row["price"])
            comm = abs(float(row.get("commission", 0) or 0))

            if t == "buy":
                if q > 0:
                    cost_per = (q * p + comm) / q
                    lots.append([q, cost_per])

            elif t == "sell":
                remaining = min(q, sum(l[0] for l in lots))
                net_proceeds = remaining * p - comm
                cost_basis = 0.0
                while remaining > 1e-9 and lots:
                    lot_qty, lot_price = lots[0]
                    used = min(lot_qty, remaining)
                    cost_basis += used * lot_price
                    remaining -= used
                    lots[0][0] -= used
                    if lots[0][0] < 1e-9:
                        lots.popleft()
                realized += net_proceeds - cost_basis

            elif t == "dividend":
                dividends += q * p

            elif t == "split":
                if p > 0:
                    new_lots = deque()
                    for lot_qty, lot_price in lots:
                        new_lots.append([lot_qty * p, lot_price / p])
                    lots = new_lots

            elif t == "fee":
                realized -= abs(p)

        open_qty = sum(l[0] for l in lots)
        avg_fifo  = (sum(l[0] * l[1] for l in lots) / open_qty
                     if open_qty > 1e-9 else 0.0)
        result[ticker] = {
            "open_qty":           round(open_qty, 6),
            "avg_cost":           round(avg_fifo, 6),
            "realized_pl":        round(realized, 4),
            "dividends_received": round(dividends, 4),
            "total_cost":         round(open_qty * avg_fifo, 4),
            "fifo_lots":          [(round(l[0], 4), round(l[1], 4)) for l in lots],
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Position summary with live prices
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)  # 1 min — Polygon da precios frescos
def _fetch_prices_for_positions(tickers: tuple) -> dict:
    if not tickers:
        return {}

    # ── Polygon batch (1 sola llamada) ────────────────────────────────────────
    try:
        import modules.polygon_client as pc
        if pc.api_key_set():
            snaps = pc.get_snapshots(tickers)
            prices = {t: float(s["currentPrice"])
                      for t, s in snaps.items()
                      if s.get("currentPrice")}
            # Fill any missing tickers with yfinance
            missing = [t for t in tickers if t not in prices]
            if missing:
                import yfinance as yf
                for t in missing:
                    try:
                        fi = yf.Ticker(t).fast_info
                        prices[t] = float(fi.last_price or 0)
                    except Exception:
                        prices[t] = 0.0
            return prices
    except Exception:
        pass

    # ── yfinance fallback ─────────────────────────────────────────────────────
    prices = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                fi = yf.Ticker(t).fast_info
                prices[t] = float(fi.last_price or 0)
            except Exception:
                prices[t] = 0.0
    except Exception:
        pass
    return prices


def get_position_summary(username: str) -> pd.DataFrame:
    df = get_transactions(username)
    if df.empty:
        return pd.DataFrame()

    wac  = calc_pl_wac(df)
    fifo = calc_pl_fifo(df)

    all_tickers = list(wac.keys())
    open_tickers = tuple(t for t in all_tickers
                         if wac[t]["open_qty"] > 1e-6 or fifo[t]["open_qty"] > 1e-6)
    prices = _fetch_prices_for_positions(open_tickers)

    rows = []
    for t in all_tickers:
        w = wac.get(t, {})
        f = fifo.get(t, {})
        qty   = w.get("open_qty", 0.0)
        price = prices.get(t, 0.0)
        mval  = qty * price

        wac_cost  = w.get("avg_cost", 0.0)
        fifo_cost = f.get("avg_cost", 0.0)
        wac_unr   = (price - wac_cost)  * qty if (qty > 0 and price > 0) else 0.0
        fifo_unr  = (price - fifo_cost) * qty if (qty > 0 and price > 0) else 0.0
        wac_r     = w.get("realized_pl", 0.0)
        fifo_r    = f.get("realized_pl", 0.0)
        divs      = w.get("dividends_received", 0.0)

        rows.append({
            "Ticker":         t,
            "Cant.":          qty,
            "P.Med WAC":      wac_cost,
            "P.Med FIFO":     fifo_cost,
            "Precio Actual":  price,
            "Val. Mercado":   mval,
            "No Real. WAC":   wac_unr,
            "No Real. FIFO":  fifo_unr,
            "Real. WAC":      wac_r,
            "Real. FIFO":     fifo_r,
            "Total WAC":      wac_unr + wac_r + divs,
            "Total FIFO":     fifo_unr + fifo_r + divs,
            "Dividendos":     divs,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (NaN-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_money(v, sign: bool = False) -> str:
    try:
        v = float(v)
        if np.isnan(v):
            return "—"
        prefix = ("+" if v >= 0 else "") if sign else ""
        if abs(v) >= 1e9:
            return f"{prefix}${v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{prefix}${v/1e6:.2f}M"
        return f"{prefix}${v:,.2f}"
    except Exception:
        return "—"


def _fmt_pct(v) -> str:
    try:
        v = float(v)
        return "—" if np.isnan(v) else f"{v:+.2f}%"
    except Exception:
        return "—"


def _color_val(v) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return ""
        return f"color:{POSITIVE};font-weight:600" if f >= 0 else f"color:{NEGATIVE};font-weight:600"
    except Exception:
        return ""


def _color_pct(v) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return ""
        return f"color:{POSITIVE}" if f >= 0 else f"color:{NEGATIVE}"
    except Exception:
        return ""


def _kpi(label: str, value: str, color: str = TEXT_PRIMARY, sub: str = "") -> str:
    sub_html = f"<div style='font-size:10px;color:{TEXT_MUTED};margin-top:2px;'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER_SOFT};"
        f"border-radius:10px;padding:14px 16px;height:100%;'>"
        f"<div style='font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:6px;'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{color};'>{value}</div>"
        f"{sub_html}</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_transactions(username: str):
    from modules.styles import page_header
    page_header("P&L — Operaciones", "Historial de transacciones · P&L realizado y no realizado")

    tab_resumen, tab_historial, tab_nueva, tab_import, tab_pdf = st.tabs([
        "Resumen P&L", "Historial", "Nueva operación", "Importar CSV", "Importar PDF"
    ])

    with tab_nueva:
        _render_add_form(username)

    with tab_historial:
        _render_history(username)

    with tab_resumen:
        _render_summary(username)

    with tab_import:
        csv_importer.render_csv_import(username)

    with tab_pdf:
        pdf_importer.render_pdf_import(username)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Nueva operación
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_LABELS = {
    "buy":      "Compra",
    "sell":     "Venta",
    "dividend": "Dividendo",
    "fee":      "Gasto / Comisión",
    "split":    "Split de acciones",
}

def _render_add_form(username: str):
    section_label("Registrar operación")

    col1, col2 = st.columns(2)

    with col1:
        prefill = st.session_state.pop("tx_prefill_ticker", "")
        ticker = st.text_input(
            "Ticker *", value=prefill,
            placeholder="AAPL, MSFT, BTC-USD...",
            key="tx_ticker",
        ).strip().upper()

        tx_type = st.selectbox(
            "Tipo *",
            options=list(_TYPE_LABELS.keys()),
            format_func=lambda x: _TYPE_LABELS.get(x, x),
            key="tx_type",
        )

        tx_date = st.date_input("Fecha *", value=date.today(), key="tx_date")

    with col2:
        commission = 0.0
        quantity   = 0.0
        price      = 0.0

        if tx_type == "split":
            quantity = st.number_input(
                "Ratio del split * (ej: 4 = cuatro acciones por cada una)",
                min_value=0.01, value=2.0, step=0.5, format="%.2f", key="tx_qty"
            )
            price = quantity  # stored in price field
            st.info(f"Split {quantity:.1f}:1 → cada acción se convierte en {quantity:.1f}. "
                    "El coste medio se ajusta automáticamente.")

        elif tx_type == "dividend":
            price = st.number_input(
                "Dividendo por acción ($) *",
                min_value=0.0, value=0.0, step=0.01, format="%.4f", key="tx_price"
            )
            quantity = st.number_input(
                "Nº de acciones que tenías al cobrar el dividendo *",
                min_value=0.0, value=1.0, step=1.0, format="%.4f", key="tx_qty"
            )
            if price > 0 and quantity > 0:
                st.success(f"Dividendo total: **${price * quantity:,.2f}**")

        elif tx_type == "fee":
            price = st.number_input(
                "Importe del gasto ($) *",
                min_value=0.0, value=0.0, step=0.01, format="%.2f", key="tx_price"
            )
            quantity = 1.0
            st.caption("Se descuenta del P&L realizado.")

        else:  # buy / sell
            quantity = st.number_input(
                "Cantidad (acciones / unidades) *",
                min_value=0.0001, value=1.0, step=1.0, format="%.4f", key="tx_qty"
            )
            price = st.number_input(
                "Precio por unidad ($) *",
                min_value=0.0, value=0.0, step=0.01, format="%.4f", key="tx_price"
            )
            commission = st.number_input(
                "Comisión broker ($)",
                min_value=0.0, value=0.0, step=0.01, format="%.2f", key="tx_comm"
            )

        currency = st.selectbox(
            "Divisa", ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD"],
            key="tx_currency"
        )
        notes = st.text_input(
            "Notas", key="tx_notes",
            placeholder="Opcional — posición inicial, stop loss, etc."
        )

    # Preview card
    if ticker and price > 0 and quantity > 0:
        if tx_type == "buy":
            total = quantity * price + commission
            preview = f"Compra de {quantity:,.4f} {ticker} × ${price:.4f} + ${commission:.2f} comisión = **${total:,.2f}** coste total"
        elif tx_type == "sell":
            total = quantity * price - commission
            preview = f"Venta de {quantity:,.4f} {ticker} × ${price:.4f} − ${commission:.2f} comisión = **${total:,.2f}** neto"
        elif tx_type == "dividend":
            preview = f"Dividendo ${price:.4f}/acc × {quantity:.0f} acc = **${price*quantity:,.2f}** cobrado"
        elif tx_type == "fee":
            preview = f"Gasto de **${price:,.2f}** — reduce P&L realizado"
        else:
            preview = f"Split {price:.1f}:1 sobre {ticker}"
        st.info(preview)

    col_save, col_clear = st.columns([3, 1])
    with col_save:
        if st.button("Guardar operación", type="primary", use_container_width=True, key="tx_save"):
            errors = []
            if not ticker:
                errors.append("Introduce un ticker.")
            if tx_type not in ("split", "fee") and price <= 0:
                errors.append("El precio debe ser mayor que 0.")
            if tx_type not in ("split", "fee") and quantity <= 0:
                errors.append("La cantidad debe ser mayor que 0.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    _qty   = 1.0 if tx_type == "split" else float(quantity)
                    _price = float(quantity) if tx_type == "split" else float(price)
                    add_transaction(
                        username=username,
                        ticker=ticker,
                        date_str=str(tx_date),
                        tx_type=tx_type,
                        quantity=_qty,
                        price=_price,
                        commission=float(commission),
                        currency=currency,
                        notes=notes,
                    )
                    st.success(f"Operación guardada — {_TYPE_LABELS.get(tx_type, tx_type).upper()} {ticker}")
                    # Invalidate position cache
                    _fetch_prices_for_positions.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error al guardar: {exc}")

    with col_clear:
        if st.button("Limpiar", use_container_width=True, key="tx_clear"):
            for k in ["tx_ticker", "tx_type", "tx_date", "tx_qty", "tx_price",
                      "tx_comm", "tx_currency", "tx_notes"]:
                st.session_state.pop(k, None)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Historial
# ─────────────────────────────────────────────────────────────────────────────

def _render_history(username: str):
    section_label("Historial de operaciones")
    df = get_transactions(username)

    if df.empty:
        st.info("Sin operaciones registradas. Ve a 'Nueva operación' para empezar.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        opts_t = ["Todos"] + sorted(df["ticker"].unique().tolist())
        ft = st.selectbox("Ticker", opts_t, key="hf_ticker")
    with col_f2:
        opts_ty = ["Todos"] + sorted(df["type"].unique().tolist())
        fty = st.selectbox("Tipo", opts_ty, key="hf_type")
    with col_f3:
        asc = st.selectbox("Orden", ["Más reciente primero", "Más antiguo primero"], key="hf_sort")

    filtered = df.copy()
    if ft != "Todos":
        filtered = filtered[filtered["ticker"] == ft]
    if fty != "Todos":
        filtered = filtered[filtered["type"] == fty]
    filtered = filtered.sort_values(["date", "id"], ascending=(asc != "Más reciente primero"))

    # Display
    disp = filtered[["id", "date", "ticker", "type", "quantity",
                      "price", "commission", "currency", "notes"]].copy()
    disp.columns = ["ID", "Fecha", "Ticker", "Tipo", "Cantidad",
                    "Precio ($)", "Comisión ($)", "Divisa", "Notas"]
    disp["Tipo"] = disp["Tipo"].map(lambda x: _TYPE_LABELS.get(str(x), str(x)))

    def _color_type(val):
        m = {"Compra": f"color:{POSITIVE};font-weight:600",
             "Venta":  f"color:{NEGATIVE};font-weight:600",
             "Dividendo": f"color:{GOLD};font-weight:600",
             "Gasto / Comisión": f"color:{TEXT_MUTED}",
             "Split de acciones": f"color:{TEXT_SECONDARY}"}
        return m.get(val, "")

    st.dataframe(
        disp.style
            .map(_color_type, subset=["Tipo"])
            .format({
                "Cantidad":     "{:,.4f}",
                "Precio ($)":   "{:,.4f}",
                "Comisión ($)": "{:,.2f}",
            }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"{len(filtered)} operaciones · {filtered['Ticker'].nunique()} tickers")

    # Delete with confirmation
    gold_divider()
    with st.expander("Eliminar operación"):
        st.warning("Esta acción es irreversible.")
        del_id = st.number_input("ID de la operación", min_value=1, step=1, key="del_id")
        # Show the row they're about to delete
        matching = df[df["id"] == int(del_id)]
        if not matching.empty:
            row = matching.iloc[0]
            st.markdown(
                f"**Operación #{int(del_id)}:** "
                f"{_TYPE_LABELS.get(row['type'], row['type'])} "
                f"{row['quantity']:.4f} {row['ticker']} "
                f"@ ${row['price']:.4f} el {row['date']}"
            )
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            confirm = st.checkbox("Confirmo que quiero eliminar esta operación", key="del_confirm")
        with col_d2:
            if st.button("Eliminar", type="secondary", key="del_btn", disabled=not confirm):
                delete_transaction(int(del_id))
                _fetch_prices_for_positions.clear()
                st.success(f"Operación #{int(del_id)} eliminada.")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Resumen P&L
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary(username: str):
    section_label("Resumen P&L")
    df = get_transactions(username)

    if df.empty:
        st.info("Sin operaciones registradas. Empieza en 'Nueva operación'.")
        return

    with st.spinner("Calculando P&L y precios actuales..."):
        summary = get_position_summary(username)

    if summary is None or summary.empty:
        st.warning("No hay posiciones abiertas o no se pudieron obtener precios.")
        return

    # Global KPIs
    total_market   = summary["Val. Mercado"].sum()
    total_wac_unr  = summary["No Real. WAC"].sum()
    total_fifo_unr = summary["No Real. FIFO"].sum()
    total_wac_r    = summary["Real. WAC"].sum()
    total_fifo_r   = summary["Real. FIFO"].sum()
    total_divs     = summary["Dividendos"].sum()
    total_wac      = total_wac_unr  + total_wac_r  + total_divs
    total_fifo     = total_fifo_unr + total_fifo_r + total_divs

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi("Valor Mercado", _fmt_money(total_market)), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi(
            "P&L Total (WAC)", _fmt_money(total_wac, sign=True),
            color=POSITIVE if total_wac >= 0 else NEGATIVE,
            sub=f"Real. {_fmt_money(total_wac_r, sign=True)} · No real. {_fmt_money(total_wac_unr, sign=True)}"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi(
            "P&L Total (FIFO)", _fmt_money(total_fifo, sign=True),
            color=POSITIVE if total_fifo >= 0 else NEGATIVE,
            sub=f"Real. {_fmt_money(total_fifo_r, sign=True)} · No real. {_fmt_money(total_fifo_unr, sign=True)}"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("Dividendos cobrados", _fmt_money(total_divs, sign=True),
                         color=GOLD), unsafe_allow_html=True)

    gold_divider()

    m1, m2 = st.tabs(["Coste Medio (WAC)", "FIFO"])
    with m1:
        _render_method_table(summary, "WAC")
    with m2:
        _render_method_table(summary, "FIFO")


def _render_method_table(summary: pd.DataFrame, method: str):
    import plotly.graph_objects as go
    from modules.styles import PLOTLY_DARK

    unr_col   = f"No Real. {method}"
    real_col  = f"Real. {method}"
    total_col = f"Total {method}"
    cost_col  = f"P.Med {'WAC' if method == 'WAC' else 'FIFO'}"

    tbl = summary[["Ticker", "Cant.", cost_col, "Precio Actual", "Val. Mercado",
                   unr_col, real_col, "Dividendos", total_col]].copy()
    tbl.columns = ["Ticker", "Cant.", "Coste Medio", "Precio",
                   "Val. Mercado", "No Real.", "Realizado", "Dividendos", "Total P&L"]

    # Unrealized %  (NaN-safe)
    def _unr_pct(row):
        try:
            cm = float(row["Coste Medio"])
            pr = float(row["Precio"])
            if cm > 0 and pr > 0:
                return (pr / cm - 1) * 100
        except Exception:
            pass
        return float("nan")

    tbl["No Real. %"] = tbl.apply(_unr_pct, axis=1)

    # Build display copy with formatted strings (avoids Styler lambda NaN issues)
    disp = pd.DataFrame()
    disp["Ticker"]       = tbl["Ticker"]
    disp["Cant."]        = tbl["Cant."].apply(lambda v: f"{v:,.4f}" if pd.notna(v) else "—")
    disp["Coste Medio"]  = tbl["Coste Medio"].apply(lambda v: f"${v:,.4f}" if pd.notna(v) and v > 0 else "—")
    disp["Precio"]       = tbl["Precio"].apply(lambda v: f"${v:,.4f}" if pd.notna(v) and v > 0 else "—")
    disp["No Real. %"]   = tbl["No Real. %"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    disp["No Real."]     = tbl["No Real."].apply(lambda v: _fmt_money(v, sign=True))
    disp["Realizado"]    = tbl["Realizado"].apply(lambda v: _fmt_money(v, sign=True))
    disp["Dividendos"]   = tbl["Dividendos"].apply(lambda v: _fmt_money(v, sign=True))
    disp["Val. Mercado"] = tbl["Val. Mercado"].apply(lambda v: _fmt_money(v))
    disp["Total P&L"]    = tbl["Total P&L"].apply(lambda v: _fmt_money(v, sign=True))

    # Color only the money/pct columns via map (on string values now — no NaN crash)
    def _c_green_red(val: str) -> str:
        if val.startswith("+") and val != "—":
            return f"color:{POSITIVE};font-weight:600"
        if val.startswith("-"):
            return f"color:{NEGATIVE};font-weight:600"
        return ""

    st.dataframe(
        disp.style.map(_c_green_red,
                       subset=["No Real. %", "No Real.", "Realizado", "Dividendos", "Total P&L"]),
        use_container_width=True,
        hide_index=True,
    )

    # P&L bar chart
    pl_vals = tbl["Total P&L"].fillna(0)
    if pl_vals.abs().sum() > 0:
        fig = go.Figure(go.Bar(
            x=tbl["Ticker"],
            y=pl_vals,
            marker_color=[POSITIVE if v >= 0 else NEGATIVE for v in pl_vals],
            text=[_fmt_money(v, sign=True) for v in pl_vals],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_PRIMARY),
        ))
        layout = dict(**PLOTLY_DARK)
        layout["margin"] = dict(l=10, r=10, t=30, b=40)
        layout["height"] = 260
        layout["title"]  = dict(text=f"P&L total por ticker ({method})",
                                 font=dict(size=12, color=TEXT_MUTED))
        layout["yaxis"]  = dict(showgrid=True, gridcolor=BORDER_SOFT,
                                 zeroline=True, zerolinecolor=BORDER, showticklabels=False)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

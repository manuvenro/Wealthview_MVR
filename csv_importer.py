"""
modules/csv_importer.py — Broker CSV Import Engine
===================================================
Importa operaciones desde los formatos CSV de los brokers más comunes.

Brokers soportados (auto-detección):
  • DEGIRO          — extracto de cuenta (Account.csv / Transactions.csv)
  • Interactive Brokers — Activity Statement (CSV export)
  • Schwab          — Transaction History
  • Fidelity        — Download History
  • Trading 212     — Export Data
  • Genérico        — mapeo manual de columnas

Pipeline:
  1. Subida del archivo CSV
  2. Auto-detección de broker por cabeceras
  3. Parsing → DataFrame normalizado
  4. Preview + deduplicación vs. operaciones existentes
  5. Importación con un click
"""

import io
import re
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date

import modules.auth as auth
from modules.styles import (
    GOLD, GOLD_BORDER, SURFACE, BORDER, BORDER_SOFT, BG,
    TEXT_PRIMARY, TEXT_MUTED, POSITIVE, NEGATIVE,
    section_label, gold_divider,
)


# ─────────────────────────────────────────────────────────────────────────────
# Normalised schema (what we want to end up with)
# ─────────────────────────────────────────────────────────────────────────────
# date       str  "YYYY-MM-DD"
# ticker     str  uppercase
# type       str  buy | sell | dividend | fee | split
# quantity   float > 0
# price      float ≥ 0
# commission float ≥ 0
# currency   str  "USD" | "EUR" | ...
# notes      str  (broker name + original description)
# _hash      str  sha256 for dedup


# ─────────────────────────────────────────────────────────────────────────────
# Broker detection
# ─────────────────────────────────────────────────────────────────────────────

BROKER_SIGNATURES = {
    "degiro":    {"Fecha", "Producto", "ISIN", "Bolsa", "Número"},
    "ibkr":      {"DataDiscriminator", "Asset Category", "Symbol", "T. Price"},
    "schwab":    {"Action", "Symbol", "Description", "Fees & Comm"},
    "fidelity":  {"Run Date", "Action", "Symbol", "Description", "Settlement Date"},
    "trading212":{"Action", "Time", "ISIN", "Ticker", "No. of shares"},
}


def detect_broker(columns: list[str]) -> str:
    """Return broker name or 'generic'."""
    col_set = set(str(c).strip() for c in columns)
    for broker, sig in BROKER_SIGNATURES.items():
        if sig.issubset(col_set):
            return broker
    # Fuzzy: check partial overlap
    for broker, sig in BROKER_SIGNATURES.items():
        if len(sig & col_set) >= len(sig) * 0.6:
            return broker
    return "generic"


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FMTS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
    "%Y/%m/%d", "%d.%m.%Y", "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%Y%m%d",
]

def _parse_date(val) -> str | None:
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return None
    s = str(val).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def _parse_float(val) -> float:
    if pd.isna(val) or str(val).strip() in ("", "nan", "-"):
        return 0.0
    s = str(val).strip()
    # Remove currency symbols and thousands separators
    s = re.sub(r"[€$£¥\s]", "", s)
    s = s.replace(",", ".")
    # Handle European format: 1.234,56 → 1234.56
    if s.count(".") > 1:
        s = s.replace(".", "", s.count(".") - 1)
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


def _detect_currency(df: pd.DataFrame, col_candidates: list[str]) -> str:
    for c in col_candidates:
        if c in df.columns:
            val = str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
            if "€" in val or "EUR" in val.upper():
                return "EUR"
            if "$" in val or "USD" in val.upper():
                return "USD"
            if "£" in val or "GBP" in val.upper():
                return "GBP"
    return "USD"


# ─────────────────────────────────────────────────────────────────────────────
# Action → type mapping
# ─────────────────────────────────────────────────────────────────────────────

def _map_action(action: str) -> str:
    a = str(action).strip().lower()
    buy_kws  = ("buy", "compra", "purchase", "bought", "kauf", "acqui", "reinvest",
                 "subscription", "suscripci", "opening")
    sell_kws = ("sell", "venta", "sold", "verkauf", "closing", "close")
    div_kws  = ("dividend", "divid", "dividendo", "income", "distribution",
                 "coupon", "interest", "yield", "pago")
    fee_kws  = ("fee", "comisi", "commission", "charge", "tasa", "custody",
                 "management", "transfer", "stamp", "tax", "retenci")
    split_kws = ("split", "stock split", "reverse split", "desdoblamiento")

    for kw in split_kws:
        if kw in a:
            return "split"
    for kw in div_kws:
        if kw in a:
            return "dividend"
    for kw in fee_kws:
        if kw in a:
            return "fee"
    for kw in sell_kws:
        if kw in a:
            return "sell"
    for kw in buy_kws:
        if kw in a:
            return "buy"
    return "buy"  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Row hash for deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _row_hash(date_str: str, ticker: str, tx_type: str,
              quantity: float, price: float) -> str:
    key = f"{date_str}|{ticker}|{tx_type}|{quantity:.4f}|{price:.4f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# DEGIRO parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_degiro(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # DEGIRO has both EUR and original currency rows — skip duplicate
    # Columns: Fecha, Hora, Producto, ISIN, Bolsa, ..., Número, Precio, ..., Comisión, Total
    for _, r in df.iterrows():
        date_str = _parse_date(r.get("Fecha") or r.get("Date") or "")
        if not date_str:
            continue

        product  = str(r.get("Producto") or r.get("Product") or "")
        isin     = str(r.get("ISIN") or "").strip()
        # Handle accented and non-accented variants
        qty_raw  = (r.get("Número") or r.get("Numero") or r.get("Number")
                    or r.get("Cantidad") or r.get("Qty") or 0)
        price_raw = r.get("Precio") or r.get("Price") or 0
        comm_raw  = (r.get("Comisión") or r.get("Comision")
                     or r.get("Commission") or r.get("Fee") or 0)
        # Currency: try multiple column name variants
        currency  = str(
            r.get("Moneda (Precio)") or r.get("Currency (Price)")
            or r.get("Moneda") or r.get("Currency") or "EUR"
        )[:3]

        qty   = _parse_float(qty_raw)
        price = _parse_float(price_raw)
        comm  = _parse_float(comm_raw)

        # Ticker: DEGIRO uses ISIN, try to extract ticker from product name
        ticker = isin  # fallback to ISIN
        # Product format: "Apple Inc (AAPL)" or "Apple Inc"
        match = re.search(r"\(([A-Z0-9\.\-]{1,10})\)", product)
        if match:
            ticker = match.group(1)
        elif product:
            # Use first word of product as approximate ticker
            ticker = product.split()[0][:10].upper() if product else isin

        # Type detection from Número (negative = sell, positive = buy)
        if qty == 0:
            continue
        qty_signed = 0
        try:
            qty_signed = float(str(qty_raw).replace(",", "."))
        except Exception:
            qty_signed = qty

        tx_type = "sell" if qty_signed < 0 else "buy"
        qty = abs(qty)

        rows.append({
            "date": date_str, "ticker": ticker, "type": tx_type,
            "quantity": qty, "price": price, "commission": comm,
            "currency": currency.upper(), "notes": f"DEGIRO: {product}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Brokers parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ibkr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # IBKR Activity Statement has multiple sections tagged by first column
    # Look for Trades section rows
    for _, r in df.iterrows():
        disc = str(r.get("DataDiscriminator") or r.get("Header") or "").strip()
        if disc.lower() not in ("data", "trade", "order"):
            continue

        asset_cat = str(r.get("Asset Category") or "").lower()
        if "stock" not in asset_cat and "equity" not in asset_cat and "etf" not in asset_cat:
            continue

        date_str = _parse_date(r.get("Date/Time") or r.get("TradeDate") or "")
        if not date_str:
            continue

        symbol  = str(r.get("Symbol") or "").strip().upper()
        qty     = _parse_float(r.get("Quantity") or 0)
        price   = _parse_float(r.get("T. Price") or r.get("Price") or 0)
        comm    = _parse_float(r.get("Comm/Fee") or r.get("Commission") or 0)
        curr    = str(r.get("Currency") or "USD")[:3].upper()

        try:
            qty_signed = float(str(r.get("Quantity") or 0).replace(",", ""))
        except Exception:
            qty_signed = qty

        tx_type = "sell" if qty_signed < 0 else "buy"
        qty = abs(qty)

        if not symbol or qty == 0:
            continue

        rows.append({
            "date": date_str, "ticker": symbol, "type": tx_type,
            "quantity": qty, "price": price, "commission": comm,
            "currency": curr, "notes": "Interactive Brokers",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Schwab parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_schwab(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        action   = str(r.get("Action") or "")
        date_str = _parse_date(r.get("Date") or "")
        symbol   = str(r.get("Symbol") or "").strip().upper()
        qty      = _parse_float(r.get("Quantity") or 0)
        price    = _parse_float(r.get("Price") or 0)
        comm     = _parse_float(r.get("Fees & Comm") or 0)
        desc     = str(r.get("Description") or "")

        if not date_str or not symbol:
            continue

        tx_type = _map_action(action or desc)

        # Dividend/fee may have qty=0 but amount in a separate column
        if qty == 0 and tx_type not in ("dividend", "fee"):
            continue

        # For dividends with no price/qty, use Amount as total
        if tx_type == "dividend" and qty == 0:
            total_amt = _parse_float(r.get("Amount") or 0)
            qty = 1.0
            price = total_amt

        rows.append({
            "date": date_str, "ticker": symbol, "type": tx_type,
            "quantity": qty, "price": price, "commission": comm,
            "currency": "USD", "notes": f"Schwab: {action}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        action   = str(r.get("Action") or "")
        date_str = _parse_date(r.get("Run Date") or r.get("Date") or "")
        symbol   = str(r.get("Symbol") or "").strip().upper()
        qty      = _parse_float(r.get("Quantity") or 0)
        price    = _parse_float(r.get("Price ($)") or r.get("Price") or 0)
        comm     = _parse_float(r.get("Commission ($)") or r.get("Fees ($)") or 0)
        desc     = str(r.get("Description") or "")

        if not date_str or not symbol:
            continue

        tx_type = _map_action(action or desc)
        if qty == 0 and tx_type not in ("dividend", "fee"):
            continue

        # Fidelity dividend: use Amount ($) column as total if qty=0
        if tx_type == "dividend" and qty == 0:
            total_amt = _parse_float(r.get("Amount ($)") or r.get("Amount") or 0)
            qty = 1.0
            price = total_amt

        rows.append({
            "date": date_str, "ticker": symbol, "type": tx_type,
            "quantity": max(qty, 1.0) if tx_type == "dividend" else qty,
            "price": price, "commission": comm,
            "currency": "USD", "notes": f"Fidelity: {action}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Trading 212 parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_trading212(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        action   = str(r.get("Action") or "")
        date_str = _parse_date(r.get("Time") or "")
        ticker   = str(r.get("Ticker") or r.get("ISIN") or "").strip().upper()
        qty      = _parse_float(r.get("No. of shares") or 0)
        price    = _parse_float(r.get("Price / share") or 0)
        comm     = _parse_float(r.get("Currency conversion fee") or 0)
        curr_raw = str(r.get("Currency (Price / share)") or "USD")[:3].upper()

        if not date_str:
            continue

        tx_type = _map_action(action)
        if qty == 0 and tx_type not in ("dividend", "fee"):
            continue

        # Dividend in T212: price/share comes from "Result" field
        if tx_type == "dividend":
            div_total = _parse_float(r.get("Result") or r.get("Total") or 0)
            price = div_total / qty if qty > 0 else div_total
            qty = qty or 1.0

        rows.append({
            "date": date_str, "ticker": ticker, "type": tx_type,
            "quantity": qty, "price": price, "commission": comm,
            "currency": curr_raw, "notes": f"Trading 212: {action}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Generic parser with UI column mapping
# ─────────────────────────────────────────────────────────────────────────────

def _parse_generic(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        date_str = _parse_date(r.get(col_map.get("date", "")) or "")
        ticker   = str(r.get(col_map.get("ticker", ""), "")).strip().upper()
        action   = str(r.get(col_map.get("action", ""), ""))
        qty      = _parse_float(r.get(col_map.get("quantity", ""), 0))
        price    = _parse_float(r.get(col_map.get("price", ""), 0))
        comm     = _parse_float(r.get(col_map.get("commission", ""), 0))
        curr     = str(r.get(col_map.get("currency", ""), "USD"))[:3].upper() or "USD"

        if not date_str or not ticker:
            continue

        tx_type = col_map.get("type_override") or _map_action(action)
        if qty == 0 and tx_type not in ("dividend", "fee"):
            continue

        rows.append({
            "date": date_str, "ticker": ticker, "type": tx_type,
            "quantity": qty, "price": price, "commission": comm,
            "currency": curr, "notes": f"CSV: {action}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Master parse function
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(file_bytes: bytes, filename: str,
              generic_col_map: dict | None = None) -> tuple[str, pd.DataFrame]:
    """
    Auto-detect broker and parse CSV.
    Returns (broker_name, normalized_df).
    """
    # Try multiple encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            text = None

    if text is None:
        return "error", pd.DataFrame()

    # Try to find the data start (skip IBKR header sections)
    lines = text.splitlines()
    # IBKR has metadata at top — find first real data header
    start_line = 0
    for i, line in enumerate(lines):
        if "Symbol" in line or "Ticker" in line or "Fecha" in line or "Date" in line:
            start_line = i
            break

    csv_text = "\n".join(lines[start_line:])

    # Try comma and semicolon separators
    parsed_df = None
    for sep in (",", ";", "\t"):
        try:
            parsed_df = pd.read_csv(io.StringIO(csv_text), sep=sep,
                                    dtype=str, on_bad_lines="skip")
            if len(parsed_df.columns) >= 3:
                break
        except Exception:
            continue

    if parsed_df is None or parsed_df.empty:
        return "error", pd.DataFrame()

    # Strip column names
    parsed_df.columns = [str(c).strip() for c in parsed_df.columns]
    parsed_df = parsed_df.dropna(how="all")

    broker = detect_broker(list(parsed_df.columns))

    parsers = {
        "degiro":     _parse_degiro,
        "ibkr":       _parse_ibkr,
        "schwab":     _parse_schwab,
        "fidelity":   _parse_fidelity,
        "trading212": _parse_trading212,
    }

    if broker in parsers:
        result = parsers[broker](parsed_df)
    elif generic_col_map:
        result = _parse_generic(parsed_df, generic_col_map)
        broker = "generic"
    else:
        return "generic_unmapped", parsed_df  # return raw for mapping UI

    # Add hash for dedup
    if not result.empty:
        result["_hash"] = result.apply(
            lambda r: _row_hash(
                str(r.get("date", "")), str(r.get("ticker", "")),
                str(r.get("type", "")),  float(r.get("quantity", 0)),
                float(r.get("price", 0))
            ), axis=1
        )

    return broker, result


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication against existing transactions
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(new_df: pd.DataFrame, username: str) -> tuple[pd.DataFrame, int]:
    """
    Compare new_df against existing transactions.
    Returns (filtered_df, n_duplicates_removed).
    """
    from modules.transactions import get_transactions
    existing = get_transactions(username)
    if existing.empty or new_df.empty:
        return new_df, 0

    # Build hashes of existing transactions
    existing_hashes = set(
        _row_hash(
            str(r["date"]), str(r["ticker"]), str(r["type"]),
            float(r["quantity"]), float(r["price"])
        )
        for _, r in existing.iterrows()
    )

    if "_hash" not in new_df.columns:
        new_df["_hash"] = new_df.apply(
            lambda r: _row_hash(
                str(r.get("date","")), str(r.get("ticker","")),
                str(r.get("type","")), float(r.get("quantity",0)),
                float(r.get("price",0))
            ), axis=1
        )

    mask = ~new_df["_hash"].isin(existing_hashes)
    return new_df[mask].copy(), int((~mask).sum())


# ─────────────────────────────────────────────────────────────────────────────
# UI: Render import tab
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_LABELS = {
    "buy":      "Compra",
    "sell":     "Venta",
    "dividend": "Dividendo",
    "fee":      "Gasto",
    "split":    "Split",
}

_BROKER_LABELS = {
    "degiro":     "DEGIRO",
    "ibkr":       "Interactive Brokers",
    "schwab":     "Charles Schwab",
    "fidelity":   "Fidelity",
    "trading212": "Trading 212",
    "generic":    "Formato genérico",
}


def render_csv_import(username: str):
    section_label("Importar operaciones desde CSV")
    st.caption(
        "Sube el extracto CSV de tu broker. "
        "Formatos soportados: **DEGIRO**, **Interactive Brokers**, **Schwab**, "
        "**Fidelity**, **Trading 212** y cualquier CSV con mapeo manual."
    )

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Selecciona el archivo CSV de tu broker",
        type=["csv", "txt"],
        key="csv_upload",
    )

    if uploaded is None:
        _render_broker_help()
        return

    file_bytes = uploaded.read()

    with st.spinner("Analizando CSV..."):
        broker, parsed = parse_csv(file_bytes, uploaded.name)

    # ── Generic: show column mapping UI ──────────────────────────────────────
    if broker == "generic_unmapped":
        st.info("No se detectó el formato del broker automáticamente. "
                "Indica qué columna corresponde a cada campo.")
        col_map = _render_column_mapper(parsed)
        if col_map is None:
            return
        broker, parsed = "generic", _parse_generic(parsed, col_map)
        if not parsed.empty:
            parsed["_hash"] = parsed.apply(
                lambda r: _row_hash(
                    str(r.get("date","")), str(r.get("ticker","")),
                    str(r.get("type","")), float(r.get("quantity",0)),
                    float(r.get("price",0))
                ), axis=1
            )

    if broker == "error" or parsed.empty:
        st.error("No se pudo leer el archivo. Comprueba que es un CSV válido.")
        return

    broker_label = _BROKER_LABELS.get(broker, broker.title())
    st.success(f"Broker detectado: **{broker_label}** — {len(parsed)} operaciones encontradas")

    # ── Deduplication ─────────────────────────────────────────────────────────
    clean, n_dups = deduplicate(parsed, username)

    if n_dups > 0:
        st.warning(f"Se detectaron **{n_dups} operaciones duplicadas** "
                   f"(ya existen en el historial) y se excluirán de la importación.")

    if clean.empty:
        st.info("Todas las operaciones del CSV ya están importadas. Nada nuevo que añadir.")
        return

    st.markdown(f"**{len(clean)} operaciones nuevas** listas para importar:")

    # ── Preview table ─────────────────────────────────────────────────────────
    preview = clean.copy()
    disp_cols = ["date", "ticker", "type", "quantity", "price", "commission", "currency", "notes"]
    preview = preview[[c for c in disp_cols if c in preview.columns]].copy()
    preview.columns = ["Fecha", "Ticker", "Tipo", "Cantidad", "Precio ($)",
                       "Comisión ($)", "Divisa", "Notas"][:len(preview.columns)]
    preview["Tipo"] = preview["Tipo"].map(lambda x: _TYPE_LABELS.get(str(x), str(x)))

    # Color buy/sell
    def _color_type(val):
        m = {"Compra": f"color:{POSITIVE};font-weight:600",
             "Venta":  f"color:{NEGATIVE};font-weight:600",
             "Dividendo": f"color:{GOLD};font-weight:600"}
        return m.get(val, "")

    st.dataframe(
        preview.style.map(_color_type, subset=["Tipo"]).format({
            "Cantidad":     "{:,.4f}",
            "Precio ($)":   "{:,.4f}",
            "Comisión ($)": "{:,.2f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
        height=min(400, 40 + len(preview) * 36),
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filtrar operaciones a importar", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tickers_all = sorted(clean["ticker"].unique().tolist())
            tickers_sel = st.multiselect("Tickers a incluir", tickers_all,
                                         default=tickers_all, key="ci_tickers")
        with col_f2:
            types_all = sorted(clean["type"].unique().tolist())
            types_sel = st.multiselect("Tipos a incluir", types_all,
                                       default=types_all,
                                       format_func=lambda x: _TYPE_LABELS.get(x, x),
                                       key="ci_types")
        clean = clean[
            clean["ticker"].isin(tickers_sel) &
            clean["type"].isin(types_sel)
        ]
        if len(clean) < len(parsed) - n_dups:
            st.caption(f"{len(clean)} operaciones seleccionadas tras filtros.")

    # ── Import button ─────────────────────────────────────────────────────────
    gold_divider()
    col_imp, col_cancel = st.columns([3, 1])
    with col_imp:
        if st.button(
            f"Importar {len(clean)} operaciones de {broker_label}",
            type="primary", use_container_width=True, key="ci_import_btn"
        ):
            errors = []
            imported = 0
            progress = st.progress(0, text="Importando...")
            total = len(clean)

            for i, (_, row) in enumerate(clean.iterrows()):
                try:
                    qty = float(row.get("quantity", 0))
                    price = float(row.get("price", 0))

                    # For fee type, price=amount, qty=1
                    if str(row.get("type")) == "fee":
                        qty = 1.0
                        price = qty if qty > 0 else _parse_float(row.get("price", 0))
                        price = float(row.get("price", 0)) or float(row.get("quantity", 0))
                        qty = 1.0

                    from modules.transactions import add_transaction
                    add_transaction(
                        username=username,
                        ticker=str(row.get("ticker", "UNKN")).upper()[:20],
                        date_str=str(row.get("date", date.today().isoformat())),
                        tx_type=str(row.get("type", "buy")),
                        quantity=max(abs(qty), 1e-9),
                        price=max(abs(price), 0.0),
                        commission=abs(float(row.get("commission", 0) or 0)),
                        currency=str(row.get("currency", "USD"))[:3].upper(),
                        notes=str(row.get("notes", broker_label))[:200],
                    )
                    imported += 1
                except Exception as exc:
                    errors.append(f"Fila {i+1}: {exc}")

                progress.progress((i + 1) / total,
                                  text=f"Importando {i+1}/{total}...")

            progress.empty()

            if imported > 0:
                st.success(f"✅ {imported} operaciones importadas correctamente.")
                st.balloons()
                # Clear cached positions
                try:
                    from modules.transactions import _fetch_prices_for_positions
                    _fetch_prices_for_positions.clear()
                except Exception:
                    pass
            if errors:
                with st.expander(f"⚠ {len(errors)} errores durante la importación"):
                    for e in errors:
                        st.caption(e)

    with col_cancel:
        if st.button("Cancelar", use_container_width=True, key="ci_cancel"):
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Column mapper UI (generic CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _render_column_mapper(df: pd.DataFrame) -> dict | None:
    st.markdown("**Vista previa del CSV (primeras 5 filas):**")
    st.dataframe(df.head(5), use_container_width=True)

    cols = ["— (no mapear)"] + list(df.columns)

    def _pick(label, candidates, required=False):
        default = next((c for c in candidates if c in df.columns), cols[0])
        idx = cols.index(default) if default in cols else 0
        sel = st.selectbox(label + (" *" if required else ""), cols, index=idx,
                           key=f"cmap_{label}")
        return None if sel == cols[0] else sel

    st.markdown("##### Mapeo de columnas")
    c1, c2 = st.columns(2)
    with c1:
        date_col   = _pick("Fecha",    ["Date", "Fecha", "Trade Date", "Settlement Date"], required=True)
        ticker_col = _pick("Ticker",   ["Symbol", "Ticker", "ISIN", "Instrumento"], required=True)
        qty_col    = _pick("Cantidad", ["Quantity", "Qty", "Número", "Shares", "Units"])
        price_col  = _pick("Precio",   ["Price", "Precio", "T. Price", "Price ($)"])
    with c2:
        action_col = _pick("Acción/Tipo", ["Action", "Type", "Transaction Type", "Tipo"])
        comm_col   = _pick("Comisión", ["Commission", "Fee", "Comisión", "Fees & Comm"])
        curr_col   = _pick("Divisa",   ["Currency", "Moneda", "Ccy"])

    if not date_col or not ticker_col:
        st.warning("Fecha y Ticker son obligatorios.")
        return None

    if st.button("Confirmar mapeo y continuar", type="primary", key="cmap_confirm"):
        return {
            "date": date_col,
            "ticker": ticker_col,
            "action": action_col or "",
            "quantity": qty_col or "",
            "price": price_col or "",
            "commission": comm_col or "",
            "currency": curr_col or "",
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Help: how to export from each broker
# ─────────────────────────────────────────────────────────────────────────────

def _render_broker_help():
    with st.expander("¿Cómo exportar el CSV de tu broker?"):
        st.markdown("""
**DEGIRO**
1. Inicia sesión → Actividad → Extracto de cuenta
2. Selecciona el período y descarga en formato CSV

**Interactive Brokers**
1. Reports → Activity → Create Report (Custom)
2. Sections: Trades · Date format: YYYY-MM-DD
3. Delivery: Download as CSV

**Schwab**
1. Accounts → Transaction History
2. Date range → Export (CSV)

**Fidelity**
1. Accounts & Trade → Account History
2. Download → Comma-Separated Values (.CSV)

**Trading 212**
1. Profile → History → Export CSV
2. Selecciona el rango de fechas

**Otro broker**
Sube el CSV igualmente — si las columnas no se detectan automáticamente,
podrás mapearlas manualmente.
""")

"""
modules/pdf_importer.py — PDF Broker Statement Importer
=========================================================
Extrae operaciones de extractos PDF de brokers y las importa
al mismo pipeline que csv_importer.py.

Brokers soportados:
  - DEGIRO         (extracto mensual / anual de cuenta)
  - Interactive Brokers (IBKR Activity Statement)
  - Schwab         (Trade Confirmation / Account Statement)
  - Fidelity       (Brokerage Account Statement)
  - Trading 212    (Monthly Statement)
  - Genérico       (UI manual para PDFs no reconocidos)

Motor de extracción:
  1. pdfplumber  → extrae tablas estructuradas (la mayoría de brokers)
  2. Regex       → fallback sobre texto plano del PDF

Reutiliza:
  - csv_importer.deduplicate()  → deduplicación por hash SHA256
  - auth.add_transaction()      → persistencia en SQLite
"""

import re
import io
import logging
import datetime
from typing import Optional

import pandas as pd
import streamlit as st

import modules.auth as auth
from modules.csv_importer import deduplicate, _parse_date, _parse_float, _map_action

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Broker detection by PDF text signatures
# ─────────────────────────────────────────────────────────────────────────────

BROKER_TEXT_SIGNATURES = {
    "degiro":    ["degiro", "www.degiro", "flatex"],
    "ibkr":      ["interactive brokers", "ibkr", "ib statement", "activity statement"],
    "schwab":    ["charles schwab", "schwab.com", "schwab one"],
    "fidelity":  ["fidelity investments", "fidelity.com", "fidelity brokerage"],
    "trading212":["trading 212", "trading212", "212 ltd"],
}

BROKER_LABELS = {
    "degiro":     "DEGIRO",
    "ibkr":       "Interactive Brokers",
    "schwab":     "Charles Schwab",
    "fidelity":   "Fidelity",
    "trading212": "Trading 212",
    "generic":    "Genérico (mapeo manual)",
}


def detect_broker_from_text(text: str) -> str:
    """Detecta el broker analizando el texto completo del PDF."""
    text_lower = text.lower()
    for broker, sigs in BROKER_TEXT_SIGNATURES.items():
        if any(sig in text_lower for sig in sigs):
            return broker
    return "generic"


# ─────────────────────────────────────────────────────────────────────────────
# PDF text + table extraction (pdfplumber)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(file_bytes: bytes) -> tuple[str, list[list]]:
    """
    Extrae texto completo y tablas del PDF.
    Devuelve (full_text, tables) donde tables es lista de listas de filas.
    """
    try:
        import pdfplumber
    except ImportError:
        st.error("Instala pdfplumber: `pip install pdfplumber`")
        return "", []

    full_text = []
    all_tables = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Texto completo de la página
            txt = page.extract_text() or ""
            full_text.append(txt)

            # Tablas de la página
            tables = page.extract_tables() or []
            for tbl in tables:
                if tbl:
                    all_tables.append(tbl)

    return "\n".join(full_text), all_tables


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(v) -> str:
    """Limpia una celda de tabla: None → '', strip whitespace."""
    if v is None:
        return ""
    return str(v).strip().replace("\n", " ")


def _tables_to_dfs(tables: list[list]) -> list[pd.DataFrame]:
    """Convierte tablas raw (lista de listas) en DataFrames con headers."""
    dfs = []
    for tbl in tables:
        if not tbl or len(tbl) < 2:
            continue
        # First row as header, clean None cells
        header = [_clean(c) or f"col_{i}" for i, c in enumerate(tbl[0])]
        rows = [[_clean(c) for c in row] for row in tbl[1:]]
        # Filter empty rows
        rows = [r for r in rows if any(c for c in r)]
        if rows:
            dfs.append(pd.DataFrame(rows, columns=header))
    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# DEGIRO parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_degiro_pdf(text: str, tables: list) -> pd.DataFrame:
    """
    DEGIRO monthly/yearly account statement PDF.
    Typical columns: Fecha, Hora, Producto, ISIN, Bolsa, Número, Precio, ...
    """
    dfs = _tables_to_dfs(tables)
    rows = []

    date_cols   = ["fecha", "date", "dag"]
    prod_cols   = ["producto", "product", "naam"]
    isin_cols   = ["isin"]
    qty_cols    = ["número", "numero", "number", "aantal", "qty", "quantity"]
    price_cols  = ["precio", "price", "koers", "prijs"]
    type_cols   = ["tipo", "type", "soort"]

    def _find_col(df, candidates):
        for c in candidates:
            for col in df.columns:
                if c in col.lower():
                    return col
        return None

    for df in dfs:
        date_c  = _find_col(df, date_cols)
        prod_c  = _find_col(df, prod_cols)
        isin_c  = _find_col(df, isin_cols)
        qty_c   = _find_col(df, qty_cols)
        price_c = _find_col(df, price_cols)
        type_c  = _find_col(df, type_cols)

        if not date_c or not qty_c or not price_c:
            continue

        for _, row in df.iterrows():
            date_str = _clean(row.get(date_c, ""))
            prod_str = _clean(row.get(prod_c, "")) if prod_c else ""
            isin_str = _clean(row.get(isin_c, "")) if isin_c else ""
            qty_str  = _clean(row.get(qty_c, ""))
            price_str = _clean(row.get(price_c, ""))
            type_str = _clean(row.get(type_c, "")) if type_c else ""

            date = _parse_date(date_str)
            qty  = _parse_float(qty_str)
            price = _parse_float(price_str)

            if not date or price is None:
                continue

            # DEGIRO: negative qty = sell, positive = buy
            tx_type = _map_action(type_str) if type_str else ("sell" if (qty or 0) < 0 else "buy")
            qty = abs(qty or 0)

            # Ticker: try ISIN first, then product name abbreviated
            ticker = isin_str or re.sub(r"\s+", "", prod_str[:10]).upper() or "?"
            name   = prod_str or ticker

            if qty > 0:
                rows.append({
                    "date": date, "ticker": ticker, "name": name,
                    "type": tx_type, "quantity": qty, "price": price,
                    "currency": "EUR", "broker": "DEGIRO",
                })

    # Fallback: regex on text if no table matched
    if not rows:
        rows = _degiro_regex(text)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _degiro_regex(text: str) -> list:
    """Regex fallback para PDFs de DEGIRO con formato de texto libre."""
    rows = []
    # Pattern: dd-mm-yyyy  HH:MM  Product Name  ISIN  Exchange  Qty  Price  ...
    pattern = re.compile(
        r"(\d{2}[-/]\d{2}[-/]\d{4})"          # date
        r"\s+\d{2}:\d{2}"                       # time
        r"\s+(.+?)"                             # product
        r"\s+([A-Z]{2}\w{10})"                 # ISIN
        r"\s+\w+"                               # exchange
        r"\s+([-\d,.]+)"                        # qty
        r"\s+([A-Z]{3})"                        # currency
        r"\s+([\d,.]+)"                         # price
    )
    for m in pattern.finditer(text):
        date = _parse_date(m.group(1))
        qty  = _parse_float(m.group(4))
        price = _parse_float(m.group(6))
        if date and price is not None and qty:
            tx_type = "sell" if qty < 0 else "buy"
            rows.append({
                "date": date, "ticker": m.group(3), "name": m.group(2).strip(),
                "type": tx_type, "quantity": abs(qty), "price": price,
                "currency": m.group(5), "broker": "DEGIRO",
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# IBKR parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ibkr_pdf(text: str, tables: list) -> pd.DataFrame:
    """
    Interactive Brokers Activity Statement PDF.
    Table header: Symbol, Date/Time, Quantity, T. Price, C. Price, Proceeds, ...
    """
    dfs = _tables_to_dfs(tables)
    rows = []

    for df in dfs:
        cols_lower = {c.lower(): c for c in df.columns}

        # Must have symbol + date + quantity + price
        sym_c  = next((cols_lower[k] for k in cols_lower if "symbol" in k), None)
        date_c = next((cols_lower[k] for k in cols_lower if "date" in k or "time" in k), None)
        qty_c  = next((cols_lower[k] for k in cols_lower if "quantity" in k or "qty" in k), None)
        price_c = next((cols_lower[k] for k in cols_lower if "t. price" in k or "price" in k), None)

        if not all([sym_c, date_c, qty_c, price_c]):
            continue

        for _, row in df.iterrows():
            sym   = _clean(row.get(sym_c, ""))
            date_str = _clean(row.get(date_c, ""))
            qty_str  = _clean(row.get(qty_c, ""))
            price_str = _clean(row.get(price_c, ""))

            if not sym or sym.lower() in ("symbol", "total", "subtotal", ""):
                continue

            date  = _parse_date(date_str.split(",")[0].split(" ")[0])
            qty   = _parse_float(qty_str)
            price = _parse_float(price_str)

            if not date or price is None or not qty:
                continue

            tx_type = "sell" if (qty or 0) < 0 else "buy"
            rows.append({
                "date": date, "ticker": sym.upper(), "name": sym.upper(),
                "type": tx_type, "quantity": abs(qty), "price": price,
                "currency": "USD", "broker": "IBKR",
            })

    if not rows:
        rows = _ibkr_regex(text)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _ibkr_regex(text: str) -> list:
    """Regex fallback para IBKR texto libre."""
    rows = []
    # Trades section: AAPL  Stocks  2024-01-15, 09:30:00  -100  185.50  ...
    pattern = re.compile(
        r"([A-Z]{1,5})\s+Stocks?\s+"
        r"(\d{4}-\d{2}-\d{2})[,\s]"
        r".*?"
        r"([-\d,.]+)\s+"       # qty
        r"([\d,.]+)"           # price
    )
    for m in pattern.finditer(text):
        date  = _parse_date(m.group(2))
        qty   = _parse_float(m.group(3))
        price = _parse_float(m.group(4))
        if date and qty and price:
            rows.append({
                "date": date, "ticker": m.group(1), "name": m.group(1),
                "type": "sell" if qty < 0 else "buy",
                "quantity": abs(qty), "price": price,
                "currency": "USD", "broker": "IBKR",
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Schwab parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_schwab_pdf(text: str, tables: list) -> pd.DataFrame:
    """Charles Schwab account statement / trade confirmation PDF."""
    dfs = _tables_to_dfs(tables)
    rows = []

    for df in dfs:
        cols_lower = {c.lower(): c for c in df.columns}
        date_c   = next((cols_lower[k] for k in cols_lower if "date" in k), None)
        action_c = next((cols_lower[k] for k in cols_lower if "action" in k or "type" in k or "description" in k), None)
        sym_c    = next((cols_lower[k] for k in cols_lower if "symbol" in k), None)
        qty_c    = next((cols_lower[k] for k in cols_lower if "quantity" in k or "qty" in k or "shares" in k), None)
        price_c  = next((cols_lower[k] for k in cols_lower if "price" in k and "commission" not in k), None)

        if not all([date_c, sym_c, qty_c, price_c]):
            continue

        for _, row in df.iterrows():
            date_str   = _clean(row.get(date_c, ""))
            action_str = _clean(row.get(action_c, "")) if action_c else ""
            sym        = _clean(row.get(sym_c, ""))
            qty_str    = _clean(row.get(qty_c, ""))
            price_str  = _clean(row.get(price_c, ""))

            if not sym or not date_str:
                continue

            date  = _parse_date(date_str)
            qty   = _parse_float(qty_str)
            price = _parse_float(price_str)
            tx_type = _map_action(action_str) if action_str else ("buy" if (qty or 0) > 0 else "sell")

            if not date or price is None:
                continue

            if qty == 0 and tx_type not in ("dividend", "fee"):
                continue

            rows.append({
                "date": date, "ticker": sym.upper(), "name": sym.upper(),
                "type": tx_type, "quantity": abs(qty or 0), "price": price,
                "currency": "USD", "broker": "Schwab",
            })

    if not rows:
        rows = _schwab_regex(text)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _schwab_regex(text: str) -> list:
    """Regex fallback para Schwab."""
    rows = []
    # "01/15/2024  Buy  AAPL  100  $185.50"
    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4})\s+"
        r"(Buy|Sell|Bought|Sold|Dividend|Reinvest)\s+"
        r"([A-Z]{1,5})\s+"
        r"([\d,.]+)\s+"
        r"\$([\d,.]+)"
    )
    for m in pattern.finditer(text):
        date  = _parse_date(m.group(1))
        qty   = _parse_float(m.group(4))
        price = _parse_float(m.group(5))
        if date and qty and price:
            rows.append({
                "date": date, "ticker": m.group(3), "name": m.group(3),
                "type": _map_action(m.group(2)),
                "quantity": qty, "price": price,
                "currency": "USD", "broker": "Schwab",
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fidelity_pdf(text: str, tables: list) -> pd.DataFrame:
    """Fidelity Investments brokerage statement PDF."""
    dfs = _tables_to_dfs(tables)
    rows = []

    for df in dfs:
        cols_lower = {c.lower(): c for c in df.columns}
        date_c   = next((cols_lower[k] for k in cols_lower if "date" in k), None)
        action_c = next((cols_lower[k] for k in cols_lower if "action" in k or "transaction" in k), None)
        sym_c    = next((cols_lower[k] for k in cols_lower if "symbol" in k), None)
        qty_c    = next((cols_lower[k] for k in cols_lower if "quantity" in k or "shares" in k), None)
        price_c  = next((cols_lower[k] for k in cols_lower if "price" in k), None)

        if not all([date_c, sym_c, qty_c, price_c]):
            continue

        for _, row in df.iterrows():
            date_str   = _clean(row.get(date_c, ""))
            action_str = _clean(row.get(action_c, "")) if action_c else ""
            sym        = _clean(row.get(sym_c, ""))
            qty_str    = _clean(row.get(qty_c, ""))
            price_str  = _clean(row.get(price_c, ""))

            if not sym or not date_str:
                continue

            date  = _parse_date(date_str)
            qty   = _parse_float(qty_str)
            price = _parse_float(price_str)
            tx_type = _map_action(action_str) if action_str else "buy"

            if not date or price is None:
                continue
            if qty == 0 and tx_type not in ("dividend", "fee"):
                continue

            rows.append({
                "date": date, "ticker": sym.upper(), "name": sym.upper(),
                "type": tx_type, "quantity": abs(qty or 0), "price": price,
                "currency": "USD", "broker": "Fidelity",
            })

    if not rows:
        rows = _fidelity_regex(text)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fidelity_regex(text: str) -> list:
    """Regex fallback para Fidelity."""
    rows = []
    pattern = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})[,\s]+(\d{4})\s+"
        r"(YOU BOUGHT|YOU SOLD|DIVIDEND RECEIVED|REINVESTMENT)\s+"
        r"([A-Z]{1,5})\s+"
        r"([\d,.]+)\s+\$([\d,.]+)"
    )
    for m in pattern.finditer(text):
        date_str = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
        date = _parse_date(date_str)
        qty = _parse_float(m.group(6))
        price = _parse_float(m.group(7))
        if date and qty and price:
            rows.append({
                "date": date, "ticker": m.group(5), "name": m.group(5),
                "type": _map_action(m.group(4)),
                "quantity": qty, "price": price,
                "currency": "USD", "broker": "Fidelity",
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Trading 212 parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_trading212_pdf(text: str, tables: list) -> pd.DataFrame:
    """Trading 212 monthly statement PDF."""
    dfs = _tables_to_dfs(tables)
    rows = []

    for df in dfs:
        cols_lower = {c.lower(): c for c in df.columns}
        date_c  = next((cols_lower[k] for k in cols_lower if "date" in k or "time" in k), None)
        act_c   = next((cols_lower[k] for k in cols_lower if "action" in k or "type" in k), None)
        isin_c  = next((cols_lower[k] for k in cols_lower if "isin" in k), None)
        tick_c  = next((cols_lower[k] for k in cols_lower if "ticker" in k or "symbol" in k), None)
        qty_c   = next((cols_lower[k] for k in cols_lower if "shares" in k or "quantity" in k or "no." in k), None)
        price_c = next((cols_lower[k] for k in cols_lower if "price" in k), None)

        if not all([date_c, qty_c, price_c]):
            continue

        for _, row in df.iterrows():
            date_str  = _clean(row.get(date_c, ""))
            act_str   = _clean(row.get(act_c, "")) if act_c else ""
            isin_str  = _clean(row.get(isin_c, "")) if isin_c else ""
            tick_str  = _clean(row.get(tick_c, "")) if tick_c else ""
            qty_str   = _clean(row.get(qty_c, ""))
            price_str = _clean(row.get(price_c, ""))

            date  = _parse_date(date_str.split(" ")[0])
            qty   = _parse_float(qty_str)
            price = _parse_float(price_str)
            tx_type = _map_action(act_str) if act_str else "buy"

            if not date or price is None:
                continue
            if qty == 0 and tx_type not in ("dividend", "fee"):
                continue

            ticker = tick_str or isin_str or "?"
            rows.append({
                "date": date, "ticker": ticker.upper(), "name": ticker,
                "type": tx_type, "quantity": abs(qty or 0), "price": price,
                "currency": "EUR", "broker": "Trading 212",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Generic text parser (regex heuristics)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_generic_pdf(text: str, tables: list) -> pd.DataFrame:
    """
    Parser genérico: busca patrones comunes de operaciones en cualquier PDF.
    Detecta: fecha + ticker + compra/venta + cantidad + precio.
    """
    rows = []

    # Pattern: various date formats + optional action word + ticker (1-5 caps) + qty + price
    patterns = [
        # ISO date
        re.compile(
            r"(\d{4}-\d{2}-\d{2})\s+"
            r"(?:(buy|sell|bought|sold|compra|venta|dividend|dividendo)\s+)?"
            r"([A-Z]{1,5})\s+"
            r"([\d,.]+)\s+"
            r"[@$€£]?\s*([\d,.]+)",
            re.IGNORECASE
        ),
        # European date dd/mm/yyyy or dd-mm-yyyy
        re.compile(
            r"(\d{2}[/.-]\d{2}[/.-]\d{4})\s+"
            r"(?:(buy|sell|bought|sold|compra|venta|dividend|dividendo)\s+)?"
            r"([A-Z]{1,5})\s+"
            r"([\d,.]+)\s+"
            r"[@$€£]?\s*([\d,.]+)",
            re.IGNORECASE
        ),
        # US date mm/dd/yyyy
        re.compile(
            r"(\d{2}/\d{2}/\d{4})\s+"
            r"(?:(buy|sell|bought|sold|dividend)\s+)?"
            r"([A-Z]{1,5})\s+"
            r"([\d,.]+)\s+"
            r"\$([\d,.]+)",
            re.IGNORECASE
        ),
    ]

    seen = set()
    for pat in patterns:
        for m in pat.finditer(text):
            date  = _parse_date(m.group(1))
            act   = m.group(2) or "buy"
            tick  = m.group(3).upper()
            qty   = _parse_float(m.group(4))
            price = _parse_float(m.group(5))

            if not date or not qty or price is None or price <= 0:
                continue

            key = (date, tick, round(qty, 2), round(price, 4))
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "date": date, "ticker": tick, "name": tick,
                "type": _map_action(act), "quantity": abs(qty), "price": price,
                "currency": "USD", "broker": "Genérico",
            })

    # Also try table extraction with generic column matching
    if not rows:
        dfs = _tables_to_dfs(tables)
        for df in dfs:
            cols_lower = {c.lower(): c for c in df.columns}
            date_c  = next((cols_lower[k] for k in cols_lower if "date" in k or "fecha" in k), None)
            sym_c   = next((cols_lower[k] for k in cols_lower if any(x in k for x in ["symbol", "ticker", "isin", "producto"])), None)
            qty_c   = next((cols_lower[k] for k in cols_lower if any(x in k for x in ["qty", "quantity", "shares", "número", "numero"])), None)
            price_c = next((cols_lower[k] for k in cols_lower if "price" in k or "precio" in k or "koers" in k), None)
            act_c   = next((cols_lower[k] for k in cols_lower if any(x in k for x in ["action", "type", "tipo"])), None)

            if not all([date_c, sym_c, qty_c, price_c]):
                continue

            for _, row in df.iterrows():
                date  = _parse_date(_clean(row.get(date_c, "")))
                sym   = _clean(row.get(sym_c, "")).upper()
                qty   = _parse_float(_clean(row.get(qty_c, "")))
                price = _parse_float(_clean(row.get(price_c, "")))
                act   = _clean(row.get(act_c, "")) if act_c else "buy"

                if not date or not sym or price is None or price <= 0:
                    continue

                tx_type = _map_action(act) if act else ("sell" if (qty or 0) < 0 else "buy")
                if qty == 0 and tx_type not in ("dividend", "fee"):
                    continue

                rows.append({
                    "date": date, "ticker": sym, "name": sym,
                    "type": tx_type, "quantity": abs(qty or 0), "price": price,
                    "currency": "USD", "broker": "Genérico",
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

PARSERS = {
    "degiro":     _parse_degiro_pdf,
    "ibkr":       _parse_ibkr_pdf,
    "schwab":     _parse_schwab_pdf,
    "fidelity":   _parse_fidelity_pdf,
    "trading212": _parse_trading212_pdf,
    "generic":    _parse_generic_pdf,
}


def parse_pdf(file_bytes: bytes, filename: str = "") -> tuple[str, pd.DataFrame]:
    """
    Pipeline principal: extrae texto/tablas → detecta broker → parsea.
    Devuelve (broker_id, df_con_operaciones).
    """
    text, tables = extract_pdf(file_bytes)

    broker = detect_broker_from_text(text)
    parser = PARSERS.get(broker, _parse_generic_pdf)

    df = parser(text, tables)

    # Normalize output schema
    if not df.empty:
        df = df.copy()
        for col in ["date", "ticker", "name", "type", "quantity", "price", "currency", "broker"]:
            if col not in df.columns:
                df[col] = "" if col in ("name", "currency", "broker") else None

        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df["price"]    = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df = df[df["price"] > 0].copy()

    return broker, df


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "buy":      ("#27ae60", "Compra"),
    "sell":     ("#e74c3c", "Venta"),
    "dividend": ("#3498db", "Dividendo"),
    "fee":      ("#95a5a6", "Comisión"),
    "split":    ("#9b59b6", "Split"),
}


def render_pdf_import(username: str):
    """Interfaz Streamlit para importar PDFs de brokers."""

    from modules.styles import (
        SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
        TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
        GOLD, POSITIVE, NEGATIVE,
    )

    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:0 0 16px 0;'>"
        f"Importar extracto PDF de broker</p>",
        unsafe_allow_html=True,
    )

    # ── Broker guide ─────────────────────────────────────────────────────────
    with st.expander("¿Cómo exportar tu extracto PDF?"):
        st.markdown("""
**DEGIRO** → Mi cuenta → Extractos → Extracto de cuenta → Selecciona período → Descargar PDF

**Interactive Brokers** → Reportes → Activity Statements → PDF → Descargar

**Charles Schwab** → Accounts → Statements → Monthly Statement → Download PDF

**Fidelity** → Accounts & Trade → Statements → Download (PDF)

**Trading 212** → History → Monthly Statement → Export PDF

> Los PDFs se procesan **localmente** en tu equipo. Ningún dato se envía a ningún servidor externo.
        """)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Sube tu extracto PDF",
        type=["pdf"],
        help="Extracto mensual, anual o de confirmación de operaciones de tu broker.",
    )

    if not uploaded:
        st.markdown(
            f"<div style='background:{SURFACE}; border:1px dashed {BORDER}; "
            f"border-radius:8px; padding:32px; text-align:center; color:{TEXT_MUTED}; "
            f"font-size:13px;'>Sube un PDF de tu broker para comenzar</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Parse ─────────────────────────────────────────────────────────────────
    file_bytes = uploaded.read()

    with st.spinner("Extrayendo operaciones del PDF..."):
        broker, df = parse_pdf(file_bytes, uploaded.name)

    broker_label = BROKER_LABELS.get(broker, broker.upper())

    # Broker badge
    badge_color = {
        "degiro": "#e74c3c", "ibkr": "#3498db", "schwab": "#27ae60",
        "fidelity": "#9b59b6", "trading212": "#f39c12", "generic": "#95a5a6",
    }.get(broker, "#95a5a6")

    st.markdown(
        f"<span style='background:{badge_color}22; color:{badge_color}; "
        f"border:1px solid {badge_color}44; border-radius:4px; "
        f"font-size:11px; font-weight:700; padding:3px 10px; "
        f"text-transform:uppercase; letter-spacing:0.5px;'>"
        f"● {broker_label}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── No rows found ─────────────────────────────────────────────────────────
    if df.empty:
        st.warning(
            "No se encontraron operaciones en este PDF. "
            "Puede que el PDF esté escaneado (imagen sin texto) o tenga un formato no soportado. "
            "Prueba a exportar el CSV desde el broker en su lugar."
        )
        return

    # ── Deduplication ─────────────────────────────────────────────────────────
    new_df, dupes = deduplicate(df, username)

    st.markdown(
        f"<div style='background:{SURFACE}; border:1px solid {BORDER_SOFT}; "
        f"border-radius:8px; padding:16px 20px; margin-bottom:16px; "
        f"display:flex; gap:32px;'>"
        f"<div><div style='font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:4px;'>Encontradas</div>"
        f"<div style='font-size:22px; font-weight:700; color:{TEXT_PRIMARY};'>{len(df)}</div></div>"
        f"<div><div style='font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:4px;'>Nuevas</div>"
        f"<div style='font-size:22px; font-weight:700; color:{POSITIVE};'>{len(new_df)}</div></div>"
        f"<div><div style='font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:4px;'>Duplicadas</div>"
        f"<div style='font-size:22px; font-weight:700; color:{TEXT_MUTED};'>{dupes}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if dupes > 0:
        st.info(f"{dupes} operaciones ya existían y se han omitido automáticamente.")

    if new_df.empty:
        st.success("Todas las operaciones de este PDF ya están en tu historial.")
        return

    # ── Preview table ─────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:{TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:1.2px; margin:16px 0 8px 0;'>"
        f"Vista previa — operaciones a importar</p>",
        unsafe_allow_html=True,
    )

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tickers_available = sorted(new_df["ticker"].unique().tolist())
        sel_tickers = st.multiselect("Filtrar por ticker", tickers_available, default=tickers_available,
                                     key="pdf_filter_ticker")
    with col_f2:
        types_available = sorted(new_df["type"].unique().tolist())
        sel_types = st.multiselect("Filtrar por tipo", types_available, default=types_available,
                                   key="pdf_filter_type")

    preview = new_df[
        new_df["ticker"].isin(sel_tickers) & new_df["type"].isin(sel_types)
    ].copy()

    # Format for display
    disp = pd.DataFrame({
        "Fecha":    preview["date"].apply(lambda d: d.strftime("%d/%m/%Y") if hasattr(d, "strftime") else str(d)),
        "Ticker":   preview["ticker"],
        "Tipo":     preview["type"].apply(lambda t: _TYPE_COLORS.get(t, ("#95a5a6", t))[1]),
        "Cantidad": preview["quantity"].apply(lambda v: f"{v:,.4f}".rstrip("0").rstrip(".")),
        "Precio":   preview["price"].apply(lambda v: f"${v:,.4f}" if v < 10 else f"${v:,.2f}"),
        "Broker":   preview.get("broker", "—"),
    })

    def _color_tipo(v):
        for k, (color, label) in _TYPE_COLORS.items():
            if v == label:
                return f"color: {color}; font-weight: 600"
        return ""

    st.dataframe(
        disp.style.map(_color_tipo, subset=["Tipo"]),
        use_container_width=True,
        height=min(400, 60 + len(disp) * 35),
    )

    st.markdown(f"**{len(preview)}** operaciones seleccionadas para importar.")

    # ── Import button ─────────────────────────────────────────────────────────
    if st.button("Importar operaciones", type="primary", use_container_width=True,
                 key="pdf_import_btn"):
        if preview.empty:
            st.warning("No hay operaciones seleccionadas.")
            return

        prog = st.progress(0, text="Importando...")
        total = len(preview)
        imported = 0
        errors = 0

        for i, (_, row) in enumerate(preview.iterrows()):
            try:
                date_val = row["date"]
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)

                auth.add_transaction(
                    username=username,
                    date=date_str,
                    ticker=str(row["ticker"]),
                    tx_type=str(row["type"]),
                    quantity=float(row["quantity"]),
                    price=float(row["price"]),
                    currency=str(row.get("currency", "USD")),
                    notes=f"PDF import — {row.get('broker', broker_label)}",
                    row_hash=str(row.get("_hash", "")),
                )
                imported += 1
            except Exception as exc:
                log.warning("Error importando fila %d: %s", i, exc)
                errors += 1

            prog.progress((i + 1) / total, text=f"Importando {i+1}/{total}...")

        prog.empty()

        if imported > 0:
            st.success(f"✅ {imported} operaciones importadas correctamente.")
            # Clear price cache so dashboard shows updated positions
            try:
                from modules.transactions import _fetch_prices_for_positions
                _fetch_prices_for_positions.clear()
            except Exception:
                pass
        if errors > 0:
            st.warning(f"⚠ {errors} operaciones no pudieron importarse.")

        st.rerun()

"""
modules/error_handler.py
──────────────────────────────────────────────────────────────────────────────
Manejo centralizado de errores para WealthView.

Provee:
  safe_render(fn, *args, page_name="", **kwargs)
    → Ejecuta fn con args/kwargs. Si lanza excepción, loguea y muestra
      una tarjeta de error amigable al usuario en lugar del stack trace.

  show_error_card(title, detail, suggestion, exception)
    → Renderiza una tarjeta de error estilizada.

  Validadores de inputs:
    validate_ticker(ticker)         → (ok: bool, msg: str, price: float|None)
    validate_positive(val, name)    → (ok: bool, msg: str)
    validate_non_empty(val, name)   → (ok: bool, msg: str)
    validate_date_str(date_str)     → (ok: bool, msg: str)
    validate_percentage(val, name)  → (ok: bool, msg: str)
"""

import traceback
import streamlit as st
from modules.i18n import t
from modules.logger import get_logger

_log = get_logger(__name__)


# ── Tarjeta de error ────────────────────────────────────────────────────────────

def show_error_card(
    title: str = "Algo salió mal",
    detail: str = "",
    suggestion: str = "Recarga la página o revisa la configuración.",
    exception: Exception | None = None,
) -> None:
    """
    Muestra una tarjeta de error estilizada en la UI de Streamlit.
    El stack trace técnico se oculta en un expander opcional.
    """
    st.markdown(f"""
    <div style="background:rgba(231,76,60,0.08);border:1px solid rgba(231,76,60,0.4);
                border-left:4px solid #e74c3c;border-radius:8px;
                padding:16px 20px;margin:12px 0;">
        <div style="font-size:15px;font-weight:700;color:#e74c3c;margin-bottom:6px;">
            ⚠ {title}
        </div>
        <div style="font-size:13px;color:#c9c9c9;margin-bottom:8px;">{detail}</div>
        <div style="font-size:12px;color:#8b95a8;">💡 {suggestion}</div>
    </div>
    """, unsafe_allow_html=True)

    if exception is not None:
        with st.expander(t("error.tech_details"), expanded=False):
            st.code(traceback.format_exc(), language="python")


# ── Renderizado seguro ──────────────────────────────────────────────────────────

def safe_render(fn, *args, page_name: str = "", **kwargs):
    """
    Ejecuta fn(*args, **kwargs) con captura de excepciones.
    Si falla, loguea el error y muestra show_error_card en lugar del stack trace.
    Devuelve el resultado de fn o None si hubo error.
    """
    label = page_name or getattr(fn, "__name__", str(fn))
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        _log.error("Error al renderizar '%s': %s", label, e, exc_info=True)
        show_error_card(
            title=f"Error en {label}",
            detail=str(e) or "Error inesperado al cargar este módulo.",
            suggestion="Si el problema persiste, revisa los logs en Configuración → Logs del Sistema.",
            exception=e,
        )
        return None


# ── Validadores de inputs ───────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> tuple[bool, str, float | None]:
    """
    Valida que el ticker existe en yfinance y tiene precio.
    Devuelve (ok, mensaje_de_error_o_ok, precio_actual).
    Usa un timeout corto para no bloquear la UI.
    """
    if not ticker or not ticker.strip():
        return False, "El ticker no puede estar vacío.", None

    ticker = ticker.strip().upper()

    # Caracteres inválidos
    invalid = set(ticker) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=")
    if invalid:
        return False, f"Ticker contiene caracteres inválidos: {''.join(invalid)}", None

    if len(ticker) > 12:
        return False, "El ticker es demasiado largo (máx. 12 caracteres).", None

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None)
        if price is None or float(price) <= 0:
            return False, f"'{ticker}' no devolvió precio. Verifica que el ticker sea correcto.", None
        return True, f"{ticker} — precio actual: {float(price):,.2f}", float(price)
    except Exception as e:
        _log.warning("validate_ticker('%s') falló: %s", ticker, e)
        return False, f"No se pudo verificar '{ticker}': {e}", None


def validate_positive(val, name: str = "El valor") -> tuple[bool, str]:
    """Valida que val sea un número positivo (> 0)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False, f"{name} debe ser un número."
    if v <= 0:
        return False, f"{name} debe ser mayor que cero."
    return True, ""


def validate_non_negative(val, name: str = "El valor") -> tuple[bool, str]:
    """Valida que val sea un número no negativo (>= 0)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False, f"{name} debe ser un número."
    if v < 0:
        return False, f"{name} no puede ser negativo."
    return True, ""


def validate_non_empty(val, name: str = "El campo") -> tuple[bool, str]:
    """Valida que val no sea None ni cadena vacía."""
    if val is None or str(val).strip() == "":
        return False, f"{name} no puede estar vacío."
    return True, ""


def validate_date_str(date_str: str, name: str = "La fecha") -> tuple[bool, str]:
    """Valida que date_str sea una fecha válida en formato YYYY-MM-DD."""
    if not date_str or not str(date_str).strip():
        return False, f"{name} no puede estar vacía."
    try:
        from datetime import datetime
        datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
        return True, ""
    except ValueError:
        return False, f"{name} debe estar en formato YYYY-MM-DD."


def validate_percentage(val, name: str = "El porcentaje") -> tuple[bool, str]:
    """Valida que val sea un porcentaje razonable (0–100)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False, f"{name} debe ser un número."
    if not (0.0 <= v <= 100.0):
        return False, f"{name} debe estar entre 0 y 100."
    return True, ""


def validate_all(*validations) -> tuple[bool, list[str]]:
    """
    Ejecuta múltiples validaciones y acumula todos los errores.
    Cada elemento de validations es un tuple (ok, msg).
    Devuelve (all_ok, [lista de mensajes de error]).
    """
    errors = [msg for ok, msg in validations if not ok and msg]
    return len(errors) == 0, errors


def show_validation_errors(errors: list[str]) -> None:
    """Muestra una lista de errores de validación de forma compacta."""
    if not errors:
        return
    items = "".join(f"<li>{e}</li>" for e in errors)
    st.markdown(
        f"<div style='background:rgba(231,76,60,0.08);border:1px solid rgba(231,76,60,0.35);"
        f"border-radius:6px;padding:10px 14px;margin:6px 0;'>"
        f"<ul style='margin:0;padding-left:18px;color:#e88080;font-size:13px;'>{items}</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )

"""
modules/logger.py
──────────────────────────────────────────────────────────────────────────────
Logging centralizado para WealthView.

Uso:
    from modules.logger import get_logger
    _log = get_logger(__name__)
    _log.info("Portfolio guardado para %s", username)
    _log.warning("FX fallback a 1.0 para %s/%s", from_ccy, to_ccy)
    _log.error("Error al conectar SMTP: %s", e)

Características:
  - RotatingFileHandler: data/logs/wealthview.log (10 MB × 5 ficheros)
  - StreamHandler en consola solo en modo DEBUG (variable WEALTHVIEW_DEBUG=1)
  - Formato estructurado: timestamp · nivel · módulo · mensaje
  - Un único handler por logger (evita duplicados en reruns de Streamlit)
  - Función tail_log() para mostrar las últimas N líneas en la UI
"""

import os
import logging
import logging.handlers
from typing import Optional

# ── Rutas ───────────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR  = os.path.join(_ROOT, "data", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "wealthview.log")

# ── Formato ─────────────────────────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ── Nivel global ────────────────────────────────────────────────────────────────
_DEBUG_MODE = os.environ.get("WEALTHVIEW_DEBUG", "0") == "1"
_LOG_LEVEL  = logging.DEBUG if _DEBUG_MODE else logging.INFO

# ── Registro de loggers ya configurados (evita duplicar handlers) ────────────────
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado con:
      - RotatingFileHandler → data/logs/wealthview.log
      - StreamHandler       → stdout (solo si WEALTHVIEW_DEBUG=1)

    Idempotente: llamar varias veces con el mismo name no duplica handlers.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False  # evitar que suba al root logger de Streamlit

    # ── File handler (siempre activo) ─────────────────────────────────────────
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(_LOG_LEVEL)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(fh)
    except Exception as e:
        # Si no podemos escribir en disco, al menos logueamos en consola
        logging.getLogger("wealthview.logger_setup").warning(
            "No se pudo crear RotatingFileHandler: %s", e
        )

    # ── Console handler (solo en modo debug) ──────────────────────────────────
    if _DEBUG_MODE:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(ch)

    _configured.add(name)
    return logger


def tail_log(n: int = 100) -> list[str]:
    """
    Devuelve las últimas `n` líneas del fichero de log como lista de strings.
    Útil para mostrar en la UI de settings.
    """
    if not os.path.exists(_LOG_FILE):
        return []
    try:
        with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]
    except Exception:
        return []


def log_file_path() -> str:
    """Devuelve la ruta absoluta del fichero de log actual."""
    return _LOG_FILE


def log_file_size_kb() -> float:
    """Devuelve el tamaño del fichero de log en KB."""
    try:
        return round(os.path.getsize(_LOG_FILE) / 1024, 1)
    except Exception:
        return 0.0


def clear_log() -> bool:
    """Vacía el fichero de log. Devuelve True si tuvo éxito."""
    try:
        open(_LOG_FILE, "w", encoding="utf-8").close()
        return True
    except Exception:
        return False

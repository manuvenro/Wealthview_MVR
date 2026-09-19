"""
modules/scheduler.py
──────────────────────────────────────────────────────────────────────────────
Motor de alertas en background para WealthView.

Usa APScheduler BackgroundScheduler (hilo daemon) que sobrevive a los reruns
de Streamlit porque vive a nivel de módulo Python, no de session_state.

Flujo:
  1. app.py llama a start_scheduler() en el arranque.
  2. El scheduler ejecuta _check_and_notify(username) cada N minutos.
  3. _check_and_notify() comprueba alertas activas, obtiene precios via yfinance
     y envía email si alguna se dispara.
  4. Las alertas disparadas se registran en data/scheduler_log.json para
     evitar reenvíos duplicados dentro de la misma sesión.

Configuración:
  - Intervalo por defecto: 15 minutos (configurable en Settings).
  - Requiere SMTP configurado en data/smtp_config.json.
  - Si no hay SMTP, los precios se comprueban igualmente y se guarda el log.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional
from modules.logger import get_logger

_log = get_logger(__name__)

# ── Rutas ──────────────────────────────────────────────────────────────────────
_DATA_DIR       = "data"
_SMTP_CFG_PATH  = os.path.join(_DATA_DIR, "smtp_config.json")
_SCHED_LOG_PATH = os.path.join(_DATA_DIR, "scheduler_log.json")
_STATE_PATH     = os.path.join(_DATA_DIR, "scheduler_state.json")

# ── Singleton del scheduler ────────────────────────────────────────────────────
_scheduler = None
_scheduler_lock = threading.Lock()

# Intervalo mínimo entre emails para la misma alerta (evita spam)
_ALERT_COOLDOWN_HOURS = 4


# ── Helpers de ficheros ────────────────────────────────────────────────────────

def _load_json(path: str, default) -> dict | list:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path: str, data) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _load_smtp() -> Optional[dict]:
    cfg = _load_json(_SMTP_CFG_PATH, {})
    if cfg.get("enabled") and cfg.get("host") and cfg.get("user") and cfg.get("recipient"):
        return cfg
    return None


# ── Log de alertas enviadas ────────────────────────────────────────────────────

def _load_alert_log() -> dict:
    """Devuelve dict {alert_id: last_sent_iso_str}."""
    return _load_json(_SCHED_LOG_PATH, {})


def _mark_alert_sent(alert_id: str) -> None:
    log = _load_alert_log()
    log[str(alert_id)] = datetime.utcnow().isoformat()
    _save_json(_SCHED_LOG_PATH, log)


def _should_send(alert_id: str) -> bool:
    """True si no se ha enviado o han pasado más de COOLDOWN horas."""
    log = _load_alert_log()
    last_str = log.get(str(alert_id))
    if not last_str:
        return True
    try:
        last = datetime.fromisoformat(last_str)
        return (datetime.utcnow() - last) > timedelta(hours=_ALERT_COOLDOWN_HOURS)
    except Exception:
        return True


# ── Estado del scheduler ────────────────────────────────────────────────────────

def _update_state(key: str, value) -> None:
    state = _load_json(_STATE_PATH, {})
    state[key] = value
    _save_json(_STATE_PATH, state)


def get_scheduler_state() -> dict:
    """Devuelve el estado actual del scheduler para mostrar en la UI."""
    state = _load_json(_STATE_PATH, {})
    state["running"] = _scheduler is not None and _scheduler.running
    return state


# ── Email de alerta ────────────────────────────────────────────────────────────

def _send_alert_email(smtp_cfg: dict, alerts_triggered: list) -> bool:
    """Envía un email HTML con las alertas disparadas."""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        rows_html = ""
        for a in alerts_triggered:
            direction_es = "por encima de" if a["direction"] == "above" else "por debajo de"
            color = "#27ae60" if a["direction"] == "above" else "#e74c3c"
            rows_html += (
                f"<tr>"
                f"<td style='padding:10px 16px;font-weight:700;color:{color};'>{a['ticker']}</td>"
                f"<td style='padding:10px 16px;'>Precio: <b>${a['price']:,.2f}</b></td>"
                f"<td style='padding:10px 16px;'>{direction_es} <b>${a['threshold']:,.2f}</b></td>"
                f"</tr>"
            )

        html = f"""
        <html><body style='font-family:Inter,sans-serif;background:#0e1117;color:#e8eaf6;padding:24px;'>
        <div style='max-width:600px;margin:0 auto;'>
          <div style='font-family:Georgia,serif;font-size:22px;color:#c9a84c;margin-bottom:8px;'>
            WealthView — Alerta de precio
          </div>
          <div style='color:#8b95a8;font-size:12px;margin-bottom:24px;'>
            {datetime.now().strftime('%d/%m/%Y %H:%M')} · Comprobación automática
          </div>
          <table style='width:100%;border-collapse:collapse;background:#161d2b;border-radius:8px;overflow:hidden;'>
            <thead>
              <tr style='background:#1e2a3b;color:#8b95a8;font-size:11px;text-transform:uppercase;letter-spacing:1px;'>
                <th style='padding:10px 16px;text-align:left;'>Ticker</th>
                <th style='padding:10px 16px;text-align:left;'>Precio actual</th>
                <th style='padding:10px 16px;text-align:left;'>Condición</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
          <div style='margin-top:24px;color:#8b95a8;font-size:11px;'>
            Este email fue enviado automáticamente por WealthView.<br>
            Configura tus alertas en la sección <b>Alertas</b> de la aplicación.
          </div>
        </div>
        </body></html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔔 WealthView — {len(alerts_triggered)} alerta(s) activada(s)"
        msg["From"]    = smtp_cfg["user"]
        msg["To"]      = smtp_cfg["recipient"]
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_cfg["host"], int(smtp_cfg.get("port", 587))) as server:
            server.starttls()
            server.login(smtp_cfg["user"], smtp_cfg["password"])
            server.sendmail(smtp_cfg["user"], smtp_cfg["recipient"], msg.as_string())

        return True
    except Exception as e:
        _log.error(f"Error enviando email de alerta: {e}")
        return False


# ── Job principal ──────────────────────────────────────────────────────────────

def _check_and_notify(username: str) -> None:
    """
    Job ejecutado por APScheduler:
    1. Carga alertas activas del usuario.
    2. Obtiene precios via yfinance.
    3. Filtra las disparadas que no estén en cooldown.
    4. Envía email si hay SMTP configurado.
    5. Actualiza el log y el estado.
    """
    _log.info(f"[scheduler] Comprobando alertas para {username}...")
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    _update_state("last_run", now_str)
    _update_state("last_run_user", username)

    try:
        import modules.auth as auth

        alerts_df = auth.load_alerts(username)
        if alerts_df.empty:
            _update_state("last_result", "Sin alertas activas.")
            return

        active = alerts_df[alerts_df["active"] == 1]
        if active.empty:
            _update_state("last_result", "Sin alertas activas.")
            return

        # Batch fetch — una sola petición para todos los tickers activos
        from modules.price_cache import get_prices as _get_prices
        tickers_list = [str(r["ticker"]).upper() for _, r in active.iterrows()]
        prices_map = _get_prices(tickers_list) if tickers_list else {}

        triggered_new = []
        for _, row in active.iterrows():
            try:
                ticker_key = str(row["ticker"]).upper()
                price = float(prices_map.get(ticker_key, 0.0) or 0.0)
                if price == 0.0:
                    continue

                condition_met = (
                    (row["direction"] == "above" and price >= float(row["threshold"])) or
                    (row["direction"] == "below" and price <= float(row["threshold"]))
                )
                if condition_met and _should_send(row["id"]):
                    triggered_new.append({
                        "id":        row["id"],
                        "ticker":    row["ticker"],
                        "direction": row["direction"],
                        "threshold": float(row["threshold"]),
                        "price":     round(price, 2),
                    })
            except Exception as e:
                _log.warning(f"[scheduler] Error comprobando {row.get('ticker','?')}: {e}")
                continue

        if not triggered_new:
            _update_state("last_result", f"Sin disparos nuevos ({len(active)} alertas comprobadas).")
            return

        # Intentar enviar email
        smtp_cfg = _load_smtp()
        email_sent = False
        if smtp_cfg:
            email_sent = _send_alert_email(smtp_cfg, triggered_new)

        # Marcar como enviadas (aunque email falle, para no duplicar en el mismo run)
        for a in triggered_new:
            _mark_alert_sent(a["id"])

        tickers_triggered = ", ".join(a["ticker"] for a in triggered_new)
        result = f"{len(triggered_new)} alerta(s) disparada(s): {tickers_triggered}"
        if smtp_cfg:
            result += " · Email " + ("enviado ✓" if email_sent else "fallido ✗")
        else:
            result += " · SMTP no configurado"
        _update_state("last_result", result)
        _log.info(f"[scheduler] {result}")

    except Exception as e:
        _update_state("last_result", f"Error en job: {e}")
        _log.error(f"[scheduler] Error en _check_and_notify: {e}")


# ── API pública ────────────────────────────────────────────────────────────────

def start_scheduler(username: str, interval_minutes: int = 15) -> bool:
    """
    Arranca el BackgroundScheduler si no está corriendo.
    Seguro de llamar varias veces — solo arranca una instancia.
    Devuelve True si arrancó, False si ya estaba corriendo.
    """
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return False  # ya corriendo

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger

            _scheduler = BackgroundScheduler(
                job_defaults={"coalesce": True, "max_instances": 1},
                timezone="UTC"
            )
            _scheduler.add_job(
                func=_check_and_notify,
                trigger=IntervalTrigger(minutes=interval_minutes),
                args=[username],
                id="price_alerts",
                replace_existing=True,
            )
            _scheduler.start()

            _update_state("started_at", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            _update_state("interval_minutes", interval_minutes)
            _update_state("username", username)
            _update_state("running", True)
            _log.info(f"[scheduler] Arrancado para {username} cada {interval_minutes} min.")
            return True

        except Exception as e:
            _log.error(f"[scheduler] Error al arrancar: {e}")
            _scheduler = None
            return False


def stop_scheduler() -> None:
    """Para el scheduler si está corriendo."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            try:
                _scheduler.shutdown(wait=False)
            except Exception:
                pass
            _scheduler = None
            _update_state("running", False)
            _log.info("[scheduler] Detenido.")


def update_interval(username: str, interval_minutes: int) -> bool:
    """Cambia el intervalo de comprobación sin reiniciar el scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None or not _scheduler.running:
            return start_scheduler(username, interval_minutes)
        try:
            from apscheduler.triggers.interval import IntervalTrigger
            _scheduler.reschedule_job(
                "price_alerts",
                trigger=IntervalTrigger(minutes=interval_minutes)
            )
            _update_state("interval_minutes", interval_minutes)
            _log.info(f"[scheduler] Intervalo cambiado a {interval_minutes} min.")
            return True
        except Exception as e:
            _log.error(f"[scheduler] Error cambiando intervalo: {e}")
            return False


def run_now(username: str) -> str:
    """Ejecuta el job inmediatamente (para prueba manual desde la UI)."""
    try:
        _check_and_notify(username)
        state = get_scheduler_state()
        return state.get("last_result", "Comprobación completada.")
    except Exception as e:
        return f"Error: {e}"

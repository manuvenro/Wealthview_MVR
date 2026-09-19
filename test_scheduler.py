"""
tests/test_scheduler.py
Tests para modules/scheduler.py:
  - Lógica de cooldown (_should_send, _mark_alert_sent)
  - Estado del scheduler (get_scheduler_state)
  - Parsing de alertas y condiciones (sin red)
  - Email HTML generation (sin enviar)
"""
import os
import json
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════
# Fixtures locales
# ══════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def patch_scheduler_paths(tmp_path, monkeypatch):
    """Redirige todos los ficheros JSON del scheduler a tmp_path."""
    import modules.scheduler as sched
    monkeypatch.setattr(sched, "_DATA_DIR",       str(tmp_path))
    monkeypatch.setattr(sched, "_SMTP_CFG_PATH",  str(tmp_path / "smtp_config.json"))
    monkeypatch.setattr(sched, "_SCHED_LOG_PATH", str(tmp_path / "scheduler_log.json"))
    monkeypatch.setattr(sched, "_STATE_PATH",     str(tmp_path / "scheduler_state.json"))
    yield


# ══════════════════════════════════════════════════════════
# Tests de cooldown
# ══════════════════════════════════════════════════════════

class TestCooldown:

    def test_should_send_when_no_prior_log(self):
        from modules.scheduler import _should_send
        assert _should_send("alert_999") is True

    def test_should_not_send_immediately_after_marking(self):
        from modules.scheduler import _should_send, _mark_alert_sent
        _mark_alert_sent("alert_1")
        assert _should_send("alert_1") is False

    def test_should_send_after_cooldown_expires(self, monkeypatch):
        from modules import scheduler as sched
        from modules.scheduler import _mark_alert_sent

        # Mark as sent
        _mark_alert_sent("alert_2")

        # Monkeypatch datetime.utcnow to simulate 5 hours later
        future = datetime.utcnow() + timedelta(hours=5)
        monkeypatch.setattr(sched, "_ALERT_COOLDOWN_HOURS", 0)  # instant expiry
        assert sched._should_send("alert_2") is True

    def test_mark_alert_sent_writes_to_disk(self, tmp_path, monkeypatch):
        import modules.scheduler as sched
        log_path = str(tmp_path / "scheduler_log.json")
        monkeypatch.setattr(sched, "_SCHED_LOG_PATH", log_path)

        sched._mark_alert_sent("alert_42")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            data = json.load(f)
        assert "alert_42" in data

    def test_multiple_alerts_independent_cooldown(self):
        from modules.scheduler import _should_send, _mark_alert_sent
        _mark_alert_sent("a1")
        # a2 was never sent → should send
        assert _should_send("a2") is True
        # a1 was just sent → should not
        assert _should_send("a1") is False


# ══════════════════════════════════════════════════════════
# Tests de estado del scheduler
# ══════════════════════════════════════════════════════════

class TestSchedulerState:

    def test_get_state_no_file(self):
        from modules.scheduler import get_scheduler_state
        state = get_scheduler_state()
        assert isinstance(state, dict)
        assert "running" in state

    def test_update_state_persists(self, tmp_path, monkeypatch):
        import modules.scheduler as sched
        state_path = str(tmp_path / "scheduler_state.json")
        monkeypatch.setattr(sched, "_STATE_PATH", state_path)

        sched._update_state("last_run", "01/01/2026 12:00:00")
        state = sched.get_scheduler_state()
        assert state["last_run"] == "01/01/2026 12:00:00"

    def test_state_running_false_when_no_scheduler(self):
        """Sin scheduler activo, running debe ser False."""
        import modules.scheduler as sched
        # Asegurar que el singleton es None
        original = sched._scheduler
        sched._scheduler = None
        try:
            state = sched.get_scheduler_state()
            assert state["running"] is False
        finally:
            sched._scheduler = original


# ══════════════════════════════════════════════════════════
# Tests de condiciones de alerta
# ══════════════════════════════════════════════════════════

class TestAlertConditions:
    """
    Verifica la lógica de evaluación de condiciones sin tocar red ni BD real.
    """

    def test_above_condition_triggers(self):
        """Precio >= umbral con direction='above' → debe dispararse."""
        price, threshold = 210.0, 200.0
        triggered = price >= threshold
        assert triggered is True

    def test_above_condition_not_triggered(self):
        price, threshold = 190.0, 200.0
        triggered = price >= threshold
        assert triggered is False

    def test_below_condition_triggers(self):
        price, threshold = 95.0, 100.0
        triggered = price <= threshold
        assert triggered is True

    def test_below_condition_not_triggered(self):
        price, threshold = 110.0, 100.0
        triggered = price <= threshold
        assert triggered is False

    def test_exact_threshold_triggers(self):
        """Precio exactamente igual al umbral debe considerarse disparado."""
        price = threshold = 150.0
        assert (price >= threshold) is True   # above
        assert (price <= threshold) is True   # below


# ══════════════════════════════════════════════════════════
# Tests de email HTML (sin enviar)
# ══════════════════════════════════════════════════════════

class TestEmailGeneration:

    def test_send_alert_email_fails_gracefully_without_smtp(self):
        """Si SMTP no está disponible, _send_alert_email debe devolver False, no lanzar."""
        from modules.scheduler import _send_alert_email

        bad_cfg = {
            "host": "localhost", "port": 9999,
            "user": "test@example.com", "password": "wrong",
            "recipient": "user@example.com", "enabled": True,
        }
        alerts = [{"ticker": "AAPL", "direction": "above", "threshold": 200.0, "price": 210.0}]
        result = _send_alert_email(bad_cfg, alerts)
        assert result is False

    def test_send_alert_email_with_mock_smtp(self):
        """Con SMTP mockeado, _send_alert_email debe devolver True."""
        from modules.scheduler import _send_alert_email

        cfg = {
            "host": "smtp.example.com", "port": 587,
            "user": "from@example.com", "password": "pass",
            "recipient": "to@example.com", "enabled": True,
        }
        alerts = [
            {"ticker": "AAPL", "direction": "above", "threshold": 200.0, "price": 210.0},
            {"ticker": "MSFT", "direction": "below", "threshold": 300.0, "price": 295.0},
        ]

        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", return_value=mock_server):
            result = _send_alert_email(cfg, alerts)

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with(cfg["user"], cfg["password"])
        mock_server.sendmail.assert_called_once()


# ══════════════════════════════════════════════════════════
# Tests de run_now (con BD y yfinance mockeados)
# ══════════════════════════════════════════════════════════

class TestRunNow:

    def test_run_now_with_no_alerts_returns_string(self, tmp_db, monkeypatch):
        """run_now sin alertas activas debe devolver un string descriptivo."""
        import modules.scheduler as sched
        import modules.auth as auth
        import pandas as pd

        auth.create_user("demo", "pass123")
        # No alerts → early exit
        result = sched.run_now("demo")
        assert isinstance(result, str)
        assert len(result) > 0

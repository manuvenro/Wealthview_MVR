"""
tests/test_auth.py
Tests para modules/auth.py:
  - Registro y login de usuarios
  - CRUD del portfolio (incluyendo avg_cost y currency)
  - Backup automático: backup_portfolio, list_backups, restore_backup
  - save_portfolio dispara backup automáticamente
"""
import os
import pytest
import pandas as pd
import modules.auth as auth


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

USERNAME = "testuser"
PASSWORD = "secret123"


def _register_and_login(username=USERNAME, password=PASSWORD):
    ok = auth.create_user(username, password)
    assert ok, "create_user debe retornar True al crear usuario nuevo"
    ok, _ = auth.login_user(username, password)
    assert ok, "login debe funcionar con contraseña correcta"
    return username


# ══════════════════════════════════════════════════════════
# Tests de autenticación
# ══════════════════════════════════════════════════════════

class TestAuth:

    def test_create_user_ok(self, tmp_db):
        assert auth.create_user(USERNAME, PASSWORD) is True

    def test_create_user_duplicate(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        assert auth.create_user(USERNAME, PASSWORD) is False

    def test_login_correct_password(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        ok, _ = auth.login_user(USERNAME, PASSWORD)
        assert ok is True

    def test_login_wrong_password(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        ok, _ = auth.login_user(USERNAME, "wrongpass")
        assert ok is False

    def test_login_nonexistent_user(self, tmp_db):
        ok, _ = auth.login_user("nobody", "pass")
        assert ok is False


# ══════════════════════════════════════════════════════════
# Tests de portfolio CRUD
# ══════════════════════════════════════════════════════════

class TestPortfolio:

    def test_save_and_load_portfolio(self, tmp_db, tmp_backup_dir, sample_portfolio):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_portfolio(USERNAME, sample_portfolio)
        loaded = auth.load_portfolio(USERNAME)
        assert not loaded.empty
        assert set(loaded["Ticker"].tolist()) == {"AAPL", "MSFT"}

    def test_load_empty_portfolio(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        df = auth.load_portfolio(USERNAME)
        assert df.empty

    def test_avg_cost_persists(self, tmp_db, tmp_backup_dir, sample_portfolio):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_portfolio(USERNAME, sample_portfolio)
        loaded = auth.load_portfolio(USERNAME)
        assert "Avg Cost" in loaded.columns
        aapl_row = loaded[loaded["Ticker"] == "AAPL"].iloc[0]
        assert abs(aapl_row["Avg Cost"] - 175.0) < 0.01

    def test_currency_persists(self, tmp_db, tmp_backup_dir, sample_portfolio):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_portfolio(USERNAME, sample_portfolio)
        loaded = auth.load_portfolio(USERNAME)
        assert "Currency" in loaded.columns
        assert (loaded["Currency"] == "USD").all()

    def test_overwrite_portfolio(self, tmp_db, tmp_backup_dir, sample_portfolio):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_portfolio(USERNAME, sample_portfolio)
        # Save only one ticker
        single = sample_portfolio[sample_portfolio["Ticker"] == "AAPL"].copy()
        auth.save_portfolio(USERNAME, single)
        loaded = auth.load_portfolio(USERNAME)
        assert len(loaded) == 1
        assert loaded.iloc[0]["Ticker"] == "AAPL"


# ══════════════════════════════════════════════════════════
# Tests de backup
# ══════════════════════════════════════════════════════════

class TestBackup:

    def test_backup_creates_file(self, tmp_backup_dir, sample_portfolio):
        path = auth.backup_portfolio(USERNAME, sample_portfolio)
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_backup_empty_portfolio_returns_none(self, tmp_backup_dir):
        result = auth.backup_portfolio(USERNAME, pd.DataFrame())
        assert result is None

    def test_backup_none_returns_none(self, tmp_backup_dir):
        result = auth.backup_portfolio(USERNAME, None)
        assert result is None

    def test_backup_prunes_old_files(self, tmp_backup_dir, sample_portfolio):
        # Create 5 backups with max=3 → only 3 should remain
        for _ in range(5):
            auth.backup_portfolio(USERNAME, sample_portfolio, max_backups=3)
        backups = auth.list_backups(USERNAME)
        assert len(backups) <= 3

    def test_list_backups_empty(self, tmp_backup_dir):
        result = auth.list_backups("nobody")
        assert result == []

    def test_list_backups_returns_metadata(self, tmp_backup_dir, sample_portfolio):
        auth.backup_portfolio(USERNAME, sample_portfolio)
        backups = auth.list_backups(USERNAME)
        assert len(backups) == 1
        bk = backups[0]
        assert "filename" in bk
        assert "timestamp_str" in bk
        assert bk["n_positions"] == 2
        assert "AAPL" in bk["tickers"] or "MSFT" in bk["tickers"]
        assert bk["size_kb"] > 0

    def test_list_backups_sorted_newest_first(self, tmp_backup_dir, sample_portfolio):
        import time
        auth.backup_portfolio(USERNAME, sample_portfolio)
        time.sleep(1.1)
        auth.backup_portfolio(USERNAME, sample_portfolio)
        backups = auth.list_backups(USERNAME)
        assert len(backups) == 2
        # Newest first → filename with higher timestamp comes first
        assert backups[0]["filename"] > backups[1]["filename"]

    def test_restore_backup_roundtrip(self, tmp_backup_dir, sample_portfolio):
        auth.backup_portfolio(USERNAME, sample_portfolio)
        backups = auth.list_backups(USERNAME)
        restored = auth.restore_backup(USERNAME, backups[0]["filename"])
        assert restored is not None
        assert set(restored["Ticker"].tolist()) == {"AAPL", "MSFT"}

    def test_restore_missing_backup_returns_none(self, tmp_backup_dir):
        result = auth.restore_backup(USERNAME, "nonexistent.json")
        assert result is None

    def test_save_portfolio_auto_backup(self, tmp_db, tmp_backup_dir, sample_portfolio):
        """save_portfolio debe crear automáticamente un backup antes de escribir."""
        auth.create_user(USERNAME, PASSWORD)
        auth.save_portfolio(USERNAME, sample_portfolio)
        backups = auth.list_backups(USERNAME)
        assert len(backups) >= 1, "save_portfolio debe generar al menos un backup"


# ══════════════════════════════════════════════════════════
# Tests de alertas
# ══════════════════════════════════════════════════════════

class TestAlerts:

    def test_save_and_load_alert(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_alert(USERNAME, "AAPL", "above", 200.0)
        df = auth.load_alerts(USERNAME)
        assert not df.empty
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["direction"] == "above"
        assert abs(df.iloc[0]["threshold"] - 200.0) < 0.01

    def test_delete_alert(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_alert(USERNAME, "AAPL", "above", 200.0)
        df = auth.load_alerts(USERNAME)
        alert_id = df.iloc[0]["id"]
        auth.delete_alert(alert_id)
        df2 = auth.load_alerts(USERNAME)
        assert df2.empty

    def test_multiple_alerts(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_alert(USERNAME, "AAPL", "above", 200.0)
        auth.save_alert(USERNAME, "MSFT", "below", 300.0)
        df = auth.load_alerts(USERNAME)
        assert len(df) == 2


# ══════════════════════════════════════════════════════════
# Tests de watchlist
# ══════════════════════════════════════════════════════════

class TestWatchlist:

    def test_save_and_load_watchlist(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_watchlist(USERNAME, ["AAPL", "GOOGL"])
        tickers = auth.load_watchlist(USERNAME)
        assert set(tickers) == {"AAPL", "GOOGL"}

    def test_watchlist_overwrite(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        auth.save_watchlist(USERNAME, ["AAPL", "GOOGL"])
        auth.save_watchlist(USERNAME, ["TSLA"])
        tickers = auth.load_watchlist(USERNAME)
        assert tickers == ["TSLA"]

    def test_empty_watchlist(self, tmp_db):
        auth.create_user(USERNAME, PASSWORD)
        tickers = auth.load_watchlist(USERNAME)
        assert tickers == []

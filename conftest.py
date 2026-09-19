"""
tests/conftest.py
Fixtures compartidas para la suite de tests de WealthView.
"""
import os
import sys
import tempfile
import pytest
import pandas as pd

# ── Asegurarse de que el root del proyecto esté en el path ─────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Fixture: base de datos SQLite temporal ─────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """
    Redirige modules.auth a una DB SQLite temporal para cada test.
    La DB se elimina automáticamente al final del test.
    """
    import modules.auth as auth
    db_file = str(tmp_path / "test_wealthview.db")
    monkeypatch.setattr(auth, "DB_PATH", db_file)
    auth.init_db()
    yield db_file


# ── Fixture: directorio de backups temporal ────────────────────────────────────

@pytest.fixture
def tmp_backup_dir(tmp_path, monkeypatch):
    """Redirige el directorio de backups a un path temporal."""
    import modules.auth as auth
    backup_dir = str(tmp_path / "backups")
    monkeypatch.setattr(auth, "_BACKUP_DIR_BASE", backup_dir)
    yield backup_dir


# ── Fixture: portfolio de muestra ──────────────────────────────────────────────

@pytest.fixture
def sample_portfolio():
    """DataFrame de portfolio mínimo válido para tests."""
    return pd.DataFrame([
        {
            "Ticker": "AAPL", "Shares": 10.0, "Asset Type": "equity",
            "Avg Cost": 175.0, "Currency": "USD",
            "Face Value": None, "Coupon %": None, "Maturity": None,
            "Name": "Apple Inc.", "Strike": None, "Multiplier": None,
            "Option Type": None, "Premium": None,
        },
        {
            "Ticker": "MSFT", "Shares": 5.0, "Asset Type": "equity",
            "Avg Cost": 310.0, "Currency": "USD",
            "Face Value": None, "Coupon %": None, "Maturity": None,
            "Name": "Microsoft Corp.", "Strike": None, "Multiplier": None,
            "Option Type": None, "Premium": None,
        },
    ])


# ── Fixture: config de usuario temporal ───────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """
    Crea un directorio data/ temporal y parchea las rutas de fx.py y scheduler.py
    para que usen esa ubicación.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield str(data_dir)

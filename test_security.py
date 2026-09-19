"""
tests/test_security.py
Tests para el módulo de seguridad: bcrypt, rate limiting, lockout,
change_password, get_last_login, session expiration.
"""
import pytest
import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_db(tmp_path, monkeypatch):
    """Redirige la base de datos a un fichero temporal para cada test."""
    db_file = str(tmp_path / "test_auth.db")
    import modules.auth as auth
    monkeypatch.setattr(auth, "DB_PATH", db_file)
    auth.init_db()
    yield


@pytest.fixture()
def auth():
    import modules.auth as _auth
    return _auth


@pytest.fixture()
def user(auth):
    """Crea un usuario de prueba y devuelve (username, password)."""
    username, password = "testuser", "SecurePass123"
    auth.create_user(username, password)
    return username, password


# ─────────────────────────────────────────────────────────────
# 1. Hashing bcrypt
# ─────────────────────────────────────────────────────────────

class TestBcrypt:
    def test_hash_is_bcrypt_format(self, auth):
        h = auth.hash_password("mypassword")
        assert h.startswith("$2")

    def test_verify_correct_password(self, auth):
        h = auth.hash_password("correct")
        assert auth.verify_password("correct", h) is True

    def test_verify_wrong_password(self, auth):
        h = auth.hash_password("correct")
        assert auth.verify_password("wrong", h) is False

    def test_different_salts(self, auth):
        h1 = auth.hash_password("same")
        h2 = auth.hash_password("same")
        assert h1 != h2  # salts diferentes

    def test_verify_legacy_sha256(self, auth):
        """Contraseñas SHA-256 legacy deben verificarse correctamente."""
        import hashlib
        password = "legacypass"
        sha_hash = hashlib.sha256(password.encode()).hexdigest()
        assert len(sha_hash) == 64
        assert not sha_hash.startswith("$2")
        assert auth.verify_password(password, sha_hash) is True

    def test_reject_wrong_legacy_sha256(self, auth):
        import hashlib
        sha_hash = hashlib.sha256("correct".encode()).hexdigest()
        assert auth.verify_password("wrong", sha_hash) is False


# ─────────────────────────────────────────────────────────────
# 2. Login básico
# ─────────────────────────────────────────────────────────────

class TestLogin:
    def test_login_success(self, auth, user):
        username, password = user
        ok, msg = auth.login_user(username, password)
        assert ok is True
        assert msg == ""

    def test_login_wrong_password(self, auth, user):
        username, _ = user
        ok, msg = auth.login_user(username, "WrongPass!")
        assert ok is False
        assert len(msg) > 0

    def test_login_nonexistent_user(self, auth):
        ok, msg = auth.login_user("nobody", "whatever")
        assert ok is False

    def test_login_returns_tuple(self, auth, user):
        username, password = user
        result = auth.login_user(username, password)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────
# 3. Rate limiting y lockout
# ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_failed_attempts_increments(self, auth, user):
        username, _ = user
        for _ in range(3):
            auth.login_user(username, "bad")
        assert auth.get_failed_attempts(username) == 3

    def test_no_attempts_initially(self, auth, user):
        username, _ = user
        assert auth.get_failed_attempts(username) == 0

    def test_lockout_after_max_attempts(self, auth, user):
        username, _ = user
        max_attempts = auth._MAX_FAILED_ATTEMPTS
        for _ in range(max_attempts):
            auth.login_user(username, "bad")
        locked, until_str = auth._is_locked(username)
        assert locked is True
        assert until_str != ""

    def test_locked_user_cannot_login(self, auth, user):
        username, password = user
        max_attempts = auth._MAX_FAILED_ATTEMPTS
        for _ in range(max_attempts):
            auth.login_user(username, "bad")
        ok, msg = auth.login_user(username, password)
        assert ok is False
        assert "bloqueada" in msg.lower() or "locked" in msg.lower()

    def test_error_message_shows_remaining(self, auth, user):
        username, _ = user
        ok, msg = auth.login_user(username, "bad")
        assert ok is False
        # Primer fallo: quedan _MAX_FAILED_ATTEMPTS - 1 intentos
        remaining = auth._MAX_FAILED_ATTEMPTS - 1
        assert str(remaining) in msg or "intento" in msg.lower()

    def test_successful_login_clears_failed_attempts(self, auth, user):
        username, password = user
        auth.login_user(username, "bad")
        auth.login_user(username, "bad")
        auth.login_user(username, password)  # login exitoso
        assert auth.get_failed_attempts(username) == 0

    def test_not_locked_initially(self, auth, user):
        username, _ = user
        locked, _ = auth._is_locked(username)
        assert locked is False


# ─────────────────────────────────────────────────────────────
# 4. unlock_user
# ─────────────────────────────────────────────────────────────

class TestUnlockUser:
    def test_unlock_clears_lockout(self, auth, user):
        username, _ = user
        for _ in range(auth._MAX_FAILED_ATTEMPTS):
            auth.login_user(username, "bad")
        locked, _ = auth._is_locked(username)
        assert locked is True

        auth.unlock_user(username)
        locked_after, _ = auth._is_locked(username)
        assert locked_after is False

    def test_unlock_resets_failed_count(self, auth, user):
        username, _ = user
        for _ in range(auth._MAX_FAILED_ATTEMPTS):
            auth.login_user(username, "bad")
        auth.unlock_user(username)
        assert auth.get_failed_attempts(username) == 0

    def test_unlock_allows_login(self, auth, user):
        username, password = user
        for _ in range(auth._MAX_FAILED_ATTEMPTS):
            auth.login_user(username, "bad")
        auth.unlock_user(username)
        ok, _ = auth.login_user(username, password)
        assert ok is True


# ─────────────────────────────────────────────────────────────
# 5. change_password
# ─────────────────────────────────────────────────────────────

class TestChangePassword:
    def test_change_password_success(self, auth, user):
        username, password = user
        ok, msg = auth.change_password(username, password, "NewPass456!")
        assert ok is True
        assert msg == ""

    def test_new_password_works_after_change(self, auth, user):
        username, password = user
        auth.change_password(username, password, "NewPass456!")
        ok, _ = auth.login_user(username, "NewPass456!")
        assert ok is True

    def test_old_password_fails_after_change(self, auth, user):
        username, password = user
        auth.change_password(username, password, "NewPass456!")
        ok, _ = auth.login_user(username, password)
        assert ok is False

    def test_change_password_wrong_old(self, auth, user):
        username, _ = user
        ok, msg = auth.change_password(username, "WrongOld", "NewPass456!")
        assert ok is False
        assert len(msg) > 0

    def test_change_password_nonexistent_user(self, auth):
        ok, msg = auth.change_password("ghost", "any", "new")
        assert ok is False

    def test_change_password_returns_tuple(self, auth, user):
        username, password = user
        result = auth.change_password(username, password, "NewPass456!")
        assert isinstance(result, tuple) and len(result) == 2


# ─────────────────────────────────────────────────────────────
# 6. get_last_login
# ─────────────────────────────────────────────────────────────

class TestLastLogin:
    def test_last_login_none_before_login(self, auth, user):
        username, _ = user
        result = auth.get_last_login(username)
        # Puede ser None o vacío antes de cualquier login
        assert result is None or result == ""

    def test_last_login_set_after_login(self, auth, user):
        username, password = user
        auth.login_user(username, password)
        result = auth.get_last_login(username)
        assert result is not None and result != ""

    def test_last_login_nonexistent_user(self, auth):
        result = auth.get_last_login("nobody")
        assert result is None


# ─────────────────────────────────────────────────────────────
# 7. get_login_history
# ─────────────────────────────────────────────────────────────

class TestLoginHistory:
    def test_empty_history_initially(self, auth, user):
        username, _ = user
        history = auth.get_login_history(username)
        assert isinstance(history, list)

    def test_history_records_success(self, auth, user):
        username, password = user
        auth.login_user(username, password)
        history = auth.get_login_history(username)
        assert any(h["success"] for h in history)

    def test_history_records_failure(self, auth, user):
        username, _ = user
        auth.login_user(username, "bad")
        history = auth.get_login_history(username)
        assert any(not h["success"] for h in history)

    def test_history_limit_respected(self, auth, user):
        username, password = user
        for _ in range(5):
            auth.login_user(username, password)
        history = auth.get_login_history(username, limit=3)
        assert len(history) <= 3


# ─────────────────────────────────────────────────────────────
# 8. Constantes de seguridad
# ─────────────────────────────────────────────────────────────

class TestSecurityConstants:
    def test_max_failed_attempts_positive(self, auth):
        assert auth._MAX_FAILED_ATTEMPTS > 0

    def test_lockout_minutes_positive(self, auth):
        assert auth._LOCKOUT_MINUTES > 0

    def test_session_timeout_positive(self, auth):
        assert auth._SESSION_TIMEOUT_HOURS > 0

    def test_session_timeout_reasonable(self, auth):
        # Entre 1 y 24 horas
        assert 1 <= auth._SESSION_TIMEOUT_HOURS <= 24

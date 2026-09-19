"""
tests/test_error_handler.py
Tests para modules/error_handler.py:
  - Validadores de inputs (sin red)
  - validate_all combinator
  - safe_render captura excepciones sin relanzarlas
"""
import pytest
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════
# Tests de validate_positive
# ══════════════════════════════════════════════════════════

class TestValidatePositive:

    def test_positive_float(self):
        from modules.error_handler import validate_positive
        ok, msg = validate_positive(10.5, "Shares")
        assert ok is True
        assert msg == ""

    def test_zero_fails(self):
        from modules.error_handler import validate_positive
        ok, msg = validate_positive(0, "Shares")
        assert ok is False
        assert "mayor que cero" in msg

    def test_negative_fails(self):
        from modules.error_handler import validate_positive
        ok, msg = validate_positive(-1.0, "Shares")
        assert ok is False

    def test_non_numeric_fails(self):
        from modules.error_handler import validate_positive
        ok, msg = validate_positive("abc", "Shares")
        assert ok is False
        assert "número" in msg


# ══════════════════════════════════════════════════════════
# Tests de validate_non_negative
# ══════════════════════════════════════════════════════════

class TestValidateNonNegative:

    def test_zero_ok(self):
        from modules.error_handler import validate_non_negative
        ok, _ = validate_non_negative(0.0, "Commission")
        assert ok is True

    def test_positive_ok(self):
        from modules.error_handler import validate_non_negative
        ok, _ = validate_non_negative(5.0, "Commission")
        assert ok is True

    def test_negative_fails(self):
        from modules.error_handler import validate_non_negative
        ok, msg = validate_non_negative(-0.01, "Commission")
        assert ok is False
        assert "negativo" in msg


# ══════════════════════════════════════════════════════════
# Tests de validate_non_empty
# ══════════════════════════════════════════════════════════

class TestValidateNonEmpty:

    def test_non_empty_string(self):
        from modules.error_handler import validate_non_empty
        ok, _ = validate_non_empty("AAPL", "Ticker")
        assert ok is True

    def test_empty_string_fails(self):
        from modules.error_handler import validate_non_empty
        ok, msg = validate_non_empty("", "Ticker")
        assert ok is False
        assert "vacío" in msg

    def test_whitespace_fails(self):
        from modules.error_handler import validate_non_empty
        ok, msg = validate_non_empty("   ", "Ticker")
        assert ok is False

    def test_none_fails(self):
        from modules.error_handler import validate_non_empty
        ok, msg = validate_non_empty(None, "Campo")
        assert ok is False


# ══════════════════════════════════════════════════════════
# Tests de validate_date_str
# ══════════════════════════════════════════════════════════

class TestValidateDateStr:

    def test_valid_date(self):
        from modules.error_handler import validate_date_str
        ok, _ = validate_date_str("2026-12-31")
        assert ok is True

    def test_invalid_format(self):
        from modules.error_handler import validate_date_str
        ok, msg = validate_date_str("31/12/2026")
        assert ok is False
        assert "YYYY-MM-DD" in msg

    def test_empty_date(self):
        from modules.error_handler import validate_date_str
        ok, _ = validate_date_str("")
        assert ok is False

    def test_invalid_date_values(self):
        from modules.error_handler import validate_date_str
        ok, _ = validate_date_str("2026-13-01")   # mes 13
        assert ok is False


# ══════════════════════════════════════════════════════════
# Tests de validate_percentage
# ══════════════════════════════════════════════════════════

class TestValidatePercentage:

    def test_valid_percentage(self):
        from modules.error_handler import validate_percentage
        ok, _ = validate_percentage(45.5)
        assert ok is True

    def test_zero_ok(self):
        from modules.error_handler import validate_percentage
        ok, _ = validate_percentage(0.0)
        assert ok is True

    def test_hundred_ok(self):
        from modules.error_handler import validate_percentage
        ok, _ = validate_percentage(100.0)
        assert ok is True

    def test_over_hundred_fails(self):
        from modules.error_handler import validate_percentage
        ok, msg = validate_percentage(100.1)
        assert ok is False

    def test_negative_fails(self):
        from modules.error_handler import validate_percentage
        ok, _ = validate_percentage(-1.0)
        assert ok is False


# ══════════════════════════════════════════════════════════
# Tests de validate_ticker (con yfinance mockeado)
# ══════════════════════════════════════════════════════════

class TestValidateTicker:

    def _mock_fast_info(self, price):
        m = MagicMock()
        m.fast_info.last_price = price
        return m

    def test_empty_ticker_fails(self):
        from modules.error_handler import validate_ticker
        ok, msg, price = validate_ticker("")
        assert ok is False
        assert price is None

    def test_ticker_too_long_fails(self):
        from modules.error_handler import validate_ticker
        ok, msg, price = validate_ticker("A" * 15)
        assert ok is False

    def test_ticker_invalid_chars_fails(self):
        from modules.error_handler import validate_ticker
        ok, msg, price = validate_ticker("AAP L!")
        assert ok is False

    def test_valid_ticker_with_price(self):
        from modules.error_handler import validate_ticker
        with patch("yfinance.Ticker", return_value=self._mock_fast_info(180.0)):
            ok, msg, price = validate_ticker("AAPL")
        assert ok is True
        assert price == 180.0
        assert "180" in msg

    def test_ticker_zero_price_fails(self):
        from modules.error_handler import validate_ticker
        with patch("yfinance.Ticker", return_value=self._mock_fast_info(0.0)):
            ok, msg, price = validate_ticker("FAKE")
        assert ok is False
        assert price is None

    def test_ticker_network_error_fails(self):
        from modules.error_handler import validate_ticker
        with patch("yfinance.Ticker", side_effect=ConnectionError("timeout")):
            ok, msg, price = validate_ticker("AAPL")
        assert ok is False
        assert price is None


# ══════════════════════════════════════════════════════════
# Tests de validate_all
# ══════════════════════════════════════════════════════════

class TestValidateAll:

    def test_all_pass(self):
        from modules.error_handler import validate_all
        ok, errors = validate_all(
            (True, ""),
            (True, ""),
            (True, ""),
        )
        assert ok is True
        assert errors == []

    def test_one_fails(self):
        from modules.error_handler import validate_all
        ok, errors = validate_all(
            (True, ""),
            (False, "Campo inválido"),
            (True, ""),
        )
        assert ok is False
        assert "Campo inválido" in errors

    def test_multiple_fail(self):
        from modules.error_handler import validate_all
        ok, errors = validate_all(
            (False, "Error 1"),
            (False, "Error 2"),
        )
        assert ok is False
        assert len(errors) == 2


# ══════════════════════════════════════════════════════════
# Tests de safe_render
# ══════════════════════════════════════════════════════════

class TestSafeRender:

    def test_returns_fn_result_on_success(self, monkeypatch):
        from modules.error_handler import safe_render
        result = safe_render(lambda: 42, page_name="test")
        assert result == 42

    def test_catches_exception_returns_none(self, monkeypatch):
        """safe_render no debe propagar excepciones."""
        import streamlit as st

        # Mock st.markdown and st.expander so they don't fail in test context
        monkeypatch.setattr(st, "markdown", lambda *a, **kw: None)
        monkeypatch.setattr(st, "expander", MagicMock())
        monkeypatch.setattr(st, "code", lambda *a, **kw: None)

        def bad_fn():
            raise RuntimeError("something broke")

        from modules.error_handler import safe_render
        result = safe_render(bad_fn, page_name="BadPage")
        assert result is None  # no exception propagated

    def test_passes_args_to_fn(self):
        from modules.error_handler import safe_render
        result = safe_render(lambda x, y: x + y, 3, 4, page_name="math")
        assert result == 7

    def test_passes_kwargs_to_fn(self):
        from modules.error_handler import safe_render
        result = safe_render(lambda x, mul=1: x * mul, 5, mul=3, page_name="math")
        assert result == 15

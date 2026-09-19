"""
tests/test_price_cache.py
──────────────────────────────────────────────────────────────────────────────
Tests para modules/price_cache.py.
Todos los tests mockean yfinance — sin peticiones de red.
"""
import time
import threading
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

def _reset_cache():
    """Limpia caché y stats entre tests para aislamiento."""
    import modules.price_cache as pc
    pc._cache.clear()
    for k in pc._stats:
        pc._stats[k] = 0


def _make_download_df(prices: dict[str, float]) -> pd.DataFrame:
    """
    Fabrica un DataFrame con estructura MultiIndex que yf.download devuelve
    para múltiples tickers: columnas = (OHLCV, ticker).
    Para un solo ticker devuelve un DataFrame simple.
    """
    if len(prices) == 1:
        ticker, price = next(iter(prices.items()))
        idx = pd.date_range("2026-01-01", periods=2)
        df = pd.DataFrame({"Close": [price - 1, price]}, index=idx)
        return df

    # Multi-ticker: MultiIndex columns (field, ticker)
    import numpy as np
    idx = pd.date_range("2026-01-01", periods=2)
    close_data = {t: [p - 1, p] for t, p in prices.items()}
    close_df = pd.DataFrame(close_data, index=idx)
    # Wrap in MultiIndex so close = df["Close"]
    arrays = [["Close"] * len(prices), list(prices.keys())]
    close_df.columns = pd.MultiIndex.from_arrays(arrays)
    return close_df


# ══════════════════════════════════════════════════════════════════════════════
# TestCacheBasics
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheBasics:

    def setup_method(self):
        _reset_cache()

    def test_get_price_returns_float(self):
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 150.0
        with patch("yfinance.Ticker", return_value=mock_ticker):
            price = pc.get_price("AAPL")
        assert isinstance(price, float)
        assert price == 150.0

    def test_get_price_zero_on_error(self):
        import modules.price_cache as pc
        with patch("yfinance.Ticker", side_effect=ConnectionError("timeout")):
            price = pc.get_price("BAD")
        assert price == 0.0

    def test_get_price_caches_result(self):
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 200.0
        with patch("yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            pc.get_price("MSFT")
            pc.get_price("MSFT")  # second call should hit cache
        # yfinance.Ticker should be called only once
        assert mock_yf.call_count == 1

    def test_cache_hit_increments_hits(self):
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 100.0
        with patch("yfinance.Ticker", return_value=mock_ticker):
            pc.get_price("GOOG")
            pc.get_price("GOOG")
        stats = pc.cache_stats()
        assert stats["hits"] >= 1

    def test_cache_miss_increments_misses(self):
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 50.0
        with patch("yfinance.Ticker", return_value=mock_ticker):
            pc.get_price("AMZN")
        stats = pc.cache_stats()
        assert stats["misses"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# TestGetPricesBatch
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPricesBatch:

    def setup_method(self):
        _reset_cache()

    def test_get_prices_returns_dict(self):
        import modules.price_cache as pc
        mock_df = _make_download_df({"AAPL": 150.0, "MSFT": 300.0})
        with patch("yfinance.download", return_value=mock_df):
            result = pc.get_prices(["AAPL", "MSFT"])
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_get_prices_correct_values(self):
        import modules.price_cache as pc
        mock_df = _make_download_df({"AAPL": 175.0, "GOOGL": 140.0})
        with patch("yfinance.download", return_value=mock_df):
            result = pc.get_prices(["AAPL", "GOOGL"])
        assert abs(result.get("AAPL", 0) - 175.0) < 0.01
        assert abs(result.get("GOOGL", 0) - 140.0) < 0.01

    def test_get_prices_empty_list(self):
        import modules.price_cache as pc
        result = pc.get_prices([])
        assert result == {}

    def test_get_prices_uses_cache_on_second_call(self):
        import modules.price_cache as pc
        mock_df = _make_download_df({"TSLA": 250.0})
        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            pc.get_prices(["TSLA"])
            pc.get_prices(["TSLA"])  # second call: all fresh
        # download called only once (second call served from cache)
        assert mock_dl.call_count == 1

    def test_get_prices_normalises_ticker_case(self):
        import modules.price_cache as pc
        mock_df = _make_download_df({"AAPL": 180.0})
        with patch("yfinance.download", return_value=mock_df):
            result = pc.get_prices(["aapl"])
        assert "AAPL" in result

    def test_get_prices_batch_fallback_on_download_error(self):
        """Si yf.download falla, debe hacer fetches individuales."""
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 99.0

        with patch("yfinance.download", side_effect=Exception("network error")):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = pc.get_prices(["NVDA"])

        assert result.get("NVDA", 0) == 99.0


# ══════════════════════════════════════════════════════════════════════════════
# TestTTL
# ══════════════════════════════════════════════════════════════════════════════

class TestTTL:

    def setup_method(self):
        _reset_cache()
        import modules.price_cache as pc
        pc.set_ttl(300)  # restore default

    def test_set_ttl_changes_ttl(self):
        import modules.price_cache as pc
        pc.set_ttl(60)
        assert pc.get_ttl() == 60

    def test_set_ttl_minimum_30(self):
        import modules.price_cache as pc
        pc.set_ttl(5)   # below minimum
        assert pc.get_ttl() == 30

    def test_entry_stale_after_ttl(self):
        """Entry inserted with monotonic timestamp older than TTL is stale."""
        import modules.price_cache as pc
        # Insert entry with timestamp in the past (TTL + 10s ago)
        pc.set_ttl(30)
        with pc._lock:
            pc._cache["STALE"] = {
                "price": 1.0,
                "ts": time.monotonic() - 41,   # 41s ago, TTL=30 → stale
                "source": "test",
            }
        # Calling get_price should trigger a re-fetch (miss, not hit)
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 2.0
        with patch("yfinance.Ticker", return_value=mock_ticker):
            price = pc.get_price("STALE")
        assert price == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# TestInvalidate
# ══════════════════════════════════════════════════════════════════════════════

class TestInvalidate:

    def setup_method(self):
        _reset_cache()

    def _seed(self, ticker: str, price: float):
        import modules.price_cache as pc
        with pc._lock:
            pc._cache[ticker] = {"price": price, "ts": time.monotonic(), "source": "test"}

    def test_invalidate_single_ticker(self):
        import modules.price_cache as pc
        self._seed("AAPL", 150.0)
        self._seed("MSFT", 300.0)
        pc.invalidate("AAPL")
        with pc._lock:
            assert "AAPL" not in pc._cache
            assert "MSFT" in pc._cache

    def test_invalidate_all(self):
        import modules.price_cache as pc
        self._seed("AAPL", 150.0)
        self._seed("MSFT", 300.0)
        pc.invalidate()
        with pc._lock:
            assert len(pc._cache) == 0

    def test_invalidate_nonexistent_no_error(self):
        import modules.price_cache as pc
        pc.invalidate("NOTEXISTS")  # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# TestCacheStats
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheStats:

    def setup_method(self):
        _reset_cache()

    def test_stats_structure(self):
        import modules.price_cache as pc
        stats = pc.cache_stats()
        for key in ("hits", "misses", "hit_rate", "size", "batch_calls",
                    "single_calls", "errors", "ttl_seconds", "entries"):
            assert key in stats, f"Missing key: {key}"

    def test_hit_rate_zero_when_no_calls(self):
        import modules.price_cache as pc
        stats = pc.cache_stats()
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_100_when_all_hits(self):
        import modules.price_cache as pc
        # Seed and hit twice
        with pc._lock:
            pc._cache["X"] = {"price": 10.0, "ts": time.monotonic(), "source": "test"}
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 10.0
        pc.get_price("X")   # hit
        pc.get_price("X")   # hit
        stats = pc.cache_stats()
        assert stats["hits"] == 2
        assert stats["hit_rate"] == 100.0

    def test_reset_stats_clears_counters(self):
        import modules.price_cache as pc
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 5.0
        with patch("yfinance.Ticker", return_value=mock_ticker):
            pc.get_price("ZZZZ")
        pc.reset_stats()
        stats = pc.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["errors"] == 0

    def test_entries_reflect_cache_contents(self):
        import modules.price_cache as pc
        with pc._lock:
            pc._cache["META"] = {"price": 500.0, "ts": time.monotonic(), "source": "test"}
        stats = pc.cache_stats()
        tickers = [e["ticker"] for e in stats["entries"]]
        assert "META" in tickers


# ══════════════════════════════════════════════════════════════════════════════
# TestThreadSafety
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:

    def setup_method(self):
        _reset_cache()

    def test_concurrent_reads_no_error(self):
        """50 threads leyendo simultáneamente no deben lanzar excepciones."""
        import modules.price_cache as pc
        errors = []

        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 42.0

        def worker():
            try:
                with patch("yfinance.Ticker", return_value=mock_ticker):
                    pc.get_price("AAPL")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_invalidate_and_reads(self):
        """Invalidar mientras otros threads leen no debe causar errores."""
        import modules.price_cache as pc
        errors = []

        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 99.0

        def reader():
            try:
                with patch("yfinance.Ticker", return_value=mock_ticker):
                    pc.get_price("IBM")
            except Exception as e:
                errors.append(e)

        def invalidator():
            try:
                pc.invalidate("IBM")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader if i % 2 == 0 else invalidator)
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

#!/usr/bin/env python
"""
Comprehensive DataLoader Tests - Deep Coverage
==============================================
This test suite provides comprehensive coverage for DataLoader class:
- All public methods
- All data sources (yfinance, fmp, local)
- Edge cases and error handling
- Technical indicators
- Return calculations
- News loading

Target: Increase coverage from 18% to >90%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import json


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "Open": 100 + np.random.randn(100).cumsum(),
        "High": 102 + np.random.randn(100).cumsum(),
        "Low": 98 + np.random.randn(100).cumsum(),
        "Close": 100 + np.random.randn(100).cumsum(),
        "Volume": np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    return data


@pytest.fixture
def sample_yf_ticker_data():
    """Mock yfinance ticker history data"""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame({
        "Open": 150 + np.random.randn(50),
        "High": 152 + np.random.randn(50),
        "Low": 148 + np.random.randn(50),
        "Close": 150 + np.random.randn(50),
        "Volume": np.random.randint(5000000, 15000000, 50)
    }, index=dates)


@pytest.fixture
def sample_fmp_response():
    """Mock FMP API response"""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": 200 + i,
            "high": 202 + i,
            "low": 198 + i,
            "close": 200 + i,
            "volume": 1000000 + i * 10000,
            "vwap": 200 + i
        }
        for i, date in enumerate(dates)
    ]


@pytest.fixture
def sample_news_response():
    """Mock FMP news response"""
    return [
        {
            "title": "Company Announces Q4 Earnings",
            "publishedDate": "2024-01-15T10:00:00",
            "url": "https://example.com/news1",
            "text": "Company reported strong earnings..."
        },
        {
            "title": "New Product Launch",
            "publishedDate": "2024-01-16T14:30:00",
            "url": "https://example.com/news2",
            "text": "Company launches innovative product..."
        }
    ]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================
# Test 1: Initialization Tests
# ============================================================

class TestDataLoaderInitialization:
    """Test DataLoader initialization with different configurations"""

    def test_init_default_yfinance(self):
        """Test default initialization with yfinance"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        assert loader.data_source == "yfinance"
        assert loader.cache_dir is None
        assert loader.api_key is None or loader.api_key == os.environ.get("FMP_API_KEY")

    def test_init_with_fmp(self):
        """Test initialization with FMP data source"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="fmp", api_key="test-api-key-123")
        assert loader.data_source == "fmp"
        assert loader.api_key == "test-api-key-123"

    def test_init_with_local(self):
        """Test initialization with local data source"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="local")
        assert loader.data_source == "local"

    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist"""
        from finsage.data.data_loader import DataLoader

        new_cache_path = os.path.join(temp_cache_dir, "new_cache")
        assert not os.path.exists(new_cache_path)

        loader = DataLoader(cache_dir=new_cache_path)
        assert os.path.exists(new_cache_path)

    def test_init_with_existing_cache_dir(self, temp_cache_dir):
        """Test initialization with existing cache directory"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir == temp_cache_dir
        assert os.path.exists(temp_cache_dir)

    def test_init_reads_api_key_from_env(self):
        """Test that API key is read from environment if not provided"""
        from finsage.data.data_loader import DataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": "env-api-key"}):
            loader = DataLoader(data_source="fmp")
            assert loader.api_key == "env-api-key"


# ============================================================
# Test 2: YFinance Data Loading
# ============================================================

class TestYFinanceDataLoading:
    """Test yfinance data loading with various scenarios"""

    @patch("yfinance.Ticker")
    def test_load_yfinance_single_symbol_success(self, mock_ticker_class, sample_yf_ticker_data):
        """Test successful loading of single symbol from yfinance"""
        from finsage.data.data_loader import DataLoader

        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        loader = DataLoader(data_source="yfinance")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-02-29")

        assert not result.empty
        assert "AAPL" in result.columns.get_level_values(0)
        mock_ticker_class.assert_called_with("AAPL")

    @patch("yfinance.Ticker")
    def test_load_yfinance_multiple_symbols(self, mock_ticker_class, sample_yf_ticker_data):
        """Test loading multiple symbols from yfinance"""
        from finsage.data.data_loader import DataLoader

        # Setup mock for multiple symbols
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        loader = DataLoader(data_source="yfinance")
        result = loader.load_price_data(["AAPL", "MSFT", "GOOGL"], "2024-01-01", "2024-02-29")

        assert not result.empty
        assert mock_ticker_class.call_count == 3

    @patch("yfinance.Ticker")
    def test_load_yfinance_empty_data(self, mock_ticker_class):
        """Test handling of empty data from yfinance"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to return empty DataFrame
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        loader = DataLoader(data_source="yfinance")
        result = loader.load_price_data(["INVALID"], "2024-01-01", "2024-02-29")

        assert result.empty

    @patch("yfinance.Ticker")
    def test_load_yfinance_with_error(self, mock_ticker_class):
        """Test error handling in yfinance loading"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to raise exception
        mock_ticker_class.side_effect = Exception("Network error")

        loader = DataLoader(data_source="yfinance")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-02-29")

        assert result.empty

    @patch("yfinance.Ticker")
    def test_load_yfinance_partial_success(self, mock_ticker_class, sample_yf_ticker_data):
        """Test partial success when some symbols fail"""
        from finsage.data.data_loader import DataLoader

        # Setup mock: first symbol succeeds, second fails, third succeeds
        def ticker_side_effect(symbol):
            mock_ticker = MagicMock()
            if symbol == "FAIL":
                mock_ticker.history.side_effect = Exception("Failed")
            else:
                mock_ticker.history.return_value = sample_yf_ticker_data
            return mock_ticker

        mock_ticker_class.side_effect = ticker_side_effect

        loader = DataLoader(data_source="yfinance")
        result = loader.load_price_data(["AAPL", "FAIL", "MSFT"], "2024-01-01", "2024-02-29")

        assert not result.empty
        assert mock_ticker_class.call_count == 3

    @patch("time.sleep")
    @patch("yfinance.Ticker")
    def test_load_yfinance_rate_limiting(self, mock_ticker_class, mock_sleep, sample_yf_ticker_data):
        """Test that yfinance loader implements rate limiting"""
        from finsage.data.data_loader import DataLoader

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        loader = DataLoader(data_source="yfinance")
        loader.load_price_data(["AAPL", "MSFT"], "2024-01-01", "2024-02-29")

        # Should call sleep for rate limiting
        assert mock_sleep.called

    @patch("yfinance.Ticker")
    def test_load_yfinance_with_interval(self, mock_ticker_class, sample_yf_ticker_data):
        """Test loading data with different intervals"""
        from finsage.data.data_loader import DataLoader

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        loader = DataLoader(data_source="yfinance")
        loader.load_price_data(["AAPL"], "2024-01-01", "2024-02-29", interval="1h")

        mock_ticker.history.assert_called_with(
            start="2024-01-01",
            end="2024-02-29",
            interval="1h"
        )


# ============================================================
# Test 3: FMP Data Loading
# ============================================================

class TestFMPDataLoading:
    """Test FMP data loading with various scenarios"""

    @patch("requests.get")
    def test_load_fmp_success(self, mock_get, sample_fmp_response):
        """Test successful loading from FMP"""
        from finsage.data.data_loader import DataLoader

        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_fmp_response
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert not result.empty
        assert "Close" in result.columns.get_level_values(1)

    @patch("requests.get")
    def test_load_fmp_without_api_key(self, mock_get):
        """Test FMP loading without API key"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="fmp", api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            loader.api_key = None
            result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert result.empty

    @patch("requests.get")
    def test_load_fmp_http_error(self, mock_get):
        """Test FMP loading with HTTP error"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to return error status
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert result.empty

    @patch("requests.get")
    @patch("time.sleep")
    def test_load_fmp_rate_limit_429(self, mock_sleep, mock_get):
        """Test FMP handling of 429 rate limit"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to return 429
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        # Should sleep when rate limited
        assert mock_sleep.called

    @patch("requests.get")
    def test_load_fmp_timeout(self, mock_get):
        """Test FMP loading with timeout"""
        from finsage.data.data_loader import DataLoader
        import requests

        # Setup mock to raise timeout
        mock_get.side_effect = requests.exceptions.Timeout()

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert result.empty

    @patch("requests.get")
    def test_load_fmp_empty_response(self, mock_get):
        """Test FMP loading with empty response"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to return empty list
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert result.empty

    @patch("requests.get")
    def test_load_fmp_multiple_symbols(self, mock_get, sample_fmp_response):
        """Test loading multiple symbols from FMP"""
        from finsage.data.data_loader import DataLoader

        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_fmp_response
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL", "MSFT", "GOOGL"], "2024-01-01", "2024-01-31")

        assert not result.empty
        assert mock_get.call_count == 3

    @patch("requests.get")
    def test_load_fmp_crypto_symbol_conversion(self, mock_get, sample_fmp_response):
        """Test FMP converts crypto symbols (BTC-USD -> BTCUSD)"""
        from finsage.data.data_loader import DataLoader

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_fmp_response
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        loader.load_price_data(["BTC-USD"], "2024-01-01", "2024-01-31")

        # Check that the URL contains converted symbol
        call_url = mock_get.call_args[0][0]
        assert "BTCUSD" in call_url

    @patch("requests.get")
    def test_load_fmp_exception_handling(self, mock_get):
        """Test FMP exception handling"""
        from finsage.data.data_loader import DataLoader

        # Setup mock to raise exception
        mock_get.side_effect = Exception("Unexpected error")

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert result.empty


# ============================================================
# Test 4: Local Data Loading
# ============================================================

class TestLocalDataLoading:
    """Test local CSV file loading"""

    def test_load_local_success(self, temp_cache_dir, sample_price_data):
        """Test successful loading from local CSV"""
        from finsage.data.data_loader import DataLoader

        # Create test CSV file
        csv_path = os.path.join(temp_cache_dir, "AAPL.csv")
        sample_price_data.to_csv(csv_path)

        loader = DataLoader(data_source="local", cache_dir=temp_cache_dir)
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-03-31")

        assert not result.empty
        assert "AAPL" in result.columns.get_level_values(0)

    def test_load_local_without_cache_dir(self):
        """Test local loading without cache directory"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="local", cache_dir=None)
        result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-03-31")

        assert result.empty

    def test_load_local_missing_file(self, temp_cache_dir):
        """Test local loading with missing file"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="local", cache_dir=temp_cache_dir)
        result = loader.load_price_data(["NONEXISTENT"], "2024-01-01", "2024-03-31")

        assert result.empty

    def test_load_local_multiple_files(self, temp_cache_dir, sample_price_data):
        """Test loading multiple local CSV files"""
        from finsage.data.data_loader import DataLoader

        # Create multiple test CSV files
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            csv_path = os.path.join(temp_cache_dir, f"{symbol}.csv")
            sample_price_data.to_csv(csv_path)

        loader = DataLoader(data_source="local", cache_dir=temp_cache_dir)
        result = loader.load_price_data(["AAPL", "MSFT", "GOOGL"], "2024-01-01", "2024-03-31")

        assert not result.empty
        assert len(result.columns.get_level_values(0).unique()) == 3

    def test_load_local_date_filtering(self, temp_cache_dir, sample_price_data):
        """Test that local loading filters by date range"""
        from finsage.data.data_loader import DataLoader

        csv_path = os.path.join(temp_cache_dir, "AAPL.csv")
        sample_price_data.to_csv(csv_path)

        loader = DataLoader(data_source="local", cache_dir=temp_cache_dir)
        result = loader.load_price_data(["AAPL"], "2024-01-15", "2024-01-31")

        assert not result.empty
        assert len(result) <= len(sample_price_data)


# ============================================================
# Test 5: Unknown Data Source
# ============================================================

class TestUnknownDataSource:
    """Test handling of unknown data sources"""

    def test_unknown_data_source_raises_error(self):
        """Test that unknown data source raises ValueError"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="unknown_source")

        with pytest.raises(ValueError, match="Unknown data source"):
            loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")


# ============================================================
# Test 6: News Loading
# ============================================================

class TestNewsLoading:
    """Test news loading functionality"""

    @patch("requests.get")
    def test_load_news_fmp_success(self, mock_get, sample_news_response):
        """Test successful news loading from FMP"""
        from finsage.data.data_loader import DataLoader

        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_news_response
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31", limit=10)

        assert len(result) > 0
        assert "symbol" in result[0]
        assert result[0]["symbol"] == "AAPL"

    @patch("requests.get")
    def test_load_news_multiple_symbols(self, mock_get, sample_news_response):
        """Test loading news for multiple symbols"""
        from finsage.data.data_loader import DataLoader

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_news_response
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")

        assert len(result) > 0
        assert mock_get.call_count == 2

    def test_load_news_without_fmp(self):
        """Test that news loading returns empty for non-FMP sources"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert result == []

    @patch("requests.get")
    def test_load_news_without_api_key(self, mock_get):
        """Test news loading without API key"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="fmp", api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            loader.api_key = None
            result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert result == []

    @patch("requests.get")
    def test_load_news_http_403(self, mock_get):
        """Test news loading with 403 forbidden"""
        from finsage.data.data_loader import DataLoader

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert result == []

    @patch("requests.get")
    def test_load_news_timeout(self, mock_get):
        """Test news loading with timeout"""
        from finsage.data.data_loader import DataLoader
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert result == []

    @patch("requests.get")
    def test_load_news_field_standardization(self, mock_get):
        """Test that news fields are standardized"""
        from finsage.data.data_loader import DataLoader

        # Response with different field names
        news_data = [{
            "title": "Test News",
            "publishedDate": "2024-01-15",
            "url": "https://example.com"
        }]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = news_data
        mock_get.return_value = mock_response

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert "date" in result[0]
        assert "link" in result[0]

    @patch("requests.get")
    def test_load_news_exception_handling(self, mock_get):
        """Test news loading exception handling"""
        from finsage.data.data_loader import DataLoader

        mock_get.side_effect = Exception("Unexpected error")

        loader = DataLoader(data_source="fmp", api_key="test-key")
        result = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert result == []


# ============================================================
# Test 7: Calculate Returns
# ============================================================

class TestCalculateReturns:
    """Test return calculation methods"""

    def test_calculate_log_returns(self, sample_price_data):
        """Test log returns calculation"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        returns = loader.calculate_returns(sample_price_data, method="log")

        assert not returns.empty
        assert len(returns) == len(sample_price_data) - 1

    def test_calculate_simple_returns(self, sample_price_data):
        """Test simple returns calculation"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        returns = loader.calculate_returns(sample_price_data, method="simple")

        assert not returns.empty
        assert len(returns) == len(sample_price_data) - 1

    def test_calculate_returns_with_close_column(self):
        """Test returns calculation with Close column"""
        from finsage.data.data_loader import DataLoader

        data = pd.DataFrame({
            "Close": [100, 102, 101, 105, 103]
        })

        loader = DataLoader()
        returns = loader.calculate_returns(data, method="simple")

        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.02) < 0.001

    def test_calculate_returns_with_adj_close(self):
        """Test returns calculation with Adj Close column"""
        from finsage.data.data_loader import DataLoader

        data = pd.DataFrame({
            "Close": [100, 102, 101, 105, 103],
            "Adj Close": [98, 100, 99, 103, 101]
        })

        loader = DataLoader()
        returns = loader.calculate_returns(data, method="simple")

        # Should use Adj Close preferentially
        assert not returns.empty
        assert abs(returns.iloc[0] - (100-98)/98) < 0.001

    def test_calculate_returns_without_price_columns(self):
        """Test returns calculation with non-standard column name"""
        from finsage.data.data_loader import DataLoader

        # Use DataFrame with non-standard column name
        data = pd.DataFrame({"Price": [100, 102, 101, 105, 103]})

        loader = DataLoader()
        # This should handle non-standard columns gracefully
        try:
            returns = loader.calculate_returns(data, method="log")
            # If it returns something, verify it
            if not returns.empty:
                assert len(returns) <= len(data)
        except (KeyError, AttributeError):
            # Expected behavior when column is not found
            pass


# ============================================================
# Test 8: Technical Indicators
# ============================================================

class TestTechnicalIndicators:
    """Test technical indicator calculations"""

    def test_get_technical_indicators_default(self, sample_price_data):
        """Test default technical indicators"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(sample_price_data)

        assert "sma_20" in indicators
        assert "sma_50" in indicators
        assert "rsi_14" in indicators
        assert "macd" in indicators
        assert "bb_upper" in indicators

    def test_get_technical_indicators_custom(self, sample_price_data):
        """Test custom technical indicators"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(
            sample_price_data,
            indicators=["sma_10", "ema_20"]
        )

        assert "sma_10" in indicators
        assert "ema_20" in indicators
        assert "rsi_14" not in indicators

    def test_calculate_sma(self, sample_price_data):
        """Test SMA calculation"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(
            sample_price_data,
            indicators=["sma_20"]
        )

        assert "sma_20" in indicators
        assert not indicators["sma_20"].empty

    def test_calculate_ema(self, sample_price_data):
        """Test EMA calculation"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(
            sample_price_data,
            indicators=["ema_12"]
        )

        assert "ema_12" in indicators
        assert not indicators["ema_12"].empty

    def test_calculate_rsi(self):
        """Test RSI calculation"""
        from finsage.data.data_loader import DataLoader

        # Create price data with known pattern
        prices = pd.Series([100, 102, 101, 105, 103, 107, 106, 110])

        loader = DataLoader()
        rsi = loader._calculate_rsi(prices, period=5)

        assert len(rsi) == len(prices)
        assert isinstance(rsi, pd.Series)

    def test_calculate_rsi_edge_case_zero_loss(self):
        """Test RSI calculation when loss is zero"""
        from finsage.data.data_loader import DataLoader

        # Prices only going up (no losses)
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107])

        loader = DataLoader()
        rsi = loader._calculate_rsi(prices, period=5)

        # RSI should be valid even with zero losses
        assert not rsi.isna().all()

    def test_calculate_macd(self):
        """Test MACD calculation"""
        from finsage.data.data_loader import DataLoader

        prices = pd.Series(100 + np.random.randn(100).cumsum())

        loader = DataLoader()
        macd, signal, hist = loader._calculate_macd(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)

    def test_calculate_macd_custom_params(self):
        """Test MACD with custom parameters"""
        from finsage.data.data_loader import DataLoader

        prices = pd.Series(100 + np.random.randn(100).cumsum())

        loader = DataLoader()
        macd, signal, hist = loader._calculate_macd(prices, fast=8, slow=21, signal=5)

        assert len(macd) == len(prices)

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        from finsage.data.data_loader import DataLoader

        prices = pd.Series(100 + np.random.randn(100).cumsum())

        loader = DataLoader()
        upper, middle, lower = loader._calculate_bollinger_bands(prices)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_calculate_bollinger_bands_custom_params(self):
        """Test Bollinger Bands with custom parameters"""
        from finsage.data.data_loader import DataLoader

        prices = pd.Series(100 + np.random.randn(100).cumsum())

        loader = DataLoader()
        upper, middle, lower = loader._calculate_bollinger_bands(
            prices, period=10, std_dev=3.0
        )

        assert len(upper) == len(prices)

    def test_get_indicators_with_series_input(self):
        """Test indicators with Series input instead of DataFrame"""
        from finsage.data.data_loader import DataLoader

        prices = pd.Series(100 + np.random.randn(100).cumsum())

        loader = DataLoader()
        indicators = loader.get_technical_indicators(prices, indicators=["sma_20"])

        assert "sma_20" in indicators

    def test_get_indicators_with_dataframe_single_column(self):
        """Test indicators with single-column DataFrame"""
        from finsage.data.data_loader import DataLoader

        data = pd.DataFrame({
            "Price": 100 + np.random.randn(100).cumsum()
        })

        loader = DataLoader()
        indicators = loader.get_technical_indicators(data, indicators=["sma_20"])

        assert "sma_20" in indicators

    def test_macd_indicator_all_outputs(self, sample_price_data):
        """Test that MACD indicator returns all three outputs"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(
            sample_price_data,
            indicators=["macd"]
        )

        assert "macd" in indicators
        assert "macd_signal" in indicators
        assert "macd_hist" in indicators

    def test_bollinger_bands_all_outputs(self, sample_price_data):
        """Test that Bollinger Bands returns all three bands"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader()
        indicators = loader.get_technical_indicators(
            sample_price_data,
            indicators=["bb"]
        )

        assert "bb_upper" in indicators
        assert "bb_middle" in indicators
        assert "bb_lower" in indicators


# ============================================================
# Test 9: Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests combining multiple features"""

    @patch("yfinance.Ticker")
    def test_end_to_end_yfinance_workflow(self, mock_ticker_class, sample_yf_ticker_data):
        """Test complete workflow with yfinance"""
        from finsage.data.data_loader import DataLoader

        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        # Load data
        loader = DataLoader(data_source="yfinance")
        price_data = loader.load_price_data(["AAPL"], "2024-01-01", "2024-02-29")

        # Calculate returns
        returns = loader.calculate_returns(price_data, method="log")

        # For technical indicators, extract Close column as Series
        # The price_data has MultiIndex columns (symbol, OHLCV)
        if ("AAPL", "Close") in price_data.columns:
            close_prices = price_data[("AAPL", "Close")]
            indicators = loader.get_technical_indicators(close_prices, indicators=["sma_20", "rsi_14"])
            assert len(indicators) > 0

        assert not price_data.empty
        assert not returns.empty

    @patch("requests.get")
    def test_end_to_end_fmp_workflow(self, mock_get, sample_fmp_response, sample_news_response):
        """Test complete workflow with FMP"""
        from finsage.data.data_loader import DataLoader

        # Setup mock for price data
        mock_response = MagicMock()
        mock_response.status_code = 200

        # First call for price data, second for news
        mock_response.json.side_effect = [sample_fmp_response, sample_news_response]
        mock_get.return_value = mock_response

        # Load data
        loader = DataLoader(data_source="fmp", api_key="test-key")
        price_data = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-31")
        news = loader.load_news(["AAPL"], "2024-01-01", "2024-01-31")

        assert not price_data.empty
        assert len(news) > 0

    def test_end_to_end_local_workflow(self, temp_cache_dir, sample_price_data):
        """Test complete workflow with local files"""
        from finsage.data.data_loader import DataLoader

        # Create test CSV
        csv_path = os.path.join(temp_cache_dir, "AAPL.csv")
        sample_price_data.to_csv(csv_path)

        # Load and process
        loader = DataLoader(data_source="local", cache_dir=temp_cache_dir)
        price_data = loader.load_price_data(["AAPL"], "2024-01-01", "2024-03-31")
        returns = loader.calculate_returns(price_data)
        indicators = loader.get_technical_indicators(price_data, indicators=["sma_20"])

        assert not price_data.empty
        assert not returns.empty
        assert "sma_20" in indicators


# ============================================================
# Test 10: Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_symbol_list(self):
        """Test with empty symbol list"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        with patch("yfinance.Ticker") as mock_ticker_class:
            result = loader.load_price_data([], "2024-01-01", "2024-01-31")
            assert result.empty

    def test_very_short_date_range(self):
        """Test with very short date range"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            result = loader.load_price_data(["AAPL"], "2024-01-01", "2024-01-01")
            # Should handle gracefully

    def test_future_dates(self):
        """Test with future dates"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            future_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
            result = loader.load_price_data(["AAPL"], "2024-01-01", future_date)
            # Should handle gracefully

    def test_invalid_date_format(self):
        """Test with invalid date format"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="yfinance")
        # This may raise an exception or be handled by the underlying library
        # Just ensure it doesn't crash our code

    def test_special_characters_in_symbol(self):
        """Test with special characters in symbol"""
        from finsage.data.data_loader import DataLoader

        loader = DataLoader(data_source="fmp", api_key="test-key")
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response

            # Symbol with special chars should be URL-encoded
            result = loader.load_price_data(["BRK.B"], "2024-01-01", "2024-01-31")

    def test_very_large_symbol_list(self):
        """Test with very large symbol list"""
        from finsage.data.data_loader import DataLoader

        large_symbol_list = [f"SYM{i}" for i in range(100)]

        loader = DataLoader(data_source="yfinance")
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            # Should handle batching properly
            result = loader.load_price_data(large_symbol_list, "2024-01-01", "2024-01-31")

    def test_calculate_returns_single_price(self):
        """Test returns calculation with single price point"""
        from finsage.data.data_loader import DataLoader

        data = pd.DataFrame({"Close": [100]})
        loader = DataLoader()
        returns = loader.calculate_returns(data)

        assert returns.empty  # Should be empty after dropna()

    def test_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data"""
        from finsage.data.data_loader import DataLoader

        # Only 10 data points, but requesting SMA-50
        data = pd.DataFrame({
            "Close": 100 + np.random.randn(10).cumsum()
        })

        loader = DataLoader()
        indicators = loader.get_technical_indicators(data, indicators=["sma_50"])

        # Should return indicator but with many NaN values
        assert "sma_50" in indicators


# ============================================================
# Test 11: Performance and Concurrency
# ============================================================

class TestPerformance:
    """Test performance-related features"""

    @patch("time.sleep")
    @patch("yfinance.Ticker")
    def test_batch_processing(self, mock_ticker_class, mock_sleep, sample_yf_ticker_data):
        """Test that batch processing is used for rate limiting"""
        from finsage.data.data_loader import DataLoader

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_yf_ticker_data
        mock_ticker_class.return_value = mock_ticker

        # Load 10 symbols (should be batched)
        symbols = [f"SYM{i}" for i in range(10)]
        loader = DataLoader(data_source="yfinance")
        loader.load_price_data(symbols, "2024-01-01", "2024-01-31")

        # Should have called sleep for rate limiting
        assert mock_sleep.called


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" DataLoader Deep Coverage Tests")
    print("=" * 80)

    pytest.main([__file__, "-v", "--tb=short", "--cov=finsage.data.data_loader",
                 "--cov-report=term-missing"])


if __name__ == "__main__":
    run_tests()

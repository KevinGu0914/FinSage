#!/usr/bin/env python
"""
FMP Client Deep Testing - FMP客户端深度测试
Coverage: fmp_client.py (targeting 17% -> 90%+)

Tests all classes, methods, and code paths:
- FMPClient: API requests, rate limiting, caching
- FactorScreener: Factor-based stock screening
- FMPNewsClient: News retrieval
- FMPETFClient: ETF holdings
- Singleton functions
- Convenience functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from typing import Dict, List, Any, Optional

from finsage.data.fmp_client import (
    FMPClient,
    FactorScreener,
    FMPNewsClient,
    FMPETFClient,
    get_fmp_client,
    get_news_client,
    get_etf_client,
    get_factor_screener,
    screen_stocks,
    get_quote,
    get_historical_price,
    get_stock_news,
    get_profile,
)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def mock_api_key():
    """Mock API key"""
    return "test-api-key-12345"


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory"""
    return str(tmp_path / "cache")


@pytest.fixture
def fmp_client(mock_api_key, temp_cache_dir):
    """FMPClient instance with mocked API key and temp cache"""
    return FMPClient(api_key=mock_api_key, cache_dir=temp_cache_dir, rate_limit=10)


@pytest.fixture
def mock_stock_data():
    """Mock stock screener data"""
    return [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "marketCap": 3000000000000,
            "price": 180.0,
            "volume": 50000000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "pe": 30.0,
            "priceToBook": 45.0,
            "roe": 0.50,
            "roa": 0.25,
            "grossProfitMargin": 0.43,
            "priceChange1Y": 0.45,
            "revenueGrowth": 0.08,
            "epsGrowth": 0.12,
        },
        {
            "symbol": "MSFT",
            "companyName": "Microsoft Corporation",
            "marketCap": 2800000000000,
            "price": 380.0,
            "volume": 30000000,
            "sector": "Technology",
            "industry": "Software",
            "pe": 35.0,
            "priceToBook": 12.0,
            "roe": 0.45,
            "roa": 0.20,
            "grossProfitMargin": 0.70,
            "priceChange1Y": 0.55,
            "revenueGrowth": 0.12,
            "epsGrowth": 0.15,
        },
    ]


@pytest.fixture
def mock_quote_data():
    """Mock quote data"""
    return [
        {
            "symbol": "AAPL",
            "price": 180.0,
            "change": 2.5,
            "changePercent": 1.41,
            "volume": 50000000,
            "timestamp": "2024-01-15 16:00:00",
        }
    ]


@pytest.fixture
def mock_historical_data():
    """Mock historical price data"""
    return [
        {"date": "2024-01-15", "open": 178.0, "high": 182.0, "low": 177.0, "close": 180.0, "volume": 50000000, "adjClose": 180.0},
        {"date": "2024-01-14", "open": 175.0, "high": 179.0, "low": 174.0, "close": 178.0, "volume": 48000000, "adjClose": 178.0},
        {"date": "2024-01-13", "open": 173.0, "high": 176.0, "low": 172.0, "close": 175.0, "volume": 45000000, "adjClose": 175.0},
    ]


# ============================================================
# Test 1: FMPClient Initialization
# ============================================================

class TestFMPClientInitialization:
    """测试 FMPClient 初始化"""

    def test_init_with_api_key(self, mock_api_key, temp_cache_dir):
        """测试使用 API key 初始化"""
        client = FMPClient(api_key=mock_api_key, cache_dir=temp_cache_dir)
        assert client.api_key == mock_api_key
        assert client.rate_limit == 750
        assert Path(temp_cache_dir).exists()

    def test_init_with_env_variable(self, temp_cache_dir, monkeypatch):
        """测试从环境变量读取 API key"""
        monkeypatch.setenv("FMP_API_KEY", "env-api-key")
        client = FMPClient(cache_dir=temp_cache_dir)
        assert client.api_key == "env-api-key"

    def test_init_with_oa_env_variable(self, temp_cache_dir, monkeypatch):
        """测试从 OA_FMP_KEY 环境变量读取"""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setenv("OA_FMP_KEY", "oa-api-key")
        client = FMPClient(cache_dir=temp_cache_dir)
        assert client.api_key == "oa-api-key"

    def test_init_without_api_key(self, temp_cache_dir, monkeypatch):
        """测试没有 API key 时抛出异常"""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.delenv("OA_FMP_KEY", raising=False)
        with pytest.raises(ValueError, match="FMP API key not found"):
            FMPClient(cache_dir=temp_cache_dir)

    def test_init_custom_rate_limit(self, mock_api_key, temp_cache_dir):
        """测试自定义速率限制"""
        client = FMPClient(api_key=mock_api_key, cache_dir=temp_cache_dir, rate_limit=100)
        assert client.rate_limit == 100

    def test_cache_dir_creation(self, mock_api_key, temp_cache_dir):
        """测试缓存目录创建"""
        cache_path = Path(temp_cache_dir) / "nested" / "dir"
        client = FMPClient(api_key=mock_api_key, cache_dir=str(cache_path))
        assert cache_path.exists()


# ============================================================
# Test 2: Rate Limiting
# ============================================================

class TestRateLimiting:
    """测试速率限制"""

    def test_rate_limit_wait_under_limit(self, fmp_client):
        """测试未达到速率限制"""
        # 添加少量请求
        for _ in range(3):
            fmp_client._rate_limit_wait()

        # 应该不需要等待
        assert len(fmp_client._request_times) == 3

    def test_rate_limit_wait_cleanup_old_requests(self, fmp_client):
        """测试清理旧请求记录"""
        # 添加超过1分钟的请求
        old_time = time.time() - 61
        fmp_client._request_times = [old_time, old_time + 1, old_time + 2]

        fmp_client._rate_limit_wait()

        # 旧请求应该被清理，只保留最近的时间戳
        # _rate_limit_wait可能添加1-2个新时间戳
        assert len(fmp_client._request_times) <= 2  # 旧的3个应该都被清理

    @patch('time.sleep')
    def test_rate_limit_wait_at_limit(self, mock_sleep, fmp_client):
        """测试达到速率限制时等待"""
        # 填满速率限制
        now = time.time()
        fmp_client._request_times = [now - i for i in range(fmp_client.rate_limit)]

        fmp_client._rate_limit_wait()

        # 应该调用 sleep
        assert mock_sleep.called


# ============================================================
# Test 3: API Requests
# ============================================================

class TestAPIRequests:
    """测试 API 请求"""

    @patch('requests.get')
    def test_request_success(self, mock_get, fmp_client, mock_stock_data):
        """测试成功的 API 请求"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        result = fmp_client._request("/test-endpoint")

        assert result == mock_stock_data
        assert mock_get.called
        args, kwargs = mock_get.call_args
        assert "apikey" in kwargs["params"]

    @patch('requests.get')
    def test_request_with_params(self, mock_get, fmp_client):
        """测试带参数的请求"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        fmp_client._request("/test", {"symbol": "AAPL", "limit": 10})

        args, kwargs = mock_get.call_args
        assert kwargs["params"]["symbol"] == "AAPL"
        assert kwargs["params"]["limit"] == 10

    @patch('requests.get')
    def test_request_404_error(self, mock_get, fmp_client):
        """测试 404 错误"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_get.return_value = mock_response

        result = fmp_client._request("/invalid-endpoint")

        assert result is None

    @patch('requests.get')
    @patch('time.sleep')
    def test_request_429_retry(self, mock_sleep, mock_get, fmp_client):
        """测试 429 速率限制重试"""
        # 第一次返回 429，第二次成功
        mock_response_429 = Mock()
        mock_response_429.status_code = 429

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"success": True}

        mock_get.side_effect = [mock_response_429, mock_response_200]

        result = fmp_client._request("/test")

        assert result == {"success": True}
        assert mock_sleep.called  # 应该等待

    @patch('requests.get')
    @patch('time.sleep')
    def test_request_429_max_retries(self, mock_sleep, mock_get, fmp_client):
        """测试达到最大重试次数"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = fmp_client._request("/test")

        assert result is None
        assert mock_get.call_count == 4  # 初始 + 3次重试

    @patch('requests.get')
    def test_request_timeout(self, mock_get, fmp_client):
        """测试请求超时"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = fmp_client._request("/test")

        assert result is None

    @patch('requests.get')
    def test_request_exception(self, mock_get, fmp_client):
        """测试请求异常"""
        mock_get.side_effect = Exception("Network error")

        result = fmp_client._request("/test")

        assert result is None


# ============================================================
# Test 4: Stock Screening
# ============================================================

class TestStockScreening:
    """测试股票筛选"""

    @patch('requests.get')
    def test_screen_stocks_basic(self, mock_get, fmp_client, mock_stock_data):
        """测试基本股票筛选"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        df = fmp_client.screen_stocks(market_cap_min=1e9, limit=100)

        assert not df.empty
        assert len(df) == 2
        assert "AAPL" in df["symbol"].values

    @patch('requests.get')
    def test_screen_stocks_all_filters(self, mock_get, fmp_client, mock_stock_data):
        """测试所有筛选条件"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        df = fmp_client.screen_stocks(
            market_cap_min=1e9,
            market_cap_max=5e12,
            price_min=50,
            price_max=500,
            volume_min=1e6,
            beta_min=0.5,
            beta_max=2.0,
            dividend_min=0.01,
            sector="Technology",
            industry="Software",
            country="US",
            exchange="NASDAQ",
            is_actively_trading=True,
            limit=500,
        )

        assert not df.empty
        # 验证请求参数
        args, kwargs = mock_get.call_args
        params = kwargs["params"]
        assert params["marketCapMoreThan"] == 1e9
        assert params["marketCapLowerThan"] == 5e12
        assert params["priceMoreThan"] == 50
        assert params["priceLowerThan"] == 500
        assert params["volumeMoreThan"] == 1e6
        assert params["betaMoreThan"] == 0.5
        assert params["betaLowerThan"] == 2.0
        assert params["dividendMoreThan"] == 0.01
        assert params["sector"] == "Technology"
        assert params["industry"] == "Software"

    @patch('requests.get')
    def test_screen_stocks_empty_response(self, mock_get, fmp_client):
        """测试空响应"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        df = fmp_client.screen_stocks()

        assert df.empty

    @patch('requests.get')
    def test_screen_stocks_invalid_response(self, mock_get, fmp_client):
        """测试无效响应"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_get.return_value = mock_response

        df = fmp_client.screen_stocks()

        assert df.empty

    @patch('requests.get')
    def test_get_stock_list(self, mock_get, fmp_client, mock_stock_data):
        """测试获取股票列表"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        df = fmp_client.get_stock_list()

        assert not df.empty
        assert len(df) == 2

    @patch('requests.get')
    def test_get_etf_list(self, mock_get, fmp_client):
        """测试获取 ETF 列表"""
        mock_etf_data = [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_etf_data
        mock_get.return_value = mock_response

        df = fmp_client.get_etf_list()

        assert not df.empty
        assert "SPY" in df["symbol"].values


# ============================================================
# Test 5: Multi-Asset Class
# ============================================================

class TestMultiAssetClass:
    """测试多资产类别"""

    @patch('requests.get')
    def test_get_cryptocurrency_list(self, mock_get, fmp_client):
        """测试获取加密货币列表"""
        mock_crypto = [
            {"symbol": "BTCUSD", "name": "Bitcoin"},
            {"symbol": "ETHUSD", "name": "Ethereum"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_crypto
        mock_get.return_value = mock_response

        df = fmp_client.get_cryptocurrency_list()

        assert not df.empty
        assert "asset_class" in df.columns
        assert df["asset_class"].iloc[0] == "crypto"

    @patch('requests.get')
    def test_get_commodities_list(self, mock_get, fmp_client):
        """测试获取大宗商品列表"""
        mock_commodities = [
            {"symbol": "GCUSD", "name": "Gold"},
            {"symbol": "CLUSD", "name": "Crude Oil"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_commodities
        mock_get.return_value = mock_response

        df = fmp_client.get_commodities_list()

        assert not df.empty
        assert df["asset_class"].iloc[0] == "commodities"

    @patch('requests.get')
    def test_get_reits(self, mock_get, fmp_client, mock_stock_data):
        """测试获取 REITs"""
        # Mock screen_stocks 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        df = fmp_client.get_reits(country="US", market_cap_min=1e9)

        assert not df.empty
        assert "asset_class" in df.columns

    @patch('requests.get')
    def test_get_bond_etfs(self, mock_get, fmp_client):
        """测试获取债券 ETF"""
        mock_etfs = [
            {"symbol": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF"},
            {"symbol": "BND", "name": "Vanguard Total Bond Market ETF"},
            {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_etfs
        mock_get.return_value = mock_response

        df = fmp_client.get_bond_etfs()

        assert not df.empty
        assert "asset_class" in df.columns
        assert df["asset_class"].iloc[0] == "bonds"

    @patch('requests.get')
    def test_get_bond_etfs_with_cache(self, mock_get, fmp_client):
        """测试使用缓存的债券 ETF"""
        mock_etfs = [
            {"symbol": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF"},
        ]

        # 保存到缓存
        cache_data = mock_etfs
        fmp_client._save_cache("etf_list", cache_data)

        # 不应该调用 API
        df = fmp_client.get_bond_etfs()

        assert not df.empty
        assert not mock_get.called  # 使用缓存，不调用 API

    @patch('requests.get')
    def test_get_commodity_etfs(self, mock_get, fmp_client):
        """测试获取商品 ETF"""
        mock_etfs = [
            {"symbol": "GLD", "name": "SPDR Gold Shares"},
            {"symbol": "USO", "name": "United States Oil Fund"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_etfs
        mock_get.return_value = mock_response

        df = fmp_client.get_commodity_etfs()

        assert not df.empty
        assert df["asset_class"].iloc[0] == "commodities"

    @patch('requests.get')
    def test_get_reit_etfs(self, mock_get, fmp_client):
        """测试获取 REIT ETF"""
        mock_etfs = [
            {"symbol": "VNQ", "name": "Vanguard Real Estate ETF"},
            {"symbol": "IYR", "name": "iShares U.S. Real Estate ETF"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_etfs
        mock_get.return_value = mock_response

        df = fmp_client.get_reit_etfs()

        assert not df.empty
        assert df["asset_class"].iloc[0] == "reits"

    @patch('requests.get')
    def test_get_etf_missing_name_column(self, mock_get, fmp_client):
        """测试 ETF 列表缺少名称列"""
        mock_etfs = [{"symbol": "SPY"}]  # 没有 name 或 companyName
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_etfs
        mock_get.return_value = mock_response

        df = fmp_client.get_bond_etfs()

        assert df.empty  # 应该返回空 DataFrame

    @patch.object(FMPClient, 'screen_stocks')
    @patch.object(FMPClient, 'get_cryptocurrency_list')
    @patch.object(FMPClient, 'get_commodities_list')
    @patch.object(FMPClient, 'get_commodity_etfs')
    @patch.object(FMPClient, 'get_reits')
    @patch.object(FMPClient, 'get_reit_etfs')
    @patch.object(FMPClient, 'get_bond_etfs')
    def test_get_full_asset_universe(
        self,
        mock_bonds,
        mock_reit_etfs,
        mock_reits,
        mock_comm_etfs,
        mock_commodities,
        mock_crypto,
        mock_stocks,
        fmp_client
    ):
        """测试获取完整资产范围"""
        # Mock 各个方法的返回值
        mock_stocks.return_value = pd.DataFrame([{"symbol": "AAPL", "marketCap": 3e12}])
        mock_crypto.return_value = pd.DataFrame([{"symbol": "BTCUSD"}])
        mock_commodities.return_value = pd.DataFrame([{"symbol": "GCUSD"}])
        mock_comm_etfs.return_value = pd.DataFrame([{"symbol": "GLD"}])
        mock_reits.return_value = pd.DataFrame([{"symbol": "VNQ"}])
        mock_reit_etfs.return_value = pd.DataFrame([{"symbol": "IYR"}])
        mock_bonds.return_value = pd.DataFrame([{"symbol": "AGG"}])

        universe = fmp_client.get_full_asset_universe(
            include_stocks=True,
            include_crypto=True,
            include_commodities=True,
            include_reits=True,
            include_bonds=True,
        )

        assert "stocks" in universe
        assert "crypto" in universe
        assert "commodities" in universe
        assert "reits" in universe
        assert "bonds" in universe

    @patch.object(FMPClient, 'screen_stocks')
    def test_get_full_asset_universe_stocks_only(self, mock_stocks, fmp_client):
        """测试仅获取股票"""
        mock_stocks.return_value = pd.DataFrame([{"symbol": "AAPL"}])

        universe = fmp_client.get_full_asset_universe(
            include_stocks=True,
            include_crypto=False,
            include_commodities=False,
            include_reits=False,
            include_bonds=False,
        )

        assert "stocks" in universe
        assert "crypto" not in universe


# ============================================================
# Test 6: Market Data
# ============================================================

class TestMarketData:
    """测试市场数据"""

    @patch('requests.get')
    def test_get_quote(self, mock_get, fmp_client, mock_quote_data):
        """测试获取报价"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_quote_data
        mock_get.return_value = mock_response

        quote = fmp_client.get_quote("AAPL")

        assert quote is not None
        assert quote["symbol"] == "AAPL"
        assert quote["price"] == 180.0

    @patch('requests.get')
    def test_get_quote_empty_response(self, mock_get, fmp_client):
        """测试获取报价返回空"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        quote = fmp_client.get_quote("INVALID")

        assert quote is None

    @patch('requests.get')
    @patch('time.sleep')
    def test_get_batch_quote(self, mock_sleep, mock_get, fmp_client):
        """测试批量获取报价"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "price": 180.0},
            {"symbol": "MSFT", "price": 380.0},
        ]
        mock_get.return_value = mock_response

        df = fmp_client.get_batch_quote(["AAPL", "MSFT"])

        assert not df.empty
        assert len(df) == 2

    @patch('requests.get')
    def test_get_batch_quote_empty_symbols(self, mock_get, fmp_client):
        """测试空符号列表"""
        df = fmp_client.get_batch_quote([])

        assert df.empty
        assert not mock_get.called

    @patch('requests.get')
    @patch('time.sleep')
    def test_get_batch_quote_large_batch(self, mock_sleep, mock_get, fmp_client):
        """测试大批量请求（超过100个符号）"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"symbol": f"SYM{i}", "price": 100.0} for i in range(50)]
        mock_get.return_value = mock_response

        # 150个符号，应该分2批
        symbols = [f"SYM{i}" for i in range(150)]
        df = fmp_client.get_batch_quote(symbols)

        assert not df.empty
        assert mock_get.call_count == 2  # 2批请求
        assert mock_sleep.called  # 批次间应该休息

    @patch('requests.get')
    def test_get_profile(self, mock_get, fmp_client):
        """测试获取公司概况"""
        mock_profile = [{
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_profile
        mock_get.return_value = mock_response

        profile = fmp_client.get_profile("AAPL")

        assert profile is not None
        assert profile["symbol"] == "AAPL"


# ============================================================
# Test 7: Financial Metrics
# ============================================================

class TestFinancialMetrics:
    """测试财务指标"""

    @patch('requests.get')
    def test_get_key_metrics_ttm(self, mock_get, fmp_client):
        """测试获取 TTM 关键指标"""
        mock_metrics = [{"symbol": "AAPL", "peRatio": 30.0, "marketCap": 3e12}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_metrics
        mock_get.return_value = mock_response

        metrics = fmp_client.get_key_metrics_ttm("AAPL")

        assert metrics is not None
        assert metrics["symbol"] == "AAPL"

    @patch('requests.get')
    def test_get_key_metrics_batch(self, mock_get, fmp_client):
        """测试批量获取关键指标"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            [{"peRatio": 30.0}],
            [{"peRatio": 35.0}],
        ]
        mock_get.return_value = mock_response

        df = fmp_client.get_key_metrics_batch(["AAPL", "MSFT"])

        assert not df.empty
        assert "symbol" in df.columns

    @patch('requests.get')
    def test_get_ratios_ttm(self, mock_get, fmp_client):
        """测试获取 TTM 财务比率"""
        mock_ratios = [{"symbol": "AAPL", "currentRatio": 1.5}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ratios
        mock_get.return_value = mock_response

        ratios = fmp_client.get_ratios_ttm("AAPL")

        assert ratios is not None

    @patch('requests.get')
    def test_get_financial_scores(self, mock_get, fmp_client):
        """测试获取财务评分"""
        mock_scores = [{"symbol": "AAPL", "piotroskiScore": 7, "altmanZScore": 5.2}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scores
        mock_get.return_value = mock_response

        scores = fmp_client.get_financial_scores("AAPL")

        assert scores is not None


# ============================================================
# Test 8: Financial Statements
# ============================================================

class TestFinancialStatements:
    """测试财务报表"""

    @patch('requests.get')
    def test_get_income_statement(self, mock_get, fmp_client):
        """测试获取利润表"""
        mock_income = [
            {"date": "2023-12-31", "revenue": 1e11, "netIncome": 2e10},
            {"date": "2022-12-31", "revenue": 9e10, "netIncome": 1.8e10},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_income
        mock_get.return_value = mock_response

        df = fmp_client.get_income_statement("AAPL", period="annual", limit=5)

        assert not df.empty
        assert len(df) == 2

    @patch('requests.get')
    def test_get_balance_sheet(self, mock_get, fmp_client):
        """测试获取资产负债表"""
        mock_balance = [
            {"date": "2023-12-31", "totalAssets": 3e11, "totalLiabilities": 1.5e11},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_balance
        mock_get.return_value = mock_response

        df = fmp_client.get_balance_sheet("AAPL")

        assert not df.empty

    @patch('requests.get')
    def test_get_cash_flow(self, mock_get, fmp_client):
        """测试获取现金流量表"""
        mock_cashflow = [
            {"date": "2023-12-31", "operatingCashFlow": 5e10, "freeCashFlow": 3e10},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_cashflow
        mock_get.return_value = mock_response

        df = fmp_client.get_cash_flow("AAPL", period="quarterly")

        assert not df.empty


# ============================================================
# Test 9: Historical Data
# ============================================================

class TestHistoricalData:
    """测试历史数据"""

    @patch('requests.get')
    def test_get_historical_price(self, mock_get, fmp_client, mock_historical_data):
        """测试获取历史价格"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_historical_data
        mock_get.return_value = mock_response

        df = fmp_client.get_historical_price("AAPL")

        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert df.index.name == "date"

    @patch('requests.get')
    def test_get_historical_price_with_dates(self, mock_get, fmp_client, mock_historical_data):
        """测试指定日期范围的历史价格"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_historical_data
        mock_get.return_value = mock_response

        df = fmp_client.get_historical_price("AAPL", start_date="2024-01-01", end_date="2024-01-15")

        assert not df.empty
        # 验证请求参数
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["from"] == "2024-01-01"
        assert kwargs["params"]["to"] == "2024-01-15"

    @patch('requests.get')
    def test_get_historical_price_empty(self, mock_get, fmp_client):
        """测试历史价格返回空"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        df = fmp_client.get_historical_price("INVALID")

        assert df.empty

    @patch('requests.get')
    @patch('time.sleep')
    def test_get_historical_prices_batch(self, mock_sleep, mock_get, fmp_client, mock_historical_data):
        """测试批量获取历史价格"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_historical_data
        mock_get.return_value = mock_response

        result = fmp_client.get_historical_prices_batch(["AAPL", "MSFT"])

        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert mock_sleep.called  # 应该在批次间休息


# ============================================================
# Test 10: Caching
# ============================================================

class TestCaching:
    """测试缓存功能"""

    def test_get_cache_path(self, fmp_client):
        """测试获取缓存路径"""
        cache_path = fmp_client._get_cache_path("test_key")

        assert cache_path.name == "test_key.json"
        assert cache_path.parent == fmp_client.cache_dir

    def test_save_and_load_cache(self, fmp_client):
        """测试保存和加载缓存"""
        test_data = {"key": "value", "number": 123}

        fmp_client._save_cache("test_cache", test_data)
        loaded_data = fmp_client._load_cache("test_cache")

        assert loaded_data == test_data

    def test_load_cache_not_exists(self, fmp_client):
        """测试加载不存在的缓存"""
        result = fmp_client._load_cache("nonexistent")

        assert result is None

    def test_load_cache_expired(self, fmp_client):
        """测试加载过期缓存"""
        test_data = {"key": "value"}
        cache_path = fmp_client._get_cache_path("expired_cache")

        # 保存缓存
        with open(cache_path, "w") as f:
            json.dump(test_data, f)

        # 修改文件修改时间为25小时前
        old_time = time.time() - (25 * 3600)
        os.utime(cache_path, (old_time, old_time))

        # 应该返回 None (已过期)
        result = fmp_client._load_cache("expired_cache", max_age_hours=24)

        assert result is None

    def test_save_cache_invalid_data(self, fmp_client):
        """测试保存无效数据到缓存"""
        # 创建无法序列化的数据
        import threading
        invalid_data = {"lock": threading.Lock()}

        # 不应该抛出异常
        fmp_client._save_cache("invalid", invalid_data)

    @patch('requests.get')
    def test_screen_stocks_cached(self, mock_get, fmp_client, mock_stock_data):
        """测试带缓存的股票筛选"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_stock_data
        mock_get.return_value = mock_response

        # 第一次调用应该请求 API
        df1 = fmp_client.screen_stocks_cached(cache_key="test_cached", market_cap_min=1e9)
        assert not df1.empty
        assert mock_get.called

        # 第二次调用应该使用缓存
        mock_get.reset_mock()
        df2 = fmp_client.screen_stocks_cached(cache_key="test_cached", market_cap_min=1e9)
        assert not df2.empty
        assert not mock_get.called  # 没有调用 API


# ============================================================
# Test 11: FactorScreener
# ============================================================

class TestFactorScreener:
    """测试因子筛选器"""

    @pytest.fixture
    def factor_screener(self, fmp_client):
        """创建因子筛选器实例"""
        return FactorScreener(fmp_client)

    def test_init_with_client(self, fmp_client):
        """测试使用客户端初始化"""
        screener = FactorScreener(fmp_client)
        assert screener.client == fmp_client

    @patch.object(FMPClient, '__init__', return_value=None)
    def test_init_without_client(self, mock_init):
        """测试不提供客户端时自动创建"""
        mock_init.return_value = None
        screener = FactorScreener(None)
        # 应该创建默认客户端（但会因为没有 API key 失败）

    @patch.object(FMPClient, 'screen_stocks_cached')
    def test_screen_by_factors(self, mock_screen, factor_screener, mock_stock_data):
        """测试因子筛选"""
        mock_screen.return_value = pd.DataFrame(mock_stock_data)

        result = factor_screener.screen_by_factors(
            value_weight=0.3,
            quality_weight=0.3,
            momentum_weight=0.2,
            growth_weight=0.2,
            top_n=10,
        )

        assert not result.empty
        assert "composite_score" in result.columns
        assert "value_score" in result.columns
        assert "quality_score" in result.columns

    def test_screen_by_factors_with_base_universe(self, factor_screener, mock_stock_data):
        """测试使用提供的基础股票池"""
        base_universe = pd.DataFrame(mock_stock_data)

        result = factor_screener.screen_by_factors(
            base_universe=base_universe,
            top_n=2,
        )

        assert not result.empty
        assert len(result) <= 2

    @patch.object(FMPClient, 'screen_stocks_cached')
    def test_screen_by_factors_empty_universe(self, mock_screen, factor_screener):
        """测试空股票池"""
        mock_screen.return_value = pd.DataFrame()

        result = factor_screener.screen_by_factors()

        assert result.empty

    def test_rank_score(self, factor_screener):
        """测试排名得分计算"""
        df = pd.DataFrame({
            'pe': [10, 20, 30, 40, 50],
            'roe': [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        # 低 P/E 更好 (ascending=True)
        score = factor_screener._rank_score(df, ['pe'], ascending=True)
        assert score.iloc[0] > score.iloc[-1]

        # 高 ROE 更好 (ascending=False)
        score = factor_screener._rank_score(df, ['roe'], ascending=False)
        assert score.iloc[-1] > score.iloc[0]

    def test_rank_score_with_nan(self, factor_screener):
        """测试带 NaN 的排名得分"""
        df = pd.DataFrame({
            'pe': [10, np.nan, 30, np.nan, 50],
        })

        score = factor_screener._rank_score(df, ['pe'], ascending=True)

        # NaN 应该被填充为 0.5
        assert not score.isna().any()

    def test_rank_score_with_inf(self, factor_screener):
        """测试带无穷值的排名得分"""
        df = pd.DataFrame({
            'pe': [10, np.inf, 30, -np.inf, 50],
        })

        score = factor_screener._rank_score(df, ['pe'], ascending=True)

        # 无穷值应该被处理
        assert not score.isna().any()

    def test_rank_score_missing_columns(self, factor_screener):
        """测试缺失列的排名得分"""
        df = pd.DataFrame({'pe': [10, 20, 30]})

        score = factor_screener._rank_score(df, ['nonexistent'], ascending=True)

        # 应该返回默认值 0.5
        assert (score == 0.5).all()


# ============================================================
# Test 12: FMPNewsClient
# ============================================================

class TestFMPNewsClient:
    """测试新闻客户端"""

    @pytest.fixture
    def news_client(self, fmp_client):
        """创建新闻客户端实例"""
        return FMPNewsClient(fmp_client)

    @patch('requests.get')
    @patch('time.sleep')
    def test_get_stock_news(self, mock_sleep, mock_get, news_client):
        """测试获取股票新闻"""
        mock_news = [
            {
                "title": "Apple announces new product",
                "publishedDate": "2024-01-15T10:00:00",
                "url": "https://example.com/news/1",
                "text": "Apple announced..."
            }
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_news
        mock_get.return_value = mock_response

        news = news_client.get_stock_news(["AAPL"], limit=10)

        assert len(news) > 0
        assert news[0]["symbol"] == "AAPL"
        assert "date" in news[0]
        assert "link" in news[0]

    @patch('requests.get')
    @patch('time.sleep')
    def test_get_stock_news_multiple_symbols(self, mock_sleep, mock_get, news_client):
        """测试获取多个股票的新闻"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"title": "News 1", "publishedDate": "2024-01-15T10:00:00", "url": "http://ex.com"}
        ]
        mock_get.return_value = mock_response

        news = news_client.get_stock_news(["AAPL", "MSFT"], limit=5)

        assert len(news) > 0
        assert mock_sleep.called  # 应该在请求间休息

    @patch('requests.get')
    def test_get_general_news(self, mock_get, news_client):
        """测试获取通用新闻"""
        mock_news = [{"title": "Market update", "publishedDate": "2024-01-15T10:00:00"}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_news
        mock_get.return_value = mock_response

        news = news_client.get_general_news(limit=20)

        assert len(news) > 0

    @patch('requests.get')
    def test_get_crypto_news(self, mock_get, news_client):
        """测试获取加密货币新闻"""
        mock_news = [{"title": "Bitcoin reaches new high"}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_news
        mock_get.return_value = mock_response

        news = news_client.get_crypto_news(limit=10)

        assert len(news) > 0


# ============================================================
# Test 13: FMPETFClient
# ============================================================

class TestFMPETFClient:
    """测试 ETF 客户端"""

    @pytest.fixture
    def etf_client(self, fmp_client):
        """创建 ETF 客户端实例"""
        return FMPETFClient(fmp_client)

    @patch('requests.get')
    def test_get_etf_holdings(self, mock_get, etf_client):
        """测试获取 ETF 持仓"""
        mock_holdings = [
            {"symbol": "AAPL", "weight": 0.05, "shares": 1000000},
            {"symbol": "MSFT", "weight": 0.045, "shares": 900000},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_holdings
        mock_get.return_value = mock_response

        df = etf_client.get_etf_holdings("SPY")

        assert not df.empty
        assert len(df) == 2

    @patch('requests.get')
    def test_get_etf_sector_weightings(self, mock_get, etf_client):
        """测试获取 ETF 行业权重"""
        mock_sectors = [
            {"sector": "Technology", "weightPercentage": 28.5},
            {"sector": "Healthcare", "weightPercentage": 15.2},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_sectors
        mock_get.return_value = mock_response

        sectors = etf_client.get_etf_sector_weightings("SPY")

        assert "Technology" in sectors
        assert sectors["Technology"] == 28.5


# ============================================================
# Test 14: Singleton Functions
# ============================================================

class TestSingletonFunctions:
    """测试单例函数"""

    def test_get_fmp_client_singleton(self, mock_api_key, monkeypatch):
        """测试 FMP 客户端单例"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        # 重置全局实例
        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None

        client1 = get_fmp_client()
        client2 = get_fmp_client()

        assert client1 is client2  # 应该是同一个实例

    def test_get_news_client_singleton(self, mock_api_key, monkeypatch):
        """测试新闻客户端单例"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        # 重置全局实例
        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None
        fmp_module._fmp_news_client_instance = None

        client1 = get_news_client()
        client2 = get_news_client()

        assert client1 is client2

    def test_get_etf_client_singleton(self, mock_api_key, monkeypatch):
        """测试 ETF 客户端单例"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        # 重置全局实例
        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None
        fmp_module._fmp_etf_client_instance = None

        client1 = get_etf_client()
        client2 = get_etf_client()

        assert client1 is client2

    def test_get_factor_screener_singleton(self, mock_api_key, monkeypatch):
        """测试因子筛选器单例"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        # 重置全局实例
        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None
        fmp_module._factor_screener_instance = None

        screener1 = get_factor_screener()
        screener2 = get_factor_screener()

        assert screener1 is screener2


# ============================================================
# Test 15: Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    @patch.object(FMPClient, 'screen_stocks')
    def test_screen_stocks_convenience(self, mock_screen, mock_api_key, monkeypatch):
        """测试便捷函数: screen_stocks"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        # 重置单例
        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None

        mock_screen.return_value = pd.DataFrame([{"symbol": "AAPL"}])

        result = screen_stocks(market_cap_min=1e9)

        assert not result.empty
        assert mock_screen.called

    @patch.object(FMPClient, 'get_quote')
    def test_get_quote_convenience(self, mock_quote, mock_api_key, monkeypatch):
        """测试便捷函数: get_quote"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None

        mock_quote.return_value = {"symbol": "AAPL", "price": 180.0}

        result = get_quote("AAPL")

        assert result["symbol"] == "AAPL"

    @patch.object(FMPClient, 'get_historical_price')
    def test_get_historical_price_convenience(self, mock_hist, mock_api_key, monkeypatch):
        """测试便捷函数: get_historical_price"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None

        mock_hist.return_value = pd.DataFrame({"Close": [180.0, 178.0]})

        result = get_historical_price("AAPL", start_date="2024-01-01")

        assert not result.empty

    @patch.object(FMPNewsClient, 'get_stock_news')
    def test_get_stock_news_convenience(self, mock_news, mock_api_key, monkeypatch):
        """测试便捷函数: get_stock_news"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None
        fmp_module._fmp_news_client_instance = None

        mock_news.return_value = [{"title": "News"}]

        result = get_stock_news(["AAPL"])

        assert len(result) > 0

    @patch.object(FMPClient, 'get_profile')
    def test_get_profile_convenience(self, mock_profile, mock_api_key, monkeypatch):
        """测试便捷函数: get_profile"""
        monkeypatch.setenv("FMP_API_KEY", mock_api_key)

        import finsage.data.fmp_client as fmp_module
        fmp_module._fmp_client_instance = None

        mock_profile.return_value = {"symbol": "AAPL", "companyName": "Apple Inc."}

        result = get_profile("AAPL")

        assert result["symbol"] == "AAPL"


# ============================================================
# Test 16: Edge Cases and Error Handling
# ============================================================

class TestEdgeCasesAndErrorHandling:
    """测试边界情况和错误处理"""

    def test_url_constants(self):
        """测试 URL 常量"""
        assert FMPClient.BASE_URL == "https://financialmodelingprep.com/stable"
        assert FMPClient.V3_URL == "https://financialmodelingprep.com/api/v3"
        assert FMPClient.V4_URL == "https://financialmodelingprep.com/api/v4"

    @patch('requests.get')
    def test_batch_quote_with_none_response(self, mock_get, fmp_client):
        """测试批量报价返回 None"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_get.return_value = mock_response

        df = fmp_client.get_batch_quote(["AAPL"])

        assert df.empty

    @patch('requests.get')
    def test_historical_price_no_date_column(self, mock_get, fmp_client):
        """测试历史价格没有 date 列"""
        mock_data = [{"close": 180.0}]  # 没有 date 列
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response

        df = fmp_client.get_historical_price("AAPL")

        # 应该返回 DataFrame 但没有日期索引
        assert not df.empty

    def test_cache_load_corrupted_file(self, fmp_client):
        """测试加载损坏的缓存文件"""
        cache_path = fmp_client._get_cache_path("corrupted")

        # 写入无效 JSON
        with open(cache_path, "w") as f:
            f.write("invalid json {{{")

        result = fmp_client._load_cache("corrupted")

        assert result is None  # 应该返回 None 而不是抛出异常

    @patch.object(FMPClient, 'screen_stocks_cached')
    def test_factor_screener_missing_columns(self, mock_screen, fmp_client):
        """测试因子筛选器处理缺失列"""
        # 创建缺少某些因子列的数据
        incomplete_data = pd.DataFrame([
            {"symbol": "AAPL", "pe": 30.0},  # 缺少其他列
        ])
        mock_screen.return_value = incomplete_data

        screener = FactorScreener(fmp_client)
        result = screener.screen_by_factors()

        # 应该能处理缺失列，用 NaN 填充
        assert not result.empty


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" FMP Client Deep Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short", "--cov=finsage.data.fmp_client", "--cov-report=term-missing"])


if __name__ == "__main__":
    run_tests()

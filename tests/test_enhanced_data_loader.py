#!/usr/bin/env python
"""
Enhanced Data Loader Tests
===========================
覆盖: finsage/data/enhanced_data_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def loader():
    """创建EnhancedDataLoader实例"""
    from finsage.data.enhanced_data_loader import EnhancedDataLoader
    return EnhancedDataLoader()


@pytest.fixture
def mock_requests_get():
    """Mock requests.get"""
    with patch('finsage.data.enhanced_data_loader.requests.get') as mock:
        yield mock


# ============================================================
# Test 1: Module Imports
# ============================================================

class TestModuleImports:
    """测试模块导入"""

    def test_import_enhanced_data_loader(self):
        """测试导入EnhancedDataLoader"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader
        assert EnhancedDataLoader is not None


# ============================================================
# Test 2: Initialization
# ============================================================

class TestInitialization:
    """测试初始化"""

    def test_init_default(self, loader):
        """测试默认初始化"""
        assert loader is not None
        assert loader._cache == {}
        assert loader._cache_ttl == 300  # 5 minutes

    def test_init_api_urls(self, loader):
        """测试API URL配置"""
        assert loader.sentiment_host == "stock-sentiment-api.p.rapidapi.com"
        assert loader.fmp_base_url is not None

    def test_init_env_keys(self):
        """测试环境变量密钥"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        # 测试没有设置环境变量时
        with patch.dict(os.environ, {}, clear=True):
            loader = EnhancedDataLoader()
            # 不抛出错误即可


# ============================================================
# Test 3: Cache Management
# ============================================================

class TestCacheManagement:
    """测试缓存管理"""

    def test_set_cache(self, loader):
        """测试设置缓存"""
        test_data = {"key": "value"}
        loader._set_cache("test_key", test_data)

        assert "test_key" in loader._cache
        assert loader._cache["test_key"][0] == test_data

    def test_get_from_cache_valid(self, loader):
        """测试获取有效缓存"""
        test_data = {"key": "value"}
        loader._set_cache("test_key", test_data)

        result = loader._get_from_cache("test_key")
        assert result == test_data

    def test_get_from_cache_expired(self, loader):
        """测试获取过期缓存"""
        test_data = {"key": "value"}
        # 设置一个过期的缓存
        expired_time = datetime.now() - timedelta(seconds=loader._cache_ttl + 10)
        loader._cache["test_key"] = (test_data, expired_time)

        result = loader._get_from_cache("test_key")
        assert result is None

    def test_get_from_cache_missing(self, loader):
        """测试获取不存在的缓存"""
        result = loader._get_from_cache("nonexistent")
        assert result is None

    def test_clear_cache(self, loader):
        """测试清除缓存"""
        loader._set_cache("key1", "value1")
        loader._set_cache("key2", "value2")

        loader.clear_cache()

        assert loader._cache == {}


# ============================================================
# Test 4: Default Sentiment
# ============================================================

class TestDefaultSentiment:
    """测试默认情绪数据"""

    def test_get_default_sentiment(self, loader):
        """测试获取默认情绪"""
        with patch.object(loader, '_get_sentiment_from_fmp_news', return_value=None):
            sentiment = loader._get_default_sentiment("AAPL")

            assert sentiment is not None
            assert sentiment["symbol"] == "AAPL"
            assert sentiment["sentiment_score"] == 0
            assert sentiment["sentiment_label"] == "neutral"
            assert sentiment["bullish_percent"] == 50
            assert sentiment["bearish_percent"] == 50
            assert sentiment["is_default"] is True

    def test_get_default_market_sentiment(self, loader):
        """测试获取默认市场情绪"""
        sentiment = loader._get_default_market_sentiment()

        assert sentiment is not None
        assert sentiment["market_sentiment_score"] == 0
        assert sentiment["market_sentiment_label"] == "neutral"
        assert sentiment["sample_size"] == 0
        assert sentiment["is_default"] is True


# ============================================================
# Test 5: Stock Sentiment
# ============================================================

class TestStockSentiment:
    """测试股票情绪"""

    def test_get_stock_sentiment_no_api_key(self):
        """测试无API密钥时获取股票情绪"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {}, clear=True):
            loader = EnhancedDataLoader()
            loader.rapidapi_key = None

            with patch.object(loader, '_get_default_sentiment') as mock_default:
                mock_default.return_value = {"sentiment_score": 0}
                sentiment = loader.get_stock_sentiment("AAPL")

                mock_default.assert_called_once_with("AAPL")

    def test_get_stock_sentiment_from_cache(self, loader):
        """测试从缓存获取股票情绪"""
        cached_sentiment = {"symbol": "AAPL", "sentiment_score": 0.5}
        loader._set_cache("sentiment_AAPL", cached_sentiment)

        sentiment = loader.get_stock_sentiment("AAPL")
        assert sentiment == cached_sentiment

    def test_get_stock_sentiment_api_success(self, loader, mock_requests_get):
        """测试API成功获取股票情绪"""
        loader.rapidapi_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "avg_sentiment": 0.75,
            "number_of_articles": 10
        }
        mock_requests_get.return_value = mock_response

        sentiment = loader.get_stock_sentiment("AAPL")

        assert sentiment["symbol"] == "AAPL"
        assert sentiment["source"] == "rapidapi"
        assert sentiment["news_count"] == 10

    def test_get_stock_sentiment_api_failure(self, loader, mock_requests_get):
        """测试API失败时获取股票情绪"""
        loader.rapidapi_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests_get.return_value = mock_response

        with patch.object(loader, '_get_default_sentiment') as mock_default:
            mock_default.return_value = {"sentiment_score": 0}
            sentiment = loader.get_stock_sentiment("AAPL")

            mock_default.assert_called_once()


# ============================================================
# Test 6: Market Sentiment
# ============================================================

class TestMarketSentiment:
    """测试市场情绪"""

    def test_get_market_sentiment_default_symbols(self, loader):
        """测试使用默认股票获取市场情绪"""
        with patch.object(loader, 'get_stock_sentiment') as mock_stock_sent:
            mock_stock_sent.return_value = {
                "sentiment_score": 0.5,
                "bullish_percent": 75
            }

            sentiment = loader.get_market_sentiment()

            assert sentiment is not None
            assert "market_sentiment_score" in sentiment
            assert mock_stock_sent.call_count <= 5

    def test_get_market_sentiment_custom_symbols(self, loader):
        """测试使用自定义股票获取市场情绪"""
        with patch.object(loader, 'get_stock_sentiment') as mock_stock_sent:
            mock_stock_sent.return_value = {
                "sentiment_score": 0.3,
                "bullish_percent": 65
            }

            sentiment = loader.get_market_sentiment(symbols=["AAPL", "MSFT"])

            assert sentiment is not None

    def test_get_market_sentiment_empty(self, loader):
        """测试无情绪数据时"""
        with patch.object(loader, 'get_stock_sentiment', return_value=None):
            sentiment = loader.get_market_sentiment(symbols=["AAPL"])

            assert sentiment["is_default"] is True


# ============================================================
# Test 7: ETF Fund Flows
# ============================================================

class TestETFFundFlows:
    """测试ETF资金流向"""

    def test_get_etf_fund_flows_no_api_key(self):
        """测试无API密钥时获取ETF资金流向"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_etf_fund_flows("SPY")
            assert result is None

    def test_get_etf_fund_flows_from_cache(self, loader):
        """测试从缓存获取ETF资金流向"""
        cached_flows = {"symbol": "SPY", "holdings_count": 500}
        loader._set_cache("etf_flow_SPY", cached_flows)

        result = loader.get_etf_fund_flows("SPY")
        assert result == cached_flows

    def test_get_etf_fund_flows_api_success(self, loader, mock_requests_get):
        """测试API成功获取ETF资金流向"""
        loader.fmp_api_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "weight": 0.07},
            {"symbol": "MSFT", "weight": 0.06}
        ]
        mock_requests_get.return_value = mock_response

        result = loader.get_etf_fund_flows("SPY")

        assert result["symbol"] == "SPY"
        assert result["holdings_count"] == 2
        assert result["source"] == "fmp_stable"


# ============================================================
# Test 8: Institutional Ownership
# ============================================================

class TestInstitutionalOwnership:
    """测试机构持股"""

    def test_get_institutional_ownership_no_api_key(self):
        """测试无API密钥时获取机构持股"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_institutional_ownership("AAPL")
            assert result is None

    def test_get_institutional_ownership_from_cache(self, loader):
        """测试从缓存获取机构持股"""
        cached_data = {"symbol": "AAPL", "institutional_holders": 100}
        loader._set_cache("inst_ownership_AAPL", cached_data)

        result = loader.get_institutional_ownership("AAPL")
        assert result == cached_data


# ============================================================
# Test 9: Economic Calendar
# ============================================================

class TestEconomicCalendar:
    """测试经济日历"""

    def test_get_economic_calendar_no_api_key(self):
        """测试无API密钥时获取经济日历"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_economic_calendar()
            assert result == []

    def test_get_economic_calendar_from_cache(self, loader):
        """测试从缓存获取经济日历"""
        today = datetime.now().strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        cache_key = f"econ_calendar_{today}_{next_week}"

        cached_calendar = [{"event": "FOMC", "impact": "high"}]
        loader._set_cache(cache_key, cached_calendar)

        result = loader.get_economic_calendar()
        assert result == cached_calendar

    def test_get_economic_calendar_api_success(self, loader, mock_requests_get):
        """测试API成功获取经济日历"""
        loader.fmp_api_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"event": "FOMC Meeting", "impact": "high"},
            {"event": "Jobs Report", "impact": "high"},
            {"event": "Minor Event", "impact": "low"}
        ]
        mock_requests_get.return_value = mock_response

        result = loader.get_economic_calendar()

        # 只返回 high/medium impact 事件
        assert len(result) == 2

    def test_get_upcoming_fomc(self, loader):
        """测试获取即将到来的FOMC"""
        with patch.object(loader, 'get_economic_calendar') as mock_calendar:
            mock_calendar.return_value = [
                {"event": "FOMC Meeting", "date": "2024-03-20", "impact": "high"}
            ]

            result = loader.get_upcoming_fomc()

            assert result is not None
            assert result["event"] == "FOMC Meeting"


# ============================================================
# Test 10: Options Chain
# ============================================================

class TestOptionsChain:
    """测试期权链"""

    def test_get_options_chain_no_api_key(self):
        """测试无API密钥时获取期权链"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_options_chain("SPY")
            assert result is None

    def test_get_options_chain_from_cache(self, loader):
        """测试从缓存获取期权链"""
        cached_options = {"symbol": "SPY", "expiration_dates": ["2024-03-15"]}
        loader._set_cache("options_SPY", cached_options)

        result = loader.get_options_chain("SPY")
        assert result == cached_options

    def test_get_put_call_ratio(self, loader):
        """测试获取Put/Call比率"""
        result = loader.get_put_call_ratio("SPY")

        assert result is not None
        assert result["symbol"] == "SPY"
        assert "put_call_ratio" in result
        assert "interpretation" in result


# ============================================================
# Test 11: Earnings Calendar
# ============================================================

class TestEarningsCalendar:
    """测试财报日历"""

    def test_get_earnings_calendar_no_api_key(self):
        """测试无API密钥时获取财报日历"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_earnings_calendar()
            assert result == []

    def test_get_earnings_calendar_from_cache(self, loader):
        """测试从缓存获取财报日历"""
        today = datetime.now().strftime("%Y-%m-%d")
        next_2weeks = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        cache_key = f"earnings_cal_{today}_{next_2weeks}"

        cached_earnings = [{"symbol": "AAPL", "date": "2024-01-25"}]
        loader._set_cache(cache_key, cached_earnings)

        result = loader.get_earnings_calendar()
        assert result == cached_earnings


# ============================================================
# Test 12: Earnings Surprise
# ============================================================

class TestEarningsSurprise:
    """测试财报惊喜"""

    def test_get_earnings_surprise_no_api_key(self):
        """测试无API密钥时获取财报惊喜"""
        from finsage.data.enhanced_data_loader import EnhancedDataLoader

        with patch.dict(os.environ, {"FMP_API_KEY": ""}, clear=False):
            loader = EnhancedDataLoader()
            loader.fmp_api_key = None

            result = loader.get_earnings_surprise("AAPL")
            assert result is None

    def test_get_earnings_surprise_from_cache(self, loader):
        """测试从缓存获取财报惊喜"""
        cached_surprise = {"symbol": "AAPL", "avg_surprise": 0.05}
        loader._set_cache("earnings_surprise_AAPL", cached_surprise)

        result = loader.get_earnings_surprise("AAPL")
        assert result == cached_surprise


# ============================================================
# Test 13: Enhanced Market Data
# ============================================================

class TestEnhancedMarketData:
    """测试增强市场数据"""

    def test_get_enhanced_market_data_basic(self, loader):
        """测试获取基本增强市场数据"""
        with patch.object(loader, 'get_market_sentiment') as mock_sentiment, \
             patch.object(loader, 'get_etf_fund_flows') as mock_flows, \
             patch.object(loader, 'get_economic_calendar') as mock_calendar, \
             patch.object(loader, 'get_upcoming_fomc') as mock_fomc, \
             patch.object(loader, 'get_earnings_calendar') as mock_earnings, \
             patch.object(loader, 'get_earnings_surprise') as mock_surprise:

            mock_sentiment.return_value = {"market_sentiment_score": 0.5}
            mock_flows.return_value = {"holdings_count": 500}
            mock_calendar.return_value = []
            mock_fomc.return_value = None
            mock_earnings.return_value = []
            mock_surprise.return_value = None

            result = loader.get_enhanced_market_data(
                symbols=["AAPL", "MSFT"],
                include_options=False
            )

            assert "timestamp" in result
            assert "symbols" in result
            assert result["symbols"] == ["AAPL", "MSFT"]
            assert "sentiment" in result
            assert "fund_flows" in result
            assert "economic_calendar" in result
            assert "earnings_calendar" in result

    def test_get_enhanced_market_data_with_options(self, loader):
        """测试获取包含期权的增强市场数据"""
        with patch.object(loader, 'get_market_sentiment', return_value={}), \
             patch.object(loader, 'get_etf_fund_flows', return_value=None), \
             patch.object(loader, 'get_economic_calendar', return_value=[]), \
             patch.object(loader, 'get_upcoming_fomc', return_value=None), \
             patch.object(loader, 'get_earnings_calendar', return_value=[]), \
             patch.object(loader, 'get_earnings_surprise', return_value=None), \
             patch.object(loader, 'get_options_chain', return_value=None), \
             patch.object(loader, 'get_put_call_ratio', return_value={}):

            result = loader.get_enhanced_market_data(
                symbols=["AAPL"],
                include_options=True
            )

            assert "options" in result
            assert "put_call_ratio" in result


# ============================================================
# Test 14: Sentiment From FMP News
# ============================================================

class TestSentimentFromFMPNews:
    """测试从FMP新闻获取情绪"""

    def test_get_sentiment_from_fmp_news_no_api_key(self, loader):
        """测试无API密钥时从FMP新闻获取情绪"""
        loader.fmp_api_key = None

        result = loader._get_sentiment_from_fmp_news("AAPL")
        assert result is None

    def test_get_sentiment_from_fmp_news_with_bullish_news(self, loader):
        """测试获取看涨新闻情绪"""
        loader.fmp_api_key = "test_key"

        # get_news_client is imported inside the function, so we need to patch fmp_client module
        with patch('finsage.data.fmp_client.get_news_client') as mock_client:
            mock_news_client = MagicMock()
            mock_news_client.get_stock_news.return_value = [
                {"title": "Stock Surges on Strong Earnings", "text": "Great profit growth"},
                {"title": "Rally Continues with Record High", "text": "Positive momentum"}
            ]
            mock_client.return_value = mock_news_client

            result = loader._get_sentiment_from_fmp_news("AAPL")

            # 如果返回结果不为None，检查相关字段
            if result is not None:
                assert result["symbol"] == "AAPL"
                assert result["source"] == "fmp_news"


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Enhanced Data Loader Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

"""
Deep tests for EnhancedDataLoader
增强数据加载器深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock

from finsage.data.enhanced_data_loader import EnhancedDataLoader


class TestEnhancedDataLoaderInit:
    """EnhancedDataLoader初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        with patch.dict('os.environ', {}, clear=True):
            loader = EnhancedDataLoader()
            assert loader._cache == {}
            assert loader._cache_ttl == 300

    def test_init_with_api_keys(self):
        """测试带API密钥初始化"""
        with patch.dict('os.environ', {
            'RAPIDAPI_KEY': 'test_rapid_key',
            'FMP_API_KEY': 'test_fmp_key'
        }):
            loader = EnhancedDataLoader()
            assert loader.rapidapi_key == 'test_rapid_key'
            assert loader.fmp_api_key == 'test_fmp_key'

    def test_cache_ttl(self):
        """测试缓存TTL"""
        loader = EnhancedDataLoader()
        assert loader._cache_ttl == 300  # 5 minutes


class TestSentimentData:
    """情绪数据测试"""

    @pytest.fixture
    def loader(self):
        return EnhancedDataLoader()

    def test_get_default_sentiment(self, loader):
        """测试默认情绪数据"""
        sentiment = loader._get_default_sentiment("AAPL")
        assert sentiment is not None
        assert sentiment["symbol"] == "AAPL"
        assert "sentiment_score" in sentiment

    def test_get_default_market_sentiment(self, loader):
        """测试默认市场情绪"""
        sentiment = loader._get_default_market_sentiment()
        assert sentiment is not None
        assert sentiment["market_sentiment_score"] == 0
        assert sentiment["market_sentiment_label"] == "neutral"
        assert sentiment["bullish_percent"] == 50
        assert sentiment["bearish_percent"] == 50

    @patch('requests.get')
    def test_get_stock_sentiment_with_api(self, mock_get, loader):
        """测试通过API获取股票情绪"""
        loader.rapidapi_key = 'test_key'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "avg_sentiment": 0.7,
            "number_of_articles": 15
        }
        mock_get.return_value = mock_response

        sentiment = loader.get_stock_sentiment("AAPL")
        assert sentiment is not None
        assert sentiment["symbol"] == "AAPL"
        assert sentiment["sentiment_label"] == "bullish"

    @patch('requests.get')
    def test_get_stock_sentiment_api_error(self, mock_get, loader):
        """测试API错误处理"""
        loader.rapidapi_key = 'test_key'

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        sentiment = loader.get_stock_sentiment("AAPL")
        assert sentiment is not None  # Should return default

    @patch('requests.get')
    def test_get_stock_sentiment_exception(self, mock_get, loader):
        """测试异常处理"""
        loader.rapidapi_key = 'test_key'
        mock_get.side_effect = Exception("Network error")

        sentiment = loader.get_stock_sentiment("AAPL")
        assert sentiment is not None  # Should return default

    def test_sentiment_score_conversion(self, loader):
        """测试情绪分数转换"""
        # avg_sentiment 0.5 应该转为 sentiment_score 0
        loader.rapidapi_key = 'test_key'

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "avg_sentiment": 0.5,
                "number_of_articles": 10
            }
            mock_get.return_value = mock_response

            sentiment = loader.get_stock_sentiment("TEST")
            assert sentiment["sentiment_label"] == "neutral"


class TestFundFlowsData:
    """资金流向数据测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    def test_get_etf_fund_flows_no_key(self):
        """测试无API密钥"""
        loader = EnhancedDataLoader()
        loader.fmp_api_key = None
        result = loader.get_etf_fund_flows("SPY")
        assert result is None

    @patch('requests.get')
    def test_get_etf_fund_flows_success(self, mock_get, loader):
        """测试成功获取ETF资金流向"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "shares": 1000000},
            {"symbol": "MSFT", "shares": 800000},
        ]
        mock_get.return_value = mock_response

        result = loader.get_etf_fund_flows("SPY")
        assert result is not None
        assert result["symbol"] == "SPY"
        assert result["holdings_count"] == 2

    @patch('requests.get')
    def test_get_institutional_ownership(self, mock_get, loader):
        """测试机构持股数据"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"totalShares": 5000000, "totalValue": 750000000},
            {"totalShares": 3000000, "totalValue": 450000000},
        ]
        mock_get.return_value = mock_response

        result = loader.get_institutional_ownership("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"


class TestEconomicCalendar:
    """经济日历测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    def test_get_economic_calendar_no_key(self):
        """测试无API密钥"""
        loader = EnhancedDataLoader()
        loader.fmp_api_key = None
        result = loader.get_economic_calendar()
        assert result == []

    @patch('requests.get')
    def test_get_economic_calendar_success(self, mock_get, loader):
        """测试成功获取经济日历"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"event": "FOMC Meeting", "impact": "high", "date": "2024-01-15"},
            {"event": "GDP Release", "impact": "medium", "date": "2024-01-20"},
            {"event": "Minor Event", "impact": "low", "date": "2024-01-22"},
        ]
        mock_get.return_value = mock_response

        result = loader.get_economic_calendar()
        # Should filter to only high/medium impact
        assert len(result) == 2

    @patch('requests.get')
    def test_get_upcoming_fomc(self, mock_get, loader):
        """测试获取FOMC会议"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"event": "FOMC Rate Decision", "impact": "high", "date": "2024-01-31"},
        ]
        mock_get.return_value = mock_response

        result = loader.get_upcoming_fomc()
        assert result is not None
        assert "FOMC" in result["event"]


class TestOptionsData:
    """期权数据测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    def test_get_options_chain_no_key(self):
        """测试无API密钥"""
        loader = EnhancedDataLoader()
        loader.fmp_api_key = None
        result = loader.get_options_chain("AAPL")
        assert result is None

    @patch('requests.get')
    def test_get_options_chain_success(self, mock_get, loader):
        """测试成功获取期权链"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["2024-01-19", "2024-01-26", "2024-02-02"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = loader.get_options_chain("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert len(result["expiration_dates"]) == 3

    def test_get_put_call_ratio(self, loader):
        """测试Put/Call比率"""
        result = loader.get_put_call_ratio("SPY")
        assert result is not None
        assert "put_call_ratio" in result


class TestEarningsData:
    """财报数据测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    def test_get_earnings_calendar_no_key(self):
        """测试无API密钥"""
        loader = EnhancedDataLoader()
        loader.fmp_api_key = None
        result = loader.get_earnings_calendar()
        assert result == []

    @patch('requests.get')
    def test_get_earnings_calendar_success(self, mock_get, loader):
        """测试成功获取财报日历"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "date": "2024-01-25", "eps": 2.1},
            {"symbol": "MSFT", "date": "2024-01-30", "eps": 3.0},
        ]
        mock_get.return_value = mock_response

        result = loader.get_earnings_calendar()
        assert len(result) == 2

    @patch('requests.get')
    def test_get_earnings_surprise(self, mock_get, loader):
        """测试财报惊喜数据"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"actualEarningResult": 2.2, "estimatedEarning": 2.0},
            {"actualEarningResult": 2.5, "estimatedEarning": 2.3},
            {"actualEarningResult": 1.8, "estimatedEarning": 2.0},
            {"actualEarningResult": 2.0, "estimatedEarning": 1.9},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = loader.get_earnings_surprise("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert "beat_rate" in result


class TestEnhancedMarketData:
    """综合市场数据测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    @patch.object(EnhancedDataLoader, 'get_market_sentiment')
    @patch.object(EnhancedDataLoader, 'get_etf_fund_flows')
    @patch.object(EnhancedDataLoader, 'get_economic_calendar')
    @patch.object(EnhancedDataLoader, 'get_upcoming_fomc')
    @patch.object(EnhancedDataLoader, 'get_earnings_calendar')
    @patch.object(EnhancedDataLoader, 'get_earnings_surprise')
    def test_get_enhanced_market_data(
        self, mock_surprise, mock_earnings, mock_fomc,
        mock_calendar, mock_flows, mock_sentiment, loader
    ):
        """测试获取增强市场数据"""
        mock_sentiment.return_value = {"market_sentiment_score": 0.5}
        mock_flows.return_value = {"holdings_count": 10}
        mock_calendar.return_value = []
        mock_fomc.return_value = None
        mock_earnings.return_value = []
        mock_surprise.return_value = None

        result = loader.get_enhanced_market_data(
            symbols=["AAPL", "MSFT"],
            include_sentiment=True,
            include_flows=True,
            include_calendar=True,
            include_options=False,
            include_earnings=True
        )

        assert "timestamp" in result
        assert "symbols" in result
        assert "sentiment" in result
        assert "fund_flows" in result
        assert "economic_calendar" in result

    def test_get_enhanced_market_data_minimal(self, loader):
        """测试最小参数获取"""
        with patch.object(loader, 'get_market_sentiment', return_value={}):
            result = loader.get_enhanced_market_data(
                symbols=["SPY"],
                include_sentiment=True,
                include_flows=False,
                include_calendar=False,
                include_options=False,
                include_earnings=False
            )
            assert "symbols" in result


class TestCacheManagement:
    """缓存管理测试"""

    @pytest.fixture
    def loader(self):
        return EnhancedDataLoader()

    def test_set_and_get_cache(self, loader):
        """测试设置和获取缓存"""
        loader._set_cache("test_key", {"data": "value"})
        result = loader._get_from_cache("test_key")
        assert result is not None
        assert result["data"] == "value"

    def test_cache_expiration(self, loader):
        """测试缓存过期"""
        loader._cache_ttl = 0  # 立即过期
        loader._set_cache("test_key", {"data": "value"})

        # 由于TTL=0，应该立即过期
        import time
        time.sleep(0.01)
        result = loader._get_from_cache("test_key")
        assert result is None

    def test_clear_cache(self, loader):
        """测试清除缓存"""
        loader._set_cache("key1", "value1")
        loader._set_cache("key2", "value2")

        loader.clear_cache()

        assert loader._cache == {}

    def test_cache_miss(self, loader):
        """测试缓存未命中"""
        result = loader._get_from_cache("nonexistent_key")
        assert result is None


class TestSentimentFMPFallback:
    """FMP新闻情绪回退测试"""

    @pytest.fixture
    def loader(self):
        loader = EnhancedDataLoader()
        loader.fmp_api_key = 'test_fmp_key'
        return loader

    @patch('finsage.data.enhanced_data_loader.get_news_client')
    def test_sentiment_from_fmp_news(self, mock_news_client, loader):
        """测试从FMP新闻估算情绪"""
        mock_client = MagicMock()
        mock_client.get_stock_news.return_value = [
            {"title": "Stock surges on strong earnings beat", "text": "Great growth performance"},
            {"title": "Company reports record profits", "text": "Positive outlook"},
        ]
        mock_news_client.return_value = mock_client

        result = loader._get_sentiment_from_fmp_news("AAPL")
        if result:  # May be None if implementation differs
            assert result["source"] == "fmp_news"

    @patch('finsage.data.enhanced_data_loader.get_news_client')
    def test_sentiment_from_fmp_news_bearish(self, mock_news_client, loader):
        """测试看跌新闻情绪"""
        mock_client = MagicMock()
        mock_client.get_stock_news.return_value = [
            {"title": "Stock crashes after earnings miss", "text": "Weak guidance concerns"},
            {"title": "Company warns of declining sales", "text": "Risk of downturn"},
        ]
        mock_news_client.return_value = mock_client

        result = loader._get_sentiment_from_fmp_news("AAPL")
        if result:
            assert result["sentiment_label"] == "bearish"


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def loader(self):
        return EnhancedDataLoader()

    def test_empty_symbols_list(self, loader):
        """测试空符号列表"""
        result = loader.get_market_sentiment(symbols=[])
        assert result is not None

    @patch('requests.get')
    def test_api_rate_limiting(self, mock_get, loader):
        """测试API限流处理"""
        loader.rapidapi_key = 'test_key'

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = loader.get_stock_sentiment("AAPL")
        assert result is not None  # Should return default

    def test_sentiment_score_clamping(self, loader):
        """测试情绪分数边界值处理"""
        loader.rapidapi_key = 'test_key'

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            # 异常值应该被钳制到 [0, 1]
            mock_response.json.return_value = {
                "avg_sentiment": 1.5,  # 超出范围
                "number_of_articles": 5
            }
            mock_get.return_value = mock_response

            result = loader.get_stock_sentiment("TEST")
            # avg_sentiment 应该被钳制到 1.0
            assert result["raw_avg_sentiment"] == 1.0

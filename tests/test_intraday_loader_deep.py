"""
Deep tests for IntradayDataLoader

覆盖 finsage/data/intraday_loader.py (目标从63%提升到80%+)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from finsage.data.intraday_loader import IntradayDataLoader, create_fmp_loader


class TestIntradayDataLoaderInit:
    """测试IntradayDataLoader初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch.dict('os.environ', {'OA_FMP_KEY': 'test_key'}):
            loader = IntradayDataLoader()
            assert loader.api_key == 'test_key'
            assert loader.cache_enabled is True
            assert loader.cache_ttl_minutes == 5

    def test_init_with_api_key(self):
        """测试指定API密钥"""
        loader = IntradayDataLoader(fmp_api_key='custom_key')
        assert loader.api_key == 'custom_key'

    def test_init_without_api_key(self):
        """测试无API密钥"""
        with patch.dict('os.environ', {}, clear=True):
            loader = IntradayDataLoader(fmp_api_key=None)
            assert loader.api_key is None

    def test_init_cache_settings(self):
        """测试缓存设置"""
        loader = IntradayDataLoader(
            fmp_api_key='key',
            cache_enabled=False,
            cache_ttl_minutes=10
        )
        assert loader.cache_enabled is False
        assert loader.cache_ttl_minutes == 10


class TestIntradayDataLoaderIntervalMapping:
    """测试时间间隔映射"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_interval_mapping(self, loader):
        """测试间隔映射"""
        assert loader.INTERVAL_MAPPING["1m"] == "1min"
        assert loader.INTERVAL_MAPPING["5m"] == "5min"
        assert loader.INTERVAL_MAPPING["15m"] == "15min"
        assert loader.INTERVAL_MAPPING["30m"] == "30min"
        assert loader.INTERVAL_MAPPING["1h"] == "1hour"
        assert loader.INTERVAL_MAPPING["4h"] == "4hour"

    def test_interval_minutes(self, loader):
        """测试间隔分钟数"""
        assert loader.INTERVAL_MINUTES["1m"] == 1
        assert loader.INTERVAL_MINUTES["5m"] == 5
        assert loader.INTERVAL_MINUTES["1h"] == 60
        assert loader.INTERVAL_MINUTES["4h"] == 240


class TestIntradayDataLoaderCache:
    """测试缓存功能"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_save_and_get_from_cache(self, loader):
        """测试保存和获取缓存"""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        loader._save_to_cache('test_key', df)

        result = loader._get_from_cache('test_key')

        assert result is not None
        assert len(result) == 3

    def test_cache_expiry(self, loader):
        """测试缓存过期"""
        loader.cache_ttl_minutes = 0  # 立即过期
        df = pd.DataFrame({'Close': [100, 101, 102]})
        loader._save_to_cache('test_key', df)

        import time
        time.sleep(0.1)

        result = loader._get_from_cache('test_key')
        assert result is None

    def test_cache_disabled(self, loader):
        """测试禁用缓存"""
        loader.cache_enabled = False
        df = pd.DataFrame({'Close': [100, 101, 102]})
        loader._save_to_cache('test_key', df)

        result = loader._get_from_cache('test_key')
        assert result is None

    def test_clear_cache(self, loader):
        """测试清除缓存"""
        df = pd.DataFrame({'Close': [100]})
        loader._save_to_cache('key1', df)
        loader._save_to_cache('key2', df)

        loader.clear_cache()

        assert loader._get_from_cache('key1') is None
        assert loader._get_from_cache('key2') is None


class TestIntradayDataLoaderLoadHourlyData:
    """测试加载小时数据"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_load_hourly_data(self, loader):
        """测试加载小时数据"""
        with patch.object(loader, '_load_intraday_data') as mock_load:
            mock_load.return_value = {'AAPL': pd.DataFrame({'Close': [100]})}

            result = loader.load_hourly_data(['AAPL'], lookback_hours=24)

            mock_load.assert_called_once()
            assert 'AAPL' in result


class TestIntradayDataLoaderLoadMinuteData:
    """测试加载分钟数据"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_load_minute_data_5m(self, loader):
        """测试加载5分钟数据"""
        with patch.object(loader, '_load_intraday_data') as mock_load:
            mock_load.return_value = {'AAPL': pd.DataFrame({'Close': [100]})}

            result = loader.load_minute_data(['AAPL'], interval='5m')

            mock_load.assert_called_once()

    def test_load_minute_data_invalid_interval(self, loader):
        """测试无效间隔"""
        with pytest.raises(ValueError, match="Invalid interval"):
            loader.load_minute_data(['AAPL'], interval='invalid')


class TestIntradayDataLoaderLoadIntradayData:
    """测试核心日内数据加载方法"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_load_intraday_data_from_cache(self, loader):
        """测试从缓存加载"""
        cached_df = pd.DataFrame({'Close': [100, 101, 102]})
        loader._save_to_cache('AAPL_1h_24', cached_df)

        result = loader._load_intraday_data(['AAPL'], '1h', 24)

        assert 'AAPL' in result
        assert len(result['AAPL']) == 3

    def test_load_intraday_data_from_api(self, loader):
        """测试从API加载"""
        with patch.object(loader, '_load_from_fmp') as mock_fmp:
            mock_fmp.return_value = pd.DataFrame({
                'Open': [100],
                'High': [101],
                'Low': [99],
                'Close': [100.5],
                'Volume': [1000000],
            })

            result = loader._load_intraday_data(['AAPL'], '1h', 24)

            mock_fmp.assert_called_once()
            assert 'AAPL' in result

    def test_load_intraday_data_api_error(self, loader):
        """测试API错误处理"""
        with patch.object(loader, '_load_from_fmp') as mock_fmp:
            mock_fmp.side_effect = Exception("API Error")

            result = loader._load_intraday_data(['AAPL'], '1h', 24)

            assert 'AAPL' not in result


class TestIntradayDataLoaderLoadFromFMP:
    """测试从FMP API加载数据"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_load_from_fmp_success(self, loader):
        """测试成功加载"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [
                {'date': '2024-01-15 10:00:00', 'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000},
                {'date': '2024-01-15 11:00:00', 'open': 100.5, 'high': 102, 'low': 100, 'close': 101.5, 'volume': 1200},
            ]

            result = loader._load_from_fmp('AAPL', '1h', 24, datetime.now())

            assert result is not None
            assert len(result) == 2
            assert 'Close' in result.columns

    def test_load_from_fmp_empty_response(self, loader):
        """测试空响应"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = []

            result = loader._load_from_fmp('AAPL', '1h', 24, datetime.now())

            assert result is None

    def test_load_from_fmp_error_response(self, loader):
        """测试错误响应"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = {'Error Message': 'Invalid symbol'}

            result = loader._load_from_fmp('INVALID', '1h', 24, datetime.now())

            assert result is None

    def test_load_from_fmp_exception(self, loader):
        """测试异常处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("Network error")

            result = loader._load_from_fmp('AAPL', '1h', 24, datetime.now())

            assert result is None


class TestIntradayDataLoaderRealtimeQuote:
    """测试实时报价"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_realtime_quote_success(self, loader):
        """测试成功获取报价"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [{
                'symbol': 'AAPL',
                'price': 180.50,
                'change': 2.5,
                'changesPercentage': 1.4,
                'open': 178.0,
                'dayHigh': 181.0,
                'dayLow': 177.5,
                'previousClose': 178.0,
                'volume': 50000000,
                'avgVolume': 60000000,
                'marketCap': 2800000000000,
                'pe': 28.5,
                'yearHigh': 199.0,
                'yearLow': 140.0,
                'timestamp': 1705334400,
            }]

            result = loader.get_realtime_quote('AAPL')

            assert result is not None
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 180.50
            assert result['change'] == 2.5

    def test_get_realtime_quote_empty(self, loader):
        """测试空响应"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = []

            result = loader.get_realtime_quote('INVALID')

            assert result is None

    def test_get_realtime_quote_error(self, loader):
        """测试错误处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            result = loader.get_realtime_quote('AAPL')

            assert result is None


class TestIntradayDataLoaderRealtimeSnapshot:
    """测试实时快照"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_realtime_snapshot_success(self, loader):
        """测试成功获取多股票快照"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [
                {'symbol': 'AAPL', 'price': 180.5, 'change': 2.5, 'changesPercentage': 1.4,
                 'open': 178, 'dayHigh': 181, 'dayLow': 177.5, 'previousClose': 178, 'volume': 50000000},
                {'symbol': 'MSFT', 'price': 400.0, 'change': 5.0, 'changesPercentage': 1.25,
                 'open': 395, 'dayHigh': 401, 'dayLow': 394, 'previousClose': 395, 'volume': 30000000},
            ]

            result = loader.get_realtime_snapshot(['AAPL', 'MSFT'])

            assert 'AAPL' in result
            assert 'MSFT' in result
            assert result['AAPL']['price'] == 180.5

    def test_get_realtime_snapshot_fallback(self, loader):
        """测试批量查询失败后的降级"""
        with patch.object(loader, '_make_fmp_request') as mock_request, \
             patch.object(loader, 'get_realtime_quote') as mock_quote:
            mock_request.side_effect = Exception("Batch error")
            mock_quote.return_value = {'price': 180.5}

            result = loader.get_realtime_snapshot(['AAPL'])

            assert 'AAPL' in result


class TestIntradayDataLoaderVIX:
    """测试VIX获取"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_vix_level_success(self, loader):
        """测试成功获取VIX"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [{'price': 20.5}]

            vix = loader.get_vix_level()

            assert vix == 20.5

    def test_get_vix_level_fallback(self, loader):
        """测试VIX备用符号"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = [
                [],  # 第一次调用返回空
                [{'price': 21.0}]  # 备用符号
            ]

            vix = loader.get_vix_level()

            assert vix == 21.0

    def test_get_vix_level_error(self, loader):
        """测试错误处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            vix = loader.get_vix_level()

            assert vix is None


class TestIntradayDataLoaderMarketHours:
    """测试市场时段状态"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_market_hours_status_success(self, loader):
        """测试成功获取市场状态"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = {
                'stockExchange': {'isTheStockMarketOpen': True}
            }

            status = loader.get_market_hours_status()

            assert 'stockExchange' in status

    def test_get_market_hours_status_error(self, loader):
        """测试错误处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            status = loader.get_market_hours_status()

            assert status == {}


class TestIntradayDataLoaderStockNews:
    """测试股票新闻"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_stock_news_success(self, loader):
        """测试成功获取新闻"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [
                {'title': 'Apple News', 'text': 'Content', 'publishedDate': '2024-01-15'},
            ]

            news = loader.get_stock_news('AAPL', limit=5)

            assert len(news) == 1
            assert news[0]['title'] == 'Apple News'

    def test_get_stock_news_error(self, loader):
        """测试错误处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            news = loader.get_stock_news('AAPL')

            assert news == []


class TestIntradayDataLoaderPriceChange:
    """测试价格变化"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_get_price_change_success(self, loader):
        """测试成功获取价格变化"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [
                {'1D': 1.5, '5D': 3.2, '1M': 5.0, '3M': 10.0, '1Y': 25.0}
            ]

            change = loader.get_price_change('AAPL')

            assert change is not None
            assert change['1D'] == 1.5

    def test_get_price_change_empty(self, loader):
        """测试空响应"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = []

            change = loader.get_price_change('INVALID')

            assert change is None

    def test_get_price_change_error(self, loader):
        """测试错误处理"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            change = loader.get_price_change('AAPL')

            assert change is None


class TestIntradayDataLoaderAPIConnection:
    """测试API连接检查"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_check_api_connection_success(self, loader):
        """测试API连接成功"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = [{'symbol': 'AAPL', 'price': 180.0}]

            result = loader.check_api_connection()

            assert result is True

    def test_check_api_connection_failure(self, loader):
        """测试API连接失败"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.side_effect = Exception("Connection Error")

            result = loader.check_api_connection()

            assert result is False

    def test_check_api_connection_empty(self, loader):
        """测试API返回空"""
        with patch.object(loader, '_make_fmp_request') as mock_request:
            mock_request.return_value = []

            result = loader.check_api_connection()

            assert result is False


class TestIntradayDataLoaderMakeFMPRequest:
    """测试FMP API请求"""

    @pytest.fixture
    def loader(self):
        return IntradayDataLoader(fmp_api_key='test_key')

    def test_make_fmp_request_no_api_key(self):
        """测试无API密钥"""
        from tenacity import RetryError

        with patch.dict('os.environ', {}, clear=True):
            loader = IntradayDataLoader(fmp_api_key=None)
            loader.api_key = None  # 确保API密钥为None

            # 由于retry装饰器，ValueError会被包装成RetryError
            with pytest.raises((ValueError, RetryError)):
                loader._make_fmp_request('/quote')

    def test_make_fmp_request_success(self, loader):
        """测试成功请求"""
        with patch.object(loader.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [{'price': 180.0}]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = loader._make_fmp_request('/quote', {'symbol': 'AAPL'})

            assert result == [{'price': 180.0}]


class TestCreateFMPLoader:
    """测试便捷函数"""

    def test_create_fmp_loader(self):
        """测试创建加载器"""
        loader = create_fmp_loader('test_key')

        assert isinstance(loader, IntradayDataLoader)
        assert loader.api_key == 'test_key'

    def test_create_fmp_loader_no_key(self):
        """测试无密钥创建"""
        with patch.dict('os.environ', {}, clear=True):
            loader = create_fmp_loader()
            assert loader is not None

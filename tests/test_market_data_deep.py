"""
Deep tests for MarketDataProvider

覆盖 finsage/data/market_data.py (目标从57%提升到80%+)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from finsage.data.market_data import MarketDataProvider


class TestMarketDataProviderInit:
    """测试MarketDataProvider初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch('finsage.data.market_data.DataLoader'), \
             patch('finsage.data.market_data.MacroDataLoader'), \
             patch('finsage.data.market_data.EnhancedDataLoader'):
            provider = MarketDataProvider()
            assert provider is not None

    def test_init_with_loaders(self):
        """测试指定加载器"""
        mock_loader = Mock()
        mock_macro = Mock()
        mock_enhanced = Mock()

        provider = MarketDataProvider(
            data_loader=mock_loader,
            macro_loader=mock_macro,
            enhanced_loader=mock_enhanced,
        )

        assert provider.loader == mock_loader
        assert provider.macro_loader == mock_macro
        assert provider.enhanced_loader == mock_enhanced


class TestMarketDataProviderGetMarketSnapshot:
    """测试市场快照获取"""

    @pytest.fixture
    def provider(self):
        mock_loader = Mock()
        mock_macro = Mock()
        mock_enhanced = Mock()

        # 模拟价格数据
        mock_loader.load_price_data.return_value = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0],
            'Open': [99.0, 100.0, 101.0],
            'High': [101.0, 102.0, 103.0],
            'Low': [98.0, 99.0, 100.0],
            'Volume': [1000000, 1100000, 1200000],
        }, index=pd.date_range('2024-01-01', periods=3))

        mock_loader.load_news.return_value = [
            {'title': 'Test News', 'sentiment': 'positive'}
        ]

        mock_loader.get_technical_indicators.return_value = pd.DataFrame({
            'sma_20': [100.0],
            'sma_50': [99.0],
            'rsi_14': [55.0],
            'macd': [0.5],
            'macd_signal': [0.3],
            'macd_hist': [0.2],
            'bb_upper': [105.0],
            'bb_middle': [100.0],
            'bb_lower': [95.0],
        })

        mock_macro.get_macro_for_experts.return_value = {
            'vix': 20.0,
            'dxy': 103.5,
        }

        mock_enhanced.get_market_sentiment.return_value = {
            'market_sentiment_score': 0.5,
            'market_sentiment_label': 'neutral',
        }

        return MarketDataProvider(
            data_loader=mock_loader,
            macro_loader=mock_macro,
            enhanced_loader=mock_enhanced,
        )

    def test_get_market_snapshot_basic(self, provider):
        """测试基本市场快照"""
        snapshot = provider.get_market_snapshot(
            symbols=['AAPL'],
            date='2024-01-03',
            lookback_days=30,
            include_news=True,
            include_technicals=True,
            include_sentiment=False,
        )

        assert 'macro' in snapshot

    def test_get_market_snapshot_with_sentiment(self, provider):
        """测试带情绪的市场快照"""
        snapshot = provider.get_market_snapshot(
            symbols=['AAPL'],
            date='2024-01-03',
            include_sentiment=True,
        )

        assert 'sentiment' in snapshot


class TestMarketDataProviderReturnsMatrix:
    """测试收益率矩阵"""

    @pytest.fixture
    def provider(self):
        mock_loader = Mock()

        # 创建MultiIndex价格数据
        dates = pd.date_range('2024-01-01', periods=30)
        price_data = pd.DataFrame({
            'Close': np.random.randn(30).cumsum() + 100,
        }, index=dates)
        mock_loader.load_price_data.return_value = price_data

        return MarketDataProvider(
            data_loader=mock_loader,
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_get_returns_matrix(self, provider):
        """测试收益率矩阵计算"""
        returns = provider.get_returns_matrix(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-30',
        )

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == 29  # pct_change后少一行


class TestMarketDataProviderCovarianceMatrix:
    """测试协方差矩阵"""

    @pytest.fixture
    def provider(self):
        mock_loader = Mock()

        # 创建价格数据
        dates = pd.date_range('2024-01-01', periods=30)
        price_data = pd.DataFrame({
            'Close': np.random.randn(30).cumsum() + 100,
        }, index=dates)
        mock_loader.load_price_data.return_value = price_data

        return MarketDataProvider(
            data_loader=mock_loader,
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_get_covariance_matrix(self, provider):
        """测试协方差矩阵计算"""
        cov = provider.get_covariance_matrix(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-30',
        )

        assert isinstance(cov, pd.DataFrame)

    def test_get_covariance_matrix_annualized(self, provider):
        """测试年化协方差矩阵"""
        cov = provider.get_covariance_matrix(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-30',
            annualize=True,
        )

        # 年化后值应该更大
        assert isinstance(cov, pd.DataFrame)


class TestMarketDataProviderExtractSymbolData:
    """测试提取符号数据"""

    @pytest.fixture
    def provider(self):
        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_extract_symbol_data_single_index(self, provider):
        """测试单索引数据提取"""
        df = pd.DataFrame({
            'Close': [100.0, 101.0],
            'Open': [99.0, 100.0],
            'High': [101.0, 102.0],
            'Low': [98.0, 99.0],
            'Volume': [1000000, 1100000],
        })

        result = provider._extract_symbol_data(df, 'AAPL')

        assert result['close'] == 101.0
        assert result['volume'] == 1100000

    def test_extract_symbol_data_empty(self, provider):
        """测试空数据提取"""
        df = pd.DataFrame()

        result = provider._extract_symbol_data(df, 'AAPL')

        assert result is None

    def test_extract_symbol_data_multiindex(self, provider):
        """测试MultiIndex数据提取"""
        arrays = [['AAPL', 'AAPL', 'MSFT', 'MSFT'],
                  ['Close', 'Volume', 'Close', 'Volume']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)

        df = pd.DataFrame(
            [[100.0, 1000000, 200.0, 2000000],
             [101.0, 1100000, 201.0, 2100000]],
            columns=index
        )

        result = provider._extract_symbol_data(df, 'AAPL')

        assert result is not None
        assert result['close'] == 101.0


class TestMarketDataProviderTechnicals:
    """测试技术指标"""

    @pytest.fixture
    def provider(self):
        mock_loader = Mock()
        mock_loader.get_technical_indicators.return_value = pd.DataFrame({
            'sma_20': [100.0],
            'sma_50': [99.0],
            'rsi_14': [55.0],
            'macd': [0.5],
            'macd_signal': [0.3],
            'macd_hist': [0.2],
            'bb_upper': [105.0],
            'bb_middle': [100.0],
            'bb_lower': [95.0],
        })

        return MarketDataProvider(
            data_loader=mock_loader,
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_get_technicals(self, provider):
        """测试获取技术指标"""
        df = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0],
        })

        technicals = provider._get_technicals(df, 'AAPL')

        assert 'sma_20' in technicals
        assert 'rsi_14' in technicals
        assert 'macd' in technicals
        assert 'bb_upper' in technicals
        assert 'trend' in technicals

    def test_get_default_technicals(self, provider):
        """测试默认技术指标"""
        defaults = provider._get_default_technicals()

        assert defaults['sma_20'] == 0
        assert defaults['rsi_14'] == 50
        assert defaults['trend'] == 'sideways'


class TestMarketDataProviderMacroData:
    """测试宏观数据"""

    @pytest.fixture
    def provider(self):
        mock_macro = Mock()
        mock_macro.get_macro_for_experts.return_value = {
            'vix': 20.0,
            'dxy': 103.5,
            'treasury_10y': 4.2,
        }
        mock_macro.get_full_macro_snapshot.return_value = {'vix': 20.0}
        mock_macro.get_crypto_data.return_value = {'btc': {'price': 45000}}
        mock_macro.get_commodities.return_value = {'gold': {'price': 2050}}
        mock_macro.get_sector_performance.return_value = {'Technology': 1.5}
        mock_macro.get_bond_expert_data.return_value = {'fed_funds': 5.25}
        mock_macro.get_reits_expert_data.return_value = {'avg_cap_rate': 5.0}
        mock_macro.get_reit_cap_rate.return_value = 4.5
        mock_macro.get_reits_average_cap_rate.return_value = {'avg_cap_rate': 5.0}

        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=mock_macro,
            enhanced_loader=Mock(),
        )

    def test_get_macro_data(self, provider):
        """测试获取宏观数据"""
        macro = provider._get_macro_data('2024-01-15')

        assert macro['vix'] == 20.0
        assert macro['dxy'] == 103.5

    def test_get_full_macro_snapshot(self, provider):
        """测试获取完整宏观快照"""
        snapshot = provider.get_full_macro_snapshot()

        assert 'vix' in snapshot

    def test_get_crypto_onchain_data(self, provider):
        """测试获取加密货币数据"""
        crypto = provider.get_crypto_onchain_data()

        assert 'btc' in crypto

    def test_get_commodities_data(self, provider):
        """测试获取商品数据"""
        commodities = provider.get_commodities_data()

        assert 'gold' in commodities

    def test_get_sector_performance(self, provider):
        """测试获取板块表现"""
        sectors = provider.get_sector_performance()

        assert 'Technology' in sectors

    def test_get_bond_data(self, provider):
        """测试获取债券数据"""
        bond = provider.get_bond_data()

        assert 'fed_funds' in bond

    def test_get_rates_data(self, provider):
        """测试获取利率数据"""
        rates = provider.get_rates_data()

        assert 'fed_funds' in rates

    def test_get_reits_data(self, provider):
        """测试获取REITs数据"""
        reits = provider.get_reits_data()

        assert 'avg_cap_rate' in reits

    def test_get_reits_cap_rate_symbol(self, provider):
        """测试获取单个REITs Cap Rate"""
        result = provider.get_reits_cap_rate('VNQ')

        assert result['symbol'] == 'VNQ'
        assert result['cap_rate'] == 4.5

    def test_get_reits_cap_rate_all(self, provider):
        """测试获取平均Cap Rate"""
        result = provider.get_reits_cap_rate()

        assert 'avg_cap_rate' in result


class TestMarketDataProviderSentiment:
    """测试情绪数据"""

    @pytest.fixture
    def provider(self):
        mock_enhanced = Mock()
        mock_enhanced.get_market_sentiment.return_value = {
            'market_sentiment_score': 0.5,
            'market_sentiment_label': 'bullish',
            'bullish_percent': 60,
            'bearish_percent': 40,
        }
        mock_enhanced.get_stock_sentiment.return_value = {
            'sentiment_score': 0.6,
        }

        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=Mock(),
            enhanced_loader=mock_enhanced,
        )

    def test_get_sentiment_data(self, provider):
        """测试获取情绪数据"""
        sentiment = provider._get_sentiment_data(['AAPL'])

        assert 'market' in sentiment
        assert 'individual' in sentiment

    def test_get_market_sentiment(self, provider):
        """测试获取市场情绪"""
        sentiment = provider.get_market_sentiment(['AAPL'])

        assert 'market_sentiment_score' in sentiment

    def test_get_stock_sentiment(self, provider):
        """测试获取股票情绪"""
        sentiment = provider.get_stock_sentiment('AAPL')

        assert 'sentiment_score' in sentiment


class TestMarketDataProviderEnrichNews:
    """测试新闻增强"""

    @pytest.fixture
    def provider(self):
        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_enrich_news_with_sentiment(self, provider):
        """测试新闻增强"""
        news = [
            {'title': 'Test News 1', 'sentiment': 'neutral'},
            {'title': 'Test News 2'},
        ]
        sentiment_data = {
            'market': {
                'market_sentiment_score': 0.5,
                'market_sentiment_label': 'bullish',
                'bullish_percent': 60,
                'bearish_percent': 40,
            }
        }

        enriched = provider._enrich_news_with_sentiment(news, sentiment_data)

        # 应该添加一条情绪汇总
        assert len(enriched) == 3
        assert enriched[0]['is_sentiment_summary'] is True

    def test_enrich_news_empty(self, provider):
        """测试空新闻增强"""
        enriched = provider._enrich_news_with_sentiment([], {})

        assert enriched == []


class TestMarketDataProviderEnhancedData:
    """测试增强数据"""

    @pytest.fixture
    def provider(self):
        mock_enhanced = Mock()
        mock_enhanced.get_etf_fund_flows.return_value = {'inflow': 1000000}
        mock_enhanced.get_institutional_ownership.return_value = {'ownership_pct': 0.65}
        mock_enhanced.get_economic_calendar.return_value = [{'event': 'CPI'}]
        mock_enhanced.get_upcoming_fomc.return_value = {'date': '2024-01-31'}
        mock_enhanced.get_options_chain.return_value = {'calls': [], 'puts': []}
        mock_enhanced.get_put_call_ratio.return_value = {'pcr': 0.8}
        mock_enhanced.get_earnings_calendar.return_value = [{'company': 'AAPL'}]
        mock_enhanced.get_earnings_surprise.return_value = {'surprise': 0.05}
        mock_enhanced.get_enhanced_market_data.return_value = {'sentiment': {}}

        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=Mock(),
            enhanced_loader=mock_enhanced,
        )

    def test_get_fund_flows(self, provider):
        """测试获取资金流向"""
        flows = provider.get_fund_flows('SPY')

        assert 'inflow' in flows

    def test_get_institutional_holdings(self, provider):
        """测试获取机构持仓"""
        holdings = provider.get_institutional_holdings('AAPL')

        assert 'ownership_pct' in holdings

    def test_get_economic_calendar(self, provider):
        """测试获取经济日历"""
        calendar = provider.get_economic_calendar()

        assert len(calendar) > 0

    def test_get_fomc_schedule(self, provider):
        """测试获取FOMC日程"""
        fomc = provider.get_fomc_schedule()

        assert 'date' in fomc

    def test_get_options_data(self, provider):
        """测试获取期权数据"""
        options = provider.get_options_data('AAPL')

        assert 'calls' in options

    def test_get_put_call_ratio(self, provider):
        """测试获取PCR"""
        pcr = provider.get_put_call_ratio('AAPL')

        assert 'pcr' in pcr

    def test_get_earnings_calendar(self, provider):
        """测试获取财报日历"""
        calendar = provider.get_earnings_calendar()

        assert len(calendar) > 0

    def test_get_earnings_surprise(self, provider):
        """测试获取财报惊喜"""
        surprise = provider.get_earnings_surprise('AAPL')

        assert 'surprise' in surprise

    def test_get_enhanced_market_data(self, provider):
        """测试获取增强市场数据"""
        data = provider.get_enhanced_market_data(
            symbols=['AAPL'],
            include_sentiment=True,
        )

        assert 'sentiment' in data


class TestMarketDataProviderCache:
    """测试缓存功能"""

    @pytest.fixture
    def provider(self):
        return MarketDataProvider(
            data_loader=Mock(),
            macro_loader=Mock(),
            enhanced_loader=Mock(),
        )

    def test_clear_cache(self, provider):
        """测试清除缓存"""
        provider._price_cache['test'] = 'value'
        provider._news_cache['test'] = 'value'

        provider.clear_cache()

        assert len(provider._price_cache) == 0
        assert len(provider._news_cache) == 0

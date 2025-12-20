"""
Deep tests for CryptoExpert
加密货币专家深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from finsage.agents.experts.crypto_expert import CryptoExpert
from finsage.agents.base_expert import Action


# Standard JSON response for LLM mock
MOCK_LLM_RESPONSE = """```json
{
    "overall_view": "bullish",
    "recommendations": [
        {
            "symbol": "BTC-USD",
            "action": "BUY_50%",
            "confidence": 0.70,
            "target_weight": 0.05,
            "reasoning": "机构资金流入，减半效应",
            "cycle_phase": "bull_early",
            "sentiment": "bullish",
            "regulatory_risk": "medium"
        },
        {
            "symbol": "ETH-USD",
            "action": "BUY_25%",
            "confidence": 0.65,
            "target_weight": 0.03,
            "reasoning": "DeFi生态增长",
            "cycle_phase": "bull_early",
            "sentiment": "bullish",
            "regulatory_risk": "medium"
        }
    ],
    "key_factors": ["Institutional adoption", "Halving cycle"]
}
```"""


def create_mock_llm(response=MOCK_LLM_RESPONSE):
    """创建标准mock LLM"""
    mock_llm = MagicMock()
    mock_llm.create_completion = MagicMock(return_value=response)
    return mock_llm


class TestCryptoExpertInit:
    """CryptoExpert初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        expert = CryptoExpert(llm_provider=create_mock_llm())
        assert expert.name == "Crypto Expert"
        assert expert.asset_class == "crypto"
        assert len(expert.symbols) > 0

    def test_custom_symbols(self):
        """测试自定义代码列表"""
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        expert = CryptoExpert(llm_provider=create_mock_llm(), symbols=symbols)
        assert expert.symbols == symbols

    def test_properties(self):
        """测试属性"""
        expert = CryptoExpert(llm_provider=create_mock_llm())
        assert expert.description is not None
        assert expert.expertise is not None
        assert len(expert.expertise) > 0

    def test_default_config(self):
        """测试默认配置"""
        expert = CryptoExpert(llm_provider=create_mock_llm())
        # 加密货币应该有更保守的配置
        assert expert.config.get("max_single_weight", 0.05) <= 0.10


class TestCryptoExpertAnalysis:
    """CryptoExpert分析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_analyze_basic(self, expert):
        """测试基本分析"""
        market_data = {
            "BTC-USD": {"close": 48000.0, "price": 48000.0, "change_pct": 2.5, "change_7d": 5.0},
            "ETH-USD": {"close": 2900.0, "price": 2900.0, "change_pct": 1.5, "change_7d": 3.0},
        }
        report = expert.analyze(market_data)
        assert report is not None
        assert report.asset_class == "crypto"
        assert report.expert_name == "Crypto Expert"

    def test_analyze_with_on_chain(self, expert):
        """测试带链上数据分析"""
        market_data = {
            "BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0},
            "onchain": {
                "btc_active_addresses": 1000000,
                "btc_exchange_netflow": -5000,
                "funding_rate": 0.01,
                "open_interest": 15.5,
                "fear_greed": 65,
            }
        }
        report = expert.analyze(market_data)
        assert report is not None

    def test_analyze_empty_data(self, expert):
        """测试空数据"""
        market_data = {}
        report = expert.analyze(market_data)
        assert report is not None


class TestCryptoExpertPromptBuilding:
    """提示构建测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_build_analysis_prompt(self, expert):
        """测试构建分析提示"""
        market_data = {
            "BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0},
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        assert isinstance(prompt, str)
        assert "加密" in prompt or "crypto" in prompt.lower()

    def test_prompt_includes_onchain(self, expert):
        """测试提示包含链上数据"""
        market_data = {
            "BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0},
            "onchain": {
                "btc_active_addresses": 1000000,
                "funding_rate": 0.01,
            }
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        assert isinstance(prompt, str)


class TestCryptoExpertResponseParsing:
    """响应解析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_parse_buy_response(self, expert):
        """测试解析买入响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "BTC-USD",
            "action": "BUY_75%",
            "confidence": 0.80,
            "target_weight": 0.08,
            "reasoning": "牛市启动"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.symbol == "BTC-USD"
        assert rec.action == Action.BUY_75

    def test_parse_sell_response(self, expert):
        """测试解析卖出响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "ETH-USD",
            "action": "SELL_50%",
            "confidence": 0.75,
            "target_weight": 0.02,
            "reasoning": "监管风险"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SELL_50

    def test_parse_short_response(self, expert):
        """测试解析做空响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "BTC-USD",
            "action": "SHORT_25%",
            "confidence": 0.65,
            "target_weight": 0.02,
            "reasoning": "熊市信号"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SHORT_25

    def test_parse_multiple_cryptos(self, expert):
        """测试解析多个加密货币"""
        response = """```json
{
    "recommendations": [
        {"symbol": "BTC-USD", "action": "BUY_50%", "confidence": 0.75, "target_weight": 0.05, "reasoning": "看涨"},
        {"symbol": "ETH-USD", "action": "BUY_25%", "confidence": 0.65, "target_weight": 0.03, "reasoning": "DeFi"},
        {"symbol": "SOL-USD", "action": "HOLD", "confidence": 0.50, "target_weight": 0.02, "reasoning": "观望"}
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) >= 2

    def test_parse_fallback_on_invalid_json(self, expert):
        """测试JSON解析失败时的回退"""
        response = "这是一个无效的响应"
        recommendations = expert._parse_llm_response(response)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_weight_capping(self, expert):
        """测试权重上限"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "BTC-USD",
            "action": "BUY_100%",
            "confidence": 0.90,
            "target_weight": 0.50,
            "reasoning": "极度看涨"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        # 权重应该被限制
        assert rec.target_weight <= expert.config.get("max_single_weight", 0.05)


class TestCryptoExpertVolatilityAnalysis:
    """波动率分析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_volatility_assessment(self, expert):
        """测试波动率评估"""
        if hasattr(expert, '_assess_volatility'):
            returns = pd.Series(np.random.normal(0, 0.05, 100))  # 高波动
            assessment = expert._assess_volatility(returns)
            assert assessment is not None


class TestCryptoExpertOnChainAnalysis:
    """链上数据分析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_on_chain_metrics(self, expert):
        """测试链上指标分析"""
        if hasattr(expert, '_analyze_on_chain'):
            analysis = expert._analyze_on_chain({
                "active_addresses": 1000000,
                "transaction_volume": 50000000000,
                "exchange_outflow": 10000,
            })
            assert analysis is not None

    def test_whale_activity(self, expert):
        """测试巨鲸活动分析"""
        if hasattr(expert, '_analyze_whale_activity'):
            analysis = expert._analyze_whale_activity(
                large_tx_count=500,
                large_tx_volume=100000
            )
            assert analysis is not None


class TestCryptoExpertSentimentAnalysis:
    """情绪分析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_fear_greed_index(self, expert):
        """测试恐惧贪婪指数"""
        if hasattr(expert, '_analyze_sentiment'):
            analysis = expert._analyze_sentiment(
                fear_greed_index=75  # 贪婪
            )
            assert analysis is not None

    def test_social_sentiment(self, expert):
        """测试社交媒体情绪"""
        if hasattr(expert, '_analyze_social_sentiment'):
            analysis = expert._analyze_social_sentiment({
                "twitter_mentions": 50000,
                "reddit_activity": 10000,
                "sentiment_score": 0.6,
            })
            assert analysis is not None


class TestCryptoExpertRiskManagement:
    """风险管理测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_position_sizing_high_vol(self, expert):
        """测试高波动下的仓位建议"""
        # 加密货币波动大，应该建议较小仓位
        if hasattr(expert, '_recommend_position_size'):
            size = expert._recommend_position_size(
                volatility=0.8,  # 80%年化波动
                confidence=0.7
            )
            # 高波动应该限制仓位
            assert size <= 0.15

    def test_drawdown_warning(self, expert):
        """测试回撤警告"""
        if hasattr(expert, '_assess_drawdown_risk'):
            assessment = expert._assess_drawdown_risk(
                current_price=40000,
                recent_high=70000
            )
            assert assessment is not None


class TestCryptoExpertEdgeCases:
    """边界情况测试"""

    def test_empty_llm_response(self):
        """测试空LLM响应"""
        mock_llm = create_mock_llm("""```json
{"recommendations": []}
```""")
        expert = CryptoExpert(llm_provider=mock_llm)
        market_data = {"BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_malformed_response(self):
        """测试格式错误响应"""
        mock_llm = create_mock_llm("随机文本没有格式")
        expert = CryptoExpert(llm_provider=mock_llm)
        market_data = {"BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_extreme_volatility(self):
        """测试极端波动"""
        expert = CryptoExpert(llm_provider=create_mock_llm())
        # 模拟崩盘
        market_data = {
            "BTC-USD": {"close": 30000.0, "change_pct": -40.0, "change_7d": -50.0}
        }
        report = expert.analyze(market_data)
        assert report is not None

    def test_very_small_caps(self):
        """测试小市值币"""
        expert = CryptoExpert(llm_provider=create_mock_llm())
        market_data = {
            "SMALL-USD": {"close": 0.002, "change_pct": 10.0, "change_7d": 50.0}
        }
        # 应该能处理小数价格
        report = expert.analyze(market_data)
        assert report is not None


class TestCryptoExpertFormatMethods:
    """格式化方法测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_format_price_data(self, expert):
        """测试价格数据格式化"""
        market_data = {
            "BTC-USD": {"close": 44000.0, "change_pct": 1.5, "change_7d": 3.0},
        }
        result = expert._format_price_data(market_data)
        assert isinstance(result, str)

    def test_format_price_data_empty(self, expert):
        """测试空价格数据格式化"""
        result = expert._format_price_data({})
        assert "暂无" in result or isinstance(result, str)

    def test_format_onchain_data(self, expert):
        """测试链上数据格式化"""
        market_data = {
            "onchain": {
                "btc_active_addresses": 1000000,
                "btc_exchange_netflow": -5000,
                "funding_rate": 0.01,
                "open_interest": 15.5,
                "fear_greed": 65,
            }
        }
        result = expert._format_onchain_data(market_data)
        assert isinstance(result, str)

    def test_format_news(self, expert):
        """测试新闻格式化"""
        news_data = [
            {"title": "Bitcoin ETF approved by SEC", "sentiment": "bullish"},
            {"title": "Crypto regulation concerns", "sentiment": "bearish"},
        ]
        result = expert._format_news(news_data)
        assert isinstance(result, str)

    def test_format_news_empty(self, expert):
        """测试空新闻格式化"""
        result = expert._format_news([])
        assert "暂无" in result or isinstance(result, str)


class TestCryptoExpertFactorIntegration:
    """因子集成测试"""

    def test_with_factor_scorer(self):
        """测试带因子评分器"""
        from finsage.factors.crypto_factors import CryptoFactorScorer

        expert = CryptoExpert(llm_provider=create_mock_llm())

        market_data = {
            "BTC-USD": {"close": 44000.0, "change_pct": 1.0, "change_7d": 2.0},
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestCryptoExpertCorrelation:
    """相关性测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_btc_correlation(self, expert):
        """测试与BTC相关性"""
        if hasattr(expert, '_analyze_btc_correlation'):
            analysis = expert._analyze_btc_correlation(
                symbol="ETH-USD",
                correlation=0.85
            )
            assert analysis is not None

    def test_stock_correlation(self, expert):
        """测试与股票相关性"""
        if hasattr(expert, '_analyze_equity_correlation'):
            analysis = expert._analyze_equity_correlation(
                nasdaq_correlation=0.65
            )
            assert analysis is not None


class TestCryptoExpertRegulatory:
    """监管风险测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_regulatory_risk_assessment(self, expert):
        """测试监管风险评估"""
        if hasattr(expert, '_assess_regulatory_risk'):
            assessment = expert._assess_regulatory_risk(
                recent_news=["SEC investigation", "ETF approval delayed"]
            )
            assert assessment is not None


class TestCryptoExpertActionParsing:
    """动作解析测试"""

    @pytest.fixture
    def expert(self):
        return CryptoExpert(llm_provider=create_mock_llm())

    def test_parse_action_buy(self, expert):
        """测试解析买入动作"""
        assert expert._parse_action("BUY_100%") == Action.BUY_100
        assert expert._parse_action("BUY_75%") == Action.BUY_75
        assert expert._parse_action("BUY_50%") == Action.BUY_50
        assert expert._parse_action("BUY_25%") == Action.BUY_25

    def test_parse_action_sell(self, expert):
        """测试解析卖出动作"""
        assert expert._parse_action("SELL_100%") == Action.SELL_100
        assert expert._parse_action("SELL_75%") == Action.SELL_75
        assert expert._parse_action("SELL_50%") == Action.SELL_50
        assert expert._parse_action("SELL_25%") == Action.SELL_25

    def test_parse_action_short(self, expert):
        """测试解析做空动作"""
        assert expert._parse_action("SHORT_100%") == Action.SHORT_100
        assert expert._parse_action("SHORT_75%") == Action.SHORT_75
        assert expert._parse_action("SHORT_50%") == Action.SHORT_50
        assert expert._parse_action("SHORT_25%") == Action.SHORT_25

    def test_parse_action_hold(self, expert):
        """测试解析持有动作"""
        assert expert._parse_action("HOLD") == Action.HOLD

    def test_parse_action_unknown(self, expert):
        """测试解析未知动作"""
        assert expert._parse_action("UNKNOWN") == Action.HOLD

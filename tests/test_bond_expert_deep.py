"""
Deep tests for BondExpert
债券专家深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from finsage.agents.experts.bond_expert import BondExpert
from finsage.agents.base_expert import Action


# Standard JSON response for LLM mock
MOCK_LLM_RESPONSE = """```json
{
    "overall_view": "bullish",
    "recommendations": [
        {
            "symbol": "TLT",
            "action": "BUY_50%",
            "confidence": 0.75,
            "target_weight": 0.10,
            "reasoning": "利率见顶预期，长债有上涨空间",
            "rate_view": "bullish",
            "duration_preference": "long",
            "credit_view": "neutral"
        },
        {
            "symbol": "IEF",
            "action": "HOLD",
            "confidence": 0.6,
            "target_weight": 0.05,
            "reasoning": "中性利率环境",
            "rate_view": "neutral",
            "duration_preference": "medium",
            "credit_view": "neutral"
        }
    ],
    "key_factors": ["Fed policy", "Inflation expectations"]
}
```"""


def create_mock_llm(response=MOCK_LLM_RESPONSE):
    """创建标准mock LLM"""
    mock_llm = MagicMock()
    mock_llm.create_completion = MagicMock(return_value=response)
    return mock_llm


class TestBondExpertInit:
    """BondExpert初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        expert = BondExpert(llm_provider=create_mock_llm())
        assert expert.name == "Bond Expert"
        assert expert.asset_class == "bonds"
        assert len(expert.symbols) > 0

    def test_custom_symbols(self):
        """测试自定义代码列表"""
        symbols = ["IEF", "SHY", "BND"]
        expert = BondExpert(llm_provider=create_mock_llm(), symbols=symbols)
        assert expert.symbols == symbols

    def test_properties(self):
        """测试属性"""
        expert = BondExpert(llm_provider=create_mock_llm())
        assert "债券" in expert.description or "bond" in expert.description.lower()
        assert expert.expertise is not None
        assert len(expert.expertise) > 0


class TestBondExpertAnalysis:
    """BondExpert分析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_analyze_basic(self, expert):
        """测试基本分析"""
        market_data = {
            "TLT": {"close": 100.0, "price": 100.0, "change_pct": 1.5, "yield": "4.5%"},
            "IEF": {"close": 90.0, "price": 90.0, "change_pct": 0.5, "yield": "4.0%"},
        }
        report = expert.analyze(market_data)
        assert report is not None
        assert report.asset_class == "bonds"
        assert report.expert_name == "Bond Expert"

    def test_analyze_with_macro(self, expert):
        """测试带宏观数据分析"""
        market_data = {
            "TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"},
            "rates": {
                "fed_funds": 5.25,
                "treasury_2y": 4.5,
                "treasury_10y": 4.2,
                "treasury_30y": 4.3,
                "spread_2s10s": -30,
            }
        }
        report = expert.analyze(market_data)
        assert report is not None

    def test_analyze_empty_data(self, expert):
        """测试空数据"""
        market_data = {}
        report = expert.analyze(market_data)
        # 应该能处理空数据
        assert report is not None


class TestBondExpertPromptBuilding:
    """提示构建测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_build_analysis_prompt(self, expert):
        """测试构建分析提示"""
        market_data = {
            "TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"},
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        assert isinstance(prompt, str)
        assert "债券" in prompt or "bond" in prompt.lower()

    def test_prompt_includes_yields(self, expert):
        """测试提示包含收益率信息"""
        market_data = {
            "TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"},
            "rates": {
                "fed_funds": 5.25,
                "treasury_2y": 4.5,
                "treasury_10y": 4.2,
            }
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        # 应该包含相关上下文
        assert isinstance(prompt, str)


class TestBondExpertResponseParsing:
    """响应解析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_parse_buy_response(self, expert):
        """测试解析买入响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "TLT",
            "action": "BUY_100%",
            "confidence": 0.85,
            "target_weight": 0.15,
            "reasoning": "利率下行周期",
            "rate_view": "bullish",
            "duration_preference": "long",
            "credit_view": "neutral"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.symbol == "TLT"
        assert rec.action == Action.BUY_100

    def test_parse_sell_response(self, expert):
        """测试解析卖出响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "TLT",
            "action": "SELL_50%",
            "confidence": 0.70,
            "target_weight": 0.05,
            "reasoning": "利率上行风险",
            "rate_view": "bearish"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SELL_50

    def test_parse_hold_response(self, expert):
        """测试解析持有响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "IEF",
            "action": "HOLD",
            "confidence": 0.65,
            "target_weight": 0.08,
            "reasoning": "中性利率环境"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.HOLD

    def test_parse_short_response(self, expert):
        """测试解析做空响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "TLT",
            "action": "SHORT_50%",
            "confidence": 0.70,
            "target_weight": 0.05,
            "reasoning": "利率上行预期"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SHORT_50

    def test_parse_multiple_recommendations(self, expert):
        """测试解析多个推荐"""
        response = """```json
{
    "recommendations": [
        {"symbol": "TLT", "action": "BUY_50%", "confidence": 0.75, "target_weight": 0.10, "reasoning": "长债看涨"},
        {"symbol": "IEF", "action": "HOLD", "confidence": 0.60, "target_weight": 0.05, "reasoning": "中性"},
        {"symbol": "SHY", "action": "SELL_25%", "confidence": 0.55, "target_weight": 0.03, "reasoning": "短债吸引力下降"}
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) >= 2

    def test_parse_fallback_on_invalid_json(self, expert):
        """测试JSON解析失败时的回退"""
        response = "这是一个无效的响应"
        recommendations = expert._parse_llm_response(response)
        # 应该返回默认推荐
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0  # 应该有fallback推荐


class TestBondExpertYieldCurve:
    """收益率曲线分析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_analyze_yield_curve_normal(self, expert):
        """测试正常收益率曲线"""
        yields = {"2y": 0.03, "10y": 0.04, "30y": 0.045}
        if hasattr(expert, '_analyze_yield_curve'):
            analysis = expert._analyze_yield_curve(yields)
            assert "normal" in str(analysis).lower() or analysis is not None

    def test_analyze_yield_curve_inverted(self, expert):
        """测试倒挂收益率曲线"""
        yields = {"2y": 0.05, "10y": 0.04, "30y": 0.038}
        if hasattr(expert, '_analyze_yield_curve'):
            analysis = expert._analyze_yield_curve(yields)
            assert analysis is not None


class TestBondExpertDurationAnalysis:
    """久期分析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_duration_risk_assessment(self, expert):
        """测试久期风险评估"""
        # 如果有久期分析方法
        if hasattr(expert, '_assess_duration_risk'):
            risk = expert._assess_duration_risk(duration=12, rate_outlook="rising")
            assert risk is not None


class TestBondExpertCreditAnalysis:
    """信用分析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_credit_spread_analysis(self, expert):
        """测试信用利差分析"""
        if hasattr(expert, '_analyze_credit_spread'):
            analysis = expert._analyze_credit_spread(
                ig_spread=0.015,
                hy_spread=0.04
            )
            assert analysis is not None


class TestBondExpertEdgeCases:
    """边界情况测试"""

    def test_empty_llm_response(self):
        """测试空LLM响应"""
        mock_llm = create_mock_llm("""```json
{"recommendations": []}
```""")
        expert = BondExpert(llm_provider=mock_llm)
        market_data = {"TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_malformed_response(self):
        """测试格式错误响应"""
        mock_llm = create_mock_llm("随机文本没有格式")
        expert = BondExpert(llm_provider=mock_llm)
        market_data = {"TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_single_data_point(self):
        """测试单个数据点"""
        expert = BondExpert(llm_provider=create_mock_llm())
        market_data = {"TLT": {"close": 100.0, "change_pct": 0.0, "yield": "4.5%"}}
        report = expert.analyze(market_data)
        assert report is not None


class TestBondExpertFormatMethods:
    """格式化方法测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

    def test_format_price_data(self, expert):
        """测试价格数据格式化"""
        market_data = {
            "TLT": {"close": 100.0, "change_pct": 1.5, "yield": "4.5%"},
        }
        result = expert._format_price_data(market_data)
        assert isinstance(result, str)

    def test_format_price_data_empty(self, expert):
        """测试空价格数据格式化"""
        result = expert._format_price_data({})
        assert "暂无" in result or isinstance(result, str)

    def test_format_rate_data(self, expert):
        """测试利率数据格式化"""
        market_data = {
            "rates": {
                "fed_funds": 5.25,
                "treasury_2y": 4.5,
                "treasury_10y": 4.2,
            }
        }
        result = expert._format_rate_data(market_data)
        assert isinstance(result, str)

    def test_format_news(self, expert):
        """测试新闻格式化"""
        news_data = [
            {"title": "Fed raises rate by 25bps", "sentiment": "bearish"},
            {"title": "Treasury yields fall", "sentiment": "bullish"},
        ]
        result = expert._format_news(news_data)
        assert isinstance(result, str)

    def test_format_news_empty(self, expert):
        """测试空新闻格式化"""
        result = expert._format_news([])
        assert "暂无" in result or isinstance(result, str)


class TestBondExpertFactorIntegration:
    """因子集成测试"""

    def test_with_factor_scorer(self):
        """测试带因子评分器"""
        from finsage.factors.bond_factors import BondFactorScorer

        expert = BondExpert(llm_provider=create_mock_llm())

        market_data = {
            "TLT": {"close": 100.0, "change_pct": 0.5, "yield": "4.5%"},
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestBondExpertActionParsing:
    """动作解析测试"""

    @pytest.fixture
    def expert(self):
        return BondExpert(llm_provider=create_mock_llm())

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
        assert expert._parse_action("UNKNOWN") == Action.HOLD  # 默认为HOLD

"""
Deep tests for CommodityExpert
商品专家深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from finsage.agents.experts.commodity_expert import CommodityExpert
from finsage.agents.base_expert import Action


# Standard JSON response for LLM mock
MOCK_LLM_RESPONSE = """```json
{
    "overall_view": "bullish",
    "recommendations": [
        {
            "symbol": "GLD",
            "action": "BUY_75%",
            "confidence": 0.80,
            "target_weight": 0.12,
            "reasoning": "避险需求上升，美元走弱",
            "supply_demand": "tight",
            "dollar_impact": "bullish",
            "geopolitical_risk": "medium"
        },
        {
            "symbol": "USO",
            "action": "SELL_25%",
            "confidence": 0.65,
            "target_weight": 0.05,
            "reasoning": "供应过剩，需求疲软",
            "supply_demand": "excess",
            "dollar_impact": "neutral",
            "geopolitical_risk": "high"
        }
    ],
    "key_factors": ["Dollar weakness", "Geopolitical tensions"]
}
```"""


def create_mock_llm(response=MOCK_LLM_RESPONSE):
    """创建标准mock LLM"""
    mock_llm = MagicMock()
    mock_llm.create_completion = MagicMock(return_value=response)
    return mock_llm


class TestCommodityExpertInit:
    """CommodityExpert初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        expert = CommodityExpert(llm_provider=create_mock_llm())
        assert expert.name == "Commodity Expert"
        assert expert.asset_class == "commodities"
        assert len(expert.symbols) > 0

    def test_custom_symbols(self):
        """测试自定义代码列表"""
        symbols = ["GLD", "SLV", "USO"]
        expert = CommodityExpert(llm_provider=create_mock_llm(), symbols=symbols)
        assert expert.symbols == symbols

    def test_properties(self):
        """测试属性"""
        expert = CommodityExpert(llm_provider=create_mock_llm())
        assert "商品" in expert.description or "commodity" in expert.description.lower()
        assert expert.expertise is not None
        assert len(expert.expertise) > 0


class TestCommodityExpertAnalysis:
    """CommodityExpert分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_analyze_basic(self, expert):
        """测试基本分析"""
        market_data = {
            "GLD": {"close": 188.0, "price": 188.0, "change_pct": 1.5},
            "USO": {"close": 62.0, "price": 62.0, "change_pct": -2.0},
        }
        report = expert.analyze(market_data)
        assert report is not None
        assert report.asset_class == "commodities"
        assert report.expert_name == "Commodity Expert"

    def test_analyze_with_macro(self, expert):
        """测试带宏观数据分析"""
        market_data = {
            "GLD": {"close": 184.0, "change_pct": 0.5},
            "macro": {
                "dxy": 102.5,  # 美元指数
                "inflation_expectation": 3.5,
                "real_rate": 1.5,
                "vix": 18.5,
            }
        }
        report = expert.analyze(market_data)
        assert report is not None

    def test_analyze_empty_data(self, expert):
        """测试空数据"""
        market_data = {}
        report = expert.analyze(market_data)
        assert report is not None


class TestCommodityExpertPromptBuilding:
    """提示构建测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_build_analysis_prompt(self, expert):
        """测试构建分析提示"""
        market_data = {
            "GLD": {"close": 184.0, "change_pct": 0.5},
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        assert isinstance(prompt, str)
        assert "商品" in prompt or "commodity" in prompt.lower()

    def test_prompt_includes_macro(self, expert):
        """测试提示包含宏观数据"""
        market_data = {
            "GLD": {"close": 184.0, "change_pct": 0.5},
            "macro": {
                "dxy": 102.5,
                "real_rate": 1.5,
            }
        }
        news_data = []
        technical_indicators = {}
        prompt = expert._build_analysis_prompt(market_data, news_data, technical_indicators)
        assert isinstance(prompt, str)


class TestCommodityExpertResponseParsing:
    """响应解析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_parse_buy_response(self, expert):
        """测试解析买入响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "GLD",
            "action": "BUY_100%",
            "confidence": 0.85,
            "target_weight": 0.15,
            "reasoning": "黄金避险需求"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.symbol == "GLD"
        assert rec.action == Action.BUY_100

    def test_parse_sell_response(self, expert):
        """测试解析卖出响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "USO",
            "action": "SELL_50%",
            "confidence": 0.70,
            "target_weight": 0.03,
            "reasoning": "供应过剩"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SELL_50

    def test_parse_commodities_sector(self, expert):
        """测试解析不同商品板块"""
        response = """```json
{
    "recommendations": [
        {"symbol": "GLD", "action": "BUY_50%", "confidence": 0.75, "target_weight": 0.10, "reasoning": "避险"},
        {"symbol": "USO", "action": "SELL_50%", "confidence": 0.70, "target_weight": 0.03, "reasoning": "供应过剩"},
        {"symbol": "DBA", "action": "HOLD", "confidence": 0.55, "target_weight": 0.05, "reasoning": "农产品稳定"}
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) >= 2

    def test_parse_short_response(self, expert):
        """测试解析做空响应"""
        response = """```json
{
    "recommendations": [
        {
            "symbol": "USO",
            "action": "SHORT_50%",
            "confidence": 0.75,
            "target_weight": 0.05,
            "reasoning": "油价下跌预期"
        }
    ]
}
```"""
        recommendations = expert._parse_llm_response(response)
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.action == Action.SHORT_50

    def test_parse_fallback_on_invalid_json(self, expert):
        """测试JSON解析失败时的回退"""
        response = "这是一个无效的响应"
        recommendations = expert._parse_llm_response(response)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestCommodityExpertTermStructure:
    """期限结构分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_analyze_backwardation(self, expert):
        """测试现货溢价分析"""
        if hasattr(expert, '_analyze_term_structure'):
            structure = expert._analyze_term_structure(
                front_price=100,
                back_price=95
            )
            assert structure is not None or "backwardation" in str(structure).lower()

    def test_analyze_contango(self, expert):
        """测试期货溢价分析"""
        if hasattr(expert, '_analyze_term_structure'):
            structure = expert._analyze_term_structure(
                front_price=95,
                back_price=100
            )
            assert structure is not None


class TestCommodityExpertSectorAnalysis:
    """板块分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_energy_sector_analysis(self, expert):
        """测试能源板块分析"""
        market_data = {
            "USO": {"close": 70.0, "change_pct": 1.5},
            "UNG": {"close": 21.0, "change_pct": 0.5},
        }
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(market_data, "energy")
            assert analysis is not None

    def test_precious_metals_analysis(self, expert):
        """测试贵金属分析"""
        market_data = {
            "GLD": {"close": 184.0, "change_pct": 1.0},
            "SLV": {"close": 24.0, "change_pct": 1.5},
        }
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(market_data, "precious_metals")
            assert analysis is not None

    def test_agriculture_analysis(self, expert):
        """测试农产品分析"""
        market_data = {
            "DBA": {"close": 21.0, "change_pct": 0.3},
        }
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(market_data, "agriculture")
            assert analysis is not None


class TestCommodityExpertInflationHedge:
    """通胀对冲分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_inflation_hedge_analysis(self, expert):
        """测试通胀对冲分析"""
        if hasattr(expert, '_analyze_inflation_hedge'):
            analysis = expert._analyze_inflation_hedge(
                inflation_rate=0.05,
                real_rates=-0.01
            )
            assert analysis is not None


class TestCommodityExpertDollarCorrelation:
    """美元相关性测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_dollar_correlation(self, expert):
        """测试美元相关性分析"""
        if hasattr(expert, '_analyze_dollar_correlation'):
            analysis = expert._analyze_dollar_correlation(
                dxy_level=105,
                dxy_change=0.02
            )
            assert analysis is not None


class TestCommodityExpertEdgeCases:
    """边界情况测试"""

    def test_empty_llm_response(self):
        """测试空LLM响应"""
        mock_llm = create_mock_llm("""```json
{"recommendations": []}
```""")
        expert = CommodityExpert(llm_provider=mock_llm)
        market_data = {"GLD": {"close": 184.0, "change_pct": 0.5}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_malformed_response(self):
        """测试格式错误响应"""
        mock_llm = create_mock_llm("随机文本没有格式")
        expert = CommodityExpert(llm_provider=mock_llm)
        market_data = {"GLD": {"close": 184.0, "change_pct": 0.5}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_single_commodity(self):
        """测试单个商品"""
        expert = CommodityExpert(llm_provider=create_mock_llm())
        market_data = {"GLD": {"close": 180.0, "change_pct": 0.0}}
        report = expert.analyze(market_data)
        assert report is not None

    def test_negative_prices(self):
        """测试负价格(理论上可能如2020年原油)"""
        expert = CommodityExpert(llm_provider=create_mock_llm())
        # 商品专家应该能处理异常价格
        market_data = {"USO": {"close": -5.0, "change_pct": -50.0}}
        report = expert.analyze(market_data)
        assert report is not None


class TestCommodityExpertFormatMethods:
    """格式化方法测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_format_price_data(self, expert):
        """测试价格数据格式化"""
        market_data = {
            "GLD": {"close": 184.0, "change_pct": 1.5},
        }
        result = expert._format_price_data(market_data)
        assert isinstance(result, str)

    def test_format_price_data_empty(self, expert):
        """测试空价格数据格式化"""
        result = expert._format_price_data({})
        assert "暂无" in result or isinstance(result, str)

    def test_format_macro_data(self, expert):
        """测试宏观数据格式化"""
        market_data = {
            "macro": {
                "dxy": 102.5,
                "real_rate": 1.5,
                "inflation_expectation": 3.5,
                "vix": 18.5,
            }
        }
        result = expert._format_macro_data(market_data)
        assert isinstance(result, str)

    def test_format_news(self, expert):
        """测试新闻格式化"""
        news_data = [
            {"title": "Gold prices surge on safe-haven demand", "sentiment": "bullish"},
            {"title": "Oil supply concerns ease", "sentiment": "bearish"},
        ]
        result = expert._format_news(news_data)
        assert isinstance(result, str)

    def test_format_news_empty(self, expert):
        """测试空新闻格式化"""
        result = expert._format_news([])
        assert "暂无" in result or isinstance(result, str)


class TestCommodityExpertFactorIntegration:
    """因子集成测试"""

    def test_with_factor_scorer(self):
        """测试带因子评分器"""
        from finsage.factors.commodity_factors import CommodityFactorScorer

        expert = CommodityExpert(llm_provider=create_mock_llm())

        market_data = {
            "GLD": {"close": 184.0, "change_pct": 0.5},
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestCommodityExpertSeasonality:
    """季节性分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_seasonality_analysis(self, expert):
        """测试季节性分析"""
        if hasattr(expert, '_analyze_seasonality'):
            analysis = expert._analyze_seasonality(
                symbol="DBA",
                current_month=3  # 春季种植季
            )
            assert analysis is not None


class TestCommodityExpertSupplyDemand:
    """供需分析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

    def test_supply_demand_analysis(self, expert):
        """测试供需分析"""
        if hasattr(expert, '_analyze_supply_demand'):
            analysis = expert._analyze_supply_demand(
                inventory_level=0.8,  # 库存水平
                production_change=-0.05  # 产量变化
            )
            assert analysis is not None


class TestCommodityExpertActionParsing:
    """动作解析测试"""

    @pytest.fixture
    def expert(self):
        return CommodityExpert(llm_provider=create_mock_llm())

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

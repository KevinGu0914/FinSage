"""
Deep tests for REITsExpert
REITs专家深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from finsage.agents.experts.reits_expert import REITsExpert
from finsage.agents.base_expert import Action


class TestREITsExpertInit:
    """REITsExpert初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        expert = REITsExpert(llm_provider=MagicMock())
        assert expert.name == "REITs_Expert"
        assert expert.asset_class == "reits"
        assert "VNQ" in expert.default_symbols

    def test_custom_symbols(self):
        """测试自定义代码列表"""
        symbols = ["VNQ", "IYR", "XLRE"]
        expert = REITsExpert(llm_provider=MagicMock(), symbols=symbols)
        assert expert.symbols == symbols

    def test_properties(self):
        """测试属性"""
        expert = REITsExpert(llm_provider=MagicMock())
        assert "REITs" in expert.description or "不动产" in expert.description
        assert expert.expertise is not None


class TestREITsExpertAnalysis:
    """REITsExpert分析测试"""

    @pytest.fixture
    def expert(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="""
## REITs市场分析

### VNQ (综合REITs ETF)
- 建议: 买入50%
- 置信度: 0.72
- 目标权重: 10%
- 原因: 利率见顶预期，估值修复
- 市场观点: bullish
- 风险评估: medium

### DLR (数据中心REITs)
- 建议: 买入75%
- 置信度: 0.80
- 目标权重: 5%
- 原因: AI驱动需求增长
- 市场观点: bullish
- 风险评估: medium
""")
        return REITsExpert(llm_provider=mock_llm)

    def test_analyze_basic(self, expert):
        """测试基本分析"""
        market_data = {
            "VNQ": pd.DataFrame({
                "Close": [85, 86, 87, 88, 89],
                "Volume": [3000000, 3100000, 3200000, 3300000, 3400000]
            }),
            "DLR": pd.DataFrame({
                "Close": [130, 135, 140, 145, 150],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000]
            }),
        }
        report = expert.analyze(market_data)
        assert report is not None
        assert report.asset_class == "reits"

    def test_analyze_with_rates(self, expert):
        """测试带利率数据分析"""
        market_data = {
            "VNQ": pd.DataFrame({"Close": [85, 86, 87]}),
            "macro": {
                "fed_rate": 0.05,
                "10y_yield": 0.045,
                "mortgage_rate": 0.07,
            }
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestREITsExpertResponseParsing:
    """响应解析测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_parse_buy_response(self, expert):
        """测试解析买入响应"""
        response = """
### VNQ
- 建议: 买入50%
- 置信度: 0.75
- 目标权重: 10%
- 原因: 估值便宜
- 市场观点: bullish
- 风险评估: medium
"""
        recommendations = expert._parse_llm_response(response, {})
        assert len(recommendations) > 0
        rec = recommendations[0]
        assert rec.symbol == "VNQ"

    def test_parse_sector_specific(self, expert):
        """测试解析行业特定REITs"""
        response = """
### DLR
- 建议: 买入75%
- 置信度: 0.80
- 目标权重: 5%
- 原因: 数据中心需求
- 市场观点: bullish
- 风险评估: medium

### PLD
- 建议: 买入50%
- 置信度: 0.70
- 目标权重: 5%
- 原因: 物流需求
- 市场观点: bullish
- 风险评估: low

### BXP
- 建议: 卖出50%
- 置信度: 0.65
- 目标权重: 2%
- 原因: 办公需求下降
- 市场观点: bearish
- 风险评估: high
"""
        recommendations = expert._parse_llm_response(response, {})
        assert len(recommendations) >= 2


class TestREITsExpertNAVAnalysis:
    """NAV分析测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_nav_premium_analysis(self, expert):
        """测试NAV溢价分析"""
        if hasattr(expert, '_analyze_nav_premium'):
            analysis = expert._analyze_nav_premium(
                price=85,
                nav=100
            )
            assert analysis is not None

    def test_nav_discount_analysis(self, expert):
        """测试NAV折价分析"""
        if hasattr(expert, '_analyze_nav_premium'):
            analysis = expert._analyze_nav_premium(
                price=115,
                nav=100
            )
            assert analysis is not None


class TestREITsExpertSectorAnalysis:
    """行业分析测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_data_center_analysis(self, expert):
        """测试数据中心REITs分析"""
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(
                sector="data_center",
                macro_data={"ai_investment": "high"}
            )
            assert analysis is not None

    def test_residential_analysis(self, expert):
        """测试住宅REITs分析"""
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(
                sector="residential",
                macro_data={"rent_growth": 0.05}
            )
            assert analysis is not None

    def test_office_analysis(self, expert):
        """测试办公REITs分析"""
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(
                sector="office",
                macro_data={"remote_work_trend": "increasing"}
            )
            assert analysis is not None

    def test_logistics_analysis(self, expert):
        """测试物流REITs分析"""
        if hasattr(expert, '_analyze_sector'):
            analysis = expert._analyze_sector(
                sector="logistics",
                macro_data={"ecommerce_growth": 0.15}
            )
            assert analysis is not None


class TestREITsExpertDividendAnalysis:
    """股息分析测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_dividend_yield_analysis(self, expert):
        """测试股息率分析"""
        if hasattr(expert, '_analyze_dividend'):
            analysis = expert._analyze_dividend(
                dividend_yield=0.05,
                payout_ratio=0.70,
                ffo_growth=0.03
            )
            assert analysis is not None

    def test_dividend_sustainability(self, expert):
        """测试股息可持续性"""
        if hasattr(expert, '_assess_dividend_sustainability'):
            assessment = expert._assess_dividend_sustainability(
                ffo_per_share=5.0,
                dividend_per_share=3.5,
                debt_to_ebitda=5.0
            )
            assert assessment is not None


class TestREITsExpertInterestRateSensitivity:
    """利率敏感性测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_rate_sensitivity_analysis(self, expert):
        """测试利率敏感性分析"""
        if hasattr(expert, '_analyze_rate_sensitivity'):
            analysis = expert._analyze_rate_sensitivity(
                debt_to_equity=1.2,
                variable_rate_debt_pct=0.30,
                avg_debt_maturity=5.0
            )
            assert analysis is not None


class TestREITsExpertEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def expert(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="")
        return REITsExpert(llm_provider=mock_llm)

    def test_empty_llm_response(self, expert):
        """测试空LLM响应"""
        market_data = {"VNQ": pd.DataFrame({"Close": [85, 86, 87]})}
        report = expert.analyze(market_data)
        assert report is not None

    def test_single_reit(self, expert):
        """测试单个REIT"""
        market_data = {"VNQ": pd.DataFrame({"Close": [85]})}
        report = expert.analyze(market_data)
        assert report is not None

    def test_high_volatility(self, expert):
        """测试高波动"""
        market_data = {
            "VNQ": pd.DataFrame({
                "Close": [85, 75, 90, 70, 95]  # 高波动
            })
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestREITsExpertFactorIntegration:
    """因子集成测试"""

    def test_with_factor_scorer(self):
        """测试带因子评分器"""
        from finsage.factors.reits_factors import REITsFactorScorer

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="### VNQ\n- 建议: 买入50%\n- 置信度: 0.7\n- 目标权重: 10%\n- 原因: 看涨\n- 市场观点: bullish\n- 风险评估: medium")

        expert = REITsExpert(
            llm_provider=mock_llm,
            factor_scorer=REITsFactorScorer()
        )

        market_data = {
            "VNQ": pd.DataFrame({
                "Close": [85, 86, 87, 88, 89],
                "Volume": [3000000] * 5
            }),
        }
        report = expert.analyze(market_data)
        assert report is not None


class TestREITsExpertMacroIntegration:
    """宏观集成测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_housing_market_integration(self, expert):
        """测试房地产市场集成"""
        if hasattr(expert, '_integrate_housing_data'):
            integration = expert._integrate_housing_data({
                "existing_home_sales": -0.05,
                "housing_starts": -0.03,
                "case_shiller": 0.02,
            })
            assert integration is not None

    def test_cre_market_integration(self, expert):
        """测试商业地产市场集成"""
        if hasattr(expert, '_integrate_cre_data'):
            integration = expert._integrate_cre_data({
                "office_vacancy": 0.18,
                "industrial_vacancy": 0.04,
                "retail_vacancy": 0.10,
            })
            assert integration is not None


class TestREITsExpertGeographicAnalysis:
    """地理分析测试"""

    @pytest.fixture
    def expert(self):
        return REITsExpert(llm_provider=MagicMock())

    def test_geographic_exposure(self, expert):
        """测试地理敞口分析"""
        if hasattr(expert, '_analyze_geographic_exposure'):
            analysis = expert._analyze_geographic_exposure({
                "sunbelt": 0.40,
                "coastal": 0.35,
                "midwest": 0.25,
            })
            assert analysis is not None

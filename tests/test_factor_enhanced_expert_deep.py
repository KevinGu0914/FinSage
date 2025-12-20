"""
Deep tests for FactorEnhancedExpertMixin
因子增强专家混入类深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from finsage.agents.factor_enhanced_expert import (
    FactorEnhancedExpertMixin,
    EnhancedRecommendation,
    create_factor_enhanced_expert,
    get_enhanced_experts,
)
from finsage.agents.base_expert import (
    BaseExpert,
    ExpertRecommendation,
    ExpertReport,
    Action,
)
from finsage.factors.base_factor import (
    BaseFactorScorer,
    FactorScore,
    FactorExposure,
    FactorType,
)


class TestEnhancedRecommendation:
    """EnhancedRecommendation数据类测试"""

    def test_create_enhanced_recommendation(self):
        """测试创建增强建议"""
        rec = EnhancedRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_50,
            confidence=0.75,
            target_weight=0.10,
            reasoning="技术面看涨",
            market_view="bullish",
            risk_assessment="medium",
            factor_score=None,
            factor_signal="BUY",
            factor_alpha=0.05,
        )
        assert rec.symbol == "AAPL"
        assert rec.factor_signal == "BUY"

    def test_to_dict_with_factor_score(self):
        """测试带因子评分的字典转换"""
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.7,
            expected_alpha=0.05,
            risk_contribution=0.3,
            signal="BUY",
            reasoning="因子看涨",
        )
        rec = EnhancedRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_50,
            confidence=0.75,
            target_weight=0.10,
            reasoning="技术面看涨",
            market_view="bullish",
            risk_assessment="medium",
            factor_score=factor_score,
            factor_signal="BUY",
            factor_alpha=0.05,
        )
        result = rec.to_dict()
        assert "factor_score" in result
        assert result["factor_signal"] == "BUY"

    def test_to_dict_without_factor_score(self):
        """测试无因子评分的字典转换"""
        rec = EnhancedRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.HOLD,
            confidence=0.50,
            target_weight=0.05,
            reasoning="持有",
            market_view="neutral",
            risk_assessment="low",
            factor_score=None,
            factor_signal=None,
            factor_alpha=None,
        )
        result = rec.to_dict()
        assert result["factor_signal"] is None


class MockFactorScorer(BaseFactorScorer):
    """用于测试的Mock因子评分器"""

    @property
    def asset_class(self) -> str:
        return "stocks"

    @property
    def supported_factors(self):
        return [FactorType.MOMENTUM, FactorType.VALUE]

    def _default_weights(self):
        return {"momentum": 0.5, "value": 0.5}

    def _compute_factor_exposures(self, symbol, data, returns=None):
        return {
            "momentum": FactorExposure(
                factor_type=FactorType.MOMENTUM,
                exposure=0.5,
                z_score=1.0,
                percentile=80,
                signal="LONG",
                confidence=0.7,
            ),
            "value": FactorExposure(
                factor_type=FactorType.VALUE,
                exposure=0.3,
                z_score=0.5,
                percentile=70,
                signal="LONG",
                confidence=0.6,
            ),
        }


class MockBaseExpert(BaseExpert):
    """用于测试的Mock专家类"""

    def __init__(self, llm_provider=None, symbols=None, **kwargs):
        self.llm = llm_provider
        self.symbols = symbols or ["AAPL", "MSFT"]
        # Handle factor_scorer and factor_weight from kwargs
        self.factor_scorer = kwargs.get('factor_scorer')
        self.factor_weight = kwargs.get('factor_weight', 0.3)
        self._factor_scores = {}

    @property
    def name(self) -> str:
        return "MockExpert"

    @property
    def description(self) -> str:
        return "Mock expert for testing"

    @property
    def expertise(self) -> str:
        return "Testing"

    def _build_analysis_prompt(self, market_data, **kwargs) -> str:
        return "Mock analysis prompt"

    def _parse_llm_response(self, response: str, market_data: dict):
        return [
            ExpertRecommendation(
                asset_class="stocks",
                symbol="AAPL",
                action=Action.BUY_50,
                confidence=0.7,
                target_weight=0.10,
                reasoning="看涨",
                market_view="bullish",
                risk_assessment="medium",
            )
        ]

    def analyze(self, market_data, **kwargs):
        return ExpertReport(
            expert_name="MockExpert",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            recommendations=[
                ExpertRecommendation(
                    asset_class="stocks",
                    symbol="AAPL",
                    action=Action.BUY_50,
                    confidence=0.7,
                    target_weight=0.10,
                    reasoning="看涨",
                    market_view="bullish",
                    risk_assessment="medium",
                )
            ],
            overall_view="bullish",
            sector_allocation={"tech": 0.6, "finance": 0.4},
            key_factors=["技术面向好"],
        )


class EnhancedMockExpert(FactorEnhancedExpertMixin, MockBaseExpert):
    """增强版Mock专家"""
    pass


class TestFactorEnhancedExpertMixin:
    """FactorEnhancedExpertMixin测试"""

    @pytest.fixture
    def enhanced_expert(self):
        """创建增强专家"""
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL", "MSFT"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_init_with_factor_scorer(self, enhanced_expert):
        """测试带因子评分器初始化"""
        assert enhanced_expert.factor_scorer is not None
        assert enhanced_expert.factor_weight == 0.4

    def test_init_without_factor_scorer(self):
        """测试无因子评分器初始化"""
        expert = EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
        )
        assert expert.factor_scorer is None

    def test_compute_factor_scores(self, enhanced_expert):
        """测试计算因子评分"""
        market_data = {
            "AAPL": {"price": 180, "volume": 1000000},
            "MSFT": {"price": 350, "volume": 800000},
        }
        enhanced_expert._compute_factor_scores(market_data, None, None)
        assert "AAPL" in enhanced_expert._factor_scores
        assert "MSFT" in enhanced_expert._factor_scores

    def test_enhance_market_data(self, enhanced_expert):
        """测试增强市场数据"""
        enhanced_expert._factor_scores = {
            "AAPL": FactorScore(
                symbol="AAPL",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                factor_exposures={
                    "momentum": FactorExposure(
                        factor_type=FactorType.MOMENTUM,
                        exposure=0.5,
                        z_score=1.0,
                        percentile=80,
                        signal="LONG",
                        confidence=0.7,
                    )
                },
                composite_score=0.7,
                expected_alpha=0.05,
                risk_contribution=0.3,
                signal="BUY",
                reasoning="看涨",
            )
        }
        market_data = {"AAPL": {"price": 180}}
        enhanced = enhanced_expert._enhance_market_data_with_factors(market_data)
        assert "factor_score" in enhanced["AAPL"]
        assert "factor_signal" in enhanced["AAPL"]


class TestAdjustConfidence:
    """置信度调整测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_adjust_confidence_consistent_signals(self, enhanced_expert):
        """测试一致信号提高置信度"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_50,
            confidence=0.7,
            target_weight=0.10,
            reasoning="看涨",
            market_view="bullish",
            risk_assessment="medium",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.8,
            expected_alpha=0.05,
            risk_contribution=0.3,
            signal="BUY",
            reasoning="因子看涨",
        )
        adjusted = enhanced_expert._adjust_confidence(rec, factor_score)
        # 一致信号应提高置信度
        assert adjusted >= 0.7

    def test_adjust_confidence_conflicting_signals(self, enhanced_expert):
        """测试冲突信号降低置信度"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_50,
            confidence=0.7,
            target_weight=0.10,
            reasoning="看涨",
            market_view="bullish",
            risk_assessment="medium",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.3,
            expected_alpha=-0.02,
            risk_contribution=0.3,
            signal="SELL",
            reasoning="因子看跌",
        )
        adjusted = enhanced_expert._adjust_confidence(rec, factor_score)
        # 冲突信号应降低置信度
        assert adjusted < 0.7


class TestAdjustAction:
    """动作调整测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_adjust_action_strong_buy(self, enhanced_expert):
        """测试强烈买入信号"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_100,
            confidence=0.9,
            target_weight=0.15,
            reasoning="",
            market_view="bullish",
            risk_assessment="low",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.9,
            expected_alpha=0.08,
            risk_contribution=0.2,
            signal="STRONG_BUY",
            reasoning="强烈看涨",
        )
        adjusted = enhanced_expert._adjust_action(rec, factor_score)
        # 应保持买入动作
        assert "BUY" in adjusted.value

    def test_adjust_action_sell_signal(self, enhanced_expert):
        """测试卖出信号"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.SELL_50,
            confidence=0.7,
            target_weight=0.05,
            reasoning="",
            market_view="bearish",
            risk_assessment="high",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.2,
            expected_alpha=-0.05,
            risk_contribution=0.4,
            signal="STRONG_SELL",
            reasoning="强烈看跌",
        )
        adjusted = enhanced_expert._adjust_action(rec, factor_score)
        # 应保持卖出动作
        assert "SELL" in adjusted.value


class TestAdjustTargetWeight:
    """目标权重调整测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_adjust_weight_up(self, enhanced_expert):
        """测试向上调整权重"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_50,
            confidence=0.7,
            target_weight=0.10,
            reasoning="",
            market_view="bullish",
            risk_assessment="medium",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=0.9,  # 高评分
            expected_alpha=0.08,
            risk_contribution=0.2,
            signal="STRONG_BUY",
            reasoning="",
        )
        adjusted = enhanced_expert._adjust_target_weight(rec, factor_score)
        # 高因子评分应增加权重
        assert adjusted > 0

    def test_weight_capped_at_25(self, enhanced_expert):
        """测试权重上限"""
        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="AAPL",
            action=Action.BUY_100,
            confidence=1.0,
            target_weight=0.30,  # 原本超过25%
            reasoning="",
            market_view="bullish",
            risk_assessment="low",
        )
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={},
            composite_score=1.0,
            expected_alpha=0.10,
            risk_contribution=0.1,
            signal="STRONG_BUY",
            reasoning="",
        )
        adjusted = enhanced_expert._adjust_target_weight(rec, factor_score)
        assert adjusted <= 0.25


class TestEnhanceReasoning:
    """决策理由增强测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_enhance_reasoning(self, enhanced_expert):
        """测试增强决策理由"""
        original = "技术面看涨"
        factor_score = FactorScore(
            symbol="AAPL",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            factor_exposures={
                "momentum": FactorExposure(
                    factor_type=FactorType.MOMENTUM,
                    exposure=0.5,
                    z_score=1.0,
                    percentile=80,
                    signal="LONG",
                    confidence=0.7,
                )
            },
            composite_score=0.7,
            expected_alpha=0.05,
            risk_contribution=0.3,
            signal="BUY",
            reasoning="动量因子强劲",
        )
        enhanced = enhanced_expert._enhance_reasoning(original, factor_score)
        assert "因子分析" in enhanced
        assert "动量因子强劲" in enhanced


class TestGenerateFactorSummary:
    """因子摘要生成测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL", "MSFT"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_generate_factor_summary(self, enhanced_expert):
        """测试生成因子摘要"""
        enhanced_expert._factor_scores = {
            "AAPL": FactorScore(
                symbol="AAPL",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                factor_exposures={},
                composite_score=0.8,
                expected_alpha=0.06,
                risk_contribution=0.3,
                signal="BUY",
                reasoning="",
            ),
            "MSFT": FactorScore(
                symbol="MSFT",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                factor_exposures={},
                composite_score=0.3,
                expected_alpha=-0.02,
                risk_contribution=0.4,
                signal="SELL",
                reasoning="",
            ),
        }
        summary = enhanced_expert._generate_factor_summary()
        assert len(summary) >= 1
        assert any("因子" in s for s in summary)

    def test_empty_factor_scores(self, enhanced_expert):
        """测试空因子评分"""
        enhanced_expert._factor_scores = {}
        summary = enhanced_expert._generate_factor_summary()
        assert summary == []


class TestGetFactorReport:
    """因子报告获取测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_get_factor_report(self, enhanced_expert):
        """测试获取因子报告"""
        enhanced_expert._factor_scores = {
            "AAPL": FactorScore(
                symbol="AAPL",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                factor_exposures={
                    "momentum": FactorExposure(
                        factor_type=FactorType.MOMENTUM,
                        exposure=0.5,
                        z_score=1.0,
                        percentile=80,
                        signal="LONG",
                        confidence=0.7,
                    )
                },
                composite_score=0.7,
                expected_alpha=0.05,
                risk_contribution=0.3,
                signal="BUY",
                reasoning="动量强劲",
            )
        }
        report = enhanced_expert.get_factor_report("AAPL")
        assert report is not None
        assert "AAPL" in report
        assert "综合评分" in report

    def test_get_factor_report_not_found(self, enhanced_expert):
        """测试未找到因子报告"""
        enhanced_expert._factor_scores = {}
        report = enhanced_expert.get_factor_report("AAPL")
        assert report is None


class TestAnalyzeWithFactors:
    """带因子的分析测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        return EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL", "MSFT"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )

    def test_analyze_with_factors(self, enhanced_expert):
        """测试带因子分析"""
        market_data = {
            "AAPL": {"price": 180, "volume": 1000000},
            "MSFT": {"price": 350, "volume": 800000},
        }
        report = enhanced_expert.analyze_with_factors(
            market_data=market_data,
            returns=None,
        )
        assert isinstance(report, ExpertReport)
        assert len(report.recommendations) > 0


class TestCreateFactorEnhancedExpert:
    """工厂函数测试"""

    def test_create_enhanced_expert(self):
        """测试创建增强专家"""
        scorer = MockFactorScorer()
        expert = create_factor_enhanced_expert(
            MockBaseExpert,
            scorer,
            llm_provider=MagicMock(),
            symbols=["AAPL"],
        )
        assert expert.factor_scorer is not None
        assert hasattr(expert, 'analyze_with_factors')


class TestGetEnhancedExperts:
    """获取预定义增强专家测试"""

    def test_get_enhanced_experts(self):
        """测试获取所有预定义增强专家"""
        experts = get_enhanced_experts()
        assert "stocks" in experts
        assert "bonds" in experts
        assert "commodities" in experts
        assert "reits" in experts
        assert "crypto" in experts
        # 每个条目应该是(类, 评分器类)元组
        for asset_class, (expert_cls, scorer_cls) in experts.items():
            assert expert_cls is not None
            assert scorer_cls is not None


class TestEnhanceRecommendationsWithFactors:
    """因子增强建议测试"""

    @pytest.fixture
    def enhanced_expert(self):
        scorer = MockFactorScorer()
        expert = EnhancedMockExpert(
            llm_provider=MagicMock(),
            symbols=["AAPL"],
            factor_scorer=scorer,
            factor_weight=0.4,
        )
        expert._factor_scores = {
            "AAPL": FactorScore(
                symbol="AAPL",
                asset_class="stocks",
                timestamp=datetime.now().isoformat(),
                factor_exposures={},
                composite_score=0.7,
                expected_alpha=0.05,
                risk_contribution=0.3,
                signal="BUY",
                reasoning="因子看涨",
            )
        }
        return expert

    def test_enhance_with_factors(self, enhanced_expert):
        """测试因子增强建议"""
        recommendations = [
            ExpertRecommendation(
                asset_class="stocks",
                symbol="AAPL",
                action=Action.BUY_50,
                confidence=0.7,
                target_weight=0.10,
                reasoning="技术面看涨",
                market_view="bullish",
                risk_assessment="medium",
            )
        ]
        enhanced = enhanced_expert._enhance_recommendations_with_factors(recommendations)
        assert len(enhanced) == 1
        assert isinstance(enhanced[0], EnhancedRecommendation)
        assert enhanced[0].factor_score is not None

    def test_enhance_without_factor_score(self, enhanced_expert):
        """测试无因子评分的增强"""
        enhanced_expert._factor_scores = {}
        recommendations = [
            ExpertRecommendation(
                asset_class="stocks",
                symbol="MSFT",  # 没有因子评分
                action=Action.HOLD,
                confidence=0.5,
                target_weight=0.05,
                reasoning="持有",
                market_view="neutral",
                risk_assessment="low",
            )
        ]
        enhanced = enhanced_expert._enhance_recommendations_with_factors(recommendations)
        assert len(enhanced) == 1
        assert enhanced[0].factor_score is None

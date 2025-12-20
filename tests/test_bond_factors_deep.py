"""
Deep tests for BondFactorScorer
债券因子评分器深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.factors.bond_factors import BondFactorScorer
from finsage.factors.base_factor import FactorType, FactorExposure, FactorScore


class TestBondFactorScorerInit:
    """BondFactorScorer初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        scorer = BondFactorScorer()
        assert scorer.asset_class == "bonds"
        assert FactorType.CARRY in scorer.supported_factors
        assert FactorType.VALUE in scorer.supported_factors

    def test_default_weights(self):
        """测试默认权重"""
        scorer = BondFactorScorer()
        weights = scorer._default_weights()
        assert "value" in weights
        assert "carry" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_rating_scores(self):
        """测试评级分数映射"""
        scorer = BondFactorScorer()
        assert scorer.RATING_SCORES["AAA"] == 1.0
        assert scorer.RATING_SCORES["BBB"] == 0.60
        assert scorer.RATING_SCORES["D"] == 0.0


class TestComputeValueExposure:
    """价值因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_high_spread_percentile(self, scorer):
        """测试高利差分位数(便宜)"""
        data = {"credit_spread": 0.04, "spread_percentile": 85}
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure > 0  # 高分位数=便宜=正暴露
        assert exposure.signal == "CHEAP"

    def test_low_spread_percentile(self, scorer):
        """测试低利差分位数(贵)"""
        data = {"credit_spread": 0.01, "spread_percentile": 15}
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure < 0  # 低分位数=贵=负暴露
        assert exposure.signal == "EXPENSIVE"

    def test_fair_value(self, scorer):
        """测试公允价值"""
        data = {"credit_spread": 0.02, "spread_percentile": 50}
        exposure = scorer._compute_value_exposure(data)
        assert abs(exposure.exposure) < 0.1
        assert exposure.signal == "FAIR_VALUE"

    def test_default_values(self, scorer):
        """测试默认值"""
        exposure = scorer._compute_value_exposure({})
        assert isinstance(exposure, FactorExposure)


class TestComputeMomentumExposure:
    """动量因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_positive_momentum(self, scorer):
        """测试正动量"""
        data = {"excess_return_6m": 0.05}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure > 0
        assert exposure.signal == "POSITIVE_MOMENTUM"

    def test_negative_momentum(self, scorer):
        """测试负动量"""
        data = {"excess_return_6m": -0.03}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure < 0
        assert exposure.signal == "NEGATIVE_MOMENTUM"

    def test_neutral_momentum(self, scorer):
        """测试中性动量"""
        data = {"excess_return_6m": 0.01}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.signal == "NEUTRAL"

    def test_with_returns_series(self, scorer):
        """测试带收益率序列"""
        returns = pd.Series(np.random.normal(0.001, 0.005, 130))
        exposure = scorer._compute_momentum_exposure({}, returns)
        assert isinstance(exposure, FactorExposure)


class TestComputeCarryExposure:
    """套息因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_high_carry(self, scorer):
        """测试高套息"""
        data = {"ytm": 0.10, "treasury_rate": 0.02, "credit_spread": 0.08}
        exposure = scorer._compute_carry_exposure(data)
        assert exposure.exposure > 0
        # 高套息信号
        assert exposure.signal in ["HIGH_CARRY", "MODERATE_CARRY"]

    def test_low_carry(self, scorer):
        """测试低套息"""
        data = {"ytm": 0.04, "treasury_rate": 0.035, "credit_spread": 0.005}
        exposure = scorer._compute_carry_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "LOW_CARRY"

    def test_moderate_carry(self, scorer):
        """测试中等套息"""
        data = {"ytm": 0.05, "treasury_rate": 0.03, "credit_spread": 0.02}
        exposure = scorer._compute_carry_exposure(data)
        assert exposure.signal == "MODERATE_CARRY"


class TestComputeLowRiskExposure:
    """低风险因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_defensive_bond(self, scorer):
        """测试防御性债券"""
        data = {"rating": "AA", "volatility": 0.03}
        exposure = scorer._compute_low_risk_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "DEFENSIVE"

    def test_high_risk_bond(self, scorer):
        """测试高风险债券"""
        data = {"rating": "B", "volatility": 0.12}
        exposure = scorer._compute_low_risk_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "HIGH_RISK"

    def test_neutral_risk(self, scorer):
        """测试中性风险"""
        data = {"rating": "BBB", "volatility": 0.05}
        exposure = scorer._compute_low_risk_exposure(data)
        assert exposure.signal == "NEUTRAL"


class TestComputeDurationExposure:
    """久期因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_short_duration(self, scorer):
        """测试短久期"""
        data = {"duration": 2.0}
        exposure = scorer._compute_duration_exposure(data)
        assert exposure.exposure < 0  # 短久期=负暴露
        assert exposure.signal == "LOW_RATE_RISK"

    def test_long_duration(self, scorer):
        """测试长久期"""
        data = {"duration": 12.0}
        exposure = scorer._compute_duration_exposure(data)
        assert exposure.exposure > 0  # 长久期=正暴露
        assert exposure.signal == "HIGH_RATE_RISK"

    def test_intermediate_duration(self, scorer):
        """测试中等久期"""
        data = {"duration": 5.0}
        exposure = scorer._compute_duration_exposure(data)
        assert exposure.signal == "MODERATE_RATE_RISK"


class TestComputeCreditExposure:
    """信用因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_investment_grade(self, scorer):
        """测试投资级"""
        data = {"rating": "A", "rating_trend": "stable"}
        exposure = scorer._compute_credit_exposure(data)
        assert exposure.signal == "INVESTMENT_GRADE"

    def test_high_yield(self, scorer):
        """测试高收益"""
        data = {"rating": "BB", "rating_trend": "stable"}
        exposure = scorer._compute_credit_exposure(data)
        assert exposure.signal == "HIGH_YIELD"

    def test_rating_upgrade(self, scorer):
        """测试评级上调"""
        data = {"rating": "BBB", "rating_trend": "upgrade"}
        exposure = scorer._compute_credit_exposure(data)
        # 上调应增加暴露
        assert exposure.exposure > 0

    def test_rating_downgrade(self, scorer):
        """测试评级下调"""
        data = {"rating": "BBB", "rating_trend": "downgrade"}
        exposure = scorer._compute_credit_exposure(data)
        # 下调应减少暴露
        base_exposure = scorer._compute_credit_exposure({"rating": "BBB", "rating_trend": "stable"})
        assert exposure.exposure < base_exposure.exposure


class TestComputeFactorExposures:
    """因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_compute_all_exposures(self, scorer):
        """测试计算所有暴露"""
        data = {
            "ytm": 0.06,
            "duration": 7.0,
            "credit_spread": 0.02,
            "spread_percentile": 60,
            "rating": "A",
            "volatility": 0.04,
        }
        exposures = scorer._compute_factor_exposures("LQD", data)
        assert "value" in exposures
        assert "momentum" in exposures
        assert "carry" in exposures
        assert "low_risk" in exposures
        assert "duration" in exposures
        assert "credit" in exposures


class TestGetFactorPremiums:
    """因子溢价获取测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()
        assert "carry" in premiums
        assert "value" in premiums
        assert premiums["carry"] > 0


class TestGetDurationRecommendation:
    """久期建议测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_rising_rates_recommendation(self, scorer):
        """测试利率上升建议"""
        rec = scorer.get_duration_recommendation("rising", 7.0)
        assert rec["action"] == "REDUCE_DURATION"
        assert rec["target_duration"] < 7.0

    def test_falling_rates_recommendation(self, scorer):
        """测试利率下降建议"""
        rec = scorer.get_duration_recommendation("falling", 7.0)
        assert rec["action"] == "EXTEND_DURATION"
        assert rec["target_duration"] > 7.0

    def test_stable_rates_recommendation(self, scorer):
        """测试利率稳定建议"""
        rec = scorer.get_duration_recommendation("stable", 7.0)
        assert rec["action"] == "MAINTAIN"
        assert rec["target_duration"] == 7.0


class TestGetCreditAllocation:
    """信用配置建议测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_conservative_allocation(self, scorer):
        """测试保守配置"""
        allocation = scorer.get_credit_allocation("conservative", {})
        assert allocation["treasury"] >= 0.25
        assert allocation["high_yield"] <= 0.15

    def test_aggressive_allocation(self, scorer):
        """测试激进配置"""
        allocation = scorer.get_credit_allocation("aggressive", {})
        assert allocation["high_yield"] >= 0.35
        assert allocation["treasury"] <= 0.15

    def test_moderate_allocation(self, scorer):
        """测试中性配置"""
        allocation = scorer.get_credit_allocation("moderate", {})
        total = sum(allocation.values())
        assert abs(total - 1.0) < 0.01


class TestScore:
    """完整评分测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_score_basic(self, scorer):
        """测试基本评分"""
        data = {
            "ytm": 0.055,
            "duration": 6.0,
            "credit_spread": 0.025,
            "spread_percentile": 65,
            "rating": "A-",
            "volatility": 0.04,
            "treasury_rate": 0.03,
        }
        score = scorer.score("LQD", data)
        assert isinstance(score, FactorScore)
        assert score.asset_class == "bonds"
        assert 0 <= score.composite_score <= 1

    def test_score_with_returns(self, scorer):
        """测试带收益率评分"""
        data = {"ytm": 0.05, "duration": 5.0, "rating": "BBB"}
        returns = pd.Series(np.random.normal(0.0002, 0.003, 130))
        score = scorer.score("HYG", data, returns)
        assert isinstance(score, FactorScore)

    def test_score_with_regime(self, scorer):
        """测试带市场体制评分"""
        data = {"ytm": 0.06, "duration": 8.0, "rating": "A"}
        score = scorer.score("TLT", data, market_regime="bear")
        assert isinstance(score, FactorScore)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_unknown_rating(self, scorer):
        """测试未知评级"""
        data = {"rating": "NR"}  # Not Rated
        exposure = scorer._compute_low_risk_exposure(data)
        assert isinstance(exposure, FactorExposure)

    def test_zero_volatility(self, scorer):
        """测试零波动率"""
        data = {"rating": "AAA", "volatility": 0.0}
        exposure = scorer._compute_low_risk_exposure(data)
        assert exposure.exposure > 0  # 零波动应该是好的

    def test_extreme_duration(self, scorer):
        """测试极端久期"""
        data = {"duration": 30.0}
        exposure = scorer._compute_duration_exposure(data)
        assert exposure.exposure <= 1.0  # 应该被裁剪

    def test_negative_spread(self, scorer):
        """测试负利差(罕见)"""
        data = {"ytm": 0.03, "treasury_rate": 0.04, "credit_spread": -0.01}
        exposure = scorer._compute_carry_exposure(data)
        assert isinstance(exposure, FactorExposure)


class TestRatingScores:
    """评级分数测试"""

    @pytest.fixture
    def scorer(self):
        return BondFactorScorer()

    def test_all_ratings(self, scorer):
        """测试所有评级分数"""
        for rating, score in scorer.RATING_SCORES.items():
            assert 0 <= score <= 1
            data = {"rating": rating, "volatility": 0.05}
            exposure = scorer._compute_low_risk_exposure(data)
            assert isinstance(exposure, FactorExposure)

    def test_rating_order(self, scorer):
        """测试评级顺序"""
        assert scorer.RATING_SCORES["AAA"] > scorer.RATING_SCORES["AA"]
        assert scorer.RATING_SCORES["AA"] > scorer.RATING_SCORES["A"]
        assert scorer.RATING_SCORES["A"] > scorer.RATING_SCORES["BBB"]
        assert scorer.RATING_SCORES["BBB"] > scorer.RATING_SCORES["BB"]

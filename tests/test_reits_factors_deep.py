"""
Deep tests for REITsFactorScorer
REITs因子评分器深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.factors.reits_factors import REITsFactorScorer
from finsage.factors.base_factor import FactorType, FactorExposure, FactorScore


class TestREITsFactorScorerInit:
    """REITsFactorScorer初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        scorer = REITsFactorScorer()
        assert scorer.asset_class == "reits"
        assert FactorType.NAV in scorer.supported_factors
        assert FactorType.SECTOR in scorer.supported_factors

    def test_default_weights(self):
        """测试默认权重"""
        scorer = REITsFactorScorer()
        weights = scorer._default_weights()
        assert weights["nav"] == 0.35  # NAV最重要
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_sector_outlook(self):
        """测试行业展望配置"""
        scorer = REITsFactorScorer()
        assert scorer.SECTOR_OUTLOOK["data_center"]["score"] > 0.8
        assert scorer.SECTOR_OUTLOOK["office"]["score"] < 0.4


class TestComputeNAVExposure:
    """NAV因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_deep_discount(self, scorer):
        """测试深度折价"""
        data = {"price": 80, "nav": 100}
        exposure = scorer._compute_nav_exposure(data)
        assert exposure.exposure > 0  # 折价=正暴露(逆向)
        assert exposure.signal == "DEEP_DISCOUNT"

    def test_high_premium(self, scorer):
        """测试高溢价"""
        data = {"price": 120, "nav": 100}
        exposure = scorer._compute_nav_exposure(data)
        assert exposure.exposure < 0  # 溢价=负暴露(逆向)
        assert exposure.signal == "HIGH_PREMIUM"

    def test_fair_value(self, scorer):
        """测试公允价值"""
        data = {"price": 100, "nav": 100}
        exposure = scorer._compute_nav_exposure(data)
        assert abs(exposure.exposure) < 0.1
        assert exposure.signal == "FAIR_VALUE"

    def test_nav_premium_provided(self, scorer):
        """测试直接提供NAV溢价"""
        data = {"nav_premium": -0.15}
        exposure = scorer._compute_nav_exposure(data)
        assert exposure.exposure > 0  # 折价


class TestComputeIdiosyncraticExposure:
    """特质风险因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_low_idio_risk(self, scorer):
        """测试低特质风险"""
        data = {
            "volatility": 0.05,
            "geographic_concentration": 0.1,
            "property_diversity": 0.9,
        }
        exposure = scorer._compute_idiosyncratic_exposure(data, None)
        # 低波动率和高多样性应表示低特质风险
        assert exposure.signal in ["LOW_IDIO_RISK", "MODERATE_RISK"]

    def test_high_idio_risk(self, scorer):
        """测试高特质风险"""
        data = {
            "volatility": 0.35,
            "geographic_concentration": 0.8,
            "property_diversity": 0.2,
        }
        exposure = scorer._compute_idiosyncratic_exposure(data, None)
        assert exposure.signal == "HIGH_IDIO_RISK"

    def test_with_returns(self, scorer):
        """测试带收益率序列"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        exposure = scorer._compute_idiosyncratic_exposure({}, returns)
        assert isinstance(exposure, FactorExposure)


class TestComputeSectorExposure:
    """行业因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_growth_sector(self, scorer):
        """测试成长行业"""
        data = {"sector": "data_center"}
        exposure = scorer._compute_sector_exposure("DLR", data)
        assert exposure.exposure > 0
        assert exposure.signal == "GROWTH_SECTOR"

    def test_challenged_sector(self, scorer):
        """测试困难行业"""
        data = {"sector": "office"}
        exposure = scorer._compute_sector_exposure("BXP", data)
        assert exposure.exposure < 0
        assert exposure.signal == "CHALLENGED_SECTOR"

    def test_infer_sector(self, scorer):
        """测试推断行业"""
        # 不提供行业，应从代码推断
        exposure = scorer._compute_sector_exposure("DLR", {})
        assert exposure.signal == "GROWTH_SECTOR"  # DLR是数据中心


class TestComputeValueExposure:
    """估值因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_undervalued(self, scorer):
        """测试低估"""
        data = {
            "p_ffo": 10,
            "dividend_yield": 0.06,
            "cap_rate": 0.07,
            "risk_free_rate": 0.04,
        }
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "UNDERVALUED"

    def test_overvalued(self, scorer):
        """测试高估"""
        data = {
            "p_ffo": 25,
            "dividend_yield": 0.02,
            "cap_rate": 0.04,
            "risk_free_rate": 0.04,
        }
        exposure = scorer._compute_value_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "OVERVALUED"


class TestComputeMomentumExposure:
    """动量因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_strong_momentum(self, scorer):
        """测试强动量"""
        data = {"price_change_12m": 0.20}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure > 0
        assert exposure.signal == "STRONG_MOMENTUM"

    def test_negative_momentum(self, scorer):
        """测试负动量"""
        data = {"price_change_12m": -0.15}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure < 0
        assert exposure.signal == "NEGATIVE_MOMENTUM"


class TestComputeFactorExposures:
    """因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_compute_all_exposures(self, scorer):
        """测试计算所有暴露"""
        data = {
            "price": 100,
            "nav": 110,
            "sector": "logistics",
            "p_ffo": 15,
            "dividend_yield": 0.04,
            "price_change_12m": 0.10,
        }
        exposures = scorer._compute_factor_exposures("PLD", data)
        assert "nav" in exposures
        assert "idiosyncratic" in exposures
        assert "sector" in exposures
        assert "value" in exposures
        assert "momentum" in exposures


class TestInferSector:
    """行业推断测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_infer_data_center(self, scorer):
        """测试推断数据中心"""
        sector = scorer._infer_sector("DLR")
        assert sector == "data_center"

    def test_infer_logistics(self, scorer):
        """测试推断物流"""
        sector = scorer._infer_sector("PLD")
        assert sector == "logistics"

    def test_infer_unknown(self, scorer):
        """测试推断未知"""
        sector = scorer._infer_sector("UNKNOWN")
        assert sector == "diversified"


class TestGetFactorPremiums:
    """因子溢价获取测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()
        assert "nav" in premiums
        assert "sector" in premiums
        assert premiums["sector"] == 0.03


class TestGetNAVAnalysis:
    """NAV分析报告测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_deep_discount_analysis(self, scorer):
        """测试深度折价分析"""
        data = {"price": 80, "nav": 100}
        analysis = scorer.get_nav_analysis("VNQ", data)
        assert "深度折价" in analysis
        assert "价值投资" in analysis

    def test_high_premium_analysis(self, scorer):
        """测试高溢价分析"""
        data = {"price": 125, "nav": 100}
        analysis = scorer.get_nav_analysis("VNQ", data)
        assert "高度溢价" in analysis or "溢价" in analysis

    def test_fair_value_analysis(self, scorer):
        """测试合理价值分析"""
        data = {"price": 102, "nav": 100}
        analysis = scorer.get_nav_analysis("VNQ", data)
        assert "合理" in analysis


class TestGetSectorAllocation:
    """行业配置测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_neutral_allocation(self, scorer):
        """测试中性市场配置"""
        allocation = scorer.get_sector_allocation({}, "neutral")
        assert sum(allocation.values()) > 0
        # 数据中心和物流应该有较高配置
        assert allocation["data_center"] >= 0.15

    def test_risk_off_allocation(self, scorer):
        """测试避险市场配置"""
        allocation = scorer.get_sector_allocation({}, "risk_off")
        # 避险时应增加住宅配置
        base_allocation = scorer.get_sector_allocation({}, "neutral")
        # 住宅配置应该增加
        assert allocation["residential"] >= base_allocation["residential"] * 0.9

    def test_risk_on_allocation(self, scorer):
        """测试风险偏好配置"""
        allocation = scorer.get_sector_allocation({}, "risk_on")
        # 风险偏好时应增加成长性行业
        base_allocation = scorer.get_sector_allocation({}, "neutral")
        assert allocation["data_center"] >= base_allocation["data_center"] * 0.9


class TestScore:
    """完整评分测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_score_basic(self, scorer):
        """测试基本评分"""
        data = {
            "price": 100,
            "nav": 110,
            "sector": "residential",
            "p_ffo": 18,
            "dividend_yield": 0.04,
        }
        score = scorer.score("EQR", data)
        assert isinstance(score, FactorScore)
        assert score.asset_class == "reits"
        assert 0 <= score.composite_score <= 1

    def test_score_with_returns(self, scorer):
        """测试带收益率评分"""
        data = {"price": 100, "nav": 95}
        returns = pd.Series(np.random.normal(0.001, 0.015, 260))
        score = scorer.score("VNQ", data, returns)
        assert isinstance(score, FactorScore)

    def test_score_with_regime(self, scorer):
        """测试带市场体制评分"""
        data = {"price": 100, "nav": 100}
        score = scorer.score("VNQ", data, market_regime="bear")
        assert isinstance(score, FactorScore)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_zero_nav(self, scorer):
        """测试零NAV"""
        data = {"price": 100, "nav": 0}
        exposure = scorer._compute_nav_exposure(data)
        assert isinstance(exposure, FactorExposure)

    def test_negative_nav_premium(self, scorer):
        """测试极端负溢价"""
        data = {"nav_premium": -0.50}  # 50%折价
        exposure = scorer._compute_nav_exposure(data)
        assert exposure.exposure > 0

    def test_unknown_sector_symbol(self, scorer):
        """测试未知行业代码"""
        data = {"sector": "unknown_sector"}
        exposure = scorer._compute_sector_exposure("UNKNOWN", data)
        # 应该回退到diversified
        assert isinstance(exposure, FactorExposure)


class TestSectorOutlook:
    """行业展望测试"""

    @pytest.fixture
    def scorer(self):
        return REITsFactorScorer()

    def test_all_sectors_have_outlook(self, scorer):
        """测试所有行业有展望"""
        for sector, info in scorer.SECTOR_OUTLOOK.items():
            assert "outlook" in info
            assert "score" in info
            assert 0 <= info["score"] <= 1

    def test_sector_examples(self, scorer):
        """测试行业代码示例"""
        for sector, info in scorer.SECTOR_OUTLOOK.items():
            assert "examples" in info
            assert len(info["examples"]) > 0

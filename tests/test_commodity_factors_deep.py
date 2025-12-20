"""
Deep tests for CommodityFactorScorer
商品因子评分器深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.factors.commodity_factors import CommodityFactorScorer
from finsage.factors.base_factor import FactorType, FactorExposure, FactorScore


class TestCommodityFactorScorerInit:
    """CommodityFactorScorer初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        scorer = CommodityFactorScorer()
        assert scorer.asset_class == "commodities"
        assert FactorType.TERM_STRUCTURE in scorer.supported_factors
        assert FactorType.MOMENTUM in scorer.supported_factors

    def test_default_weights(self):
        """测试默认权重"""
        scorer = CommodityFactorScorer()
        weights = scorer._default_weights()
        assert weights["term_structure"] == 0.40  # 期限结构最重要
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_commodity_sectors(self):
        """测试商品板块分类"""
        scorer = CommodityFactorScorer()
        assert "energy" in scorer.COMMODITY_SECTORS
        assert "precious_metals" in scorer.COMMODITY_SECTORS
        assert "GLD" in scorer.COMMODITY_SECTORS["precious_metals"]


class TestComputeTermStructureExposure:
    """期限结构因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_backwardation(self, scorer):
        """测试Backwardation(现货溢价)"""
        data = {"front_price": 100, "back_price": 95, "days_to_roll": 30}
        exposure = scorer._compute_term_structure_exposure(data)
        assert exposure.exposure > 0  # Backwardation=正收益
        assert exposure.signal == "LONG"

    def test_contango(self, scorer):
        """测试Contango(期货溢价)"""
        data = {"front_price": 95, "back_price": 100, "days_to_roll": 30}
        exposure = scorer._compute_term_structure_exposure(data)
        assert exposure.exposure < 0  # Contango=负收益
        assert exposure.signal == "AVOID"

    def test_flat_structure(self, scorer):
        """测试平坦结构"""
        data = {"front_price": 100, "back_price": 99.8, "days_to_roll": 30}
        exposure = scorer._compute_term_structure_exposure(data)
        # 非常小的价差可能仍被视为LONG或NEUTRAL
        assert exposure.signal in ["NEUTRAL", "LONG"]

    def test_default_values(self, scorer):
        """测试默认值"""
        exposure = scorer._compute_term_structure_exposure({})
        assert isinstance(exposure, FactorExposure)

    def test_zero_back_price(self, scorer):
        """测试零远月价格处理"""
        data = {"front_price": 100, "back_price": 0, "days_to_roll": 30}
        exposure = scorer._compute_term_structure_exposure(data)
        assert isinstance(exposure, FactorExposure)


class TestComputeMomentumExposure:
    """动量因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_strong_positive_momentum(self, scorer):
        """测试强正动量"""
        data = {"price_change_12m": 0.20}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure > 0
        assert exposure.signal == "STRONG_MOMENTUM"

    def test_weak_positive_momentum(self, scorer):
        """测试弱正动量"""
        data = {"price_change_12m": 0.05}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.signal == "LONG"

    def test_negative_momentum(self, scorer):
        """测试负动量"""
        data = {"price_change_12m": -0.20}
        exposure = scorer._compute_momentum_exposure(data, None)
        assert exposure.exposure < 0
        assert exposure.signal == "SHORT"

    def test_with_returns_series(self, scorer):
        """测试带收益率序列"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 260))
        exposure = scorer._compute_momentum_exposure({}, returns)
        assert isinstance(exposure, FactorExposure)


class TestComputeBasisExposure:
    """基差因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_tight_supply(self, scorer):
        """测试供应紧张(现货溢价)"""
        data = {"spot_price": 105, "front_price": 100}
        exposure = scorer._compute_basis_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "TIGHT_SUPPLY"

    def test_oversupply(self, scorer):
        """测试供应过剩(期货溢价)"""
        data = {"spot_price": 95, "front_price": 100}
        exposure = scorer._compute_basis_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "OVERSUPPLY"

    def test_neutral_basis(self, scorer):
        """测试中性基差"""
        data = {"spot_price": 100, "front_price": 100}
        exposure = scorer._compute_basis_exposure(data)
        assert exposure.signal == "NEUTRAL"

    def test_zero_futures_price(self, scorer):
        """测试零期货价格处理"""
        data = {"spot_price": 100, "front_price": 0}
        exposure = scorer._compute_basis_exposure(data)
        assert isinstance(exposure, FactorExposure)


class TestComputeCarryExposure:
    """套息因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_high_carry(self, scorer):
        """测试高Carry"""
        data = {
            "front_price": 100,
            "back_price": 95,
            "days_to_roll": 30,
            "risk_free_rate": 0.05,
            "storage_cost": 0.01,
        }
        exposure = scorer._compute_carry_exposure(data)
        assert exposure.exposure > 0
        assert exposure.signal == "HIGH_CARRY"

    def test_negative_carry(self, scorer):
        """测试负Carry"""
        data = {
            "front_price": 95,
            "back_price": 100,
            "days_to_roll": 30,
            "risk_free_rate": 0.02,
            "storage_cost": 0.03,
        }
        exposure = scorer._compute_carry_exposure(data)
        assert exposure.exposure < 0
        assert exposure.signal == "NEGATIVE_CARRY"


class TestComputeFactorExposures:
    """因子暴露计算测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_compute_all_exposures(self, scorer):
        """测试计算所有暴露"""
        data = {
            "front_price": 100,
            "back_price": 98,
            "spot_price": 101,
            "price_change_12m": 0.10,
            "days_to_roll": 30,
        }
        exposures = scorer._compute_factor_exposures("GLD", data)
        assert "term_structure" in exposures
        assert "momentum" in exposures
        assert "basis" in exposures
        assert "carry" in exposures


class TestGetFactorPremiums:
    """因子溢价获取测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_get_factor_premiums(self, scorer):
        """测试获取因子溢价"""
        premiums = scorer._get_factor_premiums()
        assert premiums["term_structure"] == 0.04
        assert premiums["momentum"] == 0.05


class TestGetTermStructureSummary:
    """期限结构摘要测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_backwardation_summary(self, scorer):
        """测试Backwardation摘要"""
        data = {"front_price": 100, "back_price": 95, "days_to_roll": 30}
        summary = scorer.get_term_structure_summary("CL", data)
        assert "Backwardation" in summary
        assert "做多" in summary

    def test_contango_summary(self, scorer):
        """测试Contango摘要"""
        data = {"front_price": 95, "back_price": 100, "days_to_roll": 30}
        summary = scorer.get_term_structure_summary("CL", data)
        assert "Contango" in summary
        assert "展期损失" in summary


class TestGetSectorAllocation:
    """板块配置测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_sector_allocation_basic(self, scorer):
        """测试基本板块配置"""
        scores = {
            "GLD": FactorScore(
                symbol="GLD",
                asset_class="commodities",
                timestamp=datetime.now().isoformat(),
                factor_exposures={},
                composite_score=0.7,
                expected_alpha=0.05,
                risk_contribution=0.3,
                signal="BUY",
                reasoning="",
            ),
            "USO": FactorScore(
                symbol="USO",
                asset_class="commodities",
                timestamp=datetime.now().isoformat(),
                factor_exposures={},
                composite_score=0.5,
                expected_alpha=0.02,
                risk_contribution=0.4,
                signal="HOLD",
                reasoning="",
            ),
        }
        allocation = scorer.get_sector_allocation(scores)
        assert sum(allocation.values()) > 0

    def test_empty_scores(self, scorer):
        """测试空评分"""
        allocation = scorer.get_sector_allocation({})
        # 应该返回等权配置
        assert abs(sum(allocation.values()) - 1.0) < 0.01


class TestScore:
    """完整评分测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_score_basic(self, scorer):
        """测试基本评分"""
        data = {
            "front_price": 100,
            "back_price": 98,
            "spot_price": 101,
            "price_change_12m": 0.10,
            "days_to_roll": 30,
        }
        score = scorer.score("GLD", data)
        assert isinstance(score, FactorScore)
        assert score.asset_class == "commodities"
        assert 0 <= score.composite_score <= 1

    def test_score_with_returns(self, scorer):
        """测试带收益率评分"""
        data = {"front_price": 100, "back_price": 95}
        returns = pd.Series(np.random.normal(0.001, 0.02, 260))
        score = scorer.score("GLD", data, returns)
        assert isinstance(score, FactorScore)

    def test_score_with_regime(self, scorer):
        """测试带市场体制评分"""
        data = {"front_price": 100, "back_price": 98}
        score = scorer.score("GLD", data, market_regime="bull")
        assert isinstance(score, FactorScore)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_zero_days_to_roll(self, scorer):
        """测试零展期天数"""
        data = {"front_price": 100, "back_price": 95, "days_to_roll": 0}
        exposure = scorer._compute_term_structure_exposure(data)
        assert isinstance(exposure, FactorExposure)

    def test_negative_prices(self, scorer):
        """测试负价格(理论上可能,如天然气)"""
        data = {"front_price": -5, "back_price": -3, "days_to_roll": 30}
        # 应该能处理而不崩溃
        exposure = scorer._compute_term_structure_exposure(data)
        assert isinstance(exposure, FactorExposure)

    def test_very_large_roll_yield(self, scorer):
        """测试极大展期收益"""
        data = {"front_price": 100, "back_price": 50, "days_to_roll": 30}
        exposure = scorer._compute_term_structure_exposure(data)
        assert exposure.exposure <= 1.0  # 应该被裁剪

    def test_missing_optional_data(self, scorer):
        """测试缺少可选数据"""
        data = {"price": 100}  # 只有基本价格
        exposures = scorer._compute_factor_exposures("GLD", data)
        assert len(exposures) > 0


class TestCommoditySectors:
    """商品板块测试"""

    @pytest.fixture
    def scorer(self):
        return CommodityFactorScorer()

    def test_energy_sector(self, scorer):
        """测试能源板块"""
        assert "CL" in scorer.COMMODITY_SECTORS["energy"]
        assert "NG" in scorer.COMMODITY_SECTORS["energy"]

    def test_precious_metals_sector(self, scorer):
        """测试贵金属板块"""
        assert "GC" in scorer.COMMODITY_SECTORS["precious_metals"]
        assert "GLD" in scorer.COMMODITY_SECTORS["precious_metals"]

    def test_agriculture_sector(self, scorer):
        """测试农产品板块"""
        assert "ZC" in scorer.COMMODITY_SECTORS["agriculture"]
        assert "ZS" in scorer.COMMODITY_SECTORS["agriculture"]

"""
Deep tests for Core-Satellite Strategy
核心卫星策略深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.strategies.core_satellite import CoreSatelliteStrategy


class TestCoreSatelliteInit:
    """CoreSatelliteStrategy初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        strategy = CoreSatelliteStrategy()
        assert strategy.core_ratio == 0.70
        assert strategy.min_core_ratio == 0.50
        assert strategy.max_core_ratio == 0.90

    def test_custom_init(self):
        """测试自定义初始化"""
        strategy = CoreSatelliteStrategy(
            core_ratio=0.60,
            min_core_ratio=0.40,
            max_core_ratio=0.80
        )
        assert strategy.core_ratio == 0.60
        assert strategy.min_core_ratio == 0.40
        assert strategy.max_core_ratio == 0.80

    def test_name_property(self):
        """测试名称属性"""
        strategy = CoreSatelliteStrategy()
        assert strategy.name == "core_satellite"

    def test_description_property(self):
        """测试描述属性"""
        strategy = CoreSatelliteStrategy()
        assert "核心卫星" in strategy.description
        assert "Core-Satellite" in strategy.description

    def test_rebalance_frequency(self):
        """测试再平衡频率"""
        strategy = CoreSatelliteStrategy()
        assert strategy.rebalance_frequency == "quarterly"


class TestComputeAllocation:
    """配置计算测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    @pytest.fixture
    def sample_market_data(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return {
            "stocks": pd.DataFrame({
                "SPY": np.random.normal(0.001, 0.02, 100),
            }, index=dates),
            "bonds": pd.DataFrame({
                "TLT": np.random.normal(0.0005, 0.01, 100),
            }, index=dates),
            "commodities": pd.DataFrame({
                "GLD": np.random.normal(0.0003, 0.015, 100),
            }, index=dates),
        }

    def test_basic_allocation(self, strategy, sample_market_data):
        """测试基本配置计算"""
        allocation = strategy.compute_allocation(sample_market_data)

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_allocation_with_risk_profiles(self, strategy, sample_market_data):
        """测试不同风险偏好的配置"""
        alloc_conservative = strategy.compute_allocation(
            sample_market_data, risk_profile="conservative"
        )
        alloc_aggressive = strategy.compute_allocation(
            sample_market_data, risk_profile="aggressive"
        )

        assert abs(sum(alloc_conservative.values()) - 1.0) < 0.01
        assert abs(sum(alloc_aggressive.values()) - 1.0) < 0.01

    def test_allocation_with_expert_views(self, strategy, sample_market_data):
        """测试带专家观点的配置"""
        expert_views = {
            "stocks": {"sentiment": 0.5, "conviction": 0.7},
            "bonds": {"sentiment": -0.2, "conviction": 0.5},
        }
        allocation = strategy.compute_allocation(
            sample_market_data,
            expert_views=expert_views
        )

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01


class TestAdjustCoreRatio:
    """核心比例调整测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_bull_market_adjustment(self, strategy):
        """测试牛市调整"""
        ratio = strategy._adjust_core_ratio("bull", "moderate")
        # 牛市应该减少核心比例
        assert ratio < strategy.core_ratio + 0.01

    def test_bear_market_adjustment(self, strategy):
        """测试熊市调整"""
        ratio = strategy._adjust_core_ratio("bear", "moderate")
        # 熊市应该增加核心比例
        assert ratio > strategy.core_ratio - 0.01

    def test_volatile_market_adjustment(self, strategy):
        """测试震荡市调整"""
        ratio = strategy._adjust_core_ratio("volatile", "moderate")
        assert strategy.min_core_ratio <= ratio <= strategy.max_core_ratio

    def test_conservative_profile_adjustment(self, strategy):
        """测试保守型配置调整"""
        ratio = strategy._adjust_core_ratio("normal", "conservative")
        # 保守型应该有更高的核心比例
        assert ratio >= strategy.core_ratio

    def test_aggressive_profile_adjustment(self, strategy):
        """测试激进型配置调整"""
        ratio = strategy._adjust_core_ratio("normal", "aggressive")
        # 激进型应该有更低的核心比例
        assert ratio <= strategy.core_ratio

    def test_ratio_within_bounds(self, strategy):
        """测试比例在边界内"""
        for regime in ["bull", "bear", "volatile", "normal"]:
            for profile in ["conservative", "moderate", "aggressive"]:
                ratio = strategy._adjust_core_ratio(regime, profile)
                assert strategy.min_core_ratio <= ratio <= strategy.max_core_ratio


class TestBuildCorePortfolio:
    """核心组合构建测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_build_core_with_benchmark(self, strategy):
        """测试使用基准构建核心"""
        benchmark = {"stocks": 0.6, "bonds": 0.4}
        core = strategy._build_core_portfolio(
            ["stocks", "bonds"], {}, "moderate", benchmark
        )

        assert core == benchmark

    def test_build_core_without_benchmark(self, strategy):
        """测试不使用基准构建核心"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        core = strategy._build_core_portfolio(
            ["stocks", "bonds"], market_data, "moderate", None
        )

        assert len(core) == 2
        assert abs(sum(core.values()) - 1.0) < 0.01


class TestBuildSatellitePortfolio:
    """卫星组合构建测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_build_satellite_basic(self, strategy):
        """测试基本卫星组合构建"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        risk_params = {"max_equity": 0.6}
        satellite = strategy._build_satellite_portfolio(
            ["stocks", "bonds"], market_data, None, risk_params, None
        )

        assert len(satellite) == 2
        assert abs(sum(satellite.values()) - 1.0) < 0.01

    def test_build_satellite_with_views(self, strategy):
        """测试带观点的卫星组合"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        expert_views = {"stocks": {"sentiment": 0.8, "conviction": 0.9}}
        risk_params = {"max_equity": 0.6}
        satellite = strategy._build_satellite_portfolio(
            ["stocks", "bonds"], market_data, expert_views, risk_params, None
        )

        assert len(satellite) == 2


class TestBlendPortfolios:
    """组合混合测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_blend_basic(self, strategy):
        """测试基本混合"""
        core = {"stocks": 0.4, "bonds": 0.6}
        satellite = {"stocks": 0.8, "bonds": 0.2}
        core_ratio = 0.7

        blended = strategy._blend_portfolios(
            core, satellite, core_ratio, ["stocks", "bonds"]
        )

        expected_stocks = 0.7 * 0.4 + 0.3 * 0.8
        expected_bonds = 0.7 * 0.6 + 0.3 * 0.2

        assert abs(blended["stocks"] - expected_stocks) < 0.01
        assert abs(blended["bonds"] - expected_bonds) < 0.01


class TestGetPortfolioDecomposition:
    """组合分解测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_decomposition_structure(self, strategy):
        """测试分解结构"""
        final = {"stocks": 0.5, "bonds": 0.5}
        core = {"stocks": 0.4, "bonds": 0.6}
        satellite = {"stocks": 0.8, "bonds": 0.2}
        core_ratio = 0.7

        decomposition = strategy.get_portfolio_decomposition(
            final, core, satellite, core_ratio
        )

        assert "core_ratio" in decomposition
        assert "satellite_ratio" in decomposition
        assert "core_allocation" in decomposition
        assert "satellite_allocation" in decomposition
        assert "final_allocation" in decomposition
        assert "asset_breakdown" in decomposition


class TestComputeTrackingError:
    """跟踪误差计算测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_tracking_error_basic(self, strategy):
        """测试基本跟踪误差"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        satellite = {"stocks": 0.8, "bonds": 0.2}
        core = {"stocks": 0.4, "bonds": 0.6}

        te = strategy.compute_tracking_error(satellite, core, market_data)

        assert te >= 0
        assert isinstance(te, float)


class TestGetActiveShare:
    """主动份额计算测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_active_share_identical(self, strategy):
        """测试相同权重的主动份额"""
        weights = {"stocks": 0.5, "bonds": 0.5}
        active_share = strategy.get_active_share(weights, weights)

        assert active_share == 0

    def test_active_share_different(self, strategy):
        """测试不同权重的主动份额"""
        satellite = {"stocks": 0.8, "bonds": 0.2}
        core = {"stocks": 0.4, "bonds": 0.6}

        active_share = strategy.get_active_share(satellite, core)

        # Active share = 0.5 * (|0.8-0.4| + |0.2-0.6|) = 0.5 * 0.8 = 0.4
        assert abs(active_share - 0.4) < 0.01


class TestRecommendSatelliteAdjustments:
    """卫星调整建议测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_recommendations_structure(self, strategy):
        """测试建议结构"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        current_satellite = {"stocks": 0.5, "bonds": 0.5}
        recommendations = strategy.recommend_satellite_adjustments(
            current_satellite, market_data
        )

        assert "stocks" in recommendations
        assert "bonds" in recommendations


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def strategy(self):
        return CoreSatelliteStrategy()

    def test_full_workflow(self, strategy):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        market_data = {
            "stocks": pd.DataFrame({
                "SPY": np.random.normal(0.0004, 0.015, 252),
            }, index=dates),
            "bonds": pd.DataFrame({
                "TLT": np.random.normal(0.0002, 0.008, 252),
            }, index=dates),
            "commodities": pd.DataFrame({
                "GLD": np.random.normal(0.0001, 0.012, 252),
            }, index=dates),
        }

        expert_views = {
            "stocks": {"sentiment": 0.3, "conviction": 0.6},
        }

        allocation = strategy.compute_allocation(
            market_data,
            expert_views=expert_views,
            risk_profile="moderate",
            market_regime="normal"
        )

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in allocation.values())

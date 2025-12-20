"""
Deep tests for Tactical Asset Allocation Strategy
战术资产配置策略深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.strategies.tactical_allocation import TacticalAllocationStrategy


class TestTacticalAllocationInit:
    """TacticalAllocationStrategy初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        strategy = TacticalAllocationStrategy()
        assert strategy.max_tactical_deviation == 0.15
        assert strategy.signal_decay == 0.9

    def test_custom_init(self):
        """测试自定义初始化"""
        strategy = TacticalAllocationStrategy(
            max_tactical_deviation=0.20,
            signal_decay=0.8
        )
        assert strategy.max_tactical_deviation == 0.20
        assert strategy.signal_decay == 0.8

    def test_name_property(self):
        """测试名称属性"""
        strategy = TacticalAllocationStrategy()
        assert strategy.name == "tactical_allocation"

    def test_description_property(self):
        """测试描述属性"""
        strategy = TacticalAllocationStrategy()
        assert "战术" in strategy.description
        assert "Tactical" in strategy.description

    def test_rebalance_frequency(self):
        """测试再平衡频率"""
        strategy = TacticalAllocationStrategy()
        assert strategy.rebalance_frequency == "monthly"


class TestComputeAllocation:
    """配置计算测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    @pytest.fixture
    def sample_market_data(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        return {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.0004, 0.015, 252)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0002, 0.008, 252)}, index=dates),
            "commodities": pd.DataFrame({"GLD": np.random.normal(0.0001, 0.012, 252)}, index=dates),
        }

    def test_basic_allocation(self, strategy, sample_market_data):
        """测试基本配置计算"""
        allocation = strategy.compute_allocation(sample_market_data)

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_allocation_with_risk_profiles(self, strategy, sample_market_data):
        """测试不同风险偏好的配置"""
        for profile in ["conservative", "moderate", "aggressive"]:
            allocation = strategy.compute_allocation(
                sample_market_data, risk_profile=profile
            )
            assert len(allocation) == 3
            assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_allocation_with_strategic_weights(self, strategy, sample_market_data):
        """测试带战略配置的战术调整"""
        strategic = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        allocation = strategy.compute_allocation(
            sample_market_data,
            strategic_weights=strategic
        )

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_allocation_with_regime(self, strategy, sample_market_data):
        """测试不同市场状态的配置"""
        for regime in ["bull", "bear", "volatile", "normal"]:
            allocation = strategy.compute_allocation(
                sample_market_data,
                regime=regime
            )
            assert len(allocation) == 3
            assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_empty_market_data(self, strategy):
        """测试空市场数据"""
        allocation = strategy.compute_allocation({})

        # 应该使用默认资产类别
        assert len(allocation) > 0


class TestGetDefaultStrategicWeights:
    """默认战略配置测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_conservative_weights(self, strategy):
        """测试保守型权重"""
        weights = strategy._get_default_strategic_weights(
            "conservative", ["stocks", "bonds", "cash"]
        )

        assert "stocks" in weights
        assert weights["stocks"] <= 0.30

    def test_moderate_weights(self, strategy):
        """测试稳健型权重"""
        weights = strategy._get_default_strategic_weights(
            "moderate", ["stocks", "bonds"]
        )

        assert "stocks" in weights

    def test_aggressive_weights(self, strategy):
        """测试激进型权重"""
        weights = strategy._get_default_strategic_weights(
            "aggressive", ["stocks", "bonds", "cash"]
        )

        assert "stocks" in weights
        assert weights["stocks"] >= 0.50


class TestMomentumSignals:
    """动量信号测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_momentum_with_long_history(self, strategy):
        """测试长期历史数据的动量信号"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 300)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 300)}, index=dates),
        }

        signals = strategy._compute_momentum_signals(market_data, ["stocks", "bonds"])

        assert "stocks" in signals
        assert "bonds" in signals
        assert -1 <= signals["stocks"] <= 1
        assert -1 <= signals["bonds"] <= 1

    def test_momentum_with_short_history(self, strategy):
        """测试短期历史数据的动量信号"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
        }

        signals = strategy._compute_momentum_signals(market_data, ["stocks"])

        assert "stocks" in signals

    def test_momentum_missing_data(self, strategy):
        """测试缺失数据的动量信号"""
        signals = strategy._compute_momentum_signals({}, ["stocks", "bonds"])

        assert signals["stocks"] == 0
        assert signals["bonds"] == 0


class TestValueSignals:
    """估值信号测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_value_with_long_history(self, strategy):
        """测试长期历史数据的估值信号"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 300)}, index=dates),
        }

        signals = strategy._compute_value_signals(market_data, ["stocks"])

        assert "stocks" in signals
        assert -1 <= signals["stocks"] <= 1

    def test_value_missing_data(self, strategy):
        """测试缺失数据的估值信号"""
        signals = strategy._compute_value_signals({}, ["stocks"])

        assert signals["stocks"] == 0


class TestVolatilitySignals:
    """波动率信号测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_volatility_signals_basic(self, strategy):
        """测试基本波动率信号"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.03, 100)}, index=dates),  # 高波动
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),  # 低波动
        }

        signals = strategy._compute_volatility_signals(market_data, ["stocks", "bonds"])

        assert "stocks" in signals
        assert "bonds" in signals
        # 低波动资产应该有更正的信号
        assert signals["bonds"] > signals["stocks"]

    def test_volatility_missing_data(self, strategy):
        """测试缺失数据的波动率信号"""
        signals = strategy._compute_volatility_signals({}, ["stocks", "bonds"])

        # 应该返回0信号
        assert signals["stocks"] == 0
        assert signals["bonds"] == 0


class TestTacticalAdjustments:
    """战术调整计算测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_adjustments_basic(self, strategy):
        """测试基本调整计算"""
        momentum = {"stocks": 0.5, "bonds": -0.3}
        value = {"stocks": 0.2, "bonds": 0.4}
        volatility = {"stocks": -0.2, "bonds": 0.3}

        adjustments = strategy._compute_tactical_adjustments(
            ["stocks", "bonds"],
            momentum, value, volatility,
            None, "normal"
        )

        assert "stocks" in adjustments
        assert "bonds" in adjustments

        # 调整应该是零和的
        assert abs(sum(adjustments.values())) < 0.01

    def test_adjustments_with_expert_views(self, strategy):
        """测试带专家观点的调整"""
        momentum = {"stocks": 0.0, "bonds": 0.0}
        value = {"stocks": 0.0, "bonds": 0.0}
        volatility = {"stocks": 0.0, "bonds": 0.0}
        expert_views = {
            "stocks": {"sentiment": 0.8, "conviction": 0.9}
        }

        adjustments = strategy._compute_tactical_adjustments(
            ["stocks", "bonds"],
            momentum, value, volatility,
            expert_views, "normal"
        )

        assert "stocks" in adjustments
        # 专家观点应该影响调整
        assert adjustments["stocks"] > adjustments["bonds"]

    def test_adjustments_bull_market(self, strategy):
        """测试牛市调整"""
        momentum = {"stocks": 0.5, "bonds": 0.0}
        value = {"stocks": 0.0, "bonds": 0.0}
        volatility = {"stocks": 0.0, "bonds": 0.0}

        adjustments = strategy._compute_tactical_adjustments(
            ["stocks", "bonds"],
            momentum, value, volatility,
            None, "bull"
        )

        # 牛市中动量权重更高
        assert isinstance(adjustments["stocks"], float)

    def test_adjustments_bear_market(self, strategy):
        """测试熊市调整"""
        momentum = {"stocks": 0.0, "bonds": 0.0}
        value = {"stocks": 0.0, "bonds": 0.0}
        volatility = {"stocks": -0.5, "bonds": 0.5}

        adjustments = strategy._compute_tactical_adjustments(
            ["stocks", "bonds"],
            momentum, value, volatility,
            None, "bear"
        )

        # 熊市中波动率权重更高
        assert isinstance(adjustments["stocks"], float)


class TestGetSignalAnalysis:
    """信号分析测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_analysis_structure(self, strategy):
        """测试分析结构"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 300)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 300)}, index=dates),
        }

        analysis = strategy.get_signal_analysis(market_data)

        assert "stocks" in analysis
        assert "bonds" in analysis
        assert "momentum_signal" in analysis["stocks"]
        assert "value_signal" in analysis["stocks"]
        assert "volatility_signal" in analysis["stocks"]
        assert "composite_signal" in analysis["stocks"]


class TestComputeTrackingError:
    """跟踪误差计算测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_tracking_error_basic(self, strategy):
        """测试基本跟踪误差计算"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        tactical = {"stocks": 0.7, "bonds": 0.3}
        strategic = {"stocks": 0.5, "bonds": 0.5}

        te = strategy.compute_tracking_error(tactical, strategic, market_data)

        assert te >= 0
        assert isinstance(te, float)

    def test_tracking_error_identical(self, strategy):
        """测试相同权重的跟踪误差"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        weights = {"stocks": 0.5, "bonds": 0.5}

        te = strategy.compute_tracking_error(weights, weights, market_data)

        assert te < 0.001  # 应该接近0

    def test_tracking_error_large_deviation(self, strategy):
        """测试大偏离的跟踪误差"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        tactical = {"stocks": 0.9, "bonds": 0.1}
        strategic = {"stocks": 0.1, "bonds": 0.9}

        te = strategy.compute_tracking_error(tactical, strategic, market_data)

        assert te > 0  # 大偏离应该有较大的跟踪误差


class TestMaxDeviationConstraint:
    """最大偏离约束测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy(max_tactical_deviation=0.10)

    def test_deviation_within_bounds(self, strategy):
        """测试偏离在边界内"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.002, 0.03, 252)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 252)}, index=dates),
        }

        strategic = {"stocks": 0.5, "bonds": 0.5}
        allocation = strategy.compute_allocation(
            market_data,
            strategic_weights=strategic
        )

        # 检查偏离是否在边界内
        for ac in allocation:
            deviation = abs(allocation[ac] - strategic.get(ac, 0))
            assert deviation <= strategy.max_tactical_deviation + 0.01


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_single_asset(self, strategy):
        """测试单资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
        }

        allocation = strategy.compute_allocation(market_data)

        assert "stocks" in allocation

    def test_many_assets(self, strategy):
        """测试多资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
            "commodities": pd.DataFrame({"GLD": np.random.normal(0.0003, 0.015, 100)}, index=dates),
            "reits": pd.DataFrame({"VNQ": np.random.normal(0.0004, 0.018, 100)}, index=dates),
            "cash": pd.DataFrame({"BIL": np.random.normal(0.0001, 0.002, 100)}, index=dates),
        }

        allocation = strategy.compute_allocation(market_data)

        assert len(allocation) == 5
        assert abs(sum(allocation.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def strategy(self):
        return TacticalAllocationStrategy()

    def test_full_workflow(self, strategy):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.0004, 0.015, 300)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0002, 0.008, 300)}, index=dates),
            "commodities": pd.DataFrame({"GLD": np.random.normal(0.0001, 0.012, 300)}, index=dates),
        }

        expert_views = {
            "stocks": {"sentiment": 0.3, "conviction": 0.6},
        }

        strategic = {"stocks": 0.4, "bonds": 0.4, "commodities": 0.2}

        # 计算战术配置
        allocation = strategy.compute_allocation(
            market_data,
            expert_views=expert_views,
            risk_profile="moderate",
            strategic_weights=strategic,
            regime="normal"
        )

        assert len(allocation) == 3
        assert abs(sum(allocation.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in allocation.values())

        # 获取信号分析
        analysis = strategy.get_signal_analysis(market_data)
        assert "stocks" in analysis

        # 计算跟踪误差
        te = strategy.compute_tracking_error(allocation, strategic, market_data)
        assert te >= 0

    def test_reproducibility(self, strategy):
        """测试可重复性"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 252)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 252)}, index=dates),
        }

        alloc1 = strategy.compute_allocation(market_data)
        alloc2 = strategy.compute_allocation(market_data)

        for ac in alloc1:
            assert abs(alloc1[ac] - alloc2[ac]) < 0.01

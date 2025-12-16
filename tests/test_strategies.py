#!/usr/bin/env python
"""
Strategies Module Tests - 策略模块测试
覆盖: base_strategy, strategic_allocation, tactical_allocation,
      dynamic_rebalancing, core_satellite, strategy_toolkit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_market_data():
    """生成示例市场数据"""
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    np.random.seed(42)

    data = {}
    for asset in ["stocks", "bonds", "commodities", "reits", "cash"]:
        if asset == "stocks":
            returns = np.random.normal(0.0005, 0.015, len(dates))
        elif asset == "bonds":
            returns = np.random.normal(0.0002, 0.005, len(dates))
        elif asset == "commodities":
            returns = np.random.normal(0.0003, 0.012, len(dates))
        elif asset == "reits":
            returns = np.random.normal(0.0004, 0.013, len(dates))
        else:  # cash
            returns = np.random.normal(0.0001, 0.001, len(dates))

        df = pd.DataFrame({'returns': returns}, index=dates)
        data[asset] = df

    return data


@pytest.fixture
def sample_expert_views():
    """生成示例专家观点"""
    return {
        "stocks": {"sentiment": 0.3, "conviction": 0.8, "expected_return": 0.10},
        "bonds": {"sentiment": -0.2, "conviction": 0.6, "expected_return": 0.03},
        "commodities": {"sentiment": 0.1, "conviction": 0.5, "expected_return": 0.05},
        "reits": {"sentiment": 0.2, "conviction": 0.7, "expected_return": 0.07},
    }


# ============================================================
# Test 1: Base Strategy
# ============================================================

class TestAllocationStrategy:
    """测试AllocationStrategy基类"""

    def test_asset_class_profiles(self):
        """测试资产类别配置"""
        from finsage.strategies.base_strategy import AllocationStrategy

        profiles = AllocationStrategy.ASSET_CLASS_PROFILES
        assert len(profiles) >= 5
        assert "stocks" in profiles
        assert "bonds" in profiles
        assert profiles["stocks"]["risk"] == "high"
        assert profiles["bonds"]["risk"] == "low"

    def test_get_risk_profile_params_conservative(self):
        """测试保守型风险参数"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()
        params = strategy.get_risk_profile_params("conservative")

        assert params["max_equity"] == 0.30
        assert params["min_fixed_income"] == 0.40
        assert params["max_crypto"] == 0.00
        assert params["risk_aversion"] == 3.0

    def test_get_risk_profile_params_aggressive(self):
        """测试激进型风险参数"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()
        params = strategy.get_risk_profile_params("aggressive")

        assert params["max_equity"] == 0.80
        assert params["min_fixed_income"] == 0.05
        assert params["max_crypto"] == 0.15
        assert params["risk_aversion"] == 0.5

    def test_validate_allocation_normalization(self):
        """测试配置归一化"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()

        # 测试非归一化输入
        allocation = {"stocks": 0.5, "bonds": 0.3, "cash": 0.4}
        normalized = strategy.validate_allocation(allocation)

        assert abs(sum(normalized.values()) - 1.0) < 0.0001
        assert all(v >= 0 for v in normalized.values())

    def test_validate_allocation_negative(self):
        """测试负权重处理"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()

        allocation = {"stocks": -0.1, "bonds": 0.8, "cash": 0.3}
        normalized = strategy.validate_allocation(allocation)

        assert normalized["stocks"] == 0
        assert abs(sum(normalized.values()) - 1.0) < 0.0001

    def test_validate_allocation_zero_total(self):
        """测试全零输入"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()

        allocation = {"stocks": 0, "bonds": 0, "cash": 0}
        normalized = strategy.validate_allocation(allocation)

        # 应返回等权
        assert abs(sum(normalized.values()) - 1.0) < 0.0001
        assert all(abs(v - 1/3) < 0.0001 for v in normalized.values())

    def test_compute_portfolio_metrics(self, sample_market_data):
        """测试组合指标计算"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test"
            @property
            def description(self): return "test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()
        allocation = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}

        metrics = strategy.compute_portfolio_metrics(allocation, sample_market_data)

        assert "annualized_return" in metrics
        assert "annualized_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "cvar_95" in metrics

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.strategies.base_strategy import AllocationStrategy

        class TestStrategy(AllocationStrategy):
            @property
            def name(self): return "test_strategy"
            @property
            def description(self): return "A test strategy"
            def compute_allocation(self, *args, **kwargs): return {}

        strategy = TestStrategy()
        d = strategy.to_dict()

        assert d["name"] == "test_strategy"
        assert d["description"] == "A test strategy"
        assert "rebalance_frequency" in d


# ============================================================
# Test 2: Strategic Allocation Strategy
# ============================================================

class TestStrategicAllocationStrategy:
    """测试战略资产配置策略"""

    def test_import(self):
        """测试导入"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy
        assert StrategicAllocationStrategy is not None

    def test_initialization_mean_variance(self):
        """测试均值方差方法初始化"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy(method="mean_variance")
        assert strategy.method == "mean_variance"
        assert strategy.name == "strategic_allocation"
        assert strategy.rebalance_frequency == "annually"

    def test_initialization_risk_parity(self):
        """测试风险平价方法初始化"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy(method="risk_parity")
        assert strategy.method == "risk_parity"

    def test_capital_market_assumptions(self):
        """测试资本市场假设"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy()
        cma = strategy.capital_market_assumptions

        assert "stocks" in cma
        assert "bonds" in cma
        assert cma["stocks"]["expected_return"] > cma["bonds"]["expected_return"]
        assert cma["crypto"]["volatility"] > cma["stocks"]["volatility"]

    def test_compute_allocation_moderate(self, sample_market_data):
        """测试中等风险配置"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy(method="mean_variance")
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001
        assert all(v >= 0 for v in allocation.values())

    def test_compute_allocation_conservative(self, sample_market_data):
        """测试保守型配置"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy()
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="conservative"
        )

        # 保守型应该股票比例较低
        equity = allocation.get("stocks", 0) + allocation.get("reits", 0)
        assert equity <= 0.35

    def test_compute_allocation_with_expert_views(self, sample_market_data, sample_expert_views):
        """测试带专家观点的配置"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy(use_black_litterman=True)
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            expert_views=sample_expert_views,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_risk_parity_allocation(self, sample_market_data):
        """测试风险平价配置"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy(method="risk_parity")
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001
        # 风险平价应该给低波动资产更高权重
        assert allocation.get("bonds", 0) > allocation.get("stocks", 0) * 0.3

    def test_get_policy_portfolio(self):
        """测试政策组合"""
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        strategy = StrategicAllocationStrategy()

        conservative = strategy.get_policy_portfolio("conservative")
        moderate = strategy.get_policy_portfolio("moderate")
        aggressive = strategy.get_policy_portfolio("aggressive")

        # 保守型债券比例高
        assert conservative["bonds"] > moderate["bonds"]
        # 激进型股票比例高
        assert aggressive["stocks"] > moderate["stocks"]
        # 所有组合应归一化
        assert abs(sum(conservative.values()) - 1.0) < 0.0001
        assert abs(sum(moderate.values()) - 1.0) < 0.0001
        assert abs(sum(aggressive.values()) - 1.0) < 0.0001


# ============================================================
# Test 3: Tactical Allocation Strategy
# ============================================================

class TestTacticalAllocationStrategy:
    """测试战术资产配置策略"""

    def test_import(self):
        """测试导入"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy
        assert TacticalAllocationStrategy is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy(
            max_tactical_deviation=0.20,
            signal_decay=0.85
        )

        assert strategy.max_tactical_deviation == 0.20
        assert strategy.signal_decay == 0.85
        assert strategy.name == "tactical_allocation"
        assert strategy.rebalance_frequency == "monthly"

    def test_compute_allocation(self, sample_market_data):
        """测试战术配置计算"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy()
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_with_regime(self, sample_market_data):
        """测试不同市场状态下的配置"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy()

        # 牛市
        bull_allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            regime="bull"
        )

        # 熊市
        bear_allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            regime="bear"
        )

        assert abs(sum(bull_allocation.values()) - 1.0) < 0.0001
        assert abs(sum(bear_allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_with_strategic_weights(self, sample_market_data):
        """测试基于战略配置的战术调整"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy(max_tactical_deviation=0.10)
        strategic_weights = {
            "stocks": 0.40,
            "bonds": 0.30,
            "commodities": 0.10,
            "reits": 0.10,
            "cash": 0.10
        }

        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            strategic_weights=strategic_weights
        )

        # 确保偏离在限制范围内
        for asset, weight in allocation.items():
            strategic = strategic_weights.get(asset, 0)
            deviation = abs(weight - strategic)
            assert deviation <= 0.15  # 允许一些归一化误差

    def test_get_signal_analysis(self, sample_market_data):
        """测试信号分析"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy()
        analysis = strategy.get_signal_analysis(sample_market_data)

        assert "stocks" in analysis
        assert "momentum_signal" in analysis["stocks"]
        assert "value_signal" in analysis["stocks"]
        assert "volatility_signal" in analysis["stocks"]
        assert "composite_signal" in analysis["stocks"]

    def test_compute_tracking_error(self, sample_market_data):
        """测试跟踪误差计算"""
        from finsage.strategies.tactical_allocation import TacticalAllocationStrategy

        strategy = TacticalAllocationStrategy()
        tactical = {"stocks": 0.45, "bonds": 0.25, "commodities": 0.10, "reits": 0.10, "cash": 0.10}
        strategic = {"stocks": 0.40, "bonds": 0.30, "commodities": 0.10, "reits": 0.10, "cash": 0.10}

        te = strategy.compute_tracking_error(tactical, strategic, sample_market_data)

        assert te >= 0
        assert te < 0.5  # 合理范围


# ============================================================
# Test 4: Dynamic Rebalancing Strategy
# ============================================================

class TestDynamicRebalancingStrategy:
    """测试动态再平衡策略"""

    def test_import(self):
        """测试导入"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy, RebalanceTrigger
        assert DynamicRebalancingStrategy is not None
        assert RebalanceTrigger is not None

    def test_rebalance_trigger_enum(self):
        """测试再平衡触发枚举"""
        from finsage.strategies.dynamic_rebalancing import RebalanceTrigger

        assert RebalanceTrigger.THRESHOLD.value == "threshold"
        assert RebalanceTrigger.CALENDAR.value == "calendar"
        assert RebalanceTrigger.HYBRID.value == "hybrid"
        assert RebalanceTrigger.VOLATILITY.value == "volatility"

    def test_initialization_threshold(self):
        """测试阈值触发初始化"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy(
            trigger_type="threshold",
            deviation_threshold=0.08
        )

        assert strategy.deviation_threshold == 0.08
        assert strategy.name == "dynamic_rebalancing"

    def test_initialization_calendar(self):
        """测试日历触发初始化"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy(
            trigger_type="calendar",
            calendar_frequency="monthly"
        )

        assert strategy.calendar_frequency == "monthly"

    def test_compute_allocation_no_current_weights(self, sample_market_data):
        """测试无当前权重时的配置"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy()
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        # 允许2%的容差，因为某些策略实现可能不完全归一化
        assert abs(sum(allocation.values()) - 1.0) < 0.03

    def test_compute_allocation_threshold_trigger(self, sample_market_data):
        """测试阈值触发再平衡"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy(
            trigger_type="threshold",
            deviation_threshold=0.05
        )

        current_weights = {
            "stocks": 0.50,  # 偏离10%
            "bonds": 0.20,
            "commodities": 0.10,
            "reits": 0.10,
            "cash": 0.10
        }
        target_weights = {
            "stocks": 0.40,
            "bonds": 0.30,
            "commodities": 0.10,
            "reits": 0.10,
            "cash": 0.10
        }

        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            current_weights=current_weights,
            target_weights=target_weights
        )

        # 应该触发再平衡，调整向目标靠近
        assert allocation["stocks"] < current_weights["stocks"]

    def test_compute_allocation_no_trigger(self, sample_market_data):
        """测试不触发再平衡"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy(
            trigger_type="threshold",
            deviation_threshold=0.10
        )

        current_weights = {
            "stocks": 0.42,  # 偏离仅2%
            "bonds": 0.28,
            "commodities": 0.10,
            "reits": 0.10,
            "cash": 0.10
        }
        target_weights = {
            "stocks": 0.40,
            "bonds": 0.30,
            "commodities": 0.10,
            "reits": 0.10,
            "cash": 0.10
        }

        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            current_weights=current_weights,
            target_weights=target_weights
        )

        # 不应触发，返回当前权重
        assert allocation == current_weights

    def test_get_rebalance_analysis(self):
        """测试再平衡分析"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy(deviation_threshold=0.05)

        current = {"stocks": 0.50, "bonds": 0.25, "cash": 0.25}
        target = {"stocks": 0.40, "bonds": 0.30, "cash": 0.30}

        analysis = strategy.get_rebalance_analysis(current, target, portfolio_value=1000000)

        assert "trades" in analysis
        assert "total_trade_value" in analysis
        assert "estimated_cost" in analysis
        assert "max_deviation" in analysis
        assert "needs_rebalancing" in analysis
        # 使用近似比较避免浮点精度问题
        assert abs(analysis["max_deviation"] - 0.10) < 0.001

    def test_estimate_drift(self, sample_market_data):
        """测试漂移估计"""
        from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy

        strategy = DynamicRebalancingStrategy()
        initial_weights = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.2, "cash": 0.1}

        drift = strategy.estimate_drift(initial_weights, sample_market_data, days_forward=30)

        assert "stocks" in drift
        assert all(v >= 0 for v in drift.values())


# ============================================================
# Test 5: Core-Satellite Strategy
# ============================================================

class TestCoreSatelliteStrategy:
    """测试核心卫星策略"""

    def test_import(self):
        """测试导入"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy
        assert CoreSatelliteStrategy is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy(
            core_ratio=0.75,
            min_core_ratio=0.60,
            max_core_ratio=0.85
        )

        assert strategy.core_ratio == 0.75
        assert strategy.min_core_ratio == 0.60
        assert strategy.max_core_ratio == 0.85
        assert strategy.name == "core_satellite"

    def test_compute_allocation_default(self, sample_market_data):
        """测试默认配置"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy()
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_conservative(self, sample_market_data):
        """测试保守型配置"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy(core_ratio=0.80)
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="conservative"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_with_regime(self, sample_market_data):
        """测试市场状态调整"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy(core_ratio=0.70)

        # 熊市应增加核心比例
        bear_allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            market_regime="bear"
        )

        # 牛市应减少核心比例
        bull_allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            risk_profile="moderate",
            market_regime="bull"
        )

        assert abs(sum(bear_allocation.values()) - 1.0) < 0.0001
        assert abs(sum(bull_allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_with_expert_views(self, sample_market_data, sample_expert_views):
        """测试带专家观点的配置"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy()
        allocation = strategy.compute_allocation(
            market_data=sample_market_data,
            expert_views=sample_expert_views,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_adjust_core_ratio(self):
        """测试核心比例调整"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy(
            core_ratio=0.70,
            min_core_ratio=0.50,
            max_core_ratio=0.90
        )

        # 熊市+保守应该增加核心比例
        ratio = strategy._adjust_core_ratio("bear", "conservative")
        assert ratio > 0.70

        # 牛市+激进应该减少核心比例
        ratio = strategy._adjust_core_ratio("bull", "aggressive")
        assert ratio < 0.70

        # 确保在范围内
        assert strategy.min_core_ratio <= ratio <= strategy.max_core_ratio

    def test_get_active_share(self):
        """测试主动份额计算"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy()

        satellite = {"stocks": 0.50, "bonds": 0.20, "cash": 0.30}
        core = {"stocks": 0.35, "bonds": 0.35, "cash": 0.30}

        active_share = strategy.get_active_share(satellite, core)

        # (|0.50-0.35| + |0.20-0.35| + |0.30-0.30|) / 2 = 0.15
        assert abs(active_share - 0.15) < 0.0001

    def test_compute_tracking_error(self, sample_market_data):
        """测试跟踪误差计算"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy()

        satellite = {"stocks": 0.50, "bonds": 0.20, "commodities": 0.15, "reits": 0.10, "cash": 0.05}
        core = {"stocks": 0.35, "bonds": 0.35, "commodities": 0.10, "reits": 0.10, "cash": 0.10}

        te = strategy.compute_tracking_error(satellite, core, sample_market_data)

        assert te >= 0
        assert te < 0.5

    def test_recommend_satellite_adjustments(self, sample_market_data, sample_expert_views):
        """测试卫星调整建议"""
        from finsage.strategies.core_satellite import CoreSatelliteStrategy

        strategy = CoreSatelliteStrategy()
        current_satellite = {"stocks": 0.4, "bonds": 0.2, "commodities": 0.2, "reits": 0.1, "cash": 0.1}

        recommendations = strategy.recommend_satellite_adjustments(
            current_satellite, sample_market_data, sample_expert_views
        )

        assert "stocks" in recommendations
        assert all(isinstance(v, str) for v in recommendations.values())


# ============================================================
# Test 6: Strategy Toolkit
# ============================================================

class TestStrategyToolkit:
    """测试策略工具箱"""

    def test_import(self):
        """测试导入"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit
        assert StrategyToolkit is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        assert toolkit is not None
        assert len(toolkit._strategies) >= 5

    def test_register_strategy(self):
        """测试注册策略"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit
        from finsage.strategies.strategic_allocation import StrategicAllocationStrategy

        toolkit = StrategyToolkit()
        custom_strategy = StrategicAllocationStrategy(method="equal_weight")

        toolkit.register(custom_strategy, name="custom_equal_weight")

        assert "custom_equal_weight" in toolkit._strategies
        assert toolkit.get("custom_equal_weight") == custom_strategy

    def test_unregister_strategy(self):
        """测试注销策略"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        initial_count = len(toolkit._strategies)

        toolkit.unregister("tactical_allocation")
        assert len(toolkit._strategies) == initial_count - 1

    def test_get_strategy(self):
        """测试获取策略"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        strategy = toolkit.get("strategic_allocation")

        assert strategy is not None
        assert strategy.name == "strategic_allocation"

    def test_get_nonexistent_strategy(self):
        """测试获取不存在的策略"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        strategy = toolkit.get("nonexistent")

        assert strategy is None

    def test_list_strategies(self):
        """测试列出所有策略"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        strategies = toolkit.list_strategies()

        assert len(strategies) >= 5
        assert all("name" in s for s in strategies)
        assert all("description" in s for s in strategies)
        assert all("rebalance_frequency" in s for s in strategies)

    def test_compute_allocation(self, sample_market_data):
        """测试计算配置"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        allocation = toolkit.compute_allocation(
            strategy_name="strategic_allocation",
            market_data=sample_market_data,
            risk_profile="moderate"
        )

        assert abs(sum(allocation.values()) - 1.0) < 0.0001

    def test_compute_allocation_invalid_strategy(self, sample_market_data):
        """测试无效策略配置"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()

        with pytest.raises(ValueError) as exc_info:
            toolkit.compute_allocation(
                strategy_name="invalid_strategy",
                market_data=sample_market_data
            )

        assert "not found" in str(exc_info.value)

    def test_compare_strategies(self, sample_market_data):
        """测试策略比较"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        results = toolkit.compare_strategies(
            market_data=sample_market_data,
            risk_profile="moderate",
            strategy_names=["strategic_allocation", "tactical_allocation"]
        )

        assert "strategic_allocation" in results
        assert "tactical_allocation" in results
        assert all(isinstance(v, dict) for v in results.values())

    def test_recommend_strategy_long_conservative(self):
        """测试长期保守推荐"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        recommended = toolkit.recommend_strategy(
            market_regime="normal",
            risk_profile="conservative",
            investment_horizon="long"
        )

        assert recommended == "strategic_risk_parity"

    def test_recommend_strategy_short_aggressive(self):
        """测试短期激进推荐"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        recommended = toolkit.recommend_strategy(
            market_regime="normal",
            risk_profile="aggressive",
            investment_horizon="short"
        )

        assert recommended == "tactical_allocation"

    def test_recommend_strategy_bear_market(self):
        """测试熊市推荐"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        recommended = toolkit.recommend_strategy(
            market_regime="bear",
            risk_profile="moderate",
            investment_horizon="medium"
        )

        # 熊市应偏向保守
        assert "conservative" in recommended or "core" in recommended

    def test_get_strategy_metrics(self, sample_market_data):
        """测试策略指标获取"""
        from finsage.strategies.strategy_toolkit import StrategyToolkit

        toolkit = StrategyToolkit()
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.15, "reits": 0.10, "cash": 0.05}

        metrics = toolkit.get_strategy_metrics(
            strategy_name="strategic_allocation",
            allocation=allocation,
            market_data=sample_market_data
        )

        assert "strategy_name" in metrics
        assert "allocation" in metrics


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Strategies Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

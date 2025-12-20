"""
Deep tests for StrategyToolkit
策略工具箱深度测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock

from finsage.strategies.strategy_toolkit import StrategyToolkit
from finsage.strategies.base_strategy import AllocationStrategy


class MockStrategy(AllocationStrategy):
    """测试用模拟策略"""

    def __init__(self, name="mock_strategy"):
        self._name = name
        self._description = "Mock strategy for testing"
        self._rebalance_frequency = "monthly"

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def rebalance_frequency(self):
        return self._rebalance_frequency

    def compute_allocation(self, market_data, expert_views=None,
                          risk_profile="moderate", constraints=None, **kwargs):
        return {"stocks": 0.6, "bonds": 0.3, "commodities": 0.1}

    def validate_allocation(self, allocation):
        return allocation

    def compute_portfolio_metrics(self, allocation, market_data):
        return {"sharpe": 1.5, "volatility": 0.12}


class TestStrategyToolkitInit:
    """StrategyToolkit初始化测试"""

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_default_init(self, mock_register):
        """测试默认初始化"""
        toolkit = StrategyToolkit()
        mock_register.assert_called_once()

    def test_init_registers_strategies(self):
        """测试初始化注册策略"""
        toolkit = StrategyToolkit()
        strategies = toolkit.list_strategies()
        assert len(strategies) > 0

    def test_default_strategies_registered(self):
        """测试默认策略注册"""
        toolkit = StrategyToolkit()

        # 检查关键策略是否注册
        assert toolkit.get("strategic_allocation") is not None or \
               toolkit.get("tactical_allocation") is not None


class TestStrategyRegistration:
    """策略注册测试"""

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_register_strategy(self, mock_register):
        """测试注册策略"""
        toolkit = StrategyToolkit()
        mock_strategy = MockStrategy("test_strategy")

        toolkit.register(mock_strategy)

        assert toolkit.get("test_strategy") is not None
        assert toolkit.get("test_strategy") == mock_strategy

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_register_with_custom_name(self, mock_register):
        """测试使用自定义名称注册"""
        toolkit = StrategyToolkit()
        mock_strategy = MockStrategy("original_name")

        toolkit.register(mock_strategy, name="custom_name")

        assert toolkit.get("custom_name") is not None
        assert toolkit.get("original_name") is None

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_unregister_strategy(self, mock_register):
        """测试注销策略"""
        toolkit = StrategyToolkit()
        mock_strategy = MockStrategy("to_remove")

        toolkit.register(mock_strategy)
        assert toolkit.get("to_remove") is not None

        toolkit.unregister("to_remove")
        assert toolkit.get("to_remove") is None

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_unregister_nonexistent(self, mock_register):
        """测试注销不存在的策略"""
        toolkit = StrategyToolkit()
        # 不应该抛出异常
        toolkit.unregister("nonexistent")


class TestListStrategies:
    """列出策略测试"""

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_list_strategies(self, mock_register):
        """测试列出策略"""
        toolkit = StrategyToolkit()
        toolkit.register(MockStrategy("strategy1"))
        toolkit.register(MockStrategy("strategy2"))

        strategies = toolkit.list_strategies()

        assert len(strategies) == 2
        assert strategies[0]["name"] == "strategy1"
        assert "description" in strategies[0]
        assert "rebalance_frequency" in strategies[0]

    @patch.object(StrategyToolkit, '_register_default_strategies')
    def test_list_strategies_empty(self, mock_register):
        """测试空策略列表"""
        toolkit = StrategyToolkit()
        strategies = toolkit.list_strategies()
        assert strategies == []


class TestComputeAllocation:
    """计算配置测试"""

    @pytest.fixture
    def toolkit(self):
        with patch.object(StrategyToolkit, '_register_default_strategies'):
            toolkit = StrategyToolkit()
            toolkit.register(MockStrategy("test_strategy"))
            return toolkit

    @pytest.fixture
    def market_data(self):
        return {
            "stocks": pd.DataFrame({
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000000] * 5
            }),
            "bonds": pd.DataFrame({
                "Close": [50, 50.5, 51, 51.5, 52],
                "Volume": [500000] * 5
            }),
        }

    def test_compute_allocation_basic(self, toolkit, market_data):
        """测试基本配置计算"""
        allocation = toolkit.compute_allocation(
            strategy_name="test_strategy",
            market_data=market_data,
        )

        assert "stocks" in allocation
        assert "bonds" in allocation
        assert allocation["stocks"] == 0.6

    def test_compute_allocation_with_views(self, toolkit, market_data):
        """测试带专家观点的配置计算"""
        expert_views = {
            "stock_expert": {"stocks": 0.7},
            "bond_expert": {"bonds": 0.3},
        }

        allocation = toolkit.compute_allocation(
            strategy_name="test_strategy",
            market_data=market_data,
            expert_views=expert_views,
        )

        assert allocation is not None

    def test_compute_allocation_with_constraints(self, toolkit, market_data):
        """测试带约束的配置计算"""
        constraints = {
            "max_weight": 0.4,
            "min_cash": 0.1,
        }

        allocation = toolkit.compute_allocation(
            strategy_name="test_strategy",
            market_data=market_data,
            constraints=constraints,
        )

        assert allocation is not None

    def test_compute_allocation_invalid_strategy(self, toolkit, market_data):
        """测试无效策略名称"""
        with pytest.raises(ValueError) as excinfo:
            toolkit.compute_allocation(
                strategy_name="nonexistent_strategy",
                market_data=market_data,
            )

        assert "not found" in str(excinfo.value)

    def test_compute_allocation_risk_profiles(self, toolkit, market_data):
        """测试不同风险偏好"""
        for risk_profile in ["conservative", "moderate", "aggressive"]:
            allocation = toolkit.compute_allocation(
                strategy_name="test_strategy",
                market_data=market_data,
                risk_profile=risk_profile,
            )
            assert allocation is not None


class TestCompareStrategies:
    """比较策略测试"""

    @pytest.fixture
    def toolkit(self):
        with patch.object(StrategyToolkit, '_register_default_strategies'):
            toolkit = StrategyToolkit()
            toolkit.register(MockStrategy("strategy1"))
            toolkit.register(MockStrategy("strategy2"))
            return toolkit

    @pytest.fixture
    def market_data(self):
        return {
            "stocks": pd.DataFrame({"Close": [100, 101, 102]}),
        }

    def test_compare_all_strategies(self, toolkit, market_data):
        """测试比较所有策略"""
        results = toolkit.compare_strategies(market_data=market_data)

        assert "strategy1" in results
        assert "strategy2" in results
        assert results["strategy1"]["stocks"] == 0.6

    def test_compare_specific_strategies(self, toolkit, market_data):
        """测试比较特定策略"""
        results = toolkit.compare_strategies(
            market_data=market_data,
            strategy_names=["strategy1"],
        )

        assert "strategy1" in results
        assert "strategy2" not in results

    def test_compare_with_failing_strategy(self, toolkit, market_data):
        """测试比较时策略失败"""
        # 创建一个会失败的策略
        failing_strategy = MagicMock()
        failing_strategy.name = "failing_strategy"
        failing_strategy.compute_allocation.side_effect = Exception("Test error")
        toolkit.register(failing_strategy, name="failing_strategy")

        results = toolkit.compare_strategies(
            market_data=market_data,
            strategy_names=["strategy1", "failing_strategy"],
        )

        assert results["strategy1"]["stocks"] == 0.6
        assert results["failing_strategy"] == {}  # 失败应该返回空


class TestRecommendStrategy:
    """推荐策略测试"""

    @pytest.fixture
    def toolkit(self):
        return StrategyToolkit()

    def test_recommend_long_conservative(self, toolkit):
        """测试长期保守推荐"""
        recommendation = toolkit.recommend_strategy(
            market_regime="normal",
            risk_profile="conservative",
            investment_horizon="long",
        )
        assert recommendation is not None

    def test_recommend_short_aggressive(self, toolkit):
        """测试短期激进推荐"""
        recommendation = toolkit.recommend_strategy(
            market_regime="normal",
            risk_profile="aggressive",
            investment_horizon="short",
        )
        assert recommendation is not None

    def test_recommend_bear_market_adjustment(self, toolkit):
        """测试熊市调整"""
        normal_rec = toolkit.recommend_strategy(
            market_regime="normal",
            risk_profile="aggressive",
            investment_horizon="medium",
        )

        bear_rec = toolkit.recommend_strategy(
            market_regime="bear",
            risk_profile="aggressive",
            investment_horizon="medium",
        )

        # 熊市应该更保守
        # 检查是否有调整（具体断言取决于实现）
        assert bear_rec is not None

    def test_recommend_volatile_market(self, toolkit):
        """测试震荡市推荐"""
        recommendation = toolkit.recommend_strategy(
            market_regime="volatile",
            risk_profile="moderate",
            investment_horizon="medium",
        )

        # 震荡市应该偏向动态再平衡
        assert recommendation is not None

    def test_recommend_all_combinations(self, toolkit):
        """测试所有组合"""
        regimes = ["bull", "bear", "volatile", "normal"]
        profiles = ["conservative", "moderate", "aggressive"]
        horizons = ["short", "medium", "long"]

        for regime in regimes:
            for profile in profiles:
                for horizon in horizons:
                    rec = toolkit.recommend_strategy(
                        market_regime=regime,
                        risk_profile=profile,
                        investment_horizon=horizon,
                    )
                    assert rec is not None


class TestGetStrategyMetrics:
    """获取策略指标测试"""

    @pytest.fixture
    def toolkit(self):
        with patch.object(StrategyToolkit, '_register_default_strategies'):
            toolkit = StrategyToolkit()
            toolkit.register(MockStrategy("test_strategy"))
            return toolkit

    @pytest.fixture
    def market_data(self):
        return {
            "stocks": pd.DataFrame({"Close": [100, 101, 102]}),
        }

    def test_get_metrics(self, toolkit, market_data):
        """测试获取指标"""
        allocation = {"stocks": 0.6, "bonds": 0.4}

        metrics = toolkit.get_strategy_metrics(
            strategy_name="test_strategy",
            allocation=allocation,
            market_data=market_data,
        )

        assert metrics["strategy_name"] == "test_strategy"
        assert metrics["allocation"] == allocation
        assert "sharpe" in metrics

    def test_get_metrics_nonexistent_strategy(self, toolkit, market_data):
        """测试不存在策略的指标"""
        allocation = {"stocks": 0.6}

        metrics = toolkit.get_strategy_metrics(
            strategy_name="nonexistent",
            allocation=allocation,
            market_data=market_data,
        )

        assert metrics == {}


class TestDefaultStrategiesIntegration:
    """默认策略集成测试"""

    @pytest.fixture
    def toolkit(self):
        return StrategyToolkit()

    def test_strategic_allocation_registered(self, toolkit):
        """测试战略配置策略注册"""
        # 检查是否有战略配置类策略
        strategies = toolkit.list_strategies()
        names = [s["name"] for s in strategies]
        assert any("strategic" in n.lower() for n in names)

    def test_tactical_allocation_registered(self, toolkit):
        """测试战术配置策略注册"""
        strategies = toolkit.list_strategies()
        names = [s["name"] for s in strategies]
        assert any("tactical" in n.lower() for n in names)

    def test_dynamic_rebalancing_registered(self, toolkit):
        """测试动态再平衡策略注册"""
        strategies = toolkit.list_strategies()
        names = [s["name"] for s in strategies]
        assert any("dynamic" in n.lower() for n in names)

    def test_core_satellite_registered(self, toolkit):
        """测试核心卫星策略注册"""
        strategies = toolkit.list_strategies()
        names = [s["name"] for s in strategies]
        assert any("core" in n.lower() for n in names)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def toolkit(self):
        with patch.object(StrategyToolkit, '_register_default_strategies'):
            toolkit = StrategyToolkit()
            toolkit.register(MockStrategy("test_strategy"))
            return toolkit

    def test_empty_market_data(self, toolkit):
        """测试空市场数据"""
        allocation = toolkit.compute_allocation(
            strategy_name="test_strategy",
            market_data={},
        )
        assert allocation is not None

    def test_none_expert_views(self, toolkit):
        """测试None专家观点"""
        market_data = {"stocks": pd.DataFrame({"Close": [100]})}

        allocation = toolkit.compute_allocation(
            strategy_name="test_strategy",
            market_data=market_data,
            expert_views=None,
        )
        assert allocation is not None

    def test_strategy_exception_handling(self, toolkit):
        """测试策略异常处理"""
        failing_strategy = MagicMock()
        failing_strategy.name = "failing"
        failing_strategy.compute_allocation.side_effect = ValueError("Test error")
        toolkit.register(failing_strategy, name="failing")

        with pytest.raises(ValueError):
            toolkit.compute_allocation(
                strategy_name="failing",
                market_data={},
            )

    def test_compare_with_empty_strategy_list(self, toolkit):
        """测试空策略列表比较"""
        market_data = {"stocks": pd.DataFrame({"Close": [100]})}

        results = toolkit.compare_strategies(
            market_data=market_data,
            strategy_names=[],
        )

        assert results == {}


class TestStrategyMetadata:
    """策略元数据测试"""

    @pytest.fixture
    def toolkit(self):
        return StrategyToolkit()

    def test_strategy_has_description(self, toolkit):
        """测试策略有描述"""
        strategies = toolkit.list_strategies()
        for strategy in strategies:
            assert "description" in strategy
            # 描述不应为空
            assert strategy["description"] is not None

    def test_strategy_has_rebalance_frequency(self, toolkit):
        """测试策略有再平衡频率"""
        strategies = toolkit.list_strategies()
        for strategy in strategies:
            assert "rebalance_frequency" in strategy

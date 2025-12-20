"""
Deep tests for Base Allocation Strategy
资产配置策略基类深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.strategies.base_strategy import AllocationStrategy


class ConcreteStrategy(AllocationStrategy):
    """用于测试的具体策略实现"""

    @property
    def name(self) -> str:
        return "test_strategy"

    @property
    def description(self) -> str:
        return "Test strategy for unit tests"

    def compute_allocation(self, market_data, expert_views=None, risk_profile="moderate", constraints=None, **kwargs):
        n = len(market_data)
        if n == 0:
            return {}
        return {ac: 1.0/n for ac in market_data.keys()}


class TestAllocationStrategyInit:
    """AllocationStrategy初始化测试"""

    def test_asset_class_profiles_defined(self):
        """测试资产类别配置文件已定义"""
        assert "stocks" in AllocationStrategy.ASSET_CLASS_PROFILES
        assert "bonds" in AllocationStrategy.ASSET_CLASS_PROFILES
        assert "commodities" in AllocationStrategy.ASSET_CLASS_PROFILES
        assert "reits" in AllocationStrategy.ASSET_CLASS_PROFILES
        assert "crypto" in AllocationStrategy.ASSET_CLASS_PROFILES
        assert "cash" in AllocationStrategy.ASSET_CLASS_PROFILES

    def test_asset_class_profiles_structure(self):
        """测试资产类别配置文件结构"""
        for ac, profile in AllocationStrategy.ASSET_CLASS_PROFILES.items():
            assert "risk" in profile
            assert "return" in profile
            assert "liquidity" in profile


class TestAbstractMethods:
    """抽象方法测试"""

    def test_concrete_strategy_name(self):
        """测试具体策略名称"""
        strategy = ConcreteStrategy()
        assert strategy.name == "test_strategy"

    def test_concrete_strategy_description(self):
        """测试具体策略描述"""
        strategy = ConcreteStrategy()
        assert "Test strategy" in strategy.description

    def test_default_rebalance_frequency(self):
        """测试默认再平衡频率"""
        strategy = ConcreteStrategy()
        assert strategy.rebalance_frequency == "monthly"


class TestGetRiskProfileParams:
    """风险偏好参数测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    def test_conservative_profile(self, strategy):
        """测试保守型配置"""
        params = strategy.get_risk_profile_params("conservative")

        assert params["max_equity"] == 0.30
        assert params["min_fixed_income"] == 0.40
        assert params["max_crypto"] == 0.00
        assert params["risk_aversion"] == 3.0

    def test_moderate_profile(self, strategy):
        """测试稳健型配置"""
        params = strategy.get_risk_profile_params("moderate")

        assert params["max_equity"] == 0.60
        assert params["min_fixed_income"] == 0.20
        assert params["max_crypto"] == 0.05
        assert params["risk_aversion"] == 1.5

    def test_aggressive_profile(self, strategy):
        """测试激进型配置"""
        params = strategy.get_risk_profile_params("aggressive")

        assert params["max_equity"] == 0.80
        assert params["min_fixed_income"] == 0.05
        assert params["max_crypto"] == 0.15
        assert params["risk_aversion"] == 0.5

    def test_unknown_profile_defaults_to_moderate(self, strategy):
        """测试未知配置文件默认为稳健型"""
        params = strategy.get_risk_profile_params("unknown")

        assert params["max_equity"] == 0.60


class TestValidateAllocation:
    """验证配置测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    def test_validates_normal_allocation(self, strategy):
        """测试正常配置验证"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "cash": 0.3}
        validated = strategy.validate_allocation(allocation)

        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_normalizes_unnormalized_allocation(self, strategy):
        """测试归一化未归一的配置"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "cash": 0.1}  # 总和0.8
        validated = strategy.validate_allocation(allocation)

        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_handles_negative_weights(self, strategy):
        """测试处理负权重"""
        allocation = {"stocks": 0.5, "bonds": -0.1, "cash": 0.6}
        validated = strategy.validate_allocation(allocation)

        assert all(v >= 0 for v in validated.values())
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_handles_all_zero_weights(self, strategy):
        """测试处理全零权重"""
        allocation = {"stocks": 0, "bonds": 0, "cash": 0}
        validated = strategy.validate_allocation(allocation)

        # 应该返回等权
        assert all(abs(v - 1/3) < 0.001 for v in validated.values())

    def test_handles_empty_allocation(self, strategy):
        """测试处理空配置"""
        allocation = {}
        validated = strategy.validate_allocation(allocation)

        assert validated == {}


class TestComputePortfolioMetrics:
    """组合指标计算测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        return {
            "stocks": pd.DataFrame({
                "SPY": np.random.normal(0.0004, 0.015, 252),
                "QQQ": np.random.normal(0.0005, 0.02, 252),
            }, index=dates),
            "bonds": pd.DataFrame({
                "TLT": np.random.normal(0.0002, 0.008, 252),
            }, index=dates),
        }

    def test_compute_basic_metrics(self, strategy, sample_returns):
        """测试基本指标计算"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        metrics = strategy.compute_portfolio_metrics(allocation, sample_returns)

        assert "annualized_return" in metrics
        assert "annualized_volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "cvar_95" in metrics

    def test_metrics_with_zero_weight_assets(self, strategy, sample_returns):
        """测试零权重资产的指标"""
        allocation = {"stocks": 1.0, "bonds": 0.0}
        metrics = strategy.compute_portfolio_metrics(allocation, sample_returns)

        assert "annualized_return" in metrics

    def test_metrics_with_missing_data(self, strategy):
        """测试缺失数据时的指标"""
        returns_data = {}
        allocation = {"stocks": 0.6, "bonds": 0.4}
        metrics = strategy.compute_portfolio_metrics(allocation, returns_data)

        assert metrics == {}

    def test_max_drawdown_negative(self, strategy, sample_returns):
        """测试最大回撤为负数"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        metrics = strategy.compute_portfolio_metrics(allocation, sample_returns)

        assert metrics["max_drawdown"] <= 0


class TestToDict:
    """转换为字典测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    def test_to_dict_structure(self, strategy):
        """测试字典结构"""
        result = strategy.to_dict()

        assert "name" in result
        assert "description" in result
        assert "rebalance_frequency" in result

    def test_to_dict_values(self, strategy):
        """测试字典值"""
        result = strategy.to_dict()

        assert result["name"] == "test_strategy"
        assert "Test strategy" in result["description"]
        assert result["rebalance_frequency"] == "monthly"


class TestComputeAllocation:
    """配置计算测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    def test_basic_allocation(self, strategy):
        """测试基本配置计算"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        allocation = strategy.compute_allocation(market_data)

        assert len(allocation) == 2
        assert abs(sum(allocation.values()) - 1.0) < 0.01

    def test_empty_market_data(self, strategy):
        """测试空市场数据"""
        allocation = strategy.compute_allocation({})

        assert allocation == {}


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

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

        # 计算配置
        allocation = strategy.compute_allocation(market_data, risk_profile="moderate")

        # 验证配置
        validated = strategy.validate_allocation(allocation)

        # 计算指标
        metrics = strategy.compute_portfolio_metrics(validated, market_data)

        # 转换为字典
        info = strategy.to_dict()

        assert len(validated) == 3
        assert abs(sum(validated.values()) - 1.0) < 0.01
        assert "annualized_return" in metrics
        assert "name" in info

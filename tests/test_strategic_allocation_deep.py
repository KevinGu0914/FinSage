"""
Deep tests for Strategic Asset Allocation
战略资产配置策略深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.strategies.strategic_allocation import StrategicAllocationStrategy


class TestStrategicAllocationInit:
    """StrategicAllocationStrategy初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        strategy = StrategicAllocationStrategy()
        assert strategy.method == "mean_variance"
        assert strategy.use_black_litterman == False

    def test_custom_init(self):
        """测试自定义初始化"""
        strategy = StrategicAllocationStrategy(
            method="risk_parity",
            use_black_litterman=True
        )
        assert strategy.method == "risk_parity"
        assert strategy.use_black_litterman == True

    def test_name_property(self):
        """测试名称属性"""
        strategy = StrategicAllocationStrategy()
        assert strategy.name == "strategic_allocation"

    def test_description_property(self):
        """测试描述属性"""
        strategy = StrategicAllocationStrategy()
        assert "战略" in strategy.description
        assert "Strategic" in strategy.description

    def test_rebalance_frequency(self):
        """测试再平衡频率"""
        strategy = StrategicAllocationStrategy()
        assert strategy.rebalance_frequency == "annually"

    def test_capital_market_assumptions(self):
        """测试资本市场假设"""
        strategy = StrategicAllocationStrategy()
        assert "stocks" in strategy.capital_market_assumptions
        assert "bonds" in strategy.capital_market_assumptions
        assert "expected_return" in strategy.capital_market_assumptions["stocks"]
        assert "volatility" in strategy.capital_market_assumptions["stocks"]


class TestComputeAllocation:
    """配置计算测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy()

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

    def test_empty_market_data(self, strategy):
        """测试空市场数据"""
        allocation = strategy.compute_allocation({})

        # 应该使用默认资产类别
        assert len(allocation) > 0


class TestMeanVarianceOptimization:
    """均值方差优化测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy(method="mean_variance")

    def test_mean_variance_basic(self, strategy):
        """测试基本均值方差优化"""
        mu = np.array([0.08, 0.03, 0.05])
        sigma = np.array([
            [0.0324, 0.0018, 0.0054],
            [0.0018, 0.0025, -0.0005],
            [0.0054, -0.0005, 0.0225]
        ])
        risk_params = {"risk_aversion": 1.5}

        weights = strategy._mean_variance_optimization(
            mu, sigma, risk_params, None, ["stocks", "bonds", "commodities"]
        )

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_mean_variance_with_constraints(self, strategy):
        """测试带约束的均值方差优化"""
        mu = np.array([0.08, 0.03])
        sigma = np.array([[0.0324, 0.0018], [0.0018, 0.0025]])
        risk_params = {"risk_aversion": 1.5}
        constraints = {"min_weight": 0.2, "max_weight": 0.6}

        weights = strategy._mean_variance_optimization(
            mu, sigma, risk_params, constraints, ["stocks", "bonds"]
        )

        for w in weights.values():
            assert w >= 0.19
            assert w <= 0.61


class TestRiskParityOptimization:
    """风险平价优化测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy(method="risk_parity")

    def test_risk_parity_basic(self, strategy):
        """测试基本风险平价优化"""
        sigma = np.array([
            [0.0324, 0.0018, 0.0054],
            [0.0018, 0.0025, -0.0005],
            [0.0054, -0.0005, 0.0225]
        ])

        weights = strategy._risk_parity_optimization(
            sigma, ["stocks", "bonds", "commodities"]
        )

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_risk_parity_favors_low_vol(self, strategy):
        """测试风险平价偏向低波动资产"""
        # 对角协方差矩阵（无相关性）
        sigma = np.array([
            [0.0324, 0.0, 0.0],   # 18% vol
            [0.0, 0.0025, 0.0],   # 5% vol
            [0.0, 0.0, 0.0225]    # 15% vol
        ])

        weights = strategy._risk_parity_optimization(
            sigma, ["high_vol", "low_vol", "med_vol"]
        )

        # 低波动资产应该有更高权重
        assert weights["low_vol"] > weights["high_vol"]


class TestEqualWeightMethod:
    """等权方法测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy(method="equal_weight")

    def test_equal_weight_allocation(self, strategy):
        """测试等权配置"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
            "commodities": pd.DataFrame({"GLD": np.random.normal(0.0003, 0.015, 100)}, index=dates),
        }

        allocation = strategy.compute_allocation(market_data)

        # 等权应该接近1/3
        for w in allocation.values():
            assert abs(w - 1/3) < 0.15


class TestBlackLitterman:
    """Black-Litterman测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy(use_black_litterman=True)

    def test_apply_black_litterman(self, strategy):
        """测试应用Black-Litterman"""
        mu = np.array([0.08, 0.03, 0.05])
        sigma = np.array([
            [0.0324, 0.0018, 0.0054],
            [0.0018, 0.0025, -0.0005],
            [0.0054, -0.0005, 0.0225]
        ])
        expert_views = {
            "stocks": {"expected_return": 0.12, "confidence": 0.7}
        }

        adjusted_mu = strategy._apply_black_litterman(
            mu, sigma, expert_views, ["stocks", "bonds", "commodities"]
        )

        # 调整后的股票预期收益应该向专家观点移动
        assert adjusted_mu[0] > mu[0]


class TestApplyRiskConstraints:
    """风险约束应用测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy()

    def test_max_equity_constraint(self, strategy):
        """测试最大权益约束"""
        weights = {"stocks": 0.7, "reits": 0.2, "bonds": 0.1}
        risk_params = {"max_equity": 0.6, "min_fixed_income": 0.1, "max_crypto": 0.05}

        constrained = strategy._apply_risk_constraints(
            weights, risk_params, ["stocks", "reits", "bonds"]
        )

        equity_weight = constrained.get("stocks", 0) + constrained.get("reits", 0)
        assert equity_weight <= 0.61

    def test_min_fixed_income_constraint(self, strategy):
        """测试最小固定收益约束"""
        weights = {"stocks": 0.8, "bonds": 0.1, "cash": 0.1}
        risk_params = {"max_equity": 0.9, "min_fixed_income": 0.3, "max_crypto": 0.05}

        constrained = strategy._apply_risk_constraints(
            weights, risk_params, ["stocks", "bonds", "cash"]
        )

        fixed_income = constrained.get("bonds", 0) + constrained.get("cash", 0)
        assert fixed_income >= 0.29


class TestGetPolicyPortfolio:
    """政策组合测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy()

    def test_conservative_policy(self, strategy):
        """测试保守型政策组合"""
        policy = strategy.get_policy_portfolio("conservative")

        assert policy["stocks"] == 0.20
        assert policy["bonds"] == 0.50
        assert abs(sum(policy.values()) - 1.0) < 0.01

    def test_moderate_policy(self, strategy):
        """测试稳健型政策组合"""
        policy = strategy.get_policy_portfolio("moderate")

        assert policy["stocks"] == 0.40
        assert abs(sum(policy.values()) - 1.0) < 0.01

    def test_aggressive_policy(self, strategy):
        """测试激进型政策组合"""
        policy = strategy.get_policy_portfolio("aggressive")

        assert policy["stocks"] == 0.55
        assert abs(sum(policy.values()) - 1.0) < 0.01


class TestGetExpectedReturnsAndCov:
    """预期收益和协方差估计测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy()

    def test_estimate_with_data(self, strategy):
        """测试有数据时的估计"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.0004, 0.015, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0002, 0.008, 100)}, index=dates),
        }

        mu, sigma = strategy._get_expected_returns_and_cov(
            ["stocks", "bonds"], market_data, None
        )

        assert len(mu) == 2
        assert sigma.shape == (2, 2)

    def test_estimate_without_data(self, strategy):
        """测试无数据时的估计"""
        mu, sigma = strategy._get_expected_returns_and_cov(
            ["stocks", "bonds"], {}, None
        )

        # 应该使用CMA
        assert len(mu) == 2


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def strategy(self):
        return StrategicAllocationStrategy()

    def test_full_workflow(self, strategy):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.0004, 0.015, 252)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0002, 0.008, 252)}, index=dates),
            "commodities": pd.DataFrame({"GLD": np.random.normal(0.0001, 0.012, 252)}, index=dates),
            "reits": pd.DataFrame({"VNQ": np.random.normal(0.0003, 0.014, 252)}, index=dates),
        }

        expert_views = {
            "stocks": {"expected_return": 0.10, "confidence": 0.6},
        }

        allocation = strategy.compute_allocation(
            market_data,
            expert_views=expert_views,
            risk_profile="moderate"
        )

        assert len(allocation) == 4
        assert abs(sum(allocation.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in allocation.values())

        # 验证政策组合
        policy = strategy.get_policy_portfolio("moderate")
        assert len(policy) == 6

    def test_reproducibility(self, strategy):
        """测试可重复性"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        market_data = {
            "stocks": pd.DataFrame({"SPY": np.random.normal(0.001, 0.02, 100)}, index=dates),
            "bonds": pd.DataFrame({"TLT": np.random.normal(0.0005, 0.01, 100)}, index=dates),
        }

        alloc1 = strategy.compute_allocation(market_data)
        alloc2 = strategy.compute_allocation(market_data)

        for ac in alloc1:
            assert abs(alloc1[ac] - alloc2[ac]) < 0.01

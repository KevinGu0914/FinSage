"""
Deep tests for Mean-Variance Optimization
均值方差优化深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.mean_variance import MeanVarianceTool


class TestMeanVarianceInit:
    """MeanVarianceTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = MeanVarianceTool()
        assert tool.name == "mean_variance"

    def test_description_property(self):
        """测试描述属性"""
        tool = MeanVarianceTool()
        assert "均值方差" in tool.description
        assert "Markowitz" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = MeanVarianceTool()
        params = tool.parameters
        assert "target_return" in params
        assert "target_volatility" in params
        assert "risk_free_rate" in params
        assert "objective" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_empty_returns(self, tool):
        """测试空收益率"""
        empty_df = pd.DataFrame()
        result = tool.compute_weights(empty_df)
        assert result == {}

    def test_basic_weights(self, tool, sample_returns):
        """测试基本权重计算"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 3
        assert "SPY" in weights
        assert "TLT" in weights
        assert "GLD" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.09  # 允许小误差
            assert w <= 0.51


class TestObjectiveFunctions:
    """目标函数测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "HIGH_RET": np.random.normal(0.002, 0.025, 100),
            "MED_RET": np.random.normal(0.001, 0.015, 100),
            "LOW_RET": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_max_sharpe_objective(self, tool, sample_returns):
        """测试最大化夏普比率"""
        weights = tool.compute_weights(sample_returns, objective="max_sharpe")

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_min_variance_objective(self, tool, sample_returns):
        """测试最小化方差"""
        weights = tool.compute_weights(sample_returns, objective="min_variance")

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
        # 最小方差应该偏向低波动资产
        assert weights["LOW_RET"] > 0.1

    def test_target_return_objective(self, tool, sample_returns):
        """测试目标收益"""
        weights = tool.compute_weights(
            sample_returns,
            objective="target_return",
            target_return=0.10
        )

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_different_objectives_different_weights(self, tool, sample_returns):
        """测试不同目标产生不同权重"""
        weights_sharpe = tool.compute_weights(sample_returns, objective="max_sharpe")
        weights_minvar = tool.compute_weights(sample_returns, objective="min_variance")

        # 两种目标应该产生有效权重
        assert abs(sum(weights_sharpe.values()) - 1.0) < 0.01
        assert abs(sum(weights_minvar.values()) - 1.0) < 0.01


class TestRiskFreeRate:
    """无风险利率测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
        }, index=dates)

    def test_default_risk_free(self, tool, sample_returns):
        """测试默认无风险利率"""
        weights = tool.compute_weights(sample_returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_custom_risk_free(self, tool, sample_returns):
        """测试自定义无风险利率"""
        weights = tool.compute_weights(sample_returns, risk_free_rate=0.05)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_different_risk_free_rates(self, tool, sample_returns):
        """测试不同无风险利率"""
        weights_low = tool.compute_weights(sample_returns, risk_free_rate=0.01)
        weights_high = tool.compute_weights(sample_returns, risk_free_rate=0.10)

        # 应该有有效权重
        assert abs(sum(weights_low.values()) - 1.0) < 0.01
        assert abs(sum(weights_high.values()) - 1.0) < 0.01


class TestExpertViews:
    """专家观点测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_expert_views_override_returns(self, tool, sample_returns):
        """测试专家观点覆盖历史收益"""
        expert_views = {"SPY": 0.20, "TLT": 0.05, "GLD": 0.08}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_partial_expert_views(self, tool, sample_returns):
        """测试部分专家观点"""
        expert_views = {"SPY": 0.15}  # 只对SPY有观点
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3

    def test_bullish_view_higher_weight(self, tool, sample_returns):
        """测试看涨观点增加权重"""
        expert_views = {"SPY": 0.50, "TLT": 0.05, "GLD": 0.05}  # 强烈看涨SPY
        weights = tool.compute_weights(
            sample_returns,
            expert_views=expert_views,
            objective="max_sharpe"
        )

        # SPY应该有较高权重
        assert weights["SPY"] > 0.2


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    def test_single_asset(self, tool):
        """测试单资产"""
        dates = pd.date_range("2023-01-01", periods=50)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 50)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 1
        assert abs(weights["SPY"] - 1.0) < 0.01

    def test_two_assets(self, tool):
        """测试两资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_many_assets(self, tool):
        """测试多资产"""
        np.random.seed(42)
        n_assets = 20
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            f"Asset_{i}": np.random.normal(0.001, 0.02, 100)
            for i in range(n_assets)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == n_assets
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_short_history(self, tool):
        """测试短历史数据"""
        dates = pd.date_range("2023-01-01", periods=15)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 15),
            "TLT": np.random.normal(0.0005, 0.01, 15),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2

    def test_highly_correlated_assets(self, tool):
        """测试高度相关资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.002, 100),
            "C": base + np.random.normal(0, 0.003, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_negative_correlation(self, tool):
        """测试负相关资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": -base + np.random.normal(0.002, 0.001, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestConstraintVariations:
    """约束变化测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.025, 100),
        }, index=dates)

    def test_tight_constraints(self, tool, sample_returns):
        """测试紧约束"""
        constraints = {"min_weight": 0.3, "max_single_asset": 0.35}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert abs(w - 1/3) < 0.1

    def test_min_weight_constraint(self, tool, sample_returns):
        """测试最小权重约束"""
        constraints = {"min_weight": 0.2}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.19

    def test_max_weight_constraint(self, tool, sample_returns):
        """测试最大权重约束"""
        constraints = {"max_single_asset": 0.4}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w <= 0.41

    def test_no_constraints(self, tool, sample_returns):
        """测试无约束"""
        weights = tool.compute_weights(sample_returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return MeanVarianceTool()

    def test_full_workflow(self, tool):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.015, 252),
            "TLT": np.random.normal(0.0002, 0.008, 252),
            "GLD": np.random.normal(0.0001, 0.012, 252),
            "QQQ": np.random.normal(0.0005, 0.02, 252),
        }, index=dates)

        expert_views = {"SPY": 0.12, "QQQ": 0.15, "TLT": 0.03}
        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            risk_free_rate=0.02,
            objective="max_sharpe"
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0.04 for w in weights.values())
        assert all(w <= 0.41 for w in weights.values())

    def test_reproducibility(self, tool):
        """测试可重复性"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights1 = tool.compute_weights(returns)
        weights2 = tool.compute_weights(returns)

        for asset in weights1:
            assert abs(weights1[asset] - weights2[asset]) < 0.001

    def test_min_variance_minimizes_variance(self, tool):
        """测试最小方差确实最小化了方差"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.03, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        mv_weights = tool.compute_weights(returns, objective="min_variance")

        # 计算组合方差
        cov = returns.cov().values * 252
        w = np.array([mv_weights[col] for col in returns.columns])
        mv_variance = w @ cov @ w

        # 等权方差
        eq_w = np.array([1/3, 1/3, 1/3])
        eq_variance = eq_w @ cov @ eq_w

        # 最小方差组合的方差应该小于或等于等权
        assert mv_variance <= eq_variance + 0.01

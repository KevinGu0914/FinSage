"""
Deep tests for Minimum Variance Portfolio
最小方差组合深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.minimum_variance import MinimumVarianceTool


class TestMinimumVarianceInit:
    """MinimumVariance初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = MinimumVarianceTool()
        assert tool.name == "minimum_variance"

    def test_description_property(self):
        """测试描述属性"""
        tool = MinimumVarianceTool()
        assert "最小方差" in tool.description
        assert "Markowitz" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = MinimumVarianceTool()
        params = tool.parameters
        assert "min_weight" in params
        assert "max_weight" in params
        assert "allow_short" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.008, 100),
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

    def test_weights_favor_low_volatility(self, tool):
        """测试权重偏向低波动资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        # TLT有更低的波动率
        returns = pd.DataFrame({
            "HIGH_VOL": np.random.normal(0.001, 0.04, 100),
            "LOW_VOL": np.random.normal(0.001, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 低波动资产应该有更高权重
        assert weights["LOW_VOL"] > weights["HIGH_VOL"]

    def test_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.09  # 允许小误差
            assert w <= 0.51


class TestOptimizationBehavior:
    """优化行为测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    def test_optimization_converges(self, tool):
        """测试优化收敛"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.025, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 优化应该收敛
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= -0.01 for w in weights.values())

    def test_optimization_with_tight_constraints(self, tool):
        """测试紧约束下的优化"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.025, 100),
        }, index=dates)

        # 非常紧的约束
        constraints = {"min_weight": 0.3, "max_single_asset": 0.35}
        weights = tool.compute_weights(returns, constraints=constraints)

        # 约束应该使权重接近等权
        for w in weights.values():
            assert abs(w - 1/3) < 0.1


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    def test_single_asset(self, tool):
        """测试单资产"""
        dates = pd.date_range("2023-01-01", periods=50)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 50)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 1
        assert abs(weights["SPY"] - 1.0) < 0.01

    def test_two_assets_perfectly_correlated(self, tool):
        """测试完全相关的两资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": base * 2,  # 完全相关
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_two_assets_negative_correlation(self, tool):
        """测试负相关的两资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": -base + np.random.normal(0, 0.001, 100),  # 负相关
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 负相关资产提供分散化收益
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_many_assets(self, tool):
        """测试多资产"""
        np.random.seed(42)
        n_assets = 20
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            f"Asset_{i}": np.random.normal(0.001, 0.02 + i*0.001, 100)
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
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_very_different_volatilities(self, tool):
        """测试波动率差异很大的资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "LOW_VOL": np.random.normal(0.001, 0.005, 100),
            "MED_VOL": np.random.normal(0.001, 0.02, 100),
            "HIGH_VOL": np.random.normal(0.001, 0.05, 100),
        }, index=dates)

        constraints = {"max_single_asset": 0.6}
        weights = tool.compute_weights(returns, constraints=constraints)

        # 低波动应该权重最高
        assert weights["LOW_VOL"] >= weights["MED_VOL"]


class TestConstraintVariations:
    """约束变化测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.025, 100),
        }, index=dates)

    def test_no_constraints(self, tool, sample_returns):
        """测试无约束"""
        weights = tool.compute_weights(sample_returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_min_weight_constraint(self, tool, sample_returns):
        """测试最小权重约束"""
        constraints = {"min_weight": 0.2}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.19  # 允许小误差

    def test_max_weight_constraint(self, tool, sample_returns):
        """测试最大权重约束"""
        constraints = {"max_single_asset": 0.4}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w <= 0.41  # 允许小误差

    def test_both_constraints(self, tool, sample_returns):
        """测试双重约束"""
        constraints = {"min_weight": 0.15, "max_single_asset": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.14
            assert w <= 0.51


class TestExpertViewsIgnored:
    """专家观点忽略测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_expert_views_no_effect(self, tool, sample_returns):
        """测试专家观点不影响结果"""
        weights_no_views = tool.compute_weights(sample_returns)
        weights_with_views = tool.compute_weights(
            sample_returns,
            expert_views={"SPY": 0.5, "TLT": -0.5}
        )

        # 最小方差不使用预期收益，所以观点不应影响
        for asset in weights_no_views:
            assert abs(weights_no_views[asset] - weights_with_views[asset]) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return MinimumVarianceTool()

    def test_full_workflow(self, tool):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.015, 252),
            "TLT": np.random.normal(0.0002, 0.006, 252),
            "GLD": np.random.normal(0.0001, 0.012, 252),
            "QQQ": np.random.normal(0.0005, 0.02, 252),
        }, index=dates)

        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(returns, constraints=constraints)

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

    def test_portfolio_variance_is_minimized(self, tool):
        """测试组合方差是最小的"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.03, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        mv_weights = tool.compute_weights(returns)

        # 计算组合方差
        cov = returns.cov().values * 252
        w = np.array([mv_weights[col] for col in returns.columns])
        mv_variance = w @ cov @ w

        # 等权方差
        eq_w = np.array([1/3, 1/3, 1/3])
        eq_variance = eq_w @ cov @ eq_w

        # 最小方差组合的方差应该小于或等于等权
        assert mv_variance <= eq_variance + 0.01

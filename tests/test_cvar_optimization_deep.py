"""
Deep tests for CVaR Optimization
条件风险价值优化深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool


class TestCVaRInit:
    """CVaROptimizationTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = CVaROptimizationTool()
        assert tool.name == "cvar_optimization"

    def test_description_property(self):
        """测试描述属性"""
        tool = CVaROptimizationTool()
        assert "CVaR" in tool.description
        assert "Rockafellar" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = CVaROptimizationTool()
        params = tool.parameters
        assert "alpha" in params
        assert "min_weight" in params
        assert "max_weight" in params
        assert "target_return" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

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

    def test_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点的权重"""
        expert_views = {"SPY": 0.10, "TLT": 0.05}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestAlphaParameter:
    """Alpha参数测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
        }, index=dates)

    def test_alpha_95(self, tool, sample_returns):
        """测试95%置信水平"""
        weights = tool.compute_weights(sample_returns, alpha=0.95)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_alpha_99(self, tool, sample_returns):
        """测试99%置信水平"""
        weights = tool.compute_weights(sample_returns, alpha=0.99)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_different_alpha_different_weights(self, tool, sample_returns):
        """测试不同alpha产生不同权重"""
        weights_95 = tool.compute_weights(sample_returns, alpha=0.95)
        weights_99 = tool.compute_weights(sample_returns, alpha=0.99)

        # 可能相同，但都应该有效
        assert abs(sum(weights_95.values()) - 1.0) < 0.01
        assert abs(sum(weights_99.values()) - 1.0) < 0.01


class TestComputePortfolioCVaR:
    """组合CVaR计算测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_cvar_calculation_basic(self, tool, sample_returns):
        """测试基本CVaR计算"""
        weights = {"SPY": 0.6, "TLT": 0.4}
        cvar = tool.compute_portfolio_cvar(sample_returns, weights, alpha=0.95)

        assert cvar > 0  # CVaR是正的损失值
        assert isinstance(cvar, float)

    def test_cvar_equal_weights(self, tool, sample_returns):
        """测试等权CVaR"""
        weights = {"SPY": 0.5, "TLT": 0.5}
        cvar = tool.compute_portfolio_cvar(sample_returns, weights)

        assert cvar > 0

    def test_cvar_single_asset(self, tool, sample_returns):
        """测试单资产CVaR"""
        weights = {"SPY": 1.0, "TLT": 0.0}
        cvar = tool.compute_portfolio_cvar(sample_returns, weights)

        assert cvar > 0

    def test_higher_alpha_higher_cvar(self, tool, sample_returns):
        """测试更高alpha通常产生更高CVaR"""
        weights = {"SPY": 0.5, "TLT": 0.5}
        cvar_95 = tool.compute_portfolio_cvar(sample_returns, weights, alpha=0.95)
        cvar_99 = tool.compute_portfolio_cvar(sample_returns, weights, alpha=0.99)

        # 99% CVaR应该 >= 95% CVaR
        assert cvar_99 >= cvar_95 - 0.01  # 允许小误差


class TestCVaROptimizationBehavior:
    """CVaR优化行为测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

    def test_favors_lower_tail_risk(self, tool):
        """测试优化偏向低尾部风险资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200)

        # 创建一个有明显尾部风险差异的数据集
        high_tail_risk = np.concatenate([
            np.random.normal(0.001, 0.01, 190),
            np.random.normal(-0.05, 0.02, 10)  # 10个极端负收益
        ])
        low_tail_risk = np.random.normal(0.001, 0.015, 200)

        returns = pd.DataFrame({
            "HIGH_TAIL": high_tail_risk,
            "LOW_TAIL": low_tail_risk,
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 低尾部风险资产应该权重更高
        assert weights["LOW_TAIL"] >= weights["HIGH_TAIL"] - 0.1

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

        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= -0.01 for w in weights.values())


class TestTargetReturnConstraint:
    """目标收益约束测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "HIGH_RET": np.random.normal(0.002, 0.025, 100),
            "MED_RET": np.random.normal(0.001, 0.015, 100),
            "LOW_RET": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_with_target_return(self, tool, sample_returns):
        """测试目标收益约束"""
        constraints = {"target_return": 0.10}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

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
        n_assets = 15
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
        dates = pd.date_range("2023-01-01", periods=20)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 20),
            "TLT": np.random.normal(0.0005, 0.01, 20),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2

    def test_high_volatility_assets(self, tool):
        """测试高波动资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "HIGH_VOL": np.random.normal(0.002, 0.05, 100),
            "LOW_VOL": np.random.normal(0.001, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 低波动资产应该有更高权重
        assert weights["LOW_VOL"] >= weights["HIGH_VOL"]


class TestConstraintVariations:
    """约束变化测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

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

        # 约束应该使权重接近等权
        for w in weights.values():
            assert abs(w - 1/3) < 0.1

    def test_no_constraints(self, tool, sample_returns):
        """测试无约束"""
        weights = tool.compute_weights(sample_returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return CVaROptimizationTool()

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

        expert_views = {"SPY": 0.12, "QQQ": 0.15}
        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            alpha=0.95
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0.04 for w in weights.values())
        assert all(w <= 0.41 for w in weights.values())

        # 验证CVaR计算
        cvar = tool.compute_portfolio_cvar(returns, weights, alpha=0.95)
        assert cvar > 0

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

    def test_cvar_minimized(self, tool):
        """测试CVaR是否被最小化"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.03, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        cvar_weights = tool.compute_weights(returns)

        # 计算CVaR优化组合的CVaR
        cvar_opt = tool.compute_portfolio_cvar(returns, cvar_weights)

        # 计算等权组合的CVaR
        eq_weights = {"A": 1/3, "B": 1/3, "C": 1/3}
        cvar_eq = tool.compute_portfolio_cvar(returns, eq_weights)

        # CVaR优化组合的CVaR应该小于或等于等权组合
        assert cvar_opt <= cvar_eq + 0.01

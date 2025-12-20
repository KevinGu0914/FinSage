"""
Deep tests for Black-Litterman Model
Black-Litterman模型深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.black_litterman import BlackLittermanTool


class TestBlackLittermanInit:
    """BlackLittermanTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = BlackLittermanTool()
        assert tool.name == "black_litterman"

    def test_description_property(self):
        """测试描述属性"""
        tool = BlackLittermanTool()
        assert "Black-Litterman" in tool.description
        assert "均衡收益" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = BlackLittermanTool()
        params = tool.parameters
        assert "risk_aversion" in params
        assert "tau" in params
        assert "market_weights" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

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

    def test_basic_weights_no_views(self, tool, sample_returns):
        """测试无观点时的基本权重"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 3
        assert "SPY" in weights
        assert "TLT" in weights
        assert "GLD" in weights

        # 权重和为1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点的权重"""
        expert_views = {"SPY": 0.10, "TLT": 0.05}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= -0.01  # 允许小误差

    def test_custom_risk_aversion(self, tool, sample_returns):
        """测试自定义风险厌恶系数"""
        weights_low = tool.compute_weights(sample_returns, risk_aversion=1.0)
        weights_high = tool.compute_weights(sample_returns, risk_aversion=5.0)

        # 两组权重应该都有效
        assert abs(sum(weights_low.values()) - 1.0) < 0.01
        assert abs(sum(weights_high.values()) - 1.0) < 0.01

    def test_custom_tau(self, tool, sample_returns):
        """测试自定义tau参数"""
        expert_views = {"SPY": 0.15}
        weights_low_tau = tool.compute_weights(
            sample_returns, expert_views=expert_views, tau=0.01
        )
        weights_high_tau = tool.compute_weights(
            sample_returns, expert_views=expert_views, tau=0.1
        )

        # 两种tau设置都应该产生有效的权重分配
        assert len(weights_low_tau) == 3
        assert len(weights_high_tau) == 3
        assert abs(sum(weights_low_tau.values()) - 1.0) < 0.01
        assert abs(sum(weights_high_tau.values()) - 1.0) < 0.01


class TestBuildViewMatrices:
    """观点矩阵构建测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

    def test_no_views(self, tool):
        """测试无观点时"""
        assets = ["SPY", "TLT", "GLD"]
        cov_matrix = np.eye(3)
        tau = 0.05

        P, Q, omega = tool._build_view_matrices(assets, {}, cov_matrix, tau)

        assert P is None
        assert Q == []
        assert omega is None

    def test_single_view(self, tool):
        """测试单个观点"""
        assets = ["SPY", "TLT", "GLD"]
        cov_matrix = np.eye(3) * 0.04
        tau = 0.05
        expert_views = {"SPY": 0.10}

        P, Q, omega = tool._build_view_matrices(assets, expert_views, cov_matrix, tau)

        assert P.shape == (1, 3)
        assert len(Q) == 1
        assert Q[0] == 0.10
        assert omega.shape == (1, 1)

    def test_multiple_views(self, tool):
        """测试多个观点"""
        assets = ["SPY", "TLT", "GLD"]
        cov_matrix = np.eye(3) * 0.04
        tau = 0.05
        expert_views = {"SPY": 0.10, "TLT": 0.05}

        P, Q, omega = tool._build_view_matrices(assets, expert_views, cov_matrix, tau)

        assert P.shape == (2, 3)
        assert len(Q) == 2
        assert omega.shape == (2, 2)

    def test_invalid_asset_in_views(self, tool):
        """测试无效资产的观点"""
        assets = ["SPY", "TLT", "GLD"]
        cov_matrix = np.eye(3) * 0.04
        tau = 0.05
        expert_views = {"INVALID": 0.10}

        P, Q, omega = tool._build_view_matrices(assets, expert_views, cov_matrix, tau)

        assert P is None  # 无有效观点


class TestMeanVarianceOptimize:
    """均值方差优化测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

    def test_basic_optimization(self, tool):
        """测试基本优化"""
        expected_returns = np.array([0.10, 0.05, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.002],
            [0.005, 0.002, 0.03]
        ])

        weights = tool._mean_variance_optimize(expected_returns, cov_matrix)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01

    def test_with_constraints(self, tool):
        """测试带约束的优化"""
        expected_returns = np.array([0.10, 0.05, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.002],
            [0.005, 0.002, 0.03]
        ])
        constraints = {"max_single_asset": 0.5}

        weights = tool._mean_variance_optimize(expected_returns, cov_matrix, constraints)

        assert max(weights) <= 0.51  # 允许小误差

    def test_with_target_return(self, tool):
        """测试目标收益约束"""
        expected_returns = np.array([0.10, 0.05, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.002],
            [0.005, 0.002, 0.03]
        ])
        constraints = {"target_return": 0.07}

        weights = tool._mean_variance_optimize(expected_returns, cov_matrix, constraints)

        # 应该返回权重
        assert len(weights) == 3


class TestMarketWeights:
    """市场权重测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_custom_market_weights(self, tool, sample_returns):
        """测试自定义市场权重"""
        market_weights = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}

        weights = tool.compute_weights(
            sample_returns,
            market_weights=market_weights
        )

        assert len(weights) == 3

    def test_default_equal_market_weights(self, tool, sample_returns):
        """测试默认等权市场权重"""
        weights = tool.compute_weights(sample_returns)

        # 应该使用等权作为默认
        assert len(weights) == 3


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

    def test_single_asset(self, tool):
        """测试单资产"""
        dates = pd.date_range("2023-01-01", periods=50)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 50)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 1
        assert abs(weights["SPY"] - 1.0) < 0.01

    def test_highly_correlated_assets(self, tool):
        """测试高度相关资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.001, 100),
            "C": base + np.random.normal(0, 0.002, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

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
        dates = pd.date_range("2023-01-01", periods=10)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 10),
            "TLT": np.random.normal(0.0005, 0.01, 10),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2

    def test_extreme_views(self, tool):
        """测试极端观点"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        extreme_views = {"SPY": 1.0}  # 100% 预期收益

        weights = tool.compute_weights(returns, expert_views=extreme_views)

        # 应该仍然返回有效权重
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return BlackLittermanTool()

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
        market_weights = {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.2, "GLD": 0.1}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            risk_aversion=3.0,
            tau=0.05,
            market_weights=market_weights
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= -0.01 for w in weights.values())

    def test_reproducibility(self, tool):
        """测试可重复性"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights1 = tool.compute_weights(returns, expert_views={"SPY": 0.10})
        weights2 = tool.compute_weights(returns, expert_views={"SPY": 0.10})

        # 相同输入应该产生相同输出
        for asset in weights1:
            assert abs(weights1[asset] - weights2[asset]) < 0.001

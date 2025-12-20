"""
Deep tests for Hierarchical Risk Parity (HRP)
层次风险平价深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.hrp import HierarchicalRiskParityTool


class TestHRPInit:
    """HRP初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = HierarchicalRiskParityTool()
        assert tool.name == "hrp"

    def test_description_property(self):
        """测试描述属性"""
        tool = HierarchicalRiskParityTool()
        assert "层次风险平价" in tool.description
        assert "聚类" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = HierarchicalRiskParityTool()
        params = tool.parameters
        assert "linkage_method" in params
        assert "distance_metric" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
            "QQQ": np.random.normal(0.0006, 0.025, 100),
        }, index=dates)

    def test_empty_returns(self, tool):
        """测试空收益率"""
        empty_df = pd.DataFrame()
        result = tool.compute_weights(empty_df)
        assert result == {}

    def test_short_returns(self, tool):
        """测试短数据"""
        dates = pd.date_range("2023-01-01", periods=5)
        short_df = pd.DataFrame({
            "SPY": [0.01, -0.01, 0.02, -0.005, 0.01],
            "TLT": [0.005, 0.002, -0.001, 0.003, -0.002],
        }, index=dates)

        result = tool.compute_weights(short_df)
        # 短数据应返回等权
        assert len(result) == 2
        assert abs(result["SPY"] - 0.5) < 0.01
        assert abs(result["TLT"] - 0.5) < 0.01

    def test_basic_weights(self, tool, sample_returns):
        """测试基本权重计算"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 4
        # 权重和为1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.4}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        assert len(weights) == 4
        for w in weights.values():
            assert w >= 0.09  # 允许小误差
            assert w <= 0.41


class TestQuasiDiag:
    """准对角化测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    def test_quasi_diag_basic(self, tool):
        """测试基本准对角化"""
        corr_matrix = np.array([
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0]
        ])

        order = tool._get_quasi_diag(corr_matrix)

        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_quasi_diag_identity(self, tool):
        """测试单位相关矩阵"""
        corr_matrix = np.eye(5)

        order = tool._get_quasi_diag(corr_matrix)

        assert len(order) == 5

    def test_different_linkage_methods(self, tool):
        """测试不同聚类方法"""
        corr_matrix = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])

        order_single = tool._get_quasi_diag(corr_matrix, "single")
        order_complete = tool._get_quasi_diag(corr_matrix, "complete")
        order_average = tool._get_quasi_diag(corr_matrix, "average")

        # 都应返回有效顺序
        assert len(order_single) == 3
        assert len(order_complete) == 3
        assert len(order_average) == 3


class TestRecursiveBisection:
    """递归二分测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    def test_recursive_bisection_basic(self, tool):
        """测试基本递归二分"""
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.03]
        ])
        order = [0, 1, 2]

        weights = tool._recursive_bisection(cov_matrix, order)

        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01

    def test_recursive_bisection_two_assets(self, tool):
        """测试两资产递归二分"""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.02]
        ])
        order = [0, 1]

        weights = tool._recursive_bisection(cov_matrix, order)

        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 0.01

    def test_recursive_bisection_single_asset(self, tool):
        """测试单资产"""
        cov_matrix = np.array([[0.04]])
        order = [0]

        weights = tool._recursive_bisection(cov_matrix, order)

        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 0.01


class TestGetClusterVar:
    """聚类方差测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    def test_cluster_var_single(self, tool):
        """测试单资产聚类方差"""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.02]
        ])

        var = tool._get_cluster_var(cov_matrix, [0])

        assert var > 0

    def test_cluster_var_multiple(self, tool):
        """测试多资产聚类方差"""
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.03]
        ])

        var = tool._get_cluster_var(cov_matrix, [0, 1])

        assert var > 0


class TestApplyConstraints:
    """约束应用测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    def test_apply_min_constraint(self, tool):
        """测试最小权重约束"""
        weights = np.array([0.05, 0.15, 0.8])
        constraints = {"min_weight": 0.1}

        constrained = tool._apply_constraints(weights, constraints)

        assert all(constrained >= 0.1)
        assert abs(constrained.sum() - 1.0) < 0.01

    def test_apply_max_constraint(self, tool):
        """测试最大权重约束"""
        weights = np.array([0.1, 0.2, 0.7])
        constraints = {"max_single_asset": 0.4}

        constrained = tool._apply_constraints(weights, constraints)

        assert all(constrained <= 0.4)
        assert abs(constrained.sum() - 1.0) < 0.01

    def test_apply_both_constraints(self, tool):
        """测试双重约束"""
        weights = np.array([0.05, 0.15, 0.8])
        constraints = {"min_weight": 0.15, "max_single_asset": 0.5}

        constrained = tool._apply_constraints(weights, constraints)

        assert all(constrained >= 0.15)
        assert all(constrained <= 0.5)
        assert abs(constrained.sum() - 1.0) < 0.01


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

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

    def test_uncorrelated_assets(self, tool):
        """测试不相关资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.02, 100),
            "C": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 不相关资产应该接近等权
        for w in weights.values():
            assert abs(w - 1/3) < 0.15

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


class TestDifferentLinkageMethods:
    """不同聚类方法测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_single_linkage(self, tool, sample_returns):
        """测试单连接"""
        weights = tool.compute_weights(sample_returns, linkage_method="single")
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_complete_linkage(self, tool, sample_returns):
        """测试完全连接"""
        weights = tool.compute_weights(sample_returns, linkage_method="complete")
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_average_linkage(self, tool, sample_returns):
        """测试平均连接"""
        weights = tool.compute_weights(sample_returns, linkage_method="average")
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return HierarchicalRiskParityTool()

    def test_full_workflow(self, tool):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.015, 252),
            "TLT": np.random.normal(0.0002, 0.008, 252),
            "GLD": np.random.normal(0.0001, 0.012, 252),
            "QQQ": np.random.normal(0.0005, 0.02, 252),
            "IEF": np.random.normal(0.0001, 0.005, 252),
        }, index=dates)

        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            constraints=constraints,
            linkage_method="single"
        )

        assert len(weights) == 5
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

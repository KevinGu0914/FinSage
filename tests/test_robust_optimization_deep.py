"""
Deep tests for Robust Portfolio Optimization
鲁棒组合优化深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.robust_optimization import RobustOptimizationTool


class TestRobustOptimizationInit:
    """RobustOptimizationTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = RobustOptimizationTool()
        assert tool.name == "robust_optimization"

    def test_description_property(self):
        """测试描述属性"""
        tool = RobustOptimizationTool()
        assert "鲁棒" in tool.description
        assert "Goldfarb" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = RobustOptimizationTool()
        params = tool.parameters
        assert "uncertainty_level" in params
        assert "risk_aversion" in params
        assert "min_weight" in params
        assert "max_weight" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

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


class TestUncertaintyLevel:
    """不确定性水平测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.002, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_low_uncertainty(self, tool, sample_returns):
        """测试低不确定性"""
        weights = tool.compute_weights(sample_returns, uncertainty_level=0.01)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_high_uncertainty(self, tool, sample_returns):
        """测试高不确定性"""
        weights = tool.compute_weights(sample_returns, uncertainty_level=0.5)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_higher_uncertainty_more_conservative(self, tool, sample_returns):
        """测试更高不确定性更保守"""
        weights_low = tool.compute_weights(sample_returns, uncertainty_level=0.01)
        weights_high = tool.compute_weights(sample_returns, uncertainty_level=0.5)

        # 高不确定性应该产生更分散的权重
        variance_low = np.var(list(weights_low.values()))
        variance_high = np.var(list(weights_high.values()))

        # 高不确定性权重应该更接近等权（方差更低）
        assert variance_high <= variance_low + 0.05


class TestRiskAversion:
    """风险厌恶系数测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "HIGH_VOL": np.random.normal(0.002, 0.03, 100),
            "LOW_VOL": np.random.normal(0.001, 0.01, 100),
        }, index=dates)

    def test_low_risk_aversion(self, tool, sample_returns):
        """测试低风险厌恶"""
        weights = tool.compute_weights(sample_returns, risk_aversion=0.5)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_high_risk_aversion(self, tool, sample_returns):
        """测试高风险厌恶"""
        weights = tool.compute_weights(sample_returns, risk_aversion=5.0)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_higher_risk_aversion_favors_low_vol(self, tool, sample_returns):
        """测试高风险厌恶偏好低波动"""
        weights = tool.compute_weights(sample_returns, risk_aversion=5.0)

        # 高风险厌恶应该偏好低波动资产
        assert weights["LOW_VOL"] >= weights["HIGH_VOL"]


class TestExpertViews:
    """专家观点测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_expert_views_reduce_uncertainty(self, tool, sample_returns):
        """测试专家观点减少不确定性"""
        expert_views = {"SPY": 0.5}  # 高信心
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_no_expert_views(self, tool, sample_returns):
        """测试无专家观点"""
        weights = tool.compute_weights(sample_returns, expert_views=None)

        assert len(weights) == 2


class TestMakePositiveDefinite:
    """正定矩阵处理测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

    def test_already_positive_definite(self, tool):
        """测试已经正定的矩阵"""
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = tool._make_positive_definite(matrix)

        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)

    def test_non_positive_definite(self, tool):
        """测试非正定矩阵"""
        # 创建一个近似奇异的矩阵
        matrix = np.array([[1.0, 0.99], [0.99, 1.0]])
        result = tool._make_positive_definite(matrix)

        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)

    def test_negative_eigenvalues(self, tool):
        """测试负特征值矩阵"""
        # 构造一个有负特征值的矩阵
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = tool._make_positive_definite(matrix)

        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)


class TestComputeWorstCaseReturn:
    """最坏情况收益计算测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

    def test_worst_case_return_basic(self, tool, sample_returns):
        """测试基本最坏情况收益"""
        weights = {"SPY": 0.6, "TLT": 0.4}
        worst_case = tool.compute_worst_case_return(
            sample_returns, weights, uncertainty_level=0.1
        )

        assert isinstance(worst_case, float)

    def test_worst_case_less_than_expected(self, tool, sample_returns):
        """测试最坏情况小于期望收益"""
        weights = {"SPY": 0.5, "TLT": 0.5}

        worst_case = tool.compute_worst_case_return(
            sample_returns, weights, uncertainty_level=0.1
        )

        # 计算期望收益
        w = np.array([0.5, 0.5])
        mu = sample_returns.mean().values * 252
        expected = mu @ w

        assert worst_case <= expected

    def test_higher_uncertainty_lower_worst_case(self, tool, sample_returns):
        """测试更高不确定性产生更低最坏情况"""
        weights = {"SPY": 0.5, "TLT": 0.5}

        worst_case_low = tool.compute_worst_case_return(
            sample_returns, weights, uncertainty_level=0.05
        )
        worst_case_high = tool.compute_worst_case_return(
            sample_returns, weights, uncertainty_level=0.3
        )

        assert worst_case_high <= worst_case_low


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

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
        dates = pd.date_range("2023-01-01", periods=15)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 15),
            "TLT": np.random.normal(0.0005, 0.01, 15),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2

    def test_highly_correlated(self, tool):
        """测试高度相关资产"""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.002, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestConstraintVariations:
    """约束变化测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

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


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return RobustOptimizationTool()

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

        expert_views = {"SPY": 0.3, "QQQ": 0.4}
        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            uncertainty_level=0.1,
            risk_aversion=2.0
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0.04 for w in weights.values())
        assert all(w <= 0.41 for w in weights.values())

        # 验证最坏情况收益
        worst_case = tool.compute_worst_case_return(returns, weights)
        assert isinstance(worst_case, float)

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

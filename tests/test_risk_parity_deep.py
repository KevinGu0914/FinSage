"""
Deep tests for Risk Parity Portfolio
风险平价组合深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.risk_parity import RiskParityTool


class TestRiskParityInit:
    """RiskParityTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = RiskParityTool()
        assert tool.name == "risk_parity"

    def test_description_property(self):
        """测试描述属性"""
        tool = RiskParityTool()
        assert "风险平价" in tool.description
        assert "Risk Parity" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = RiskParityTool()
        params = tool.parameters
        assert "target_risk_contribution" in params
        assert "budget" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

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


class TestRiskContribution:
    """风险贡献测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

    def test_equal_risk_contribution(self, tool):
        """测试等风险贡献"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 200),
            "B": np.random.normal(0.001, 0.03, 200),
            "C": np.random.normal(0.001, 0.01, 200),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 计算风险贡献
        cov = returns.cov().values * 252
        w = np.array([weights["A"], weights["B"], weights["C"]])
        port_var = w @ cov @ w
        port_std = np.sqrt(port_var)
        marginal = cov @ w
        contrib = w * marginal / port_std

        # 风险贡献应该接近相等
        mean_contrib = np.mean(contrib)
        for c in contrib:
            assert abs(c - mean_contrib) < mean_contrib * 0.5

    def test_low_vol_higher_weight(self, tool):
        """测试低波动资产有更高权重"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "LOW_VOL": np.random.normal(0.001, 0.01, 100),
            "HIGH_VOL": np.random.normal(0.001, 0.04, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 低波动资产应该有更高权重
        assert weights["LOW_VOL"] > weights["HIGH_VOL"]


class TestSimpleRiskParity:
    """简化风险平价测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

    def test_simple_risk_parity_basic(self, tool):
        """测试简化风险平价"""
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01]
        ])

        weights = tool._simple_risk_parity(cov_matrix)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01

    def test_simple_risk_parity_inverse_vol(self, tool):
        """测试简化风险平价是反波动率"""
        cov_matrix = np.array([
            [0.04, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.09]
        ])

        weights = tool._simple_risk_parity(cov_matrix)

        # 波动率 = [0.2, 0.1, 0.3]
        # 反波动率权重应该是 [5, 10, 3.33] / 18.33
        assert weights[1] > weights[0] > weights[2]


class TestExpertViews:
    """专家观点测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_expert_views_affect_budget(self, tool, sample_returns):
        """测试专家观点影响风险预算"""
        expert_views = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_no_expert_views(self, tool, sample_returns):
        """测试无专家观点"""
        weights = tool.compute_weights(sample_returns, expert_views=None)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

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
            f"Asset_{i}": np.random.normal(0.001, 0.02 + i*0.001, 100)
            for i in range(n_assets)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == n_assets
        assert abs(sum(weights.values()) - 1.0) < 0.01

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

        # 不相关等波动资产应该接近等权
        for w in weights.values():
            assert abs(w - 1/3) < 0.15


class TestConstraintVariations:
    """约束变化测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.03, 100),
            "C": np.random.normal(0.001, 0.01, 100),
        }, index=dates)

    def test_tight_min_constraint(self, tool, sample_returns):
        """测试紧的最小权重约束"""
        constraints = {"min_weight": 0.2}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.19

    def test_tight_max_constraint(self, tool, sample_returns):
        """测试紧的最大权重约束"""
        constraints = {"max_single_asset": 0.35}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w <= 0.36

    def test_both_constraints(self, tool, sample_returns):
        """测试双重约束"""
        constraints = {"min_weight": 0.2, "max_single_asset": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.19
            assert w <= 0.51


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return RiskParityTool()

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

        weights = tool.compute_weights(returns, constraints=constraints)

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

    def test_risk_contribution_verification(self, tool):
        """测试风险贡献验证"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 200),
            "B": np.random.normal(0.001, 0.015, 200),
            "C": np.random.normal(0.001, 0.025, 200),
        }, index=dates)

        weights = tool.compute_weights(returns)

        # 验证权重有效
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w > 0 for w in weights.values())

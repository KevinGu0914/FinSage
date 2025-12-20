"""
Deep tests for Factor-Based Hedging
因子对冲策略深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.factor_hedging import FactorHedgingTool


class TestFactorHedgingInit:
    """FactorHedgingTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = FactorHedgingTool()
        assert tool.name == "factor_hedging"

    def test_description_property(self):
        """测试描述属性"""
        tool = FactorHedgingTool()
        assert "因子对冲" in tool.description
        assert "Fama-French" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = FactorHedgingTool()
        params = tool.parameters
        assert "target_factors" in params
        assert "factor_tolerance" in params
        assert "use_momentum" in params
        assert "lookback_period" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

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

    def test_basic_weights(self, tool, sample_returns):
        """测试基本权重计算"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.4}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.09  # 允许小误差
            assert w <= 0.41

    def test_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点的权重"""
        expert_views = {"SPY": 0.15, "QQQ": 0.20}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestEstimateFactorBetas:
    """因子Beta估计测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "HIGH_BETA": np.random.normal(0.002, 0.03, 100),
            "LOW_BETA": np.random.normal(0.001, 0.01, 100),
            "MED_BETA": np.random.normal(0.0015, 0.02, 100),
        }, index=dates)

    def test_estimate_basic_factors(self, tool, sample_returns):
        """测试基本因子估计"""
        factor_betas, factor_returns = tool._estimate_factor_betas(
            sample_returns, use_momentum=False
        )

        assert "market" in factor_betas
        assert "size" in factor_betas
        assert "value" in factor_betas

    def test_estimate_with_momentum(self, tool, sample_returns):
        """测试包含动量因子"""
        factor_betas, factor_returns = tool._estimate_factor_betas(
            sample_returns, use_momentum=True, lookback=60
        )

        if "momentum" in factor_betas:
            assert len(factor_betas["momentum"]) == 3

    def test_short_lookback(self, tool):
        """测试短回溯期"""
        dates = pd.date_range("2023-01-01", periods=25)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 25),
            "B": np.random.normal(0.001, 0.015, 25),
        }, index=dates)

        factor_betas, factor_returns = tool._estimate_factor_betas(
            returns, lookback=60
        )

        assert "market" in factor_betas

    def test_factor_returns_shape(self, tool, sample_returns):
        """测试因子收益形状"""
        factor_betas, factor_returns = tool._estimate_factor_betas(sample_returns)

        assert isinstance(factor_returns, pd.DataFrame)
        assert len(factor_returns) <= len(sample_returns)


class TestComputeResidualCov:
    """残差协方差计算测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

    def test_residual_cov_basic(self, tool):
        """测试基本残差协方差计算"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
        }, index=dates)

        factor_betas, factor_returns = tool._estimate_factor_betas(returns)
        residual_cov = tool._compute_residual_cov(returns, factor_betas, factor_returns)

        assert residual_cov.shape == (2, 2)
        # 应该是正定的
        eigenvalues = np.linalg.eigvalsh(residual_cov)
        assert all(e >= 0 for e in eigenvalues)

    def test_residual_cov_dimensions(self, tool):
        """测试残差协方差维度"""
        np.random.seed(42)
        n_assets = 5
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            f"Asset_{i}": np.random.normal(0.001, 0.02, 100)
            for i in range(n_assets)
        }, index=dates)

        factor_betas, factor_returns = tool._estimate_factor_betas(returns)
        residual_cov = tool._compute_residual_cov(returns, factor_betas, factor_returns)

        assert residual_cov.shape == (n_assets, n_assets)


class TestGetFactorExposures:
    """因子敞口计算测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_get_exposures_basic(self, tool, sample_returns):
        """测试基本因子敞口计算"""
        weights = {"SPY": 0.4, "TLT": 0.3, "GLD": 0.3}
        exposures = tool.get_factor_exposures(sample_returns, weights)

        assert isinstance(exposures, dict)
        assert "market" in exposures

    def test_exposures_with_equal_weights(self, tool, sample_returns):
        """测试等权下的因子敞口"""
        weights = {"SPY": 1/3, "TLT": 1/3, "GLD": 1/3}
        exposures = tool.get_factor_exposures(sample_returns, weights)

        # 等权组合的市场因子敞口应该接近1
        assert abs(exposures["market"] - 1.0) < 0.5


class TestTargetFactorNeutralization:
    """目标因子中和测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

    def test_market_neutralization(self, tool):
        """测试市场因子中和"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(-0.001, 0.02, 100),
            "C": np.random.normal(0.0, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(
            returns,
            target_factors=["market"],
            factor_tolerance=0.2
        )

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

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
            "GLD": np.random.normal(0.0003, 0.015, 15),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 3

    def test_high_correlation_assets(self, tool):
        """测试高相关资产"""
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


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return FactorHedgingTool()

    def test_full_workflow(self, tool):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0004, 0.015, 252),
            "QQQ": np.random.normal(0.0005, 0.02, 252),
            "TLT": np.random.normal(0.0002, 0.008, 252),
            "GLD": np.random.normal(0.0001, 0.012, 252),
        }, index=dates)

        expert_views = {"SPY": 0.12, "QQQ": 0.15}
        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            target_factors=["market"],
            factor_tolerance=0.15,
            use_momentum=True,
            lookback_period=60
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

        weights1 = tool.compute_weights(returns)
        weights2 = tool.compute_weights(returns)

        for asset in weights1:
            assert abs(weights1[asset] - weights2[asset]) < 0.001

    def test_factor_exposures_after_optimization(self, tool):
        """测试优化后的因子敞口"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)
        exposures = tool.get_factor_exposures(returns, weights)

        assert isinstance(exposures, dict)
        assert "market" in exposures

"""
Deep tests for Regime-Switching Hedging
机制转换对冲策略深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.tools.regime_switching import RegimeSwitchingTool


class TestRegimeSwitchingInit:
    """RegimeSwitchingTool初始化测试"""

    def test_name_property(self):
        """测试名称属性"""
        tool = RegimeSwitchingTool()
        assert tool.name == "regime_switching"

    def test_description_property(self):
        """测试描述属性"""
        tool = RegimeSwitchingTool()
        assert "机制转换" in tool.description
        assert "Hamilton" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = RegimeSwitchingTool()
        params = tool.parameters
        assert "lookback_period" in params
        assert "n_regimes" in params
        assert "transition_cost" in params
        assert "smoothing_window" in params


class TestComputeWeightsBasic:
    """基本权重计算测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

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


class TestIdentifyRegime:
    """市场状态识别测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    def test_bull_market_identification(self, tool):
        """测试牛市识别"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        # 高收益、低波动 -> 牛市
        returns = pd.DataFrame({
            "A": np.random.normal(0.005, 0.01, 100),
            "B": np.random.normal(0.004, 0.008, 100),
        }, index=dates)

        regime, probs = tool._identify_regime(returns, lookback=60)

        assert regime in ["bull", "bear", "volatile"]
        assert "bull" in probs
        assert "bear" in probs
        assert "volatile" in probs
        assert abs(sum(probs.values()) - 1.0) < 0.1

    def test_bear_market_identification(self, tool):
        """测试熊市识别"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        # 负收益、高波动 -> 熊市
        returns = pd.DataFrame({
            "A": np.random.normal(-0.005, 0.04, 100),
            "B": np.random.normal(-0.004, 0.035, 100),
        }, index=dates)

        regime, probs = tool._identify_regime(returns, lookback=60)

        assert regime in ["bull", "bear", "volatile"]

    def test_volatile_market_identification(self, tool):
        """测试震荡市识别"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        # 低收益、中等波动 -> 震荡
        returns = pd.DataFrame({
            "A": np.random.normal(0.0, 0.025, 100),
            "B": np.random.normal(0.0, 0.02, 100),
        }, index=dates)

        regime, probs = tool._identify_regime(returns, lookback=60)

        assert regime in ["bull", "bear", "volatile"]

    def test_short_lookback(self, tool):
        """测试短回溯期"""
        dates = pd.date_range("2023-01-01", periods=20)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 20),
        }, index=dates)

        regime, probs = tool._identify_regime(returns, lookback=60)

        assert regime in ["bull", "bear", "volatile"]


class TestBullStrategy:
    """牛市策略测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.002, 0.015, 100),
            "QQQ": np.random.normal(0.003, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.008, 100),
        }, index=dates)

    def test_bull_strategy_basic(self, tool, sample_returns):
        """测试牛市策略基本计算"""
        weights = tool._bull_strategy(sample_returns, 0.0, 1.0, None)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_bull_strategy_with_constraints(self, tool, sample_returns):
        """测试带约束的牛市策略"""
        weights = tool._bull_strategy(sample_returns, 0.1, 0.5, None)

        for w in weights.values():
            assert w >= 0.09
            assert w <= 0.51

    def test_bull_strategy_with_views(self, tool, sample_returns):
        """测试带观点的牛市策略"""
        views = {"SPY": 0.2, "QQQ": 0.3}
        weights = tool._bull_strategy(sample_returns, 0.0, 1.0, views)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestBearStrategy:
    """熊市策略测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(-0.001, 0.025, 100),
            "TLT": np.random.normal(0.001, 0.01, 100),
            "GLD": np.random.normal(0.0005, 0.015, 100),
        }, index=dates)

    def test_bear_strategy_basic(self, tool, sample_returns):
        """测试熊市策略基本计算"""
        weights = tool._bear_strategy(sample_returns, 0.0, 1.0, None)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_bear_strategy_with_negative_views(self, tool, sample_returns):
        """测试带负面观点的熊市策略"""
        views = {"SPY": -0.2}  # 对SPY悲观
        weights = tool._bear_strategy(sample_returns, 0.0, 1.0, views)

        assert len(weights) == 3


class TestVolatileStrategy:
    """震荡市策略测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "A": np.random.normal(0.0, 0.02, 100),
            "B": np.random.normal(0.0, 0.03, 100),
            "C": np.random.normal(0.0, 0.01, 100),
        }, index=dates)

    def test_volatile_strategy_favors_low_vol(self, tool, sample_returns):
        """测试震荡策略偏向低波动资产"""
        weights = tool._volatile_strategy(sample_returns, 0.0, 1.0, None)

        # 低波动资产C应该有更高权重
        assert weights["C"] > weights["B"]

    def test_volatile_strategy_with_constraints(self, tool, sample_returns):
        """测试带约束的震荡策略"""
        weights = tool._volatile_strategy(sample_returns, 0.15, 0.5, None)

        for w in weights.values():
            assert w >= 0.14
            assert w <= 0.51


class TestEnsurePositiveDefinite:
    """正定矩阵处理测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    def test_positive_definite_matrix(self, tool):
        """测试已经正定的矩阵"""
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = tool._ensure_positive_definite(matrix)

        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)

    def test_non_positive_definite_matrix(self, tool):
        """测试非正定矩阵"""
        # 创建一个非正定矩阵
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = tool._ensure_positive_definite(matrix)

        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)


class TestGetRegimeAnalysis:
    """市场状态分析测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    def test_regime_analysis_structure(self, tool):
        """测试分析报告结构"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        analysis = tool.get_regime_analysis(returns, lookback=60)

        assert "current_regime" in analysis
        assert "regime_probabilities" in analysis
        assert "market_statistics" in analysis
        assert "recommended_strategy" in analysis

    def test_market_statistics(self, tool):
        """测试市场统计指标"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        analysis = tool.get_regime_analysis(returns)

        stats = analysis["market_statistics"]
        assert "annualized_return" in stats
        assert "annualized_volatility" in stats
        assert "sharpe_ratio" in stats
        assert "recent_trend" in stats


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

    def test_single_asset(self, tool):
        """测试单资产"""
        dates = pd.date_range("2023-01-01", periods=50)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 50)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 1
        assert abs(weights["SPY"] - 1.0) < 0.01

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


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return RegimeSwitchingTool()

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

        expert_views = {"SPY": 0.1, "TLT": -0.05}
        constraints = {"min_weight": 0.05, "max_single_asset": 0.4}

        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            lookback_period=60
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

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

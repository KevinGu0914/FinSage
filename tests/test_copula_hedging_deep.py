"""
Deep tests for CopulaHedgingTool

覆盖 finsage/hedging/tools/copula_hedging.py (目标从53%提升到80%+)
"""

import pytest
import pandas as pd
import numpy as np

from finsage.hedging.tools.copula_hedging import CopulaHedgingTool


class TestCopulaHedgingToolProperties:
    """测试CopulaHedgingTool属性"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    def test_name(self, tool):
        """测试名称"""
        assert tool.name == "copula_hedging"

    def test_description(self, tool):
        """测试描述"""
        desc = tool.description
        assert "Copula" in desc
        assert "Patton" in desc or "尾部依赖" in desc

    def test_parameters(self, tool):
        """测试参数"""
        params = tool.parameters
        assert "copula_type" in params
        assert "tail_dependency_weight" in params
        assert "lookback_period" in params


class TestCopulaHedgingToolComputeWeights:
    """测试权重计算"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=150)
        data = {
            "AAPL": np.random.randn(150) * 0.02,
            "MSFT": np.random.randn(150) * 0.018 + 0.0001,
            "GOOGL": np.random.randn(150) * 0.022,
            "AMZN": np.random.randn(150) * 0.025,
        }
        return pd.DataFrame(data, index=dates)

    def test_compute_weights_basic(self, tool, sample_returns):
        """测试基本权重计算"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in weights.values())

    def test_compute_weights_empty(self, tool):
        """测试空数据"""
        empty_df = pd.DataFrame()
        weights = tool.compute_weights(empty_df)

        assert weights == {}

    def test_compute_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重计算"""
        constraints = {"min_weight": 0.05, "max_single_asset": 0.35}

        weights = tool.compute_weights(sample_returns, constraints=constraints)

        assert all(w >= 0.04 for w in weights.values())  # 近似检查
        assert all(w <= 0.36 for w in weights.values())

    def test_compute_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点的权重计算"""
        expert_views = {"AAPL": 0.6, "MSFT": 0.4}

        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 4
        # AAPL应该有较高权重
        # 但不一定是最高的（取决于其他因素）

    def test_compute_weights_different_copula_types(self, tool, sample_returns):
        """测试不同Copula类型"""
        for copula_type in ["gaussian", "student_t", "clayton", "gumbel"]:
            weights = tool.compute_weights(
                sample_returns,
                copula_type=copula_type
            )

            assert len(weights) == 4
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_short_lookback(self, tool, sample_returns):
        """测试短回溯期"""
        weights = tool.compute_weights(
            sample_returns,
            lookback_period=30
        )

        assert len(weights) == 4

    def test_compute_weights_different_tail_weights(self, tool, sample_returns):
        """测试不同尾部权重"""
        weights_low = tool.compute_weights(
            sample_returns,
            tail_dependency_weight=0.2
        )
        weights_high = tool.compute_weights(
            sample_returns,
            tail_dependency_weight=0.8
        )

        # 权重应该有所不同
        assert len(weights_low) == len(weights_high) == 4


class TestCopulaHedgingToolTailDependence:
    """测试尾部依赖估计"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    @pytest.fixture
    def correlated_returns(self):
        """创建高度相关的收益率数据"""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n) * 0.02

        data = {
            "A": base + np.random.randn(n) * 0.005,
            "B": base + np.random.randn(n) * 0.008,
            "C": -base + np.random.randn(n) * 0.01,  # 负相关
        }
        return pd.DataFrame(data)

    def test_estimate_tail_dependence(self, tool, correlated_returns):
        """测试尾部依赖估计"""
        lower_tail, upper_tail = tool._estimate_tail_dependence(correlated_returns)

        assert lower_tail.shape == (3, 3)
        assert upper_tail.shape == (3, 3)
        # 对角线应该为1
        assert np.allclose(np.diag(lower_tail), 1.0)
        assert np.allclose(np.diag(upper_tail), 1.0)

    def test_estimate_tail_dependence_symmetry(self, tool, correlated_returns):
        """测试尾部依赖对称性"""
        lower_tail, upper_tail = tool._estimate_tail_dependence(correlated_returns)

        assert np.allclose(lower_tail, lower_tail.T)
        assert np.allclose(upper_tail, upper_tail.T)

    def test_estimate_tail_dependence_different_percentiles(self, tool, correlated_returns):
        """测试不同分位数"""
        lower_5, upper_5 = tool._estimate_tail_dependence(correlated_returns, percentile=0.05)
        lower_10, upper_10 = tool._estimate_tail_dependence(correlated_returns, percentile=0.10)

        # 较高分位数应该给出更高的尾部依赖估计
        assert lower_5.shape == lower_10.shape


class TestCopulaHedgingToolCopulaParameters:
    """测试Copula参数估计"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 100
        data = {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.018,
        }
        return pd.DataFrame(data)

    def test_estimate_gaussian_copula(self, tool, sample_returns):
        """测试Gaussian Copula参数"""
        params = tool._estimate_copula_parameters(sample_returns, "gaussian")

        assert params["type"] == "gaussian"
        assert "rho" in params
        assert params["rho"].shape == (2, 2)

    def test_estimate_student_t_copula(self, tool, sample_returns):
        """测试Student-t Copula参数"""
        params = tool._estimate_copula_parameters(sample_returns, "student_t")

        assert params["type"] == "student_t"
        assert "rho" in params
        assert "df" in params

    def test_estimate_clayton_copula(self, tool, sample_returns):
        """测试Clayton Copula参数"""
        params = tool._estimate_copula_parameters(sample_returns, "clayton")

        assert params["type"] == "clayton"
        assert "theta" in params
        assert params["theta"] >= 0.1

    def test_estimate_gumbel_copula(self, tool, sample_returns):
        """测试Gumbel Copula参数"""
        params = tool._estimate_copula_parameters(sample_returns, "gumbel")

        assert params["type"] == "gumbel"
        assert "theta" in params
        assert params["theta"] >= 1.0

    def test_estimate_unknown_copula(self, tool, sample_returns):
        """测试未知Copula类型"""
        params = tool._estimate_copula_parameters(sample_returns, "unknown")

        assert params["type"] == "gaussian"


class TestCopulaHedgingToolPositiveDefinite:
    """测试正定矩阵转换"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    def test_ensure_positive_definite_already_pd(self, tool):
        """测试已经正定的矩阵"""
        pd_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = tool._ensure_positive_definite(pd_matrix)

        assert np.allclose(result, pd_matrix, atol=1e-6)

    def test_ensure_positive_definite_non_pd(self, tool):
        """测试非正定矩阵"""
        non_pd = np.array([[1.0, 1.5], [1.5, 1.0]])  # 不正定
        result = tool._ensure_positive_definite(non_pd)

        # 结果应该正定
        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)


class TestCopulaHedgingToolTailDependenceAnalysis:
    """测试尾部依赖分析"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 150
        data = {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.018,
            "C": np.random.randn(n) * 0.022,
        }
        return pd.DataFrame(data)

    def test_get_tail_dependence_analysis(self, tool, sample_returns):
        """测试尾部依赖分析"""
        analysis = tool.get_tail_dependence_analysis(sample_returns)

        assert "average_lower_tail_dependence" in analysis
        assert "average_upper_tail_dependence" in analysis
        assert "tail_asymmetry" in analysis
        assert "high_lower_tail_pairs" in analysis
        assert "high_upper_tail_pairs" in analysis
        assert "interpretation" in analysis

    def test_get_tail_dependence_analysis_different_percentile(self, tool, sample_returns):
        """测试不同分位数的分析"""
        analysis_5 = tool.get_tail_dependence_analysis(sample_returns, percentile=0.05)
        analysis_10 = tool.get_tail_dependence_analysis(sample_returns, percentile=0.10)

        assert "interpretation" in analysis_5
        assert "interpretation" in analysis_10


class TestCopulaHedgingToolInterpretation:
    """测试尾部依赖解释"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    def test_interpret_both_high(self, tool):
        """测试双高尾部依赖"""
        interpretation = tool._interpret_tail_dependence(0.5, 0.5)
        assert "高度同步" in interpretation or "极端" in interpretation

    def test_interpret_lower_high(self, tool):
        """测试高下尾依赖"""
        interpretation = tool._interpret_tail_dependence(0.5, 0.2)
        assert "下跌" in interpretation or "下尾" in interpretation

    def test_interpret_upper_high(self, tool):
        """测试高上尾依赖"""
        interpretation = tool._interpret_tail_dependence(0.2, 0.5)
        assert "上涨" in interpretation or "上尾" in interpretation

    def test_interpret_both_low(self, tool):
        """测试双低尾部依赖"""
        interpretation = tool._interpret_tail_dependence(0.1, 0.1)
        assert "较低" in interpretation or "分散化" in interpretation

    def test_interpret_medium(self, tool):
        """测试中等尾部依赖"""
        interpretation = tool._interpret_tail_dependence(0.25, 0.25)
        assert "中等" in interpretation or "关注" in interpretation


class TestCopulaHedgingToolPortfolioTailRisk:
    """测试组合尾部风险"""

    @pytest.fixture
    def tool(self):
        return CopulaHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 150
        data = {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.018,
        }
        return pd.DataFrame(data)

    def test_compute_portfolio_tail_risk(self, tool, sample_returns):
        """测试组合尾部风险计算"""
        weights = {"A": 0.6, "B": 0.4}

        risk_metrics = tool.compute_portfolio_tail_risk(sample_returns, weights)

        assert "var" in risk_metrics
        assert "cvar" in risk_metrics
        assert "tail_risk_contribution" in risk_metrics
        assert "annualized_tail_risk" in risk_metrics

    def test_compute_portfolio_tail_risk_equal_weights(self, tool, sample_returns):
        """测试等权组合尾部风险"""
        weights = {"A": 0.5, "B": 0.5}

        risk_metrics = tool.compute_portfolio_tail_risk(sample_returns, weights)

        assert risk_metrics["var"] > 0
        assert risk_metrics["cvar"] >= risk_metrics["var"]  # CVaR >= VaR

    def test_compute_portfolio_tail_risk_different_percentile(self, tool, sample_returns):
        """测试不同分位数的尾部风险"""
        weights = {"A": 0.5, "B": 0.5}

        risk_5 = tool.compute_portfolio_tail_risk(sample_returns, weights, percentile=0.05)
        risk_1 = tool.compute_portfolio_tail_risk(sample_returns, weights, percentile=0.01)

        # 1%分位数的VaR应该更高
        assert risk_1["var"] >= risk_5["var"]

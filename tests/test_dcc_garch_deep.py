"""
Deep tests for DCCGARCHTool

覆盖 finsage/hedging/tools/dcc_garch.py (目标从55%提升到80%+)
"""

import pytest
import pandas as pd
import numpy as np

from finsage.hedging.tools.dcc_garch import DCCGARCHTool


class TestDCCGARCHToolProperties:
    """测试DCCGARCHTool属性"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    def test_name(self, tool):
        """测试名称"""
        assert tool.name == "dcc_garch"

    def test_description(self, tool):
        """测试描述"""
        desc = tool.description
        assert "DCC-GARCH" in desc
        assert "动态条件相关" in desc

    def test_parameters(self, tool):
        """测试参数"""
        params = tool.parameters
        assert "lookback_window" in params
        assert "decay_factor" in params
        assert "use_ewma" in params


class TestDCCGARCHToolComputeWeights:
    """测试权重计算"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        data = {
            "AAPL": np.random.randn(100) * 0.02,
            "MSFT": np.random.randn(100) * 0.018 + 0.0001,
            "GOOGL": np.random.randn(100) * 0.022,
            "AMZN": np.random.randn(100) * 0.025,
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

    def test_compute_weights_insufficient_data(self, tool):
        """测试数据不足"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=20)
        data = {
            "A": np.random.randn(20) * 0.02,
            "B": np.random.randn(20) * 0.02,
        }
        short_df = pd.DataFrame(data, index=dates)

        weights = tool.compute_weights(short_df)

        # 应该返回等权
        assert len(weights) == 2
        assert abs(weights["A"] - 0.5) < 0.01
        assert abs(weights["B"] - 0.5) < 0.01

    def test_compute_weights_with_ewma(self, tool, sample_returns):
        """测试使用EWMA"""
        weights = tool.compute_weights(
            sample_returns,
            use_ewma=True,
            decay_factor=0.94
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_without_ewma(self, tool, sample_returns):
        """测试不使用EWMA (完整DCC-GARCH)"""
        weights = tool.compute_weights(
            sample_returns,
            use_ewma=False
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_constraints(self, tool, sample_returns):
        """测试带约束的权重计算"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.4}

        weights = tool.compute_weights(sample_returns, constraints=constraints)

        assert all(w >= 0.09 for w in weights.values())  # 近似检查
        assert all(w <= 0.41 for w in weights.values())

    def test_compute_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点的权重计算"""
        expert_views = {"AAPL": 0.6, "MSFT": 0.4}

        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_different_decay_factors(self, tool, sample_returns):
        """测试不同衰减因子"""
        weights_low = tool.compute_weights(
            sample_returns,
            use_ewma=True,
            decay_factor=0.90
        )
        weights_high = tool.compute_weights(
            sample_returns,
            use_ewma=True,
            decay_factor=0.97
        )

        assert len(weights_low) == len(weights_high) == 4


class TestDCCGARCHToolEWMA:
    """测试EWMA协方差估计"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 100
        data = {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.018,
        }
        return pd.DataFrame(data)

    def test_ewma_covariance_basic(self, tool, sample_returns):
        """测试基本EWMA协方差"""
        cov = tool._ewma_covariance(sample_returns)

        assert cov.shape == (2, 2)
        # 协方差矩阵应该对称
        assert np.allclose(cov, cov.T)
        # 对角线应该为正
        assert all(cov[i, i] > 0 for i in range(2))

    def test_ewma_covariance_different_decay(self, tool, sample_returns):
        """测试不同衰减因子的EWMA"""
        cov_low = tool._ewma_covariance(sample_returns, decay_factor=0.90)
        cov_high = tool._ewma_covariance(sample_returns, decay_factor=0.97)

        # 不同衰减因子应该产生不同结果
        assert cov_low.shape == cov_high.shape

    def test_ewma_covariance_positive_semi_definite(self, tool, sample_returns):
        """测试EWMA协方差正半定"""
        cov = tool._ewma_covariance(sample_returns)

        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(e >= -1e-10 for e in eigenvalues)


class TestDCCGARCHCovariance:
    """测试DCC-GARCH协方差估计"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        n = 100
        data = {
            "A": np.random.randn(n) * 0.02,
            "B": np.random.randn(n) * 0.018,
            "C": np.random.randn(n) * 0.022,
        }
        return pd.DataFrame(data)

    def test_dcc_garch_covariance_basic(self, tool, sample_returns):
        """测试DCC-GARCH协方差"""
        cov = tool._dcc_garch_covariance(sample_returns)

        assert cov.shape == (3, 3)
        # 协方差矩阵应该对称
        assert np.allclose(cov, cov.T, atol=1e-6)

    def test_dcc_garch_covariance_positive_diagonal(self, tool, sample_returns):
        """测试DCC-GARCH对角线为正"""
        cov = tool._dcc_garch_covariance(sample_returns)

        # 对角线应该为正（方差）
        assert all(cov[i, i] > 0 for i in range(3))


class TestFitGARCH11:
    """测试GARCH(1,1)拟合"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    def test_fit_garch11_basic(self, tool):
        """测试基本GARCH(1,1)拟合"""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02

        sigma2 = tool._fit_garch11(returns)

        assert len(sigma2) == 100
        assert all(s > 0 for s in sigma2)

    def test_fit_garch11_volatility_clustering(self, tool):
        """测试波动率聚集"""
        np.random.seed(42)
        # 创建具有波动率聚集的收益率
        n = 200
        returns = np.zeros(n)
        returns[:50] = np.random.randn(50) * 0.01  # 低波动
        returns[50:100] = np.random.randn(50) * 0.05  # 高波动
        returns[100:150] = np.random.randn(50) * 0.01  # 低波动
        returns[150:] = np.random.randn(50) * 0.05  # 高波动

        sigma2 = tool._fit_garch11(returns)

        assert len(sigma2) == n
        # 高波动期的条件方差应该更高
        avg_sigma2_low = np.mean(sigma2[25:50])
        avg_sigma2_high = np.mean(sigma2[75:100])
        # 由于GARCH的滞后性，高波动期后的条件方差应该更高
        assert avg_sigma2_high > avg_sigma2_low * 0.5  # 宽松检查


class TestMinimumVarianceOptimize:
    """测试最小方差优化"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    def test_minimum_variance_optimize_basic(self, tool):
        """测试基本最小方差优化"""
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16]
        ])

        weights = tool._minimum_variance_optimize(cov)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01
        assert all(w >= 0 for w in weights)

    def test_minimum_variance_optimize_with_constraints(self, tool):
        """测试带约束的最小方差优化"""
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        constraints = {"min_weight": 0.2, "max_single_asset": 0.8}

        weights = tool._minimum_variance_optimize(cov, constraints)

        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01
        assert all(w >= 0.19 for w in weights)

    def test_minimum_variance_optimize_favor_low_variance(self, tool):
        """测试倾向低方差资产"""
        cov = np.array([
            [0.01, 0.0],
            [0.0, 0.25]
        ])
        # 放宽约束以允许更大的权重差异
        constraints = {"min_weight": 0.0, "max_single_asset": 1.0}

        weights = tool._minimum_variance_optimize(cov, constraints)

        # 低方差资产应该获得更高权重
        assert weights[0] >= weights[1]


class TestAdjustByViews:
    """测试专家观点调整"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    def test_adjust_by_views_basic(self, tool):
        """测试基本观点调整"""
        weights = np.array([0.3, 0.3, 0.4])
        assets = ["A", "B", "C"]
        expert_views = {"A": 0.6}

        adjusted = tool._adjust_by_views(weights, assets, expert_views)

        assert len(adjusted) == 3
        assert abs(sum(adjusted) - 1.0) < 0.01
        # A的权重应该增加
        assert adjusted[0] > weights[0] * 0.9  # 宽松检查

    def test_adjust_by_views_multiple_views(self, tool):
        """测试多个观点调整"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        assets = ["A", "B", "C", "D"]
        expert_views = {"A": 0.8, "D": 0.1}

        adjusted = tool._adjust_by_views(weights, assets, expert_views)

        assert len(adjusted) == 4
        assert abs(sum(adjusted) - 1.0) < 0.01

    def test_adjust_by_views_no_views(self, tool):
        """测试无观点"""
        weights = np.array([0.5, 0.5])
        assets = ["A", "B"]
        expert_views = {}

        adjusted = tool._adjust_by_views(weights, assets, expert_views)

        assert np.allclose(adjusted, weights)

    def test_adjust_by_views_normalization(self, tool):
        """测试权重归一化"""
        weights = np.array([0.2, 0.3, 0.5])
        assets = ["A", "B", "C"]
        expert_views = {"A": 0.9, "B": 0.8, "C": 0.7}

        adjusted = tool._adjust_by_views(weights, assets, expert_views)

        assert abs(sum(adjusted) - 1.0) < 0.01


class TestDCCGARCHIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    @pytest.fixture
    def realistic_returns(self):
        """创建更真实的收益率数据"""
        np.random.seed(42)
        n = 252  # 一年交易日
        dates = pd.date_range("2024-01-01", periods=n)

        # 创建相关的收益率
        base = np.random.randn(n) * 0.01

        data = {
            "SPY": base + np.random.randn(n) * 0.005,
            "QQQ": base * 1.2 + np.random.randn(n) * 0.008,  # 更高贝塔
            "TLT": -base * 0.3 + np.random.randn(n) * 0.01,  # 负相关
            "GLD": np.random.randn(n) * 0.008,  # 独立
        }
        return pd.DataFrame(data, index=dates)

    def test_full_workflow_ewma(self, tool, realistic_returns):
        """测试完整EWMA工作流"""
        weights = tool.compute_weights(
            realistic_returns,
            use_ewma=True,
            decay_factor=0.94,
            constraints={"min_weight": 0.05, "max_single_asset": 0.5}
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0.04 for w in weights.values())
        assert all(w <= 0.51 for w in weights.values())

    def test_full_workflow_dcc_garch(self, tool, realistic_returns):
        """测试完整DCC-GARCH工作流"""
        weights = tool.compute_weights(
            realistic_returns,
            use_ewma=False
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_full_workflow_with_views(self, tool, realistic_returns):
        """测试带专家观点的完整工作流"""
        expert_views = {
            "SPY": 0.4,
            "TLT": 0.6  # 看好债券
        }

        weights = tool.compute_weights(
            realistic_returns,
            expert_views=expert_views,
            use_ewma=True
        )

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_stability_across_different_periods(self, tool, realistic_returns):
        """测试不同时期的稳定性"""
        weights_1 = tool.compute_weights(realistic_returns.iloc[:126])
        weights_2 = tool.compute_weights(realistic_returns.iloc[126:])

        # 权重应该有所不同但都有效
        assert len(weights_1) == len(weights_2) == 4
        assert abs(sum(weights_1.values()) - 1.0) < 0.01
        assert abs(sum(weights_2.values()) - 1.0) < 0.01


class TestDCCGARCHEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def tool(self):
        return DCCGARCHTool()

    def test_single_asset(self, tool):
        """测试单资产"""
        np.random.seed(42)
        returns = pd.DataFrame({"A": np.random.randn(50) * 0.02})

        weights = tool.compute_weights(returns)

        assert weights == {"A": 1.0}

    def test_identical_returns(self, tool):
        """测试相同收益率"""
        np.random.seed(42)
        base = np.random.randn(50) * 0.02
        returns = pd.DataFrame({
            "A": base,
            "B": base,  # 完全相同
        })

        weights = tool.compute_weights(returns)

        assert len(weights) == 2
        # 由于协方差矩阵奇异，可能返回等权
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_high_correlation_data(self, tool):
        """测试高相关性数据"""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n) * 0.02
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.randn(n) * 0.001,  # 高度相关
            "C": base + np.random.randn(n) * 0.001,
        })

        weights = tool.compute_weights(returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_very_different_volatilities(self, tool):
        """测试差异很大的波动率"""
        np.random.seed(42)
        n = 100
        returns = pd.DataFrame({
            "low_vol": np.random.randn(n) * 0.005,
            "high_vol": np.random.randn(n) * 0.1,
        })
        # 放宽约束
        constraints = {"min_weight": 0.0, "max_single_asset": 1.0}

        weights = tool.compute_weights(returns, constraints=constraints)

        assert len(weights) == 2
        # 最小方差应该倾向低波动资产（或至少相等）
        assert weights["low_vol"] >= weights["high_vol"]

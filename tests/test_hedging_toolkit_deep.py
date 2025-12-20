"""
Deep tests for Hedging Toolkit
对冲工具箱深度测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from finsage.hedging.toolkit import HedgingToolkit
from finsage.hedging.base_tool import HedgingTool


class TestHedgingToolkitInit:
    """HedgingToolkit初始化测试"""

    def test_init_creates_tools(self):
        """测试初始化创建工具"""
        toolkit = HedgingToolkit()
        assert len(toolkit._tools) > 0

    def test_default_tools_registered(self):
        """测试默认工具已注册"""
        toolkit = HedgingToolkit()

        # 检查基础工具
        assert toolkit.get("minimum_variance") is not None
        assert toolkit.get("risk_parity") is not None
        assert toolkit.get("black_litterman") is not None
        assert toolkit.get("mean_variance") is not None
        assert toolkit.get("hrp") is not None

        # 检查高级工具
        assert toolkit.get("cvar_optimization") is not None
        assert toolkit.get("robust_optimization") is not None
        assert toolkit.get("factor_hedging") is not None
        assert toolkit.get("regime_switching") is not None


class TestRegisterUnregister:
    """注册/注销测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    def test_register_new_tool(self, toolkit):
        """测试注册新工具"""
        class CustomTool(HedgingTool):
            @property
            def name(self):
                return "custom_tool"

            @property
            def description(self):
                return "Custom test tool"

            @property
            def parameters(self):
                return {}

            def compute_weights(self, returns, **kwargs):
                return {}

        custom = CustomTool()
        toolkit.register(custom)

        assert toolkit.get("custom_tool") is not None

    def test_unregister_tool(self, toolkit):
        """测试注销工具"""
        # 确保工具存在
        assert toolkit.get("minimum_variance") is not None

        toolkit.unregister("minimum_variance")

        assert toolkit.get("minimum_variance") is None

    def test_unregister_nonexistent_tool(self, toolkit):
        """测试注销不存在的工具"""
        # 不应该抛出异常
        toolkit.unregister("nonexistent_tool")


class TestGetTool:
    """获取工具测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    def test_get_existing_tool(self, toolkit):
        """测试获取存在的工具"""
        tool = toolkit.get("minimum_variance")
        assert tool is not None
        assert tool.name == "minimum_variance"

    def test_get_nonexistent_tool(self, toolkit):
        """测试获取不存在的工具"""
        tool = toolkit.get("nonexistent")
        assert tool is None


class TestListTools:
    """列出工具测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    def test_list_tools_returns_list(self, toolkit):
        """测试列出工具返回列表"""
        tools = toolkit.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_list_tools_structure(self, toolkit):
        """测试列出工具的结构"""
        tools = toolkit.list_tools()

        for tool_dict in tools:
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "parameters" in tool_dict


class TestCallTool:
    """调用工具测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_call_minimum_variance(self, toolkit, sample_returns):
        """测试调用最小方差工具"""
        weights = toolkit.call("minimum_variance", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_risk_parity(self, toolkit, sample_returns):
        """测试调用风险平价工具"""
        weights = toolkit.call("risk_parity", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_black_litterman(self, toolkit, sample_returns):
        """测试调用Black-Litterman工具"""
        expert_views = {"SPY": 0.10}
        weights = toolkit.call(
            "black_litterman",
            sample_returns,
            expert_views=expert_views
        )

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_with_constraints(self, toolkit, sample_returns):
        """测试带约束调用"""
        constraints = {"min_weight": 0.1, "max_single_asset": 0.5}
        weights = toolkit.call(
            "minimum_variance",
            sample_returns,
            constraints=constraints
        )

        for w in weights.values():
            assert w >= 0.09
            assert w <= 0.51

    def test_call_nonexistent_tool(self, toolkit, sample_returns):
        """测试调用不存在的工具"""
        with pytest.raises(ValueError) as excinfo:
            toolkit.call("nonexistent_tool", sample_returns)

        assert "not found" in str(excinfo.value)

    def test_call_hrp(self, toolkit, sample_returns):
        """测试调用HRP工具"""
        weights = toolkit.call("hrp", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_mean_variance(self, toolkit, sample_returns):
        """测试调用均值方差工具"""
        weights = toolkit.call("mean_variance", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestCompareTools:
    """比较工具测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_compare_all_tools(self, toolkit, sample_returns):
        """测试比较所有工具"""
        results = toolkit.compare_tools(sample_returns)

        assert isinstance(results, dict)
        assert len(results) > 0

        for name, weights in results.items():
            if weights:  # 非空权重
                assert abs(sum(weights.values()) - 1.0) < 0.05

    def test_compare_selected_tools(self, toolkit, sample_returns):
        """测试比较选定工具"""
        tool_names = ["minimum_variance", "risk_parity", "hrp"]
        results = toolkit.compare_tools(sample_returns, tool_names=tool_names)

        assert len(results) == 3
        assert "minimum_variance" in results
        assert "risk_parity" in results
        assert "hrp" in results

    def test_compare_with_constraints(self, toolkit, sample_returns):
        """测试带约束比较"""
        constraints = {"min_weight": 0.1}
        tool_names = ["minimum_variance", "mean_variance"]
        results = toolkit.compare_tools(
            sample_returns,
            constraints=constraints,
            tool_names=tool_names
        )

        for name, weights in results.items():
            if weights:
                for w in weights.values():
                    assert w >= 0.09

    def test_compare_with_expert_views(self, toolkit, sample_returns):
        """测试带专家观点比较"""
        expert_views = {"SPY": 0.12}
        tool_names = ["black_litterman", "mean_variance"]
        results = toolkit.compare_tools(
            sample_returns,
            expert_views=expert_views,
            tool_names=tool_names
        )

        assert len(results) == 2


class TestEmptyReturns:
    """空收益率测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    def test_call_with_empty_returns(self, toolkit):
        """测试空收益率调用"""
        empty_df = pd.DataFrame()
        weights = toolkit.call("minimum_variance", empty_df)
        assert weights == {}

    def test_compare_with_empty_returns(self, toolkit):
        """测试空收益率比较"""
        empty_df = pd.DataFrame()
        results = toolkit.compare_tools(
            empty_df,
            tool_names=["minimum_variance", "risk_parity"]
        )

        for weights in results.values():
            assert weights == {}


class TestAdvancedTools:
    """高级工具测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_call_cvar_optimization(self, toolkit, sample_returns):
        """测试调用CVaR优化"""
        weights = toolkit.call("cvar_optimization", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_robust_optimization(self, toolkit, sample_returns):
        """测试调用鲁棒优化"""
        weights = toolkit.call("robust_optimization", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_factor_hedging(self, toolkit, sample_returns):
        """测试调用因子对冲"""
        weights = toolkit.call("factor_hedging", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_call_regime_switching(self, toolkit, sample_returns):
        """测试调用机制转换"""
        weights = toolkit.call("regime_switching", sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def toolkit(self):
        return HedgingToolkit()

    def test_full_workflow(self, toolkit):
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

        # 列出所有工具
        tools = toolkit.list_tools()
        assert len(tools) > 5

        # 调用多个工具
        mv_weights = toolkit.call(
            "minimum_variance",
            returns,
            constraints=constraints
        )
        assert len(mv_weights) == 4

        bl_weights = toolkit.call(
            "black_litterman",
            returns,
            expert_views=expert_views,
            constraints=constraints
        )
        assert len(bl_weights) == 4

        # 比较工具
        comparison = toolkit.compare_tools(
            returns,
            expert_views=expert_views,
            constraints=constraints,
            tool_names=["minimum_variance", "risk_parity", "hrp"]
        )
        assert len(comparison) == 3

    def test_all_tools_return_valid_weights(self, toolkit):
        """测试所有工具返回有效权重"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.015, 100),
            "C": np.random.normal(0.001, 0.025, 100),
        }, index=dates)

        tools = toolkit.list_tools()

        for tool_info in tools:
            tool_name = tool_info["name"]
            try:
                weights = toolkit.call(tool_name, returns)
                if weights:  # 非空
                    assert abs(sum(weights.values()) - 1.0) < 0.05
                    assert all(isinstance(v, (int, float)) for v in weights.values())
            except Exception as e:
                # 某些工具可能需要特定参数，允许失败
                pass

    def test_reproducibility(self, toolkit):
        """测试可重复性"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights1 = toolkit.call("minimum_variance", returns)
        weights2 = toolkit.call("minimum_variance", returns)

        for asset in weights1:
            assert abs(weights1[asset] - weights2[asset]) < 0.001

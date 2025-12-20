"""
Deep tests for Base Hedging Tool
对冲工具基类深度测试
"""

import pytest
import pandas as pd
import numpy as np
from abc import ABC

from finsage.hedging.base_tool import HedgingTool


class ConcreteHedgingTool(HedgingTool):
    """用于测试的具体对冲工具实现"""

    @property
    def name(self) -> str:
        return "test_hedging_tool"

    @property
    def description(self) -> str:
        return "Test hedging tool for unit tests"

    @property
    def parameters(self):
        return {
            "param1": "test parameter 1",
            "param2": "test parameter 2",
        }

    def compute_weights(self, returns, expert_views=None, constraints=None, **kwargs):
        if returns.empty:
            return {}
        n = len(returns.columns)
        return {col: 1.0 / n for col in returns.columns}


class TestHedgingToolAbstract:
    """HedgingTool抽象类测试"""

    def test_is_abstract_class(self):
        """测试是抽象类"""
        assert issubclass(HedgingTool, ABC)

    def test_cannot_instantiate_directly(self):
        """测试不能直接实例化"""
        with pytest.raises(TypeError):
            HedgingTool()


class TestConcreteHedgingToolInit:
    """具体对冲工具初始化测试"""

    def test_can_instantiate_concrete(self):
        """测试可以实例化具体类"""
        tool = ConcreteHedgingTool()
        assert tool is not None

    def test_name_property(self):
        """测试名称属性"""
        tool = ConcreteHedgingTool()
        assert tool.name == "test_hedging_tool"

    def test_description_property(self):
        """测试描述属性"""
        tool = ConcreteHedgingTool()
        assert "Test hedging tool" in tool.description

    def test_parameters_property(self):
        """测试参数属性"""
        tool = ConcreteHedgingTool()
        params = tool.parameters
        assert "param1" in params
        assert "param2" in params


class TestComputeWeights:
    """权重计算测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        return pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

    def test_compute_weights_basic(self, tool, sample_returns):
        """测试基本权重计算"""
        weights = tool.compute_weights(sample_returns)

        assert len(weights) == 3
        assert "SPY" in weights
        assert "TLT" in weights
        assert "GLD" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_empty(self, tool):
        """测试空收益率"""
        empty_df = pd.DataFrame()
        weights = tool.compute_weights(empty_df)

        assert weights == {}

    def test_compute_weights_with_expert_views(self, tool, sample_returns):
        """测试带专家观点"""
        expert_views = {"SPY": 0.10, "TLT": 0.05}
        weights = tool.compute_weights(sample_returns, expert_views=expert_views)

        assert len(weights) == 3

    def test_compute_weights_with_constraints(self, tool, sample_returns):
        """测试带约束"""
        constraints = {"min_weight": 0.1, "max_weight": 0.5}
        weights = tool.compute_weights(sample_returns, constraints=constraints)

        assert len(weights) == 3


class TestValidateWeights:
    """权重验证测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    def test_validate_normal_weights(self, tool):
        """测试正常权重验证"""
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        validated = tool.validate_weights(weights)

        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_unnormalized_weights(self, tool):
        """测试归一化未归一的权重"""
        weights = {"A": 0.4, "B": 0.3, "C": 0.1}  # 总和0.8
        validated = tool.validate_weights(weights)

        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_negative_weights(self, tool):
        """测试处理负权重"""
        weights = {"A": 0.5, "B": -0.1, "C": 0.6}
        validated = tool.validate_weights(weights)

        assert all(v >= 0 for v in validated.values())
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_all_zero_weights(self, tool):
        """测试处理全零权重"""
        weights = {"A": 0, "B": 0, "C": 0}
        validated = tool.validate_weights(weights)

        # 全零时总和为0，无法归一化
        assert sum(validated.values()) == 0

    def test_validate_all_negative_weights(self, tool):
        """测试处理全负权重"""
        weights = {"A": -0.3, "B": -0.4, "C": -0.3}
        validated = tool.validate_weights(weights)

        # 全负变为全零
        assert all(v == 0 for v in validated.values())

    def test_validate_empty_weights(self, tool):
        """测试处理空权重"""
        weights = {}
        validated = tool.validate_weights(weights)

        assert validated == {}

    def test_validate_large_weights(self, tool):
        """测试处理大权重"""
        weights = {"A": 10, "B": 20, "C": 30}
        validated = tool.validate_weights(weights)

        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_validate_small_weights(self, tool):
        """测试处理小权重"""
        weights = {"A": 0.001, "B": 0.002, "C": 0.003}
        validated = tool.validate_weights(weights)

        assert abs(sum(validated.values()) - 1.0) < 0.001


class TestToDict:
    """转换为字典测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    def test_to_dict_structure(self, tool):
        """测试字典结构"""
        result = tool.to_dict()

        assert "name" in result
        assert "description" in result
        assert "parameters" in result

    def test_to_dict_values(self, tool):
        """测试字典值"""
        result = tool.to_dict()

        assert result["name"] == "test_hedging_tool"
        assert "Test hedging tool" in result["description"]
        assert "param1" in result["parameters"]


class TestDefaultParameters:
    """默认参数测试"""

    def test_default_parameters_empty(self):
        """测试默认参数为空"""
        # 创建一个不覆盖parameters的具体类
        class MinimalTool(HedgingTool):
            @property
            def name(self):
                return "minimal"

            @property
            def description(self):
                return "minimal tool"

            def compute_weights(self, returns, expert_views=None, constraints=None, **kwargs):
                return {}

        tool = MinimalTool()
        assert tool.parameters == {}


class TestMultipleAssets:
    """多资产测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    def test_two_assets(self, tool):
        """测试两资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_many_assets(self, tool):
        """测试多资产"""
        np.random.seed(42)
        n_assets = 10
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            f"Asset_{i}": np.random.normal(0.001, 0.02, 100)
            for i in range(n_assets)
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == n_assets
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_single_asset(self, tool):
        """测试单资产"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert len(weights) == 1
        assert abs(weights["SPY"] - 1.0) < 0.01


class TestKwargs:
    """额外参数测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    def test_compute_weights_with_kwargs(self, tool):
        """测试带额外参数"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(
            returns,
            extra_param1="value1",
            extra_param2=123
        )

        assert len(weights) == 2


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tool(self):
        return ConcreteHedgingTool()

    def test_full_workflow(self, tool):
        """测试完整工作流"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        expert_views = {"SPY": 0.10, "TLT": 0.05}
        constraints = {"min_weight": 0.1}

        # 计算权重
        weights = tool.compute_weights(
            returns,
            expert_views=expert_views,
            constraints=constraints
        )

        # 验证权重
        validated = tool.validate_weights(weights)

        # 转换为字典
        info = tool.to_dict()

        assert len(validated) == 3
        assert abs(sum(validated.values()) - 1.0) < 0.01
        assert "name" in info

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

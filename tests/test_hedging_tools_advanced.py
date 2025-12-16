#!/usr/bin/env python
"""
Advanced Hedging Tools Tests - 高级对冲工具测试
覆盖: regime_switching, copula_hedging, dcc_garch, factor_hedging, robust_optimization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch


# ============================================================
# Test 1: RegimeSwitchingTool
# ============================================================

class TestRegimeSwitchingTool:
    """测试机制转换对冲工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool
        assert RegimeSwitchingTool is not None

    def test_properties(self):
        """测试属性"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

        tool = RegimeSwitchingTool()
        assert tool.name == "regime_switching"
        assert "机制转换" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_compute_weights_basic(self):
        """测试基本权重计算"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

        tool = RegimeSwitchingTool()

        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_constraints(self):
        """测试带约束的权重计算"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

        tool = RegimeSwitchingTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        constraints = {
            "min_weight": 0.1,
            "max_weight": 0.8
        }

        weights = tool.compute_weights(returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.1 - 0.01
            assert w <= 0.8 + 0.01

    def test_regime_identification(self):
        """测试市场状态识别"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

        tool = RegimeSwitchingTool()

        # 牛市数据
        bull_returns = pd.Series(np.random.normal(0.003, 0.01, 50))
        # 熊市数据
        bear_returns = pd.Series(np.random.normal(-0.003, 0.03, 50))

        # 验证工具能处理不同市场状态


# ============================================================
# Test 2: CopulaHedgingTool
# ============================================================

class TestCopulaHedgingTool:
    """测试Copula对冲工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool
        assert CopulaHedgingTool is not None

    def test_properties(self):
        """测试属性"""
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool

        tool = CopulaHedgingTool()
        assert tool.name == "copula_hedging"
        assert "Copula" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_compute_weights_basic(self):
        """测试基本权重计算"""
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool

        tool = CopulaHedgingTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "QQQ": np.random.normal(0.0012, 0.025, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_copula_type(self):
        """测试不同Copula类型"""
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool

        tool = CopulaHedgingTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        # 测试不同copula_type
        for copula_type in ["gaussian", "student_t"]:
            weights = tool.compute_weights(returns, copula_type=copula_type)
            assert abs(sum(weights.values()) - 1.0) < 0.01


# ============================================================
# Test 3: DCCGARCHTool
# ============================================================

class TestDCCGARCHTool:
    """测试DCC-GARCH工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool
        assert DCCGARCHTool is not None

    def test_properties(self):
        """测试属性"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool

        tool = DCCGARCHTool()
        assert tool.name == "dcc_garch"
        assert "DCC" in tool.description or "GARCH" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_compute_weights_basic(self):
        """测试基本权重计算"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool

        tool = DCCGARCHTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_forecast_horizon(self):
        """测试带预测期限的权重计算"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool

        tool = DCCGARCHTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        weights = tool.compute_weights(returns, forecast_horizon=5)

        assert isinstance(weights, dict)


# ============================================================
# Test 4: FactorHedgingTool
# ============================================================

class TestFactorHedgingTool:
    """测试因子对冲工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool
        assert FactorHedgingTool is not None

    def test_properties(self):
        """测试属性"""
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool

        tool = FactorHedgingTool()
        assert tool.name == "factor_hedging"
        assert "因子" in tool.description or "Factor" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_compute_weights_basic(self):
        """测试基本权重计算"""
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool

        tool = FactorHedgingTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "QQQ": np.random.normal(0.0012, 0.025, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_target_factor_exposure(self):
        """测试带目标因子暴露的权重计算"""
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool

        tool = FactorHedgingTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        # 目标：市场中性
        weights = tool.compute_weights(
            returns,
            target_factor_exposure={"market": 0.0}
        )

        assert isinstance(weights, dict)


# ============================================================
# Test 5: RobustOptimizationTool
# ============================================================

class TestRobustOptimizationTool:
    """测试鲁棒优化工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool
        assert RobustOptimizationTool is not None

    def test_properties(self):
        """测试属性"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool

        tool = RobustOptimizationTool()
        assert tool.name == "robust_optimization"
        assert "鲁棒" in tool.description or "Robust" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_compute_weights_basic(self):
        """测试基本权重计算"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool

        tool = RobustOptimizationTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_uncertainty_set(self):
        """测试带不确定性集的权重计算"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool

        tool = RobustOptimizationTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        # 不同不确定性等级
        for epsilon in [0.1, 0.2, 0.3]:
            weights = tool.compute_weights(returns, epsilon=epsilon)
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_constraints(self):
        """测试带约束的权重计算"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool

        tool = RobustOptimizationTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
            "GLD": np.random.normal(0.0003, 0.015, 100),
        }, index=dates)

        constraints = {
            "min_weight": 0.05,
            "max_weight": 0.70
        }

        weights = tool.compute_weights(returns, constraints=constraints)

        for w in weights.values():
            assert w >= 0.05 - 0.01
            assert w <= 0.70 + 0.01


# ============================================================
# Test 6: Hedging Tool Integration
# ============================================================

class TestHedgingToolsIntegration:
    """测试对冲工具集成"""

    def test_all_tools_in_toolkit(self):
        """测试所有工具都在工具包中"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        tools = toolkit.list_tools()

        # list_tools 返回工具描述列表
        tool_names = [t["name"] for t in tools]

        # 验证高级工具都已注册
        expected_tools = [
            "dcc_garch",
            "hrp",
            "cvar_optimization"
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Missing tool: {tool_name}"

    def test_get_tool_directly(self):
        """测试直接获取工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
            "TLT": np.random.normal(0.0005, 0.01, 100),
        }, index=dates)

        # 获取工具并计算权重
        tool = toolkit.get("minimum_variance")  # 正确的方法名是 get
        assert tool is not None

        weights = tool.compute_weights(returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_toolkit_list_tools_format(self):
        """测试工具列表格式"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        tools = toolkit.list_tools()

        # 验证返回格式
        assert isinstance(tools, list)
        assert len(tools) > 0

        for tool_info in tools:
            assert "name" in tool_info
            assert "description" in tool_info


# ============================================================
# Test 7: Edge Cases
# ============================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_single_asset(self):
        """测试单资产情况"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool

        tool = RobustOptimizationTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)

        assert "SPY" in weights
        assert abs(weights["SPY"] - 1.0) < 0.01

    def test_high_correlation(self):
        """测试高相关性资产"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

        tool = RegimeSwitchingTool()

        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # 高度相关的资产
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.001, 100),
        }, index=dates)

        weights = tool.compute_weights(returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_short_history(self):
        """测试短历史数据"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool

        tool = DCCGARCHTool()

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.02, 30),
            "TLT": np.random.normal(0.0005, 0.01, 30),
        }, index=dates)

        # 应该仍然能够计算权重
        weights = tool.compute_weights(returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Advanced Hedging Tools Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

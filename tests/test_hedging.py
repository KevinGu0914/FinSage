#!/usr/bin/env python
"""
Hedging Module Tests - 对冲模块测试
覆盖: base_tool, toolkit, risk_parity, cvar_optimization, minimum_variance,
      black_litterman, mean_variance, dcc_garch, hrp, robust_optimization,
      factor_hedging, regime_switching, copula_hedging, dynamic_selector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_returns():
    """生成示例收益率数据"""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)

    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.012, len(dates)),
        'QQQ': np.random.normal(0.0006, 0.015, len(dates)),
        'TLT': np.random.normal(0.0002, 0.007, len(dates)),
        'GLD': np.random.normal(0.0003, 0.010, len(dates)),
        'VIX': np.random.normal(-0.0002, 0.030, len(dates)),
        'IWM': np.random.normal(0.0004, 0.014, len(dates)),
    }, index=dates)

    return returns


@pytest.fixture
def sample_expert_views():
    """生成示例专家观点"""
    return {
        'SPY': 0.25,
        'TLT': 0.35,
        'GLD': 0.25,
        'VIX': 0.05,
    }


@pytest.fixture
def sample_constraints():
    """生成示例约束条件"""
    return {
        'min_weight': 0.05,
        'max_single_asset': 0.30,
        'max_volatility': 0.15,
    }


# ============================================================
# Test 1: Base Hedging Tool
# ============================================================

class TestHedgingTool:
    """测试HedgingTool基类"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.base_tool import HedgingTool
        assert HedgingTool is not None

    def test_abstract_methods(self):
        """测试抽象方法"""
        from finsage.hedging.base_tool import HedgingTool

        # 应该无法直接实例化抽象类
        with pytest.raises(TypeError):
            HedgingTool()

    def test_validate_weights_normalization(self):
        """测试权重验证和归一化"""
        from finsage.hedging.base_tool import HedgingTool

        class MockTool(HedgingTool):
            @property
            def name(self): return "mock"
            @property
            def description(self): return "mock tool"
            def compute_weights(self, *args, **kwargs): return {}

        tool = MockTool()

        # 测试归一化
        weights = {'A': 0.3, 'B': 0.4, 'C': 0.5}
        normalized = tool.validate_weights(weights)

        assert abs(sum(normalized.values()) - 1.0) < 0.0001
        assert all(v >= 0 for v in normalized.values())

    def test_validate_weights_negative(self):
        """测试负权重处理"""
        from finsage.hedging.base_tool import HedgingTool

        class MockTool(HedgingTool):
            @property
            def name(self): return "mock"
            @property
            def description(self): return "mock tool"
            def compute_weights(self, *args, **kwargs): return {}

        tool = MockTool()

        weights = {'A': -0.1, 'B': 0.6, 'C': 0.5}
        normalized = tool.validate_weights(weights)

        assert normalized['A'] == 0
        assert abs(sum(normalized.values()) - 1.0) < 0.0001

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.hedging.base_tool import HedgingTool

        class MockTool(HedgingTool):
            @property
            def name(self): return "mock_tool"
            @property
            def description(self): return "A mock hedging tool"
            @property
            def parameters(self): return {"param1": "desc1"}
            def compute_weights(self, *args, **kwargs): return {}

        tool = MockTool()
        d = tool.to_dict()

        assert d['name'] == 'mock_tool'
        assert d['description'] == 'A mock hedging tool'
        assert 'param1' in d['parameters']


# ============================================================
# Test 2: Hedging Toolkit
# ============================================================

class TestHedgingToolkit:
    """测试对冲工具箱"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.toolkit import HedgingToolkit
        assert HedgingToolkit is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        assert toolkit is not None
        assert len(toolkit._tools) >= 6  # 至少6个默认工具

    def test_list_tools(self):
        """测试列出所有工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        tools = toolkit.list_tools()

        assert len(tools) >= 6
        assert all('name' in t for t in tools)
        assert all('description' in t for t in tools)

    def test_get_tool(self):
        """测试获取工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()

        tool = toolkit.get('risk_parity')
        assert tool is not None
        assert tool.name == 'risk_parity'

    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        tool = toolkit.get('nonexistent')

        assert tool is None

    def test_register_tool(self):
        """测试注册工具"""
        from finsage.hedging.toolkit import HedgingToolkit
        from finsage.hedging.base_tool import HedgingTool

        class CustomTool(HedgingTool):
            @property
            def name(self): return "custom"
            @property
            def description(self): return "custom tool"
            def compute_weights(self, *args, **kwargs): return {}

        toolkit = HedgingToolkit()
        toolkit.register(CustomTool())

        assert 'custom' in toolkit._tools
        assert toolkit.get('custom') is not None

    def test_unregister_tool(self):
        """测试注销工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        initial_count = len(toolkit._tools)

        toolkit.unregister('risk_parity')
        assert len(toolkit._tools) == initial_count - 1
        assert toolkit.get('risk_parity') is None

    def test_call_tool(self, sample_returns, sample_expert_views, sample_constraints):
        """测试调用工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        weights = toolkit.call(
            tool_name='risk_parity',
            returns=sample_returns,
            expert_views=sample_expert_views,
            constraints=sample_constraints
        )

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.0001

    def test_call_invalid_tool(self, sample_returns):
        """测试调用无效工具"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()

        with pytest.raises(ValueError) as exc_info:
            toolkit.call(tool_name='invalid', returns=sample_returns)

        assert 'not found' in str(exc_info.value)

    def test_compare_tools(self, sample_returns):
        """测试工具比较"""
        from finsage.hedging.toolkit import HedgingToolkit

        toolkit = HedgingToolkit()
        results = toolkit.compare_tools(
            returns=sample_returns,
            tool_names=['risk_parity', 'minimum_variance']
        )

        assert 'risk_parity' in results
        assert 'minimum_variance' in results
        assert all(isinstance(v, dict) for v in results.values())


# ============================================================
# Test 3: Risk Parity Tool
# ============================================================

class TestRiskParityTool:
    """测试风险平价工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.risk_parity import RiskParityTool
        assert RiskParityTool is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.hedging.tools.risk_parity import RiskParityTool

        tool = RiskParityTool()
        assert tool.name == 'risk_parity'
        assert '风险平价' in tool.description

    def test_compute_weights(self, sample_returns):
        """测试权重计算"""
        from finsage.hedging.tools.risk_parity import RiskParityTool

        tool = RiskParityTool()
        weights = tool.compute_weights(sample_returns)

        assert isinstance(weights, dict)
        assert len(weights) == len(sample_returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_with_expert_views(self, sample_returns, sample_expert_views):
        """测试带专家观点的权重计算"""
        from finsage.hedging.tools.risk_parity import RiskParityTool

        tool = RiskParityTool()
        weights = tool.compute_weights(sample_returns, expert_views=sample_expert_views)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_weights_empty_returns(self):
        """测试空收益率"""
        from finsage.hedging.tools.risk_parity import RiskParityTool

        tool = RiskParityTool()
        weights = tool.compute_weights(pd.DataFrame())

        assert weights == {}


# ============================================================
# Test 4: CVaR Optimization Tool
# ============================================================

class TestCVaROptimizationTool:
    """测试CVaR优化工具"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool
        assert CVaROptimizationTool is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool

        tool = CVaROptimizationTool()
        assert tool.name == 'cvar_optimization'
        assert 'CVaR' in tool.description

    def test_parameters(self):
        """测试参数说明"""
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool

        tool = CVaROptimizationTool()
        params = tool.parameters

        assert 'alpha' in params
        assert 'min_weight' in params

    def test_compute_weights(self, sample_returns):
        """测试权重计算"""
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool

        tool = CVaROptimizationTool()
        weights = tool.compute_weights(sample_returns, alpha=0.95)

        assert isinstance(weights, dict)
        assert len(weights) == len(sample_returns.columns)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_compute_portfolio_cvar(self, sample_returns):
        """测试组合CVaR计算"""
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool

        tool = CVaROptimizationTool()
        weights = {'SPY': 0.3, 'QQQ': 0.3, 'TLT': 0.2, 'GLD': 0.2}

        cvar = tool.compute_portfolio_cvar(sample_returns, weights, alpha=0.95)

        assert isinstance(cvar, float)
        assert cvar >= 0  # CVaR表示损失，应为正


# ============================================================
# Test 5: Other Hedging Tools (Import Tests)
# ============================================================

class TestOtherHedgingTools:
    """测试其他对冲工具导入"""

    def test_minimum_variance_import(self):
        """测试最小方差工具"""
        from finsage.hedging.tools.minimum_variance import MinimumVarianceTool
        tool = MinimumVarianceTool()
        assert tool.name == 'minimum_variance'

    def test_black_litterman_import(self):
        """测试Black-Litterman工具"""
        from finsage.hedging.tools.black_litterman import BlackLittermanTool
        tool = BlackLittermanTool()
        assert tool.name == 'black_litterman'

    def test_mean_variance_import(self):
        """测试均值方差工具"""
        from finsage.hedging.tools.mean_variance import MeanVarianceTool
        tool = MeanVarianceTool()
        assert tool.name == 'mean_variance'

    def test_dcc_garch_import(self):
        """测试DCC-GARCH工具"""
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool
        tool = DCCGARCHTool()
        assert tool.name == 'dcc_garch'

    def test_hrp_import(self):
        """测试HRP工具"""
        from finsage.hedging.tools.hrp import HierarchicalRiskParityTool
        tool = HierarchicalRiskParityTool()
        assert tool.name == 'hrp'

    def test_robust_optimization_import(self):
        """测试稳健优化工具"""
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool
        tool = RobustOptimizationTool()
        assert tool.name == 'robust_optimization'

    def test_factor_hedging_import(self):
        """测试因子对冲工具"""
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool
        tool = FactorHedgingTool()
        assert tool.name == 'factor_hedging'

    def test_regime_switching_import(self):
        """测试状态转换工具"""
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool
        tool = RegimeSwitchingTool()
        assert tool.name == 'regime_switching'

    def test_copula_hedging_import(self):
        """测试Copula对冲工具"""
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool
        tool = CopulaHedgingTool()
        assert tool.name == 'copula_hedging'


# ============================================================
# Test 6: Hedging Tools Compute Weights
# ============================================================

class TestHedgingToolsComputeWeights:
    """测试各对冲工具的权重计算"""

    def test_minimum_variance_weights(self, sample_returns):
        """测试最小方差权重"""
        from finsage.hedging.tools.minimum_variance import MinimumVarianceTool

        tool = MinimumVarianceTool()
        weights = tool.compute_weights(sample_returns)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_black_litterman_weights(self, sample_returns, sample_expert_views):
        """测试Black-Litterman权重"""
        from finsage.hedging.tools.black_litterman import BlackLittermanTool

        tool = BlackLittermanTool()
        weights = tool.compute_weights(sample_returns, expert_views=sample_expert_views)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_mean_variance_weights(self, sample_returns):
        """测试均值方差权重"""
        from finsage.hedging.tools.mean_variance import MeanVarianceTool

        tool = MeanVarianceTool()
        weights = tool.compute_weights(sample_returns)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_hrp_weights(self, sample_returns):
        """测试HRP权重"""
        from finsage.hedging.tools.hrp import HierarchicalRiskParityTool

        tool = HierarchicalRiskParityTool()
        weights = tool.compute_weights(sample_returns)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ============================================================
# Test 7: Dynamic Hedge Selector
# ============================================================

class TestDynamicHedgeSelector:
    """测试动态对冲选择器"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.dynamic_selector import (
            DynamicHedgeSelector,
            HedgeObjective,
            PortfolioExposure,
            HedgeCandidate,
            HedgeRecommendation
        )
        assert DynamicHedgeSelector is not None
        assert HedgeObjective is not None

    def test_hedge_objective_enum(self):
        """测试对冲目标枚举"""
        from finsage.hedging.dynamic_selector import HedgeObjective

        assert HedgeObjective.BETA_NEUTRAL.value == "beta_neutral"
        assert HedgeObjective.TAIL_RISK.value == "tail_risk"
        assert HedgeObjective.DIVERSIFICATION.value == "diversification"

    def test_portfolio_exposure_dataclass(self):
        """测试组合敞口数据类"""
        from finsage.hedging.dynamic_selector import PortfolioExposure

        exposure = PortfolioExposure(
            beta=1.2,
            volatility=0.15,
            concentration_hhi=0.2,
            sector_exposure={'tech': 0.4, 'finance': 0.3},
            top_holdings=[{'symbol': 'AAPL', 'weight': 0.1}],
            correlation_with_spy=0.85,
            var_95=-0.025
        )

        assert exposure.beta == 1.2
        assert exposure.volatility == 0.15

        d = exposure.to_dict()
        assert 'beta' in d
        assert 'volatility' in d

    def test_hedge_candidate_dataclass(self):
        """测试对冲候选数据类"""
        from finsage.hedging.dynamic_selector import HedgeCandidate
        from finsage.hedging.hedge_universe import HedgeAsset, HedgeCategory

        asset = HedgeAsset(
            symbol='SH',
            name='ProShares Short S&P500',
            category=HedgeCategory.INVERSE_EQUITY,
            leverage=-1.0
        )

        candidate = HedgeCandidate(
            asset=asset,
            correlation_score=0.8,
            liquidity_score=0.7,
            cost_score=0.9,
            efficiency_score=0.6,
            total_score=0.75,
            raw_correlation=-0.6
        )

        assert candidate.total_score == 0.75
        d = candidate.to_dict()
        assert 'symbol' in d
        assert 'total_score' in d

    def test_selector_initialization(self):
        """测试选择器初始化"""
        from finsage.hedging.dynamic_selector import DynamicHedgeSelector

        selector = DynamicHedgeSelector()
        assert selector is not None
        assert selector.universe is not None

    def test_analyze_portfolio_exposure(self, sample_returns):
        """测试组合敞口分析"""
        from finsage.hedging.dynamic_selector import DynamicHedgeSelector

        selector = DynamicHedgeSelector()
        portfolio_weights = {'SPY': 0.5, 'QQQ': 0.3, 'TLT': 0.2}

        exposure = selector.analyze_portfolio_exposure(
            portfolio_weights, sample_returns
        )

        assert exposure.volatility > 0
        assert 0 < exposure.beta < 3

    def test_identify_hedge_objective(self):
        """测试对冲目标识别"""
        from finsage.hedging.dynamic_selector import (
            DynamicHedgeSelector, PortfolioExposure, HedgeObjective
        )

        selector = DynamicHedgeSelector()

        # 高Beta情况
        high_beta_exposure = PortfolioExposure(beta=1.5, volatility=0.12)
        objective = selector.identify_hedge_objective(
            high_beta_exposure, "dynamic_hedge"
        )
        assert objective == HedgeObjective.BETA_NEUTRAL

        # 高波动情况
        high_vol_exposure = PortfolioExposure(beta=1.0, volatility=0.25)
        objective = selector.identify_hedge_objective(
            high_vol_exposure, "tail_hedge"
        )
        assert objective == HedgeObjective.TAIL_RISK

    def test_select_candidates(self):
        """测试候选资产筛选"""
        from finsage.hedging.dynamic_selector import (
            DynamicHedgeSelector, PortfolioExposure, HedgeObjective
        )

        selector = DynamicHedgeSelector()
        exposure = PortfolioExposure()

        candidates = selector.select_candidates(
            HedgeObjective.BETA_NEUTRAL, exposure
        )

        assert isinstance(candidates, list)

    def test_get_universe_summary(self):
        """测试获取资产全集摘要"""
        from finsage.hedging.dynamic_selector import DynamicHedgeSelector

        selector = DynamicHedgeSelector()
        summary = selector.get_universe_summary()

        assert 'total_assets' in summary
        assert 'categories' in summary
        assert 'config' in summary


# ============================================================
# Test 8: Hedge Universe
# ============================================================

class TestHedgeUniverse:
    """测试对冲资产全集"""

    def test_import(self):
        """测试导入"""
        from finsage.hedging.hedge_universe import (
            HedgeAssetUniverse, HedgeAsset, HedgeCategory
        )
        assert HedgeAssetUniverse is not None
        assert HedgeAsset is not None
        assert HedgeCategory is not None

    def test_hedge_category_enum(self):
        """测试对冲类别枚举"""
        from finsage.hedging.hedge_universe import HedgeCategory

        assert HedgeCategory.INVERSE_EQUITY.value == "inverse_equity"
        assert HedgeCategory.SAFE_HAVEN.value == "safe_haven"
        assert HedgeCategory.VOLATILITY.value == "volatility"

    def test_hedge_asset_dataclass(self):
        """测试对冲资产数据类"""
        from finsage.hedging.hedge_universe import HedgeAsset, HedgeCategory

        asset = HedgeAsset(
            symbol='GLD',
            name='SPDR Gold Shares',
            category=HedgeCategory.SAFE_HAVEN,
            leverage=1.0,
            expense_ratio=0.004,
            avg_daily_volume=10000000
        )

        assert asset.symbol == 'GLD'
        assert asset.leverage == 1.0
        assert not asset.is_inverse

    def test_universe_initialization(self):
        """测试资产全集初始化"""
        from finsage.hedging.hedge_universe import HedgeAssetUniverse

        universe = HedgeAssetUniverse()
        assert universe is not None
        assert len(universe.get_all_symbols()) > 0

    def test_universe_get_by_category(self):
        """测试按类别获取资产"""
        from finsage.hedging.hedge_universe import HedgeAssetUniverse, HedgeCategory

        universe = HedgeAssetUniverse()
        safe_haven = universe.get_by_category(HedgeCategory.SAFE_HAVEN)

        assert isinstance(safe_haven, list)

    def test_universe_summary(self):
        """测试资产全集摘要"""
        from finsage.hedging.hedge_universe import HedgeAssetUniverse

        universe = HedgeAssetUniverse()
        summary = universe.summary()

        assert isinstance(summary, dict)


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Hedging Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python
"""
Agents Module Tests - 智能体模块测试
覆盖: base_expert, portfolio_manager, position_sizing_agent, hedging_agent,
      manager_coordinator, risk_controller, factor_enhanced_expert
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: Action Enum and Data Classes
# ============================================================

class TestActionEnum:
    """测试Action枚举类"""

    def test_action_values(self):
        """测试所有13个交易动作"""
        from finsage.agents.base_expert import Action

        # 验证13个动作
        actions = list(Action)
        assert len(actions) == 13

        # 验证做空动作
        short_actions = [a for a in actions if "SHORT" in a.value]
        assert len(short_actions) == 4

        # 验证买入动作
        buy_actions = [a for a in actions if "BUY" in a.value]
        assert len(buy_actions) == 4

        # 验证卖出动作
        sell_actions = [a for a in actions if "SELL" in a.value]
        assert len(sell_actions) == 4

        # 验证HOLD
        assert Action.HOLD.value == "HOLD"

    def test_action_ordering(self):
        """测试动作的强度顺序"""
        from finsage.agents.base_expert import Action

        # 买入强度从低到高
        assert Action.BUY_25.value == "BUY_25%"
        assert Action.BUY_100.value == "BUY_100%"


class TestExpertRecommendation:
    """测试ExpertRecommendation数据类"""

    def test_recommendation_creation(self):
        """测试建议对象创建"""
        from finsage.agents.base_expert import ExpertRecommendation, Action

        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="SPY",
            action=Action.BUY_50,
            confidence=0.8,
            target_weight=0.15,
            reasoning="Strong momentum signal",
            market_view={"trend": "bullish"},
            risk_assessment={"volatility": 0.15}
        )

        assert rec.symbol == "SPY"
        assert rec.confidence == 0.8
        assert rec.action == Action.BUY_50

    def test_recommendation_to_dict(self):
        """测试转换为字典"""
        from finsage.agents.base_expert import ExpertRecommendation, Action

        rec = ExpertRecommendation(
            asset_class="bonds",
            symbol="TLT",
            action=Action.HOLD,
            confidence=0.6,
            target_weight=0.1,
            reasoning="Neutral outlook",
            market_view={"trend": "neutral"},
            risk_assessment={"duration": 0.05}
        )

        d = rec.to_dict()
        assert d["symbol"] == "TLT"
        assert d["action"] == "HOLD"
        assert d["confidence"] == 0.6


class TestExpertReport:
    """测试ExpertReport数据类"""

    def test_report_creation(self):
        """测试报告创建"""
        from finsage.agents.base_expert import ExpertReport, ExpertRecommendation, Action

        recommendations = [
            ExpertRecommendation(
                asset_class="stocks",
                symbol="SPY",
                action=Action.BUY_25,
                confidence=0.7,
                target_weight=0.1,
                reasoning="Test",
                market_view={},
                risk_assessment={}
            )
        ]

        report = ExpertReport(
            expert_name="Stock Expert",
            asset_class="stocks",
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations,
            overall_view="bullish",
            sector_allocation={"SPY": 0.5},
            key_factors=["momentum", "volume"]
        )

        assert report.expert_name == "Stock Expert"
        assert len(report.recommendations) == 1
        assert report.overall_view == "bullish"

    def test_report_to_dict(self):
        """测试报告转字典"""
        from finsage.agents.base_expert import ExpertReport, ExpertRecommendation, Action

        rec = ExpertRecommendation(
            asset_class="crypto",
            symbol="BTC",
            action=Action.SHORT_25,
            confidence=0.65,
            target_weight=0.05,
            reasoning="Overbought",
            market_view={"sentiment": "bearish"},
            risk_assessment={"crash_risk": 0.3}
        )

        report = ExpertReport(
            expert_name="Crypto Expert",
            asset_class="crypto",
            timestamp="2024-01-15T10:00:00",
            recommendations=[rec],
            overall_view="bearish",
            sector_allocation={"BTC": 1.0},
            key_factors=["overbought"]
        )

        d = report.to_dict()
        assert d["expert_name"] == "Crypto Expert"
        assert len(d["recommendations"]) == 1
        assert d["recommendations"][0]["action"] == "SHORT_25%"


# ============================================================
# Test 2: Base Expert (Mock LLM)
# ============================================================

class TestBaseExpertMethods:
    """测试BaseExpert基类方法"""

    def test_determine_overall_view_bullish(self):
        """测试看多观点判断"""
        from finsage.agents.base_expert import BaseExpert, ExpertRecommendation, Action

        # 创建Mock子类
        class MockExpert(BaseExpert):
            @property
            def name(self): return "Mock"
            @property
            def description(self): return "Mock expert"
            @property
            def expertise(self): return ["testing"]
            def _build_analysis_prompt(self, *args): return "test"
            def _parse_llm_response(self, response): return []

        mock_llm = Mock()
        expert = MockExpert(mock_llm, "test", ["SPY"])

        # 创建看多建议
        recommendations = [
            ExpertRecommendation("stocks", "SPY", Action.BUY_75, 0.9, 0.2, "", {}, {}),
            ExpertRecommendation("stocks", "QQQ", Action.BUY_50, 0.8, 0.15, "", {}, {}),
        ]

        view = expert._determine_overall_view(recommendations)
        assert view == "bullish"

    def test_determine_overall_view_bearish(self):
        """测试看空观点判断"""
        from finsage.agents.base_expert import BaseExpert, ExpertRecommendation, Action

        class MockExpert(BaseExpert):
            @property
            def name(self): return "Mock"
            @property
            def description(self): return "Mock expert"
            @property
            def expertise(self): return ["testing"]
            def _build_analysis_prompt(self, *args): return "test"
            def _parse_llm_response(self, response): return []

        mock_llm = Mock()
        expert = MockExpert(mock_llm, "test", ["SPY"])

        # 创建看空建议(包含SHORT)
        recommendations = [
            ExpertRecommendation("stocks", "SPY", Action.SHORT_50, 0.85, 0.1, "", {}, {}),
            ExpertRecommendation("stocks", "QQQ", Action.SELL_75, 0.8, 0.1, "", {}, {}),
        ]

        view = expert._determine_overall_view(recommendations)
        assert view == "bearish"

    def test_determine_overall_view_neutral(self):
        """测试中性观点判断"""
        from finsage.agents.base_expert import BaseExpert, ExpertRecommendation, Action

        class MockExpert(BaseExpert):
            @property
            def name(self): return "Mock"
            @property
            def description(self): return "Mock expert"
            @property
            def expertise(self): return ["testing"]
            def _build_analysis_prompt(self, *args): return "test"
            def _parse_llm_response(self, response): return []

        mock_llm = Mock()
        expert = MockExpert(mock_llm, "test", ["SPY"])

        # 创建平衡建议
        recommendations = [
            ExpertRecommendation("stocks", "SPY", Action.BUY_25, 0.5, 0.1, "", {}, {}),
            ExpertRecommendation("stocks", "QQQ", Action.SELL_25, 0.5, 0.1, "", {}, {}),
        ]

        view = expert._determine_overall_view(recommendations)
        assert view == "neutral"

    def test_calculate_sector_allocation(self):
        """测试细分配置计算"""
        from finsage.agents.base_expert import BaseExpert, ExpertRecommendation, Action

        class MockExpert(BaseExpert):
            @property
            def name(self): return "Mock"
            @property
            def description(self): return "Mock expert"
            @property
            def expertise(self): return ["testing"]
            def _build_analysis_prompt(self, *args): return "test"
            def _parse_llm_response(self, response): return []

        mock_llm = Mock()
        expert = MockExpert(mock_llm, "test", ["SPY", "QQQ"])

        recommendations = [
            ExpertRecommendation("stocks", "SPY", Action.BUY_50, 0.8, 0.3, "", {}, {}),
            ExpertRecommendation("stocks", "QQQ", Action.BUY_25, 0.7, 0.2, "", {}, {}),
        ]

        allocation = expert._calculate_sector_allocation(recommendations)

        # 验证归一化
        assert abs(sum(allocation.values()) - 1.0) < 0.001
        assert allocation["SPY"] == 0.6  # 0.3 / 0.5
        assert allocation["QQQ"] == 0.4  # 0.2 / 0.5

    def test_update_symbols(self):
        """测试动态更新资产池"""
        from finsage.agents.base_expert import BaseExpert

        class MockExpert(BaseExpert):
            @property
            def name(self): return "Mock"
            @property
            def description(self): return "Mock expert"
            @property
            def expertise(self): return ["testing"]
            def _build_analysis_prompt(self, *args): return "test"
            def _parse_llm_response(self, response): return []

        mock_llm = Mock()
        expert = MockExpert(mock_llm, "test", ["SPY", "QQQ"])

        assert len(expert.symbols) == 2

        expert.update_symbols(["SPY", "QQQ", "IWM", "DIA"])
        assert len(expert.symbols) == 4
        assert "IWM" in expert.symbols


# ============================================================
# Test 3: Position Sizing Agent
# ============================================================

class TestPositionSizingAgent:
    """测试仓位规模智能体"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.position_sizing_agent import PositionSizingAgent
        assert PositionSizingAgent is not None

    def test_position_sizing_methods(self):
        """测试仓位计算方法"""
        from finsage.agents.position_sizing_agent import PositionSizingAgent

        # 只测试类存在和基本属性
        # 实际的analyze需要LLM，这里不测试
        assert hasattr(PositionSizingAgent, 'analyze')


# ============================================================
# Test 4: Hedging Agent
# ============================================================

class TestHedgingAgent:
    """测试对冲智能体"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.hedging_agent import HedgingAgent, HedgingDecision
        assert HedgingAgent is not None
        assert HedgingDecision is not None

    def test_hedging_decision_dataclass(self):
        """测试对冲决策数据类"""
        from finsage.agents.hedging_agent import HedgingDecision

        decision = HedgingDecision(
            timestamp="2024-01-15T10:00:00",
            hedging_strategy="put_protection",
            hedge_ratio=0.15,
            hedge_instruments=[{"type": "SPY_PUT", "strike": 420}],
            expected_cost=0.02,
            expected_protection=0.05,
            reasoning="High tail risk detected, implementing put protection",
            tail_risk_metrics={"var_95": 0.03, "cvar_99": 0.05},
            dynamic_recommendation=None
        )

        assert decision.hedging_strategy == "put_protection"
        assert decision.hedge_ratio == 0.15


# ============================================================
# Test 5: Manager Coordinator
# ============================================================

class TestManagerCoordinator:
    """测试管理协调器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.manager_coordinator import ManagerCoordinator
        assert ManagerCoordinator is not None


# ============================================================
# Test 6: Risk Controller
# ============================================================

class TestRiskController:
    """测试风险控制器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.risk_controller import RiskController, RiskAssessment
        assert RiskController is not None
        assert RiskAssessment is not None

    def test_risk_assessment_dataclass(self):
        """测试风险评估数据类"""
        from finsage.agents.risk_controller import RiskAssessment

        assessment = RiskAssessment(
            timestamp="2024-01-15T10:00:00",
            portfolio_var_95=0.025,
            portfolio_cvar_99=0.04,
            current_drawdown=-0.02,
            max_drawdown=-0.05,
            volatility=0.15,
            sharpe_ratio=1.2,
            concentration_risk="low",
            violations=[],
            warnings=["High volatility detected"],
            veto=False,
            recommendations={"reduce_exposure": 0.1},
            defensive_allocation=None
        )

        assert assessment.veto == False
        assert len(assessment.warnings) == 1
        assert assessment.concentration_risk == "low"


# ============================================================
# Test 7: Factor Enhanced Expert
# ============================================================

class TestFactorEnhancedExpert:
    """测试因子增强专家"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.factor_enhanced_expert import FactorEnhancedExpertMixin
        assert FactorEnhancedExpertMixin is not None


# ============================================================
# Test 8: Expert Implementations
# ============================================================

class TestExpertImplementations:
    """测试各资产类别专家实现"""

    def test_stock_expert_import(self):
        """测试股票专家导入"""
        from finsage.agents.experts.stock_expert import StockExpert
        assert StockExpert is not None

    def test_bond_expert_import(self):
        """测试债券专家导入"""
        from finsage.agents.experts.bond_expert import BondExpert
        assert BondExpert is not None

    def test_commodity_expert_import(self):
        """测试商品专家导入"""
        from finsage.agents.experts.commodity_expert import CommodityExpert
        assert CommodityExpert is not None

    def test_reits_expert_import(self):
        """测试REITs专家导入"""
        from finsage.agents.experts.reits_expert import REITsExpert
        assert REITsExpert is not None

    def test_crypto_expert_import(self):
        """测试加密货币专家导入"""
        from finsage.agents.experts.crypto_expert import CryptoExpert
        assert CryptoExpert is not None


# ============================================================
# Test 9: Portfolio Manager
# ============================================================

class TestPortfolioManager:
    """测试组合管理器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.portfolio_manager import PortfolioManager, PortfolioDecision
        assert PortfolioManager is not None
        assert PortfolioDecision is not None

    def test_portfolio_decision_dataclass(self):
        """测试组合决策数据类"""
        from finsage.agents.portfolio_manager import PortfolioDecision

        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"stocks": 0.4, "bonds": 0.3, "cash": 0.3},
            trades=[{"asset": "SPY", "action": "BUY", "weight": 0.1}],
            hedging_tool_used="risk_parity",
            reasoning="Balanced allocation based on expert consensus",
            risk_metrics={"var_95": 0.02},
            expert_summary={"bullish": "3", "neutral": "1", "bearish": "1"}
        )

        assert decision.hedging_tool_used == "risk_parity"
        assert "stocks" in decision.target_allocation


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Agents Module Tests")
    print("=" * 60)

    # 使用pytest运行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

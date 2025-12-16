#!/usr/bin/env python
"""
Advanced Experts Tests - 专家代理深度测试
覆盖: stock_expert, bond_expert, commodity_expert, crypto_expert, reits_expert
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
# Test 1: Expert Imports
# ============================================================

class TestExpertImports:
    """测试专家模块导入"""

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

    def test_crypto_expert_import(self):
        """测试加密货币专家导入"""
        from finsage.agents.experts.crypto_expert import CryptoExpert
        assert CryptoExpert is not None

    def test_reits_expert_import(self):
        """测试REITs专家导入"""
        from finsage.agents.experts.reits_expert import REITsExpert
        assert REITsExpert is not None

    def test_enhanced_commodity_expert_import(self):
        """测试增强商品专家导入"""
        from finsage.agents.experts.enhanced_commodity_expert import EnhancedCommodityExpert
        assert EnhancedCommodityExpert is not None


# ============================================================
# Test 2: BaseExpert Class
# ============================================================

class TestBaseExpert:
    """测试基础专家类"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.base_expert import BaseExpert
        assert BaseExpert is not None

    def test_expert_recommendation_import(self):
        """测试专家推荐数据类导入"""
        from finsage.agents.base_expert import ExpertRecommendation
        assert ExpertRecommendation is not None

    def test_expert_report_import(self):
        """测试专家报告数据类导入"""
        from finsage.agents.base_expert import ExpertReport
        assert ExpertReport is not None

    def test_expert_recommendation_creation(self):
        """测试创建专家推荐"""
        from finsage.agents.base_expert import ExpertRecommendation

        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="SPY",
            action="BUY",
            confidence=0.85,
            target_weight=0.15,
            reasoning="Strong momentum signals",
            market_view="bullish",
            risk_assessment="medium"
        )

        assert rec.symbol == "SPY"
        assert rec.action == "BUY"
        assert rec.confidence == 0.85

    def test_expert_report_creation(self):
        """测试创建专家报告"""
        from finsage.agents.base_expert import ExpertReport, ExpertRecommendation

        rec = ExpertRecommendation(
            asset_class="stocks",
            symbol="SPY",
            action="BUY",
            confidence=0.8,
            target_weight=0.1,
            reasoning="Test",
            market_view="bullish",
            risk_assessment="low"
        )

        report = ExpertReport(
            expert_name="Stock_Expert",
            asset_class="stocks",
            timestamp="2024-01-15T10:00:00",
            recommendations=[rec],
            overall_view="bullish",
            sector_allocation={"technology": 0.4},
            key_factors=["momentum", "value"]
        )

        assert report.expert_name == "Stock_Expert"
        assert report.overall_view == "bullish"
        assert len(report.recommendations) == 1


# ============================================================
# Test 3: Expert Class Structure
# ============================================================

class TestExpertClassStructure:
    """测试专家类结构"""

    def test_stock_expert_has_required_attributes(self):
        """测试股票专家有必要属性"""
        from finsage.agents.experts.stock_expert import StockExpert

        # 验证类定义中包含必要的属性/方法
        assert hasattr(StockExpert, '__init__')
        assert hasattr(StockExpert, 'analyze')

    def test_bond_expert_has_required_attributes(self):
        """测试债券专家有必要属性"""
        from finsage.agents.experts.bond_expert import BondExpert

        assert hasattr(BondExpert, '__init__')
        assert hasattr(BondExpert, 'analyze')

    def test_commodity_expert_has_required_attributes(self):
        """测试商品专家有必要属性"""
        from finsage.agents.experts.commodity_expert import CommodityExpert

        assert hasattr(CommodityExpert, '__init__')
        assert hasattr(CommodityExpert, 'analyze')

    def test_crypto_expert_has_required_attributes(self):
        """测试加密货币专家有必要属性"""
        from finsage.agents.experts.crypto_expert import CryptoExpert

        assert hasattr(CryptoExpert, '__init__')
        assert hasattr(CryptoExpert, 'analyze')

    def test_reits_expert_has_required_attributes(self):
        """测试REITs专家有必要属性"""
        from finsage.agents.experts.reits_expert import REITsExpert

        assert hasattr(REITsExpert, '__init__')
        assert hasattr(REITsExpert, 'analyze')


# ============================================================
# Test 4: FactorEnhancedExpert
# ============================================================

class TestFactorEnhancedExpert:
    """测试因子增强专家"""

    def test_mixin_import(self):
        """测试Mixin导入"""
        from finsage.agents.factor_enhanced_expert import FactorEnhancedExpertMixin
        assert FactorEnhancedExpertMixin is not None

    def test_enhanced_recommendation_import(self):
        """测试增强推荐导入"""
        from finsage.agents.factor_enhanced_expert import EnhancedRecommendation
        assert EnhancedRecommendation is not None

    def test_create_enhanced_expert_function(self):
        """测试创建增强专家函数"""
        from finsage.agents.factor_enhanced_expert import create_factor_enhanced_expert
        assert create_factor_enhanced_expert is not None

    def test_get_enhanced_experts_function(self):
        """测试获取增强专家函数"""
        from finsage.agents.factor_enhanced_expert import get_enhanced_experts
        assert get_enhanced_experts is not None


# ============================================================
# Test 5: HedgingAgent
# ============================================================

class TestHedgingAgent:
    """测试对冲代理"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.hedging_agent import HedgingAgent
        assert HedgingAgent is not None

    def test_hedging_decision_import(self):
        """测试对冲决策数据类导入"""
        from finsage.agents.hedging_agent import HedgingDecision
        assert HedgingDecision is not None

    def test_hedging_decision_creation(self):
        """测试创建对冲决策"""
        from finsage.agents.hedging_agent import HedgingDecision

        decision = HedgingDecision(
            timestamp="2024-01-15T10:00:00",
            hedging_strategy="put_protection",
            hedge_ratio=0.15,
            hedge_instruments=[{"type": "SPY_PUT", "strike": 420}],
            expected_cost=0.02,
            expected_protection=0.05,
            reasoning="High tail risk detected",
            tail_risk_metrics={"var_95": 0.03},
            dynamic_recommendation=None
        )

        assert decision.hedging_strategy == "put_protection"
        assert decision.hedge_ratio == 0.15


# ============================================================
# Test 6: PositionSizingAgent
# ============================================================

class TestPositionSizingAgent:
    """测试仓位管理代理"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.position_sizing_agent import PositionSizingAgent
        assert PositionSizingAgent is not None

    def test_class_has_init(self):
        """测试类有初始化方法"""
        from finsage.agents.position_sizing_agent import PositionSizingAgent

        # 验证有初始化方法
        assert hasattr(PositionSizingAgent, '__init__')


# ============================================================
# Test 7: RiskController
# ============================================================

class TestRiskController:
    """测试风险控制器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.risk_controller import RiskController
        assert RiskController is not None

    def test_risk_assessment_import(self):
        """测试风险评估数据类导入"""
        from finsage.agents.risk_controller import RiskAssessment
        assert RiskAssessment is not None

    def test_risk_assessment_creation(self):
        """测试创建风险评估"""
        from finsage.agents.risk_controller import RiskAssessment

        assessment = RiskAssessment(
            timestamp="2024-01-15T10:00:00",
            portfolio_var_95=0.02,
            portfolio_cvar_99=0.03,
            current_drawdown=0.05,
            max_drawdown=0.15,
            volatility=0.12,
            sharpe_ratio=1.2,
            concentration_risk=0.25,
            violations=[],
            warnings=["high volatility"],
            veto=False,
            recommendations=["reduce equity exposure"],
            intraday_alerts=[],
            emergency_rebalance=False,
            defensive_allocation=None
        )

        assert assessment.volatility == 0.12
        assert assessment.sharpe_ratio == 1.2


# ============================================================
# Test 8: ManagerCoordinator
# ============================================================

class TestManagerCoordinator:
    """测试管理协调器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.manager_coordinator import ManagerCoordinator
        assert ManagerCoordinator is not None

    def test_class_attributes(self):
        """测试类属性"""
        from finsage.agents.manager_coordinator import ManagerCoordinator

        assert hasattr(ManagerCoordinator, '__init__')


# ============================================================
# Test 9: PortfolioManager
# ============================================================

class TestPortfolioManager:
    """测试投资组合管理器"""

    def test_import(self):
        """测试导入"""
        from finsage.agents.portfolio_manager import PortfolioManager
        assert PortfolioManager is not None

    def test_portfolio_decision_import(self):
        """测试投资组合决策数据类导入"""
        from finsage.agents.portfolio_manager import PortfolioDecision
        assert PortfolioDecision is not None

    def test_portfolio_decision_creation(self):
        """测试创建投资组合决策"""
        from finsage.agents.portfolio_manager import PortfolioDecision

        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "cash": 0.1},
            trades=[{"symbol": "SPY", "action": "BUY", "shares": 100}],
            hedging_tool_used="risk_parity",
            reasoning="Rebalancing to target weights",
            risk_metrics={"volatility": 0.12, "var_95": 0.02},
            expert_summary={"stock": "bullish", "bond": "neutral"}
        )

        assert decision.hedging_tool_used == "risk_parity"
        assert "SPY" in decision.target_allocation


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Advanced Experts Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

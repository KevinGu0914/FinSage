"""
Deep tests for ManagerCoordinator
管理层协调器深度测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from concurrent.futures import ThreadPoolExecutor

from finsage.agents.manager_coordinator import (
    ManagerCoordinator,
    IntegratedDecision,
)
from finsage.agents.portfolio_manager import PortfolioDecision
from finsage.agents.position_sizing_agent import PositionSizingDecision
from finsage.agents.hedging_agent import HedgingDecision


class TestIntegratedDecision:
    """IntegratedDecision数据类测试"""

    def test_create_decision(self):
        """测试创建整合决策"""
        decision = IntegratedDecision(
            timestamp="2024-01-15",
            target_allocation={"stocks": 0.5, "bonds": 0.5},
            position_sizes={"stocks": 0.5, "bonds": 0.5},
            hedging_strategy="dynamic",
            hedge_ratio=0.1,
            hedge_instruments=[],
            trades=[],
            risk_metrics={"var_95": 0.02},
            tail_risk_metrics={"cvar_99": 0.03},
            discussion_rounds=2,
            consensus_reached=True,
            individual_decisions={},
            final_reasoning="综合决策",
        )
        assert decision.hedging_strategy == "dynamic"
        assert decision.consensus_reached is True

    def test_to_dict(self):
        """测试转换为字典"""
        decision = IntegratedDecision(
            timestamp="2024-01-15",
            target_allocation={"stocks": 0.6},
            position_sizes={"stocks": 0.6},
            hedging_strategy="none",
            hedge_ratio=0.0,
            hedge_instruments=[],
            trades=[{"asset": "stocks", "action": "BUY"}],
            risk_metrics={},
            tail_risk_metrics={},
            discussion_rounds=1,
            consensus_reached=True,
            individual_decisions={},
            final_reasoning="Test",
        )
        result = decision.to_dict()
        assert "target_allocation" in result
        assert "discussion_rounds" in result
        assert result["consensus_reached"] is True


class TestManagerCoordinatorInit:
    """ManagerCoordinator初始化测试"""

    def test_init(self):
        """测试初始化"""
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()

        coordinator = ManagerCoordinator(
            portfolio_manager=pm,
            position_sizing_agent=sizing,
            hedging_agent=hedging,
            llm_provider=llm,
        )
        assert coordinator.max_discussion_rounds == 2
        assert coordinator.consensus_threshold == 0.85
        assert coordinator.parallel_execution is True

    def test_custom_config(self):
        """测试自定义配置"""
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()

        config = {
            "max_discussion_rounds": 3,
            "consensus_threshold": 0.90,
            "parallel_execution": False,
        }
        coordinator = ManagerCoordinator(
            portfolio_manager=pm,
            position_sizing_agent=sizing,
            hedging_agent=hedging,
            llm_provider=llm,
            config=config,
        )
        assert coordinator.max_discussion_rounds == 3
        assert coordinator.parallel_execution is False


class TestGenerateTrades:
    """交易生成测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_generate_buy_trade(self, coordinator):
        """测试生成买入交易"""
        current = {"stocks": 0.30}
        target = {"stocks": 0.50}
        trades = coordinator._generate_trades(current, target)
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"
        assert trades[0]["asset"] == "stocks"

    def test_generate_sell_trade(self, coordinator):
        """测试生成卖出交易"""
        current = {"stocks": 0.60}
        target = {"stocks": 0.40}
        trades = coordinator._generate_trades(current, target)
        assert len(trades) == 1
        assert trades[0]["action"] == "SELL"

    def test_no_trade_below_threshold(self, coordinator):
        """测试低于阈值不交易"""
        current = {"stocks": 0.50}
        target = {"stocks": 0.51}  # 差值<2%
        trades = coordinator._generate_trades(current, target)
        assert len(trades) == 0

    def test_multiple_trades(self, coordinator):
        """测试多笔交易"""
        current = {"stocks": 0.50, "bonds": 0.30, "cash": 0.20}
        target = {"stocks": 0.30, "bonds": 0.50, "cash": 0.20}
        trades = coordinator._generate_trades(current, target)
        assert len(trades) == 2

    def test_new_asset_trade(self, coordinator):
        """测试新资产交易"""
        current = {"stocks": 0.80}
        target = {"stocks": 0.60, "bonds": 0.40}
        trades = coordinator._generate_trades(current, target)
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        assert len(buy_trades) == 1
        assert buy_trades[0]["asset"] == "bonds"


class TestCheckConsensus:
    """共识检查测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_consensus_reached(self, coordinator):
        """测试达成共识"""
        old_sizing = MagicMock()
        old_sizing.position_sizes = {"stocks": 0.50, "bonds": 0.50}
        new_sizing = MagicMock()
        new_sizing.position_sizes = {"stocks": 0.51, "bonds": 0.49}

        old_hedging = MagicMock()
        old_hedging.hedge_ratio = 0.10
        new_hedging = MagicMock()
        new_hedging.hedge_ratio = 0.11

        result = coordinator._check_consensus(old_sizing, new_sizing, old_hedging, new_hedging)
        assert result is True

    def test_consensus_not_reached(self, coordinator):
        """测试未达成共识"""
        old_sizing = MagicMock()
        old_sizing.position_sizes = {"stocks": 0.50, "bonds": 0.50}
        new_sizing = MagicMock()
        new_sizing.position_sizes = {"stocks": 0.30, "bonds": 0.70}

        old_hedging = MagicMock()
        old_hedging.hedge_ratio = 0.10
        new_hedging = MagicMock()
        new_hedging.hedge_ratio = 0.20

        result = coordinator._check_consensus(old_sizing, new_sizing, old_hedging, new_hedging)
        assert result is False


class TestGenerateFinalReasoning:
    """最终决策理由生成测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_generate_reasoning(self, coordinator):
        """测试生成最终理由"""
        pm_decision = MagicMock()
        pm_decision.reasoning = "市场看涨，增加股票配置。增加债券对冲。"

        sizing_decision = MagicMock()
        sizing_decision.reasoning = "使用风险平价方法配置仓位。均衡风险贡献。"

        hedging_decision = MagicMock()
        hedging_decision.reasoning = "动态对冲策略，对冲尾部风险。保护下行。"

        reasoning = coordinator._generate_final_reasoning(
            pm_decision, sizing_decision, hedging_decision
        )
        assert "综合决策" in reasoning
        assert "组合管理" in reasoning
        assert "仓位管理" in reasoning


class TestIntegrateDecisions:
    """决策整合测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_integrate_basic(self, coordinator):
        """测试基本整合"""
        pm_decision = MagicMock()
        pm_decision.target_allocation = {"stocks": 0.50, "bonds": 0.30, "cash": 0.20}
        pm_decision.expert_summary = "看涨"
        pm_decision.risk_metrics = {"var": 0.02}
        pm_decision.reasoning = "增加股票"
        pm_decision.to_dict.return_value = {}

        sizing_decision = MagicMock()
        sizing_decision.position_sizes = {"stocks": 0.45, "bonds": 0.35, "cash": 0.20}
        sizing_decision.sizing_method = "risk_parity"
        sizing_decision.reasoning = "风险平价"
        sizing_decision.to_dict.return_value = {}

        hedging_decision = MagicMock()
        hedging_decision.hedging_strategy = "dynamic"
        hedging_decision.hedge_ratio = 0.1
        hedging_decision.hedge_instruments = []
        hedging_decision.tail_risk_metrics = {}
        hedging_decision.reasoning = "动态对冲"
        hedging_decision.to_dict.return_value = {}

        current_portfolio = {"stocks": 0.40, "bonds": 0.40, "cash": 0.20}

        result = coordinator._integrate_decisions(
            pm_decision, sizing_decision, hedging_decision,
            current_portfolio, discussion_rounds=2
        )

        assert isinstance(result, IntegratedDecision)
        assert len(result.target_allocation) > 0
        assert abs(sum(result.target_allocation.values()) - 1.0) < 0.1

    def test_integrate_with_hedging(self, coordinator):
        """测试带对冲的整合"""
        pm_decision = MagicMock()
        pm_decision.target_allocation = {"stocks": 0.60, "bonds": 0.20, "cash": 0.20}
        pm_decision.expert_summary = ""
        pm_decision.risk_metrics = {}
        pm_decision.reasoning = ""
        pm_decision.to_dict.return_value = {}

        sizing_decision = MagicMock()
        sizing_decision.position_sizes = {"stocks": 0.60, "bonds": 0.20, "cash": 0.20}
        sizing_decision.sizing_method = "equal_weight"
        sizing_decision.reasoning = ""
        sizing_decision.to_dict.return_value = {}

        hedging_decision = MagicMock()
        hedging_decision.hedging_strategy = "tail_risk"
        hedging_decision.hedge_ratio = 0.2
        hedging_decision.hedge_instruments = [{"type": "put"}]
        hedging_decision.tail_risk_metrics = {}
        hedging_decision.reasoning = ""
        hedging_decision.to_dict.return_value = {}

        result = coordinator._integrate_decisions(
            pm_decision, sizing_decision, hedging_decision,
            {}, discussion_rounds=1
        )

        # 对冲应减少股票配置
        assert result.target_allocation.get("stocks", 0) < 0.60


class TestPhase1ParallelAnalysis:
    """阶段1并行分析测试"""

    def test_parallel_execution(self):
        """测试并行执行"""
        pm = MagicMock()
        pm.decide.return_value = MagicMock(
            target_allocation={"stocks": 0.5},
            spec=['target_allocation']
        )
        pm._get_default_allocation.return_value = {"stocks": 0.5}

        sizing = MagicMock()
        sizing.analyze.return_value = MagicMock(
            position_sizes={"stocks": 0.5},
            sizing_method="equal_weight"
        )

        hedging = MagicMock()
        hedging.analyze.return_value = MagicMock(
            hedging_strategy="none",
            hedge_ratio=0.0
        )

        llm = MagicMock()

        coordinator = ManagerCoordinator(pm, sizing, hedging, llm)

        result = coordinator._phase1_parallel_analysis(
            expert_reports={},
            market_data={},
            current_portfolio={},
            risk_constraints={},
            portfolio_value=1000000,
        )

        assert "pm" in result
        assert "sizing" in result
        assert "hedging" in result

    def test_sequential_execution(self):
        """测试顺序执行"""
        pm = MagicMock()
        pm.decide.return_value = MagicMock(
            target_allocation={"stocks": 0.5}
        )

        sizing = MagicMock()
        sizing.analyze.return_value = MagicMock(
            position_sizes={"stocks": 0.5},
            sizing_method="equal_weight"
        )

        hedging = MagicMock()
        hedging.analyze.return_value = MagicMock(
            hedging_strategy="none",
            hedge_ratio=0.0
        )

        llm = MagicMock()

        coordinator = ManagerCoordinator(
            pm, sizing, hedging, llm,
            config={"parallel_execution": False}
        )

        result = coordinator._phase1_parallel_analysis(
            expert_reports={},
            market_data={},
            current_portfolio={},
            risk_constraints={},
            portfolio_value=1000000,
        )

        assert "pm" in result


class TestPhase2Discussion:
    """阶段2讨论测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        sizing.revise_based_on_feedback.return_value = MagicMock(
            position_sizes={"stocks": 0.5},
            sizing_method="revised"
        )

        hedging = MagicMock()
        hedging.revise_based_on_feedback.return_value = MagicMock(
            hedging_strategy="revised",
            hedge_ratio=0.1
        )

        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_discussion_reaches_consensus(self, coordinator):
        """测试讨论达成共识"""
        pm_decision = MagicMock()
        pm_decision.target_allocation = {"stocks": 0.5}
        pm_decision.expert_summary = ""
        pm_decision.risk_metrics = {}

        sizing_decision = MagicMock()
        sizing_decision.position_sizes = {"stocks": 0.5}
        sizing_decision.sizing_method = "equal_weight"
        sizing_decision.risk_contribution = {}

        hedging_decision = MagicMock()
        hedging_decision.hedging_strategy = "none"
        hedging_decision.hedge_ratio = 0.0
        hedging_decision.tail_risk_metrics = {}
        hedging_decision.expected_cost = 0.0

        # Mock返回几乎相同的决策以达成共识
        coordinator.sizing_agent.revise_based_on_feedback.return_value = MagicMock(
            position_sizes={"stocks": 0.51}
        )
        coordinator.hedging_agent.revise_based_on_feedback.return_value = MagicMock(
            hedge_ratio=0.01
        )

        result, rounds = coordinator._phase2_discussion(
            pm_decision, sizing_decision, hedging_decision,
            {}, {}
        )

        assert "pm" in result
        assert rounds <= coordinator.max_discussion_rounds


class TestCoordinate:
    """完整协调流程测试"""

    def test_coordinate_full_flow(self):
        """测试完整协调流程"""
        # Setup mocks
        pm = MagicMock()
        pm.decide.return_value = MagicMock(
            target_allocation={"stocks": 0.5, "bonds": 0.5},
            expert_summary="",
            risk_metrics={},
            reasoning="PM reasoning" * 20,
            to_dict=MagicMock(return_value={})
        )
        pm._get_default_allocation.return_value = {"stocks": 0.5}

        sizing = MagicMock()
        sizing.analyze.return_value = MagicMock(
            position_sizes={"stocks": 0.5, "bonds": 0.5},
            sizing_method="risk_parity",
            risk_contribution={},
            reasoning="Sizing reasoning" * 20,
            to_dict=MagicMock(return_value={})
        )
        sizing.revise_based_on_feedback.return_value = MagicMock(
            position_sizes={"stocks": 0.5, "bonds": 0.5}
        )

        hedging = MagicMock()
        hedging.analyze.return_value = MagicMock(
            hedging_strategy="dynamic",
            hedge_ratio=0.1,
            hedge_instruments=[],
            tail_risk_metrics={},
            expected_cost=0.01,
            reasoning="Hedging reasoning" * 20,
            to_dict=MagicMock(return_value={})
        )
        hedging.revise_based_on_feedback.return_value = MagicMock(
            hedge_ratio=0.11
        )

        llm = MagicMock()

        coordinator = ManagerCoordinator(pm, sizing, hedging, llm)

        result = coordinator.coordinate(
            expert_reports={},
            market_data={},
            current_portfolio={"stocks": 0.4, "bonds": 0.4, "cash": 0.2},
            risk_constraints={},
            portfolio_value=1000000,
        )

        assert isinstance(result, IntegratedDecision)
        assert len(result.target_allocation) > 0


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def coordinator(self):
        pm = MagicMock()
        sizing = MagicMock()
        hedging = MagicMock()
        llm = MagicMock()
        return ManagerCoordinator(pm, sizing, hedging, llm)

    def test_empty_current_portfolio(self, coordinator):
        """测试空当前组合"""
        trades = coordinator._generate_trades({}, {"stocks": 0.5})
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"

    def test_empty_target_allocation(self, coordinator):
        """测试空目标配置"""
        trades = coordinator._generate_trades({"stocks": 0.5}, {})
        assert len(trades) == 0  # 不生成卖出到0的交易

    def test_negative_hedge_ratio(self, coordinator):
        """测试负对冲比率处理"""
        pm_decision = MagicMock()
        pm_decision.target_allocation = {"stocks": 0.5}
        pm_decision.to_dict.return_value = {}
        pm_decision.reasoning = "test"
        pm_decision.risk_metrics = {}

        sizing_decision = MagicMock()
        sizing_decision.position_sizes = {}
        sizing_decision.to_dict.return_value = {}
        sizing_decision.reasoning = "test"

        hedging_decision = MagicMock()
        hedging_decision.hedging_strategy = "none"
        hedging_decision.hedge_ratio = -0.1  # 负值
        hedging_decision.hedge_instruments = []
        hedging_decision.tail_risk_metrics = {}
        hedging_decision.to_dict.return_value = {}
        hedging_decision.reasoning = "test"

        result = coordinator._integrate_decisions(
            pm_decision, sizing_decision, hedging_decision, {}, 1
        )
        # 应该正常处理
        assert isinstance(result, IntegratedDecision)

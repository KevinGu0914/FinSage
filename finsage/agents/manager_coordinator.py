"""
Manager Coordinator
管理层协调器 - 负责协调三个管理智能体的并行讨论和最终决策整合
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from finsage.agents.base_expert import ExpertReport
from finsage.agents.portfolio_manager import PortfolioManager, PortfolioDecision
from finsage.agents.position_sizing_agent import PositionSizingAgent, PositionSizingDecision
from finsage.agents.hedging_agent import HedgingAgent, HedgingDecision

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDecision:
    """整合后的最终决策"""
    timestamp: str

    # 资产配置
    target_allocation: Dict[str, float]
    position_sizes: Dict[str, float]

    # 对冲
    hedging_strategy: str
    hedge_ratio: float
    hedge_instruments: List[Dict]

    # 交易
    trades: List[Dict[str, Any]]

    # 风险
    risk_metrics: Dict[str, float]
    tail_risk_metrics: Dict[str, float]

    # 决策过程
    discussion_rounds: int
    consensus_reached: bool
    individual_decisions: Dict[str, Any]
    final_reasoning: str

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "target_allocation": self.target_allocation,
            "position_sizes": self.position_sizes,
            "hedging_strategy": self.hedging_strategy,
            "hedge_ratio": self.hedge_ratio,
            "hedge_instruments": self.hedge_instruments,
            "trades": self.trades,
            "risk_metrics": self.risk_metrics,
            "tail_risk_metrics": self.tail_risk_metrics,
            "discussion_rounds": self.discussion_rounds,
            "consensus_reached": self.consensus_reached,
            "final_reasoning": self.final_reasoning,
        }


class ManagerCoordinator:
    """
    管理层协调器

    架构:
    ```
    阶段1: 并行分析（各自独立）
      ├── Portfolio Manager → 初步配置建议
      ├── Position Sizing Agent → 仓位建议
      └── Hedging Agent → 对冲建议

    阶段2: 讨论整合（1-2轮）
      └── 三个Agent看到彼此建议后修正
          → 最终整合决策
    ```

    核心职责:
    1. 协调三个管理智能体的执行
    2. 管理并行讨论过程
    3. 整合最终决策
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        position_sizing_agent: PositionSizingAgent,
        hedging_agent: HedgingAgent,
        llm_provider: Any,
        config: Optional[Dict] = None
    ):
        """
        初始化协调器

        Args:
            portfolio_manager: 组合管理智能体
            position_sizing_agent: 仓位规模智能体
            hedging_agent: 对冲策略智能体
            llm_provider: LLM服务提供者
            config: 配置参数
        """
        self.pm = portfolio_manager
        self.sizing_agent = position_sizing_agent
        self.hedging_agent = hedging_agent
        self.llm = llm_provider
        self.config = config or {}

        # 配置参数
        self.max_discussion_rounds = self.config.get("max_discussion_rounds", 2)
        self.consensus_threshold = self.config.get("consensus_threshold", 0.85)
        self.parallel_execution = self.config.get("parallel_execution", True)

        logger.info("Manager Coordinator initialized")

    def coordinate(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        current_portfolio: Dict[str, float],
        risk_constraints: Dict[str, float],
        portfolio_value: float,
    ) -> IntegratedDecision:
        """
        协调三个管理智能体进行决策

        Args:
            expert_reports: 5位专家的报告
            market_data: 市场数据
            current_portfolio: 当前持仓
            risk_constraints: 风控约束
            portfolio_value: 组合总价值

        Returns:
            IntegratedDecision: 整合后的最终决策
        """
        logger.info("Starting manager coordination process")

        # 阶段1: 并行独立分析
        phase1_results = self._phase1_parallel_analysis(
            expert_reports, market_data, current_portfolio,
            risk_constraints, portfolio_value
        )

        pm_decision = phase1_results["pm"]
        sizing_decision = phase1_results["sizing"]
        hedging_decision = phase1_results["hedging"]

        logger.info(f"Phase 1 complete: PM allocation={len(pm_decision.target_allocation)}, "
                    f"Sizing method={sizing_decision.sizing_method}, "
                    f"Hedging strategy={hedging_decision.hedging_strategy}")

        # 阶段2: 讨论整合
        final_decisions, discussion_rounds = self._phase2_discussion(
            pm_decision, sizing_decision, hedging_decision,
            market_data, risk_constraints
        )

        # 阶段3: 最终整合
        integrated = self._integrate_decisions(
            final_decisions["pm"],
            final_decisions["sizing"],
            final_decisions["hedging"],
            current_portfolio,
            discussion_rounds
        )

        logger.info(f"Coordination complete: {discussion_rounds} rounds, "
                    f"consensus={integrated.consensus_reached}")

        return integrated

    def _phase1_parallel_analysis(
        self,
        expert_reports: Dict[str, ExpertReport],
        market_data: Dict[str, Any],
        current_portfolio: Dict[str, float],
        risk_constraints: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """
        阶段1: 并行独立分析

        三个智能体同时独立分析，不互相影响
        """
        if self.parallel_execution:
            # 使用线程池并行执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 提交任务
                pm_future = executor.submit(
                    self.pm.decide,
                    expert_reports, market_data, current_portfolio, risk_constraints
                )

                # Position Sizing 需要先有 PM 的初步配置
                # 这里使用默认配置作为初始输入
                default_allocation = self.pm._get_default_allocation()
                sizing_future = executor.submit(
                    self.sizing_agent.analyze,
                    default_allocation, market_data, risk_constraints, portfolio_value
                )

                hedging_future = executor.submit(
                    self.hedging_agent.analyze,
                    default_allocation, {},  # 初始时没有仓位
                    market_data, risk_constraints
                )

                # 收集结果 (设置超时防止无限等待)
                pm_decision = pm_future.result(timeout=120)
                sizing_decision = sizing_future.result(timeout=120)
                hedging_decision = hedging_future.result(timeout=120)
        else:
            # 顺序执行
            pm_decision = self.pm.decide(
                expert_reports, market_data, current_portfolio, risk_constraints
            )

            sizing_decision = self.sizing_agent.analyze(
                pm_decision.target_allocation, market_data,
                risk_constraints, portfolio_value
            )

            hedging_decision = self.hedging_agent.analyze(
                pm_decision.target_allocation, sizing_decision.position_sizes,
                market_data, risk_constraints
            )

        return {
            "pm": pm_decision,
            "sizing": sizing_decision,
            "hedging": hedging_decision,
        }

    def _phase2_discussion(
        self,
        pm_decision: PortfolioDecision,
        sizing_decision: PositionSizingDecision,
        hedging_decision: HedgingDecision,
        market_data: Dict[str, Any],
        risk_constraints: Dict[str, float],
    ) -> tuple:
        """
        阶段2: 讨论整合

        智能体看到彼此的决策后进行修正 (1-2轮)
        """
        current_pm = pm_decision
        current_sizing = sizing_decision
        current_hedging = hedging_decision

        rounds_completed = 0

        for round_num in range(self.max_discussion_rounds):
            rounds_completed += 1
            logger.info(f"Discussion round {round_num + 1}")

            # 构建反馈
            pm_feedback = {
                "sizing_agent": {
                    "sizing_method": current_sizing.sizing_method,
                    "position_sizes": current_sizing.position_sizes,
                    "risk_contribution": current_sizing.risk_contribution,
                },
                "hedging_agent": {
                    "strategy": current_hedging.hedging_strategy,
                    "hedge_ratio": current_hedging.hedge_ratio,
                    "tail_risk": current_hedging.tail_risk_metrics,
                }
            }

            sizing_feedback = {
                "portfolio_manager": {
                    "target_allocation": current_pm.target_allocation,
                    "expert_summary": current_pm.expert_summary,
                },
                "hedging_agent": {
                    "strategy": current_hedging.hedging_strategy,
                    "hedge_ratio": current_hedging.hedge_ratio,
                    "expected_cost": current_hedging.expected_cost,
                }
            }

            hedging_feedback = {
                "portfolio_manager": {
                    "target_allocation": current_pm.target_allocation,
                    "risk_metrics": current_pm.risk_metrics,
                },
                "sizing_agent": {
                    "position_sizes": current_sizing.position_sizes,
                    "risk_contribution": current_sizing.risk_contribution,
                }
            }

            # 并行修正
            if self.parallel_execution:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sizing_future = executor.submit(
                        self.sizing_agent.revise_based_on_feedback,
                        current_sizing, sizing_feedback, market_data
                    )
                    hedging_future = executor.submit(
                        self.hedging_agent.revise_based_on_feedback,
                        current_hedging, hedging_feedback, market_data
                    )

                    new_sizing = sizing_future.result(timeout=120)
                    new_hedging = hedging_future.result(timeout=120)
            else:
                new_sizing = self.sizing_agent.revise_based_on_feedback(
                    current_sizing, sizing_feedback, market_data
                )
                new_hedging = self.hedging_agent.revise_based_on_feedback(
                    current_hedging, hedging_feedback, market_data
                )

            # 检查是否达成共识 (变化小于阈值)
            if self._check_consensus(current_sizing, new_sizing,
                                      current_hedging, new_hedging):
                logger.info(f"Consensus reached at round {round_num + 1}")
                current_sizing = new_sizing
                current_hedging = new_hedging
                break

            current_sizing = new_sizing
            current_hedging = new_hedging

        return {
            "pm": current_pm,
            "sizing": current_sizing,
            "hedging": current_hedging,
        }, rounds_completed

    def _check_consensus(
        self,
        old_sizing: PositionSizingDecision,
        new_sizing: PositionSizingDecision,
        old_hedging: HedgingDecision,
        new_hedging: HedgingDecision,
    ) -> bool:
        """
        检查是否达成共识

        如果决策变化很小，认为已达成共识
        """
        # 检查仓位变化
        sizing_diff = 0.0
        for asset in old_sizing.position_sizes:
            old_size = old_sizing.position_sizes.get(asset, 0)
            new_size = new_sizing.position_sizes.get(asset, 0)
            sizing_diff += abs(old_size - new_size)

        # 检查对冲比例变化
        hedge_diff = abs(old_hedging.hedge_ratio - new_hedging.hedge_ratio)

        # 如果变化都很小，认为达成共识
        consensus = (sizing_diff < 0.05) and (hedge_diff < 0.02)

        logger.debug(f"Consensus check: sizing_diff={sizing_diff:.4f}, "
                     f"hedge_diff={hedge_diff:.4f}, consensus={consensus}")

        return consensus

    def _integrate_decisions(
        self,
        pm_decision: PortfolioDecision,
        sizing_decision: PositionSizingDecision,
        hedging_decision: HedgingDecision,
        current_portfolio: Dict[str, float],
        discussion_rounds: int,
    ) -> IntegratedDecision:
        """
        整合三个智能体的决策
        """
        # 使用 PM 的资产类别配置作为基础
        # 用 Sizing Agent 的仓位细化

        # 最终配置 = PM配置 * Sizing调整
        final_allocation = {}
        for asset_class, pm_weight in pm_decision.target_allocation.items():
            if asset_class in sizing_decision.position_sizes:
                # 根据 sizing agent 的建议微调
                sizing_weight = sizing_decision.position_sizes.get(asset_class, pm_weight)
                # 加权平均，PM权重0.6，Sizing权重0.4
                final_allocation[asset_class] = pm_weight * 0.6 + sizing_weight * 0.4
            else:
                final_allocation[asset_class] = pm_weight

        # 归一化
        total = sum(final_allocation.values())
        if total > 0:
            final_allocation = {k: v / total for k, v in final_allocation.items()}

        # 应用对冲调整
        if hedging_decision.hedge_ratio > 0 and hedging_decision.hedging_strategy != "none":
            # 从风险资产中减少，增加对冲资产
            hedge_ratio = hedging_decision.hedge_ratio

            # 假设对冲工具算作现金等价物
            if "cash" in final_allocation:
                final_allocation["cash"] += hedge_ratio * 0.5

            # 减少股票配置
            if "stocks" in final_allocation:
                final_allocation["stocks"] = max(0.1, final_allocation["stocks"] - hedge_ratio * 0.3)

            # 重新归一化
            total = sum(final_allocation.values())
            if total > 0:
                final_allocation = {k: v / total for k, v in final_allocation.items()}

        # 生成交易指令
        trades = self._generate_trades(current_portfolio, final_allocation)

        # 生成最终理由
        final_reasoning = self._generate_final_reasoning(
            pm_decision, sizing_decision, hedging_decision
        )

        # 检查是否达成共识
        consensus = discussion_rounds < self.max_discussion_rounds

        return IntegratedDecision(
            timestamp=datetime.now().isoformat(),
            target_allocation=final_allocation,
            position_sizes=sizing_decision.position_sizes,
            hedging_strategy=hedging_decision.hedging_strategy,
            hedge_ratio=hedging_decision.hedge_ratio,
            hedge_instruments=hedging_decision.hedge_instruments,
            trades=trades,
            risk_metrics=pm_decision.risk_metrics,
            tail_risk_metrics=hedging_decision.tail_risk_metrics,
            discussion_rounds=discussion_rounds,
            consensus_reached=consensus,
            individual_decisions={
                "pm": pm_decision.to_dict(),
                "sizing": sizing_decision.to_dict(),
                "hedging": hedging_decision.to_dict(),
            },
            final_reasoning=final_reasoning,
        )

    def _generate_trades(
        self,
        current_portfolio: Dict[str, float],
        target_allocation: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """生成交易指令"""
        trades = []
        threshold = 0.02  # 2% 阈值

        for asset, target_weight in target_allocation.items():
            current_weight = current_portfolio.get(asset, 0.0)
            diff = target_weight - current_weight

            if abs(diff) > threshold:
                action = "BUY" if diff > 0 else "SELL"
                trades.append({
                    "asset": asset,
                    "action": action,
                    "weight_change": abs(diff),
                    "from_weight": current_weight,
                    "to_weight": target_weight,
                })

        return trades

    def _generate_final_reasoning(
        self,
        pm_decision: PortfolioDecision,
        sizing_decision: PositionSizingDecision,
        hedging_decision: HedgingDecision,
    ) -> str:
        """生成最终决策理由"""
        reasoning = "【综合决策】"

        # PM 观点
        reasoning += f"\n组合管理: {pm_decision.reasoning[:100]}..."

        # Sizing 观点
        reasoning += f"\n仓位管理: {sizing_decision.reasoning[:100]}..."

        # Hedging 观点
        reasoning += f"\n对冲策略: {hedging_decision.reasoning[:100]}..."

        return reasoning

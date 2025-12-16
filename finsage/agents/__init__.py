"""
FinSage Agents Module
"""

from finsage.agents.base_expert import BaseExpert
from finsage.agents.portfolio_manager import PortfolioManager
from finsage.agents.risk_controller import RiskController
from finsage.agents.position_sizing_agent import PositionSizingAgent, PositionSizingDecision
from finsage.agents.hedging_agent import HedgingAgent, HedgingDecision
from finsage.agents.manager_coordinator import ManagerCoordinator, IntegratedDecision
from finsage.agents.factor_enhanced_expert import (
    FactorEnhancedExpertMixin,
    EnhancedRecommendation,
    create_factor_enhanced_expert,
    get_enhanced_experts,
)

__all__ = [
    "BaseExpert",
    "PortfolioManager",
    "RiskController",
    # New manager agents
    "PositionSizingAgent",
    "PositionSizingDecision",
    "HedgingAgent",
    "HedgingDecision",
    "ManagerCoordinator",
    "IntegratedDecision",
    # Factor-enhanced experts
    "FactorEnhancedExpertMixin",
    "EnhancedRecommendation",
    "create_factor_enhanced_expert",
    "get_enhanced_experts",
]

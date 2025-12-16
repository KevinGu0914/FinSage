"""
FinSage Hedging Tools
对冲策略工具集
"""

from finsage.hedging.tools.minimum_variance import MinimumVarianceTool
from finsage.hedging.tools.risk_parity import RiskParityTool
from finsage.hedging.tools.black_litterman import BlackLittermanTool
from finsage.hedging.tools.mean_variance import MeanVarianceTool
from finsage.hedging.tools.dcc_garch import DCCGARCHTool
from finsage.hedging.tools.hrp import HierarchicalRiskParityTool
from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool
from finsage.hedging.tools.robust_optimization import RobustOptimizationTool
from finsage.hedging.tools.copula_hedging import CopulaHedgingTool
from finsage.hedging.tools.factor_hedging import FactorHedgingTool
from finsage.hedging.tools.regime_switching import RegimeSwitchingTool

__all__ = [
    "MinimumVarianceTool",
    "RiskParityTool",
    "BlackLittermanTool",
    "MeanVarianceTool",
    "DCCGARCHTool",
    "HierarchicalRiskParityTool",
    "CVaROptimizationTool",
    "RobustOptimizationTool",
    "CopulaHedgingTool",
    "FactorHedgingTool",
    "RegimeSwitchingTool",
]

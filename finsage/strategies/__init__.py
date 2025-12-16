"""
Multi-Asset Allocation Strategies
多资产配置策略模块

提供不同层次的资产配置策略：
- Strategic Asset Allocation (SAA): 长期战略配置
- Tactical Asset Allocation (TAA): 短期战术调整
- Dynamic Rebalancing: 动态再平衡
- Core-Satellite: 核心卫星策略

参考文献：
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Brinson, G.P., et al. (1986). Determinants of Portfolio Performance.
- Perold, A.F. & Sharpe, W.F. (1988). Dynamic Strategies for Asset Allocation.
- Waring, M.B. & Siegel, L.B. (2003). The Dimensions of Active Management.
"""

from finsage.strategies.base_strategy import AllocationStrategy
from finsage.strategies.strategic_allocation import StrategicAllocationStrategy
from finsage.strategies.tactical_allocation import TacticalAllocationStrategy
from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy
from finsage.strategies.core_satellite import CoreSatelliteStrategy
from finsage.strategies.strategy_toolkit import StrategyToolkit

__all__ = [
    "AllocationStrategy",
    "StrategicAllocationStrategy",
    "TacticalAllocationStrategy",
    "DynamicRebalancingStrategy",
    "CoreSatelliteStrategy",
    "StrategyToolkit",
]

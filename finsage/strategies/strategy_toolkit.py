"""
Strategy Toolkit
策略工具箱 - 统一管理多资产配置策略
"""

from typing import Dict, List, Any, Optional, Type
import pandas as pd
import logging

from finsage.strategies.base_strategy import AllocationStrategy
from finsage.strategies.strategic_allocation import StrategicAllocationStrategy
from finsage.strategies.tactical_allocation import TacticalAllocationStrategy
from finsage.strategies.dynamic_rebalancing import DynamicRebalancingStrategy
from finsage.strategies.core_satellite import CoreSatelliteStrategy

logger = logging.getLogger(__name__)


class StrategyToolkit:
    """
    策略工具箱

    管理所有可用的多资产配置策略，
    供Portfolio Manager根据投资目标和市场环境选择调用。
    """

    def __init__(self):
        """初始化策略工具箱"""
        self._strategies: Dict[str, AllocationStrategy] = {}
        self._register_default_strategies()
        logger.info(f"StrategyToolkit initialized with {len(self._strategies)} strategies")

    def _register_default_strategies(self):
        """注册默认策略"""
        # Strategic Asset Allocation - 长期战略配置
        self.register(StrategicAllocationStrategy(method="mean_variance"))
        self.register(StrategicAllocationStrategy(method="risk_parity"), name="strategic_risk_parity")

        # Tactical Asset Allocation - 短期战术调整
        self.register(TacticalAllocationStrategy())

        # Dynamic Rebalancing - 动态再平衡
        self.register(DynamicRebalancingStrategy(trigger_type="threshold"))
        self.register(DynamicRebalancingStrategy(trigger_type="hybrid"), name="dynamic_hybrid")
        self.register(DynamicRebalancingStrategy(trigger_type="calendar"), name="dynamic_calendar")

        # Core-Satellite - 核心卫星策略
        self.register(CoreSatelliteStrategy(core_ratio=0.70))
        self.register(CoreSatelliteStrategy(core_ratio=0.80), name="core_satellite_conservative")
        self.register(CoreSatelliteStrategy(core_ratio=0.60), name="core_satellite_aggressive")

    def register(self, strategy: AllocationStrategy, name: Optional[str] = None):
        """
        注册策略

        Args:
            strategy: AllocationStrategy实例
            name: 可选的自定义名称
        """
        strategy_name = name or strategy.name
        self._strategies[strategy_name] = strategy
        logger.debug(f"Registered strategy: {strategy_name}")

    def unregister(self, strategy_name: str):
        """注销策略"""
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            logger.debug(f"Unregistered strategy: {strategy_name}")

    def get(self, strategy_name: str) -> Optional[AllocationStrategy]:
        """获取策略"""
        return self._strategies.get(strategy_name)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """列出所有可用策略"""
        return [
            {
                "name": name,
                "description": strategy.description,
                "rebalance_frequency": strategy.rebalance_frequency
            }
            for name, strategy in self._strategies.items()
        ]

    def compute_allocation(
        self,
        strategy_name: str,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        使用指定策略计算配置

        Args:
            strategy_name: 策略名称
            market_data: 市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            资产类别配置

        Raises:
            ValueError: 策略不存在
        """
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            available = list(self._strategies.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")

        logger.info(f"Computing allocation using strategy: {strategy_name}")

        try:
            allocation = strategy.compute_allocation(
                market_data=market_data,
                expert_views=expert_views,
                risk_profile=risk_profile,
                constraints=constraints,
                **kwargs
            )
            allocation = strategy.validate_allocation(allocation)
            logger.info(f"Strategy {strategy_name} computed allocation for {len(allocation)} asset classes")
            return allocation

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            raise

    def compare_strategies(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        比较多个策略的配置输出

        Args:
            market_data: 市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 约束条件
            strategy_names: 要比较的策略列表，None表示全部

        Returns:
            {strategy_name: allocation}
        """
        if strategy_names is None:
            strategy_names = list(self._strategies.keys())

        results = {}
        for name in strategy_names:
            try:
                allocation = self.compute_allocation(
                    strategy_name=name,
                    market_data=market_data,
                    expert_views=expert_views,
                    risk_profile=risk_profile,
                    constraints=constraints
                )
                results[name] = allocation
            except Exception as e:
                logger.warning(f"Strategy {name} comparison failed: {e}")
                results[name] = {}

        return results

    def recommend_strategy(
        self,
        market_regime: str,
        risk_profile: str,
        investment_horizon: str
    ) -> str:
        """
        根据市场状态、风险偏好和投资期限推荐策略

        Args:
            market_regime: 市场状态 (bull, bear, volatile, normal)
            risk_profile: 风险偏好 (conservative, moderate, aggressive)
            investment_horizon: 投资期限 (short, medium, long)

        Returns:
            推荐的策略名称
        """
        # 策略推荐矩阵
        recommendations = {
            # 长期投资
            ("long", "conservative"): "strategic_risk_parity",
            ("long", "moderate"): "strategic_allocation",
            ("long", "aggressive"): "core_satellite_aggressive",

            # 中期投资
            ("medium", "conservative"): "core_satellite_conservative",
            ("medium", "moderate"): "core_satellite",
            ("medium", "aggressive"): "tactical_allocation",

            # 短期投资
            ("short", "conservative"): "dynamic_rebalancing",
            ("short", "moderate"): "tactical_allocation",
            ("short", "aggressive"): "tactical_allocation",
        }

        base_recommendation = recommendations.get(
            (investment_horizon, risk_profile),
            "strategic_allocation"
        )

        # 根据市场状态调整
        if market_regime == "bear":
            # 熊市：偏向保守策略
            if "aggressive" in base_recommendation:
                base_recommendation = base_recommendation.replace("aggressive", "conservative")
            elif base_recommendation == "tactical_allocation":
                base_recommendation = "core_satellite_conservative"

        elif market_regime == "volatile":
            # 震荡市：偏向动态再平衡
            if "tactical" not in base_recommendation:
                base_recommendation = "dynamic_hybrid"

        return base_recommendation

    def get_strategy_metrics(
        self,
        strategy_name: str,
        allocation: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        获取策略配置的性能指标

        Args:
            strategy_name: 策略名称
            allocation: 配置结果
            market_data: 市场数据

        Returns:
            性能指标
        """
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            return {}

        metrics = strategy.compute_portfolio_metrics(allocation, market_data)
        metrics["strategy_name"] = strategy_name
        metrics["allocation"] = allocation

        return metrics

"""
Hedging Toolkit
对冲工具箱 - 管理和调用各种对冲策略工具
"""

from typing import Dict, List, Any, Optional, Type
import pandas as pd
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class HedgingToolkit:
    """
    对冲工具箱

    管理所有可用的对冲策略工具，
    供Portfolio Manager根据市场环境选择调用。
    """

    def __init__(self):
        """初始化工具箱"""
        self._tools: Dict[str, HedgingTool] = {}
        self._register_default_tools()
        logger.info(f"HedgingToolkit initialized with {len(self._tools)} tools")

    def _register_default_tools(self):
        """注册默认工具"""
        # 延迟导入避免循环依赖
        # 基础对冲工具
        from finsage.hedging.tools.minimum_variance import MinimumVarianceTool
        from finsage.hedging.tools.risk_parity import RiskParityTool
        from finsage.hedging.tools.black_litterman import BlackLittermanTool
        from finsage.hedging.tools.mean_variance import MeanVarianceTool
        from finsage.hedging.tools.dcc_garch import DCCGARCHTool
        from finsage.hedging.tools.hrp import HierarchicalRiskParityTool

        # 高级对冲工具 (基于金融文献)
        from finsage.hedging.tools.cvar_optimization import CVaROptimizationTool
        from finsage.hedging.tools.robust_optimization import RobustOptimizationTool
        from finsage.hedging.tools.factor_hedging import FactorHedgingTool
        from finsage.hedging.tools.regime_switching import RegimeSwitchingTool
        from finsage.hedging.tools.copula_hedging import CopulaHedgingTool

        # 注册基础工具
        self.register(MinimumVarianceTool())
        self.register(RiskParityTool())
        self.register(BlackLittermanTool())
        self.register(MeanVarianceTool())
        self.register(DCCGARCHTool())
        self.register(HierarchicalRiskParityTool())

        # 注册高级工具
        self.register(CVaROptimizationTool())       # Rockafellar & Uryasev (2000)
        self.register(RobustOptimizationTool())     # Goldfarb & Iyengar (2003)
        self.register(FactorHedgingTool())          # Fama-French (1993, 2015)
        self.register(RegimeSwitchingTool())        # Hamilton (1989)
        self.register(CopulaHedgingTool())          # Patton (2006, 2012)

    def register(self, tool: HedgingTool):
        """
        注册新工具

        Args:
            tool: HedgingTool实例
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered hedging tool: {tool.name}")

    def unregister(self, tool_name: str):
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered hedging tool: {tool_name}")

    def get(self, tool_name: str) -> Optional[HedgingTool]:
        """获取工具"""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [tool.to_dict() for tool in self._tools.values()]

    def call(
        self,
        tool_name: str,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        调用指定工具计算权重

        Args:
            tool_name: 工具名称
            returns: 收益率数据
            expert_views: 专家观点
            constraints: 风控约束
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重

        Raises:
            ValueError: 工具不存在
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            available = list(self._tools.keys())
            raise ValueError(f"Tool '{tool_name}' not found. Available: {available}")

        logger.info(f"Calling hedging tool: {tool_name}")

        try:
            weights = tool.compute_weights(
                returns=returns,
                expert_views=expert_views,
                constraints=constraints,
                **kwargs
            )
            weights = tool.validate_weights(weights)
            logger.info(f"Tool {tool_name} computed weights for {len(weights)} assets")
            return weights

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    def compare_tools(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        tool_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        比较多个工具的输出

        Args:
            returns: 收益率数据
            expert_views: 专家观点
            constraints: 风控约束
            tool_names: 要比较的工具名列表，None表示全部

        Returns:
            Dict[str, Dict[str, float]]: {tool_name: weights}
        """
        if tool_names is None:
            tool_names = list(self._tools.keys())

        results = {}
        for name in tool_names:
            try:
                weights = self.call(
                    tool_name=name,
                    returns=returns,
                    expert_views=expert_views,
                    constraints=constraints
                )
                results[name] = weights
            except Exception as e:
                logger.warning(f"Tool {name} comparison failed: {e}")
                results[name] = {}

        return results

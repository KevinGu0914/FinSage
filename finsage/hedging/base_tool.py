"""
Base Hedging Tool
对冲工具基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class HedgingTool(ABC):
    """
    对冲工具基类

    所有对冲策略工具都继承自此类，
    供Portfolio Manager调用来计算资产配置权重。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """工具参数说明"""
        return {}

    @abstractmethod
    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算资产配置权重

        Args:
            returns: 资产收益率DataFrame (columns=资产名, rows=日期)
            expert_views: 专家观点权重建议
            constraints: 风控约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重字典 {asset: weight}
        """
        pass

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """验证并规范化权重"""
        # 确保权重非负
        weights = {k: max(0, v) for k, v in weights.items()}

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典描述"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

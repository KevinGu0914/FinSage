"""
Base Allocation Strategy
资产配置策略基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AllocationStrategy(ABC):
    """
    资产配置策略基类

    定义了所有配置策略的通用接口和方法。
    """

    # 资产类别及其默认特性
    ASSET_CLASS_PROFILES = {
        "stocks": {"risk": "high", "return": "high", "liquidity": "high"},
        "bonds": {"risk": "low", "return": "low", "liquidity": "high"},
        "commodities": {"risk": "medium", "return": "medium", "liquidity": "medium"},
        "reits": {"risk": "medium", "return": "medium", "liquidity": "medium"},
        "crypto": {"risk": "very_high", "return": "very_high", "liquidity": "high"},
        "cash": {"risk": "none", "return": "very_low", "liquidity": "very_high"},
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass

    @property
    def rebalance_frequency(self) -> str:
        """再平衡频率 (daily, weekly, monthly, quarterly, annually)"""
        return "monthly"

    @abstractmethod
    def compute_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算资产类别配置

        Args:
            market_data: 各资产类别的市场数据 {asset_class: returns_df}
            expert_views: 专家观点 {asset_class: {metric: value}}
            risk_profile: 风险偏好 (conservative, moderate, aggressive)
            constraints: 配置约束
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 各资产类别的目标配置 {asset_class: weight}
        """
        pass

    def get_risk_profile_params(self, risk_profile: str) -> Dict[str, float]:
        """
        获取风险偏好对应的参数

        Args:
            risk_profile: conservative, moderate, aggressive

        Returns:
            风险参数字典
        """
        profiles = {
            "conservative": {
                "max_equity": 0.30,
                "min_fixed_income": 0.40,
                "max_alternatives": 0.15,
                "max_crypto": 0.00,
                "risk_aversion": 3.0,
                "target_volatility": 0.08,
            },
            "moderate": {
                "max_equity": 0.60,
                "min_fixed_income": 0.20,
                "max_alternatives": 0.25,
                "max_crypto": 0.05,
                "risk_aversion": 1.5,
                "target_volatility": 0.12,
            },
            "aggressive": {
                "max_equity": 0.80,
                "min_fixed_income": 0.05,
                "max_alternatives": 0.35,
                "max_crypto": 0.15,
                "risk_aversion": 0.5,
                "target_volatility": 0.18,
            },
        }
        return profiles.get(risk_profile, profiles["moderate"])

    def validate_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        验证并规范化配置

        Args:
            allocation: 原始配置

        Returns:
            规范化后的配置
        """
        # 确保非负
        allocation = {k: max(0, v) for k, v in allocation.items()}

        # 归一化使总和为1
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}
        else:
            # 如果全为0，返回等权
            n = len(allocation)
            allocation = {k: 1.0 / n for k in allocation.keys()}

        return allocation

    def compute_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        returns_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        计算组合指标

        Args:
            allocation: 配置权重
            returns_data: 各资产类别收益率数据

        Returns:
            组合指标
        """
        # 构建组合收益
        portfolio_returns = None
        for asset_class, weight in allocation.items():
            if asset_class in returns_data and weight > 0:
                asset_returns = returns_data[asset_class].mean(axis=1)
                if portfolio_returns is None:
                    portfolio_returns = weight * asset_returns
                else:
                    portfolio_returns = portfolio_returns + weight * asset_returns

        if portfolio_returns is None or len(portfolio_returns) == 0:
            return {}

        # 计算指标
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # CVaR (95%)
        var_95 = np.percentile(-portfolio_returns, 95)
        cvar_95 = -portfolio_returns[-portfolio_returns <= -var_95].mean() if len(portfolio_returns[-portfolio_returns <= -var_95]) > 0 else var_95

        return {
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "cvar_95": float(cvar_95),
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "rebalance_frequency": self.rebalance_frequency,
        }

"""
Mean-Variance Optimization
均值方差优化 - Markowitz (1952)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from finsage.hedging.base_tool import HedgingTool


class MeanVarianceTool(HedgingTool):
    """
    均值方差优化(Mean-Variance Optimization)

    经典的Markowitz投资组合优化方法，
    在给定风险水平下最大化预期收益，
    或在给定收益水平下最小化风险。

    参考文献:
    Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
    """

    @property
    def name(self) -> str:
        return "mean_variance"

    @property
    def description(self) -> str:
        return """均值方差优化(Mean-Variance Optimization)
经典Markowitz方法，在风险和收益之间寻找最优平衡。
适用场景：有明确收益预期、标准资产配置优化。
优点：理论基础扎实，可解释性强。
缺点：对输入参数敏感，尤其是预期收益估计。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "target_return": "目标收益率 (可选)",
            "target_volatility": "目标波动率 (可选)",
            "risk_free_rate": "无风险利率 (默认0.02)",
            "objective": "优化目标: 'max_sharpe', 'min_variance', 'target_return'",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算均值方差最优组合权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点的预期收益
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty:
            return {}

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 计算预期收益和协方差
        if expert_views:
            # 使用专家预期收益
            expected_returns = np.array([
                expert_views.get(asset, returns[asset].mean() * 252)
                for asset in assets
            ])
        else:
            # 使用历史收益
            expected_returns = returns.mean().values * 252

        cov_matrix = returns.cov().values * 252

        # 参数
        risk_free = kwargs.get("risk_free_rate", 0.02)
        objective = kwargs.get("objective", "max_sharpe")

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        # 定义目标函数
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def neg_sharpe_ratio(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free) / vol if vol > 0 else 0

        # 选择目标函数
        if objective == "max_sharpe":
            objective_func = neg_sharpe_ratio
        elif objective == "min_variance":
            objective_func = lambda w: np.dot(w.T, np.dot(cov_matrix, w))
        else:
            objective_func = neg_sharpe_ratio

        # 初始权重
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        # 目标收益约束
        target_return = kwargs.get("target_return")
        if target_return is not None and objective == "target_return":
            constraints_list.append({
                "type": "eq",
                "fun": lambda w: portfolio_return(w) - target_return
            })
            objective_func = lambda w: portfolio_volatility(w)

        # 边界
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        result = minimize(
            objective_func,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if result.success:
            weights = result.x
        else:
            weights = init_weights

        return dict(zip(assets, weights))

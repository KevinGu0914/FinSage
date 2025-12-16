"""
Minimum Variance Portfolio
最小方差组合 - Markowitz (1952)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from finsage.hedging.base_tool import HedgingTool


class MinimumVarianceTool(HedgingTool):
    """
    最小方差组合优化

    基于Markowitz (1952)的现代投资组合理论，
    寻找给定资产集合中方差最小的投资组合。

    参考文献:
    Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
    """

    @property
    def name(self) -> str:
        return "minimum_variance"

    @property
    def description(self) -> str:
        return """最小方差组合优化(Minimum Variance Portfolio)
基于Markowitz现代投资组合理论，寻找风险最小化的资产配置。
适用场景：高波动市场、风险厌恶型投资者、防守型配置。
优点：不依赖预期收益估计，只需协方差矩阵。
缺点：可能产生极端权重，需要配合约束条件使用。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "min_weight": "单资产最小权重 (默认0.0)",
            "max_weight": "单资产最大权重 (默认1.0)",
            "allow_short": "是否允许做空 (默认False)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算最小方差组合权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (本工具不使用)
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty:
            return {}

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 计算协方差矩阵
        cov_matrix = returns.cov().values * 252  # 年化

        # 约束条件
        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.15)

        # 目标函数: 最小化组合方差
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # 初始权重: 等权
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束: 权重和为1
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        # 边界条件
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        result = minimize(
            portfolio_variance,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if result.success:
            weights = result.x
        else:
            # 优化失败，返回等权
            weights = init_weights

        return dict(zip(assets, weights))

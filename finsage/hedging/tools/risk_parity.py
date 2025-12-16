"""
Risk Parity Portfolio
风险平价组合 - Qian (2005), Maillard et al. (2010)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from finsage.hedging.base_tool import HedgingTool


class RiskParityTool(HedgingTool):
    """
    风险平价组合优化

    使每个资产对组合总风险的贡献相等。
    目标是实现风险的均衡分配，而非资金的均衡分配。

    参考文献:
    - Qian, E. (2005). Risk Parity Portfolios. PanAgora Asset Management.
    - Maillard, S., Roncalli, T., Teïletche, J. (2010). The Properties of
      Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management.
    """

    @property
    def name(self) -> str:
        return "risk_parity"

    @property
    def description(self) -> str:
        return """风险平价组合(Risk Parity Portfolio)
使每个资产对组合总风险的贡献相等，实现风险的均衡分配。
适用场景：多资产配置、长期投资、全天候策略。
优点：风险分散均衡，对市场环境适应性强。
缺点：可能需要杠杆才能达到目标收益。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "target_risk_contribution": "目标风险贡献比例 (默认等权)",
            "budget": "风险预算 (默认均等分配)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算风险平价组合权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (可用于调整风险预算)
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

        # 风险预算 (默认等权)
        if expert_views:
            # 根据专家看好程度调整风险预算
            risk_budget = np.array([
                expert_views.get(asset, 1.0 / n_assets)
                for asset in assets
            ])
            risk_budget = risk_budget / risk_budget.sum()
        else:
            risk_budget = np.array([1.0 / n_assets] * n_assets)

        # 风险贡献函数
        def risk_contribution(weights):
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            port_std = np.sqrt(port_var)
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / port_std
            return contrib

        # 目标函数: 最小化风险贡献与目标的偏差
        def objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = risk_budget * np.sum(contrib)
            return np.sum((contrib - target_contrib) ** 2)

        # 初始权重
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        # 边界条件
        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.01)
        max_weight = constraints_dict.get("max_single_asset", 0.40)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        if result.success:
            weights = result.x
        else:
            # 优化失败，使用简化的风险平价
            weights = self._simple_risk_parity(cov_matrix)

        return dict(zip(assets, weights))

    def _simple_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """简化的风险平价: 按波动率倒数分配"""
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights

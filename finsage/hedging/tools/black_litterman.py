"""
Black-Litterman Model
Black-Litterman模型 - Black & Litterman (1992)
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from finsage.hedging.base_tool import HedgingTool


class BlackLittermanTool(HedgingTool):
    """
    Black-Litterman资产配置模型

    将市场均衡收益与投资者观点相结合，
    生成更稳定、更直观的资产配置。

    参考文献:
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
    Financial Analysts Journal.
    """

    @property
    def name(self) -> str:
        return "black_litterman"

    @property
    def description(self) -> str:
        return """Black-Litterman资产配置模型
将市场均衡收益与投资者主观观点相结合，生成更稳定的配置。
适用场景：有明确市场观点、需要融合多方意见的配置。
优点：解决了Markowitz模型对输入敏感的问题，观点表达灵活。
缺点：需要准确估计市场均衡和观点置信度。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "risk_aversion": "风险厌恶系数 (默认2.5)",
            "tau": "不确定性参数 (默认0.05)",
            "market_weights": "市场权重 (用于计算均衡收益)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算Black-Litterman组合权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 {asset: expected_return}
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

        # 参数
        risk_aversion = kwargs.get("risk_aversion", 2.5)
        tau = kwargs.get("tau", 0.05)

        # 市场权重 (默认等权)
        market_weights = kwargs.get("market_weights", None)
        if market_weights is None:
            market_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            market_weights = np.array([market_weights.get(a, 1.0/n_assets) for a in assets])
            market_weights = market_weights / market_weights.sum()

        # 计算隐含均衡收益
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        # 如果没有专家观点，使用均衡收益进行优化
        if not expert_views:
            expected_returns = pi
        else:
            # 构建观点矩阵
            P, Q, omega = self._build_view_matrices(assets, expert_views, cov_matrix, tau)

            if P is not None and len(Q) > 0:
                # Black-Litterman公式
                tau_cov = tau * cov_matrix
                tau_cov_inv = np.linalg.inv(tau_cov)
                omega_inv = np.linalg.inv(omega)

                # 后验收益 = 先验 + 调整
                M = np.linalg.inv(tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P)))
                expected_returns = np.dot(M, np.dot(tau_cov_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q)))
            else:
                expected_returns = pi

        # 使用后验收益进行均值方差优化
        weights = self._mean_variance_optimize(
            expected_returns, cov_matrix, constraints
        )

        return dict(zip(assets, weights))

    def _build_view_matrices(
        self,
        assets: List[str],
        expert_views: Dict[str, float],
        cov_matrix: np.ndarray,
        tau: float
    ) -> tuple:
        """构建Black-Litterman观点矩阵"""
        n_assets = len(assets)
        views = []
        view_returns = []

        for asset, view_return in expert_views.items():
            if asset in assets:
                idx = assets.index(asset)
                view_vec = np.zeros(n_assets)
                view_vec[idx] = 1.0
                views.append(view_vec)
                view_returns.append(view_return)

        if not views:
            return None, [], None

        P = np.array(views)
        Q = np.array(view_returns)

        # 观点不确定性矩阵 (对角矩阵，使用tau * P * Sigma * P')
        omega = tau * np.dot(P, np.dot(cov_matrix, P.T))
        # 添加小量正则化
        omega = np.diag(np.diag(omega)) + np.eye(len(Q)) * 1e-6

        return P, Q, omega

    def _mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """均值方差优化"""
        n_assets = len(expected_returns)

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)
        target_return = constraints_dict.get("target_return", None)

        # 目标函数: 最大化夏普比率
        def neg_sharpe(weights, risk_free=0.02):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - risk_free) / port_vol if port_vol > 0 else 0

        init_weights = np.array([1.0 / n_assets] * n_assets)

        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        if target_return is not None:
            constraints_list.append({
                "type": "eq",
                "fun": lambda w: np.dot(w, expected_returns) - target_return
            })

        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        result = minimize(
            neg_sharpe,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000}
        )

        return result.x if result.success else init_weights

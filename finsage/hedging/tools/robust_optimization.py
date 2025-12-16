"""
Robust Portfolio Optimization
鲁棒组合优化 - Goldfarb & Iyengar (2003)

处理参数估计的不确定性，通过考虑最坏情况来构建稳健的投资组合。
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class RobustOptimizationTool(HedgingTool):
    """
    鲁棒组合优化

    传统均值方差优化对参数估计非常敏感（尤其是预期收益）。
    鲁棒优化通过引入不确定性集，优化最坏情况下的性能。

    参考文献:
    - Goldfarb, D. & Iyengar, G. (2003). Robust Portfolio Selection Problems.
      Mathematics of Operations Research, 28(1), 1-38.
    - Tutuncu, R.H. & Koenig, M. (2004). Robust Asset Allocation.
      Annals of Operations Research.
    - Ben-Tal, A. & Nemirovski, A. (1998). Robust Convex Optimization.
      Mathematics of Operations Research.

    数学框架:
    min max { -μ'w + λ*(w'Σw) } subject to μ ∈ U_μ, Σ ∈ U_Σ

    简化版本使用椭球不确定性集:
    U_μ = { μ : ||Σ^(-1/2)(μ - μ̂)||_2 ≤ ε }
    """

    @property
    def name(self) -> str:
        return "robust_optimization"

    @property
    def description(self) -> str:
        return """鲁棒组合优化 (Robust Portfolio Optimization)
基于Goldfarb & Iyengar (2003)的方法，处理参数估计不确定性。
在最坏情况（参数估计偏差最大）下优化组合表现。
适用场景：数据有限、估计误差大、需要稳健配置的情况。
优点：对参数误差具有免疫性、避免极端权重。
缺点：可能过于保守、计算复杂度较高。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "uncertainty_level": "不确定性水平 (默认0.1，越大越保守)",
            "risk_aversion": "风险厌恶系数 (默认1.0)",
            "min_weight": "单资产最小权重 (默认0.0)",
            "max_weight": "单资产最大权重 (默认1.0)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算鲁棒最优组合权重

        使用简化的椭球不确定性集方法:
        max { μ̂'w - ε*||Σ^(1/2)w|| - λ/2 * w'Σw }

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (可调整预期收益的不确定性)
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty:
            return {}

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 参数
        uncertainty = kwargs.get("uncertainty_level", 0.1)
        risk_aversion = kwargs.get("risk_aversion", 1.0)
        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        # 估计参数
        mu_hat = returns.mean().values * 252  # 年化期望收益
        cov_matrix = returns.cov().values * 252  # 年化协方差

        # 计算协方差矩阵的平方根（Cholesky分解）
        try:
            # 确保协方差矩阵正定
            cov_matrix = self._make_positive_definite(cov_matrix)
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed, using eigenvalue approach")
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # 如果有专家观点，调整不确定性
        if expert_views:
            confidence_adjustments = np.ones(n_assets)
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    # 专家越有信心，不确定性越小
                    confidence_adjustments[idx] = 1 - abs(view) * 0.3
            uncertainty_vec = uncertainty * confidence_adjustments
        else:
            uncertainty_vec = np.ones(n_assets) * uncertainty

        # 目标函数: 最大化鲁棒效用
        # max { μ̂'w - ε*||Lw|| - λ/2 * w'Σw }
        # 等价于 min { -μ̂'w + ε*||Lw|| + λ/2 * w'Σw }
        def robust_objective(w):
            expected_return = mu_hat @ w
            # 不确定性惩罚
            uncertainty_penalty = np.mean(uncertainty_vec) * np.linalg.norm(L @ w)
            # 风险惩罚
            risk_penalty = risk_aversion / 2 * (w @ cov_matrix @ w)
            # 最小化负效用
            return -expected_return + uncertainty_penalty + risk_penalty

        # 初始权重: 等权
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        # 边界条件
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        try:
            result = minimize(
                robust_objective,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 1000, "ftol": 1e-8}
            )

            if result.success:
                weights = result.x
                logger.info(f"Robust optimization converged")
            else:
                logger.warning(f"Robust optimization did not converge: {result.message}")
                weights = init_weights

        except Exception as e:
            logger.error(f"Robust optimization failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def _make_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """
        确保矩阵正定

        Args:
            matrix: 输入矩阵
            min_eigenvalue: 最小特征值

        Returns:
            正定矩阵
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def compute_worst_case_return(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        uncertainty_level: float = 0.1
    ) -> float:
        """
        计算给定权重的最坏情况收益

        Args:
            returns: 收益率数据
            weights: 权重字典
            uncertainty_level: 不确定性水平

        Returns:
            最坏情况年化收益
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        mu_hat = returns.mean().values * 252
        cov_matrix = returns.cov().values * 252
        cov_matrix = self._make_positive_definite(cov_matrix)

        # 最坏情况: μ'w - ε*||Σ^(1/2)w||
        L = np.linalg.cholesky(cov_matrix)
        expected_return = mu_hat @ w
        uncertainty_penalty = uncertainty_level * np.linalg.norm(L @ w)

        return expected_return - uncertainty_penalty

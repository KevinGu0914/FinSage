"""
Copula-Based Hedging
Copula依赖结构对冲 - Patton (2006, 2012)

利用Copula函数建模资产间的非线性依赖关系，特别是尾部依赖。
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t, kendalltau, spearmanr
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class CopulaHedgingTool(HedgingTool):
    """
    Copula依赖结构对冲策略

    传统相关系数只能捕捉线性依赖，而Copula可以建模非线性依赖，
    特别是尾部依赖（在极端情况下的共同运动）。

    参考文献:
    - Patton, A.J. (2006). Modelling Asymmetric Exchange Rate Dependence.
      International Economic Review, 47(2), 527-556.
    - Patton, A.J. (2012). A Review of Copula Models for Economic Time Series.
      Journal of Multivariate Analysis.
    - Embrechts, P., McNeil, A.J., & Straumann, D. (2002). Correlation and
      Dependence in Risk Management: Properties and Pitfalls.
    - Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.

    Copula类型:
    1. Gaussian Copula: 对称依赖，无尾部依赖
    2. Student-t Copula: 对称依赖，有尾部依赖
    3. Clayton Copula: 下尾依赖（熊市时相关性增强）
    4. Gumbel Copula: 上尾依赖（牛市时相关性增强）

    对冲策略:
    - 高尾部依赖资产在危机时一起下跌，需要降低配置
    - 低尾部依赖资产提供更好的分散化
    """

    @property
    def name(self) -> str:
        return "copula_hedging"

    @property
    def description(self) -> str:
        return """Copula依赖结构对冲 (Copula-Based Hedging)
基于Patton (2006, 2012)的方法，建模资产间非线性依赖。
特别关注尾部依赖 - 极端情况下资产的共同运动。
适用场景：市场危机对冲、尾部风险管理、非线性相关性建模。
优点：捕捉非线性依赖、识别危机传染风险。
缺点：计算复杂、需要足够样本量。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "copula_type": "Copula类型 (gaussian/student_t/clayton/gumbel)",
            "tail_dependency_weight": "尾部依赖权重 (默认0.5)",
            "lookback_period": "回溯期 (默认120天)",
            "percentile_threshold": "尾部定义分位数 (默认0.05)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算基于Copula依赖结构的对冲权重

        核心思想：
        1. 估计资产间的尾部依赖
        2. 惩罚高尾部依赖的资产对
        3. 优化得到降低尾部风险的权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点
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
        copula_type = kwargs.get("copula_type", "student_t")
        tail_weight = kwargs.get("tail_dependency_weight", 0.5)
        lookback = kwargs.get("lookback_period", 120)
        percentile = kwargs.get("percentile_threshold", 0.05)

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        # 使用回溯期数据
        if len(returns) > lookback:
            returns_used = returns.iloc[-lookback:]
        else:
            returns_used = returns

        # Step 1: 估计尾部依赖矩阵
        lower_tail_dep, upper_tail_dep = self._estimate_tail_dependence(
            returns_used, percentile=percentile
        )

        # Step 2: 估计协方差和相关矩阵
        cov_matrix = returns_used.cov().values * 252
        cov_matrix = self._ensure_positive_definite(cov_matrix)

        # Step 3: 构建综合依赖矩阵
        # 使用下尾依赖（危机时的相关性）作为主要考量
        combined_dep = tail_weight * lower_tail_dep + (1 - tail_weight) * upper_tail_dep

        # Step 4: 构建优化目标
        # 目标: 最小化 组合方差 + λ * 尾部依赖惩罚
        mu = returns_used.mean().values * 252

        # 专家观点调整
        if expert_views:
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    mu[idx] += view * 0.03

        def copula_objective(w):
            # 组合方差
            variance = w @ cov_matrix @ w

            # 尾部依赖惩罚: 高权重资产对的尾部依赖
            tail_penalty = 0
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    tail_penalty += w[i] * w[j] * combined_dep[i, j]

            # 期望收益
            expected_return = mu @ w

            # 最小化: 方差 + 尾部惩罚 - 收益
            return variance + 0.5 * tail_penalty - 0.1 * expected_return

        # 初始权重
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # 边界
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        try:
            result = minimize(
                copula_objective,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 1000, "ftol": 1e-8}
            )

            if result.success:
                weights = result.x
                logger.info(f"Copula hedging optimization converged")
            else:
                logger.warning(f"Copula hedging did not converge: {result.message}")
                weights = init_weights

        except Exception as e:
            logger.error(f"Copula hedging failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def _estimate_tail_dependence(
        self,
        returns: pd.DataFrame,
        percentile: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计尾部依赖系数

        使用非参数方法估计下尾和上尾依赖:
        - 下尾依赖: P(U1 < u | U2 < u) as u -> 0
        - 上尾依赖: P(U1 > u | U2 > u) as u -> 1

        Args:
            returns: 收益率数据
            percentile: 尾部定义的分位数

        Returns:
            (lower_tail_matrix, upper_tail_matrix)
        """
        n_assets = len(returns.columns)
        lower_tail = np.zeros((n_assets, n_assets))
        upper_tail = np.zeros((n_assets, n_assets))

        # 转换为秩 (Probability Integral Transform)
        ranks = returns.rank() / (len(returns) + 1)

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    lower_tail[i, j] = 1.0
                    upper_tail[i, j] = 1.0
                else:
                    u_i = ranks.iloc[:, i].values
                    u_j = ranks.iloc[:, j].values

                    # 下尾依赖: P(U_j < q | U_i < q)
                    lower_mask = u_i < percentile
                    if lower_mask.sum() > 0:
                        lower_tail[i, j] = (u_j[lower_mask] < percentile).mean()
                    else:
                        lower_tail[i, j] = 0

                    # 上尾依赖: P(U_j > 1-q | U_i > 1-q)
                    upper_mask = u_i > (1 - percentile)
                    if upper_mask.sum() > 0:
                        upper_tail[i, j] = (u_j[upper_mask] > (1 - percentile)).mean()
                    else:
                        upper_tail[i, j] = 0

        # 对称化
        lower_tail = (lower_tail + lower_tail.T) / 2
        upper_tail = (upper_tail + upper_tail.T) / 2

        return lower_tail, upper_tail

    def _estimate_copula_parameters(
        self,
        returns: pd.DataFrame,
        copula_type: str = "student_t"
    ) -> Dict[str, Any]:
        """
        估计Copula参数

        Args:
            returns: 收益率数据
            copula_type: Copula类型

        Returns:
            参数字典
        """
        n_assets = len(returns.columns)

        # 计算Kendall's tau (秩相关)
        kendall_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    kendall_matrix[i, j] = 1.0
                else:
                    tau, _ = kendalltau(
                        returns.iloc[:, i].values,
                        returns.iloc[:, j].values
                    )
                    kendall_matrix[i, j] = tau if not np.isnan(tau) else 0

        if copula_type == "gaussian":
            # Gaussian Copula: ρ = sin(π * τ / 2)
            rho_matrix = np.sin(np.pi * kendall_matrix / 2)
            return {"type": "gaussian", "rho": rho_matrix}

        elif copula_type == "student_t":
            # Student-t Copula: 同样的ρ转换，加上自由度估计
            rho_matrix = np.sin(np.pi * kendall_matrix / 2)
            # 简化: 使用固定自由度
            df = 5  # 典型金融数据的自由度
            return {"type": "student_t", "rho": rho_matrix, "df": df}

        elif copula_type == "clayton":
            # Clayton Copula: θ = 2τ / (1-τ)
            theta = 2 * kendall_matrix.mean() / (1 - kendall_matrix.mean() + 1e-8)
            return {"type": "clayton", "theta": max(theta, 0.1)}

        elif copula_type == "gumbel":
            # Gumbel Copula: θ = 1 / (1-τ)
            theta = 1 / (1 - kendall_matrix.mean() + 1e-8)
            return {"type": "gumbel", "theta": max(theta, 1.0)}

        return {"type": "gaussian", "rho": np.eye(n_assets)}

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
        """确保矩阵正定"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eig)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_tail_dependence_analysis(
        self,
        returns: pd.DataFrame,
        percentile: float = 0.05
    ) -> Dict[str, Any]:
        """
        获取详细的尾部依赖分析

        Args:
            returns: 收益率数据
            percentile: 尾部分位数

        Returns:
            尾部依赖分析报告
        """
        assets = returns.columns.tolist()
        lower_tail, upper_tail = self._estimate_tail_dependence(returns, percentile)

        # Kendall's tau
        kendall_matrix = np.zeros((len(assets), len(assets)))
        for i in range(len(assets)):
            for j in range(len(assets)):
                if i != j:
                    tau, _ = kendalltau(
                        returns.iloc[:, i].values,
                        returns.iloc[:, j].values
                    )
                    kendall_matrix[i, j] = tau if not np.isnan(tau) else 0
                else:
                    kendall_matrix[i, j] = 1.0

        # 找出高尾部依赖的资产对
        high_lower_tail_pairs = []
        high_upper_tail_pairs = []

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                if lower_tail[i, j] > 0.3:  # 阈值
                    high_lower_tail_pairs.append({
                        "assets": (assets[i], assets[j]),
                        "lower_tail_dependence": float(lower_tail[i, j])
                    })
                if upper_tail[i, j] > 0.3:
                    high_upper_tail_pairs.append({
                        "assets": (assets[i], assets[j]),
                        "upper_tail_dependence": float(upper_tail[i, j])
                    })

        # 平均尾部依赖
        avg_lower = lower_tail[np.triu_indices(len(assets), k=1)].mean()
        avg_upper = upper_tail[np.triu_indices(len(assets), k=1)].mean()

        analysis = {
            "average_lower_tail_dependence": float(avg_lower),
            "average_upper_tail_dependence": float(avg_upper),
            "tail_asymmetry": float(avg_lower - avg_upper),  # 正值表示熊市相关性更强
            "high_lower_tail_pairs": high_lower_tail_pairs,
            "high_upper_tail_pairs": high_upper_tail_pairs,
            "average_kendall_tau": float(kendall_matrix[np.triu_indices(len(assets), k=1)].mean()),
            "interpretation": self._interpret_tail_dependence(avg_lower, avg_upper)
        }

        return analysis

    def _interpret_tail_dependence(
        self,
        avg_lower: float,
        avg_upper: float
    ) -> str:
        """解释尾部依赖结果"""
        if avg_lower > 0.4 and avg_upper > 0.4:
            return "资产组合在极端市场条件下（无论涨跌）都表现出高度同步，分散化效果有限"
        elif avg_lower > 0.4:
            return "资产在市场下跌时高度相关（下尾依赖强），危机时分散化效果减弱"
        elif avg_upper > 0.4:
            return "资产在市场上涨时高度相关（上尾依赖强），牛市收益可能更集中"
        elif avg_lower < 0.2 and avg_upper < 0.2:
            return "尾部依赖较低，资产在极端情况下仍能提供较好的分散化"
        else:
            return "中等尾部依赖，需要关注特定资产对的极端相关性"

    def compute_portfolio_tail_risk(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        percentile: float = 0.05
    ) -> Dict[str, float]:
        """
        计算组合的尾部风险指标

        Args:
            returns: 收益率数据
            weights: 权重字典
            percentile: 尾部分位数

        Returns:
            尾部风险指标
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.values @ w

        # VaR
        var_level = np.percentile(-portfolio_returns, (1 - percentile) * 100)

        # CVaR (Expected Shortfall)
        losses = -portfolio_returns
        cvar = losses[losses >= var_level].mean() if len(losses[losses >= var_level]) > 0 else var_level

        # 下尾概率加权
        lower_tail, _ = self._estimate_tail_dependence(returns, percentile)
        tail_risk_contribution = w @ lower_tail @ w

        return {
            "var": float(var_level),
            "cvar": float(cvar),
            "tail_risk_contribution": float(tail_risk_contribution),
            "annualized_tail_risk": float(cvar * np.sqrt(252))
        }

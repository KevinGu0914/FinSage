"""
DCC-GARCH Model for Dynamic Portfolio Allocation
DCC-GARCH动态条件相关模型 - Engle (2002)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

from finsage.hedging.base_tool import HedgingTool


class DCCGARCHTool(HedgingTool):
    """
    DCC-GARCH动态资产配置

    使用动态条件相关(DCC)模型估计时变协方差矩阵，
    捕捉资产间相关性的动态变化，实现更精准的风险对冲。

    参考文献:
    - Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of
      Multivariate GARCH Models. Journal of Business & Economic Statistics.
    - Engle, R., & Sheppard, K. (2001). Theoretical and Empirical Properties
      of Dynamic Conditional Correlation Multivariate GARCH. NBER Working Paper.
    """

    @property
    def name(self) -> str:
        return "dcc_garch"

    @property
    def description(self) -> str:
        return """DCC-GARCH动态资产配置模型
使用动态条件相关模型估计时变协方差矩阵，捕捉资产相关性变化。
适用场景：市场波动加剧期、相关性不稳定的市场环境。
优点：能够捕捉时变相关性，对冲效果更精准。
缺点：计算复杂度高，参数估计困难。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback_window": "回看窗口 (默认60天)",
            "decay_factor": "衰减因子 (默认0.94)",
            "use_ewma": "是否使用EWMA简化版 (默认True)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算DCC-GARCH动态配置权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点
            constraints: 约束条件
            **kwargs: 其他参数

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty or len(returns) < 30:
            # 数据不足，返回等权
            assets = returns.columns.tolist()
            n = len(assets)
            return {a: 1.0/n for a in assets} if n > 0 else {}

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 参数
        use_ewma = kwargs.get("use_ewma", True)
        decay_factor = kwargs.get("decay_factor", 0.94)

        if use_ewma:
            # 使用EWMA简化版
            cov_matrix = self._ewma_covariance(returns, decay_factor)
        else:
            # 完整DCC-GARCH (简化实现)
            cov_matrix = self._dcc_garch_covariance(returns)

        # 年化
        cov_matrix = cov_matrix * 252

        # 使用动态协方差进行最小方差优化
        weights = self._minimum_variance_optimize(cov_matrix, constraints)

        # 如果有专家观点，进行调整
        if expert_views:
            weights = self._adjust_by_views(weights, assets, expert_views)

        return dict(zip(assets, weights))

    def _ewma_covariance(
        self,
        returns: pd.DataFrame,
        decay_factor: float = 0.94
    ) -> np.ndarray:
        """
        计算EWMA协方差矩阵
        """
        returns_array = returns.values
        n_obs, n_assets = returns_array.shape

        # 初始协方差
        cov_matrix = np.cov(returns_array.T)

        # EWMA更新
        for t in range(1, n_obs):
            rt = returns_array[t].reshape(-1, 1)
            cov_matrix = decay_factor * cov_matrix + (1 - decay_factor) * np.dot(rt, rt.T)

        return cov_matrix

    def _dcc_garch_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        DCC-GARCH协方差估计 (简化实现)

        完整的DCC-GARCH包括:
        1. 对每个资产拟合单变量GARCH(1,1)
        2. 标准化残差
        3. 估计动态相关参数
        4. 构建时变协方差矩阵

        这里使用简化版本
        """
        returns_array = returns.values
        n_obs, n_assets = returns_array.shape

        # Step 1: 估计各资产的条件方差 (GARCH(1,1))
        conditional_vars = np.zeros((n_obs, n_assets))
        for i in range(n_assets):
            conditional_vars[:, i] = self._fit_garch11(returns_array[:, i])

        # Step 2: 标准化残差
        std_residuals = returns_array / np.sqrt(conditional_vars + 1e-8)

        # Step 3: 计算无条件相关矩阵
        Q_bar = np.corrcoef(std_residuals.T)

        # Step 4: DCC参数 (简化: 使用固定参数)
        alpha = 0.05
        beta = 0.90

        # Step 5: 动态相关矩阵
        Q_t = Q_bar.copy()
        for t in range(1, n_obs):
            et = std_residuals[t].reshape(-1, 1)
            Q_t = (1 - alpha - beta) * Q_bar + alpha * np.dot(et, et.T) + beta * Q_t

        # Step 6: 标准化得到相关矩阵
        Q_diag = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(Q_diag, Q_diag)

        # Step 7: 构建协方差矩阵
        D_t = np.diag(np.sqrt(conditional_vars[-1]))
        cov_matrix = np.dot(D_t, np.dot(R_t, D_t))

        return cov_matrix

    def _fit_garch11(self, returns: np.ndarray) -> np.ndarray:
        """
        拟合GARCH(1,1)模型

        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """
        n = len(returns)
        omega = np.var(returns) * 0.05
        alpha = 0.05
        beta = 0.90

        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        return sigma2

    def _minimum_variance_optimize(
        self,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """最小方差优化"""
        n_assets = cov_matrix.shape[0]

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        init_weights = np.array([1.0 / n_assets] * n_assets)

        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                portfolio_variance,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 1000}
            )

        return result.x if result.success else init_weights

    def _adjust_by_views(
        self,
        weights: np.ndarray,
        assets: list,
        expert_views: Dict[str, float]
    ) -> np.ndarray:
        """根据专家观点调整权重"""
        adjusted = weights.copy()
        total_adjustment = 0

        for i, asset in enumerate(assets):
            if asset in expert_views:
                view_strength = expert_views[asset]
                adjustment = (view_strength - 1.0/len(assets)) * 0.3
                adjusted[i] += adjustment
                total_adjustment += adjustment

        # 重新归一化
        adjusted = np.maximum(adjusted, 0)
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()

        return adjusted

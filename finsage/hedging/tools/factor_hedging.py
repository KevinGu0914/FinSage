"""
Factor-Based Hedging
因子对冲策略 - 基于Fama-French因子模型

通过识别和对冲系统性风险因子来降低组合风险。
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class FactorHedgingTool(HedgingTool):
    """
    因子对冲策略

    基于Fama-French因子模型，识别资产对各风险因子的敏感度（Beta），
    然后构建对冲组合以中和特定因子敞口。

    参考文献:
    - Fama, E.F. & French, K.R. (1993). Common Risk Factors in the Returns
      on Stocks and Bonds. Journal of Financial Economics.
    - Fama, E.F. & French, K.R. (2015). A Five-Factor Asset Pricing Model.
      Journal of Financial Economics.
    - Ang, A. (2014). Asset Management: A Systematic Approach to Factor Investing.

    因子模型:
    r_i = α_i + β_MKT * MKT + β_SMB * SMB + β_HML * HML + β_MOM * MOM + ε_i

    对冲策略:
    1. 估计各资产的因子Beta
    2. 选择要中和的因子（如市场因子）
    3. 构建权重使得组合的目标因子敞口为零
    """

    @property
    def name(self) -> str:
        return "factor_hedging"

    @property
    def description(self) -> str:
        return """因子对冲策略 (Factor-Based Hedging)
基于Fama-French因子模型，识别并对冲系统性风险因子。
可以选择性中和市场、规模、价值、动量等因子敞口。
适用场景：对冲系统性风险、构建市场中性策略、因子轮动。
优点：可精确控制因子敞口、理论基础扎实。
缺点：需要因子数据、因子Beta时变。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "target_factors": "要中和的因子列表 (默认['market'])",
            "factor_tolerance": "因子敞口容忍度 (默认0.1)",
            "use_momentum": "是否使用动量因子 (默认True)",
            "lookback_period": "Beta估计回溯期 (默认60天)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算因子对冲权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (可用于调整因子偏好)
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
        target_factors = kwargs.get("target_factors", ["market"])
        factor_tolerance = kwargs.get("factor_tolerance", 0.1)
        use_momentum = kwargs.get("use_momentum", True)
        lookback = kwargs.get("lookback_period", 60)

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        # 估计因子敞口
        factor_betas, factor_returns = self._estimate_factor_betas(
            returns, use_momentum=use_momentum, lookback=lookback
        )

        # 构建约束条件
        def factor_exposure(w, factor_name):
            """计算组合对某因子的敞口"""
            betas = factor_betas.get(factor_name, np.zeros(n_assets))
            return np.dot(w, betas)

        # 目标函数: 最小化剩余风险（特质风险）
        residual_cov = self._compute_residual_cov(returns, factor_betas, factor_returns)

        def residual_variance(w):
            return w @ residual_cov @ w

        # 专家观点调整
        if expert_views:
            expected_returns = returns.mean().values * 252
            view_adjustments = np.zeros(n_assets)
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    view_adjustments[idx] = view * 0.1

            adjusted_returns = expected_returns + view_adjustments

            def adjusted_objective(w):
                # 最大化调整后收益，最小化残差风险
                return -adjusted_returns @ w + 0.5 * residual_variance(w)

            objective = adjusted_objective
        else:
            objective = residual_variance

        # 初始权重
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束列表
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # 权重和为1
        ]

        # 添加因子中和约束
        for factor in target_factors:
            if factor in factor_betas:
                constraints_list.append({
                    "type": "ineq",
                    "fun": lambda w, f=factor: factor_tolerance - abs(factor_exposure(w, f))
                })

        # 边界条件
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # 优化
        try:
            result = minimize(
                objective,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 1000, "ftol": 1e-8}
            )

            if result.success:
                weights = result.x
                logger.info(f"Factor hedging optimization converged")
            else:
                logger.warning(f"Factor hedging did not converge: {result.message}")
                weights = init_weights

        except Exception as e:
            logger.error(f"Factor hedging failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def _estimate_factor_betas(
        self,
        returns: pd.DataFrame,
        use_momentum: bool = True,
        lookback: int = 60
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        估计因子Beta

        使用简化的因子构建方法（基于资产自身数据）

        Args:
            returns: 收益率数据
            use_momentum: 是否使用动量因子
            lookback: 回溯期

        Returns:
            factor_betas: {factor_name: beta_array}
            factor_returns: 因子收益DataFrame
        """
        n_assets = len(returns.columns)
        n_periods = len(returns)

        # 使用数据的最近lookback期
        if n_periods > lookback:
            returns_used = returns.iloc[-lookback:]
        else:
            returns_used = returns

        # 简化的因子构建
        factor_returns = pd.DataFrame(index=returns_used.index)

        # 1. 市场因子 (使用等权平均作为代理)
        factor_returns["market"] = returns_used.mean(axis=1)

        # 2. 规模因子 (使用波动率作为规模代理: 高波动=小盘股)
        volatilities = returns_used.std()
        high_vol = returns_used.columns[volatilities > volatilities.median()]
        low_vol = returns_used.columns[volatilities <= volatilities.median()]
        if len(high_vol) > 0 and len(low_vol) > 0:
            factor_returns["size"] = (
                returns_used[high_vol].mean(axis=1) -
                returns_used[low_vol].mean(axis=1)
            )
        else:
            factor_returns["size"] = 0

        # 3. 价值因子 (使用近期收益作为价值代理: 低收益=价值股)
        recent_returns = returns_used.sum()
        value_stocks = returns_used.columns[recent_returns < recent_returns.median()]
        growth_stocks = returns_used.columns[recent_returns >= recent_returns.median()]
        if len(value_stocks) > 0 and len(growth_stocks) > 0:
            factor_returns["value"] = (
                returns_used[value_stocks].mean(axis=1) -
                returns_used[growth_stocks].mean(axis=1)
            )
        else:
            factor_returns["value"] = 0

        # 4. 动量因子
        if use_momentum and n_periods > 20:
            # 过去一个月的动量
            past_month = returns.iloc[-22:-2].sum() if n_periods > 22 else returns.iloc[:-2].sum()
            winners = returns_used.columns[past_month > past_month.median()]
            losers = returns_used.columns[past_month <= past_month.median()]
            if len(winners) > 0 and len(losers) > 0:
                factor_returns["momentum"] = (
                    returns_used[winners].mean(axis=1) -
                    returns_used[losers].mean(axis=1)
                )
            else:
                factor_returns["momentum"] = 0

        # 估计每个资产的因子Beta (使用OLS回归)
        factor_betas = {}
        for factor in factor_returns.columns:
            betas = []
            X = factor_returns[factor].values
            for asset in returns_used.columns:
                Y = returns_used[asset].values
                # 简单OLS: beta = Cov(X,Y) / Var(X)
                cov_xy = np.cov(X, Y)[0, 1]
                var_x = np.var(X)
                if var_x > 0:
                    beta = cov_xy / var_x
                else:
                    beta = 0
                betas.append(beta)
            factor_betas[factor] = np.array(betas)

        return factor_betas, factor_returns

    def _compute_residual_cov(
        self,
        returns: pd.DataFrame,
        factor_betas: Dict[str, np.ndarray],
        factor_returns: pd.DataFrame
    ) -> np.ndarray:
        """
        计算残差协方差矩阵

        Args:
            returns: 资产收益率
            factor_betas: 因子Beta
            factor_returns: 因子收益

        Returns:
            残差协方差矩阵
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 计算因子解释的部分
        factor_cov = factor_returns.cov().values

        # 构建Beta矩阵
        B = np.column_stack([
            factor_betas.get(f, np.zeros(n_assets))
            for f in factor_returns.columns
        ])

        # 因子协方差贡献
        systematic_cov = B @ factor_cov @ B.T

        # 总协方差
        total_cov = returns.cov().values * 252

        # 残差协方差
        residual_cov = total_cov - systematic_cov

        # 确保正定
        eigenvalues = np.linalg.eigvalsh(residual_cov)
        if eigenvalues.min() < 0:
            residual_cov = residual_cov + (-eigenvalues.min() + 1e-6) * np.eye(n_assets)

        return residual_cov

    def get_factor_exposures(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算给定权重的因子敞口

        Args:
            returns: 收益率数据
            weights: 权重字典

        Returns:
            各因子敞口
        """
        factor_betas, _ = self._estimate_factor_betas(returns)
        w = np.array([weights.get(col, 0) for col in returns.columns])

        exposures = {}
        for factor, betas in factor_betas.items():
            exposures[factor] = float(np.dot(w, betas))

        return exposures

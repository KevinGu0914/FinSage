"""
CVaR (Conditional Value at Risk) Optimization
条件风险价值优化 - Rockafellar & Uryasev (2000, 2002)

CVaR是一种下行风险度量，关注尾部损失的期望值。
相比VaR，CVaR对极端损失更敏感，是一致性风险度量。
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class CVaROptimizationTool(HedgingTool):
    """
    CVaR (条件风险价值) 优化

    最小化组合的CVaR（Expected Shortfall），
    即在最差的α%情况下的平均损失。

    参考文献:
    - Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
      Value-at-Risk. Journal of Risk, 2(3), 21-41.
    - Rockafellar, R.T. & Uryasev, S. (2002). Conditional Value-at-Risk for
      General Loss Distributions. Journal of Banking & Finance.

    数学公式:
    CVaR_α(X) = E[X | X ≤ VaR_α(X)]

    对于组合优化:
    min CVaR_α(w) = min { VaR + 1/(1-α) * E[max(-r·w - VaR, 0)] }

    这是一个线性规划问题（给定样本数据）。
    """

    @property
    def name(self) -> str:
        return "cvar_optimization"

    @property
    def description(self) -> str:
        return """CVaR (条件风险价值) 优化
基于Rockafellar & Uryasev (2000)的方法，最小化组合的尾部风险。
CVaR衡量最坏α%情况下的平均损失，比VaR更适合捕捉极端风险。
适用场景：市场动荡期、尾部风险管理、监管资本计算。
优点：凸优化问题、考虑尾部损失、一致性风险度量。
缺点：对历史数据敏感、计算复杂度较高。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "alpha": "置信水平 (默认0.95，即关注最差5%情况)",
            "min_weight": "单资产最小权重 (默认0.0)",
            "max_weight": "单资产最大权重 (默认1.0)",
            "target_return": "目标收益率 (可选，用于约束)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算CVaR最优组合权重

        使用历史模拟法 + 线性规划近似

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (可用于调整目标收益)
            constraints: 约束条件
            **kwargs: 其他参数 (alpha等)

        Returns:
            Dict[str, float]: 资产权重
        """
        if returns.empty:
            return {}

        assets = returns.columns.tolist()
        n_assets = len(assets)
        n_scenarios = len(returns)

        # 参数
        alpha = kwargs.get("alpha", 0.95)
        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)
        target_return = constraints_dict.get("target_return", None)

        # 收益率矩阵
        R = returns.values  # (n_scenarios, n_assets)

        # 使用scipy优化CVaR
        # CVaR可以通过以下方式优化:
        # min_w,VaR { VaR + 1/((1-α)*S) * Σ max(-r_s·w - VaR, 0) }
        # 这是一个非光滑优化问题，我们使用近似方法

        def cvar_objective(w):
            """计算组合的CVaR（负收益的CVaR）"""
            portfolio_returns = R @ w
            # 计算VaR (第(1-alpha)分位数的损失)
            var_alpha = np.percentile(-portfolio_returns, alpha * 100)
            # CVaR是超过VaR部分的平均损失
            losses = -portfolio_returns
            tail_losses = losses[losses >= var_alpha]
            if len(tail_losses) > 0:
                cvar = tail_losses.mean()
            else:
                cvar = var_alpha
            return cvar

        def neg_expected_return(w):
            """负期望收益（用于约束）"""
            return -np.mean(R @ w)

        # 初始权重: 等权
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # 约束
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # 权重和为1
        ]

        # 如果有目标收益约束
        if target_return is not None:
            constraints_list.append({
                "type": "ineq",
                "fun": lambda w: np.mean(R @ w) * 252 - target_return
            })

        # 如果有专家观点，可以用于调整
        if expert_views:
            # 专家看好的资产给予轻微的收益调整
            view_bonus = np.zeros(n_assets)
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    view_bonus[idx] = view * 0.01  # 轻微调整

            def adjusted_cvar(w):
                portfolio_returns = (R + view_bonus) @ w
                var_alpha = np.percentile(-portfolio_returns, alpha * 100)
                losses = -portfolio_returns
                tail_losses = losses[losses >= var_alpha]
                if len(tail_losses) > 0:
                    return tail_losses.mean()
                return var_alpha

            objective = adjusted_cvar
        else:
            objective = cvar_objective

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
                logger.info(f"CVaR optimization converged: CVaR={result.fun:.4f}")
            else:
                logger.warning(f"CVaR optimization did not converge: {result.message}")
                weights = init_weights

        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def compute_portfolio_cvar(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        alpha: float = 0.95
    ) -> float:
        """
        计算给定权重的组合CVaR

        Args:
            returns: 收益率数据
            weights: 权重字典
            alpha: 置信水平

        Returns:
            CVaR值 (正数表示损失)
        """
        w = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.values @ w
        var_alpha = np.percentile(-portfolio_returns, alpha * 100)
        losses = -portfolio_returns
        tail_losses = losses[losses >= var_alpha]
        if len(tail_losses) > 0:
            return tail_losses.mean()
        return var_alpha

"""
Regime-Switching Hedging
机制转换对冲策略 - Hamilton (1989), Ang & Bekaert (2002)

根据市场状态（牛市/熊市/震荡）动态调整对冲策略。
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging

from finsage.hedging.base_tool import HedgingTool

logger = logging.getLogger(__name__)


class RegimeSwitchingTool(HedgingTool):
    """
    机制转换对冲策略

    基于市场状态的识别，在不同市场环境下采用不同的对冲策略。
    使用隐马尔可夫模型(HMM)或简化的规则识别市场状态。

    参考文献:
    - Hamilton, J.D. (1989). A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle. Econometrica.
    - Ang, A. & Bekaert, G. (2002). International Asset Allocation with
      Regime Shifts. Review of Financial Studies.
    - Guidolin, M. & Timmermann, A. (2007). Asset Allocation Under
      Multivariate Regime Switching. Journal of Economic Dynamics & Control.

    市场状态:
    1. 牛市 (Bull): 高收益、低波动 -> 增加风险敞口
    2. 熊市 (Bear): 低收益、高波动 -> 减少风险敞口、增加对冲
    3. 震荡 (Volatile): 低收益、高波动但双向 -> 中性配置

    策略适配:
    - 牛市: 风险平价/最大化夏普
    - 熊市: 最小方差/CVaR最小化
    - 震荡: 鲁棒优化/等权配置
    """

    @property
    def name(self) -> str:
        return "regime_switching"

    @property
    def description(self) -> str:
        return """机制转换对冲策略 (Regime-Switching Hedging)
基于Hamilton (1989)和Ang & Bekaert (2002)的方法。
识别市场状态（牛市/熊市/震荡），根据不同状态采用不同策略。
适用场景：市场环境多变、需要适应性策略的情况。
优点：动态适应市场、避免单一策略失效。
缺点：状态识别可能滞后、转换成本。"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback_period": "状态识别回溯期 (默认60天)",
            "n_regimes": "市场状态数量 (默认3: 牛市/熊市/震荡)",
            "transition_cost": "状态转换惩罚 (默认0.01)",
            "smoothing_window": "平滑窗口 (默认5天)",
        }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        expert_views: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算机制转换对冲权重

        Args:
            returns: 资产收益率DataFrame
            expert_views: 专家观点 (可用于辅助状态判断)
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
        lookback = kwargs.get("lookback_period", 60)
        n_regimes = kwargs.get("n_regimes", 3)
        transition_cost = kwargs.get("transition_cost", 0.01)
        smoothing = kwargs.get("smoothing_window", 5)

        constraints_dict = constraints or {}
        min_weight = constraints_dict.get("min_weight", 0.0)
        max_weight = constraints_dict.get("max_single_asset", 0.25)

        # Step 1: 识别当前市场状态
        current_regime, regime_probs = self._identify_regime(
            returns, lookback=lookback, n_regimes=n_regimes, smoothing=smoothing
        )

        logger.info(f"Identified regime: {current_regime}, probabilities: {regime_probs}")

        # Step 2: 根据状态选择策略并计算权重
        if current_regime == "bull":
            weights = self._bull_strategy(returns, min_weight, max_weight, expert_views)
        elif current_regime == "bear":
            weights = self._bear_strategy(returns, min_weight, max_weight, expert_views)
        else:  # volatile
            weights = self._volatile_strategy(returns, min_weight, max_weight, expert_views)

        # Step 3: 根据状态概率进行混合 (避免硬切换)
        if len(regime_probs) == 3:
            bull_weights = self._bull_strategy(returns, min_weight, max_weight, expert_views)
            bear_weights = self._bear_strategy(returns, min_weight, max_weight, expert_views)
            volatile_weights = self._volatile_strategy(returns, min_weight, max_weight, expert_views)

            # 概率加权混合
            blended_weights = {}
            for asset in assets:
                blended_weights[asset] = (
                    regime_probs.get("bull", 0) * bull_weights.get(asset, 0) +
                    regime_probs.get("bear", 0) * bear_weights.get(asset, 0) +
                    regime_probs.get("volatile", 0) * volatile_weights.get(asset, 0)
                )

            # 归一化
            total = sum(blended_weights.values())
            if total > 0:
                weights = {k: v / total for k, v in blended_weights.items()}
            else:
                weights = {asset: 1.0 / n_assets for asset in assets}

        return weights

    def _identify_regime(
        self,
        returns: pd.DataFrame,
        lookback: int = 60,
        n_regimes: int = 3,
        smoothing: int = 5
    ) -> Tuple[str, Dict[str, float]]:
        """
        识别市场状态

        使用简化的规则方法 (可扩展为HMM):
        - 计算近期收益和波动率
        - 基于阈值判断状态

        Args:
            returns: 收益率数据
            lookback: 回溯期
            n_regimes: 状态数量
            smoothing: 平滑窗口

        Returns:
            (当前状态, 各状态概率)
        """
        n_periods = len(returns)
        if n_periods < lookback:
            lookback = n_periods

        recent_returns = returns.iloc[-lookback:]

        # 计算市场级指标 (使用等权组合作为市场代理)
        market_returns = recent_returns.mean(axis=1)

        # 滚动统计
        if len(market_returns) >= smoothing:
            rolling_mean = market_returns.rolling(window=smoothing).mean().iloc[-1]
            rolling_vol = market_returns.rolling(window=smoothing).std().iloc[-1]
        else:
            rolling_mean = market_returns.mean()
            rolling_vol = market_returns.std()

        # 年化
        ann_return = rolling_mean * 252
        ann_vol = rolling_vol * np.sqrt(252)

        # 计算最近趋势
        if len(market_returns) >= 20:
            recent_trend = market_returns.iloc[-20:].sum()
        else:
            recent_trend = market_returns.sum()

        # 计算动量指标
        if len(market_returns) >= 10:
            momentum = market_returns.iloc[-5:].sum() - market_returns.iloc[-10:-5].sum()
        else:
            momentum = 0

        # 简化的状态判断规则
        # 牛市: 正收益 + 低波动
        # 熊市: 负收益 + 高波动
        # 震荡: 其他情况

        bull_score = 0
        bear_score = 0
        volatile_score = 0

        # 基于收益
        if ann_return > 0.10:  # 年化 > 10%
            bull_score += 2
        elif ann_return < -0.10:  # 年化 < -10%
            bear_score += 2
        else:
            volatile_score += 1

        # 基于波动率
        if ann_vol < 0.15:  # 年化波动率 < 15%
            bull_score += 1
        elif ann_vol > 0.25:  # 年化波动率 > 25%
            bear_score += 1
        else:
            volatile_score += 1

        # 基于趋势
        if recent_trend > 0.05:
            bull_score += 1
        elif recent_trend < -0.05:
            bear_score += 1

        # 基于动量
        if momentum > 0.02:
            bull_score += 1
        elif momentum < -0.02:
            bear_score += 1

        # 计算概率 (softmax-like)
        total_score = bull_score + bear_score + volatile_score + 3  # 加3避免除零
        probs = {
            "bull": (bull_score + 1) / total_score,
            "bear": (bear_score + 1) / total_score,
            "volatile": (volatile_score + 1) / total_score,
        }

        # 确定主要状态
        if bull_score > bear_score and bull_score > volatile_score:
            regime = "bull"
        elif bear_score > bull_score and bear_score > volatile_score:
            regime = "bear"
        else:
            regime = "volatile"

        return regime, probs

    def _bull_strategy(
        self,
        returns: pd.DataFrame,
        min_weight: float,
        max_weight: float,
        expert_views: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        牛市策略: 最大化夏普比率 (风险调整后收益)

        目标: max { μ'w / sqrt(w'Σw) }
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        mu = returns.mean().values * 252
        cov = returns.cov().values * 252
        cov = self._ensure_positive_definite(cov)

        # 整合专家观点
        if expert_views:
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    mu[idx] += view * 0.05  # 轻微调整

        def neg_sharpe(w):
            ret = mu @ w
            vol = np.sqrt(w @ cov @ w)
            if vol < 1e-8:
                return 1e10
            return -ret / vol

        init_weights = np.array([1.0 / n_assets] * n_assets)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        try:
            result = minimize(
                neg_sharpe, init_weights, method="SLSQP",
                bounds=bounds, constraints=constraints_list,
                options={"maxiter": 500}
            )
            weights = result.x if result.success else init_weights
        except Exception as e:
            logger.warning(f"Bull strategy optimization failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def _bear_strategy(
        self,
        returns: pd.DataFrame,
        min_weight: float,
        max_weight: float,
        expert_views: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        熊市策略: 最小化方差 + CVaR惩罚

        目标: min { w'Σw + λ * CVaR(w) }
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        cov = returns.cov().values * 252
        cov = self._ensure_positive_definite(cov)
        R = returns.values

        # 如果专家有强烈负面观点，进一步降低该资产权重
        penalty = np.zeros(n_assets)
        if expert_views:
            for asset, view in expert_views.items():
                if asset in assets and view < 0:
                    idx = assets.index(asset)
                    penalty[idx] = abs(view) * 0.1

        def bear_objective(w):
            # 方差
            variance = w @ cov @ w

            # 简化的CVaR估计 (5%尾部)
            portfolio_returns = R @ w
            var_95 = np.percentile(-portfolio_returns, 95)
            losses = -portfolio_returns
            cvar = losses[losses >= var_95].mean() if len(losses[losses >= var_95]) > 0 else var_95

            # 专家惩罚
            expert_penalty = penalty @ w

            return variance + 0.5 * cvar + expert_penalty

        init_weights = np.array([1.0 / n_assets] * n_assets)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        try:
            result = minimize(
                bear_objective, init_weights, method="SLSQP",
                bounds=bounds, constraints=constraints_list,
                options={"maxiter": 500}
            )
            weights = result.x if result.success else init_weights
        except Exception as e:
            logger.warning(f"Bear strategy optimization failed: {e}")
            weights = init_weights

        return dict(zip(assets, weights))

    def _volatile_strategy(
        self,
        returns: pd.DataFrame,
        min_weight: float,
        max_weight: float,
        expert_views: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        震荡市策略: 鲁棒等权 (稳健配置)

        接近等权但稍微偏向低波动资产
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 计算各资产波动率
        vols = returns.std().values * np.sqrt(252)

        # 反波动率权重 (波动率越低权重越高)
        inv_vols = 1.0 / (vols + 1e-8)

        # 专家调整
        if expert_views:
            for asset, view in expert_views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    inv_vols[idx] *= (1 + view * 0.1)

        # 归一化
        weights = inv_vols / inv_vols.sum()

        # 应用约束
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        return dict(zip(assets, weights))

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
        """确保矩阵正定"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eig)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_regime_analysis(
        self,
        returns: pd.DataFrame,
        lookback: int = 60
    ) -> Dict[str, Any]:
        """
        获取详细的市场状态分析

        Args:
            returns: 收益率数据
            lookback: 回溯期

        Returns:
            市场状态分析报告
        """
        regime, probs = self._identify_regime(returns, lookback=lookback)

        recent_returns = returns.iloc[-lookback:] if len(returns) >= lookback else returns
        market_returns = recent_returns.mean(axis=1)

        analysis = {
            "current_regime": regime,
            "regime_probabilities": probs,
            "market_statistics": {
                "annualized_return": float(market_returns.mean() * 252),
                "annualized_volatility": float(market_returns.std() * np.sqrt(252)),
                "sharpe_ratio": float(
                    market_returns.mean() / market_returns.std() * np.sqrt(252)
                ) if market_returns.std() > 0 else 0,
                "recent_trend": float(market_returns.iloc[-20:].sum()) if len(market_returns) >= 20 else float(market_returns.sum()),
            },
            "recommended_strategy": {
                "bull": "最大化夏普比率，增加风险敞口",
                "bear": "最小化方差+CVaR，降低风险敞口",
                "volatile": "鲁棒等权配置，保持中性",
            }.get(regime, "等权配置")
        }

        return analysis

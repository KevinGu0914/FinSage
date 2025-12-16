"""
Core-Satellite Strategy
核心卫星策略

将组合分为核心（被动管理）和卫星（主动管理）两部分。
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.strategies.base_strategy import AllocationStrategy

logger = logging.getLogger(__name__)


class CoreSatelliteStrategy(AllocationStrategy):
    """
    核心卫星策略 (Core-Satellite Strategy)

    将组合分为核心持仓（被动/指数跟踪）和卫星持仓（主动策略），
    核心提供稳定收益和分散化，卫星追求超额收益（Alpha）。

    参考文献:
    - Waring, M.B. & Siegel, L.B. (2003). The Dimensions of Active Management.
      Journal of Portfolio Management.
    - Leibowitz, M.L. & Bova, A. (2005). Allocation Betas.
      Financial Analysts Journal.
    - Amenc, N., et al. (2012). Diversifying the Diversifiers and Tracking
      the Tracking Error. Journal of Portfolio Management.

    关键特点:
    1. 核心持仓：低成本、分散化、市场暴露
    2. 卫星持仓：主动管理、追求Alpha、更高风险
    3. 灵活的核心/卫星比例
    4. 根据市场环境调整卫星持仓
    """

    def __init__(
        self,
        core_ratio: float = 0.70,
        min_core_ratio: float = 0.50,
        max_core_ratio: float = 0.90,
    ):
        """
        初始化核心卫星策略

        Args:
            core_ratio: 核心持仓比例（默认70%）
            min_core_ratio: 最小核心比例
            max_core_ratio: 最大核心比例
        """
        self.core_ratio = core_ratio
        self.min_core_ratio = min_core_ratio
        self.max_core_ratio = max_core_ratio

    @property
    def name(self) -> str:
        return "core_satellite"

    @property
    def description(self) -> str:
        return """核心卫星策略 (Core-Satellite Strategy)
将组合分为核心持仓（低成本、被动跟踪）和卫星持仓（主动策略、追求Alpha）。
核心提供市场暴露和分散化，卫星追求超额收益。
适用场景：机构投资者、有一定风险承受能力的长期投资。
优点：平衡被动与主动管理、灵活度高、成本可控。
缺点：卫星部分表现不确定、需要专业的主动管理能力。"""

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def compute_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算核心卫星配置

        Args:
            market_data: 各资产类别的市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 配置约束
            **kwargs:
                - satellite_candidates: 卫星候选资产
                - core_benchmark: 核心基准配置
                - market_regime: 市场状态

        Returns:
            资产类别配置
        """
        asset_classes = list(market_data.keys())
        n_assets = len(asset_classes)

        if not asset_classes:
            asset_classes = ["stocks", "bonds", "commodities", "reits", "cash"]

        # 获取风险偏好参数
        risk_params = self.get_risk_profile_params(risk_profile)

        # 动态调整核心比例
        market_regime = kwargs.get("market_regime", "normal")
        adjusted_core_ratio = self._adjust_core_ratio(market_regime, risk_profile)

        # Step 1: 构建核心配置
        core_benchmark = kwargs.get("core_benchmark", None)
        core_weights = self._build_core_portfolio(
            asset_classes, market_data, risk_profile, core_benchmark
        )

        # Step 2: 构建卫星配置
        satellite_candidates = kwargs.get("satellite_candidates", None)
        satellite_weights = self._build_satellite_portfolio(
            asset_classes, market_data, expert_views, risk_params, satellite_candidates
        )

        # Step 3: 混合核心和卫星
        final_weights = self._blend_portfolios(
            core_weights=core_weights,
            satellite_weights=satellite_weights,
            core_ratio=adjusted_core_ratio,
            asset_classes=asset_classes
        )

        return self.validate_allocation(final_weights)

    def _adjust_core_ratio(self, market_regime: str, risk_profile: str) -> float:
        """
        根据市场状态和风险偏好动态调整核心比例

        熊市/高波动 -> 增加核心比例
        牛市 -> 可适当降低核心比例
        """
        base_ratio = self.core_ratio

        # 市场状态调整
        regime_adjustments = {
            "bull": -0.05,      # 牛市：减少核心
            "bear": +0.10,     # 熊市：增加核心
            "volatile": +0.05,  # 震荡：略增核心
            "normal": 0,        # 正常：不调整
        }
        base_ratio += regime_adjustments.get(market_regime, 0)

        # 风险偏好调整
        profile_adjustments = {
            "conservative": +0.10,   # 保守：更多核心
            "moderate": 0,           # 中等：不调整
            "aggressive": -0.10,     # 激进：更多卫星
        }
        base_ratio += profile_adjustments.get(risk_profile, 0)

        # 限制在范围内
        return np.clip(base_ratio, self.min_core_ratio, self.max_core_ratio)

    def _build_core_portfolio(
        self,
        asset_classes: list,
        market_data: Dict[str, pd.DataFrame],
        risk_profile: str,
        benchmark: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        构建核心配置

        核心配置使用风险平价或简化的市场权重组合
        """
        if benchmark:
            return benchmark

        # 使用改进的风险平价
        volatilities = {}
        for ac in asset_classes:
            if ac in market_data and not market_data[ac].empty:
                vol = market_data[ac].mean(axis=1).std() * np.sqrt(252)
                volatilities[ac] = max(vol, 0.01)
            else:
                # 默认波动率
                defaults = {
                    "stocks": 0.18, "bonds": 0.05, "commodities": 0.15,
                    "reits": 0.14, "crypto": 0.60, "cash": 0.01
                }
                volatilities[ac] = defaults.get(ac, 0.15)

        # 反波动率权重
        inv_vols = {ac: 1.0 / vol for ac, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        core_weights = {ac: v / total_inv_vol for ac, v in inv_vols.items()}

        # 应用风险偏好的上限
        risk_params = self.get_risk_profile_params(risk_profile)

        # 限制股票类
        equity_types = ["stocks", "reits", "crypto"]
        equity_weight = sum(core_weights.get(ac, 0) for ac in equity_types if ac in core_weights)
        max_equity = risk_params.get("max_equity", 0.6)

        if equity_weight > max_equity:
            scale = max_equity / equity_weight
            for ac in equity_types:
                if ac in core_weights:
                    core_weights[ac] *= scale

        # 归一化
        total = sum(core_weights.values())
        if total > 0:
            core_weights = {k: v / total for k, v in core_weights.items()}

        return core_weights

    def _build_satellite_portfolio(
        self,
        asset_classes: list,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]],
        risk_params: Dict[str, float],
        candidates: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        构建卫星配置

        卫星配置基于：
        1. 动量信号
        2. 专家观点
        3. 预期Alpha

        返回偏向高Alpha预期资产的配置
        """
        if candidates is None:
            # 默认：所有资产都可作为卫星
            candidates = asset_classes

        n_candidates = len(candidates)
        if n_candidates == 0:
            return {}

        # 计算各候选资产的得分
        scores = {}
        for ac in candidates:
            score = 0

            # 动量得分
            if ac in market_data and not market_data[ac].empty:
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 63:
                    momentum = returns.iloc[-63:].sum()
                    score += np.tanh(momentum * 5) * 0.4

            # 专家观点得分
            if expert_views and ac in expert_views:
                view = expert_views[ac]
                sentiment = view.get("sentiment", 0)
                conviction = view.get("conviction", 0.5)
                score += sentiment * conviction * 0.4

            # 波动率惩罚（避免过高波动的资产）
            if ac in market_data and not market_data[ac].empty:
                vol = market_data[ac].mean(axis=1).std() * np.sqrt(252)
                score -= min(vol, 0.5) * 0.2

            scores[ac] = score

        # 将得分转换为权重（softmax-like）
        scores_arr = np.array([scores.get(ac, 0) for ac in candidates])
        exp_scores = np.exp(scores_arr - scores_arr.max())  # 数值稳定
        weights = exp_scores / exp_scores.sum()

        satellite_weights = dict(zip(candidates, weights))

        # 对非候选资产设为0
        for ac in asset_classes:
            if ac not in satellite_weights:
                satellite_weights[ac] = 0

        return satellite_weights

    def _blend_portfolios(
        self,
        core_weights: Dict[str, float],
        satellite_weights: Dict[str, float],
        core_ratio: float,
        asset_classes: list
    ) -> Dict[str, float]:
        """
        混合核心和卫星配置
        """
        satellite_ratio = 1 - core_ratio

        blended = {}
        for ac in asset_classes:
            core_w = core_weights.get(ac, 0)
            satellite_w = satellite_weights.get(ac, 0)
            blended[ac] = core_ratio * core_w + satellite_ratio * satellite_w

        return blended

    def get_portfolio_decomposition(
        self,
        final_weights: Dict[str, float],
        core_weights: Dict[str, float],
        satellite_weights: Dict[str, float],
        core_ratio: float
    ) -> Dict[str, Any]:
        """
        获取组合分解详情

        Args:
            final_weights: 最终权重
            core_weights: 核心权重
            satellite_weights: 卫星权重
            core_ratio: 核心比例

        Returns:
            组合分解报告
        """
        asset_classes = list(final_weights.keys())

        decomposition = {
            "core_ratio": core_ratio,
            "satellite_ratio": 1 - core_ratio,
            "core_allocation": core_weights,
            "satellite_allocation": satellite_weights,
            "final_allocation": final_weights,
            "asset_breakdown": {}
        }

        for ac in asset_classes:
            core_contrib = core_ratio * core_weights.get(ac, 0)
            satellite_contrib = (1 - core_ratio) * satellite_weights.get(ac, 0)
            decomposition["asset_breakdown"][ac] = {
                "final_weight": final_weights.get(ac, 0),
                "core_contribution": core_contrib,
                "satellite_contribution": satellite_contrib,
                "core_ratio_in_asset": core_contrib / final_weights.get(ac, 1) if final_weights.get(ac, 0) > 0 else 0
            }

        return decomposition

    def compute_tracking_error(
        self,
        satellite_weights: Dict[str, float],
        core_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        计算卫星相对核心的跟踪误差

        Args:
            satellite_weights: 卫星权重
            core_weights: 核心权重
            market_data: 市场数据

        Returns:
            年化跟踪误差
        """
        asset_classes = list(core_weights.keys())

        deviation = np.array([
            satellite_weights.get(ac, 0) - core_weights.get(ac, 0)
            for ac in asset_classes
        ])

        # 估计协方差
        returns_list = []
        for ac in asset_classes:
            if ac in market_data and not market_data[ac].empty:
                returns_list.append(market_data[ac].mean(axis=1))
            else:
                returns_list.append(pd.Series([0]))

        if returns_list:
            combined = pd.concat(returns_list, axis=1)
            combined.columns = asset_classes
            cov = combined.cov().values * 252
        else:
            cov = np.eye(len(asset_classes)) * 0.04

        tracking_error = np.sqrt(deviation @ cov @ deviation)
        return float(tracking_error)

    def get_active_share(
        self,
        satellite_weights: Dict[str, float],
        core_weights: Dict[str, float]
    ) -> float:
        """
        计算主动份额（Active Share）

        主动份额 = 0.5 * Σ|w_portfolio - w_benchmark|

        Args:
            satellite_weights: 卫星权重
            core_weights: 核心权重（作为基准）

        Returns:
            主动份额
        """
        all_assets = set(satellite_weights.keys()) | set(core_weights.keys())

        active_share = 0
        for ac in all_assets:
            diff = abs(satellite_weights.get(ac, 0) - core_weights.get(ac, 0))
            active_share += diff

        return active_share / 2

    def recommend_satellite_adjustments(
        self,
        current_satellite: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, str]:
        """
        推荐卫星调整

        Args:
            current_satellite: 当前卫星配置
            market_data: 市场数据
            expert_views: 专家观点

        Returns:
            调整建议 {asset: recommendation}
        """
        recommendations = {}
        asset_classes = list(current_satellite.keys())

        for ac in asset_classes:
            current_weight = current_satellite.get(ac, 0)

            # 计算信号
            momentum = 0
            if ac in market_data and not market_data[ac].empty:
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 21:
                    momentum = returns.iloc[-21:].sum()

            expert_sentiment = 0
            if expert_views and ac in expert_views:
                expert_sentiment = expert_views[ac].get("sentiment", 0)

            # 综合信号
            signal = momentum * 0.6 + expert_sentiment * 0.4

            # 生成建议
            if signal > 0.1 and current_weight < 0.2:
                recommendations[ac] = "INCREASE - Positive momentum and/or expert view"
            elif signal < -0.1 and current_weight > 0.05:
                recommendations[ac] = "DECREASE - Negative momentum and/or expert view"
            elif abs(signal) > 0.15:
                recommendations[ac] = f"REVIEW - Strong signal ({signal:.2f}), current weight {current_weight:.1%}"
            else:
                recommendations[ac] = "HOLD - No clear signal"

        return recommendations

"""
Tactical Asset Allocation (TAA)
战术资产配置策略

基于短期市场信号进行动态调整，追求超额收益。
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

from finsage.strategies.base_strategy import AllocationStrategy

logger = logging.getLogger(__name__)


def _is_valid_dataframe(data: Any) -> bool:
    """Check if data is a valid non-empty DataFrame."""
    return isinstance(data, pd.DataFrame) and not data.empty


class TacticalAllocationStrategy(AllocationStrategy):
    """
    战术资产配置策略 (Tactical Asset Allocation, TAA)

    在战略配置基础上，基于短期市场信号进行动态调整，
    追求超额收益（Alpha）。

    参考文献:
    - Perold, A.F. & Sharpe, W.F. (1988). Dynamic Strategies for Asset Allocation.
      Financial Analysts Journal.
    - Arnott, R.D., et al. (2010). The Fundamental Index.
    - Asness, C.S., et al. (2013). Value and Momentum Everywhere.
      Journal of Finance.

    关键特点:
    1. 短期视角（周到季度）
    2. 基于动量、估值、情绪等信号
    3. 在战略配置基础上进行偏离
    4. 控制跟踪误差（相对SAA的偏离）
    """

    def __init__(
        self,
        max_tactical_deviation: float = 0.15,
        signal_decay: float = 0.9,
    ):
        """
        初始化战术配置策略

        Args:
            max_tactical_deviation: 相对SAA的最大偏离
            signal_decay: 信号衰减系数
        """
        self.max_tactical_deviation = max_tactical_deviation
        self.signal_decay = signal_decay

    @property
    def name(self) -> str:
        return "tactical_allocation"

    @property
    def description(self) -> str:
        return """战术资产配置策略 (Tactical Asset Allocation, TAA)
基于短期市场信号（动量、估值、情绪）进行动态调整。
在战略配置基础上寻求超额收益，同时控制跟踪误差。
适用场景：市场波动期、有明确短期观点时。
优点：可捕捉短期机会、灵活应对市场变化。
缺点：交易成本较高、需要准确的市场判断。"""

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def compute_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算战术资产配置

        Args:
            market_data: 各资产类别的市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 配置约束
            **kwargs: 其他参数
                - strategic_weights: 战略配置权重（基准）
                - regime: 当前市场状态

        Returns:
            资产类别配置
        """
        asset_classes = list(market_data.keys())
        if not asset_classes:
            asset_classes = ["stocks", "bonds", "commodities", "reits", "cash"]

        n_assets = len(asset_classes)

        # 获取或使用默认的战略配置作为基准
        strategic_weights = kwargs.get("strategic_weights", None)
        if strategic_weights is None:
            strategic_weights = self._get_default_strategic_weights(risk_profile, asset_classes)

        # 计算各类信号
        momentum_signals = self._compute_momentum_signals(market_data, asset_classes)
        value_signals = self._compute_value_signals(market_data, asset_classes)
        volatility_signals = self._compute_volatility_signals(market_data, asset_classes)

        # 获取当前市场状态（如果提供）
        regime = kwargs.get("regime", "normal")

        # 综合信号计算战术调整
        tactical_adjustments = self._compute_tactical_adjustments(
            asset_classes=asset_classes,
            momentum=momentum_signals,
            value=value_signals,
            volatility=volatility_signals,
            expert_views=expert_views,
            regime=regime
        )

        # 应用战术调整到战略配置
        tactical_weights = {}
        for ac in asset_classes:
            base_weight = strategic_weights.get(ac, 1.0 / n_assets)
            adjustment = tactical_adjustments.get(ac, 0)
            # 限制偏离幅度
            adjustment = np.clip(
                adjustment,
                -self.max_tactical_deviation,
                self.max_tactical_deviation
            )
            tactical_weights[ac] = base_weight + adjustment

        # 确保权重非负
        tactical_weights = {k: max(0, v) for k, v in tactical_weights.items()}

        return self.validate_allocation(tactical_weights)

    def _get_default_strategic_weights(
        self,
        risk_profile: str,
        asset_classes: list
    ) -> Dict[str, float]:
        """获取默认战略配置"""
        profiles = {
            "conservative": {
                "stocks": 0.25, "bonds": 0.45, "commodities": 0.05,
                "reits": 0.10, "crypto": 0.00, "cash": 0.15
            },
            "moderate": {
                "stocks": 0.40, "bonds": 0.30, "commodities": 0.10,
                "reits": 0.10, "crypto": 0.02, "cash": 0.08
            },
            "aggressive": {
                "stocks": 0.55, "bonds": 0.15, "commodities": 0.10,
                "reits": 0.10, "crypto": 0.05, "cash": 0.05
            },
        }
        default = profiles.get(risk_profile, profiles["moderate"])
        # 仅返回存在的资产类别
        return {ac: default.get(ac, 1.0/len(asset_classes)) for ac in asset_classes}

    def _compute_momentum_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        asset_classes: list
    ) -> Dict[str, float]:
        """
        计算动量信号

        使用12个月动量（扣除最近1个月）
        """
        signals = {}
        for ac in asset_classes:
            if ac in market_data and _is_valid_dataframe(market_data[ac]):
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 252:
                    # 12个月动量，跳过最近1个月
                    mom_12m = returns.iloc[-252:-21].sum()
                elif len(returns) >= 63:
                    # 3个月动量
                    mom_12m = returns.iloc[-63:-5].sum()
                else:
                    mom_12m = 0

                # 标准化到[-1, 1]
                signals[ac] = np.tanh(mom_12m * 5)  # 缩放因子
            else:
                signals[ac] = 0

        return signals

    def _compute_value_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        asset_classes: list
    ) -> Dict[str, float]:
        """
        计算估值信号

        使用相对历史均值的偏离作为估值指标
        """
        signals = {}
        for ac in asset_classes:
            if ac in market_data and _is_valid_dataframe(market_data[ac]):
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 252:
                    # 当前累计收益相对历史均值
                    cumret = (1 + returns).cumprod()
                    current = cumret.iloc[-1]
                    hist_mean = cumret.iloc[-252:].mean()

                    # 如果当前低于历史均值，视为"便宜"（正信号）
                    value_score = (hist_mean - current) / hist_mean
                    signals[ac] = np.tanh(value_score * 3)
                else:
                    signals[ac] = 0
            else:
                signals[ac] = 0

        return signals

    def _compute_volatility_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        asset_classes: list
    ) -> Dict[str, float]:
        """
        计算波动率信号

        低波动率资产给予正信号（风险调整后更优）
        """
        signals = {}
        volatilities = {}

        for ac in asset_classes:
            if ac in market_data and _is_valid_dataframe(market_data[ac]):
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 21:
                    vol = returns.iloc[-21:].std() * np.sqrt(252)
                    volatilities[ac] = vol
                else:
                    volatilities[ac] = 0.2  # 默认
            else:
                volatilities[ac] = 0.2

        # 计算平均波动率
        avg_vol = np.mean(list(volatilities.values()))

        for ac in asset_classes:
            if avg_vol > 0:
                # 波动率低于平均给正信号
                signals[ac] = np.tanh((avg_vol - volatilities[ac]) / avg_vol * 2)
            else:
                signals[ac] = 0

        return signals

    def _compute_tactical_adjustments(
        self,
        asset_classes: list,
        momentum: Dict[str, float],
        value: Dict[str, float],
        volatility: Dict[str, float],
        expert_views: Optional[Dict[str, Dict[str, float]]],
        regime: str
    ) -> Dict[str, float]:
        """
        综合各信号计算战术调整

        Args:
            asset_classes: 资产类别列表
            momentum: 动量信号
            value: 估值信号
            volatility: 波动率信号
            expert_views: 专家观点
            regime: 市场状态

        Returns:
            各资产类别的调整量
        """
        # 根据市场状态调整信号权重
        regime_weights = {
            "bull": {"momentum": 0.5, "value": 0.2, "volatility": 0.1, "expert": 0.2},
            "bear": {"momentum": 0.2, "value": 0.3, "volatility": 0.3, "expert": 0.2},
            "volatile": {"momentum": 0.2, "value": 0.2, "volatility": 0.4, "expert": 0.2},
            "normal": {"momentum": 0.35, "value": 0.25, "volatility": 0.2, "expert": 0.2},
        }
        weights = regime_weights.get(regime, regime_weights["normal"])

        adjustments = {}
        for ac in asset_classes:
            # 各信号贡献
            mom_contrib = momentum.get(ac, 0) * weights["momentum"]
            val_contrib = value.get(ac, 0) * weights["value"]
            vol_contrib = volatility.get(ac, 0) * weights["volatility"]

            # 专家观点贡献
            expert_contrib = 0
            if expert_views and ac in expert_views:
                view = expert_views[ac]
                # 综合专家观点中的各个维度
                sentiment = view.get("sentiment", 0)  # -1 to 1
                conviction = view.get("conviction", 0.5)  # 0 to 1
                expert_contrib = sentiment * conviction * weights["expert"]

            # 综合调整量
            total_adjustment = mom_contrib + val_contrib + vol_contrib + expert_contrib

            # 缩放到合理范围
            adjustments[ac] = total_adjustment * self.max_tactical_deviation

        # 调整零和化（确保总调整为0）
        avg_adj = np.mean(list(adjustments.values()))
        adjustments = {k: v - avg_adj for k, v in adjustments.items()}

        return adjustments

    def get_signal_analysis(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        获取详细的信号分析

        Args:
            market_data: 市场数据

        Returns:
            各资产类别的信号详情
        """
        asset_classes = list(market_data.keys())

        momentum = self._compute_momentum_signals(market_data, asset_classes)
        value = self._compute_value_signals(market_data, asset_classes)
        volatility = self._compute_volatility_signals(market_data, asset_classes)

        analysis = {}
        for ac in asset_classes:
            analysis[ac] = {
                "momentum_signal": momentum.get(ac, 0),
                "value_signal": value.get(ac, 0),
                "volatility_signal": volatility.get(ac, 0),
                "composite_signal": (
                    momentum.get(ac, 0) * 0.4 +
                    value.get(ac, 0) * 0.3 +
                    volatility.get(ac, 0) * 0.3
                ),
            }

        return analysis

    def compute_tracking_error(
        self,
        tactical_weights: Dict[str, float],
        strategic_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        计算跟踪误差

        Args:
            tactical_weights: 战术配置
            strategic_weights: 战略配置（基准）
            market_data: 市场数据

        Returns:
            年化跟踪误差
        """
        asset_classes = list(market_data.keys())

        # 构建偏离向量
        deviation = np.array([
            tactical_weights.get(ac, 0) - strategic_weights.get(ac, 0)
            for ac in asset_classes
        ])

        # 估计协方差矩阵
        returns_list = []
        for ac in asset_classes:
            if ac in market_data and _is_valid_dataframe(market_data[ac]):
                returns_list.append(market_data[ac].mean(axis=1))
            else:
                returns_list.append(pd.Series([0]))

        if returns_list:
            combined = pd.concat(returns_list, axis=1)
            combined.columns = asset_classes
            cov = combined.cov().values * 252
        else:
            cov = np.eye(len(asset_classes)) * 0.04

        # 跟踪误差 = sqrt(deviation' * Cov * deviation)
        tracking_error = np.sqrt(deviation @ cov @ deviation)

        return float(tracking_error)

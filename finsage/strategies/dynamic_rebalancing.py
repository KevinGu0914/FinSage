"""
Dynamic Rebalancing Strategy
动态再平衡策略

基于触发条件自动调整组合配置，控制漂移风险。
"""

from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from finsage.strategies.base_strategy import AllocationStrategy

logger = logging.getLogger(__name__)


class RebalanceTrigger(Enum):
    """再平衡触发类型"""
    THRESHOLD = "threshold"      # 阈值触发
    CALENDAR = "calendar"        # 日历触发
    HYBRID = "hybrid"            # 混合触发
    VOLATILITY = "volatility"    # 波动率触发


class DynamicRebalancingStrategy(AllocationStrategy):
    """
    动态再平衡策略

    根据多种触发条件动态决定是否进行再平衡，
    同时考虑交易成本和市场冲击。

    参考文献:
    - Almgren, R. & Chriss, N. (2000). Optimal Execution of Portfolio Transactions.
      Journal of Risk.
    - Masters, S.J. (2003). Rebalancing. Journal of Portfolio Management.
    - Sun, W., et al. (2006). Optimal Rebalancing for Institutional Portfolios.
      Journal of Portfolio Management.

    关键特点:
    1. 多种触发机制（阈值/日历/波动率）
    2. 成本感知再平衡
    3. 部分再平衡（不完全调整到目标）
    4. 考虑税务影响（可选）
    """

    def __init__(
        self,
        trigger_type: str = "hybrid",
        deviation_threshold: float = 0.05,
        calendar_frequency: str = "quarterly",
        transaction_cost: float = 0.001,
    ):
        """
        初始化动态再平衡策略

        Args:
            trigger_type: 触发类型 (threshold, calendar, hybrid, volatility)
            deviation_threshold: 偏离阈值（触发再平衡的最小偏离）
            calendar_frequency: 日历频率 (monthly, quarterly, annually)
            transaction_cost: 交易成本（占交易金额的比例）
        """
        self.trigger_type = RebalanceTrigger(trigger_type)
        self.deviation_threshold = deviation_threshold
        self.calendar_frequency = calendar_frequency
        self.transaction_cost = transaction_cost

        # 记录上次再平衡
        self.last_rebalance_date = None
        self.rebalance_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "dynamic_rebalancing"

    @property
    def description(self) -> str:
        return """动态再平衡策略 (Dynamic Rebalancing)
基于阈值、日历或波动率触发的智能再平衡机制。
考虑交易成本和市场冲击，实现成本高效的组合维护。
适用场景：长期投资组合维护、养老基金管理、指数跟踪。
优点：风险可控、成本效率高、自动化程度高。
缺点：可能错过极端市场的调整时机。"""

    @property
    def rebalance_frequency(self) -> str:
        return self.calendar_frequency

    def compute_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        expert_views: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: str = "moderate",
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算再平衡后的配置

        Args:
            market_data: 市场数据
            expert_views: 专家观点
            risk_profile: 风险偏好
            constraints: 约束条件
            **kwargs:
                - current_weights: 当前配置
                - target_weights: 目标配置
                - current_date: 当前日期
                - portfolio_value: 组合总值

        Returns:
            再平衡后的配置
        """
        asset_classes = list(market_data.keys())
        n_assets = len(asset_classes)

        # 获取当前和目标配置
        current_weights = kwargs.get("current_weights", None)
        target_weights = kwargs.get("target_weights", None)
        current_date = kwargs.get("current_date", datetime.now())
        portfolio_value = kwargs.get("portfolio_value", 1000000)  # 默认100万

        # 如果没有当前配置，使用目标配置
        if current_weights is None:
            if target_weights is None:
                target_weights = self._get_default_target(risk_profile, asset_classes)
            return target_weights

        # 如果没有目标配置，获取默认
        if target_weights is None:
            target_weights = self._get_default_target(risk_profile, asset_classes)

        # 检查是否需要再平衡
        should_rebalance, trigger_reason = self._check_rebalance_trigger(
            current_weights=current_weights,
            target_weights=target_weights,
            market_data=market_data,
            current_date=current_date,
            asset_classes=asset_classes
        )

        if not should_rebalance:
            logger.info("No rebalancing needed")
            return current_weights

        logger.info(f"Rebalancing triggered: {trigger_reason}")

        # 计算最优再平衡（考虑成本）
        new_weights = self._compute_optimal_rebalance(
            current_weights=current_weights,
            target_weights=target_weights,
            market_data=market_data,
            portfolio_value=portfolio_value,
            asset_classes=asset_classes
        )

        # 记录再平衡
        self.last_rebalance_date = current_date
        self.rebalance_history.append({
            "date": current_date,
            "trigger": trigger_reason,
            "from_weights": current_weights,
            "to_weights": new_weights,
        })

        return self.validate_allocation(new_weights)

    def _get_default_target(
        self,
        risk_profile: str,
        asset_classes: list
    ) -> Dict[str, float]:
        """获取默认目标配置"""
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
        return {ac: default.get(ac, 1.0/len(asset_classes)) for ac in asset_classes}

    def _check_rebalance_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        asset_classes: list
    ) -> Tuple[bool, str]:
        """
        检查是否触发再平衡

        Returns:
            (是否触发, 触发原因)
        """
        if self.trigger_type == RebalanceTrigger.THRESHOLD:
            return self._check_threshold_trigger(current_weights, target_weights, asset_classes)

        elif self.trigger_type == RebalanceTrigger.CALENDAR:
            return self._check_calendar_trigger(current_date)

        elif self.trigger_type == RebalanceTrigger.VOLATILITY:
            return self._check_volatility_trigger(market_data, asset_classes)

        else:  # HYBRID
            # 检查阈值触发
            threshold_triggered, threshold_reason = self._check_threshold_trigger(
                current_weights, target_weights, asset_classes
            )
            if threshold_triggered:
                return True, threshold_reason

            # 检查日历触发
            calendar_triggered, calendar_reason = self._check_calendar_trigger(current_date)
            if calendar_triggered:
                # 但只有在有足够偏离时才触发
                max_deviation = self._compute_max_deviation(current_weights, target_weights, asset_classes)
                if max_deviation > self.deviation_threshold * 0.5:  # 降低阈值
                    return True, f"{calendar_reason} with deviation {max_deviation:.2%}"

            return False, "No trigger"

    def _check_threshold_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        asset_classes: list
    ) -> Tuple[bool, str]:
        """阈值触发检查"""
        max_deviation = 0
        max_deviation_asset = None

        for ac in asset_classes:
            current = current_weights.get(ac, 0)
            target = target_weights.get(ac, 0)
            deviation = abs(current - target)
            if deviation > max_deviation:
                max_deviation = deviation
                max_deviation_asset = ac

        if max_deviation > self.deviation_threshold:
            return True, f"Threshold breach: {max_deviation_asset} deviated {max_deviation:.2%}"
        return False, ""

    def _check_calendar_trigger(self, current_date: datetime) -> Tuple[bool, str]:
        """日历触发检查"""
        if self.last_rebalance_date is None:
            return True, "Initial rebalancing"

        days_since_last = (current_date - self.last_rebalance_date).days

        if self.calendar_frequency == "monthly" and days_since_last >= 30:
            return True, "Monthly calendar trigger"
        elif self.calendar_frequency == "quarterly" and days_since_last >= 90:
            return True, "Quarterly calendar trigger"
        elif self.calendar_frequency == "annually" and days_since_last >= 365:
            return True, "Annual calendar trigger"

        return False, ""

    def _check_volatility_trigger(
        self,
        market_data: Dict[str, pd.DataFrame],
        asset_classes: list
    ) -> Tuple[bool, str]:
        """波动率触发检查"""
        # 计算组合波动率变化
        current_vols = {}
        historical_vols = {}

        for ac in asset_classes:
            if ac in market_data and not market_data[ac].empty:
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 63:  # 至少3个月数据
                    current_vols[ac] = returns.iloc[-21:].std() * np.sqrt(252)
                    historical_vols[ac] = returns.iloc[-63:-21].std() * np.sqrt(252)

        if not current_vols:
            return False, ""

        # 检查波动率是否显著上升
        vol_increase = 0
        for ac in current_vols:
            if ac in historical_vols and historical_vols[ac] > 0:
                change = (current_vols[ac] - historical_vols[ac]) / historical_vols[ac]
                vol_increase = max(vol_increase, change)

        if vol_increase > 0.5:  # 波动率上升超过50%
            return True, f"Volatility spike: {vol_increase:.1%} increase"

        return False, ""

    def _compute_max_deviation(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        asset_classes: list
    ) -> float:
        """计算最大偏离"""
        return max(
            abs(current_weights.get(ac, 0) - target_weights.get(ac, 0))
            for ac in asset_classes
        )

    def _compute_optimal_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        asset_classes: list
    ) -> Dict[str, float]:
        """
        计算最优再平衡（考虑交易成本）

        使用部分再平衡策略：只调整偏离超过阈值的资产
        """
        new_weights = current_weights.copy()
        total_trade_value = 0

        # 计算各资产的偏离和所需调整
        trades = {}
        for ac in asset_classes:
            current = current_weights.get(ac, 0)
            target = target_weights.get(ac, 0)
            deviation = target - current

            # 只调整偏离超过阈值的一半
            if abs(deviation) > self.deviation_threshold * 0.5:
                # 使用部分调整（调整70%的偏离）
                adjustment = deviation * 0.7
                trades[ac] = adjustment
                total_trade_value += abs(adjustment) * portfolio_value

        # 计算交易成本
        estimated_cost = total_trade_value * self.transaction_cost

        # 如果成本太高（超过组合价值的0.5%），进行更保守的调整
        max_acceptable_cost = portfolio_value * 0.005
        if estimated_cost > max_acceptable_cost and total_trade_value > 0:
            scale = max_acceptable_cost / estimated_cost
            trades = {ac: v * scale for ac, v in trades.items()}
            logger.info(f"Scaled down trades due to cost: {scale:.2%}")

        # 应用调整
        for ac, adjustment in trades.items():
            new_weights[ac] = current_weights.get(ac, 0) + adjustment

        # 确保非负并归一化
        new_weights = {k: max(0, v) for k, v in new_weights.items()}
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def get_rebalance_analysis(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float = 1000000
    ) -> Dict[str, Any]:
        """
        获取再平衡分析

        Args:
            current_weights: 当前配置
            target_weights: 目标配置
            portfolio_value: 组合价值

        Returns:
            再平衡分析报告
        """
        asset_classes = list(set(current_weights.keys()) | set(target_weights.keys()))

        deviations = {}
        trades = {}
        total_trade_value = 0

        for ac in asset_classes:
            current = current_weights.get(ac, 0)
            target = target_weights.get(ac, 0)
            deviation = target - current
            deviations[ac] = deviation
            trade_value = abs(deviation) * portfolio_value
            trades[ac] = {
                "current_weight": current,
                "target_weight": target,
                "deviation": deviation,
                "trade_value": trade_value,
                "action": "buy" if deviation > 0 else "sell" if deviation < 0 else "hold"
            }
            total_trade_value += trade_value

        estimated_cost = total_trade_value * self.transaction_cost
        max_deviation = max(abs(d) for d in deviations.values())

        return {
            "trades": trades,
            "total_trade_value": total_trade_value,
            "estimated_cost": estimated_cost,
            "cost_as_percent_of_portfolio": estimated_cost / portfolio_value * 100,
            "max_deviation": max_deviation,
            "needs_rebalancing": max_deviation > self.deviation_threshold,
            "trigger_threshold": self.deviation_threshold,
        }

    def estimate_drift(
        self,
        initial_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        days_forward: int = 30
    ) -> Dict[str, float]:
        """
        估计未来漂移

        基于历史波动率估计权重可能的漂移范围

        Args:
            initial_weights: 初始权重
            market_data: 市场数据
            days_forward: 预测天数

        Returns:
            预期最大漂移
        """
        asset_classes = list(initial_weights.keys())
        expected_drift = {}

        for ac in asset_classes:
            if ac in market_data and not market_data[ac].empty:
                returns = market_data[ac].mean(axis=1)
                if len(returns) >= 21:
                    daily_vol = returns.std()
                    # 使用简化模型：漂移 ~ vol * sqrt(days)
                    expected_vol = daily_vol * np.sqrt(days_forward)
                    initial = initial_weights.get(ac, 0)
                    # 权重变化 ≈ 初始权重 * 价格变化
                    expected_drift[ac] = initial * expected_vol * 2  # 2 sigma
                else:
                    expected_drift[ac] = 0.02
            else:
                expected_drift[ac] = 0.02  # 默认2%漂移

        return expected_drift

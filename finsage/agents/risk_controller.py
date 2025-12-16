"""
Risk Controller Agent
风险控制智能体 - 负责监控和约束组合风险

增强功能:
- 日内监控机制 (VIX spike / Drawdown trigger)
- 紧急防御性调整
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from finsage.config import AssetConfig

logger = logging.getLogger(__name__)


@dataclass
class IntradayAlert:
    """日内监控警报"""
    timestamp: str
    alert_type: str  # "vix_spike", "drawdown_trigger", "volatility_regime"
    severity: str    # "warning", "critical", "emergency"
    current_value: float
    threshold: float
    message: str
    recommended_action: str


@dataclass
class RiskAssessment:
    """风险评估结果"""
    timestamp: str
    portfolio_var_95: float           # 95% VaR
    portfolio_cvar_99: float          # 99% CVaR
    current_drawdown: float           # 当前回撤
    max_drawdown: float               # 最大回撤
    volatility: float                 # 波动率
    sharpe_ratio: float               # 夏普比率
    concentration_risk: str           # 集中度风险等级
    violations: List[str]             # 违规项
    warnings: List[str]               # 警告项
    veto: bool                        # 是否否决交易
    recommendations: Dict[str, Any]   # 调整建议
    # 日内监控警报
    intraday_alerts: List[IntradayAlert] = field(default_factory=list)
    emergency_rebalance: bool = False  # 是否触发紧急再平衡
    defensive_allocation: Optional[Dict[str, float]] = None  # 防御性配置建议

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "portfolio_var_95": self.portfolio_var_95,
            "portfolio_cvar_99": self.portfolio_cvar_99,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "concentration_risk": self.concentration_risk,
            "violations": self.violations,
            "warnings": self.warnings,
            "veto": self.veto,
            "recommendations": self.recommendations,
            "intraday_alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "action": a.recommended_action,
                }
                for a in self.intraday_alerts
            ],
            "emergency_rebalance": self.emergency_rebalance,
            "defensive_allocation": self.defensive_allocation,
        }


class RiskController:
    """
    风险控制智能体

    核心职责:
    1. 监控组合风险指标
    2. 检查约束条件违规
    3. 提供风险调整建议
    4. 行使否决权阻止高风险交易
    5. 日内监控 (VIX spike / Drawdown trigger)
    """

    # 硬性约束 (违反则否决)
    DEFAULT_HARD_LIMITS = {
        "max_single_asset": 0.15,       # 单资产最大权重
        "max_asset_class": 0.50,        # 单类别最大权重
        "max_drawdown_trigger": 0.15,   # 触发减仓的回撤
        "max_portfolio_var_95": 0.03,   # 日VaR上限
    }

    # 软性约束 (违反则警告)
    DEFAULT_SOFT_LIMITS = {
        "target_volatility": 0.12,      # 目标年化波动率
        "max_correlation_cluster": 0.60, # 高相关资产合计
        "min_diversification_ratio": 1.2, # 最低分散化比率
    }

    # 日内监控阈值
    INTRADAY_THRESHOLDS = {
        # VIX 阈值
        "vix_warning": 25.0,            # VIX > 25: 警告
        "vix_critical": 30.0,           # VIX > 30: 危急 (减少风险敞口)
        "vix_emergency": 40.0,          # VIX > 40: 紧急 (触发防御性配置)

        # 回撤阈值
        "drawdown_warning": -0.05,      # 5% 回撤: 警告
        "drawdown_critical": -0.10,     # 10% 回撤: 危急
        "drawdown_emergency": -0.15,    # 15% 回撤: 紧急减仓

        # 波动率突变检测
        "vol_spike_multiplier": 1.5,    # 波动率突然增加 50%: 触发警告
    }

    # 防御性配置 (紧急情况下使用)
    DEFENSIVE_ALLOCATION = {
        "stocks": 0.20,      # 减少股票敞口
        "bonds": 0.40,       # 增加债券避险
        "commodities": 0.15, # 黄金避险
        "reits": 0.05,       # 减少 REITs
        "crypto": 0.00,      # 清空加密货币
        "cash": 0.20,        # 增加现金
    }

    def __init__(
        self,
        hard_limits: Optional[Dict] = None,
        soft_limits: Optional[Dict] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化风控Agent

        Args:
            hard_limits: 硬性约束
            soft_limits: 软性约束
            config: 其他配置
        """
        self.hard_limits = hard_limits or self.DEFAULT_HARD_LIMITS
        self.soft_limits = soft_limits or self.DEFAULT_SOFT_LIMITS
        self.config = config or {}

        # 历史数据
        self.portfolio_history = []
        self.peak_value = 1.0
        self.max_drawdown_history = 0.0

        # 日内监控状态
        self.last_vix = None
        self.last_volatility = None
        self.intraday_alerts_history: List[IntradayAlert] = []

        logger.info("Risk Controller initialized (with intraday monitoring)")

    def assess(
        self,
        current_allocation: Dict[str, float],
        proposed_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        portfolio_value: float,
    ) -> RiskAssessment:
        """
        评估当前和提议的配置风险

        Args:
            current_allocation: 当前配置
            proposed_allocation: 提议配置
            market_data: 市场数据
            portfolio_value: 组合价值

        Returns:
            RiskAssessment: 风险评估结果
        """
        violations = []
        warnings = []
        veto = False

        # 1. 检查硬性约束
        hard_violations = self._check_hard_limits(proposed_allocation, market_data)
        if hard_violations:
            violations.extend(hard_violations)
            veto = True

        # 2. 检查软性约束
        soft_violations = self._check_soft_limits(proposed_allocation, market_data)
        warnings.extend(soft_violations)

        # 3. 计算风险指标
        var_95 = self._calculate_var(proposed_allocation, market_data, 0.95)
        cvar_99 = self._calculate_cvar(proposed_allocation, market_data, 0.99)
        volatility = self._calculate_volatility(proposed_allocation, market_data)
        sharpe = self._calculate_sharpe(proposed_allocation, market_data)

        # 4. 计算回撤
        current_drawdown, max_drawdown = self._calculate_drawdown(portfolio_value)

        # 5. 评估集中度风险
        concentration_risk = self._assess_concentration(proposed_allocation)

        # 6. 生成调整建议
        recommendations = self._generate_recommendations(
            proposed_allocation, violations, warnings
        )

        assessment = RiskAssessment(
            timestamp=datetime.now().isoformat(),
            portfolio_var_95=var_95,
            portfolio_cvar_99=cvar_99,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe,
            concentration_risk=concentration_risk,
            violations=violations,
            warnings=warnings,
            veto=veto,
            recommendations=recommendations,
        )

        if veto:
            logger.warning(f"Risk Controller VETO: {violations}")
        elif warnings:
            logger.info(f"Risk Controller warnings: {warnings}")

        return assessment

    def _check_hard_limits(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> List[str]:
        """检查硬性约束"""
        violations = []

        # 定义资产类别名称
        asset_classes = {"stocks", "bonds", "commodities", "reits", "crypto", "cash"}

        # 检查单资产权重 (只对非资产类别名称检查)
        for asset, weight in allocation.items():
            # 如果是资产类别名称，使用 max_asset_class 限制
            max_asset_class = self.hard_limits.get("max_asset_class", 0.50)
            max_single_asset = self.hard_limits.get("max_single_asset", 0.15)
            if asset in asset_classes:
                if weight > max_asset_class:
                    violations.append(
                        f"资产类别{asset}权重{weight:.1%}超过限制{max_asset_class:.1%}"
                    )
            else:
                # 对具体资产使用 max_single_asset 限制
                if weight > max_single_asset:
                    violations.append(
                        f"单资产{asset}权重{weight:.1%}超过限制{max_single_asset:.1%}"
                    )

        # 如果传入的是具体资产符号，检查资产类别聚合权重
        has_individual_assets = any(a not in asset_classes for a in allocation.keys())
        if has_individual_assets:
            class_weights = self._aggregate_by_class(allocation)
            for asset_class, weight in class_weights.items():
                if weight > max_asset_class:
                    violations.append(
                        f"资产类别{asset_class}权重{weight:.1%}超过限制{max_asset_class:.1%}"
                    )

        return violations

    def _check_soft_limits(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> List[str]:
        """检查软性约束"""
        warnings = []

        # 检查波动率
        volatility = self._calculate_volatility(allocation, market_data)
        if volatility > self.soft_limits["target_volatility"]:
            warnings.append(
                f"预期波动率{volatility:.1%}超过目标{self.soft_limits['target_volatility']:.1%}"
            )

        # 检查分散化
        div_ratio = self._calculate_diversification_ratio(allocation, market_data)
        if div_ratio < self.soft_limits["min_diversification_ratio"]:
            warnings.append(
                f"分散化比率{div_ratio:.2f}低于目标{self.soft_limits['min_diversification_ratio']:.2f}"
            )

        return warnings

    def _aggregate_by_class(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """按资产类别聚合权重"""
        # 从 config.py AssetConfig 动态获取，支持扩展
        class_mapping = AssetConfig().default_universe.copy()
        # 添加常见变体符号以兼容不同数据源
        if "crypto" in class_mapping:
            class_mapping["crypto"] = class_mapping["crypto"] + ["BTCUSD", "ETHUSD"]

        class_weights = {c: 0.0 for c in class_mapping.keys()}
        class_weights["other"] = 0.0  # 用于未分类资产

        for asset, weight in allocation.items():
            if asset == "cash":
                continue  # cash 单独处理
            found = False
            for asset_class, symbols in class_mapping.items():
                if asset in symbols or asset == asset_class:
                    class_weights[asset_class] += weight
                    found = True
                    break
            if not found:
                class_weights["other"] += weight

        return class_weights

    def _calculate_var(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any],
        confidence: float = 0.95
    ) -> float:
        """计算VaR"""
        # 简化实现 - 使用历史模拟或参数法
        volatility = self._calculate_volatility(allocation, market_data)
        z_score = 1.645 if confidence == 0.95 else 2.326
        return volatility * z_score / np.sqrt(252)

    def _calculate_cvar(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any],
        confidence: float = 0.99
    ) -> float:
        """计算CVaR (Expected Shortfall)"""
        var = self._calculate_var(allocation, market_data, confidence)
        # 简化: CVaR ≈ 1.3 * VaR
        return var * 1.3

    def _calculate_volatility(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> float:
        """计算组合波动率"""
        # 简化实现
        asset_vols = market_data.get("volatilities", {})
        if not asset_vols:
            return 0.15  # 默认波动率

        weighted_vol = sum(
            allocation.get(asset, 0) * vol
            for asset, vol in asset_vols.items()
        )
        # 考虑分散化效应，打折
        return weighted_vol * 0.8

    def _calculate_sharpe(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any],
        risk_free: float = 0.05
    ) -> float:
        """计算夏普比率"""
        expected_returns = market_data.get("expected_returns", {})
        if not expected_returns:
            return 0.5  # 默认夏普

        portfolio_return = sum(
            allocation.get(asset, 0) * ret
            for asset, ret in expected_returns.items()
        )
        volatility = self._calculate_volatility(allocation, market_data)

        if volatility > 0:
            return (portfolio_return - risk_free) / volatility
        return 0

    def _calculate_drawdown(self, portfolio_value: float) -> tuple:
        """计算回撤"""
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # 当前回撤 - 添加零值检查
        if self.peak_value > 0:
            current_drawdown = (portfolio_value - self.peak_value) / self.peak_value
        else:
            current_drawdown = 0.0

        # 更新最大回撤
        if current_drawdown < self.max_drawdown_history:
            self.max_drawdown_history = current_drawdown

        return current_drawdown, self.max_drawdown_history

    def _calculate_diversification_ratio(
        self,
        allocation: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> float:
        """计算分散化比率"""
        asset_vols = market_data.get("volatilities", {})
        if not asset_vols:
            return 1.5

        weighted_avg_vol = sum(
            allocation.get(asset, 0) * vol
            for asset, vol in asset_vols.items()
        )
        portfolio_vol = self._calculate_volatility(allocation, market_data)

        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        return 1.0

    def _assess_concentration(self, allocation: Dict[str, float]) -> str:
        """评估集中度风险"""
        max_weight = max(allocation.values()) if allocation else 0

        if max_weight > 0.3:
            return "high"
        elif max_weight > 0.2:
            return "medium"
        return "low"

    def _generate_recommendations(
        self,
        allocation: Dict[str, float],
        violations: List[str],
        warnings: List[str]
    ) -> Dict[str, Any]:
        """生成调整建议"""
        recommendations = {}

        # 针对违规提建议
        for violation in violations:
            if "单资产" in violation:
                # 建议减少集中持仓
                for asset, weight in allocation.items():
                    if weight > self.hard_limits["max_single_asset"]:
                        recommendations[f"reduce_{asset}"] = {
                            "action": "reduce",
                            "asset": asset,
                            "from": weight,
                            "to": self.hard_limits["max_single_asset"],
                        }

        # 针对警告提建议
        for warning in warnings:
            if "波动率" in warning:
                recommendations["reduce_volatility"] = {
                    "action": "增加债券或现金配置",
                    "target_vol": self.soft_limits["target_volatility"],
                }

        return recommendations

    def get_constraints(self) -> Dict[str, float]:
        """
        获取约束条件供Portfolio Manager使用

        包含:
        - 硬性约束 (hard_limits)
        - 软性约束 (soft_limits)
        - 当前风险状态 (current_drawdown, max_drawdown)
        """
        # 计算当前回撤
        current_drawdown = 0.0
        max_drawdown = self.max_drawdown_history

        if self.portfolio_history:
            # 使用最近的组合价值计算回撤
            current_value = self.portfolio_history[-1] if self.portfolio_history else 1.0
            if self.peak_value > 1e-10:  # 使用 epsilon 值避免浮点精度问题
                current_drawdown = (current_value - self.peak_value) / self.peak_value

        return {
            **self.hard_limits,
            **self.soft_limits,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
        }

    # ==================== 日内监控方法 ====================

    def check_intraday_alerts(
        self,
        market_data: Dict[str, Any],
        current_drawdown: float,
    ) -> List[IntradayAlert]:
        """
        检查日内监控警报

        Args:
            market_data: 市场数据 (需包含 macro.vix)
            current_drawdown: 当前回撤 (负数)

        Returns:
            警报列表
        """
        alerts = []
        timestamp = datetime.now().isoformat()

        # 1. 检查 VIX 警报
        vix_alert = self._check_vix_alert(market_data, timestamp)
        if vix_alert:
            alerts.append(vix_alert)

        # 2. 检查回撤警报
        drawdown_alert = self._check_drawdown_alert(current_drawdown, timestamp)
        if drawdown_alert:
            alerts.append(drawdown_alert)

        # 3. 检查波动率突变
        vol_alert = self._check_volatility_spike(market_data, timestamp)
        if vol_alert:
            alerts.append(vol_alert)

        # 记录历史
        self.intraday_alerts_history.extend(alerts)

        return alerts

    def _check_vix_alert(
        self,
        market_data: Dict[str, Any],
        timestamp: str,
    ) -> Optional[IntradayAlert]:
        """检查 VIX 警报"""
        macro = market_data.get("macro", {})
        vix = macro.get("vix")

        if vix is None:
            return None

        # 更新最近 VIX 值
        self.last_vix = vix

        thresholds = self.INTRADAY_THRESHOLDS

        if vix >= thresholds["vix_emergency"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="vix_spike",
                severity="emergency",
                current_value=vix,
                threshold=thresholds["vix_emergency"],
                message=f"VIX 恐慌 ({vix:.1f}) - 触发紧急防御模式",
                recommended_action="立即切换至防御性配置，清空高风险资产",
            )
        elif vix >= thresholds["vix_critical"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="vix_spike",
                severity="critical",
                current_value=vix,
                threshold=thresholds["vix_critical"],
                message=f"VIX 危急 ({vix:.1f}) - 建议减少风险敞口",
                recommended_action="减少股票/加密货币敞口 30%，增加债券/现金",
            )
        elif vix >= thresholds["vix_warning"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="vix_spike",
                severity="warning",
                current_value=vix,
                threshold=thresholds["vix_warning"],
                message=f"VIX 警告 ({vix:.1f}) - 市场波动加剧",
                recommended_action="密切关注，考虑减少高Beta资产",
            )

        return None

    def _check_drawdown_alert(
        self,
        current_drawdown: float,
        timestamp: str,
    ) -> Optional[IntradayAlert]:
        """检查回撤警报"""
        thresholds = self.INTRADAY_THRESHOLDS

        if current_drawdown <= thresholds["drawdown_emergency"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="drawdown_trigger",
                severity="emergency",
                current_value=current_drawdown,
                threshold=thresholds["drawdown_emergency"],
                message=f"回撤紧急 ({current_drawdown:.1%}) - 触发紧急减仓",
                recommended_action="执行紧急减仓，切换至防御性配置",
            )
        elif current_drawdown <= thresholds["drawdown_critical"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="drawdown_trigger",
                severity="critical",
                current_value=current_drawdown,
                threshold=thresholds["drawdown_critical"],
                message=f"回撤危急 ({current_drawdown:.1%}) - 建议减少持仓",
                recommended_action="减少 50% 股票持仓，增加债券/现金",
            )
        elif current_drawdown <= thresholds["drawdown_warning"]:
            return IntradayAlert(
                timestamp=timestamp,
                alert_type="drawdown_trigger",
                severity="warning",
                current_value=current_drawdown,
                threshold=thresholds["drawdown_warning"],
                message=f"回撤警告 ({current_drawdown:.1%}) - 加强监控",
                recommended_action="密切关注，准备减仓计划",
            )

        return None

    def _check_volatility_spike(
        self,
        market_data: Dict[str, Any],
        timestamp: str,
    ) -> Optional[IntradayAlert]:
        """检查波动率突变"""
        # 从市场数据获取当前波动率
        volatilities = market_data.get("volatilities", {})
        if not volatilities:
            return None

        # 计算平均波动率
        current_avg_vol = np.mean(list(volatilities.values())) if volatilities else None

        if current_avg_vol is None:
            return None

        # 与上次比较 - 添加零值检查
        if self.last_volatility is not None and self.last_volatility > 1e-10:
            vol_change = current_avg_vol / self.last_volatility
            if vol_change >= self.INTRADAY_THRESHOLDS["vol_spike_multiplier"]:
                self.last_volatility = current_avg_vol
                return IntradayAlert(
                    timestamp=timestamp,
                    alert_type="volatility_regime",
                    severity="warning",
                    current_value=current_avg_vol,
                    threshold=self.last_volatility,
                    message=f"波动率突变 ({vol_change:.1%} 增加) - 市场进入高波动期",
                    recommended_action="考虑减少杠杆，增加对冲",
                )

        self.last_volatility = current_avg_vol
        return None

    def should_trigger_emergency_rebalance(
        self,
        alerts: List[IntradayAlert],
    ) -> bool:
        """
        判断是否应触发紧急再平衡

        Args:
            alerts: 当前警报列表

        Returns:
            是否触发紧急再平衡
        """
        for alert in alerts:
            if alert.severity == "emergency":
                logger.warning(f"Emergency rebalance triggered: {alert.message}")
                return True
        return False

    def get_defensive_allocation(self) -> Dict[str, float]:
        """
        获取防御性配置

        Returns:
            防御性资产配置权重
        """
        return self.DEFENSIVE_ALLOCATION.copy()

    def assess_with_intraday(
        self,
        current_allocation: Dict[str, float],
        proposed_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        portfolio_value: float,
    ) -> RiskAssessment:
        """
        带日内监控的完整风险评估

        Args:
            current_allocation: 当前配置
            proposed_allocation: 提议配置
            market_data: 市场数据
            portfolio_value: 组合价值

        Returns:
            RiskAssessment: 包含日内警报的风险评估
        """
        # 先执行标准评估
        assessment = self.assess(
            current_allocation=current_allocation,
            proposed_allocation=proposed_allocation,
            market_data=market_data,
            portfolio_value=portfolio_value,
        )

        # 检查日内警报
        intraday_alerts = self.check_intraday_alerts(
            market_data=market_data,
            current_drawdown=assessment.current_drawdown,
        )

        # 更新评估结果
        assessment.intraday_alerts = intraday_alerts

        # 检查是否触发紧急再平衡
        if self.should_trigger_emergency_rebalance(intraday_alerts):
            assessment.emergency_rebalance = True
            assessment.defensive_allocation = self.get_defensive_allocation()
            assessment.veto = True  # 否决当前配置，使用防御性配置
            assessment.violations.append("紧急情况触发：切换至防御性配置")
            logger.warning("Emergency rebalance activated - switching to defensive allocation")

        return assessment

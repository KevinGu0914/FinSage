"""
Intraday Risk Monitor
日内实时风险监控模块

基于小时级数据进行实时风险监控，在日K决策基础上增加日内异常检测和紧急响应机制。

参考文献:
- Andersen, T.G., et al. (2003). Modeling and Forecasting Realized Volatility. Econometrica.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
  Journal of Financial Econometrics.
- Bollerslev, T., et al. (2018). Risk Everywhere: Modeling and Managing Volatility.
  Review of Financial Studies.
- Christoffersen, P., et al. (2012). Is the Potential for International Diversification
  Disappearing? Review of Financial Studies.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """警报级别"""
    NORMAL = "normal"           # 正常
    ATTENTION = "attention"     # 关注
    WARNING = "warning"         # 警告
    CRITICAL = "critical"       # 危急
    EMERGENCY = "emergency"     # 紧急


class AlertType(Enum):
    """警报类型"""
    PRICE_SPIKE = "price_spike"               # 价格异常波动
    VOLUME_SURGE = "volume_surge"             # 成交量激增
    VOLATILITY_EXPLOSION = "volatility_explosion"  # 波动率爆发
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # 相关性崩塌
    LIQUIDITY_CRISIS = "liquidity_crisis"     # 流动性危机
    GAP_RISK = "gap_risk"                     # 跳空风险
    NEWS_SHOCK = "news_shock"                 # 新闻冲击
    CIRCUIT_BREAKER = "circuit_breaker"       # 熔断触发
    FLASH_CRASH = "flash_crash"               # 闪崩检测


class EmergencyAction(Enum):
    """紧急响应动作"""
    HOLD = "hold"                   # 保持现状
    REDUCE_EXPOSURE = "reduce_exposure"  # 减少敞口
    HEDGE = "hedge"                 # 对冲
    STOP_LOSS = "stop_loss"         # 止损
    FULL_EXIT = "full_exit"         # 全部平仓
    PAUSE_TRADING = "pause_trading"  # 暂停交易


@dataclass
class IntradayAlert:
    """日内警报"""
    timestamp: datetime
    alert_type: AlertType
    alert_level: AlertLevel
    symbol: str
    description: str
    metrics: Dict[str, float]
    recommended_action: EmergencyAction
    confidence: float  # 0-1

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "alert_level": self.alert_level.value,
            "symbol": self.symbol,
            "description": self.description,
            "metrics": self.metrics,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
        }


@dataclass
class IntradayRiskReport:
    """日内风险报告"""
    timestamp: datetime
    overall_level: AlertLevel
    alerts: List[IntradayAlert]
    portfolio_metrics: Dict[str, float]
    recommended_actions: List[Dict[str, Any]]
    market_regime: str  # normal, stressed, crisis

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_level": self.overall_level.value,
            "alerts": [a.to_dict() for a in self.alerts],
            "portfolio_metrics": self.portfolio_metrics,
            "recommended_actions": self.recommended_actions,
            "market_regime": self.market_regime,
        }


class IntradayRiskMonitor:
    """
    日内实时风险监控器

    核心功能:
    1. 基于小时级数据检测异常波动
    2. 多种异常检测算法
    3. 分级警报系统
    4. 紧急响应建议
    5. 与日K级决策系统协调

    监控指标:
    - 价格波动 (相对历史分布)
    - 成交量异常
    - 实现波动率 (Realized Volatility)
    - 相关性突变
    - 流动性指标
    - VIX/恐慌指数
    """

    # 默认阈值配置
    DEFAULT_THRESHOLDS = {
        # 价格波动阈值 (标准差倍数)
        "price_attention": 2.0,    # 2σ -> 关注
        "price_warning": 3.0,      # 3σ -> 警告
        "price_critical": 4.0,     # 4σ -> 危急
        "price_emergency": 5.0,    # 5σ -> 紧急

        # 成交量阈值 (相对均值倍数)
        "volume_attention": 2.0,
        "volume_warning": 3.0,
        "volume_critical": 5.0,

        # 波动率阈值 (相对历史百分位)
        "volatility_warning_percentile": 90,
        "volatility_critical_percentile": 95,
        "volatility_emergency_percentile": 99,

        # 相关性变化阈值
        "correlation_change_warning": 0.3,
        "correlation_change_critical": 0.5,

        # 流动性阈值 (价差扩大倍数)
        "spread_warning": 2.0,
        "spread_critical": 4.0,

        # VIX阈值
        "vix_attention": 25,
        "vix_warning": 30,
        "vix_critical": 40,
        "vix_emergency": 50,

        # 日内最大回撤
        "intraday_drawdown_warning": 0.03,  # 3%
        "intraday_drawdown_critical": 0.05,  # 5%
        "intraday_drawdown_emergency": 0.07,  # 7%
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        lookback_hours: int = 24,
        check_interval_minutes: int = 60,
    ):
        """
        初始化日内风险监控器

        Args:
            thresholds: 自定义阈值
            lookback_hours: 回看小时数 (用于计算历史统计)
            check_interval_minutes: 检查间隔 (分钟)
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.lookback_hours = lookback_hours
        self.check_interval_minutes = check_interval_minutes

        # 历史数据缓存
        self._hourly_cache: Dict[str, pd.DataFrame] = {}
        self._alert_history: List[IntradayAlert] = []
        self._last_check_time: Optional[datetime] = None

        # 统计基准 (用于异常检测)
        self._baseline_stats: Dict[str, Dict[str, float]] = {}

        logger.info(f"IntradayRiskMonitor initialized with {lookback_hours}h lookback")

    def monitor(
        self,
        hourly_data: Dict[str, pd.DataFrame],
        current_holdings: Dict[str, float],
        portfolio_value: float,
        vix_level: Optional[float] = None,
    ) -> IntradayRiskReport:
        """
        执行日内风险监控

        Args:
            hourly_data: 小时级价格数据 {symbol: DataFrame with OHLCV}
            current_holdings: 当前持仓 {symbol: weight}
            portfolio_value: 组合总价值
            vix_level: VIX水平 (可选)

        Returns:
            IntradayRiskReport: 日内风险报告
        """
        timestamp = datetime.now()
        alerts = []

        # 更新缓存
        self._update_cache(hourly_data)

        # 1. 检测各资产的异常
        for symbol in current_holdings.keys():
            if symbol in hourly_data and not hourly_data[symbol].empty:
                symbol_alerts = self._detect_symbol_anomalies(
                    symbol=symbol,
                    data=hourly_data[symbol],
                    weight=current_holdings.get(symbol, 0),
                )
                alerts.extend(symbol_alerts)

        # 2. 检测组合级异常
        portfolio_alerts = self._detect_portfolio_anomalies(
            hourly_data=hourly_data,
            holdings=current_holdings,
            portfolio_value=portfolio_value,
        )
        alerts.extend(portfolio_alerts)

        # 3. 检测市场级异常 (VIX等)
        if vix_level is not None:
            market_alerts = self._detect_market_anomalies(vix_level)
            alerts.extend(market_alerts)

        # 4. 检测相关性异常
        correlation_alerts = self._detect_correlation_anomalies(hourly_data, current_holdings)
        alerts.extend(correlation_alerts)

        # 5. 确定整体风险级别
        overall_level = self._determine_overall_level(alerts)

        # 6. 判断市场状态
        market_regime = self._assess_market_regime(alerts, vix_level)

        # 7. 生成建议动作
        recommended_actions = self._generate_actions(
            alerts=alerts,
            holdings=current_holdings,
            overall_level=overall_level,
        )

        # 8. 计算组合指标
        portfolio_metrics = self._calculate_portfolio_metrics(
            hourly_data=hourly_data,
            holdings=current_holdings,
            portfolio_value=portfolio_value,
        )

        # 记录
        self._alert_history.extend(alerts)
        self._last_check_time = timestamp

        report = IntradayRiskReport(
            timestamp=timestamp,
            overall_level=overall_level,
            alerts=alerts,
            portfolio_metrics=portfolio_metrics,
            recommended_actions=recommended_actions,
            market_regime=market_regime,
        )

        if alerts:
            logger.warning(f"IntradayRiskMonitor: {len(alerts)} alerts, level={overall_level.value}")

        return report

    def _update_cache(self, hourly_data: Dict[str, pd.DataFrame]):
        """更新小时数据缓存"""
        for symbol, data in hourly_data.items():
            if symbol not in self._hourly_cache:
                self._hourly_cache[symbol] = data.copy()
            else:
                # 合并新数据
                combined = pd.concat([self._hourly_cache[symbol], data])
                combined = combined[~combined.index.duplicated(keep='last')]
                # 只保留最近的数据
                cutoff = datetime.now() - timedelta(hours=self.lookback_hours * 2)
                self._hourly_cache[symbol] = combined[combined.index > cutoff]

    def _detect_symbol_anomalies(
        self,
        symbol: str,
        data: pd.DataFrame,
        weight: float,
    ) -> List[IntradayAlert]:
        """检测单个资产的异常"""
        alerts = []

        if len(data) < 2:
            return alerts

        # 获取最新数据
        latest = data.iloc[-1]

        # 计算收益率
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
        else:
            return alerts

        if len(returns) < 2:
            return alerts

        latest_return = returns.iloc[-1]

        # 1. 价格波动检测
        price_alert = self._detect_price_spike(symbol, latest_return, returns, weight)
        if price_alert:
            alerts.append(price_alert)

        # 2. 成交量异常检测
        if 'Volume' in data.columns:
            volume_alert = self._detect_volume_surge(symbol, data['Volume'], weight)
            if volume_alert:
                alerts.append(volume_alert)

        # 3. 实现波动率检测
        rv_alert = self._detect_volatility_explosion(symbol, returns, weight)
        if rv_alert:
            alerts.append(rv_alert)

        # 4. 跳空检测
        if 'Open' in data.columns and 'Close' in data.columns:
            gap_alert = self._detect_gap_risk(symbol, data, weight)
            if gap_alert:
                alerts.append(gap_alert)

        return alerts

    def _detect_price_spike(
        self,
        symbol: str,
        latest_return: float,
        returns: pd.Series,
        weight: float,
    ) -> Optional[IntradayAlert]:
        """检测价格异常波动"""
        if len(returns) < 10:
            return None

        # 计算历史统计
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return < 1e-10:  # 使用阈值代替浮点数相等比较
            return None

        # 计算Z-score
        z_score = abs(latest_return - mean_return) / std_return

        # 确定警报级别
        level = AlertLevel.NORMAL
        if z_score >= self.thresholds["price_emergency"]:
            level = AlertLevel.EMERGENCY
        elif z_score >= self.thresholds["price_critical"]:
            level = AlertLevel.CRITICAL
        elif z_score >= self.thresholds["price_warning"]:
            level = AlertLevel.WARNING
        elif z_score >= self.thresholds["price_attention"]:
            level = AlertLevel.ATTENTION
        else:
            return None

        # 确定建议动作
        action = self._determine_action_for_spike(level, latest_return, weight)

        return IntradayAlert(
            timestamp=datetime.now(),
            alert_type=AlertType.PRICE_SPIKE,
            alert_level=level,
            symbol=symbol,
            description=f"{symbol} 价格异常波动: {latest_return:.2%} (Z={z_score:.1f}σ)",
            metrics={
                "return": float(latest_return),
                "z_score": float(z_score),
                "historical_std": float(std_return),
                "position_weight": float(weight),
            },
            recommended_action=action,
            confidence=min(0.95, 0.5 + z_score * 0.1),
        )

    def _detect_volume_surge(
        self,
        symbol: str,
        volume: pd.Series,
        weight: float,
    ) -> Optional[IntradayAlert]:
        """检测成交量激增"""
        if len(volume) < 10:
            return None

        latest_volume = volume.iloc[-1]
        mean_volume = volume.iloc[:-1].mean()

        if mean_volume < 1e-10:  # 避免除零
            return None

        volume_ratio = latest_volume / mean_volume

        # 确定级别
        level = AlertLevel.NORMAL
        if volume_ratio >= self.thresholds["volume_critical"]:
            level = AlertLevel.CRITICAL
        elif volume_ratio >= self.thresholds["volume_warning"]:
            level = AlertLevel.WARNING
        elif volume_ratio >= self.thresholds["volume_attention"]:
            level = AlertLevel.ATTENTION
        else:
            return None

        return IntradayAlert(
            timestamp=datetime.now(),
            alert_type=AlertType.VOLUME_SURGE,
            alert_level=level,
            symbol=symbol,
            description=f"{symbol} 成交量激增: {volume_ratio:.1f}x 均值",
            metrics={
                "volume_ratio": float(volume_ratio),
                "latest_volume": float(latest_volume),
                "mean_volume": float(mean_volume),
            },
            recommended_action=EmergencyAction.HOLD,  # 成交量本身不决定动作
            confidence=min(0.9, 0.5 + (volume_ratio - 1) * 0.1),
        )

    def _detect_volatility_explosion(
        self,
        symbol: str,
        returns: pd.Series,
        weight: float,
    ) -> Optional[IntradayAlert]:
        """检测波动率爆发 (基于实现波动率)"""
        if len(returns) < 20:
            return None

        # 计算滚动实现波动率
        recent_rv = returns.iloc[-6:].std() * np.sqrt(252 * 6.5)  # 年化 (假设每天6.5小时)
        historical_rv = returns.iloc[:-6].std() * np.sqrt(252 * 6.5)

        if historical_rv < 1e-10:  # 使用阈值代替浮点数相等比较
            return None

        rv_ratio = recent_rv / historical_rv

        # 计算百分位
        rolling_rv = returns.rolling(6).std() * np.sqrt(252 * 6.5)
        percentile = (rolling_rv < recent_rv).sum() / len(rolling_rv) * 100

        # 确定级别
        level = AlertLevel.NORMAL
        if percentile >= self.thresholds["volatility_emergency_percentile"]:
            level = AlertLevel.EMERGENCY
        elif percentile >= self.thresholds["volatility_critical_percentile"]:
            level = AlertLevel.CRITICAL
        elif percentile >= self.thresholds["volatility_warning_percentile"]:
            level = AlertLevel.WARNING
        else:
            return None

        action = EmergencyAction.REDUCE_EXPOSURE if level.value in ["critical", "emergency"] else EmergencyAction.HOLD

        return IntradayAlert(
            timestamp=datetime.now(),
            alert_type=AlertType.VOLATILITY_EXPLOSION,
            alert_level=level,
            symbol=symbol,
            description=f"{symbol} 波动率爆发: {recent_rv:.1%}年化 ({percentile:.0f}th百分位)",
            metrics={
                "recent_rv": float(recent_rv),
                "historical_rv": float(historical_rv),
                "rv_ratio": float(rv_ratio),
                "percentile": float(percentile),
            },
            recommended_action=action,
            confidence=np.clip(percentile / 100, 0, 1),  # 约束在 [0,1]
        )

    def _detect_gap_risk(
        self,
        symbol: str,
        data: pd.DataFrame,
        weight: float,
    ) -> Optional[IntradayAlert]:
        """检测跳空风险"""
        if len(data) < 2:
            return None

        # 计算跳空幅度
        prev_close = data['Close'].iloc[-2]
        curr_open = data['Open'].iloc[-1]

        if prev_close < 1e-10:  # 避免除零
            return None

        gap = (curr_open - prev_close) / prev_close

        # 只有大跳空才报警
        if abs(gap) < 0.02:  # 2% 以下不报警
            return None

        level = AlertLevel.WARNING
        if abs(gap) >= 0.05:
            level = AlertLevel.CRITICAL
        elif abs(gap) >= 0.03:
            level = AlertLevel.WARNING

        direction = "向上" if gap > 0 else "向下"

        return IntradayAlert(
            timestamp=datetime.now(),
            alert_type=AlertType.GAP_RISK,
            alert_level=level,
            symbol=symbol,
            description=f"{symbol} {direction}跳空: {gap:.2%}",
            metrics={
                "gap_percent": float(gap),
                "prev_close": float(prev_close),
                "curr_open": float(curr_open),
            },
            recommended_action=EmergencyAction.HOLD if gap > 0 else EmergencyAction.REDUCE_EXPOSURE,
            confidence=0.8,
        )

    def _detect_portfolio_anomalies(
        self,
        hourly_data: Dict[str, pd.DataFrame],
        holdings: Dict[str, float],
        portfolio_value: float,
    ) -> List[IntradayAlert]:
        """检测组合级异常"""
        alerts = []

        # 计算组合的小时收益率
        portfolio_returns = self._calculate_portfolio_hourly_returns(hourly_data, holdings)

        if len(portfolio_returns) < 6:
            return alerts

        # 1. 日内回撤检测
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        current_drawdown = drawdown.iloc[-1]

        level = AlertLevel.NORMAL
        if abs(current_drawdown) >= self.thresholds["intraday_drawdown_emergency"]:
            level = AlertLevel.EMERGENCY
        elif abs(current_drawdown) >= self.thresholds["intraday_drawdown_critical"]:
            level = AlertLevel.CRITICAL
        elif abs(current_drawdown) >= self.thresholds["intraday_drawdown_warning"]:
            level = AlertLevel.WARNING

        if level != AlertLevel.NORMAL:
            action = EmergencyAction.STOP_LOSS if level == AlertLevel.EMERGENCY else EmergencyAction.REDUCE_EXPOSURE
            alerts.append(IntradayAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.FLASH_CRASH,
                alert_level=level,
                symbol="PORTFOLIO",
                description=f"组合日内回撤: {current_drawdown:.2%}",
                metrics={
                    "current_drawdown": float(current_drawdown),
                    "portfolio_value": float(portfolio_value),
                },
                recommended_action=action,
                confidence=0.9,
            ))

        return alerts

    def _detect_market_anomalies(self, vix_level: float) -> List[IntradayAlert]:
        """检测市场级异常 (VIX等)"""
        alerts = []

        level = AlertLevel.NORMAL
        if vix_level >= self.thresholds["vix_emergency"]:
            level = AlertLevel.EMERGENCY
        elif vix_level >= self.thresholds["vix_critical"]:
            level = AlertLevel.CRITICAL
        elif vix_level >= self.thresholds["vix_warning"]:
            level = AlertLevel.WARNING
        elif vix_level >= self.thresholds["vix_attention"]:
            level = AlertLevel.ATTENTION

        if level != AlertLevel.NORMAL:
            action_map = {
                AlertLevel.EMERGENCY: EmergencyAction.FULL_EXIT,
                AlertLevel.CRITICAL: EmergencyAction.STOP_LOSS,
                AlertLevel.WARNING: EmergencyAction.REDUCE_EXPOSURE,
                AlertLevel.ATTENTION: EmergencyAction.HEDGE,
            }

            alerts.append(IntradayAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.VOLATILITY_EXPLOSION,
                alert_level=level,
                symbol="VIX",
                description=f"VIX恐慌指数升高: {vix_level:.1f}",
                metrics={
                    "vix_level": float(vix_level),
                },
                recommended_action=action_map.get(level, EmergencyAction.HOLD),
                confidence=0.85,
            ))

        return alerts

    def _detect_correlation_anomalies(
        self,
        hourly_data: Dict[str, pd.DataFrame],
        holdings: Dict[str, float],
    ) -> List[IntradayAlert]:
        """检测相关性异常 (相关性崩塌)"""
        alerts = []

        symbols = [s for s in holdings.keys() if s in hourly_data]
        if len(symbols) < 2:
            return alerts

        # 构建收益率矩阵
        returns_dict = {}
        for symbol in symbols:
            if 'Close' in hourly_data[symbol].columns:
                returns_dict[symbol] = hourly_data[symbol]['Close'].pct_change().dropna()

        if len(returns_dict) < 2:
            return alerts

        returns_df = pd.DataFrame(returns_dict).dropna()

        if len(returns_df) < 12:  # 至少需要12小时数据
            return alerts

        # 计算最近相关性 vs 历史相关性
        recent_corr = returns_df.iloc[-6:].corr()
        historical_corr = returns_df.iloc[:-6].corr() if len(returns_df) > 12 else recent_corr

        # 检测相关性变化
        corr_change = (recent_corr - historical_corr).abs()
        max_change = corr_change.values[np.triu_indices(len(symbols), k=1)].max()

        level = AlertLevel.NORMAL
        if max_change >= self.thresholds["correlation_change_critical"]:
            level = AlertLevel.CRITICAL
        elif max_change >= self.thresholds["correlation_change_warning"]:
            level = AlertLevel.WARNING

        if level != AlertLevel.NORMAL:
            alerts.append(IntradayAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.CORRELATION_BREAKDOWN,
                alert_level=level,
                symbol="PORTFOLIO",
                description=f"相关性结构突变: 最大变化 {max_change:.2f}",
                metrics={
                    "max_correlation_change": float(max_change),
                },
                recommended_action=EmergencyAction.HEDGE,
                confidence=0.75,
            ))

        return alerts

    def _calculate_portfolio_hourly_returns(
        self,
        hourly_data: Dict[str, pd.DataFrame],
        holdings: Dict[str, float],
    ) -> pd.Series:
        """计算组合的小时收益率"""
        returns_list = []
        weights = []

        for symbol, weight in holdings.items():
            if symbol in hourly_data and 'Close' in hourly_data[symbol].columns:
                ret = hourly_data[symbol]['Close'].pct_change()
                returns_list.append(ret)
                weights.append(weight)

        if not returns_list:
            return pd.Series(dtype=float)

        # 对齐并计算加权平均
        returns_df = pd.concat(returns_list, axis=1)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化

        portfolio_returns = (returns_df * weights).sum(axis=1)

        return portfolio_returns.dropna()

    def _determine_overall_level(self, alerts: List[IntradayAlert]) -> AlertLevel:
        """确定整体风险级别"""
        if not alerts:
            return AlertLevel.NORMAL

        # 按严重程度排序
        level_priority = {
            AlertLevel.EMERGENCY: 5,
            AlertLevel.CRITICAL: 4,
            AlertLevel.WARNING: 3,
            AlertLevel.ATTENTION: 2,
            AlertLevel.NORMAL: 1,
        }

        max_level = max(alerts, key=lambda a: level_priority[a.alert_level])
        return max_level.alert_level

    def _assess_market_regime(
        self,
        alerts: List[IntradayAlert],
        vix_level: Optional[float],
    ) -> str:
        """评估市场状态"""
        # 基于警报数量和级别
        critical_count = sum(1 for a in alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY])
        warning_count = sum(1 for a in alerts if a.alert_level == AlertLevel.WARNING)

        # 结合VIX
        vix_regime = "normal"
        if vix_level:
            if vix_level >= 40:
                vix_regime = "crisis"
            elif vix_level >= 25:
                vix_regime = "stressed"

        # 综合判断
        if critical_count >= 2 or vix_regime == "crisis":
            return "crisis"
        elif critical_count >= 1 or warning_count >= 3 or vix_regime == "stressed":
            return "stressed"
        else:
            return "normal"

    def _generate_actions(
        self,
        alerts: List[IntradayAlert],
        holdings: Dict[str, float],
        overall_level: AlertLevel,
    ) -> List[Dict[str, Any]]:
        """生成建议动作"""
        actions = []

        if overall_level == AlertLevel.NORMAL:
            return actions

        # 按资产聚合警报
        asset_alerts: Dict[str, List[IntradayAlert]] = {}
        portfolio_alerts = []

        for alert in alerts:
            if alert.symbol == "PORTFOLIO" or alert.symbol == "VIX":
                portfolio_alerts.append(alert)
            else:
                if alert.symbol not in asset_alerts:
                    asset_alerts[alert.symbol] = []
                asset_alerts[alert.symbol].append(alert)

        # 针对单个资产生成动作
        for symbol, symbol_alerts in asset_alerts.items():
            worst_alert = max(symbol_alerts, key=lambda a: a.alert_level.value)
            current_weight = holdings.get(symbol, 0)

            if worst_alert.alert_level in [AlertLevel.EMERGENCY, AlertLevel.CRITICAL]:
                # 减仓或止损
                target_weight = current_weight * 0.5 if worst_alert.alert_level == AlertLevel.CRITICAL else 0
                actions.append({
                    "symbol": symbol,
                    "action": worst_alert.recommended_action.value,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "reason": worst_alert.description,
                    "urgency": "immediate",
                })

        # 针对组合级别生成动作
        if overall_level == AlertLevel.EMERGENCY:
            actions.append({
                "symbol": "ALL",
                "action": EmergencyAction.PAUSE_TRADING.value,
                "reason": "组合整体风险达到紧急级别",
                "urgency": "immediate",
            })
        elif overall_level == AlertLevel.CRITICAL:
            # 建议整体减仓
            actions.append({
                "symbol": "ALL",
                "action": EmergencyAction.REDUCE_EXPOSURE.value,
                "target_reduction": 0.3,  # 减仓30%
                "reason": "组合整体风险达到危急级别",
                "urgency": "within_hour",
            })

        return actions

    def _calculate_portfolio_metrics(
        self,
        hourly_data: Dict[str, pd.DataFrame],
        holdings: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """计算组合指标"""
        portfolio_returns = self._calculate_portfolio_hourly_returns(hourly_data, holdings)

        if len(portfolio_returns) < 2:
            return {
                "intraday_return": 0.0,
                "intraday_volatility": 0.0,
                "intraday_sharpe": 0.0,
                "max_hourly_drawdown": 0.0,
            }

        intraday_return = (1 + portfolio_returns).prod() - 1
        intraday_vol = portfolio_returns.std() * np.sqrt(6.5)  # 日内年化

        # 计算日内最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 简化Sharpe (假设日内无风险收益为0)
        sharpe = intraday_return / intraday_vol if intraday_vol > 0 else 0

        return {
            "intraday_return": float(intraday_return),
            "intraday_volatility": float(intraday_vol),
            "intraday_sharpe": float(sharpe),
            "max_hourly_drawdown": float(max_drawdown),
            "portfolio_value": float(portfolio_value),
        }

    def _determine_action_for_spike(
        self,
        level: AlertLevel,
        return_value: float,
        weight: float,
    ) -> EmergencyAction:
        """根据价格波动确定动作"""
        if level == AlertLevel.EMERGENCY:
            return EmergencyAction.STOP_LOSS
        elif level == AlertLevel.CRITICAL:
            if return_value < 0:  # 下跌
                return EmergencyAction.REDUCE_EXPOSURE
            else:  # 上涨
                return EmergencyAction.HOLD  # 暴涨时可能是好事
        elif level == AlertLevel.WARNING:
            return EmergencyAction.HOLD
        else:
            return EmergencyAction.HOLD

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取警报摘要"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self._alert_history if a.timestamp > cutoff]

        # 按级别统计
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(1 for a in recent_alerts if a.alert_level == level)

        # 按类型统计
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = sum(1 for a in recent_alerts if a.alert_type == alert_type)

        return {
            "total_alerts": len(recent_alerts),
            "by_level": level_counts,
            "by_type": type_counts,
            "most_recent": recent_alerts[-5:] if recent_alerts else [],
        }

    def clear_history(self):
        """清除历史记录"""
        self._alert_history.clear()
        self._hourly_cache.clear()
        logger.info("IntradayRiskMonitor history cleared")

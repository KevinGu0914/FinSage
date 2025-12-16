"""
Risk Management Module
风险管理模块

提供多层次的风险管理功能:
- 日级风险控制 (Risk Controller)
- 日内实时监控 (Intraday Risk Monitor)

核心组件:
1. IntradayRiskMonitor: 基于小时级数据的实时风险监控
   - 检测价格异常波动 (Z-score)
   - 检测成交量激增
   - 检测实现波动率爆发
   - 检测相关性崩塌
   - 检测流动性危机
   - 生成紧急响应建议

2. AlertLevel: 警报级别 (NORMAL, ATTENTION, WARNING, CRITICAL, EMERGENCY)

3. EmergencyAction: 紧急响应动作 (HOLD, REDUCE_EXPOSURE, HEDGE, STOP_LOSS, FULL_EXIT, PAUSE_TRADING)

参考文献:
- Andersen, T.G., et al. (2003). Modeling and Forecasting Realized Volatility.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
- Bollerslev, T., et al. (2018). Risk Everywhere: Modeling and Managing Volatility.
"""

from finsage.risk.intraday_monitor import (
    IntradayRiskMonitor,
    IntradayAlert,
    IntradayRiskReport,
    AlertLevel,
    AlertType,
    EmergencyAction,
)

__all__ = [
    "IntradayRiskMonitor",
    "IntradayAlert",
    "IntradayRiskReport",
    "AlertLevel",
    "AlertType",
    "EmergencyAction",
]

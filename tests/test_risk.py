#!/usr/bin/env python
"""
Risk Module Tests - 风险模块测试
覆盖: intraday_monitor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: Alert Enums
# ============================================================

class TestAlertEnums:
    """测试警报枚举"""

    def test_alert_level_enum(self):
        """测试警报级别枚举"""
        from finsage.risk.intraday_monitor import AlertLevel

        assert AlertLevel.NORMAL.value == "normal"
        assert AlertLevel.ATTENTION.value == "attention"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"

    def test_alert_type_enum(self):
        """测试警报类型枚举"""
        from finsage.risk.intraday_monitor import AlertType

        assert AlertType.PRICE_SPIKE.value == "price_spike"
        assert AlertType.VOLUME_SURGE.value == "volume_surge"
        assert AlertType.VOLATILITY_EXPLOSION.value == "volatility_explosion"
        assert AlertType.CORRELATION_BREAKDOWN.value == "correlation_breakdown"
        assert AlertType.LIQUIDITY_CRISIS.value == "liquidity_crisis"
        assert AlertType.FLASH_CRASH.value == "flash_crash"

    def test_emergency_action_enum(self):
        """测试紧急响应动作枚举"""
        from finsage.risk.intraday_monitor import EmergencyAction

        assert EmergencyAction.HOLD.value == "hold"
        assert EmergencyAction.REDUCE_EXPOSURE.value == "reduce_exposure"
        assert EmergencyAction.HEDGE.value == "hedge"
        assert EmergencyAction.STOP_LOSS.value == "stop_loss"
        assert EmergencyAction.FULL_EXIT.value == "full_exit"
        assert EmergencyAction.PAUSE_TRADING.value == "pause_trading"


# ============================================================
# Test 2: Intraday Alert Dataclass
# ============================================================

class TestIntradayAlert:
    """测试日内警报数据类"""

    def test_import(self):
        """测试导入"""
        from finsage.risk.intraday_monitor import IntradayAlert
        assert IntradayAlert is not None

    def test_creation(self):
        """测试创建"""
        from finsage.risk.intraday_monitor import (
            IntradayAlert, AlertType, AlertLevel, EmergencyAction
        )

        alert = IntradayAlert(
            timestamp=datetime.now(),
            alert_type=AlertType.PRICE_SPIKE,
            alert_level=AlertLevel.WARNING,
            symbol="SPY",
            description="Price dropped 3% in 5 minutes",
            metrics={"price_change": -0.03, "volume_ratio": 2.5},
            recommended_action=EmergencyAction.REDUCE_EXPOSURE,
            confidence=0.85
        )

        assert alert.symbol == "SPY"
        assert alert.alert_level == AlertLevel.WARNING
        assert alert.confidence == 0.85

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.risk.intraday_monitor import (
            IntradayAlert, AlertType, AlertLevel, EmergencyAction
        )

        alert = IntradayAlert(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            alert_type=AlertType.VOLUME_SURGE,
            alert_level=AlertLevel.ATTENTION,
            symbol="QQQ",
            description="Volume 3x normal",
            metrics={"volume_ratio": 3.0},
            recommended_action=EmergencyAction.HOLD,
            confidence=0.70
        )

        d = alert.to_dict()

        assert d["symbol"] == "QQQ"
        assert d["alert_type"] == "volume_surge"
        assert d["alert_level"] == "attention"
        assert "timestamp" in d
        assert d["confidence"] == 0.70


# ============================================================
# Test 3: Intraday Risk Report Dataclass
# ============================================================

class TestIntradayRiskReport:
    """测试日内风险报告数据类"""

    def test_import(self):
        """测试导入"""
        from finsage.risk.intraday_monitor import IntradayRiskReport
        assert IntradayRiskReport is not None

    def test_creation(self):
        """测试创建"""
        from finsage.risk.intraday_monitor import (
            IntradayRiskReport, IntradayAlert, AlertType, AlertLevel, EmergencyAction
        )

        alerts = [
            IntradayAlert(
                timestamp=datetime.now(),
                alert_type=AlertType.PRICE_SPIKE,
                alert_level=AlertLevel.WARNING,
                symbol="SPY",
                description="Test alert",
                metrics={"value": 0.05},
                recommended_action=EmergencyAction.HOLD,
                confidence=0.8
            )
        ]

        report = IntradayRiskReport(
            timestamp=datetime.now(),
            overall_level=AlertLevel.WARNING,
            alerts=alerts,
            portfolio_metrics={"var_95": 0.02, "volatility": 0.15},
            recommended_actions=[{"action": "reduce_spy", "size": 0.1}],
            market_regime="stressed"
        )

        assert report.overall_level == AlertLevel.WARNING
        assert len(report.alerts) == 1
        assert report.market_regime == "stressed"

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.risk.intraday_monitor import (
            IntradayRiskReport, AlertLevel
        )

        report = IntradayRiskReport(
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            overall_level=AlertLevel.NORMAL,
            alerts=[],
            portfolio_metrics={"var_95": 0.015},
            recommended_actions=[],
            market_regime="normal"
        )

        d = report.to_dict()

        assert d["overall_level"] == "normal"
        assert d["market_regime"] == "normal"
        assert "alerts" in d
        assert "portfolio_metrics" in d


# ============================================================
# Test 4: Intraday Risk Monitor
# ============================================================

class TestIntradayRiskMonitor:
    """测试日内风险监控器"""

    def test_import(self):
        """测试导入"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor
        assert IntradayRiskMonitor is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()
        assert monitor is not None

    def test_initialization_with_thresholds(self):
        """测试带阈值参数初始化"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        custom_thresholds = {
            "price_attention": 2.5,
            "volume_attention": 2.5,
            "vix_warning": 35,
        }
        monitor = IntradayRiskMonitor(
            thresholds=custom_thresholds,
            lookback_hours=24
        )

        assert monitor.thresholds["price_attention"] == 2.5
        assert monitor.lookback_hours == 24

    def test_default_thresholds(self):
        """测试默认阈值"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()

        # 检查是否有默认阈值
        assert hasattr(monitor, 'thresholds')
        assert "price_warning" in monitor.thresholds
        assert "vix_warning" in monitor.thresholds

    def test_check_alert_levels(self):
        """测试警报级别判断逻辑"""
        from finsage.risk.intraday_monitor import AlertLevel

        # 验证警报级别的顺序
        levels = [
            AlertLevel.NORMAL,
            AlertLevel.ATTENTION,
            AlertLevel.WARNING,
            AlertLevel.CRITICAL,
            AlertLevel.EMERGENCY
        ]

        # 确保所有级别都定义了
        assert len(levels) == 5


# ============================================================
# Test 5: Risk Monitor Methods
# ============================================================

class TestRiskMonitorMethods:
    """测试风险监控器方法"""

    @pytest.fixture
    def sample_intraday_data(self):
        """生成示例日内数据"""
        timestamps = pd.date_range(
            start='2024-01-15 09:30:00',
            periods=78,  # 6.5小时交易日
            freq='5min'
        )
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': 450 + np.cumsum(np.random.randn(78) * 0.5),
            'high': 451 + np.cumsum(np.random.randn(78) * 0.5),
            'low': 449 + np.cumsum(np.random.randn(78) * 0.5),
            'close': 450 + np.cumsum(np.random.randn(78) * 0.5),
            'volume': np.random.randint(100000, 500000, 78),
        })

        return data

    def test_monitor_has_monitor_method(self):
        """测试监控器有监控方法"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()
        assert hasattr(monitor, 'monitor')

    def test_monitor_has_detect_price_spike(self):
        """测试监控器有价格异常检测方法"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()
        assert hasattr(monitor, '_detect_price_spike')

    def test_monitor_has_detect_volume_surge(self):
        """测试监控器有成交量异常检测方法"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()
        assert hasattr(monitor, '_detect_volume_surge')

    def test_monitor_has_detect_volatility_explosion(self):
        """测试监控器有波动率爆发检测方法"""
        from finsage.risk.intraday_monitor import IntradayRiskMonitor

        monitor = IntradayRiskMonitor()
        assert hasattr(monitor, '_detect_volatility_explosion')


# ============================================================
# Test 6: Risk Metrics Calculation
# ============================================================

class TestRiskMetrics:
    """测试风险指标计算"""

    def test_var_calculation_concept(self):
        """测试VaR计算概念"""
        # 简单的VaR计算示例
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)

        # 95% VaR
        var_95 = np.percentile(returns, 5)
        assert var_95 < 0  # VaR应该是负数

    def test_cvar_calculation_concept(self):
        """测试CVaR计算概念"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)

        # 95% CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        assert cvar_95 <= var_95  # CVaR应该小于等于VaR

    def test_volatility_calculation(self):
        """测试波动率计算"""
        np.random.seed(42)
        returns = np.random.normal(0, 0.015, 252)

        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        assert 0.1 < annual_vol < 0.4  # 合理的年化波动率范围


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Risk Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

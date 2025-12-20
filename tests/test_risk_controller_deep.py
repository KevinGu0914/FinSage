"""
Deep tests for RiskController
风险控制器深度测试
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.agents.risk_controller import (
    RiskController,
    RiskAssessment,
    IntradayAlert,
)


class TestIntradayAlert:
    """IntradayAlert数据类测试"""

    def test_create_intraday_alert(self):
        """测试创建日内警报"""
        alert = IntradayAlert(
            timestamp="2024-01-15T10:30:00",
            alert_type="vix_spike",
            severity="warning",
            current_value=26.5,
            threshold=25.0,
            message="VIX警告",
            recommended_action="密切关注",
        )
        assert alert.alert_type == "vix_spike"
        assert alert.severity == "warning"
        assert alert.current_value == 26.5

    def test_alert_types(self):
        """测试不同警报类型"""
        for alert_type in ["vix_spike", "drawdown_trigger", "volatility_regime"]:
            alert = IntradayAlert(
                timestamp=datetime.now().isoformat(),
                alert_type=alert_type,
                severity="warning",
                current_value=0.5,
                threshold=0.3,
                message="Test",
                recommended_action="Test action",
            )
            assert alert.alert_type == alert_type


class TestRiskAssessment:
    """RiskAssessment数据类测试"""

    def test_create_risk_assessment(self):
        """测试创建风险评估"""
        assessment = RiskAssessment(
            timestamp="2024-01-15T10:30:00",
            portfolio_var_95=0.025,
            portfolio_cvar_99=0.035,
            current_drawdown=-0.05,
            max_drawdown=-0.08,
            volatility=0.12,
            sharpe_ratio=1.2,
            concentration_risk="medium",
            violations=[],
            warnings=["波动率较高"],
            veto=False,
            recommendations={},
        )
        assert assessment.portfolio_var_95 == 0.025
        assert assessment.veto is False

    def test_to_dict(self):
        """测试转换为字典"""
        assessment = RiskAssessment(
            timestamp="2024-01-15",
            portfolio_var_95=0.02,
            portfolio_cvar_99=0.03,
            current_drawdown=-0.03,
            max_drawdown=-0.05,
            volatility=0.10,
            sharpe_ratio=1.0,
            concentration_risk="low",
            violations=[],
            warnings=[],
            veto=False,
            recommendations={},
        )
        result = assessment.to_dict()
        assert "portfolio_var_95" in result
        assert "intraday_alerts" in result
        assert result["emergency_rebalance"] is False

    def test_with_intraday_alerts(self):
        """测试带日内警报的评估"""
        alert = IntradayAlert(
            timestamp="2024-01-15",
            alert_type="vix_spike",
            severity="critical",
            current_value=35.0,
            threshold=30.0,
            message="VIX危急",
            recommended_action="减少风险敞口",
        )
        assessment = RiskAssessment(
            timestamp="2024-01-15",
            portfolio_var_95=0.03,
            portfolio_cvar_99=0.04,
            current_drawdown=-0.08,
            max_drawdown=-0.10,
            volatility=0.15,
            sharpe_ratio=0.8,
            concentration_risk="high",
            violations=[],
            warnings=[],
            veto=False,
            recommendations={},
            intraday_alerts=[alert],
            emergency_rebalance=True,
            defensive_allocation={"bonds": 0.5, "cash": 0.3},
        )
        result = assessment.to_dict()
        assert len(result["intraday_alerts"]) == 1
        assert result["emergency_rebalance"] is True


class TestRiskControllerInit:
    """RiskController初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        rc = RiskController()
        assert rc.hard_limits == rc.DEFAULT_HARD_LIMITS
        assert rc.soft_limits == rc.DEFAULT_SOFT_LIMITS
        assert rc.peak_value == 1.0

    def test_custom_limits(self):
        """测试自定义约束"""
        hard = {"max_single_asset": 0.20, "max_asset_class": 0.60}
        soft = {"target_volatility": 0.15}
        rc = RiskController(hard_limits=hard, soft_limits=soft)
        assert rc.hard_limits["max_single_asset"] == 0.20
        assert rc.soft_limits["target_volatility"] == 0.15

    def test_intraday_thresholds(self):
        """测试日内监控阈值"""
        rc = RiskController()
        assert rc.INTRADAY_THRESHOLDS["vix_warning"] == 25.0
        assert rc.INTRADAY_THRESHOLDS["vix_critical"] == 30.0
        assert rc.INTRADAY_THRESHOLDS["vix_emergency"] == 40.0


class TestCheckHardLimits:
    """硬性约束检查测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_single_asset_violation(self, controller):
        """测试单资产权重违规"""
        allocation = {"AAPL": 0.20, "MSFT": 0.10}
        violations = controller._check_hard_limits(allocation, {})
        assert any("单资产" in v for v in violations)

    def test_asset_class_violation(self, controller):
        """测试资产类别权重违规"""
        allocation = {"stocks": 0.60, "bonds": 0.40}
        violations = controller._check_hard_limits(allocation, {})
        assert any("资产类别" in v for v in violations)

    def test_no_violation(self, controller):
        """测试无违规情况"""
        allocation = {"stocks": 0.40, "bonds": 0.30, "cash": 0.30}
        violations = controller._check_hard_limits(allocation, {})
        assert len(violations) == 0

    def test_aggregate_by_class(self, controller):
        """测试按类别聚合"""
        allocation = {"SPY": 0.15, "QQQ": 0.15, "IWM": 0.15}
        class_weights = controller._aggregate_by_class(allocation)
        assert "stocks" in class_weights


class TestCheckSoftLimits:
    """软性约束检查测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_volatility_warning(self, controller):
        """测试波动率警告"""
        allocation = {"stocks": 0.80, "bonds": 0.20}
        market_data = {"volatilities": {"stocks": 0.25, "bonds": 0.05}}
        warnings = controller._check_soft_limits(allocation, market_data)
        assert any("波动率" in w for w in warnings)

    def test_diversification_warning(self, controller):
        """测试分散化警告"""
        allocation = {"stocks": 0.90, "bonds": 0.10}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        warnings = controller._check_soft_limits(allocation, market_data)
        # 可能有分散化警告
        assert isinstance(warnings, list)


class TestRiskCalculations:
    """风险计算测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_calculate_var_95(self, controller):
        """测试95% VaR计算"""
        allocation = {"stocks": 0.60, "bonds": 0.40}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        var = controller._calculate_var(allocation, market_data, 0.95)
        assert var > 0
        assert var < 0.10  # 合理范围

    def test_calculate_var_99(self, controller):
        """测试99% VaR计算"""
        allocation = {"stocks": 0.60, "bonds": 0.40}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        var = controller._calculate_var(allocation, market_data, 0.99)
        assert var > controller._calculate_var(allocation, market_data, 0.95)

    def test_calculate_cvar(self, controller):
        """测试CVaR计算"""
        allocation = {"stocks": 0.60, "bonds": 0.40}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        cvar = controller._calculate_cvar(allocation, market_data, 0.99)
        var = controller._calculate_var(allocation, market_data, 0.99)
        assert cvar > var  # CVaR应大于VaR

    def test_calculate_volatility(self, controller):
        """测试波动率计算"""
        allocation = {"stocks": 0.50, "bonds": 0.50}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        vol = controller._calculate_volatility(allocation, market_data)
        assert vol > 0
        # 加权波动率应在两者之间
        assert vol < 0.20

    def test_calculate_volatility_no_data(self, controller):
        """测试无数据时的波动率"""
        vol = controller._calculate_volatility({}, {})
        assert vol == 0.15  # 默认值

    def test_calculate_sharpe(self, controller):
        """测试夏普比率计算"""
        allocation = {"stocks": 0.60, "bonds": 0.40}
        market_data = {
            "volatilities": {"stocks": 0.20, "bonds": 0.05},
            "expected_returns": {"stocks": 0.10, "bonds": 0.04},
        }
        sharpe = controller._calculate_sharpe(allocation, market_data)
        assert isinstance(sharpe, float)

    def test_calculate_sharpe_no_data(self, controller):
        """测试无数据时的夏普比率"""
        sharpe = controller._calculate_sharpe({}, {})
        assert sharpe == 0.5  # 默认值


class TestDrawdownCalculation:
    """回撤计算测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_no_drawdown(self, controller):
        """测试无回撤情况"""
        current, max_dd = controller._calculate_drawdown(1.0)
        assert current == 0.0
        assert max_dd == 0.0

    def test_drawdown_from_peak(self, controller):
        """测试从峰值回撤"""
        controller.peak_value = 100.0
        current, max_dd = controller._calculate_drawdown(90.0)
        assert current == -0.10
        assert max_dd == -0.10

    def test_new_peak(self, controller):
        """测试创新高"""
        controller.peak_value = 100.0
        current, max_dd = controller._calculate_drawdown(110.0)
        assert current == 0.0
        assert controller.peak_value == 110.0

    def test_max_drawdown_tracking(self, controller):
        """测试最大回撤跟踪"""
        controller.peak_value = 100.0
        controller._calculate_drawdown(85.0)  # -15%
        controller._calculate_drawdown(90.0)  # -10%
        current, max_dd = controller._calculate_drawdown(95.0)
        assert max_dd == -0.15  # 最大回撤保持

    def test_zero_peak_value(self, controller):
        """测试零峰值处理"""
        controller.peak_value = 0.0
        current, max_dd = controller._calculate_drawdown(100.0)
        assert current == 0.0


class TestConcentrationAssessment:
    """集中度评估测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_high_concentration(self, controller):
        """测试高集中度"""
        allocation = {"AAPL": 0.35, "MSFT": 0.35, "cash": 0.30}
        risk = controller._assess_concentration(allocation)
        assert risk == "high"

    def test_medium_concentration(self, controller):
        """测试中等集中度"""
        allocation = {"stocks": 0.25, "bonds": 0.25, "commodities": 0.25, "cash": 0.25}
        risk = controller._assess_concentration(allocation)
        assert risk == "medium"

    def test_low_concentration(self, controller):
        """测试低集中度"""
        allocation = {f"asset_{i}": 0.10 for i in range(10)}
        risk = controller._assess_concentration(allocation)
        assert risk == "low"

    def test_empty_allocation(self, controller):
        """测试空配置"""
        risk = controller._assess_concentration({})
        assert risk == "low"


class TestRecommendations:
    """建议生成测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_reduce_single_asset_recommendation(self, controller):
        """测试减少单资产建议"""
        allocation = {"AAPL": 0.25, "MSFT": 0.10}
        violations = ["单资产AAPL权重25.0%超过限制15.0%"]
        recommendations = controller._generate_recommendations(allocation, violations, [])
        assert "reduce_AAPL" in recommendations

    def test_reduce_volatility_recommendation(self, controller):
        """测试降低波动率建议"""
        allocation = {"stocks": 0.80}
        warnings = ["预期波动率18.0%超过目标12.0%"]
        recommendations = controller._generate_recommendations(allocation, [], warnings)
        assert "reduce_volatility" in recommendations


class TestGetConstraints:
    """获取约束测试"""

    def test_get_constraints(self):
        """测试获取约束条件"""
        rc = RiskController()
        constraints = rc.get_constraints()
        assert "max_single_asset" in constraints
        assert "target_volatility" in constraints
        assert "current_drawdown" in constraints

    def test_get_constraints_with_history(self):
        """测试有历史数据时的约束"""
        rc = RiskController()
        rc.portfolio_history = [100, 95, 90, 88]
        rc.peak_value = 100.0
        rc.max_drawdown_history = -0.12
        constraints = rc.get_constraints()
        assert constraints["max_drawdown"] == -0.12


class TestIntradayAlerts:
    """日内监控警报测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_vix_warning_alert(self, controller):
        """测试VIX警告级别"""
        market_data = {"macro": {"vix": 26.0}}
        alerts = controller.check_intraday_alerts(market_data, -0.02)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].alert_type == "vix_spike"

    def test_vix_critical_alert(self, controller):
        """测试VIX危急级别"""
        market_data = {"macro": {"vix": 35.0}}
        alerts = controller.check_intraday_alerts(market_data, -0.02)
        vix_alerts = [a for a in alerts if a.alert_type == "vix_spike"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].severity == "critical"

    def test_vix_emergency_alert(self, controller):
        """测试VIX紧急级别"""
        market_data = {"macro": {"vix": 45.0}}
        alerts = controller.check_intraday_alerts(market_data, -0.02)
        vix_alerts = [a for a in alerts if a.alert_type == "vix_spike"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].severity == "emergency"

    def test_no_vix_alert(self, controller):
        """测试无VIX警报"""
        market_data = {"macro": {"vix": 18.0}}
        alerts = controller.check_intraday_alerts(market_data, -0.02)
        vix_alerts = [a for a in alerts if a.alert_type == "vix_spike"]
        assert len(vix_alerts) == 0

    def test_drawdown_warning(self, controller):
        """测试回撤警告"""
        market_data = {"macro": {}}
        alerts = controller.check_intraday_alerts(market_data, -0.06)
        dd_alerts = [a for a in alerts if a.alert_type == "drawdown_trigger"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == "warning"

    def test_drawdown_critical(self, controller):
        """测试回撤危急"""
        market_data = {"macro": {}}
        alerts = controller.check_intraday_alerts(market_data, -0.12)
        dd_alerts = [a for a in alerts if a.alert_type == "drawdown_trigger"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == "critical"

    def test_drawdown_emergency(self, controller):
        """测试回撤紧急"""
        market_data = {"macro": {}}
        alerts = controller.check_intraday_alerts(market_data, -0.18)
        dd_alerts = [a for a in alerts if a.alert_type == "drawdown_trigger"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == "emergency"


class TestVolatilitySpike:
    """波动率突变测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_volatility_spike_detection(self, controller):
        """测试波动率突变检测"""
        # 设置上次波动率
        controller.last_volatility = 0.10
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.10}}
        alerts = controller.check_intraday_alerts(market_data, 0.0)
        vol_alerts = [a for a in alerts if a.alert_type == "volatility_regime"]
        assert len(vol_alerts) == 1

    def test_no_volatility_spike(self, controller):
        """测试无波动率突变"""
        controller.last_volatility = 0.10
        market_data = {"volatilities": {"stocks": 0.12, "bonds": 0.05}}
        alerts = controller.check_intraday_alerts(market_data, 0.0)
        vol_alerts = [a for a in alerts if a.alert_type == "volatility_regime"]
        assert len(vol_alerts) == 0


class TestEmergencyRebalance:
    """紧急再平衡测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_trigger_emergency_rebalance(self, controller):
        """测试触发紧急再平衡"""
        alert = IntradayAlert(
            timestamp="2024-01-15",
            alert_type="vix_spike",
            severity="emergency",
            current_value=50.0,
            threshold=40.0,
            message="VIX恐慌",
            recommended_action="立即防御",
        )
        should_trigger = controller.should_trigger_emergency_rebalance([alert])
        assert should_trigger is True

    def test_no_emergency_rebalance(self, controller):
        """测试不触发紧急再平衡"""
        alert = IntradayAlert(
            timestamp="2024-01-15",
            alert_type="vix_spike",
            severity="warning",
            current_value=26.0,
            threshold=25.0,
            message="VIX警告",
            recommended_action="密切关注",
        )
        should_trigger = controller.should_trigger_emergency_rebalance([alert])
        assert should_trigger is False

    def test_get_defensive_allocation(self, controller):
        """测试获取防御性配置"""
        defensive = controller.get_defensive_allocation()
        assert defensive["stocks"] == 0.20
        assert defensive["bonds"] == 0.40
        assert defensive["crypto"] == 0.00
        assert defensive["cash"] == 0.20


class TestAssess:
    """完整评估测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_assess_normal_allocation(self, controller):
        """测试正常配置评估"""
        current = {"stocks": 0.50, "bonds": 0.30, "cash": 0.20}
        proposed = {"stocks": 0.45, "bonds": 0.35, "cash": 0.20}
        market_data = {
            "volatilities": {"stocks": 0.15, "bonds": 0.05, "cash": 0.0},
            "expected_returns": {"stocks": 0.08, "bonds": 0.04, "cash": 0.02},
        }
        assessment = controller.assess(current, proposed, market_data, 1000000)
        assert isinstance(assessment, RiskAssessment)
        assert assessment.veto is False

    def test_assess_violation_triggers_veto(self, controller):
        """测试违规触发否决"""
        current = {"stocks": 0.40}
        proposed = {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.20}  # 单资产超限
        market_data = {"volatilities": {}}
        assessment = controller.assess(current, proposed, market_data, 1000000)
        assert assessment.veto is True
        assert len(assessment.violations) > 0


class TestAssessWithIntraday:
    """带日内监控的完整评估测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_assess_with_intraday_normal(self, controller):
        """测试正常情况下的日内评估"""
        current = {"stocks": 0.50, "bonds": 0.50}
        proposed = {"stocks": 0.45, "bonds": 0.55}
        market_data = {
            "macro": {"vix": 15.0},
            "volatilities": {"stocks": 0.15, "bonds": 0.05},
            "expected_returns": {"stocks": 0.08, "bonds": 0.04},
        }
        assessment = controller.assess_with_intraday(current, proposed, market_data, 1000000)
        assert assessment.emergency_rebalance is False

    def test_assess_with_intraday_emergency(self, controller):
        """测试紧急情况下的日内评估"""
        current = {"stocks": 0.50, "bonds": 0.50}
        proposed = {"stocks": 0.45, "bonds": 0.55}
        market_data = {
            "macro": {"vix": 50.0},  # 紧急VIX
            "volatilities": {"stocks": 0.15, "bonds": 0.05},
            "expected_returns": {"stocks": 0.08, "bonds": 0.04},
        }
        # 设置回撤也触发紧急
        controller.peak_value = 1000000
        assessment = controller.assess_with_intraday(current, proposed, market_data, 1000000)
        assert assessment.emergency_rebalance is True
        assert assessment.veto is True
        assert assessment.defensive_allocation is not None


class TestDiversificationRatio:
    """分散化比率测试"""

    @pytest.fixture
    def controller(self):
        return RiskController()

    def test_calculate_diversification_ratio(self, controller):
        """测试分散化比率计算"""
        allocation = {"stocks": 0.50, "bonds": 0.50}
        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.05}}
        ratio = controller._calculate_diversification_ratio(allocation, market_data)
        assert ratio > 1.0  # 分散化应>1

    def test_diversification_no_data(self, controller):
        """测试无数据时的分散化比率"""
        ratio = controller._calculate_diversification_ratio({}, {})
        assert ratio == 1.5  # 默认值

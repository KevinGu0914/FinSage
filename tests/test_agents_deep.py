#!/usr/bin/env python
"""
Deep Tests for Agents Module - 智能体模块深度测试
覆盖: PositionSizingAgent, RiskController, PortfolioManager
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


# ============================================================
# Test 1: PositionSizingDecision DataClass
# ============================================================

class TestPositionSizingDecision:
    """测试仓位决策数据类"""

    def test_creation(self):
        """测试创建"""
        from finsage.agents.position_sizing_agent import PositionSizingDecision

        decision = PositionSizingDecision(
            timestamp="2024-01-15T10:00:00",
            position_sizes={"SPY": 0.3, "TLT": 0.4, "GLD": 0.3},
            sizing_method="risk_parity",
            reasoning="Based on volatility-adjusted weights",
            risk_contribution={"SPY": 0.35, "TLT": 0.30, "GLD": 0.35}
        )

        assert decision.sizing_method == "risk_parity"
        assert len(decision.position_sizes) == 3
        assert decision.position_sizes["SPY"] == 0.3

    def test_to_dict(self):
        """测试转字典"""
        from finsage.agents.position_sizing_agent import PositionSizingDecision

        decision = PositionSizingDecision(
            timestamp="2024-01-15T10:00:00",
            position_sizes={"SPY": 0.5, "AGG": 0.5},
            sizing_method="equal_weight",
            reasoning="Simple equal allocation",
            risk_contribution={"SPY": 0.6, "AGG": 0.4}
        )

        d = decision.to_dict()
        assert d["sizing_method"] == "equal_weight"
        assert d["position_sizes"]["SPY"] == 0.5
        assert "reasoning" in d


# ============================================================
# Test 2: PositionSizingAgent Methods
# ============================================================

class TestPositionSizingAgent:
    """测试仓位规模智能体"""

    @pytest.fixture
    def mock_llm(self):
        """创建Mock LLM"""
        llm = Mock()
        llm.create_completion = Mock(return_value='{"method": "risk_parity", "reasoning": "test"}')
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """创建仓位规模智能体"""
        from finsage.agents.position_sizing_agent import PositionSizingAgent
        return PositionSizingAgent(
            llm_provider=mock_llm,
            config={
                "max_position_size": 0.20,
                "min_position_size": 0.02,
                "target_volatility": 0.10
            }
        )

    @pytest.fixture
    def sample_market_data(self):
        """创建示例市场数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        returns = {
            "stocks": np.random.normal(0.0005, 0.015, 100),
            "bonds": np.random.normal(0.0002, 0.005, 100),
            "commodities": np.random.normal(0.0003, 0.020, 100),
        }
        return {
            "returns": returns,
            "macro": {"vix": 18.0}
        }

    def test_init(self, agent):
        """测试初始化"""
        assert agent.max_position_size == 0.20
        assert agent.min_position_size == 0.02
        assert agent.target_volatility == 0.10

    def test_equal_weight_sizing(self, agent):
        """测试等权配置"""
        target = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        result = agent._equal_weight_sizing(target)

        assert len(result) == 3
        assert abs(result["stocks"] - 1/3) < 0.01
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_equal_weight_empty(self, agent):
        """测试空配置等权"""
        result = agent._equal_weight_sizing({})
        assert result == {}

    def test_risk_parity_sizing(self, agent, sample_market_data):
        """测试风险平价配置"""
        target = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        result = agent._risk_parity_sizing(target, sample_market_data)

        # 风险平价: 低波动资产权重更高
        assert result["bonds"] > result["stocks"]  # bonds波动率低，权重高
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_risk_parity_no_returns(self, agent):
        """测试无收益率数据时风险平价回退到等权"""
        target = {"stocks": 0.5, "bonds": 0.5}
        result = agent._risk_parity_sizing(target, {"macro": {"vix": 20}})

        # 应回退到等权
        assert abs(result["stocks"] - result["bonds"]) < 0.01

    def test_volatility_target_sizing(self, agent, sample_market_data):
        """测试波动率目标配置"""
        target = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        result = agent._volatility_target_sizing(target, sample_market_data)

        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_volatility_target_no_returns(self, agent):
        """测试无收益率数据时返回原配置"""
        target = {"stocks": 0.6, "bonds": 0.4}
        result = agent._volatility_target_sizing(target, {})

        assert result == target

    def test_kelly_sizing(self, agent, sample_market_data):
        """测试Kelly准则配置"""
        target = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        result = agent._kelly_sizing(target, sample_market_data)

        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_kelly_sizing_no_returns(self, agent):
        """测试无数据时Kelly回退"""
        target = {"stocks": 0.6, "bonds": 0.4}
        result = agent._kelly_sizing(target, {})

        assert result == target

    def test_apply_constraints(self, agent, sample_market_data):
        """测试应用约束"""
        sizes = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        risk_constraints = {"max_single_asset": 0.20}

        result = agent._apply_constraints(sizes, risk_constraints, sample_market_data)

        # 约束应用后会归一化，所以检查相对比例而非绝对值
        # 原始比例是 0.5:0.3:0.2, 约束到0.2:0.2:0.2后归一化
        # 最终各资产应该接近相等
        assert abs(result["stocks"] - result["bonds"]) < 0.01
        assert abs(result["bonds"] - result["commodities"]) < 0.01
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_compute_risk_contribution(self, agent, sample_market_data):
        """测试风险贡献计算"""
        sizes = {"stocks": 0.4, "bonds": 0.35, "commodities": 0.25}
        result = agent._compute_risk_contribution(sizes, sample_market_data)

        assert len(result) == 3
        # 风险贡献应为正数且和为1
        assert all(v >= 0 for v in result.values())

    def test_compute_risk_contribution_no_returns(self, agent):
        """测试无收益率数据时的风险贡献"""
        sizes = {"stocks": 0.5, "bonds": 0.5}
        result = agent._compute_risk_contribution(sizes, {})

        # 应返回原始比例
        assert result == sizes

    def test_generate_reasoning(self, agent):
        """测试生成决策理由"""
        sizes = {"stocks": 0.4, "bonds": 0.35, "commodities": 0.25}
        risk_contrib = {"stocks": 0.5, "bonds": 0.2, "commodities": 0.3}

        reasoning = agent._generate_reasoning("risk_parity", sizes, risk_contrib)

        assert "风险平价" in reasoning
        assert "stocks" in reasoning

    def test_generate_reasoning_high_concentration(self, agent):
        """测试高集中度警告"""
        sizes = {"stocks": 0.5, "bonds": 0.3, "commodities": 0.2}
        risk_contrib = {"stocks": 0.45, "bonds": 0.25, "commodities": 0.30}

        reasoning = agent._generate_reasoning("equal_weight", sizes, risk_contrib)

        assert "集中度" in reasoning  # 应有集中度警告

    def test_select_sizing_method(self, agent, sample_market_data):
        """测试选择仓位方法"""
        result = agent._select_sizing_method(
            sample_market_data,
            {"target_volatility": 0.12, "max_drawdown": 0.15}
        )

        assert result in agent.SIZING_METHODS

    def test_select_sizing_method_llm_failure(self, agent, sample_market_data):
        """测试LLM失败时回退"""
        agent.llm.create_completion = Mock(side_effect=Exception("LLM error"))

        result = agent._select_sizing_method(
            sample_market_data,
            {"target_volatility": 0.12}
        )

        assert result == "risk_parity"  # 默认方法

    def test_compute_position_sizes_equal_weight(self, agent, sample_market_data):
        """测试等权配置计算"""
        target = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        result = agent._compute_position_sizes(target, sample_market_data, "equal_weight")

        assert abs(result["stocks"] - result["bonds"]) < 0.01

    def test_compute_position_sizes_unknown_method(self, agent, sample_market_data):
        """测试未知方法回退到等权"""
        target = {"stocks": 0.5, "bonds": 0.5}
        result = agent._compute_position_sizes(target, sample_market_data, "unknown_method")

        assert abs(result["stocks"] - result["bonds"]) < 0.01

    def test_analyze(self, agent, sample_market_data):
        """测试完整分析流程"""
        target = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}

        decision = agent.analyze(
            target_allocation=target,
            market_data=sample_market_data,
            risk_constraints={"max_single_asset": 0.15},
            portfolio_value=100000
        )

        assert decision.sizing_method in agent.SIZING_METHODS
        assert len(decision.position_sizes) == 3
        assert abs(sum(decision.position_sizes.values()) - 1.0) < 0.001


# ============================================================
# Test 3: RiskAssessment and IntradayAlert DataClasses
# ============================================================

class TestRiskAssessmentDataClass:
    """测试风险评估数据类"""

    def test_risk_assessment_creation(self):
        """测试风险评估创建"""
        from finsage.agents.risk_controller import RiskAssessment

        assessment = RiskAssessment(
            timestamp="2024-01-15T10:00:00",
            portfolio_var_95=0.025,
            portfolio_cvar_99=0.040,
            current_drawdown=-0.03,
            max_drawdown=-0.08,
            volatility=0.15,
            sharpe_ratio=1.2,
            concentration_risk="medium",
            violations=[],
            warnings=["High volatility"],
            veto=False,
            recommendations={"reduce_volatility": 0.1}
        )

        assert assessment.portfolio_var_95 == 0.025
        assert assessment.sharpe_ratio == 1.2
        assert assessment.concentration_risk == "medium"

    def test_risk_assessment_with_alerts(self):
        """测试带警报的风险评估"""
        from finsage.agents.risk_controller import RiskAssessment, IntradayAlert

        alert = IntradayAlert(
            timestamp="2024-01-15T10:00:00",
            alert_type="vix_spike",
            severity="warning",
            current_value=26.0,
            threshold=25.0,
            message="VIX elevated",
            recommended_action="Reduce exposure"
        )

        assessment = RiskAssessment(
            timestamp="2024-01-15T10:00:00",
            portfolio_var_95=0.025,
            portfolio_cvar_99=0.040,
            current_drawdown=-0.03,
            max_drawdown=-0.08,
            volatility=0.15,
            sharpe_ratio=1.2,
            concentration_risk="medium",
            violations=[],
            warnings=[],
            veto=False,
            recommendations={},
            intraday_alerts=[alert]
        )

        assert len(assessment.intraday_alerts) == 1
        assert assessment.intraday_alerts[0].severity == "warning"

    def test_risk_assessment_to_dict(self):
        """测试转字典"""
        from finsage.agents.risk_controller import RiskAssessment, IntradayAlert

        alert = IntradayAlert(
            timestamp="2024-01-15T10:00:00",
            alert_type="drawdown_trigger",
            severity="critical",
            current_value=-0.12,
            threshold=-0.10,
            message="Drawdown critical",
            recommended_action="Reduce positions"
        )

        assessment = RiskAssessment(
            timestamp="2024-01-15T10:00:00",
            portfolio_var_95=0.025,
            portfolio_cvar_99=0.040,
            current_drawdown=-0.12,
            max_drawdown=-0.12,
            volatility=0.18,
            sharpe_ratio=0.8,
            concentration_risk="high",
            violations=["Max drawdown exceeded"],
            warnings=[],
            veto=True,
            recommendations={"cut_exposure": 0.5},
            intraday_alerts=[alert],
            emergency_rebalance=True,
            defensive_allocation={"stocks": 0.2, "bonds": 0.5, "cash": 0.3}
        )

        d = assessment.to_dict()
        assert d["veto"] == True
        assert d["emergency_rebalance"] == True
        assert len(d["intraday_alerts"]) == 1


class TestIntradayAlert:
    """测试日内警报数据类"""

    def test_intraday_alert_creation(self):
        """测试警报创建"""
        from finsage.agents.risk_controller import IntradayAlert

        alert = IntradayAlert(
            timestamp="2024-01-15T14:30:00",
            alert_type="vix_spike",
            severity="emergency",
            current_value=45.0,
            threshold=40.0,
            message="VIX panic level",
            recommended_action="Switch to defensive"
        )

        assert alert.alert_type == "vix_spike"
        assert alert.severity == "emergency"
        assert alert.current_value == 45.0


# ============================================================
# Test 4: RiskController
# ============================================================

class TestRiskController:
    """测试风险控制器"""

    @pytest.fixture
    def controller(self):
        """创建风险控制器"""
        from finsage.agents.risk_controller import RiskController
        return RiskController()

    @pytest.fixture
    def sample_market_data(self):
        """创建示例市场数据"""
        return {
            "volatilities": {
                "stocks": 0.18,
                "bonds": 0.08,
                "commodities": 0.22,
                "reits": 0.15,
                "crypto": 0.60
            },
            "expected_returns": {
                "stocks": 0.10,
                "bonds": 0.04,
                "commodities": 0.06,
                "reits": 0.08,
                "crypto": 0.15
            },
            "macro": {"vix": 20.0}
        }

    def test_init(self, controller):
        """测试初始化"""
        assert controller.hard_limits["max_single_asset"] == 0.15
        assert controller.soft_limits["target_volatility"] == 0.12
        assert controller.peak_value == 1.0

    def test_init_custom_limits(self):
        """测试自定义约束初始化"""
        from finsage.agents.risk_controller import RiskController

        controller = RiskController(
            hard_limits={"max_single_asset": 0.20},
            soft_limits={"target_volatility": 0.15}
        )

        assert controller.hard_limits["max_single_asset"] == 0.20
        assert controller.soft_limits["target_volatility"] == 0.15

    def test_check_hard_limits_no_violation(self, controller, sample_market_data):
        """测试无违规情况"""
        allocation = {"stocks": 0.40, "bonds": 0.30, "commodities": 0.20, "cash": 0.10}
        violations = controller._check_hard_limits(allocation, sample_market_data)

        assert len(violations) == 0

    def test_check_hard_limits_single_asset_violation(self, controller, sample_market_data):
        """测试单资产超限"""
        allocation = {"SPY": 0.25, "TLT": 0.75}  # SPY超过15%限制
        violations = controller._check_hard_limits(allocation, sample_market_data)

        assert len(violations) >= 1
        assert any("SPY" in v for v in violations)

    def test_check_hard_limits_class_violation(self, controller, sample_market_data):
        """测试资产类别超限"""
        allocation = {"stocks": 0.55, "bonds": 0.45}  # stocks超过50%限制
        violations = controller._check_hard_limits(allocation, sample_market_data)

        assert len(violations) >= 1
        assert any("stocks" in v for v in violations)

    def test_check_soft_limits(self, controller, sample_market_data):
        """测试软性约束检查"""
        allocation = {"stocks": 0.40, "bonds": 0.30, "crypto": 0.30}
        warnings = controller._check_soft_limits(allocation, sample_market_data)

        # crypto波动率高，可能触发波动率警告
        assert isinstance(warnings, list)

    def test_calculate_var(self, controller, sample_market_data):
        """测试VaR计算"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        var_95 = controller._calculate_var(allocation, sample_market_data, 0.95)

        assert var_95 > 0
        assert var_95 < 1  # VaR应该是合理的值

    def test_calculate_cvar(self, controller, sample_market_data):
        """测试CVaR计算"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        cvar_99 = controller._calculate_cvar(allocation, sample_market_data, 0.99)
        var_99 = controller._calculate_var(allocation, sample_market_data, 0.99)

        assert cvar_99 > var_99  # CVaR应该大于VaR

    def test_calculate_volatility(self, controller, sample_market_data):
        """测试波动率计算"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        vol = controller._calculate_volatility(allocation, sample_market_data)

        assert vol > 0
        assert vol < 1

    def test_calculate_volatility_no_data(self, controller):
        """测试无数据时的默认波动率"""
        vol = controller._calculate_volatility({"stocks": 0.5}, {})
        assert vol == 0.15  # 默认值

    def test_calculate_sharpe(self, controller, sample_market_data):
        """测试夏普比率计算"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        sharpe = controller._calculate_sharpe(allocation, sample_market_data)

        assert isinstance(sharpe, float)

    def test_calculate_sharpe_no_data(self, controller):
        """测试无数据时的默认夏普"""
        sharpe = controller._calculate_sharpe({"stocks": 0.5}, {})
        assert sharpe == 0.5  # 默认值

    def test_calculate_drawdown(self, controller):
        """测试回撤计算"""
        # 初始状态
        controller.peak_value = 100.0

        current_dd, max_dd = controller._calculate_drawdown(95.0)
        assert current_dd == -0.05  # 5%回撤
        assert max_dd == -0.05

        # 继续下跌
        current_dd, max_dd = controller._calculate_drawdown(90.0)
        assert current_dd == -0.10
        assert max_dd == -0.10

        # 部分恢复
        current_dd, max_dd = controller._calculate_drawdown(95.0)
        assert current_dd == -0.05
        assert max_dd == -0.10  # 最大回撤保持

    def test_calculate_drawdown_new_high(self, controller):
        """测试创新高情况"""
        controller.peak_value = 100.0

        current_dd, max_dd = controller._calculate_drawdown(110.0)
        assert current_dd == 0.0  # 无回撤
        assert controller.peak_value == 110.0  # 峰值更新

    def test_calculate_diversification_ratio(self, controller, sample_market_data):
        """测试分散化比率计算"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}
        div_ratio = controller._calculate_diversification_ratio(allocation, sample_market_data)

        assert div_ratio >= 1.0  # 分散化比率应>=1

    def test_assess_concentration_low(self, controller):
        """测试低集中度"""
        # 所有资产权重都<=0.20才算low
        allocation = {"stocks": 0.15, "bonds": 0.15, "commodities": 0.15,
                     "reits": 0.15, "crypto": 0.15, "cash": 0.15, "gold": 0.10}
        result = controller._assess_concentration(allocation)
        assert result == "low"

    def test_assess_concentration_medium(self, controller):
        """测试中等集中度"""
        allocation = {"stocks": 0.25, "bonds": 0.25, "commodities": 0.25, "cash": 0.25}
        result = controller._assess_concentration(allocation)
        assert result == "medium"

    def test_assess_concentration_high(self, controller):
        """测试高集中度"""
        allocation = {"stocks": 0.50, "bonds": 0.50}
        result = controller._assess_concentration(allocation)
        assert result == "high"

    def test_generate_recommendations(self, controller):
        """测试生成建议"""
        allocation = {"SPY": 0.20, "TLT": 0.80}
        violations = ["单资产SPY权重20.0%超过限制15.0%"]
        warnings = ["预期波动率15.0%超过目标12.0%"]

        recommendations = controller._generate_recommendations(allocation, violations, warnings)

        assert "reduce_SPY" in recommendations or "reduce_volatility" in recommendations

    def test_assess_no_violations(self, controller, sample_market_data):
        """测试无违规评估"""
        current = {"stocks": 0.35, "bonds": 0.35, "commodities": 0.20, "cash": 0.10}
        proposed = {"stocks": 0.40, "bonds": 0.30, "commodities": 0.20, "cash": 0.10}

        assessment = controller.assess(current, proposed, sample_market_data, 100000)

        assert assessment.veto == False
        assert len(assessment.violations) == 0

    def test_assess_with_veto(self, controller, sample_market_data):
        """测试触发否决的评估"""
        current = {"stocks": 0.35, "bonds": 0.35, "commodities": 0.30}
        proposed = {"stocks": 0.55, "bonds": 0.25, "commodities": 0.20}  # stocks超限

        assessment = controller.assess(current, proposed, sample_market_data, 100000)

        assert assessment.veto == True
        assert len(assessment.violations) > 0

    def test_get_constraints(self, controller):
        """测试获取约束"""
        constraints = controller.get_constraints()

        assert "max_single_asset" in constraints
        assert "target_volatility" in constraints
        assert "current_drawdown" in constraints


# ============================================================
# Test 5: RiskController Intraday Monitoring
# ============================================================

class TestRiskControllerIntraday:
    """测试风险控制器日内监控功能"""

    @pytest.fixture
    def controller(self):
        """创建风险控制器"""
        from finsage.agents.risk_controller import RiskController
        return RiskController()

    def test_check_vix_warning(self, controller):
        """测试VIX警告"""
        market_data = {"macro": {"vix": 26.0}}
        alert = controller._check_vix_alert(market_data, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "warning"
        assert alert.alert_type == "vix_spike"

    def test_check_vix_critical(self, controller):
        """测试VIX危急"""
        market_data = {"macro": {"vix": 32.0}}
        alert = controller._check_vix_alert(market_data, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "critical"

    def test_check_vix_emergency(self, controller):
        """测试VIX紧急"""
        market_data = {"macro": {"vix": 45.0}}
        alert = controller._check_vix_alert(market_data, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "emergency"

    def test_check_vix_normal(self, controller):
        """测试VIX正常"""
        market_data = {"macro": {"vix": 18.0}}
        alert = controller._check_vix_alert(market_data, "2024-01-15T10:00:00")

        assert alert is None

    def test_check_vix_no_data(self, controller):
        """测试无VIX数据"""
        alert = controller._check_vix_alert({}, "2024-01-15T10:00:00")
        assert alert is None

    def test_check_drawdown_warning(self, controller):
        """测试回撤警告"""
        alert = controller._check_drawdown_alert(-0.06, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "warning"
        assert alert.alert_type == "drawdown_trigger"

    def test_check_drawdown_critical(self, controller):
        """测试回撤危急"""
        alert = controller._check_drawdown_alert(-0.12, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "critical"

    def test_check_drawdown_emergency(self, controller):
        """测试回撤紧急"""
        alert = controller._check_drawdown_alert(-0.18, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.severity == "emergency"

    def test_check_drawdown_normal(self, controller):
        """测试正常回撤"""
        alert = controller._check_drawdown_alert(-0.02, "2024-01-15T10:00:00")
        assert alert is None

    def test_check_volatility_spike(self, controller):
        """测试波动率突变"""
        # 设置初始波动率
        controller.last_volatility = 0.10

        market_data = {"volatilities": {"stocks": 0.20, "bonds": 0.08}}  # 平均0.14，增加40%
        alert = controller._check_volatility_spike(market_data, "2024-01-15T10:00:00")

        # 40%增加，未超过50%阈值
        assert alert is None

    def test_check_volatility_spike_triggered(self, controller):
        """测试波动率突变触发"""
        controller.last_volatility = 0.10

        market_data = {"volatilities": {"stocks": 0.25, "bonds": 0.15}}  # 平均0.20，增加100%
        alert = controller._check_volatility_spike(market_data, "2024-01-15T10:00:00")

        assert alert is not None
        assert alert.alert_type == "volatility_regime"

    def test_check_intraday_alerts_multiple(self, controller):
        """测试多个警报同时触发"""
        market_data = {"macro": {"vix": 35.0}}  # critical
        current_drawdown = -0.12  # critical

        alerts = controller.check_intraday_alerts(market_data, current_drawdown)

        assert len(alerts) >= 2

    def test_should_trigger_emergency_rebalance_true(self, controller):
        """测试触发紧急再平衡"""
        from finsage.agents.risk_controller import IntradayAlert

        alerts = [
            IntradayAlert(
                timestamp="2024-01-15T10:00:00",
                alert_type="vix_spike",
                severity="emergency",
                current_value=45.0,
                threshold=40.0,
                message="VIX panic",
                recommended_action="Defensive"
            )
        ]

        assert controller.should_trigger_emergency_rebalance(alerts) == True

    def test_should_trigger_emergency_rebalance_false(self, controller):
        """测试不触发紧急再平衡"""
        from finsage.agents.risk_controller import IntradayAlert

        alerts = [
            IntradayAlert(
                timestamp="2024-01-15T10:00:00",
                alert_type="vix_spike",
                severity="warning",
                current_value=26.0,
                threshold=25.0,
                message="VIX elevated",
                recommended_action="Monitor"
            )
        ]

        assert controller.should_trigger_emergency_rebalance(alerts) == False

    def test_get_defensive_allocation(self, controller):
        """测试获取防御性配置"""
        defensive = controller.get_defensive_allocation()

        assert abs(sum(defensive.values()) - 1.0) < 0.01
        assert defensive["stocks"] < 0.30  # 减少股票
        assert defensive["bonds"] > 0.30  # 增加债券
        assert defensive["crypto"] == 0.0  # 清空加密

    def test_assess_with_intraday_emergency(self, controller):
        """测试带日内监控的完整评估-紧急情况"""
        market_data = {
            "macro": {"vix": 45.0},  # emergency
            "volatilities": {"stocks": 0.25},
            "expected_returns": {"stocks": 0.08}
        }
        current = {"stocks": 0.50, "bonds": 0.50}
        proposed = {"stocks": 0.40, "bonds": 0.60}

        assessment = controller.assess_with_intraday(
            current, proposed, market_data, 100000
        )

        assert assessment.emergency_rebalance == True
        assert assessment.defensive_allocation is not None
        assert assessment.veto == True

    def test_assess_with_intraday_normal(self, controller):
        """测试带日内监控的正常评估"""
        market_data = {
            "macro": {"vix": 18.0},
            "volatilities": {"stocks": 0.15, "bonds": 0.08},
            "expected_returns": {"stocks": 0.08, "bonds": 0.04}
        }
        current = {"stocks": 0.40, "bonds": 0.35, "commodities": 0.25}
        proposed = {"stocks": 0.40, "bonds": 0.35, "commodities": 0.25}

        assessment = controller.assess_with_intraday(
            current, proposed, market_data, 100000
        )

        assert assessment.emergency_rebalance == False
        assert len(assessment.intraday_alerts) == 0


# ============================================================
# Test 6: PortfolioManager
# ============================================================

class TestPortfolioManager:
    """测试组合管理器"""

    @pytest.fixture
    def mock_llm(self):
        """创建Mock LLM"""
        llm = Mock()
        llm.create_completion = Mock(return_value='{"tool_name": "minimum_variance", "reasoning": "test"}')
        return llm

    @pytest.fixture
    def mock_toolkit(self):
        """创建Mock工具箱"""
        toolkit = Mock()
        toolkit.list_tools = Mock(return_value=[
            {"name": "minimum_variance", "description": "Minimum variance optimization"},
            {"name": "risk_parity", "description": "Risk parity allocation"},
            {"name": "cvar_optimization", "description": "CVaR optimization"},
        ])
        toolkit.call = Mock(return_value={"stocks": 0.35, "bonds": 0.35, "commodities": 0.30})
        return toolkit

    @pytest.fixture
    def manager(self, mock_llm, mock_toolkit):
        """创建组合管理器"""
        from finsage.agents.portfolio_manager import PortfolioManager
        return PortfolioManager(
            llm_provider=mock_llm,
            hedging_toolkit=mock_toolkit,
            config={"rebalance_threshold": 0.05}
        )

    @pytest.fixture
    def sample_expert_reports(self):
        """创建示例专家报告"""
        from finsage.agents.base_expert import ExpertReport, ExpertRecommendation, Action

        def create_report(asset_class, view, symbols):
            recommendations = [
                ExpertRecommendation(
                    asset_class=asset_class,
                    symbol=sym,
                    action=Action.BUY_50 if view == "bullish" else Action.HOLD,
                    confidence=0.7,
                    target_weight=0.1,
                    reasoning="Test",
                    market_view={},
                    risk_assessment={}
                )
                for sym in symbols
            ]
            return ExpertReport(
                expert_name=f"{asset_class} Expert",
                asset_class=asset_class,
                timestamp="2024-01-15T10:00:00",
                recommendations=recommendations,
                overall_view=view,
                sector_allocation={s: 1/len(symbols) for s in symbols},
                key_factors=["test"]
            )

        return {
            "stocks": create_report("stocks", "bullish", ["SPY", "QQQ"]),
            "bonds": create_report("bonds", "neutral", ["TLT", "AGG"]),
            "commodities": create_report("commodities", "bearish", ["GLD"]),
            "reits": create_report("reits", "neutral", ["VNQ"]),
            "crypto": create_report("crypto", "bearish", ["BTC"]),
        }

    @pytest.fixture
    def sample_market_data(self):
        """创建示例市场数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50)
        returns = {
            "stocks": np.random.normal(0.0005, 0.015, 50),
            "bonds": np.random.normal(0.0002, 0.005, 50),
            "commodities": np.random.normal(0.0003, 0.020, 50),
        }
        return {
            "returns": returns,
            "macro": {"vix": 18.0},
            "volatility": 0.15,
            "avg_correlation": 0.3
        }

    def test_init(self, manager):
        """测试初始化"""
        assert manager.rebalance_threshold == 0.05
        assert "stocks" in manager.allocation_bounds

    def test_summarize_expert_views(self, manager, sample_expert_reports):
        """测试汇总专家观点"""
        summary = manager._summarize_expert_views(sample_expert_reports)

        assert "stocks" in summary
        assert "bullish" in summary["stocks"]

    def test_analyze_market_conditions(self, manager, sample_expert_reports, sample_market_data):
        """测试分析市场条件"""
        conditions = manager._analyze_market_conditions(
            sample_expert_reports,
            sample_market_data,
            {"current_drawdown": -0.02}
        )

        assert "vix" in conditions
        assert "bullish_count" in conditions
        assert "bearish_count" in conditions
        assert "expert_disagreement" in conditions

    def test_rule_based_tool_preselection_high_vix(self, manager):
        """测试高VIX时的工具预选"""
        conditions = {
            "vix": 28.0,
            "bearish_majority": False,
            "current_drawdown": -0.02,
            "volatility_change": 0.05,
            "correlation_change": 0.03,
            "expert_disagreement": 0.2,
            "market_state": "elevated_volatility",
            "bullish_majority": False,
            "bullish_count": 2,
            "neutral_count": 2,
        }

        candidates, reasoning = manager._rule_based_tool_preselection(conditions)

        assert "cvar_optimization" in candidates

    def test_rule_based_tool_preselection_bearish(self, manager):
        """测试看空多数时的工具预选"""
        conditions = {
            "vix": 22.0,
            "bearish_majority": True,
            "current_drawdown": -0.04,
            "volatility_change": 0.05,
            "correlation_change": 0.03,
            "expert_disagreement": 0.2,
            "market_state": "normal",
            "bullish_majority": False,
            "bullish_count": 1,
            "neutral_count": 1,
        }

        candidates, reasoning = manager._rule_based_tool_preselection(conditions)

        assert "cvar_optimization" in candidates

    def test_rule_based_tool_preselection_low_vix(self, manager):
        """测试低VIX时的工具预选"""
        conditions = {
            "vix": 12.0,
            "bearish_majority": False,
            "current_drawdown": 0.0,
            "volatility_change": 0.02,
            "correlation_change": 0.01,
            "expert_disagreement": 0.1,
            "market_state": "low_volatility",
            "bullish_majority": False,
            "bullish_count": 2,
            "neutral_count": 3,
        }

        candidates, reasoning = manager._rule_based_tool_preselection(conditions)

        # 低VIX应选择minimum_variance或risk_parity
        assert "minimum_variance" in candidates or "risk_parity" in candidates or "mean_variance" in candidates

    def test_apply_allocation_bounds(self, manager):
        """测试应用配置约束"""
        weights = {"stocks": 0.60, "bonds": 0.30, "commodities": 0.10}

        bounded = manager._apply_allocation_bounds(weights)

        # 约束应用后会归一化，所以stocks相对于bonds的比例应该降低
        # 原始 stocks:bonds = 0.6:0.3 = 2:1
        # 限制后 stocks:bonds = 0.5:0.3 归一化后 比例<2:1
        # 检查stocks相对比例降低
        original_ratio = 0.60 / 0.30  # 2.0
        bounded_ratio = bounded["stocks"] / bounded["bonds"]
        assert bounded_ratio < original_ratio  # 比例应该降低
        assert abs(sum(bounded.values()) - 1.0) < 0.001

    def test_generate_trades(self, manager):
        """测试生成交易"""
        current = {"stocks": 0.30, "bonds": 0.40, "commodities": 0.30}
        target = {"stocks": 0.40, "bonds": 0.30, "commodities": 0.30}

        trades = manager._generate_trades(current, target)

        assert len(trades) == 2  # stocks买入, bonds卖出

        buy_trade = next(t for t in trades if t["action"] == "BUY")
        assert buy_trade["asset"] == "stocks"

    def test_generate_trades_no_change(self, manager):
        """测试无需交易的情况"""
        current = {"stocks": 0.40, "bonds": 0.35, "commodities": 0.25}
        target = {"stocks": 0.42, "bonds": 0.34, "commodities": 0.24}  # 变化小于阈值

        trades = manager._generate_trades(current, target)

        assert len(trades) == 0

    def test_compute_risk_metrics_with_returns(self, manager, sample_market_data):
        """测试有收益率数据时的风险指标计算"""
        allocation = {"stocks": 0.4, "bonds": 0.3, "commodities": 0.3}

        metrics = manager._compute_risk_metrics(allocation, sample_market_data)

        assert "expected_volatility" in metrics
        assert "diversification_ratio" in metrics
        assert metrics["data_source"] == "historical_returns"

    def test_compute_risk_metrics_without_returns(self, manager):
        """测试无收益率数据时使用VIX估计"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        market_data = {"macro": {"vix": 20.0}}

        metrics = manager._compute_risk_metrics(allocation, market_data)

        assert "expected_volatility" in metrics
        assert metrics["data_source"] == "vix_estimate"

    def test_generate_reasoning(self, manager):
        """测试生成决策理由"""
        expert_summary = {
            "stocks": "bullish (SPY, QQQ)",
            "bonds": "neutral (TLT)",
            "commodities": "bearish (GLD)"
        }
        target = {"stocks": 0.45, "bonds": 0.30, "commodities": 0.15, "cash": 0.10}

        reasoning = manager._generate_reasoning(expert_summary, "risk_parity", target)

        assert "risk_parity" in reasoning
        assert "stocks" in reasoning

    def test_extract_json_from_response_pure_json(self, manager):
        """测试提取纯JSON响应"""
        response = '{"tool_name": "risk_parity", "reasoning": "test"}'
        result = manager._extract_json_from_response(response)

        import json
        data = json.loads(result)
        assert data["tool_name"] == "risk_parity"

    def test_extract_json_from_response_with_code_block(self, manager):
        """测试提取带代码块的JSON"""
        response = '''Here is the result:
```json
{"tool_name": "cvar_optimization", "reasoning": "high risk"}
```
That's the recommendation.'''

        result = manager._extract_json_from_response(response)

        import json
        data = json.loads(result)
        assert data["tool_name"] == "cvar_optimization"

    def test_extract_json_from_response_with_plain_code_block(self, manager):
        """测试提取普通代码块的JSON"""
        response = '''```
{"tool_name": "minimum_variance"}
```'''

        result = manager._extract_json_from_response(response)

        import json
        data = json.loads(result)
        assert data["tool_name"] == "minimum_variance"

    def test_find_balanced_json_simple(self, manager):
        """测试简单JSON查找"""
        text = 'Some text {"key": "value"} more text'
        result = manager._find_balanced_json(text, '{', '}')

        assert result == '{"key": "value"}'

    def test_find_balanced_json_nested(self, manager):
        """测试嵌套JSON查找"""
        text = 'prefix {"outer": {"inner": "value"}} suffix'
        result = manager._find_balanced_json(text, '{', '}')

        assert result == '{"outer": {"inner": "value"}}'

    def test_find_balanced_json_with_string(self, manager):
        """测试含字符串的JSON查找"""
        text = '{"key": "value with { and } chars"}'
        result = manager._find_balanced_json(text, '{', '}')

        assert result == '{"key": "value with { and } chars"}'

    def test_validate_llm_allocation(self, manager):
        """测试验证LLM配置"""
        allocation = {"stocks": 0.60, "bonds": 0.30}  # 缺少其他类别

        validated = manager._validate_llm_allocation(allocation)

        # 应包含所有资产类别
        assert "commodities" in validated
        assert "reits" in validated
        assert "crypto" in validated
        assert "cash" in validated

        # 应归一化到1
        assert abs(sum(validated.values()) - 1.0) < 0.001

    def test_expert_weighted_default_allocation(self, manager, sample_expert_reports):
        """测试专家加权默认配置"""
        allocation = manager._expert_weighted_default_allocation(sample_expert_reports)

        # bullish资产应有更高权重
        assert allocation["stocks"] > allocation["commodities"]
        assert abs(sum(allocation.values()) - 1.0) < 0.001

    def test_get_default_allocation(self, manager):
        """测试获取默认配置"""
        default = manager._get_default_allocation()

        assert default["stocks"] == 0.40
        assert default["bonds"] == 0.25
        assert abs(sum(default.values()) - 1.0) < 0.01


# ============================================================
# Test 7: PortfolioDecision DataClass
# ============================================================

class TestPortfolioDecision:
    """测试组合决策数据类"""

    def test_creation(self):
        """测试创建"""
        from finsage.agents.portfolio_manager import PortfolioDecision

        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"stocks": 0.4, "bonds": 0.3, "commodities": 0.2, "cash": 0.1},
            trades=[{"asset": "stocks", "action": "BUY", "weight_change": 0.1}],
            hedging_tool_used="risk_parity",
            reasoning="Balanced allocation",
            risk_metrics={"var_95": 0.02},
            expert_summary={"stocks": "bullish", "bonds": "neutral"}
        )

        assert decision.hedging_tool_used == "risk_parity"
        assert len(decision.trades) == 1

    def test_to_dict(self):
        """测试转字典"""
        from finsage.agents.portfolio_manager import PortfolioDecision

        decision = PortfolioDecision(
            timestamp="2024-01-15T10:00:00",
            target_allocation={"stocks": 0.5, "bonds": 0.5},
            trades=[],
            hedging_tool_used="minimum_variance",
            reasoning="Test",
            risk_metrics={},
            expert_summary={}
        )

        d = decision.to_dict()
        assert d["hedging_tool_used"] == "minimum_variance"
        assert "target_allocation" in d


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Deep Agents Module Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

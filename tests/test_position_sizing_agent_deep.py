"""
Deep tests for PositionSizingAgent
仓位规模智能体深度测试
"""

import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from finsage.agents.position_sizing_agent import (
    PositionSizingAgent,
    PositionSizingDecision,
)


class TestPositionSizingDecision:
    """PositionSizingDecision数据类测试"""

    def test_create_decision(self):
        """测试创建仓位决策"""
        decision = PositionSizingDecision(
            timestamp="2024-01-15T10:30:00",
            position_sizes={"AAPL": 0.10, "MSFT": 0.10},
            sizing_method="risk_parity",
            reasoning="风险平价配置",
            risk_contribution={"AAPL": 0.50, "MSFT": 0.50},
        )
        assert decision.sizing_method == "risk_parity"
        assert "AAPL" in decision.position_sizes

    def test_to_dict(self):
        """测试转换为字典"""
        decision = PositionSizingDecision(
            timestamp="2024-01-15",
            position_sizes={"SPY": 0.25},
            sizing_method="equal_weight",
            reasoning="等权配置",
            risk_contribution={"SPY": 1.0},
        )
        result = decision.to_dict()
        assert "position_sizes" in result
        assert "sizing_method" in result
        assert result["sizing_method"] == "equal_weight"


class TestPositionSizingAgentInit:
    """PositionSizingAgent初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        llm = MagicMock()
        agent = PositionSizingAgent(llm_provider=llm)
        assert agent.max_position_size == 0.15
        assert agent.min_position_size == 0.01
        assert agent.target_volatility == 0.12

    def test_custom_config(self):
        """测试自定义配置"""
        llm = MagicMock()
        config = {
            "max_position_size": 0.20,
            "min_position_size": 0.02,
            "target_volatility": 0.15,
        }
        agent = PositionSizingAgent(llm_provider=llm, config=config)
        assert agent.max_position_size == 0.20
        assert agent.min_position_size == 0.02

    def test_sizing_methods(self):
        """测试仓位方法定义"""
        llm = MagicMock()
        agent = PositionSizingAgent(llm_provider=llm)
        assert "equal_weight" in agent.SIZING_METHODS
        assert "risk_parity" in agent.SIZING_METHODS
        assert "kelly" in agent.SIZING_METHODS


class TestEqualWeightSizing:
    """等权配置测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_equal_weight_basic(self, agent):
        """测试基本等权配置"""
        allocation = {"SPY": 0.4, "QQQ": 0.3, "IWM": 0.3}
        result = agent._equal_weight_sizing(allocation)
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.01
        for weight in result.values():
            assert abs(weight - 1/3) < 0.01

    def test_equal_weight_empty(self, agent):
        """测试空配置"""
        result = agent._equal_weight_sizing({})
        assert result == {}

    def test_equal_weight_single(self, agent):
        """测试单一资产"""
        result = agent._equal_weight_sizing({"SPY": 1.0})
        assert result["SPY"] == 1.0


class TestRiskParitySizing:
    """风险平价配置测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_risk_parity_basic(self, agent):
        """测试基本风险平价"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        # 创建收益率数据
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._risk_parity_sizing(allocation, market_data)

        # 低波动资产应获得更高权重
        assert result["bonds"] > result["stocks"]

    def test_risk_parity_no_returns(self, agent):
        """测试无收益率数据"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        result = agent._risk_parity_sizing(allocation, {})
        # 应回退到等权
        assert abs(result["stocks"] - 0.5) < 0.01

    def test_risk_parity_normalization(self, agent):
        """测试权重归一化"""
        allocation = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.03, 252),
            "B": np.random.normal(0.001, 0.02, 252),
            "C": np.random.normal(0.001, 0.01, 252),
            "D": np.random.normal(0.001, 0.015, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._risk_parity_sizing(allocation, market_data)
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestVolatilityTargetSizing:
    """波动率目标配置测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_volatility_target_basic(self, agent):
        """测试基本波动率目标"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._volatility_target_sizing(allocation, market_data)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_volatility_target_no_returns(self, agent):
        """测试无收益率数据"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        result = agent._volatility_target_sizing(allocation, {})
        assert result == allocation


class TestKellySizing:
    """Kelly准则配置测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_kelly_basic(self, agent):
        """测试基本Kelly配置"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._kelly_sizing(allocation, market_data)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_kelly_no_returns(self, agent):
        """测试无收益率数据"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        result = agent._kelly_sizing(allocation, {})
        assert result == allocation


class TestApplyConstraints:
    """约束应用测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_apply_max_constraint(self, agent):
        """测试最大权重约束"""
        sizes = {"A": 0.30, "B": 0.40, "C": 0.30}
        constraints = {"max_single_asset": 0.15}
        result = agent._apply_constraints(sizes, constraints, {})
        # 约束后归一化，所以所有值都被夹到0.15然后归一化到等权
        # 3个值各0.15，归一化后各0.333
        assert abs(result["A"] - result["B"]) < 0.01  # 约束后应该接近等权
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_apply_min_constraint(self, agent):
        """测试最小权重约束"""
        sizes = {"A": 0.005, "B": 0.50, "C": 0.495}
        result = agent._apply_constraints(sizes, {}, {})
        assert min(result.values()) >= agent.min_position_size - 0.001

    def test_normalization_after_constraints(self, agent):
        """测试约束后归一化"""
        sizes = {"A": 0.50, "B": 0.30, "C": 0.20}
        result = agent._apply_constraints(sizes, {"max_single_asset": 0.15}, {})
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestRiskContribution:
    """风险贡献计算测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_risk_contribution_basic(self, agent):
        """测试基本风险贡献计算"""
        sizes = {"stocks": 0.6, "bonds": 0.4}
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._compute_risk_contribution(sizes, market_data)
        assert abs(sum(result.values()) - 1.0) < 0.1  # 风险贡献应归一

    def test_risk_contribution_no_returns(self, agent):
        """测试无收益率数据"""
        sizes = {"A": 0.5, "B": 0.5}
        result = agent._compute_risk_contribution(sizes, {})
        assert result == sizes


class TestGenerateReasoning:
    """决策理由生成测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_generate_reasoning(self, agent):
        """测试生成决策理由"""
        sizes = {"AAPL": 0.15, "MSFT": 0.12, "GOOGL": 0.10, "AMZN": 0.08}
        risk_contrib = {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.20}
        reasoning = agent._generate_reasoning("risk_parity", sizes, risk_contrib)
        assert "风险平价" in reasoning
        assert "AAPL" in reasoning

    def test_reasoning_high_risk_concentration(self, agent):
        """测试高风险集中度提示"""
        sizes = {"AAPL": 0.50, "MSFT": 0.50}
        risk_contrib = {"AAPL": 0.40, "MSFT": 0.60}
        reasoning = agent._generate_reasoning("equal_weight", sizes, risk_contrib)
        assert "风险集中度" in reasoning


class TestSelectSizingMethod:
    """仓位方法选择测试"""

    @pytest.fixture
    def agent(self):
        llm = MagicMock()
        llm.create_completion.return_value = '{"method": "risk_parity", "reasoning": "市场波动适中"}'
        return PositionSizingAgent(llm_provider=llm)

    def test_select_method_success(self, agent):
        """测试成功选择方法"""
        market_data = {"macro": {"vix": 20.0}}
        constraints = {"target_volatility": 0.12}
        method = agent._select_sizing_method(market_data, constraints)
        assert method in agent.SIZING_METHODS

    def test_select_method_fallback(self):
        """测试选择失败时回退"""
        llm = MagicMock()
        llm.create_completion.side_effect = Exception("API error")
        agent = PositionSizingAgent(llm_provider=llm)
        method = agent._select_sizing_method({}, {})
        assert method == "risk_parity"

    def test_select_method_invalid_response(self):
        """测试无效响应"""
        llm = MagicMock()
        llm.create_completion.return_value = '{"method": "invalid_method"}'
        agent = PositionSizingAgent(llm_provider=llm)
        method = agent._select_sizing_method({}, {})
        assert method == "risk_parity"


class TestComputePositionSizes:
    """仓位大小计算测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_compute_equal_weight(self, agent):
        """测试等权计算"""
        allocation = {"A": 0.5, "B": 0.5}
        result = agent._compute_position_sizes(allocation, {}, "equal_weight")
        assert abs(result["A"] - 0.5) < 0.01

    def test_compute_risk_parity(self, agent):
        """测试风险平价计算"""
        allocation = {"stocks": 0.5, "bonds": 0.5}
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._compute_position_sizes(allocation, market_data, "risk_parity")
        assert sum(result.values()) > 0

    def test_compute_default_method(self, agent):
        """测试默认方法"""
        allocation = {"A": 0.5, "B": 0.5}
        result = agent._compute_position_sizes(allocation, {}, "unknown_method")
        # 应回退到等权
        assert abs(result["A"] - 0.5) < 0.01


class TestAnalyze:
    """完整分析测试"""

    @pytest.fixture
    def agent(self):
        llm = MagicMock()
        llm.create_completion.return_value = '{"method": "risk_parity", "reasoning": "适中波动"}'
        return PositionSizingAgent(llm_provider=llm)

    def test_analyze_basic(self, agent):
        """测试基本分析流程"""
        allocation = {"stocks": 0.6, "bonds": 0.4}
        np.random.seed(42)
        returns = pd.DataFrame({
            "stocks": np.random.normal(0.001, 0.02, 252),
            "bonds": np.random.normal(0.0005, 0.005, 252),
        })
        market_data = {"returns": returns.to_dict(orient='list'), "macro": {"vix": 20}}
        constraints = {"max_single_asset": 0.15, "target_volatility": 0.12}

        decision = agent.analyze(allocation, market_data, constraints, 1000000)

        assert isinstance(decision, PositionSizingDecision)
        assert len(decision.position_sizes) > 0
        assert abs(sum(decision.position_sizes.values()) - 1.0) < 0.01

    def test_analyze_empty_allocation(self, agent):
        """测试空配置分析"""
        decision = agent.analyze({}, {}, {}, 1000000)
        assert len(decision.position_sizes) == 0


class TestReviseBasedOnFeedback:
    """基于反馈修正测试"""

    @pytest.fixture
    def agent(self):
        llm = MagicMock()
        llm.create_completion.return_value = '{"position_sizes": {"stocks": 0.5, "bonds": 0.5}, "reasoning": "根据反馈调整"}'
        return PositionSizingAgent(llm_provider=llm)

    def test_revise_success(self, agent):
        """测试成功修正"""
        current = PositionSizingDecision(
            timestamp="2024-01-15",
            position_sizes={"stocks": 0.6, "bonds": 0.4},
            sizing_method="risk_parity",
            reasoning="初始配置",
            risk_contribution={"stocks": 0.7, "bonds": 0.3},
        )
        feedback = {
            "portfolio_manager": {"target_allocation": {"stocks": 0.5, "bonds": 0.5}},
            "hedging_agent": {"hedge_ratio": 0.1},
        }
        revised = agent.revise_based_on_feedback(current, feedback, {})
        assert isinstance(revised, PositionSizingDecision)
        assert "revised" in revised.sizing_method

    def test_revise_failure_returns_original(self):
        """测试修正失败返回原值"""
        llm = MagicMock()
        llm.create_completion.side_effect = Exception("API error")
        agent = PositionSizingAgent(llm_provider=llm)

        current = PositionSizingDecision(
            timestamp="2024-01-15",
            position_sizes={"stocks": 0.6, "bonds": 0.4},
            sizing_method="risk_parity",
            reasoning="初始配置",
            risk_contribution={"stocks": 0.7, "bonds": 0.3},
        )
        revised = agent.revise_based_on_feedback(current, {}, {})
        assert revised == current


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def agent(self):
        return PositionSizingAgent(llm_provider=MagicMock())

    def test_zero_volatility_asset(self, agent):
        """测试零波动率资产"""
        allocation = {"cash": 1.0}
        np.random.seed(42)
        returns = pd.DataFrame({
            "cash": np.zeros(252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._risk_parity_sizing(allocation, market_data)
        assert "cash" in result

    def test_negative_returns(self, agent):
        """测试负收益"""
        allocation = {"A": 0.5, "B": 0.5}
        returns = pd.DataFrame({
            "A": np.array([-0.01] * 252),
            "B": np.array([-0.005] * 252),
        })
        market_data = {"returns": returns.to_dict(orient='list')}
        result = agent._kelly_sizing(allocation, market_data)
        assert sum(result.values()) > 0  # 仍应产生有效配置

    def test_very_large_portfolio(self, agent):
        """测试大规模组合"""
        allocation = {f"asset_{i}": 1.0/50 for i in range(50)}
        result = agent._equal_weight_sizing(allocation)
        assert len(result) == 50
        assert abs(sum(result.values()) - 1.0) < 0.01

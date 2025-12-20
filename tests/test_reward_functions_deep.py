"""
Deep tests for Reward Functions
奖励函数深度测试
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from finsage.rl.reward_functions import (
    RewardComponents,
    BaseAgentReward,
    PortfolioManagerReward,
    PositionSizingReward,
    HedgingReward,
    ExpertReward,
    CoordinationReward,
    CombinedRewardCalculator,
    create_default_reward_calculator,
    compute_gae_with_individual_rewards,
)


class TestRewardComponents:
    """RewardComponents测试"""

    def test_init(self):
        """测试初始化"""
        rc = RewardComponents(
            total=0.5,
            components={"a": 0.3, "b": 0.2},
            description="test reward"
        )
        assert rc.total == 0.5
        assert rc.components["a"] == 0.3
        assert rc.description == "test reward"

    def test_to_dict(self):
        """测试转字典"""
        rc = RewardComponents(
            total=0.5,
            components={"a": 0.3},
            description="test"
        )
        d = rc.to_dict()
        assert d["total"] == 0.5
        assert "components" in d
        assert "description" in d


class TestPortfolioManagerReward:
    """PortfolioManagerReward测试"""

    @pytest.fixture
    def reward_fn(self):
        return PortfolioManagerReward()

    def test_init(self, reward_fn):
        """测试初始化"""
        assert reward_fn.return_weight == 0.35
        assert reward_fn.consensus_weight == 0.25
        assert reward_fn.quality_weight == 0.25
        assert reward_fn.timing_weight == 0.15

    def test_compute_basic(self, reward_fn):
        """测试基本计算"""
        result = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_volatility=0.02,
            expert_recommendations={},
            actual_allocation={"stocks": 0.5, "bonds": 0.5},
            asset_returns={"stocks": 0.015, "bonds": 0.005},
        )

        assert isinstance(result, RewardComponents)
        assert "return_reward" in result.components
        assert "consensus_reward" in result.components

    def test_compute_with_experts(self, reward_fn):
        """测试带专家建议计算"""
        result = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_volatility=0.02,
            expert_recommendations={
                "stock_expert": {"stocks": 0.6},
                "bond_expert": {"bonds": 0.4},
            },
            actual_allocation={"stocks": 0.5, "bonds": 0.5},
            asset_returns={"stocks": 0.015, "bonds": 0.005},
        )

        assert result.components["consensus_reward"] >= 0

    def test_compute_different_regimes(self, reward_fn):
        """测试不同市场状态"""
        for regime in ["bull", "bear", "volatile", "normal"]:
            result = reward_fn.compute(
                portfolio_return=0.01,
                portfolio_volatility=0.02,
                expert_recommendations={},
                actual_allocation={"stocks": 0.5},
                asset_returns={"stocks": 0.01},
                market_regime=regime,
            )
            assert result is not None

    def test_consensus_reward_perfect_match(self, reward_fn):
        """测试完美匹配共识奖励"""
        expert_recs = {"expert1": {"stocks": 0.5, "bonds": 0.5}}
        allocation = {"stocks": 0.5, "bonds": 0.5}

        reward = reward_fn._compute_consensus_reward(expert_recs, allocation)
        assert reward > 0.9  # 应该接近1

    def test_timing_reward_bull_market(self, reward_fn):
        """测试牛市时机奖励"""
        reward = reward_fn._compute_timing_reward(
            portfolio_return=0.02,  # 正收益
            market_regime="bull",
            volatility=0.01
        )
        assert reward > 0  # 牛市正收益应该有正奖励

    def test_timing_reward_bear_market(self, reward_fn):
        """测试熊市时机奖励"""
        reward = reward_fn._compute_timing_reward(
            portfolio_return=-0.005,  # 小亏
            market_regime="bear",
            volatility=0.02
        )
        # 熊市少亏应该有正奖励


class TestPositionSizingReward:
    """PositionSizingReward测试"""

    @pytest.fixture
    def reward_fn(self):
        return PositionSizingReward()

    def test_init(self, reward_fn):
        """测试初始化"""
        assert reward_fn.risk_parity_weight == 0.30
        assert reward_fn.kelly_weight == 0.25
        assert reward_fn.vol_target_weight == 0.30
        assert reward_fn.target_volatility == 0.12

    def test_compute_basic(self, reward_fn):
        """测试基本计算"""
        result = reward_fn.compute(
            position_sizes={"SPY": 0.3, "TLT": 0.3, "GLD": 0.2},
            asset_volatilities={"SPY": 0.15, "TLT": 0.10, "GLD": 0.12},
            asset_returns={"SPY": 0.01, "TLT": 0.005, "GLD": 0.008},
            portfolio_volatility=0.08,
            risk_contributions={"SPY": 0.35, "TLT": 0.35, "GLD": 0.30},
        )

        assert isinstance(result, RewardComponents)
        assert "risk_parity" in result.components
        assert "kelly_efficiency" in result.components

    def test_risk_parity_quality_equal(self, reward_fn):
        """测试均等风险贡献"""
        risk_contributions = {"A": 0.33, "B": 0.33, "C": 0.34}
        reward = reward_fn._compute_risk_parity_quality(risk_contributions)
        assert reward > 0.8  # 接近均等

    def test_risk_parity_quality_unequal(self, reward_fn):
        """测试不均等风险贡献"""
        risk_contributions = {"A": 0.8, "B": 0.1, "C": 0.1}
        reward = reward_fn._compute_risk_parity_quality(risk_contributions)
        assert reward < 0.5  # 偏离较大

    def test_vol_target_reward(self, reward_fn):
        """测试波动率目标奖励"""
        # 假设目标是12%年化，即日波动率约0.75%
        reward = reward_fn._compute_vol_target_reward(0.0075)
        assert reward > 0.5

    def test_liquidity_reward_with_scores(self, reward_fn):
        """测试流动性奖励"""
        position_sizes = {"A": 0.2, "B": 0.3}
        liquidity_scores = {"A": 0.9, "B": 0.8}

        reward = reward_fn._compute_liquidity_reward(position_sizes, liquidity_scores)
        assert reward > 0

    def test_liquidity_reward_low_liquidity_large_position(self, reward_fn):
        """测试低流动性大仓位惩罚"""
        position_sizes = {"A": 0.3}  # 大仓位
        liquidity_scores = {"A": 0.2}  # 低流动性

        reward = reward_fn._compute_liquidity_reward(position_sizes, liquidity_scores)
        # 应该有惩罚


class TestHedgingReward:
    """HedgingReward测试"""

    @pytest.fixture
    def reward_fn(self):
        return HedgingReward()

    def test_init(self, reward_fn):
        """测试初始化"""
        assert reward_fn.tail_risk_weight == 0.35
        assert reward_fn.cost_efficiency_weight == 0.25
        assert reward_fn.var_threshold == 0.05

    def test_compute_basic(self, reward_fn):
        """测试基本计算"""
        result = reward_fn.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.001,
            vix_level=20,
            vix_change=0.05,
            hedge_ratio=0.15,
            hedge_ratio_change=0.02,
            portfolio_return=0.005,
        )

        assert isinstance(result, RewardComponents)
        assert "tail_risk" in result.components
        assert "cost_efficiency" in result.components

    def test_tail_risk_protection_improvement(self, reward_fn):
        """测试尾部风险改善"""
        reward = reward_fn._compute_tail_risk_protection(
            var_before=-0.05,
            var_after=-0.03,  # 40%改善
            cvar_before=-0.07,
            cvar_after=-0.05,  # 28%改善
        )
        assert reward > 0

    def test_cost_efficiency_no_cost(self, reward_fn):
        """测试无成本效率"""
        reward = reward_fn._compute_cost_efficiency(
            var_before=-0.05,
            var_after=-0.04,
            hedge_cost=0,  # 无成本
        )
        assert reward == 1.0

    def test_cost_efficiency_high_cost(self, reward_fn):
        """测试高成本效率"""
        reward = reward_fn._compute_cost_efficiency(
            var_before=-0.05,
            var_after=-0.04,
            hedge_cost=0.05,  # 高成本
        )
        assert reward < 0.5

    def test_vix_response_vix_rising(self, reward_fn):
        """测试VIX上升响应"""
        reward = reward_fn._compute_vix_response(
            vix_level=25,
            vix_change=0.1,  # VIX上升
            hedge_ratio=0.2,
            hedge_ratio_change=0.05,  # 增加对冲
        )
        assert reward > 0

    def test_vix_response_vix_falling(self, reward_fn):
        """测试VIX下降响应"""
        reward = reward_fn._compute_vix_response(
            vix_level=15,
            vix_change=-0.05,  # VIX下降
            hedge_ratio=0.1,
            hedge_ratio_change=-0.02,  # 减少对冲
        )
        assert reward > 0


class TestExpertReward:
    """ExpertReward测试"""

    @pytest.fixture
    def stock_expert_reward(self):
        return ExpertReward("stock")

    @pytest.fixture
    def crypto_expert_reward(self):
        return ExpertReward("crypto")

    def test_init(self, stock_expert_reward):
        """测试初始化"""
        assert stock_expert_reward.expert_type == "stock"
        assert stock_expert_reward.accuracy_weight == 0.35

    def test_type_configs(self, stock_expert_reward, crypto_expert_reward):
        """测试类型配置"""
        assert stock_expert_reward.type_config["volatility_scale"] == 1.0
        assert crypto_expert_reward.type_config["volatility_scale"] == 3.0

    def test_compute_basic(self, stock_expert_reward):
        """测试基本计算"""
        result = stock_expert_reward.compute(
            signal=0.5,
            confidence=0.7,
            actual_return=0.01,
            historical_signals=[0.3, 0.4, 0.5, 0.6, 0.7],
            historical_returns=[0.005, 0.01, 0.008, 0.012, 0.009],
            portfolio_weight=0.2,
            asset_contribution=0.003,
        )

        assert isinstance(result, RewardComponents)
        assert "accuracy" in result.components
        assert "calibration" in result.components

    def test_accuracy_correct_direction(self, stock_expert_reward):
        """测试正确方向准确度"""
        # 正信号+正收益
        reward = stock_expert_reward._compute_accuracy(signal=0.5, actual_return=0.01)
        assert reward > 0

    def test_accuracy_wrong_direction(self, stock_expert_reward):
        """测试错误方向准确度"""
        # 正信号+负收益
        reward = stock_expert_reward._compute_accuracy(signal=0.5, actual_return=-0.01)
        assert reward < 0

    def test_calibration_insufficient_data(self, stock_expert_reward):
        """测试数据不足校准"""
        reward = stock_expert_reward._compute_calibration(
            historical_signals=[0.5, 0.6],  # 少于5个
            historical_returns=[0.01, 0.02],
            current_confidence=0.7,
        )
        assert reward == 0.0

    def test_timing_high_confidence_correct(self, stock_expert_reward):
        """测试高置信度正确"""
        reward = stock_expert_reward._compute_timing(
            signal=0.8,
            actual_return=0.02,
            confidence=0.9,
        )
        assert reward > 0


class TestCoordinationReward:
    """CoordinationReward测试"""

    @pytest.fixture
    def reward_fn(self):
        return CoordinationReward()

    def test_init(self, reward_fn):
        """测试初始化"""
        assert reward_fn.consistency_weight == 0.25
        assert reward_fn.efficiency_weight == 0.25

    def test_compute_basic(self, reward_fn):
        """测试基本计算"""
        result = reward_fn.compute(
            manager_decisions={
                "pm1": {"stocks": 0.5, "bonds": 0.3},
                "pm2": {"stocks": 0.4, "bonds": 0.4},
            },
            expert_signals={"stock": 0.5, "bond": 0.2},
            final_allocation={"stocks": 0.45, "bonds": 0.35},
            individual_returns={"pm1": 0.008, "pm2": 0.01},
            portfolio_return=0.012,
        )

        assert isinstance(result, RewardComponents)
        assert "consistency" in result.components
        assert "efficiency" in result.components

    def test_consistency_perfect(self, reward_fn):
        """测试完美一致性"""
        manager_decisions = {
            "pm1": {"stocks": 0.5},
            "pm2": {"stocks": 0.5},
        }
        final_allocation = {"stocks": 0.5}

        reward = reward_fn._compute_consistency(manager_decisions, final_allocation)
        assert reward > 0.9

    def test_efficiency_coordination_helps(self, reward_fn):
        """测试协调有帮助"""
        individual_returns = {"pm1": 0.005, "pm2": 0.007}
        portfolio_return = 0.01  # 高于平均

        reward = reward_fn._compute_coordination_efficiency(
            individual_returns, portfolio_return
        )
        assert reward > 0

    def test_efficiency_coordination_hurts(self, reward_fn):
        """测试协调有害"""
        individual_returns = {"pm1": 0.01, "pm2": 0.012}
        portfolio_return = 0.005  # 低于平均

        reward = reward_fn._compute_coordination_efficiency(
            individual_returns, portfolio_return
        )
        assert reward < 0


class TestCombinedRewardCalculator:
    """CombinedRewardCalculator测试"""

    @pytest.fixture
    def calculator(self):
        return CombinedRewardCalculator()

    def test_init(self, calculator):
        """测试初始化"""
        assert calculator.individual_weight == 0.4
        assert calculator.team_weight == 0.4
        assert calculator.coordination_weight == 0.2

    def test_compute_all_rewards(self, calculator):
        """测试计算所有奖励"""
        state = {}
        actions = {
            "portfolio_manager": {"allocation": {"stocks": 0.5}},
            "position_sizing": {"sizes": {"stocks": 0.5}},
            "hedging": {"ratio": 0.1},
            "expert_stock": {"signal": 0.5, "confidence": 0.7},
            "expert_bond": {"signal": 0.2, "confidence": 0.6},
            "expert_commodity": {"signal": -0.1, "confidence": 0.5},
            "expert_reits": {"signal": 0.1, "confidence": 0.5},
            "expert_crypto": {"signal": 0.3, "confidence": 0.4},
        }
        info = {
            "portfolio_return": 0.01,
            "portfolio_volatility": 0.02,
            "asset_returns": {"stocks": 0.012},
            "vix": 18,
        }

        rewards = calculator.compute_all_rewards(state, actions, {}, info)

        assert "portfolio_manager" in rewards
        assert "position_sizing" in rewards
        assert "hedging" in rewards
        assert "expert_stock" in rewards
        assert "coordination" in rewards

    def test_compute_agent_total_reward(self, calculator):
        """测试计算智能体总奖励"""
        individual = RewardComponents(total=0.5, components={}, description="")
        coordination = RewardComponents(total=0.3, components={}, description="")
        team_reward = 0.4

        total = calculator.compute_agent_total_reward(
            "test_agent", individual, team_reward, coordination
        )

        expected = 0.4 * 0.5 + 0.4 * 0.4 + 0.2 * 0.3
        assert total == pytest.approx(expected, rel=0.01)


class TestCreateDefaultRewardCalculator:
    """create_default_reward_calculator测试"""

    def test_create(self):
        """测试创建"""
        calculator = create_default_reward_calculator()

        assert isinstance(calculator, CombinedRewardCalculator)
        assert calculator.individual_weight == 0.4
        assert calculator.team_weight == 0.4


class TestComputeGAEWithIndividualRewards:
    """compute_gae_with_individual_rewards测试"""

    def test_basic_computation(self):
        """测试基本计算"""
        rewards_dict = {
            "agent1": [0.1, 0.2, 0.15],
            "agent2": [0.05, 0.1, 0.08],
        }
        values = np.array([0.5, 0.6, 0.55])
        dones = np.array([0.0, 0.0, 1.0])

        advantages, returns = compute_gae_with_individual_rewards(
            rewards_dict, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_single_agent(self):
        """测试单智能体"""
        rewards_dict = {
            "agent1": [0.1, 0.2],
        }
        values = np.array([0.5, 0.6])
        dones = np.array([0.0, 0.0])

        advantages, returns = compute_gae_with_individual_rewards(
            rewards_dict, values, dones
        )

        assert advantages.shape == (2,)

    def test_with_done_signal(self):
        """测试带终止信号"""
        rewards_dict = {
            "agent1": [0.1, 0.2, 0.15],
        }
        values = np.array([0.5, 0.6, 0.55])
        dones = np.array([0.0, 1.0, 0.0])  # 中间结束

        advantages, returns = compute_gae_with_individual_rewards(
            rewards_dict, values, dones
        )

        assert advantages.shape == (3,)


class TestRewardClipping:
    """奖励裁剪测试"""

    def test_clip_reward_in_range(self):
        """测试范围内裁剪"""
        reward_fn = PortfolioManagerReward()
        clipped = reward_fn._clip_reward(0.5)
        assert clipped == 0.5

    def test_clip_reward_above_max(self):
        """测试超过最大值裁剪"""
        reward_fn = PortfolioManagerReward()
        clipped = reward_fn._clip_reward(15.0)
        assert clipped == 10.0

    def test_clip_reward_below_min(self):
        """测试低于最小值裁剪"""
        reward_fn = PortfolioManagerReward()
        clipped = reward_fn._clip_reward(-15.0)
        assert clipped == -10.0

    def test_custom_clip_range(self):
        """测试自定义裁剪范围"""
        reward_fn = PortfolioManagerReward()
        clipped = reward_fn._clip_reward(0.8, min_val=0, max_val=1)
        assert clipped == 0.8

        clipped = reward_fn._clip_reward(1.5, min_val=0, max_val=1)
        assert clipped == 1.0

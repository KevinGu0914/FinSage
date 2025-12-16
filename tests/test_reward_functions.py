#!/usr/bin/env python
"""
Reward Functions Tests - RL奖励函数模块测试
覆盖: reward_functions.py (RewardComponents, BaseAgentReward, PortfolioManagerReward,
      PositionSizingReward, HedgingReward, ExpertReward, CoordinationReward)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any


# ============================================================
# Test 1: RewardComponents
# ============================================================

class TestRewardComponents:
    """测试奖励组件数据类"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import RewardComponents
        assert RewardComponents is not None

    def test_creation(self):
        """测试创建"""
        from finsage.rl.reward_functions import RewardComponents

        reward = RewardComponents(
            total=0.75,
            components={"return": 0.5, "risk": 0.25},
            description="Test reward"
        )

        assert reward.total == 0.75
        assert reward.components["return"] == 0.5
        assert reward.description == "Test reward"

    def test_to_dict(self):
        """测试转换为字典"""
        from finsage.rl.reward_functions import RewardComponents

        reward = RewardComponents(
            total=0.8,
            components={"alpha": 0.3, "beta": 0.5},
            description="Alpha-beta reward"
        )

        d = reward.to_dict()

        assert "total" in d
        assert "components" in d
        assert "description" in d
        assert d["total"] == 0.8


# ============================================================
# Test 2: BaseAgentReward
# ============================================================

class TestBaseAgentReward:
    """测试基类奖励函数"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import BaseAgentReward
        assert BaseAgentReward is not None

    def test_is_abstract(self):
        """测试是抽象类"""
        from finsage.rl.reward_functions import BaseAgentReward

        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            BaseAgentReward()

    def test_clip_reward(self):
        """测试奖励裁剪功能"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        # 通过子类测试_clip_reward
        reward = PortfolioManagerReward()

        assert reward._clip_reward(15.0) == 10.0  # 超出上限
        assert reward._clip_reward(-15.0) == -10.0  # 超出下限
        assert reward._clip_reward(5.0) == 5.0  # 正常范围内


# ============================================================
# Test 3: PortfolioManagerReward
# ============================================================

class TestPortfolioManagerReward:
    """测试投资组合经理奖励"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import PortfolioManagerReward
        assert PortfolioManagerReward is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        assert reward.return_weight == 0.35
        assert reward.consensus_weight == 0.25
        assert reward.quality_weight == 0.25
        assert reward.timing_weight == 0.15

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        config = {
            "return_weight": 0.5,
            "consensus_weight": 0.2,
            "risk_free_rate": 0.03
        }
        reward = PortfolioManagerReward(config)

        assert reward.return_weight == 0.5
        assert reward.risk_free_rate == 0.03

    def test_compute_positive_return(self):
        """测试正收益计算"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        result = reward.compute(
            portfolio_return=0.02,
            portfolio_volatility=0.15,
            expert_recommendations={
                "stock_expert": {"SPY": 0.4, "QQQ": 0.3},
                "bond_expert": {"TLT": 0.3}
            },
            actual_allocation={"SPY": 0.4, "QQQ": 0.3, "TLT": 0.3},
            asset_returns={"SPY": 0.03, "QQQ": 0.02, "TLT": 0.01},
            market_regime="bull"
        )

        assert result.total is not None
        assert "return_reward" in result.components
        assert "consensus_reward" in result.components

    def test_compute_negative_return(self):
        """测试负收益计算"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        result = reward.compute(
            portfolio_return=-0.02,
            portfolio_volatility=0.20,
            expert_recommendations={},
            actual_allocation={"SPY": 0.5, "TLT": 0.5},
            asset_returns={"SPY": -0.03, "TLT": 0.01},
            market_regime="bear"
        )

        assert result.total is not None


# ============================================================
# Test 4: PositionSizingReward
# ============================================================

class TestPositionSizingReward:
    """测试仓位规模奖励"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import PositionSizingReward
        assert PositionSizingReward is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        assert reward.risk_parity_weight == 0.30
        assert reward.kelly_weight == 0.25
        assert reward.vol_target_weight == 0.30
        assert reward.target_volatility == 0.12

    def test_compute(self):
        """测试计算"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
            asset_volatilities={"SPY": 0.15, "TLT": 0.10, "GLD": 0.12},
            asset_returns={"SPY": 0.02, "TLT": 0.01, "GLD": 0.015},
            portfolio_volatility=0.12,
            risk_contributions={"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
            liquidity_scores={"SPY": 0.95, "TLT": 0.90, "GLD": 0.85}
        )

        assert result.total is not None
        assert "risk_parity" in result.components
        assert "kelly_efficiency" in result.components


# ============================================================
# Test 5: HedgingReward
# ============================================================

class TestHedgingReward:
    """测试对冲奖励"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import HedgingReward
        assert HedgingReward is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()
        assert hasattr(reward, 'config')

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.rl.reward_functions import HedgingReward

        config = {
            "tail_risk_weight": 0.4,
            "cost_efficiency_weight": 0.3,
            "var_threshold": 0.08
        }
        reward = HedgingReward(config)
        assert reward.tail_risk_weight == 0.4
        assert reward.cost_efficiency_weight == 0.3

    def test_compute_basic(self):
        """测试基本计算"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=-0.05,
            var_after=-0.03,
            cvar_before=-0.08,
            cvar_after=-0.06,
            hedge_cost=0.005,
            vix_level=25,
            vix_change=0.05,
            hedge_ratio=0.2,
            hedge_ratio_change=0.05,
            portfolio_return=0.01
        )

        assert result.total is not None
        assert "tail_risk" in result.components
        assert "cost_efficiency" in result.components
        assert "vix_response" in result.components

    def test_compute_high_vix(self):
        """测试高VIX环境"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=-0.10,
            var_after=-0.07,
            cvar_before=-0.15,
            cvar_after=-0.10,
            hedge_cost=0.01,
            vix_level=45,
            vix_change=0.15,
            hedge_ratio=0.4,
            hedge_ratio_change=0.1,
            portfolio_return=-0.02
        )

        assert result.total is not None

    def test_compute_no_hedge_cost(self):
        """测试无对冲成本"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0,
            vix_level=20,
            vix_change=0,
            hedge_ratio=0.1,
            hedge_ratio_change=0,
            portfolio_return=0
        )

        assert result.components["cost_efficiency"] == 1.0  # 无成本保护最佳

    def test_tail_risk_protection_var_zero(self):
        """测试VaR为零的情况"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=0,
            var_after=0,
            cvar_before=0,
            cvar_after=0,
            hedge_cost=0.001,
            vix_level=15,
            vix_change=0,
            hedge_ratio=0.05,
            hedge_ratio_change=0,
            portfolio_return=0.005
        )

        assert result.total is not None

    def test_vix_response_rising(self):
        """测试VIX上升时的响应"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        # VIX上升，增加对冲
        result = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.002,
            vix_level=30,
            vix_change=0.10,  # VIX上升
            hedge_ratio=0.25,
            hedge_ratio_change=0.05,  # 增加对冲
            portfolio_return=0
        )

        assert "vix_response" in result.components

    def test_vix_response_falling(self):
        """测试VIX下降时的响应"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.002,
            vix_level=18,
            vix_change=-0.05,  # VIX下降
            hedge_ratio=0.1,
            hedge_ratio_change=-0.02,  # 减少对冲
            portfolio_return=0.01
        )

        assert "vix_response" in result.components

    def test_dynamic_adjustment_bull_market(self):
        """测试牛市中的动态调整"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        # 牛市高收益，高对冲应该被惩罚
        result_high_hedge = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.002,
            vix_level=15,
            vix_change=0,
            hedge_ratio=0.4,  # 高对冲
            hedge_ratio_change=0,
            portfolio_return=0.02  # 高正收益
        )

        # 牛市高收益，低对冲应该更好
        result_low_hedge = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.001,
            vix_level=15,
            vix_change=0,
            hedge_ratio=0.1,  # 低对冲
            hedge_ratio_change=0,
            portfolio_return=0.02
        )

        # 两种情况都应该有有效结果
        assert result_high_hedge.total is not None
        assert result_low_hedge.total is not None

    def test_dynamic_adjustment_bear_market(self):
        """测试熊市中的动态调整"""
        from finsage.rl.reward_functions import HedgingReward

        reward = HedgingReward()

        result = reward.compute(
            var_before=-0.05,
            var_after=-0.04,
            cvar_before=-0.07,
            cvar_after=-0.06,
            hedge_cost=0.003,
            vix_level=30,
            vix_change=0.1,
            hedge_ratio=0.3,  # 适当对冲
            hedge_ratio_change=0.1,
            portfolio_return=-0.02  # 负收益
        )

        assert result.total is not None


# ============================================================
# Test 6: ExpertReward
# ============================================================

class TestExpertReward:
    """测试专家奖励"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import ExpertReward
        assert ExpertReward is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="stock")
        assert hasattr(reward, 'config')
        assert reward.expert_type == "stock"

    def test_different_expert_types(self):
        """测试不同类型专家"""
        from finsage.rl.reward_functions import ExpertReward

        for expert_type in ["stock", "bond", "commodity", "reits", "crypto"]:
            reward = ExpertReward(expert_type=expert_type)
            assert reward.expert_type == expert_type

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.rl.reward_functions import ExpertReward

        config = {"accuracy_weight": 0.4, "calibration_weight": 0.3}
        reward = ExpertReward(expert_type="stock", config=config)
        assert reward.accuracy_weight == 0.4
        assert reward.calibration_weight == 0.3

    def test_compute_stock_expert(self):
        """测试股票专家计算"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="stock")

        result = reward.compute(
            signal=0.7,
            confidence=0.8,
            actual_return=0.02,
            historical_signals=[0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.7],
            historical_returns=[0.01, 0.02, -0.01, 0.015, 0.005, 0.02, 0.01],
            portfolio_weight=0.3,
            asset_contribution=0.01
        )

        assert result.total is not None
        assert "accuracy" in result.components
        assert "calibration" in result.components
        assert "timing" in result.components
        assert "contribution" in result.components

    def test_compute_bond_expert(self):
        """测试债券专家计算"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="bond")

        result = reward.compute(
            signal=0.3,
            confidence=0.6,
            actual_return=0.005,
            historical_signals=[0.2, 0.3, 0.25, 0.2, 0.3, 0.25],
            historical_returns=[0.003, 0.004, 0.002, 0.003, 0.004, 0.003],
            portfolio_weight=0.2,
            asset_contribution=0.002
        )

        assert result.total is not None

    def test_compute_crypto_expert(self):
        """测试加密货币专家计算（高波动）"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="crypto")

        result = reward.compute(
            signal=0.8,
            confidence=0.5,  # 较低置信度
            actual_return=0.08,  # 高收益
            historical_signals=[0.5, 0.7, -0.3, 0.6, 0.8, -0.2],
            historical_returns=[0.05, 0.08, -0.05, 0.03, 0.1, -0.03],
            portfolio_weight=0.1,
            asset_contribution=0.02
        )

        assert result.total is not None

    def test_compute_negative_signal_negative_return(self):
        """测试负信号和负收益（正确预测）"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="stock")

        result = reward.compute(
            signal=-0.5,  # 卖出信号
            confidence=0.7,
            actual_return=-0.02,  # 实际下跌
            historical_signals=[0.3, -0.2, 0.4, -0.3, 0.1, -0.1],
            historical_returns=[0.01, -0.01, 0.02, -0.02, 0.005, -0.005],
            portfolio_weight=0.25,
            asset_contribution=-0.005
        )

        assert result.components["accuracy"] > 0  # 预测正确应该有正奖励

    def test_compute_wrong_prediction(self):
        """测试错误预测"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="stock")

        result = reward.compute(
            signal=0.8,  # 强买入信号
            confidence=0.9,  # 高置信度
            actual_return=-0.03,  # 实际下跌
            historical_signals=[0.5, 0.6, 0.7, 0.8, 0.5, 0.6],
            historical_returns=[0.01, 0.02, -0.01, 0.01, 0.02, 0.01],
            portfolio_weight=0.3,
            asset_contribution=-0.01
        )

        # 错误预测应该有负面影响
        assert result.total is not None

    def test_calibration_insufficient_data(self):
        """测试校准不足数据"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="stock")

        result = reward.compute(
            signal=0.5,
            confidence=0.6,
            actual_return=0.01,
            historical_signals=[0.5, 0.6],  # 只有2个历史信号
            historical_returns=[0.01, 0.02],  # 不足5个
            portfolio_weight=0.2,
            asset_contribution=0.002
        )

        assert result.components["calibration"] == 0.0  # 数据不足返回0

    def test_accuracy_different_volatility_scales(self):
        """测试不同资产类型的波动率缩放"""
        from finsage.rl.reward_functions import ExpertReward

        # 债券（低波动）
        bond_reward = ExpertReward(expert_type="bond")
        assert bond_reward.type_config["volatility_scale"] == 0.3

        # 加密货币（高波动）
        crypto_reward = ExpertReward(expert_type="crypto")
        assert crypto_reward.type_config["volatility_scale"] == 3.0

    def test_unknown_expert_type(self):
        """测试未知专家类型使用默认配置"""
        from finsage.rl.reward_functions import ExpertReward

        reward = ExpertReward(expert_type="unknown")
        assert reward.type_config["volatility_scale"] == 1.0  # 使用stock默认值


# ============================================================
# Test 7: CoordinationReward
# ============================================================

class TestCoordinationReward:
    """测试协调奖励"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import CoordinationReward
        assert CoordinationReward is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()
        assert hasattr(reward, 'config')

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.rl.reward_functions import CoordinationReward

        config = {
            "consistency_weight": 0.3,
            "info_utilization_weight": 0.3,
            "efficiency_weight": 0.2
        }
        reward = CoordinationReward(config)
        assert reward.consistency_weight == 0.3
        assert reward.info_utilization_weight == 0.3

    def test_compute_basic(self):
        """测试基本计算"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={
                "portfolio_manager": {"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
                "position_sizing": {"SPY": 0.35, "TLT": 0.35, "GLD": 0.30}
            },
            expert_signals={
                "stock_expert": 0.6,
                "bond_expert": 0.3,
                "commodity_expert": 0.1
            },
            final_allocation={"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
            individual_returns={
                "portfolio_manager": 0.008,
                "position_sizing": 0.007
            },
            portfolio_return=0.01,
            discussion_rounds=1
        )

        assert result.total is not None
        assert "consistency" in result.components
        assert "info_utilization" in result.components
        assert "conflict_resolution" in result.components
        assert "efficiency" in result.components

    def test_compute_high_disagreement(self):
        """测试高分歧情况"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={
                "manager1": {"SPY": 0.6, "TLT": 0.2, "GLD": 0.2},
                "manager2": {"SPY": 0.2, "TLT": 0.6, "GLD": 0.2},  # 分歧很大
                "manager3": {"SPY": 0.3, "TLT": 0.3, "GLD": 0.4}
            },
            expert_signals={
                "stock_expert": 0.5,
                "bond_expert": 0.5
            },
            final_allocation={"SPY": 0.35, "TLT": 0.35, "GLD": 0.3},
            individual_returns={
                "manager1": 0.012,
                "manager2": 0.006,
                "manager3": 0.009
            },
            portfolio_return=0.01,
            discussion_rounds=3  # 多轮讨论
        )

        assert result.total is not None

    def test_compute_empty_manager_decisions(self):
        """测试空经理决策"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={},
            expert_signals={},
            final_allocation={"SPY": 0.5, "TLT": 0.5},
            individual_returns={},
            portfolio_return=0.005
        )

        assert result.components["consistency"] == 0.5  # 无经理决策返回0.5

    def test_compute_empty_expert_signals(self):
        """测试空专家信号"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={"pm": {"SPY": 0.5}},
            expert_signals={},
            final_allocation={"SPY": 0.5, "TLT": 0.5},
            individual_returns={"pm": 0.01},
            portfolio_return=0.01
        )

        assert result.components["info_utilization"] == 0.5

    def test_compute_single_manager(self):
        """测试单一经理"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={"pm": {"SPY": 0.5, "TLT": 0.5}},
            expert_signals={"stock": 0.5},
            final_allocation={"SPY": 0.5, "TLT": 0.5},
            individual_returns={"pm": 0.01},
            portfolio_return=0.01,
            discussion_rounds=1
        )

        assert result.components["conflict_resolution"] == 0.5  # 单经理无冲突

    def test_coordination_efficiency_positive(self):
        """测试协调效率为正（协作优于独立）"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={
                "pm1": {"SPY": 0.4, "TLT": 0.4, "GLD": 0.2},
                "pm2": {"SPY": 0.35, "TLT": 0.45, "GLD": 0.2}
            },
            expert_signals={},
            final_allocation={"SPY": 0.4, "TLT": 0.4, "GLD": 0.2},
            individual_returns={"pm1": 0.005, "pm2": 0.006},  # 平均 0.0055
            portfolio_return=0.01,  # 高于平均
            discussion_rounds=1
        )

        # 协作收益高于个体平均，效率应该为正
        assert result.total is not None

    def test_coordination_efficiency_negative(self):
        """测试协调效率为负（协作不如独立）"""
        from finsage.rl.reward_functions import CoordinationReward

        reward = CoordinationReward()

        result = reward.compute(
            manager_decisions={
                "pm1": {"SPY": 0.4},
                "pm2": {"SPY": 0.35}
            },
            expert_signals={},
            final_allocation={"SPY": 0.4},
            individual_returns={"pm1": 0.02, "pm2": 0.015},  # 平均 0.0175
            portfolio_return=0.01,  # 低于平均
            discussion_rounds=1
        )

        assert result.total is not None


# ============================================================
# Test 8: CombinedRewardCalculator
# ============================================================

class TestCombinedRewardCalculator:
    """测试组合奖励计算器"""

    def test_import(self):
        """测试导入"""
        from finsage.rl.reward_functions import CombinedRewardCalculator
        assert CombinedRewardCalculator is not None

    def test_initialization(self):
        """测试初始化"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()
        assert calculator is not None
        assert calculator.individual_weight == 0.4
        assert calculator.team_weight == 0.4
        assert calculator.coordination_weight == 0.2

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        config = {
            "individual_weight": 0.5,
            "team_weight": 0.3,
            "coordination_weight": 0.2,
            "portfolio_manager": {"return_weight": 0.5}
        }
        calculator = CombinedRewardCalculator(config)
        assert calculator.individual_weight == 0.5
        assert calculator.team_weight == 0.3

    def test_compute_all_rewards(self):
        """测试计算所有奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {
            "portfolio_manager": {"allocation": {"stock": 0.4, "bond": 0.3, "commodity": 0.3}},
            "position_sizing": {"sizes": {"stock": 0.35, "bond": 0.35, "commodity": 0.30}},
            "hedging": {"ratio": 0.15},
            "expert_stock": {"signal": 0.5, "confidence": 0.7},
            "expert_bond": {"signal": 0.2, "confidence": 0.6},
            "expert_commodity": {"signal": -0.3, "confidence": 0.5},
            "expert_reits": {"signal": 0.1, "confidence": 0.5},
            "expert_crypto": {"signal": 0.4, "confidence": 0.4},
        }
        next_state = {}
        info = {
            "portfolio_return": 0.005,
            "portfolio_volatility": 0.015,
            "asset_returns": {"stock": 0.008, "bond": 0.002, "commodity": -0.003},
            "expert_recommendations": {"stock_expert": {"stock": 0.45}},
            "market_regime": "normal",
            "vix": 18,
            "var_before": -0.05,
            "var_after": -0.04,
            "cvar_before": -0.07,
            "cvar_after": -0.055,
            "hedge_cost": 0.001,
            "final_allocation": {"stock": 0.4, "bond": 0.3, "commodity": 0.3},
            "asset_volatilities": {"stock": 0.15, "bond": 0.08, "commodity": 0.12},
            "risk_contributions": {"stock": 0.4, "bond": 0.3, "commodity": 0.3},
        }

        rewards = calculator.compute_all_rewards(state, actions, next_state, info)

        assert "portfolio_manager" in rewards
        assert "position_sizing" in rewards
        assert "hedging" in rewards
        assert "expert_stock" in rewards
        assert "coordination" in rewards

    def test_compute_agent_total_reward(self):
        """测试计算单个智能体总奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator, RewardComponents

        calculator = CombinedRewardCalculator()

        individual_reward = RewardComponents(
            total=0.5,
            components={"test": 0.5},
            description="test"
        )
        team_reward = 0.3
        coordination_reward = RewardComponents(
            total=0.4,
            components={"coord": 0.4},
            description="coord"
        )

        total = calculator.compute_agent_total_reward(
            "test_agent",
            individual_reward,
            team_reward,
            coordination_reward
        )

        # 0.4 * 0.5 + 0.4 * 0.3 + 0.2 * 0.4 = 0.2 + 0.12 + 0.08 = 0.4
        assert abs(total - 0.4) < 0.01

    def test_compute_pm_reward(self):
        """测试计算投资组合经理奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {
            "portfolio_manager": {"allocation": {"SPY": 0.6, "TLT": 0.4}}
        }
        info = {
            "portfolio_return": 0.01,
            "portfolio_volatility": 0.02,
            "expert_recommendations": {},
            "asset_returns": {"SPY": 0.012, "TLT": 0.005},
            "market_regime": "bull"
        }

        reward = calculator._compute_pm_reward(state, actions, info)

        assert reward.total is not None
        assert "return_reward" in reward.components

    def test_compute_ps_reward(self):
        """测试计算仓位规模奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {
            "position_sizing": {"sizes": {"SPY": 0.5, "TLT": 0.5}}
        }
        info = {
            "asset_volatilities": {"SPY": 0.15, "TLT": 0.08},
            "asset_returns": {"SPY": 0.01, "TLT": 0.005},
            "portfolio_volatility": 0.10,
            "risk_contributions": {"SPY": 0.6, "TLT": 0.4}
        }

        reward = calculator._compute_ps_reward(state, actions, info)

        assert reward.total is not None

    def test_compute_hedge_reward(self):
        """测试计算对冲奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {"hedging": {"ratio": 0.2}}
        info = {
            "var_before": -0.05,
            "var_after": -0.04,
            "cvar_before": -0.07,
            "cvar_after": -0.06,
            "hedge_cost": 0.002,
            "vix": 22,
            "vix_change": 0.05,
            "hedge_ratio_change": 0.03,
            "portfolio_return": 0.005
        }

        reward = calculator._compute_hedge_reward(state, actions, info)

        assert reward.total is not None

    def test_compute_expert_reward(self):
        """测试计算专家奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {"expert_stock": {"signal": 0.6, "confidence": 0.7}}
        info = {
            "asset_returns": {"stock": 0.015},
            "stock_historical_signals": [0.5, 0.6, 0.4, 0.7, 0.5, 0.6],
            "stock_historical_returns": [0.01, 0.02, -0.01, 0.015, 0.005, 0.02],
            "portfolio_weights": {"stock": 0.4},
            "asset_contributions": {"stock": 0.006}
        }

        reward = calculator._compute_expert_reward(
            "stock",
            calculator.expert_rewards["stock"],
            state, actions, info
        )

        assert reward.total is not None

    def test_compute_coordination_reward(self):
        """测试计算协调奖励"""
        from finsage.rl.reward_functions import CombinedRewardCalculator

        calculator = CombinedRewardCalculator()

        state = {}
        actions = {
            "portfolio_manager": {"allocation": {"SPY": 0.5}},
            "position_sizing": {"sizes": {"SPY": 0.5}},
            "expert_stock": {"signal": 0.5},
            "expert_bond": {"signal": 0.3},
            "expert_commodity": {"signal": 0.2},
            "expert_reits": {"signal": 0.1},
            "expert_crypto": {"signal": 0.4}
        }
        info = {
            "final_allocation": {"SPY": 0.5},
            "individual_returns": {"pm": 0.01},
            "portfolio_return": 0.012,
            "discussion_rounds": 2
        }

        reward = calculator._compute_coordination_reward(state, actions, info)

        assert reward.total is not None


# ============================================================
# Test 9: Utility Functions
# ============================================================

class TestUtilityFunctions:
    """测试工具函数"""

    def test_create_default_reward_calculator(self):
        """测试创建默认奖励计算器"""
        from finsage.rl.reward_functions import create_default_reward_calculator

        calculator = create_default_reward_calculator()

        assert calculator is not None
        assert calculator.individual_weight == 0.4
        assert calculator.pm_reward is not None
        assert calculator.ps_reward is not None
        assert calculator.hedge_reward is not None

    def test_compute_gae_with_individual_rewards(self):
        """测试 GAE 计算"""
        from finsage.rl.reward_functions import compute_gae_with_individual_rewards

        rewards_dict = {
            "agent1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "agent2": [0.2, 0.3, 0.4, 0.5, 0.6]
        }
        values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        dones = np.array([0, 0, 0, 0, 1])

        advantages, returns = compute_gae_with_individual_rewards(
            rewards_dict, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 5
        assert len(returns) == 5

    def test_compute_gae_terminal_state(self):
        """测试终止状态的 GAE"""
        from finsage.rl.reward_functions import compute_gae_with_individual_rewards

        rewards_dict = {
            "agent1": [0.1, 0.2, 0.3],
        }
        values = np.array([0.5, 0.6, 0.7])
        dones = np.array([0, 0, 1])  # 最后一步终止

        advantages, returns = compute_gae_with_individual_rewards(
            rewards_dict, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 3
        # 终止状态的 next_value 应该是 0
        assert advantages[-1] is not None


# ============================================================
# Test 10: PositionSizingReward Additional Tests
# ============================================================

class TestPositionSizingRewardAdditional:
    """测试仓位规模奖励的额外测试"""

    def test_compute_empty_positions(self):
        """测试空仓位"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={},
            asset_volatilities={},
            asset_returns={},
            portfolio_volatility=0.01,
            risk_contributions={}
        )

        assert result.total is not None

    def test_risk_parity_single_asset(self):
        """测试单一资产风险平价"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={"SPY": 1.0},
            asset_volatilities={"SPY": 0.15},
            asset_returns={"SPY": 0.01},
            portfolio_volatility=0.15,
            risk_contributions={"SPY": 1.0}  # 单一资产贡献100%
        )

        assert result.components["risk_parity"] == 0.0  # 单一资产返回0

    def test_kelly_efficiency_zero_volatility(self):
        """测试零波动率的Kelly效率"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={"SPY": 0.5, "TLT": 0.5},
            asset_volatilities={"SPY": 0, "TLT": 0.08},  # SPY波动率为0
            asset_returns={"SPY": 0.01, "TLT": 0.005},
            portfolio_volatility=0.04,
            risk_contributions={"SPY": 0.5, "TLT": 0.5}
        )

        assert result.total is not None

    def test_liquidity_no_data(self):
        """测试无流动性数据"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={"SPY": 0.5, "TLT": 0.5},
            asset_volatilities={"SPY": 0.15, "TLT": 0.08},
            asset_returns={"SPY": 0.01, "TLT": 0.005},
            portfolio_volatility=0.10,
            risk_contributions={"SPY": 0.6, "TLT": 0.4},
            liquidity_scores=None  # 无流动性数据
        )

        assert result.components["liquidity"] == 0.5  # 无数据返回0.5

    def test_liquidity_low_liquidity_high_position(self):
        """测试低流动性高仓位"""
        from finsage.rl.reward_functions import PositionSizingReward

        reward = PositionSizingReward()

        result = reward.compute(
            position_sizes={"SPY": 0.3, "ILLIQ": 0.5},  # 非流动资产高仓位
            asset_volatilities={"SPY": 0.15, "ILLIQ": 0.25},
            asset_returns={"SPY": 0.01, "ILLIQ": 0.02},
            portfolio_volatility=0.18,
            risk_contributions={"SPY": 0.4, "ILLIQ": 0.6},
            liquidity_scores={"SPY": 0.95, "ILLIQ": 0.3}  # 低流动性
        )

        # 低流动性高仓位应该被惩罚
        assert result.total is not None


# ============================================================
# Test 11: PortfolioManagerReward Additional Tests
# ============================================================

class TestPortfolioManagerRewardAdditional:
    """测试投资组合经理奖励的额外测试"""

    def test_zero_volatility(self):
        """测试零波动率"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        result = reward.compute(
            portfolio_return=0.01,
            portfolio_volatility=0,  # 零波动率
            expert_recommendations={},
            actual_allocation={"SPY": 1.0},
            asset_returns={"SPY": 0.01},
            market_regime="normal"
        )

        assert result.total is not None

    def test_volatile_market_regime(self):
        """测试高波动市场状态"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        result = reward.compute(
            portfolio_return=0,
            portfolio_volatility=0.015,  # 低波动 (<2%)
            expert_recommendations={},
            actual_allocation={"SPY": 0.5, "TLT": 0.5},
            asset_returns={"SPY": 0.01, "TLT": -0.01},
            market_regime="volatile"  # 高波动市场
        )

        # 在高波动市场控制波动应该得到奖励
        assert result.total is not None

    def test_empty_allocations(self):
        """测试空配置"""
        from finsage.rl.reward_functions import PortfolioManagerReward

        reward = PortfolioManagerReward()

        result = reward.compute(
            portfolio_return=0,
            portfolio_volatility=0.01,
            expert_recommendations={},
            actual_allocation={},
            asset_returns={},
            market_regime="normal"
        )

        assert result.total is not None


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Reward Functions Tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()

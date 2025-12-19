#!/usr/bin/env python
"""
MARFT Integration Deep Testing
深度测试MARFT-FinSage整合模块的所有组件

目标: 提高代码覆盖率从23%到90%+
覆盖:
- FlexMGState dataclass
- FinSageFlexMGEnv class
- FinSageActionBuffer class
- FinSageAPPOTrainer class
- MARFTFinSageIntegration class
- FinSageRewardFunction class
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json


# ============================================================
# Test FlexMGState Dataclass
# ============================================================

class TestFlexMGState:
    """测试FlexMGState数据类"""

    def test_flexmg_state_creation(self):
        """测试状态对象创建"""
        from finsage.rl.marft_integration import FlexMGState

        state = FlexMGState(
            market_data={"SPY": {"close": 450, "volume": 10000}},
            news_data=[{"headline": "Market up"}],
            technical_indicators={"rsi": 55},
            portfolio_state={"value": 1000000},
            predecessor_actions=[{"action": "BUY"}],
            timestamp="2024-01-15"
        )

        assert state.market_data["SPY"]["close"] == 450
        assert len(state.news_data) == 1
        assert state.timestamp == "2024-01-15"

    def test_flexmg_state_empty_fields(self):
        """测试空字段处理"""
        from finsage.rl.marft_integration import FlexMGState

        state = FlexMGState(
            market_data={},
            news_data=[],
            technical_indicators={},
            portfolio_state={},
            predecessor_actions=[],
            timestamp=""
        )

        assert state.market_data == {}
        assert state.news_data == []
        assert state.predecessor_actions == []


# ============================================================
# Test FinSageFlexMGEnv
# ============================================================

class TestFinSageFlexMGEnv:
    """测试FinSageFlexMGEnv环境包装器"""

    @pytest.fixture
    def mock_base_env(self):
        """创建模拟的基础环境"""
        env = Mock()
        env.reset.return_value = ({"market": "data"}, {"info": "test"})
        env.step.return_value = (
            {"market": "next_data"},
            0.05,  # reward
            False,  # terminated
            False,  # truncated
            {"portfolio": {"value": 1000000}}
        )
        return env

    @pytest.fixture
    def agent_profiles(self):
        """创建测试用的agent profiles"""
        return [
            {
                "role": "Stock_Expert",
                "expertise": "Stock analysis",
                "asset_class": "stocks",
                "dependencies": [],
            },
            {
                "role": "Bond_Expert",
                "expertise": "Bond analysis",
                "asset_class": "bonds",
                "dependencies": ["Stock_Expert"],
            },
            {
                "role": "Commodity_Expert",
                "expertise": "Commodity analysis",
                "asset_class": "commodities",
                "dependencies": ["Stock_Expert", "Bond_Expert"],
            },
        ]

    def test_env_initialization(self, mock_base_env, agent_profiles):
        """测试环境初始化"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(
            base_env=mock_base_env,
            agent_profiles=agent_profiles,
            gamma=0.99
        )

        assert env.num_agents == 3
        assert env.gamma == 0.99
        assert len(env.dependency_graph) == 3

    def test_dependency_graph_construction(self, mock_base_env, agent_profiles):
        """测试依赖图构建"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)

        # Stock Expert (index 0) has no dependencies
        assert env.dependency_graph[0] == []

        # Bond Expert (index 1) depends on Stock Expert
        assert env.dependency_graph[1] == [0]

        # Commodity Expert (index 2) depends on both
        assert 0 in env.dependency_graph[2]
        assert 1 in env.dependency_graph[2]

    def test_dependency_graph_missing_dependencies(self, mock_base_env):
        """测试缺失依赖的处理"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        profiles = [
            {
                "role": "Agent1",
                "expertise": "Test",
                "asset_class": "test",
                "dependencies": ["NonExistentAgent"],  # 不存在的依赖
            }
        ]

        env = FinSageFlexMGEnv(mock_base_env, profiles)
        # 不存在的依赖应该被忽略
        assert env.dependency_graph[0] == []

    def test_get_agent_observation_no_deps(self, mock_base_env, agent_profiles):
        """测试无依赖的agent观察生成"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)
        base_obs = {"market": "data", "prices": {"SPY": 450}}

        obs = env.get_agent_observation(
            agent_idx=0,
            base_obs=base_obs,
            predecessor_actions=[]
        )

        assert "当前市场状态" in obs
        assert "Stock_Expert" not in obs  # No predecessor info
        assert "stocks" in obs

    def test_get_agent_observation_with_deps(self, mock_base_env, agent_profiles):
        """测试有依赖的agent观察生成"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)
        base_obs = {"market": "data"}
        predecessor_actions = [
            {"action": "BUY_50%", "confidence": 0.8},
            {"action": "HOLD", "confidence": 0.6},
        ]

        # Commodity Expert (index 2) depends on both predecessors
        obs = env.get_agent_observation(
            agent_idx=2,
            base_obs=base_obs,
            predecessor_actions=predecessor_actions
        )

        assert "当前市场状态" in obs
        assert "其他专家的建议" in obs
        assert "Stock_Expert" in obs
        assert "Bond_Expert" in obs
        assert "BUY_50%" in obs

    def test_format_market_data(self, mock_base_env, agent_profiles):
        """测试市场数据格式化"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)

        obs = {
            "prices": {"SPY": 450, "QQQ": 380},
            "volumes": {"SPY": 10000},
        }

        formatted = env._format_market_data(obs)
        assert "SPY" in formatted
        assert "450" in formatted

    def test_reset(self, mock_base_env, agent_profiles):
        """测试环境重置"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)
        obs, info = env.reset()

        assert obs == {"market": "data"}
        assert info == {"info": "test"}
        mock_base_env.reset.assert_called_once()

    def test_step(self, mock_base_env, agent_profiles):
        """测试环境step"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)
        joint_action = [
            {"action": "BUY_50%"},
            {"action": "HOLD"},
            {"action": "SELL_25%"},
        ]

        obs, reward, terminated, truncated, info = env.step(joint_action)

        assert obs == {"market": "next_data"}
        assert reward == 0.05
        assert terminated is False
        assert truncated is False
        mock_base_env.step.assert_called_once()

    def test_aggregate_expert_actions(self, mock_base_env, agent_profiles):
        """测试专家动作聚合"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv

        env = FinSageFlexMGEnv(mock_base_env, agent_profiles)
        expert_actions = [
            {"action": "BUY_50%", "confidence": 0.8},
            {"action": "HOLD", "confidence": 0.6},
            {"action": "SELL_25%", "confidence": 0.7},
        ]

        aggregated = env._aggregate_expert_actions(expert_actions)

        # 验证所有资产类别都被映射
        assert "stocks" in aggregated
        assert "bonds" in aggregated
        assert "commodities" in aggregated
        assert aggregated["stocks"]["action"] == "BUY_50%"


# ============================================================
# Test FinSageActionBuffer
# ============================================================

class TestFinSageActionBuffer:
    """测试FinSageActionBuffer轨迹缓冲"""

    def test_buffer_initialization(self):
        """测试buffer初始化"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=9,  # 5 Asset Experts + 4 Meta-Level Agents
            gamma=0.99,
            gae_lambda=0.95
        )

        assert buffer.episode_length == 100
        assert buffer.num_agents == 9  # 5 Asset Experts + 4 Meta-Level Agents
        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95
        assert buffer.step == 0

    def test_insert_single_step(self):
        """测试插入单步数据"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=3
        )

        buffer.insert(
            obs=["obs1", "obs2", "obs3"],
            actions=[{"a": 1}, {"a": 2}, {"a": 3}],
            log_probs=[0.1, 0.2, 0.3],
            reward=0.05,
            value=[0.5, 0.6, 0.7],
            done=False
        )

        assert buffer.step == 1
        assert len(buffer.observations) == 1
        assert len(buffer.actions) == 1
        assert buffer.rewards[0] == 0.05

    def test_insert_multiple_steps(self):
        """测试插入多步数据"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=2
        )

        for i in range(10):
            buffer.insert(
                obs=[f"obs1_{i}", f"obs2_{i}"],
                actions=[{"step": i}] * 2,
                log_probs=[float(i), float(i)],
                reward=0.01 * i,
                value=[0.0, 0.0],
                done=False
            )

        assert buffer.step == 10
        assert len(buffer.rewards) == 10

    def test_compute_gae_simple(self):
        """测试GAE计算 - 简单情况"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=5,
            num_agents=2,
            gamma=0.99,
            gae_lambda=0.95
        )

        # 插入5步数据，reward递增
        for i in range(5):
            buffer.insert(
                obs=["obs1", "obs2"],
                actions=[{"a": 1}] * 2,
                log_probs=[0.0, 0.0],
                reward=float(i),
                value=[float(i), float(i)],
                done=False
            )

        next_value = [5.0, 5.0]
        advantages, returns = buffer.compute_gae_and_returns(next_value)

        # 验证形状
        assert advantages.shape == (5, 2)
        assert returns.shape == (5, 2)

        # 验证returns = advantages + values
        values_array = np.array(buffer.values)
        np.testing.assert_array_almost_equal(
            returns, advantages + values_array
        )

    def test_compute_gae_with_terminal(self):
        """测试GAE计算 - 有终止状态"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=3,
            num_agents=2,
            gamma=0.99,
            gae_lambda=0.95
        )

        # 前两步正常，第三步终止
        for i in range(2):
            buffer.insert(
                obs=["obs1", "obs2"],
                actions=[{"a": 1}] * 2,
                log_probs=[0.0, 0.0],
                reward=1.0,
                value=[0.5, 0.5],
                done=False
            )

        # 终止步
        buffer.insert(
            obs=["obs1", "obs2"],
            actions=[{"a": 1}] * 2,
            log_probs=[0.0, 0.0],
            reward=1.0,
            value=[0.5, 0.5],
            done=True
        )

        next_value = [0.0, 0.0]  # Terminal state, next value is 0
        advantages, returns = buffer.compute_gae_and_returns(next_value)

        assert advantages.shape == (3, 2)
        assert returns.shape == (3, 2)

    def test_get_batch(self):
        """测试获取batch"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=3
        )

        for i in range(5):
            buffer.insert(
                obs=[f"o{i}"] * 3,
                actions=[{"a": i}] * 3,
                log_probs=[float(i)] * 3,
                reward=float(i),
                value=[0.0] * 3,
                done=False
            )

        batch = buffer.get_batch()

        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert len(batch["observations"]) == 5
        assert len(batch["rewards"]) == 5

    def test_clear(self):
        """测试清空buffer"""
        from finsage.rl.marft_integration import FinSageActionBuffer

        buffer = FinSageActionBuffer(
            episode_length=100,
            num_agents=2
        )

        # 插入数据
        buffer.insert(
            obs=["obs1", "obs2"],
            actions=[{"a": 1}] * 2,
            log_probs=[0.0, 0.0],
            reward=1.0,
            value=[0.0, 0.0],
            done=False
        )

        assert buffer.step == 1

        # 清空
        buffer.clear()

        assert buffer.step == 0
        assert len(buffer.observations) == 0
        assert len(buffer.rewards) == 0


# ============================================================
# Test FinSageAPPOTrainer
# ============================================================

class TestFinSageAPPOTrainer:
    """测试FinSageAPPOTrainer训练器"""

    @pytest.fixture
    def mock_expert(self):
        """创建模拟expert"""
        expert = Mock()
        expert.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        expert.get_action_log_prob.return_value = (
            torch.tensor(-0.5, requires_grad=True),
            torch.tensor(0.1, requires_grad=True)
        )
        expert.model = Mock()
        expert.model.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        return expert

    @pytest.fixture
    def mock_critic(self):
        """创建模拟critic"""
        critic = MagicMock()
        critic.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        critic.get_value.return_value = torch.tensor([0.5, 0.6])
        return critic

    @pytest.fixture
    def config(self):
        """创建训练配置"""
        return {
            "clip_param": 0.2,
            "ppo_epoch": 2,
            "num_mini_batch": 2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "kl_threshold": 0.01,
            "lr": 5e-7,
            "critic_lr": 5e-4,
            "agent_iteration_interval": 0,
        }

    def test_trainer_initialization(self, mock_expert, mock_critic, config):
        """测试训练器初始化"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        experts = {"expert1": mock_expert}

        trainer = FinSageAPPOTrainer(
            experts=experts,
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        assert trainer.clip_param == 0.2
        assert trainer.ppo_epoch == 2
        assert len(trainer.expert_optimizers) == 1
        assert trainer.critic_optimizer is not None

    def test_trainer_initialization_no_critic(self, mock_expert, config):
        """测试无critic的初始化"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=None,
            config=config,
            device="cpu"
        )

        assert trainer.critic is None
        assert trainer.critic_optimizer is None

    def test_trainer_initialization_expert_without_parameters(self, config):
        """测试没有parameters的expert"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        expert = Mock()
        expert.parameters = None  # No parameters method

        trainer = FinSageAPPOTrainer(
            experts={"expert1": expert},
            critic=None,
            config=config,
            device="cpu"
        )

        # 应该跳过没有parameters的expert
        assert "expert1" not in trainer.expert_optimizers

    def test_train_with_empty_buffer(self, mock_expert, mock_critic, config):
        """测试空buffer训练"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer, FinSageActionBuffer

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        buffer = FinSageActionBuffer(
            episode_length=10,
            num_agents=1
        )

        next_value = [0.0]
        stats = trainer.train(buffer, next_value, global_step=0)

        # 空buffer应该返回零统计
        assert stats["policy_loss"] == 0.0
        assert stats["value_loss"] == 0.0

    def test_train_with_data_no_log_prob_method(self, mock_critic, config):
        """测试expert没有get_action_log_prob方法"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer, FinSageActionBuffer

        expert = Mock()
        expert.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        # No get_action_log_prob method

        trainer = FinSageAPPOTrainer(
            experts={"expert1": expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        buffer = FinSageActionBuffer(episode_length=5, num_agents=1)
        for _ in range(3):
            buffer.insert(
                obs=["obs"],
                actions=[{"a": 1}],
                log_probs=[0.0],
                reward=0.1,
                value=[0.5],
                done=False
            )

        stats = trainer.train(buffer, [0.0], global_step=0)
        # 应该跳过没有方法的expert
        assert isinstance(stats, dict)

    def test_train_basic(self, mock_expert, mock_critic, config):
        """测试基本训练流程"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer, FinSageActionBuffer

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        buffer = FinSageActionBuffer(episode_length=10, num_agents=1)

        # 插入一些数据
        action_tokens_buffer = []
        for i in range(4):
            buffer.insert(
                obs=[f"observation_{i}"],
                actions=[{"action": "BUY"}],
                log_probs=[-0.5],
                reward=0.1,
                value=[0.5],
                done=False
            )
            action_tokens_buffer.append([torch.tensor([1, 2, 3])])

        stats = trainer.train(
            buffer,
            next_value=[0.0],
            global_step=0,
            action_tokens_buffer=action_tokens_buffer
        )

        assert "policy_loss" in stats
        assert "entropy" in stats
        assert "kl_divergence" in stats

    def test_train_agent_iteration(self, mock_expert, config):
        """测试agent轮流训练"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer, FinSageActionBuffer

        config["agent_iteration_interval"] = 10

        expert1 = Mock()
        expert1.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        expert1.get_action_log_prob.return_value = (
            torch.tensor(-0.5, requires_grad=True),
            torch.tensor(0.1, requires_grad=True)
        )
        expert1.model = Mock()
        expert1.model.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]

        expert2 = Mock()
        expert2.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]

        trainer = FinSageAPPOTrainer(
            experts={"expert1": expert1, "expert2": expert2},
            critic=None,
            config=config,
            device="cpu"
        )

        buffer = FinSageActionBuffer(episode_length=10, num_agents=2)
        buffer.insert(
            obs=["obs1", "obs2"],
            actions=[{"a": 1}, {"a": 2}],
            log_probs=[0.0, 0.0],
            reward=0.1,
            value=[0.5, 0.5],
            done=False
        )

        # global_step=0应该训练expert1 (0 // 10 % 2 = 0)
        stats = trainer.train(buffer, [0.0, 0.0], global_step=0)
        assert isinstance(stats, dict)

    def test_save_checkpoint(self, mock_expert, mock_critic, config):
        """测试保存checkpoint"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        mock_expert.save = Mock()
        mock_expert.model.save_pretrained = Mock()

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, step=100)

            # 验证checkpoint文件存在
            assert os.path.exists(os.path.join(tmpdir, "checkpoint.json"))
            assert os.path.exists(os.path.join(tmpdir, "critic.pt"))
            assert os.path.exists(os.path.join(tmpdir, "expert_optimizers.pt"))

            # 验证checkpoint内容
            with open(os.path.join(tmpdir, "checkpoint.json")) as f:
                checkpoint = json.load(f)
                assert checkpoint["step"] == 100

    def test_save_checkpoint_expert_with_save_method(self, mock_critic, config):
        """测试expert有save方法的checkpoint保存"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        expert = Mock()
        expert.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
        expert.save = Mock()

        trainer = FinSageAPPOTrainer(
            experts={"expert1": expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, step=50)
            expert.save.assert_called_once()

    def test_load_checkpoint(self, mock_expert, mock_critic, config):
        """测试加载checkpoint"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        mock_expert.load = Mock()

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=mock_critic,
            config=config,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 先保存
            trainer.save_checkpoint(tmpdir, step=100)

            # 再加载
            step = trainer.load_checkpoint(tmpdir)
            assert step == 100

    def test_load_checkpoint_nonexistent(self, mock_expert, config):
        """测试加载不存在的checkpoint"""
        from finsage.rl.marft_integration import FinSageAPPOTrainer

        trainer = FinSageAPPOTrainer(
            experts={"expert1": mock_expert},
            critic=None,
            config=config,
            device="cpu"
        )

        step = trainer.load_checkpoint("/nonexistent/path")
        assert step == 0


# ============================================================
# Test FinSageRewardFunction
# ============================================================

class TestFinSageRewardFunction:
    """测试FinSageRewardFunction奖励函数"""

    def test_reward_function_initialization(self):
        """测试奖励函数初始化"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction(
            risk_penalty_coef=0.5,
            transaction_cost_rate=0.001,
            diversification_bonus_coef=0.1,
            max_drawdown_penalty=1.0
        )

        assert reward_fn.risk_penalty_coef == 0.5
        assert reward_fn.transaction_cost_rate == 0.001

    def test_compute_reward_positive_return(self):
        """测试正收益情况"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction()

        total_reward, components = reward_fn.compute_reward(
            portfolio_return=0.02,
            portfolio_volatility=0.15,
            transaction_volume=10000,
            portfolio_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            max_drawdown=0.05
        )

        assert isinstance(total_reward, float)
        assert "return_reward" in components
        assert "risk_penalty" in components
        assert "transaction_cost" in components
        assert "diversification_bonus" in components
        assert "drawdown_penalty" in components

        # 验证收益组成
        assert components["return_reward"] == 0.02
        assert components["risk_penalty"] < 0  # 应该是负的
        assert components["transaction_cost"] < 0  # 应该是负的

    def test_compute_reward_negative_return(self):
        """测试负收益情况"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction()

        total_reward, components = reward_fn.compute_reward(
            portfolio_return=-0.01,
            portfolio_volatility=0.2,
            transaction_volume=5000,
            portfolio_weights=np.array([0.5, 0.5, 0.0, 0.0, 0.0]),
            max_drawdown=0.15
        )

        assert components["return_reward"] == -0.01
        assert total_reward < components["return_reward"]  # 惩罚应该进一步降低总奖励

    def test_compute_reward_diversification(self):
        """测试多样化奖励"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction(diversification_bonus_coef=0.1)

        # 均匀分布 vs 集中分布
        _, components_diverse = reward_fn.compute_reward(
            portfolio_return=0.0,
            portfolio_volatility=0.1,
            transaction_volume=0,
            portfolio_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            max_drawdown=0.0
        )

        _, components_concentrated = reward_fn.compute_reward(
            portfolio_return=0.0,
            portfolio_volatility=0.1,
            transaction_volume=0,
            portfolio_weights=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            max_drawdown=0.0
        )

        # 多样化的组合应该有更高的奖励
        assert components_diverse["diversification_bonus"] > components_concentrated["diversification_bonus"]

    def test_compute_reward_high_drawdown(self):
        """测试高回撤惩罚"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction(max_drawdown_penalty=1.0)

        # 低回撤
        _, components_low_dd = reward_fn.compute_reward(
            portfolio_return=0.02,
            portfolio_volatility=0.1,
            transaction_volume=0,
            portfolio_weights=np.array([0.2] * 5),
            max_drawdown=0.05
        )

        # 高回撤
        _, components_high_dd = reward_fn.compute_reward(
            portfolio_return=0.02,
            portfolio_volatility=0.1,
            transaction_volume=0,
            portfolio_weights=np.array([0.2] * 5),
            max_drawdown=0.20
        )

        # 高回撤应该有更大的惩罚
        assert components_high_dd["drawdown_penalty"] < components_low_dd["drawdown_penalty"]

    def test_compute_reward_zero_weights(self):
        """测试零权重处理"""
        from finsage.rl.marft_integration import FinSageRewardFunction

        reward_fn = FinSageRewardFunction()

        # 全零权重应该不报错
        total_reward, components = reward_fn.compute_reward(
            portfolio_return=0.0,
            portfolio_volatility=0.0,
            transaction_volume=0,
            portfolio_weights=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            max_drawdown=0.0
        )

        assert isinstance(total_reward, float)
        assert not np.isnan(total_reward)


# ============================================================
# Test MARFTFinSageIntegration
# ============================================================

class TestMARFTFinSageIntegration:
    """测试MARFTFinSageIntegration主类"""

    @pytest.fixture
    def mock_env(self):
        """创建模拟环境"""
        env = Mock()
        env.reset.return_value = ({"market": "data"}, {"info": "test"})
        env.step.return_value = (
            {"market": "next"},
            0.05,
            False,
            False,
            {"portfolio": {"return": 0.05, "volatility": 0.1, "transaction_volume": 1000}}
        )
        return env

    @pytest.fixture
    def mock_experts(self):
        """创建模拟experts"""
        experts = {}
        for asset_class in ["stocks", "bonds", "commodities", "reits", "crypto"]:
            expert = Mock()
            expert.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]
            expert.generate_action.return_value = (
                {"action": "BUY_50%", "confidence": 0.8},
                torch.tensor([1, 2, 3]),
                "raw response"
            )
            expert.get_action_log_prob.return_value = (
                torch.tensor(-0.5, requires_grad=True),
                torch.tensor(0.1, requires_grad=True)
            )
            expert.eval = Mock()
            expert.train = Mock()
            experts[asset_class] = expert
        return experts

    @pytest.fixture
    def config(self):
        """创建配置"""
        return {
            "gamma": 0.99,
            "episode_length": 10,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "ppo_epoch": 2,
            "num_mini_batch": 2,
            "lr": 5e-7,
            "critic_lr": 5e-4,
            "risk_penalty_coef": 0.5,
            "transaction_cost_rate": 0.001,
        }

    def test_integration_initialization(self, mock_experts, mock_env, config):
        """测试整合类初始化"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            critic=None,
            device="cpu"
        )

        assert integration.flex_env is not None
        assert integration.buffer is not None
        assert integration.trainer is not None
        assert integration.global_step == 0
        assert integration.episode_count == 0

    def test_integration_with_custom_critic(self, mock_experts, mock_env, config):
        """测试使用自定义critic初始化"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        critic = Mock(spec=nn.Module)
        critic.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            critic=critic,
            device="cpu"
        )

        assert integration.critic is critic

    def test_get_expert_by_index_by_asset_class(self, mock_experts, mock_env, config):
        """测试通过asset_class获取expert"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        # Stock Expert (index 0)
        expert = integration._get_expert_by_index(0)
        assert expert is mock_experts["stocks"]

        # Bond Expert (index 1)
        expert = integration._get_expert_by_index(1)
        assert expert is mock_experts["bonds"]

    def test_get_expert_by_index_by_role(self, mock_env, config):
        """测试通过role获取expert"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        experts = {
            "Stock_Expert": Mock(),
            "Bond_Expert": Mock(),
        }

        integration = MARFTFinSageIntegration(
            experts=experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        expert = integration._get_expert_by_index(0)
        assert expert is experts["Stock_Expert"]

    def test_get_expert_by_index_finsage_mas(self, mock_env, config):
        """测试从FinSageMAS获取expert"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        mas = Mock()
        mas.experts = {
            "Stock_Expert": Mock(),
            "stocks": Mock(),
        }

        integration = MARFTFinSageIntegration(
            experts=mas,
            env=mock_env,
            config=config,
            device="cpu"
        )

        expert = integration._get_expert_by_index(0)
        assert expert is not None

    def test_get_expert_by_index_not_found(self, mock_env, config):
        """测试找不到expert的情况"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts={},
            env=mock_env,
            config=config,
            device="cpu"
        )

        expert = integration._get_expert_by_index(0)
        assert expert is None

    def test_estimate_values_with_critic(self, mock_experts, mock_env, config):
        """测试value估计（有critic）"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        critic = Mock()
        critic.get_value.return_value = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            critic=critic,
            device="cpu"
        )

        obs_list = ["obs1", "obs2", "obs3", "obs4", "obs5"]
        values = integration._estimate_values(obs_list)

        assert len(values) == 5
        critic.get_value.assert_called_once()

    def test_estimate_values_without_critic(self, mock_experts, mock_env, config):
        """测试value估计（无critic）"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            critic=None,
            device="cpu"
        )

        values = integration._estimate_values(["obs1", "obs2"])
        assert values == [0.0, 0.0]

    def test_estimate_values_scalar(self, mock_experts, mock_env, config):
        """测试value估计返回标量"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        critic = Mock()
        critic.get_value.return_value = torch.tensor(0.5)  # scalar

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            critic=critic,
            device="cpu"
        )

        values = integration._estimate_values(["obs1", "obs2", "obs3"])
        assert len(values) == 3
        assert all(v == 0.5 for v in values)

    def test_collect_rollout_basic(self, mock_experts, mock_env, config):
        """测试基本rollout收集"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        stats = integration.collect_rollout(num_steps=3)

        assert "mean_reward" in stats
        assert "total_reward" in stats
        assert "num_steps" in stats
        assert stats["num_steps"] == 3

    def test_collect_rollout_with_expert_failure(self, mock_env, config):
        """测试expert生成失败的rollout"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        expert = Mock()
        expert.generate_action.side_effect = Exception("Generation failed")

        experts = {"stocks": expert}

        integration = MARFTFinSageIntegration(
            experts=experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        # 应该降级为默认动作，不应该崩溃
        stats = integration.collect_rollout(num_steps=2)
        assert "mean_reward" in stats

    def test_collect_rollout_with_episode_termination(self, mock_experts, config):
        """测试episode终止的rollout"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        env = Mock()
        env.reset.return_value = ({"market": "data"}, {})

        # 第一步不终止，第二步终止
        env.step.side_effect = [
            ({"market": "next1"}, 0.1, False, False, {"portfolio": {}}),
            ({"market": "next2"}, 0.2, True, False, {"portfolio": {}}),
            ({"market": "next3"}, 0.3, False, False, {"portfolio": {}}),
        ]

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=env,
            config=config,
            device="cpu"
        )

        stats = integration.collect_rollout(num_steps=3)

        assert stats["episodes_completed"] >= 1
        assert integration.episode_count >= 1

    def test_train_step(self, mock_experts, mock_env, config):
        """测试训练步"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        # 先收集数据
        integration.collect_rollout(num_steps=3)

        # 训练
        stats = integration.train_step()

        assert isinstance(stats, dict)
        assert integration.buffer.step == 0  # buffer应该被清空

    def test_train_step_empty_buffer(self, mock_experts, mock_env, config):
        """测试空buffer的训练步"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        # 不收集数据，直接训练
        stats = integration.train_step()
        assert isinstance(stats, dict)

    def test_evaluate(self, mock_experts, mock_env, config):
        """测试评估"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        # 设置环境终止
        env = Mock()
        env.reset.return_value = ({"market": "data"}, {})
        env.step.return_value = (
            {"market": "next"},
            0.1,
            True,  # 立即终止
            False,
            {"portfolio": {}}
        )

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=env,
            config=config,
            device="cpu"
        )

        eval_stats = integration.evaluate(num_episodes=2)

        assert "mean_episode_reward" in eval_stats
        assert "std_episode_reward" in eval_stats
        assert "mean_episode_length" in eval_stats
        assert eval_stats["num_episodes"] == 2

        # 验证eval/train模式切换
        for expert in mock_experts.values():
            expert.eval.assert_called()
            expert.train.assert_called()

    def test_evaluate_expert_without_methods(self, mock_env, config):
        """测试expert没有eval/train方法"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        expert = Mock()
        expert.generate_action.return_value = ({"action": "HOLD"}, torch.tensor([]), "")
        expert.get_action_log_prob.return_value = (torch.tensor(-0.5), torch.tensor(0.1))
        # No eval/train methods

        env = Mock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0.0, True, False, {"portfolio": {}})

        experts = {"stocks": expert}

        integration = MARFTFinSageIntegration(
            experts=experts,
            env=env,
            config=config,
            device="cpu"
        )

        # 应该不报错
        eval_stats = integration.evaluate(num_episodes=1)
        assert "mean_episode_reward" in eval_stats

    def test_run_training(self, mock_experts, mock_env, config):
        """测试完整训练循环"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=mock_env,
            config=config,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            integration.run_training(
                num_env_steps=20,  # 小数量
                rollout_length=5,
                log_interval=1,
                save_interval=2,
                eval_interval=10,
                save_dir=tmpdir
            )

            # 验证训练完成
            assert integration.global_step == 20

            # 验证checkpoint保存
            final_dir = os.path.join(tmpdir, "final")
            assert os.path.exists(final_dir)

    def test_run_training_best_model_saving(self, mock_experts, config):
        """测试最佳模型保存"""
        from finsage.rl.marft_integration import MARFTFinSageIntegration

        env = Mock()
        env.reset.return_value = ({}, {})

        # 递增的奖励
        rewards = [0.1, 0.2, 0.3]
        env.step.side_effect = [
            ({}, r, False, False, {"portfolio": {"return": r}})
            for r in rewards
        ]

        integration = MARFTFinSageIntegration(
            experts=mock_experts,
            env=env,
            config=config,
            device="cpu"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            integration.run_training(
                num_env_steps=3,
                rollout_length=1,
                log_interval=1,
                save_interval=10,
                save_dir=tmpdir
            )

            # 应该保存了最佳模型
            best_dir = os.path.join(tmpdir, "best")
            assert os.path.exists(best_dir)


# ============================================================
# Test Agent Profiles and Constants
# ============================================================

class TestAgentProfiles:
    """测试Agent Profiles配置"""

    def test_agent_profiles_structure(self):
        """测试agent profiles结构"""
        from finsage.rl.marft_integration import FINSAGE_AGENT_PROFILES

        assert len(FINSAGE_AGENT_PROFILES) == 5

        required_keys = ["role", "expertise", "asset_class", "dependencies"]
        for profile in FINSAGE_AGENT_PROFILES:
            for key in required_keys:
                assert key in profile

    def test_agent_profiles_roles(self):
        """测试角色定义"""
        from finsage.rl.marft_integration import FINSAGE_AGENT_PROFILES

        roles = [p["role"] for p in FINSAGE_AGENT_PROFILES]
        assert "Stock_Expert" in roles
        assert "Bond_Expert" in roles
        assert "Commodity_Expert" in roles
        assert "REITs_Expert" in roles
        assert "Crypto_Expert" in roles

    def test_agent_profiles_asset_classes(self):
        """测试资产类别"""
        from finsage.rl.marft_integration import FINSAGE_AGENT_PROFILES

        asset_classes = [p["asset_class"] for p in FINSAGE_AGENT_PROFILES]
        assert "stocks" in asset_classes
        assert "bonds" in asset_classes
        assert "commodities" in asset_classes
        assert "reits" in asset_classes
        assert "crypto" in asset_classes

    def test_agent_profiles_dependencies(self):
        """测试依赖关系"""
        from finsage.rl.marft_integration import FINSAGE_AGENT_PROFILES

        # Stock Expert应该没有依赖
        stock_expert = next(p for p in FINSAGE_AGENT_PROFILES if p["role"] == "Stock_Expert")
        assert stock_expert["dependencies"] == []

        # Bond Expert应该依赖Stock Expert
        bond_expert = next(p for p in FINSAGE_AGENT_PROFILES if p["role"] == "Bond_Expert")
        assert "Stock_Expert" in bond_expert["dependencies"]

    def test_default_config(self):
        """测试默认配置"""
        from finsage.rl.marft_integration import DEFAULT_MARFT_FINSAGE_CONFIG

        assert DEFAULT_MARFT_FINSAGE_CONFIG["gamma"] == 0.99
        assert DEFAULT_MARFT_FINSAGE_CONFIG["episode_length"] == 252
        assert DEFAULT_MARFT_FINSAGE_CONFIG["clip_param"] == 0.2
        assert DEFAULT_MARFT_FINSAGE_CONFIG["lr"] == 5e-7


# ============================================================
# Integration Tests
# ============================================================

class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_full_pipeline_minimal(self):
        """测试完整流程（最小版本）"""
        from finsage.rl.marft_integration import (
            FinSageFlexMGEnv,
            FinSageActionBuffer,
            FinSageRewardFunction,
            FINSAGE_AGENT_PROFILES
        )

        # 创建mock环境
        base_env = Mock()
        base_env.reset.return_value = ({"market": "data"}, {})
        base_env.step.return_value = ({}, 0.1, False, False, {"portfolio": {}})

        # 创建Flex-MG环境
        flex_env = FinSageFlexMGEnv(
            base_env=base_env,
            agent_profiles=FINSAGE_AGENT_PROFILES[:3],  # 只用3个agent
            gamma=0.99
        )

        # 创建buffer
        buffer = FinSageActionBuffer(
            episode_length=10,
            num_agents=3,
            gamma=0.99,
            gae_lambda=0.95
        )

        # 创建奖励函数
        reward_fn = FinSageRewardFunction()

        # 模拟一个step
        obs, _ = flex_env.reset()

        joint_action = [
            {"action": "BUY_50%"},
            {"action": "HOLD"},
            {"action": "SELL_25%"}
        ]

        next_obs, reward, term, trunc, info = flex_env.step(joint_action)

        # 计算增强奖励
        enhanced_reward, components = reward_fn.compute_reward(
            portfolio_return=reward,
            portfolio_volatility=0.1,
            transaction_volume=1000,
            portfolio_weights=np.array([0.3, 0.3, 0.4]),
            max_drawdown=0.05
        )

        # 存入buffer
        buffer.insert(
            obs=["obs1", "obs2", "obs3"],
            actions=joint_action,
            log_probs=[0.0, 0.0, 0.0],
            reward=enhanced_reward,
            value=[0.0, 0.0, 0.0],
            done=term or trunc
        )

        # 计算GAE
        advantages, returns = buffer.compute_gae_and_returns([0.0, 0.0, 0.0])

        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)

    def test_dependency_order_execution(self):
        """测试依赖顺序执行"""
        from finsage.rl.marft_integration import FinSageFlexMGEnv, FINSAGE_AGENT_PROFILES

        base_env = Mock()
        base_env.reset.return_value = ({}, {})

        env = FinSageFlexMGEnv(base_env, FINSAGE_AGENT_PROFILES)

        # 验证拓扑顺序
        # Stock Expert (0) 应该没有依赖
        assert env.dependency_graph[0] == []

        # 其他expert应该有依赖
        for i in range(1, len(FINSAGE_AGENT_PROFILES)):
            deps = env.dependency_graph[i]
            # 所有依赖的索引都应该小于当前索引（拓扑顺序）
            for dep_idx in deps:
                assert dep_idx < i


# ============================================================
# Main Test Runner
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

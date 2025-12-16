"""
MARFT-FinSage Integration Module
将MARFT的多智能体强化学习微调框架整合到FinSage金融投资系统

核心思想:
1. 将FinSage的5个Expert Agent映射为MARFT的LLM Agent
2. 使用Flex-MG框架建模Expert之间的依赖关系
3. 通过APPO算法对Expert进行端到端强化学习微调
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

logger = logging.getLogger(__name__)


# ============================================================
# 1. Agent Profile Configuration
# ============================================================

FINSAGE_AGENT_PROFILES = [
    {
        "role": "Stock_Expert",
        "expertise": """你是一位专业的股票投资专家，擅长:
- 基本面分析：财务报表、盈利能力、估值模型
- 技术面分析：价格趋势、成交量、技术指标
- 行业分析：行业周期、竞争格局、政策影响
- 市场情绪分析：新闻事件、投资者情绪""",
        "asset_class": "stocks",
        "dependencies": [],  # 第一个决策，无依赖
    },
    {
        "role": "Bond_Expert",
        "expertise": """你是一位专业的债券投资专家，擅长:
- 利率分析：收益率曲线、久期、凸性
- 信用分析：信用评级、违约风险、利差
- 宏观分析：央行政策、通胀预期、经济周期
- 债券估值：现金流折现、相对价值分析""",
        "asset_class": "bonds",
        "dependencies": ["Stock_Expert"],  # 参考股票专家观点
    },
    {
        "role": "Commodity_Expert",
        "expertise": """你是一位专业的大宗商品投资专家，擅长:
- 供需分析：产量、库存、消费趋势
- 宏观因素：美元走势、地缘政治、季节性
- 期货市场：期限结构、持仓分析
- 跨品种分析：能源、金属、农产品关联""",
        "asset_class": "commodities",
        "dependencies": ["Stock_Expert", "Bond_Expert"],
    },
    {
        "role": "REITs_Expert",
        "expertise": """你是一位专业的房地产投资信托(REITs)专家，擅长:
- 物业分析：租金收益、出租率、物业估值
- 行业细分：办公、零售、工业、住宅REITs
- 宏观因素：利率敏感性、经济周期影响
- 财务分析：FFO、NAV、派息率""",
        "asset_class": "reits",
        "dependencies": ["Stock_Expert", "Bond_Expert"],
    },
    {
        "role": "Crypto_Expert",
        "expertise": """你是一位专业的加密货币投资专家，擅长:
- 链上分析：交易量、活跃地址、持仓分布
- 技术发展：协议升级、生态系统、DeFi趋势
- 市场情绪：社交媒体、恐惧贪婪指数
- 监管动态：政策变化、合规发展""",
        "asset_class": "crypto",
        "dependencies": ["Stock_Expert"],  # 与风险资产相关
    },
]


# ============================================================
# 2. Flex-MG Environment Wrapper
# ============================================================

@dataclass
class FlexMGState:
    """Flex-MG 状态表示"""
    market_data: Dict[str, Any]       # 市场数据
    news_data: List[Dict]             # 新闻数据
    technical_indicators: Dict        # 技术指标
    portfolio_state: Dict             # 当前组合状态
    predecessor_actions: List[Dict]   # 前序Agent的决策
    timestamp: str


class FinSageFlexMGEnv:
    """
    将FinSage的MultiAssetTradingEnv包装为Flex-MG环境

    Flex-MG = <V, N, S, A, T, R, γ, D>
    - V: 协调变量 (market state)
    - N: Agent数量 (5 experts + 1 PM)
    - S: 状态空间 (market + portfolio + predecessor actions)
    - A: 动作空间 (13-action trading space)
    - T: 状态转移 (市场演化)
    - R: 奖励函数 (组合收益 - 风险惩罚)
    - γ: 折扣因子
    - D: 依赖函数 (Expert依赖关系)
    """

    def __init__(
        self,
        base_env,  # MultiAssetTradingEnv
        agent_profiles: List[Dict],
        gamma: float = 0.99,
    ):
        self.base_env = base_env
        self.agent_profiles = agent_profiles
        self.num_agents = len(agent_profiles)
        self.gamma = gamma

        # 构建依赖图
        self.dependency_graph = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[int, List[int]]:
        """构建Agent依赖关系图 D(i)"""
        role_to_idx = {p["role"]: i for i, p in enumerate(self.agent_profiles)}
        graph = {}
        for i, profile in enumerate(self.agent_profiles):
            deps = profile.get("dependencies", [])
            graph[i] = [role_to_idx[d] for d in deps if d in role_to_idx]
        return graph

    def get_agent_observation(
        self,
        agent_idx: int,
        base_obs: Dict[str, Any],
        predecessor_actions: List[Dict],
    ) -> str:
        """
        构建Agent的观察 (prompt)
        包含: 市场数据 + 依赖Agent的决策
        """
        profile = self.agent_profiles[agent_idx]

        # 基础市场观察
        obs_prompt = f"""## 当前市场状态
{self._format_market_data(base_obs)}

## 你的专业领域
{profile['expertise']}

## 你负责的资产类别: {profile['asset_class']}
"""

        # 添加依赖Agent的决策
        deps = self.dependency_graph.get(agent_idx, [])
        if deps and predecessor_actions:
            obs_prompt += "\n## 其他专家的建议 (供参考)\n"
            for dep_idx in deps:
                if dep_idx < len(predecessor_actions):
                    dep_action = predecessor_actions[dep_idx]
                    dep_role = self.agent_profiles[dep_idx]["role"]
                    obs_prompt += f"\n### {dep_role}:\n{json.dumps(dep_action, ensure_ascii=False, indent=2)}\n"

        return obs_prompt

    def _format_market_data(self, obs: Dict[str, Any]) -> str:
        """格式化市场数据为文本"""
        # 根据实际obs结构调整
        return json.dumps(obs, ensure_ascii=False, indent=2, default=str)

    def reset(self) -> Tuple[Dict, Dict]:
        """重置环境"""
        obs, info = self.base_env.reset()
        return obs, info

    def step(self, joint_action: List[Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行联合动作

        Args:
            joint_action: 所有Agent的动作列表

        Returns:
            obs, reward, terminated, truncated, info
        """
        # 将Expert建议聚合为组合动作
        portfolio_action = self._aggregate_expert_actions(joint_action)

        # 执行环境step
        obs, reward, terminated, truncated, info = self.base_env.step(portfolio_action)

        return obs, reward, terminated, truncated, info

    def _aggregate_expert_actions(self, expert_actions: List[Dict]) -> Dict:
        """
        聚合Expert建议为最终组合动作
        这里可以实现投票、加权平均等策略
        """
        aggregated = {}
        for i, action in enumerate(expert_actions):
            asset_class = self.agent_profiles[i]["asset_class"]
            aggregated[asset_class] = action
        return aggregated


# ============================================================
# 3. MARFT-style Trajectory Buffer for FinSage
# ============================================================

class FinSageActionBuffer:
    """
    Action-level Trajectory Buffer
    存储Expert决策轨迹，用于计算GAE和训练
    """

    def __init__(
        self,
        episode_length: int,
        num_agents: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._reset_storage()

    def _reset_storage(self):
        """重置存储"""
        self.observations = []      # List of obs strings per agent
        self.actions = []           # List of action dicts per agent
        self.action_log_probs = []  # List of log probs per agent
        self.rewards = []           # Shared team rewards
        self.values = []            # Value estimates per agent
        self.dones = []             # Episode termination flags

        self.step = 0

    def insert(
        self,
        obs: List[str],           # [num_agents] observation prompts
        actions: List[Dict],      # [num_agents] action dicts
        log_probs: List[float],   # [num_agents] action log probs
        reward: float,            # shared reward
        value: List[float],       # [num_agents] value estimates
        done: bool,
    ):
        """插入一步数据"""
        self.observations.append(obs)
        self.actions.append(actions)
        self.action_log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.step += 1

    def compute_gae_and_returns(
        self,
        next_value: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE和Returns

        遵循MARFT的反向迭代方式:
        - 先反向遍历时间步
        - 再反向遍历Agent (考虑依赖关系)
        """
        T = len(self.rewards)
        N = self.num_agents

        advantages = np.zeros((T, N))
        returns = np.zeros((T, N))

        # 转换为numpy
        values = np.array(self.values)  # [T, N]
        rewards = np.array(self.rewards)  # [T]
        dones = np.array(self.dones)  # [T]

        # 反向计算GAE
        gae = np.zeros(N)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = np.array(next_value)
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            # 对每个agent计算
            for i in reversed(range(N)):
                delta = rewards[t] + self.gamma * next_val[i] * next_non_terminal - values[t, i]
                gae[i] = delta + self.gamma * self.gae_lambda * next_non_terminal * gae[i]
                advantages[t, i] = gae[i]
                returns[t, i] = advantages[t, i] + values[t, i]

        return advantages, returns

    def get_batch(self) -> Dict[str, Any]:
        """获取训练batch"""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "action_log_probs": self.action_log_probs,
            "rewards": self.rewards,
            "values": self.values,
            "dones": self.dones,
        }

    def clear(self):
        """清空buffer"""
        self._reset_storage()


# ============================================================
# 4. APPO Trainer for FinSage Experts
# ============================================================

class FinSageAPPOTrainer:
    """
    Action-level PPO Trainer for FinSage

    训练策略:
    1. 收集Expert决策轨迹
    2. 计算GAE advantage
    3. 使用PPO更新Expert LLM (LoRA参数)
    """

    def __init__(
        self,
        experts: Dict[str, Any],  # LoRAExpert agents
        critic: nn.Module,        # Value network
        config: Dict[str, Any],
        device: str = "cuda:0",
    ):
        self.experts = experts
        self.critic = critic
        self.config = config
        self.device = torch.device(device)

        # PPO超参数
        self.clip_param = config.get("clip_param", 0.2)
        self.ppo_epoch = config.get("ppo_epoch", 5)
        self.num_mini_batch = config.get("num_mini_batch", 4)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.kl_threshold = config.get("kl_threshold", 0.01)

        # 学习率
        self.lr = config.get("lr", 5e-7)
        self.critic_lr = config.get("critic_lr", 5e-4)

        # Agent轮流训练间隔 (0=同时训练)
        self.agent_iteration_interval = config.get("agent_iteration_interval", 0)

        # 初始化优化器
        self._init_optimizers()

        logger.info(f"Initialized FinSageAPPOTrainer with {len(experts)} experts")

    def _init_optimizers(self):
        """初始化优化器"""
        # 每个Expert一个优化器 (只优化LoRA参数)
        self.expert_optimizers = {}
        for name, expert in self.experts.items():
            if hasattr(expert, 'parameters'):
                self.expert_optimizers[name] = AdamW(
                    expert.parameters(),
                    lr=self.lr,
                    weight_decay=0.0,
                )

        # Critic优化器
        if self.critic is not None:
            self.critic_optimizer = AdamW(
                self.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=0.01,
            )
        else:
            self.critic_optimizer = None

    def train(
        self,
        buffer: FinSageActionBuffer,
        next_value: List[float],
        global_step: int,
        action_tokens_buffer: List[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        执行PPO训练

        Args:
            buffer: 包含轨迹数据的buffer
            next_value: 最后状态的value估计
            global_step: 全局训练步数
            action_tokens_buffer: 动作token缓存 [T, num_agents, seq_len]

        Returns:
            训练统计信息
        """
        # 计算GAE
        advantages, returns = buffer.compute_gae_and_returns(next_value)

        # 转换为tensor
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 标准化advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        batch = buffer.get_batch()
        T = len(batch["rewards"])
        num_agents = buffer.num_agents

        # 统计信息
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        for epoch in range(self.ppo_epoch):
            # Mini-batch 训练
            indices = np.random.permutation(T)
            batch_size = max(1, T // self.num_mini_batch)

            for mb in range(self.num_mini_batch):
                mb_start = mb * batch_size
                mb_end = min((mb + 1) * batch_size, T)
                mb_indices = indices[mb_start:mb_end]

                if len(mb_indices) == 0:
                    continue

                # 获取mini-batch数据
                mb_obs = [batch["observations"][i] for i in mb_indices]
                mb_old_log_probs = np.array([batch["action_log_probs"][i] for i in mb_indices])
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # 更新每个Expert
                for expert_idx, (expert_name, expert) in enumerate(self.experts.items()):
                    # Agent轮流训练逻辑
                    if self.agent_iteration_interval > 0:
                        current_agent_turn = (global_step // self.agent_iteration_interval) % len(self.experts)
                        if current_agent_turn != expert_idx:
                            continue

                    # 检查expert是否有必要的方法
                    if not hasattr(expert, 'get_action_log_prob'):
                        continue

                    optimizer = self.expert_optimizers.get(expert_name)
                    if optimizer is None:
                        continue

                    # 收集该expert的数据
                    expert_log_probs = []
                    expert_entropies = []

                    for batch_idx, t_idx in enumerate(mb_indices):
                        # 获取该时间步该expert的观察
                        obs_t = mb_obs[batch_idx][expert_idx] if isinstance(mb_obs[batch_idx], list) else mb_obs[batch_idx]

                        # 获取动作tokens
                        if action_tokens_buffer is not None and t_idx < len(action_tokens_buffer):
                            action_tokens = action_tokens_buffer[t_idx][expert_idx]
                        else:
                            # 如果没有token缓存，跳过
                            continue

                        # 获取前序动作
                        pred_actions = []
                        if t_idx < len(batch["actions"]):
                            for pred_idx in range(expert_idx):
                                pred_actions.append(batch["actions"][t_idx][pred_idx])

                        # 计算log_prob和entropy
                        try:
                            log_prob, entropy = expert.get_action_log_prob(
                                obs_t, action_tokens, pred_actions
                            )
                            expert_log_probs.append(log_prob)
                            expert_entropies.append(entropy)
                        except Exception as e:
                            logger.warning(f"Error computing log_prob for {expert_name}: {e}")
                            continue

                    if len(expert_log_probs) == 0:
                        continue

                    # 转换为tensor
                    new_log_probs = torch.stack(expert_log_probs)
                    entropies = torch.stack(expert_entropies)

                    # 获取旧的log_probs
                    old_log_probs = torch.tensor(
                        mb_old_log_probs[:len(new_log_probs), expert_idx],
                        dtype=torch.float32,
                        device=self.device
                    )

                    # 获取advantages
                    adv = mb_advantages[:len(new_log_probs), expert_idx]

                    # PPO loss计算
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Entropy bonus
                    entropy_loss = -self.entropy_coef * entropies.mean()

                    # 总损失
                    total_loss = policy_loss + entropy_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()

                    # 梯度裁剪
                    if hasattr(expert, 'model'):
                        torch.nn.utils.clip_grad_norm_(expert.model.parameters(), self.max_grad_norm)

                    optimizer.step()

                    # 统计
                    with torch.no_grad():
                        kl = (old_log_probs - new_log_probs).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                    stats["policy_loss"] += policy_loss.item()
                    stats["entropy"] += entropies.mean().item()
                    stats["kl_divergence"] += kl
                    stats["clip_fraction"] += clip_frac
                    update_count += 1

                # 更新Critic
                if self.critic is not None and self.critic_optimizer is not None:
                    # 获取新的value估计
                    # 这里简化处理，实际需要根据critic类型调整
                    try:
                        # 假设critic可以处理文本观察
                        obs_texts = [mb_obs[i][0] if isinstance(mb_obs[i], list) else mb_obs[i]
                                     for i in range(len(mb_obs))]
                        if hasattr(self.critic, 'get_value'):
                            new_values = self.critic.get_value(obs_texts[:len(mb_returns)])
                        else:
                            new_values = mb_returns.mean(dim=1)  # fallback

                        # Value loss (MSE)
                        value_loss = self.value_loss_coef * F.mse_loss(
                            new_values, mb_returns[:len(new_values)].mean(dim=1)
                        )

                        self.critic_optimizer.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.critic_optimizer.step()

                        stats["value_loss"] += value_loss.item()
                    except Exception as e:
                        logger.warning(f"Error updating critic: {e}")

            # Early stopping based on KL divergence
            if update_count > 0 and stats["kl_divergence"] / update_count > self.kl_threshold:
                logger.info(f"Early stopping at epoch {epoch} due to KL threshold")
                break

        # 平均统计
        if update_count > 0:
            stats = {k: v / update_count for k, v in stats.items()}

        return stats

    def save_checkpoint(self, save_dir: str, step: int):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "step": step,
            "config": self.config,
        }

        # 保存每个Expert的LoRA权重
        for name, expert in self.experts.items():
            expert_dir = os.path.join(save_dir, f"expert_{name}")
            os.makedirs(expert_dir, exist_ok=True)

            if hasattr(expert, 'save'):
                expert.save(expert_dir)
            elif hasattr(expert, 'model'):
                # 保存LoRA适配器
                try:
                    expert.model.save_pretrained(expert_dir)
                    logger.info(f"Saved expert {name} to {expert_dir}")
                except Exception as e:
                    logger.warning(f"Failed to save expert {name}: {e}")

        # 保存Critic
        if self.critic is not None:
            critic_path = os.path.join(save_dir, "critic.pt")
            torch.save({
                "model_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict() if self.critic_optimizer else None,
            }, critic_path)
            logger.info(f"Saved critic to {critic_path}")

        # 保存优化器状态
        optimizer_states = {}
        for name, opt in self.expert_optimizers.items():
            optimizer_states[name] = opt.state_dict()
        torch.save(optimizer_states, os.path.join(save_dir, "expert_optimizers.pt"))

        # 保存checkpoint元信息
        checkpoint_path = os.path.join(save_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {save_dir} at step {step}")

    def load_checkpoint(self, load_dir: str) -> int:
        """加载检查点"""
        checkpoint_path = os.path.join(load_dir, "checkpoint.json")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {load_dir}")
            return 0

        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        # 加载Expert权重
        for name, expert in self.experts.items():
            expert_dir = os.path.join(load_dir, f"expert_{name}")
            if os.path.exists(expert_dir):
                if hasattr(expert, 'load'):
                    expert.load(expert_dir)
                logger.info(f"Loaded expert {name} from {expert_dir}")

        # 加载Critic
        critic_path = os.path.join(load_dir, "critic.pt")
        if self.critic is not None and os.path.exists(critic_path):
            state = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(state["model_state_dict"])
            if self.critic_optimizer and state.get("optimizer_state_dict"):
                self.critic_optimizer.load_state_dict(state["optimizer_state_dict"])
            logger.info(f"Loaded critic from {critic_path}")

        # 加载优化器状态
        opt_path = os.path.join(load_dir, "expert_optimizers.pt")
        if os.path.exists(opt_path):
            optimizer_states = torch.load(opt_path, map_location=self.device)
            for name, state in optimizer_states.items():
                if name in self.expert_optimizers:
                    self.expert_optimizers[name].load_state_dict(state)

        logger.info(f"Loaded checkpoint from {load_dir}, step={checkpoint.get('step', 0)}")
        return checkpoint.get("step", 0)


# ============================================================
# 5. Reward Function Design for FinSage
# ============================================================

class FinSageRewardFunction:
    """
    FinSage奖励函数设计

    奖励组成:
    1. 组合收益 (Portfolio Return)
    2. 风险惩罚 (Risk Penalty)
    3. 交易成本 (Transaction Cost)
    4. 多样化奖励 (Diversification Bonus)
    """

    def __init__(
        self,
        risk_penalty_coef: float = 0.5,
        transaction_cost_rate: float = 0.001,
        diversification_bonus_coef: float = 0.1,
        max_drawdown_penalty: float = 1.0,
    ):
        self.risk_penalty_coef = risk_penalty_coef
        self.transaction_cost_rate = transaction_cost_rate
        self.diversification_bonus_coef = diversification_bonus_coef
        self.max_drawdown_penalty = max_drawdown_penalty

    def compute_reward(
        self,
        portfolio_return: float,
        portfolio_volatility: float,
        transaction_volume: float,
        portfolio_weights: np.ndarray,
        max_drawdown: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励

        Returns:
            total_reward: 总奖励
            reward_components: 各组成部分
        """
        # 1. 收益奖励
        return_reward = portfolio_return

        # 2. 风险惩罚 (波动率)
        risk_penalty = -self.risk_penalty_coef * portfolio_volatility

        # 3. 交易成本
        transaction_cost = -self.transaction_cost_rate * transaction_volume

        # 4. 多样化奖励 (使用熵)
        weights = np.abs(portfolio_weights)
        weights = weights / (weights.sum() + 1e-8)
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        diversification_bonus = self.diversification_bonus_coef * entropy

        # 5. 最大回撤惩罚
        drawdown_penalty = -self.max_drawdown_penalty * max(0, max_drawdown - 0.1)

        total_reward = (
            return_reward +
            risk_penalty +
            transaction_cost +
            diversification_bonus +
            drawdown_penalty
        )

        components = {
            "return_reward": return_reward,
            "risk_penalty": risk_penalty,
            "transaction_cost": transaction_cost,
            "diversification_bonus": diversification_bonus,
            "drawdown_penalty": drawdown_penalty,
        }

        return total_reward, components


# ============================================================
# 6. Main Integration Class
# ============================================================

class MARFTFinSageIntegration:
    """
    MARFT-FinSage 整合主类

    整合流程:
    1. 初始化Expert Agents (作为MARFT的LLM Agents)
    2. 包装Trading Environment为Flex-MG格式
    3. 使用APPO训练Expert
    4. 评估和部署
    """

    def __init__(
        self,
        experts: Dict[str, Any],  # LoRAExpert agents 或 FinSageMAS
        env,                       # MultiAssetTradingEnv
        config: Dict[str, Any],
        critic: nn.Module = None,  # 可选的Critic网络
        device: str = "cuda:0",
    ):
        self.experts = experts
        self.config = config
        self.device = torch.device(device)

        # 创建Flex-MG环境
        self.flex_env = FinSageFlexMGEnv(
            base_env=env,
            agent_profiles=FINSAGE_AGENT_PROFILES,
            gamma=config.get("gamma", 0.99),
        )

        # 创建Buffer
        self.buffer = FinSageActionBuffer(
            episode_length=config.get("episode_length", 252),  # 一年交易日
            num_agents=len(FINSAGE_AGENT_PROFILES),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
        )

        # Action tokens 缓存 (用于PPO训练时重新计算log_prob)
        self.action_tokens_buffer: List[List[torch.Tensor]] = []

        # 创建奖励函数
        self.reward_fn = FinSageRewardFunction(
            risk_penalty_coef=config.get("risk_penalty_coef", 0.5),
            transaction_cost_rate=config.get("transaction_cost_rate", 0.001),
            diversification_bonus_coef=config.get("diversification_bonus_coef", 0.1),
            max_drawdown_penalty=config.get("max_drawdown_penalty", 1.0),
        )

        # 初始化Critic (如果未提供)
        if critic is None:
            critic = self._create_default_critic()
        self.critic = critic

        # 创建训练器
        self.trainer = FinSageAPPOTrainer(
            experts=experts,
            critic=self.critic,
            config=config,
            device=device,
        )

        # 跟踪训练状态
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = float('-inf')

        logger.info("MARFT-FinSage Integration initialized")

    def _create_default_critic(self) -> nn.Module:
        """创建默认的Critic网络"""
        try:
            from finsage.rl.critic import PortfolioValueCritic
            critic = PortfolioValueCritic(
                num_assets=10,  # 默认值，可配置
                hidden_size=256,
            ).to(self.device)
            logger.info("Created default PortfolioValueCritic")
            return critic
        except Exception as e:
            logger.warning(f"Failed to create default critic: {e}")
            return None

    def _get_expert_by_index(self, agent_idx: int):
        """根据索引获取Expert"""
        asset_class = FINSAGE_AGENT_PROFILES[agent_idx]["asset_class"]
        role = FINSAGE_AGENT_PROFILES[agent_idx]["role"]

        # 尝试不同的键名
        if isinstance(self.experts, dict):
            if asset_class in self.experts:
                return self.experts[asset_class]
            if role in self.experts:
                return self.experts[role]

        # 如果是 FinSageMAS
        if hasattr(self.experts, 'experts'):
            if role in self.experts.experts:
                return self.experts.experts[role]
            if asset_class in self.experts.experts:
                return self.experts.experts[asset_class]

        return None

    def collect_rollout(self, num_steps: int) -> Dict[str, float]:
        """
        收集rollout数据

        实现MARFT的sequential action generation
        """
        obs, info = self.flex_env.reset()

        episode_rewards = []
        episode_returns = []
        current_episode_reward = 0.0

        # 清空action tokens缓存
        self.action_tokens_buffer = []

        for step in range(num_steps):
            predecessor_actions = []
            step_obs = []
            step_actions = []
            step_log_probs = []
            step_action_tokens = []

            # Sequential action generation (遵循依赖顺序)
            for agent_idx in range(self.flex_env.num_agents):
                # 获取Agent观察 (包含predecessor actions)
                agent_obs = self.flex_env.get_agent_observation(
                    agent_idx, obs, predecessor_actions
                )
                step_obs.append(agent_obs)

                # 获取Expert
                expert = self._get_expert_by_index(agent_idx)

                if expert is not None and hasattr(expert, 'generate_action'):
                    try:
                        # 获取相关前序动作
                        deps = self.flex_env.dependency_graph.get(agent_idx, [])
                        relevant_predecessors = [
                            predecessor_actions[d] for d in deps
                            if d < len(predecessor_actions)
                        ]

                        # 调用Expert生成动作
                        action_dict, action_tokens, raw_response = expert.generate_action(
                            market_obs=agent_obs,
                            predecessor_actions=relevant_predecessors,
                            temperature=0.7,
                        )

                        # 计算log_prob
                        log_prob, _ = expert.get_action_log_prob(
                            agent_obs, action_tokens, relevant_predecessors
                        )

                        step_actions.append(action_dict)
                        step_log_probs.append(log_prob.item())
                        step_action_tokens.append(action_tokens)

                    except Exception as e:
                        logger.warning(f"Expert {agent_idx} generation failed: {e}")
                        # 降级为默认动作
                        action_dict = {"action": "HOLD", "confidence": 0.5}
                        step_actions.append(action_dict)
                        step_log_probs.append(0.0)
                        step_action_tokens.append(torch.tensor([]))
                else:
                    # 无Expert时使用默认动作
                    action_dict = {"action": "HOLD", "confidence": 0.5}
                    step_actions.append(action_dict)
                    step_log_probs.append(0.0)
                    step_action_tokens.append(torch.tensor([]))

                predecessor_actions.append(step_actions[-1])

            # 缓存action tokens
            self.action_tokens_buffer.append(step_action_tokens)

            # 执行联合动作
            next_obs, base_reward, terminated, truncated, info = self.flex_env.step(step_actions)

            # 计算增强奖励
            portfolio_info = info.get("portfolio", {})
            reward, reward_components = self.reward_fn.compute_reward(
                portfolio_return=portfolio_info.get("return", base_reward),
                portfolio_volatility=portfolio_info.get("volatility", 0.0),
                transaction_volume=portfolio_info.get("transaction_volume", 0.0),
                portfolio_weights=np.array(list(portfolio_info.get("weights", {}).values()) or [0.2] * 5),
                max_drawdown=portfolio_info.get("max_drawdown", 0.0),
            )

            # 获取value估计
            values = self._estimate_values(step_obs)

            # 存入buffer
            self.buffer.insert(
                obs=step_obs,
                actions=step_actions,
                log_probs=step_log_probs,
                reward=reward,
                value=values,
                done=terminated or truncated,
            )

            episode_rewards.append(reward)
            current_episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                episode_returns.append(current_episode_reward)
                current_episode_reward = 0.0
                self.episode_count += 1
                obs, info = self.flex_env.reset()

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "total_reward": np.sum(episode_rewards),
            "num_steps": num_steps,
            "episode_returns": episode_returns,
            "episodes_completed": len(episode_returns),
        }

    def _estimate_values(self, obs_list: List[str]) -> List[float]:
        """估计状态价值"""
        if self.critic is None:
            return [0.0] * len(obs_list)

        try:
            if hasattr(self.critic, 'get_value'):
                with torch.no_grad():
                    values = self.critic.get_value(obs_list)
                    if isinstance(values, torch.Tensor):
                        # 扩展到所有agents
                        if values.dim() == 0:
                            return [values.item()] * len(obs_list)
                        return values.tolist()
            return [0.0] * len(obs_list)
        except Exception as e:
            logger.warning(f"Value estimation failed: {e}")
            return [0.0] * len(obs_list)

    def train_step(self) -> Dict[str, float]:
        """执行一步训练"""
        # 获取最后状态的value估计
        if self.buffer.observations:
            last_obs = self.buffer.observations[-1]
            next_value = self._estimate_values(last_obs)
        else:
            next_value = [0.0] * self.flex_env.num_agents

        # PPO更新
        stats = self.trainer.train(
            self.buffer,
            next_value,
            self.global_step,
            action_tokens_buffer=self.action_tokens_buffer,
        )

        # 清空buffer
        self.buffer.clear()
        self.action_tokens_buffer = []

        return stats

    def run_training(
        self,
        num_env_steps: int,
        rollout_length: int = 256,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
        save_dir: str = "./checkpoints/marft",
    ):
        """
        运行完整训练循环

        Args:
            num_env_steps: 总环境步数
            rollout_length: 每次rollout的步数
            log_interval: 日志间隔
            save_interval: 保存间隔
            eval_interval: 评估间隔
            save_dir: 保存目录
        """
        num_updates = num_env_steps // rollout_length

        logger.info(f"Starting MARFT training for {num_env_steps} steps ({num_updates} updates)")
        logger.info(f"Config: rollout_length={rollout_length}, ppo_epoch={self.trainer.ppo_epoch}")

        for update in range(num_updates):
            # 收集rollout
            rollout_stats = self.collect_rollout(rollout_length)
            self.global_step += rollout_length

            # 训练
            train_stats = self.train_step()

            # 日志
            if update % log_interval == 0:
                mean_reward = rollout_stats['mean_reward']
                policy_loss = train_stats.get('policy_loss', 0.0)
                entropy = train_stats.get('entropy', 0.0)

                logger.info(
                    f"Update {update}/{num_updates} | "
                    f"Step {self.global_step} | "
                    f"Reward: {mean_reward:.4f} | "
                    f"Policy Loss: {policy_loss:.4f} | "
                    f"Entropy: {entropy:.4f}"
                )

            # 保存最佳模型
            mean_reward = rollout_stats['mean_reward']
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                best_dir = os.path.join(save_dir, "best")
                self.trainer.save_checkpoint(best_dir, self.global_step)
                logger.info(f"New best reward: {self.best_reward:.4f}")

            # 定期保存
            if update % save_interval == 0 and update > 0:
                checkpoint_dir = os.path.join(save_dir, f"step_{self.global_step}")
                self.trainer.save_checkpoint(checkpoint_dir, self.global_step)

        # 最终保存
        final_dir = os.path.join(save_dir, "final")
        self.trainer.save_checkpoint(final_dir, self.global_step)
        logger.info(f"Training completed! Best reward: {self.best_reward:.4f}")

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """评估当前策略"""
        # 设置为评估模式
        for expert in self.experts.values() if isinstance(self.experts, dict) else []:
            if hasattr(expert, 'eval'):
                expert.eval()

        episode_rewards = []
        episode_lengths = []

        for ep in range(num_episodes):
            obs, info = self.flex_env.reset()
            ep_reward = 0.0
            ep_length = 0

            done = False
            while not done:
                predecessor_actions = []
                step_actions = []

                for agent_idx in range(self.flex_env.num_agents):
                    agent_obs = self.flex_env.get_agent_observation(
                        agent_idx, obs, predecessor_actions
                    )

                    expert = self._get_expert_by_index(agent_idx)

                    if expert is not None and hasattr(expert, 'generate_action'):
                        try:
                            deps = self.flex_env.dependency_graph.get(agent_idx, [])
                            relevant_predecessors = [
                                predecessor_actions[d] for d in deps
                                if d < len(predecessor_actions)
                            ]

                            action_dict, _, _ = expert.generate_action(
                                market_obs=agent_obs,
                                predecessor_actions=relevant_predecessors,
                                temperature=0.1,  # 评估时使用低温度
                                do_sample=False,
                            )
                            step_actions.append(action_dict)
                        except Exception as e:
                            logger.warning(f"Expert {agent_idx} evaluation failed: {e}")
                            step_actions.append({"action": "HOLD", "confidence": 0.5})
                    else:
                        step_actions.append({"action": "HOLD", "confidence": 0.5})

                    predecessor_actions.append(step_actions[-1])

                obs, reward, terminated, truncated, info = self.flex_env.step(step_actions)
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        # 恢复训练模式
        for expert in self.experts.values() if isinstance(self.experts, dict) else []:
            if hasattr(expert, 'train'):
                expert.train()

        return {
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "num_episodes": num_episodes,
        }


# ============================================================
# 7. Configuration Template
# ============================================================

DEFAULT_MARFT_FINSAGE_CONFIG = {
    # Environment
    "gamma": 0.99,
    "episode_length": 252,  # Trading days per year

    # GAE
    "gae_lambda": 0.95,

    # PPO
    "clip_param": 0.2,
    "ppo_epoch": 5,
    "num_mini_batch": 4,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "kl_threshold": 0.01,

    # Learning rate
    "lr": 5e-7,           # Policy LR (low for LLM fine-tuning)
    "critic_lr": 5e-4,    # Critic LR

    # Training
    "agent_iteration_interval": 0,  # 0 = concurrent training
    "rollout_length": 256,
    "num_env_steps": 1000000,

    # Reward
    "risk_penalty_coef": 0.5,
    "transaction_cost_rate": 0.001,

    # Logging & Saving
    "log_interval": 10,
    "save_interval": 100,
}


if __name__ == "__main__":
    # 示例用法
    print("MARFT-FinSage Integration Module")
    print("=" * 50)
    print("\nAgent Profiles:")
    for i, profile in enumerate(FINSAGE_AGENT_PROFILES):
        print(f"  {i+1}. {profile['role']} ({profile['asset_class']})")
        print(f"     Dependencies: {profile['dependencies']}")

    print("\nDefault Config:")
    for key, value in DEFAULT_MARFT_FINSAGE_CONFIG.items():
        print(f"  {key}: {value}")

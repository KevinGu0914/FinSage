#!/usr/bin/env python3
"""
PPO Training with Shared Model Multi-Expert Architecture

使用SharedModelExpertManager实现5个Expert的PPO训练:
- 1个基础模型 (~65GB)
- 5个LoRA适配器 (共~80MB)
- 顺序依赖训练 (Stock->Bond->Commodity/REITs->Crypto)

Usage:
    python scripts/train_shared_experts.py --num_rounds 100 --save_interval 20
"""

import os
import sys
import torch
import numpy as np
import random
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Market Scenario Generator
# ============================================================

@dataclass
class MarketScenario:
    """市场情景"""
    date: str
    observation: str
    target_actions: Dict[str, str]  # role -> expected action
    market_regime: str  # bull/bear/sideways


def generate_market_scenarios(num_scenarios: int = 100) -> List[MarketScenario]:
    """生成多样化的市场情景用于训练"""
    scenarios = []

    for i in range(num_scenarios):
        date = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"

        # 随机市场状态
        market_regime = random.choice(["bull", "bear", "sideways"])

        # 根据市场状态生成数据
        if market_regime == "bull":
            spy_change = random.uniform(0.5, 3.0)
            vix = random.uniform(10, 18)
            btc_change = random.uniform(2, 8)
            sentiment = "乐观"
            stock_target = random.choice(["BUY_50%", "BUY_75%"])
            bond_target = random.choice(["SELL_25%", "HOLD"])
            commodity_target = random.choice(["BUY_25%", "BUY_50%"])
            crypto_target = random.choice(["BUY_50%", "BUY_75%"])
        elif market_regime == "bear":
            spy_change = random.uniform(-3.0, -0.5)
            vix = random.uniform(25, 45)
            btc_change = random.uniform(-8, -2)
            sentiment = "恐慌"
            stock_target = random.choice(["SELL_50%", "SELL_75%"])
            bond_target = random.choice(["BUY_25%", "BUY_50%"])
            commodity_target = random.choice(["BUY_25%", "HOLD"])  # 避险
            crypto_target = random.choice(["SELL_50%", "SELL_75%"])
        else:  # sideways
            spy_change = random.uniform(-1.0, 1.0)
            vix = random.uniform(15, 22)
            btc_change = random.uniform(-3, 3)
            sentiment = "中性"
            stock_target = "HOLD"
            bond_target = "HOLD"
            commodity_target = "HOLD"
            crypto_target = "HOLD"

        # 生成完整观察
        spy_price = random.uniform(380, 480)
        spy_rsi = 30 + spy_change * 10 + random.uniform(-10, 10)
        spy_rsi = max(20, min(80, spy_rsi))

        tlt_price = random.uniform(85, 110)
        tlt_change = -spy_change * 0.3 + random.uniform(-0.5, 0.5)
        yield_10y = random.uniform(3.5, 5.0)

        gld_price = random.uniform(170, 200)
        gld_change = random.uniform(-1, 2) if market_regime == "bear" else random.uniform(-1, 1)

        observation = f"""## 市场日期: {date}
## 资产类别: multi-asset

### 股票市场
- SPY: ${spy_price:.2f}, 日涨跌: {spy_change:+.2f}%, RSI: {spy_rsi:.1f}
- QQQ: ${spy_price * 0.85:.2f}, 日涨跌: {spy_change * 1.1:+.2f}%, RSI: {spy_rsi + 3:.1f}

### 债券市场
- TLT: ${tlt_price:.2f}, 日涨跌: {tlt_change:+.2f}%, 10Y Yield: {yield_10y:.2f}%
- LQD: ${tlt_price * 1.15:.2f}, 日涨跌: {tlt_change * 0.5:+.2f}%, Credit Spread: {random.uniform(1.0, 1.8):.2f}%

### 商品市场
- GLD: ${gld_price:.2f}, 日涨跌: {gld_change:+.2f}%
- USO: ${random.uniform(65, 85):.2f}, 日涨跌: {random.uniform(-2, 3):+.2f}%

### 加密货币
- BTC-USD: ${random.uniform(35000, 55000):.0f}, 日涨跌: {btc_change:+.2f}%
- ETH-USD: ${random.uniform(2000, 3500):.0f}, 日涨跌: {btc_change * 1.2:+.2f}%

### 宏观环境
- VIX: {vix:.1f}
- 美元指数: {random.uniform(100, 108):.1f}
- 市场情绪: {sentiment}
"""

        target_actions = {
            "Stock_Expert": stock_target,
            "Bond_Expert": bond_target,
            "Commodity_Expert": commodity_target,
            "REITs_Expert": stock_target,  # REITs跟随股票
            "Crypto_Expert": crypto_target,
        }

        scenarios.append(MarketScenario(
            date=date,
            observation=observation,
            target_actions=target_actions,
            market_regime=market_regime,
        ))

    return scenarios


# ============================================================
# Reward Functions
# ============================================================

def compute_expert_reward(
    action_dict: Dict,
    target_action: str,
    market_regime: str,
) -> float:
    """
    计算单个Expert的奖励

    奖励设计:
    1. 动作匹配奖励 (0.0 - 1.0)
    2. 信心度校准奖励 (-0.2 - 0.2)
    3. 推理质量奖励 (0.0 - 0.3)
    """
    actual_action = action_dict.get("action", "HOLD")
    confidence = action_dict.get("confidence", 0.5)
    reasoning = action_dict.get("reasoning", "")

    reward = 0.0

    # 1. 动作匹配奖励
    if actual_action == target_action:
        reward += 1.0
    elif _actions_same_direction(actual_action, target_action):
        reward += 0.5
    elif actual_action == "HOLD":
        reward += 0.2
    else:
        reward -= 0.5

    # 2. 信心度校准
    if actual_action == target_action:
        # 正确时高信心更好
        reward += (confidence - 0.5) * 0.4
    else:
        # 错误时低信心更好
        reward -= (confidence - 0.5) * 0.4

    # 3. 推理质量 (长度和关键词)
    if len(reasoning) > 50:
        reward += 0.1
    if any(kw in reasoning for kw in ["RSI", "VIX", "收益率", "趋势", "风险"]):
        reward += 0.1
    if any(kw in reasoning for kw in ["考虑", "建议", "分析", "因此"]):
        reward += 0.1

    return reward


def _actions_same_direction(action1: str, action2: str) -> bool:
    """判断两个动作方向是否一致"""
    buy_actions = {"BUY_25%", "BUY_50%", "BUY_75%", "BUY_100%"}
    sell_actions = {"SELL_25%", "SELL_50%", "SELL_75%", "SELL_100%"}

    if action1 in buy_actions and action2 in buy_actions:
        return True
    if action1 in sell_actions and action2 in sell_actions:
        return True
    return False


# ============================================================
# PPO Trainer
# ============================================================

class SharedExpertPPOTrainer:
    """使用PPO训练共享模型的多Expert系统"""

    def __init__(
        self,
        manager,
        lr: float = 5e-6,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.manager = manager
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # 为每个Expert创建optimizer
        self.optimizers = {}
        for role in ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]:
            self.manager.switch_expert(role)
            params = list(self.manager.parameters(role))
            self.optimizers[role] = torch.optim.AdamW(params, lr=lr)

        # 简单的Critic (共享)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        ).to(manager.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def train_step(
        self,
        scenarios: List[MarketScenario],
    ) -> Dict[str, float]:
        """
        执行一个训练步骤

        Args:
            scenarios: 训练场景列表

        Returns:
            训练统计
        """
        expert_order = ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]

        total_rewards = {role: 0.0 for role in expert_order}
        total_correct = {role: 0 for role in expert_order}
        total_loss = 0.0

        for scenario in scenarios:
            # 清理GPU缓存
            torch.cuda.empty_cache()
            all_actions = {}
            all_tokens = {}
            all_log_probs = {}

            # 1. Rollout: 按依赖顺序生成动作
            for role in expert_order:
                deps = self.manager.expert_configs[role].get("dependencies", [])
                predecessor_actions = {d: all_actions[d] for d in deps if d in all_actions}

                action_dict, tokens, _ = self.manager.generate_action(
                    role=role,
                    market_obs=scenario.observation,
                    predecessor_actions=predecessor_actions if predecessor_actions else None,
                )

                # 计算log_prob
                log_prob, entropy = self.manager.get_action_log_prob(
                    role=role,
                    obs=scenario.observation,
                    action_tokens=tokens,
                    predecessor_actions=predecessor_actions if predecessor_actions else None,
                )

                all_actions[role] = action_dict
                all_tokens[role] = tokens.detach().cpu()  # 移到CPU节省显存
                all_log_probs[role] = (log_prob.detach().item(), entropy.detach().item())

            # 2. 计算奖励
            rewards = {}
            for role in expert_order:
                reward = compute_expert_reward(
                    action_dict=all_actions[role],
                    target_action=scenario.target_actions[role],
                    market_regime=scenario.market_regime,
                )
                rewards[role] = reward
                total_rewards[role] += reward

                if all_actions[role].get("action") == scenario.target_actions[role]:
                    total_correct[role] += 1

            # 3. PPO更新 (简化版 - 使用REINFORCE)
            for role in expert_order:
                reward = rewards[role]
                old_log_prob, old_entropy = all_log_probs[role]  # 这些是Python floats

                # 重新计算log_prob用于梯度
                deps = self.manager.expert_configs[role].get("dependencies", [])
                predecessor_actions = {d: all_actions[d] for d in deps if d in all_actions}

                new_log_prob, new_entropy = self.manager.get_action_log_prob(
                    role=role,
                    obs=scenario.observation,
                    action_tokens=all_tokens[role].to(self.manager.device),
                    predecessor_actions=predecessor_actions if predecessor_actions else None,
                )

                # Policy gradient loss
                advantage = torch.tensor(reward, device=self.manager.device)
                policy_loss = -new_log_prob * advantage - self.entropy_coef * new_entropy

                # 更新
                self.optimizers[role].zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.manager.parameters(role)),
                    self.max_grad_norm
                )
                self.optimizers[role].step()

                total_loss += policy_loss.item()

                # 清理显存
                del new_log_prob, new_entropy, policy_loss
                torch.cuda.empty_cache()

        # 计算统计
        n = len(scenarios)
        stats = {
            "total_loss": total_loss / (n * len(expert_order)),
        }
        for role in expert_order:
            stats[f"{role}_reward"] = total_rewards[role] / n
            stats[f"{role}_accuracy"] = total_correct[role] / n

        return stats


# ============================================================
# Main Training Loop
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--num_rounds", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每轮场景数")
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    parser.add_argument("--save_dir", default="/root/checkpoints/shared_ppo")
    args = parser.parse_args()

    print("=" * 80)
    print(" Shared Model Multi-Expert PPO Training")
    print("=" * 80)
    print(f" Model: {args.model}")
    print(f" Rounds: {args.num_rounds}")
    print(f" Batch Size: {args.batch_size}")
    print(f" Learning Rate: {args.lr}")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 导入SharedModelExpertManager
    from finsage.rl.shared_expert_manager import SharedModelExpertManager

    # 初始化管理器
    print("\nInitializing SharedModelExpertManager...")
    manager = SharedModelExpertManager(
        model_path=args.model,
        device="cuda:0",
        bf16=True,
    )

    print(f"GPU Memory after init: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 创建训练器
    print("\nCreating PPO Trainer...")
    trainer = SharedExpertPPOTrainer(
        manager=manager,
        lr=args.lr,
    )

    # 生成训练场景
    print("\nGenerating training scenarios...")
    all_scenarios = generate_market_scenarios(args.num_rounds * args.batch_size * 2)
    print(f"Generated {len(all_scenarios)} scenarios")

    # 训练循环
    print("\n" + "=" * 80)
    print(" Starting Training")
    print("=" * 80)

    start_time = datetime.now()

    for round_idx in range(args.num_rounds):
        # 随机采样batch
        batch_scenarios = random.sample(all_scenarios, args.batch_size)

        # 训练步骤
        stats = trainer.train_step(batch_scenarios)

        # 日志
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = (round_idx + 1) * args.batch_size / elapsed

            print(f"\nRound {round_idx + 1}/{args.num_rounds}")
            print(f"  Loss: {stats['total_loss']:.4f}")
            print(f"  Speed: {speed:.2f} samples/s")
            print("  Expert Rewards:")
            for role in ["Stock_Expert", "Bond_Expert", "Commodity_Expert", "REITs_Expert", "Crypto_Expert"]:
                reward = stats[f"{role}_reward"]
                acc = stats[f"{role}_accuracy"]
                print(f"    {role}: reward={reward:.3f}, acc={acc:.1%}")

            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # 保存检查点
        if (round_idx + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.save_dir, f"round_{round_idx + 1}")
            manager.save_all_adapters(checkpoint_dir)
            print(f"\n>>> Saved checkpoint to {checkpoint_dir}")

    # 最终保存
    final_dir = os.path.join(args.save_dir, "final")
    manager.save_all_adapters(final_dir)

    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 80)
    print(" Training Complete!")
    print("=" * 80)
    print(f" Total Rounds: {args.num_rounds}")
    print(f" Total Samples: {args.num_rounds * args.batch_size}")
    print(f" Total Time: {total_time:.1f}s")
    print(f" Final Checkpoint: {final_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

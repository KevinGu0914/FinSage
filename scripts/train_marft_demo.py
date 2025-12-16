#!/usr/bin/env python
"""
MARFT Training Demo Script

简化版训练脚本，用于测试完整训练流程
支持两种模式:
1. Mock模式: 不加载真实LLM，用于测试流程
2. Full模式: 加载真实LLM进行训练
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MARFT Training Demo")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "full"],
                        help="Training mode: mock (no LLM) or full (with LLM)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model name for full mode")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--rollout_length", type=int, default=32, help="Rollout length")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/demo")
    return parser.parse_args()


class MockExpert:
    """Mock Expert for testing (no LLM required)"""

    def __init__(self, role: str, asset_class: str, device: str = "cpu"):
        self.role = role
        self.asset_class = asset_class
        self.device = device

        # Simple MLP as mock policy
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13),  # 13 actions
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

    def generate_action(self, obs_embedding: torch.Tensor):
        """Generate action from observation embedding"""
        with torch.no_grad():
            logits = self.policy(obs_embedding)
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs, 1)
        return action_idx, logits

    def get_log_prob(self, obs_embedding: torch.Tensor, action_idx: torch.Tensor):
        """Compute log probability of action"""
        logits = self.policy(obs_embedding)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(-1, action_idx)
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(-1)
        return action_log_prob, entropy


class MockCritic:
    """Mock Critic for testing"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.network = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def get_value(self, obs_embedding: torch.Tensor):
        return self.network(obs_embedding).squeeze(-1)


class MockEnvironment:
    """Mock Environment for testing"""

    def __init__(self, num_assets: int = 10):
        self.num_assets = num_assets
        self.step_count = 0
        self.portfolio_value = 1_000_000

    def reset(self):
        self.step_count = 0
        self.portfolio_value = 1_000_000
        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        # Random market movement
        market_return = np.random.normal(0.0005, 0.02)
        self.portfolio_value *= (1 + market_return)

        reward = market_return * 100  # Scale reward
        done = self.step_count >= 252  # One year

        return self._get_obs(), reward, done, {"portfolio_value": self.portfolio_value}

    def _get_obs(self):
        """Return random observation embedding"""
        return torch.randn(64)


def train_mock_mode(args):
    """训练Mock模式 - 测试流程"""

    logger.info("=" * 60)
    logger.info(" MARFT Training Demo (Mock Mode)")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    # Create mock components
    expert_configs = [
        ("Stock_Expert", "stocks"),
        ("Bond_Expert", "bonds"),
        ("Commodity_Expert", "commodities"),
        ("REITs_Expert", "reits"),
        ("Crypto_Expert", "crypto"),
    ]

    experts = [MockExpert(role, asset_class, device) for role, asset_class in expert_configs]
    critic = MockCritic(device)
    env = MockEnvironment()

    logger.info(f"Created {len(experts)} mock experts and critic")

    # Training config
    gamma = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    ppo_epochs = 4

    # Training loop
    total_steps = 0
    episode_rewards = []

    logger.info(f"\nStarting training for {args.steps} steps...")
    logger.info("-" * 60)

    while total_steps < args.steps:
        # Collect rollout
        obs = env.reset().to(device)
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_values = []
        rollout_log_probs = []
        rollout_dones = []

        episode_reward = 0

        for _ in range(args.rollout_length):
            rollout_obs.append(obs)

            # Get value estimate
            value = critic.get_value(obs.unsqueeze(0))
            rollout_values.append(value.item())

            # Sequential expert actions (simplified)
            actions = []
            log_probs = []
            for expert in experts:
                action_idx, _ = expert.generate_action(obs.unsqueeze(0))
                action_log_prob, _ = expert.get_log_prob(obs.unsqueeze(0), action_idx)
                actions.append(action_idx.item())
                log_probs.append(action_log_prob.item())

            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)

            # Environment step
            next_obs, reward, done, info = env.step(actions)
            next_obs = next_obs.to(device)

            rollout_rewards.append(reward)
            rollout_dones.append(done)
            episode_reward += reward

            obs = next_obs
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                obs = env.reset().to(device)
                episode_reward = 0

        # Compute GAE
        rollout_values = np.array(rollout_values)
        rollout_rewards = np.array(rollout_rewards)
        rollout_dones = np.array(rollout_dones)

        next_value = critic.get_value(obs.unsqueeze(0)).item()

        advantages = np.zeros_like(rollout_rewards)
        returns = np.zeros_like(rollout_rewards)
        gae = 0

        for t in reversed(range(len(rollout_rewards))):
            if t == len(rollout_rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - float(rollout_dones[t])
            else:
                next_val = rollout_values[t + 1]
                next_non_terminal = 1.0 - float(rollout_dones[t])

            delta = rollout_rewards[t] + gamma * next_val * next_non_terminal - rollout_values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + rollout_values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(ppo_epochs):
            for i in range(len(rollout_obs)):
                obs_tensor = rollout_obs[i].unsqueeze(0)

                # Update critic
                value = critic.get_value(obs_tensor)
                value_loss = 0.5 * (value - torch.tensor([returns[i]], device=device)).pow(2).mean()

                critic.optimizer.zero_grad()
                value_loss.backward()
                critic.optimizer.step()

                # Update each expert
                for j, expert in enumerate(experts):
                    action_idx = torch.tensor([[rollout_actions[i][j]]], device=device)
                    old_log_prob = rollout_log_probs[i][j]

                    new_log_prob, entropy = expert.get_log_prob(obs_tensor, action_idx)

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv = torch.tensor([advantages[i]], device=device)

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv
                    policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    expert.optimizer.zero_grad()
                    policy_loss.backward()
                    expert.optimizer.step()

        # Logging
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Step {total_steps:6d} | Avg Reward: {avg_reward:8.2f} | Episodes: {len(episode_rewards)}")

    logger.info("-" * 60)
    logger.info(f"Training completed!")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        logger.info(f"Final avg reward: {np.mean(episode_rewards[-10:]):.2f}")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint = {
        "experts": [e.policy.state_dict() for e in experts],
        "critic": critic.network.state_dict(),
        "total_steps": total_steps,
    }
    torch.save(checkpoint, os.path.join(args.save_dir, "mock_checkpoint.pt"))
    logger.info(f"Checkpoint saved to {args.save_dir}")


def train_full_mode(args):
    """训练Full模式 - 真实LLM"""

    logger.info("=" * 60)
    logger.info(" MARFT Training (Full Mode with LLM)")
    logger.info("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Full mode requires GPU.")
        logger.info("Please use --mode mock for CPU testing.")
        return

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        from finsage.rl.lora_expert import LoRAExpert, create_finsage_expert_profiles
        from finsage.rl.critic import ActionCritic
        from finsage.rl.config import MARFTFinSageConfig
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please install: pip install transformers peft accelerate")
        return

    # Load config
    config = MARFTFinSageConfig()
    config.model.model_name_or_path = args.model

    logger.info(f"Loading model: {args.model}")
    logger.info("This may take a few minutes...")

    # Create experts
    profiles = create_finsage_expert_profiles()

    # For demo, just load one expert to test
    logger.info("Creating Stock Expert...")
    expert = LoRAExpert(
        model_path=args.model,
        profile=profiles[0],
        device="cuda:0",
        load_in_4bit=True,  # Use 4-bit to save memory
    )

    logger.info("Expert created successfully!")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in expert.parameters() if p.requires_grad):,}")

    # Test generation
    logger.info("\nTesting generation...")
    test_obs = "## 市场日期: 2024-01-15\n当前SPY价格: $450, RSI: 55"
    action, tokens, response = expert.generate_action(test_obs)
    logger.info(f"Generated action: {action.get('action', 'N/A')}")
    logger.info(f"Response preview: {response[:100]}...")

    logger.info("\n✓ Full mode test successful!")
    logger.info("To run full training, please use the main training script:")
    logger.info("  python scripts/run_marft_training.py --num_env_steps 100000")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f" MARFT-FinSage Training Demo")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f" Mode: {args.mode}")
    print(f" Steps: {args.steps}")
    print(f" Device: {args.device}")
    print(f"{'='*60}\n")

    if args.mode == "mock":
        train_mock_mode(args)
    else:
        train_full_mode(args)


if __name__ == "__main__":
    main()

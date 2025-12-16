#!/usr/bin/env python
"""
MARFT Training Script for FinSage

使用MARFT框架对FinSage的Expert Agents进行强化学习微调

Usage:
    python scripts/run_marft_training.py --config config/marft_config.yaml
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finsage.rl.marft_integration import (
    MARFTFinSageIntegration,
    DEFAULT_MARFT_FINSAGE_CONFIG,
    FINSAGE_AGENT_PROFILES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MARFT Training for FinSage")

    # 基础参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")

    # 环境参数
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Training start date")
    parser.add_argument("--end_date", type=str, default="2023-12-31", help="Training end date")
    parser.add_argument("--initial_capital", type=float, default=1000000, help="Initial capital")

    # 训练参数
    parser.add_argument("--num_env_steps", type=int, default=1000000, help="Total environment steps")
    parser.add_argument("--rollout_length", type=int, default=256, help="Rollout length")
    parser.add_argument("--ppo_epoch", type=int, default=5, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=5e-7, help="Policy learning rate")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")

    # Agent训练策略
    parser.add_argument(
        "--agent_iteration_interval",
        type=int,
        default=0,
        help="Agent iteration interval (0=concurrent, >0=sequential)"
    )

    # 保存/加载
    parser.add_argument("--save_dir", type=str, default="./checkpoints/marft", help="Checkpoint directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load from checkpoint")

    # LLM参数
    parser.add_argument("--llm_provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--llm_model", type=str, default="gpt-4", help="LLM model name")

    return parser.parse_args()


def setup_environment(args):
    """设置交易环境"""
    from finsage.environment.multi_asset_env import MultiAssetTradingEnv

    # 资产配置
    asset_config = {
        "stocks": ["SPY", "QQQ", "IWM", "VTI"],
        "bonds": ["TLT", "IEF", "LQD", "HYG"],
        "commodities": ["GLD", "SLV", "USO", "DBA"],
        "reits": ["VNQ", "IYR", "XLRE"],
        "crypto": ["BTC-USD", "ETH-USD"],
    }

    env = MultiAssetTradingEnv(
        asset_config=asset_config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )

    return env


def setup_orchestrator(args):
    """设置FinSage Orchestrator"""
    from finsage.core.orchestrator import FinSageOrchestrator

    config = {
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "max_single_weight": 0.15,
        "risk_free_rate": 0.04,
    }

    orchestrator = FinSageOrchestrator(
        config=config,
        checkpoint_dir=args.save_dir,
    )

    return orchestrator


def main():
    args = get_args()

    # 设置随机种子
    import numpy as np
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("MARFT-FinSage Training")
    logger.info("=" * 60)

    # 打印配置
    logger.info("\nConfiguration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # 设置环境
    logger.info("\nSetting up environment...")
    try:
        env = setup_environment(args)
        logger.info(f"Environment created: {args.start_date} to {args.end_date}")
    except Exception as e:
        logger.warning(f"Could not create real environment: {e}")
        logger.info("Using mock environment for demonstration")
        env = None

    # 设置Orchestrator
    logger.info("\nSetting up orchestrator...")
    try:
        orchestrator = setup_orchestrator(args)
        logger.info(f"Orchestrator created with {len(orchestrator.experts)} experts")
    except Exception as e:
        logger.warning(f"Could not create real orchestrator: {e}")
        logger.info("Using mock orchestrator for demonstration")
        orchestrator = None

    # 构建训练配置
    train_config = {
        **DEFAULT_MARFT_FINSAGE_CONFIG,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "ppo_epoch": args.ppo_epoch,
        "lr": args.lr,
        "critic_lr": args.critic_lr,
        "agent_iteration_interval": args.agent_iteration_interval,
        "rollout_length": args.rollout_length,
        "num_env_steps": args.num_env_steps,
    }

    # 创建整合模块
    if env is not None and orchestrator is not None:
        logger.info("\nInitializing MARFT-FinSage Integration...")
        integration = MARFTFinSageIntegration(
            base_orchestrator=orchestrator,
            env=env,
            config=train_config,
        )

        # 开始训练
        logger.info("\nStarting training...")
        integration.run_training(
            num_env_steps=args.num_env_steps,
            rollout_length=args.rollout_length,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            save_dir=args.save_dir,
        )
    else:
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION MODE")
        logger.info("=" * 60)
        logger.info("\nTo run actual training, ensure all dependencies are installed:")
        logger.info("  1. FinSage environment and orchestrator are properly configured")
        logger.info("  2. LLM API keys are set (e.g., OPENAI_API_KEY)")
        logger.info("  3. Market data is available")
        logger.info("\nAgent Profiles configured:")
        for i, profile in enumerate(FINSAGE_AGENT_PROFILES):
            deps = profile.get("dependencies", [])
            logger.info(f"  {i+1}. {profile['role']}")
            logger.info(f"     Asset Class: {profile['asset_class']}")
            logger.info(f"     Dependencies: {deps if deps else 'None (first in chain)'}")

    logger.info("\n" + "=" * 60)
    logger.info("Training script completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

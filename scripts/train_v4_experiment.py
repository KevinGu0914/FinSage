#!/usr/bin/env python3
"""
MARFT V4 实验训练脚本

支持三种奖励方案的A/B测试:
- aggressive: 激进探索型 (适合牛市，打败XGBoost)
- balanced: 平衡型 (风险收益平衡)
- adaptive: 自适应型 (根据市场状态动态调整)

Usage:
    # 方案A: 激进探索
    python scripts/train_v4_experiment.py --reward_scheme aggressive --save_dir /root/checkpoints/exp_aggressive

    # 方案B: 平衡型
    python scripts/train_v4_experiment.py --reward_scheme balanced --save_dir /root/checkpoints/exp_balanced

    # 方案C: 自适应
    python scripts/train_v4_experiment.py --reward_scheme adaptive --save_dir /root/checkpoints/exp_adaptive
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Force unbuffered output
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Import reward configurations
from finsage.rl.reward_configs import (
    get_reward_scheme,
    compute_modified_expert_reward,
    detect_market_regime,
    RewardSchemeConfig,
    REWARD_SCHEMES,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MARFT V4 Experiment Training")

    # 奖励方案选择
    parser.add_argument("--reward_scheme", type=str, default="balanced",
                        choices=["aggressive", "balanced", "adaptive"],
                        help="Reward scheme to use")

    # 模型配置
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    # 训练配置
    parser.add_argument("--train_start", default="2023-01-01")
    parser.add_argument("--train_end", default="2024-06-30")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--rebalance_freq", type=int, default=5)
    parser.add_argument("--rollout_length", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=10)

    # 保存配置
    parser.add_argument("--save_dir", default="/root/checkpoints/marft_v4_exp")
    parser.add_argument("--experiment_name", default=None,
                        help="Experiment name (default: reward_scheme name)")

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true", default=True)

    args = parser.parse_args()

    # 获取奖励方案配置
    scheme = get_reward_scheme(args.reward_scheme)

    # 设置实验名称
    if args.experiment_name is None:
        args.experiment_name = f"exp_{args.reward_scheme}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # 更新保存目录
    args.save_dir = os.path.join(args.save_dir, args.reward_scheme)
    os.makedirs(args.save_dir, exist_ok=True)

    # 打印实验配置
    flush_print("\n" + "=" * 80)
    flush_print(f" MARFT V4 Experiment: {args.reward_scheme.upper()}")
    flush_print("=" * 80)
    flush_print(f"\n>>> Reward Scheme: {scheme.name}")
    flush_print(f">>> Description: {scheme.description[:200]}...")
    flush_print(f"\n>>> Key Parameters:")
    flush_print(f"    - Wrong direction penalty scale: {scheme.wrong_direction_penalty_scale}")
    flush_print(f"    - Timing penalty scale: {scheme.timing_penalty_scale}")
    flush_print(f"    - Trade bonus: {scheme.trade_bonus}")
    flush_print(f"    - Momentum bonus: {scheme.momentum_bonus}")
    flush_print(f"    - HOLD penalty in uptrend: {scheme.hold_penalty_in_uptrend}")
    flush_print(f"    - Entropy coefficient: {scheme.entropy_coef}")
    flush_print(f"    - Regime adaptive: {scheme.regime_adaptive}")
    flush_print(f"\n>>> Training Period: {args.train_start} ~ {args.train_end}")
    flush_print(f">>> Save Directory: {args.save_dir}")
    flush_print("=" * 80)

    # 保存实验配置
    config_path = os.path.join(args.save_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "reward_scheme": args.reward_scheme,
            "scheme_params": {
                "wrong_direction_penalty_scale": scheme.wrong_direction_penalty_scale,
                "timing_penalty_scale": scheme.timing_penalty_scale,
                "trade_bonus": scheme.trade_bonus,
                "momentum_bonus": scheme.momentum_bonus,
                "hold_penalty_in_uptrend": scheme.hold_penalty_in_uptrend,
                "entropy_coef": scheme.entropy_coef,
                "clip_param": scheme.clip_param,
                "regime_adaptive": scheme.regime_adaptive,
                "max_drawdown_tolerance": scheme.max_drawdown_tolerance,
                "volatility_target": scheme.volatility_target,
            },
            "model": args.model,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "num_epochs": args.num_epochs,
            "experiment_name": args.experiment_name,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    flush_print(f"\n>>> Config saved to: {config_path}")

    # 导入训练脚本的主要组件
    # (这里简化处理，实际需要导入完整的train_with_real_data_v4.py中的类)

    flush_print("\n>>> Starting training with modified reward scheme...")

    # 调用原始训练脚本，传递奖励方案参数
    # 构建命令行参数
    train_args = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train_with_real_data_v4.py"),
        "--model", args.model,
        "--train_start", args.train_start,
        "--train_end", args.train_end,
        "--num_epochs", str(args.num_epochs),
        "--rebalance_freq", str(args.rebalance_freq),
        "--rollout_length", str(args.rollout_length),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--save_dir", args.save_dir,
    ]

    if args.load_in_8bit:
        train_args.append("--load_in_8bit")
    if args.load_in_4bit:
        train_args.append("--load_in_4bit")
    if args.resume:
        train_args.extend(["--resume", args.resume])

    # 设置环境变量传递奖励方案
    os.environ["REWARD_SCHEME"] = args.reward_scheme
    os.environ["ENTROPY_COEF"] = str(scheme.entropy_coef)
    os.environ["CLIP_PARAM"] = str(scheme.clip_param)

    # 执行训练
    import subprocess
    flush_print(f"\n>>> Executing: {' '.join(train_args)}")
    result = subprocess.run(train_args, env=os.environ)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

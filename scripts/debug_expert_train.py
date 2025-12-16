#!/usr/bin/env python3
"""
Debug Expert Training Script
显示每个Expert的详细输入输出，用于调试
"""

import os
import sys
import torch
import logging
from datetime import datetime
from typing import Dict, List
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# 5个Expert的角色定义
EXPERT_PROFILES = [
    {
        "role": "Stock_Expert",
        "asset_class": "stocks",
        "assets": ["SPY", "QQQ", "AAPL", "MSFT"],
        "dependencies": [],
    },
    {
        "role": "Bond_Expert",
        "asset_class": "bonds",
        "assets": ["TLT", "IEF", "LQD"],
        "dependencies": ["Stock_Expert"],
    },
    {
        "role": "Commodity_Expert",
        "asset_class": "commodities",
        "assets": ["GLD", "SLV", "USO"],
        "dependencies": ["Stock_Expert", "Bond_Expert"],
    },
    {
        "role": "REITs_Expert",
        "asset_class": "reits",
        "assets": ["VNQ", "IYR"],
        "dependencies": ["Stock_Expert", "Bond_Expert"],
    },
    {
        "role": "Crypto_Expert",
        "asset_class": "crypto",
        "assets": ["BTC-USD", "ETH-USD"],
        "dependencies": ["Stock_Expert"],
    },
]


def create_market_observation(profile: Dict, date: str) -> str:
    """为特定Expert创建市场观察"""
    asset_class = profile["asset_class"]
    assets = profile["assets"]

    obs = f"""## 市场日期: {date}
## 资产类别: {asset_class}
## 你的角色: {profile['role']}

### 市场数据
"""

    for asset in assets:
        price = random.uniform(50, 500)
        change = random.uniform(-5, 5)
        rsi = random.uniform(20, 80)
        volume = random.uniform(-30, 50)

        obs += f"""
#### {asset}
- 当前价格: ${price:.2f}
- 日涨跌幅: {change:+.2f}%
- RSI(14): {rsi:.1f}
- 成交量变化: {volume:+.1f}%
"""

    # 添加宏观数据
    obs += f"""
### 宏观环境
- VIX: {random.uniform(12, 35):.1f}
- 10年期国债收益率: {random.uniform(3.5, 5.0):.2f}%
- 美元指数: {random.uniform(100, 110):.1f}
- 市场情绪: {"乐观" if random.random() > 0.5 else "谨慎"}

### 任务
请分析当前{asset_class}市场状况，给出你的投资建议。
回复格式要求JSON:
{{"action": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "reasoning": "你的分析"}}
"""

    return obs


def create_observation_with_predecessors(
    profile: Dict,
    date: str,
    predecessor_actions: Dict[str, Dict]
) -> str:
    """创建包含前置Expert决策的观察"""
    base_obs = create_market_observation(profile, date)

    if profile["dependencies"] and predecessor_actions:
        base_obs += "\n### 其他专家观点\n"
        for dep in profile["dependencies"]:
            if dep in predecessor_actions:
                action = predecessor_actions[dep]
                base_obs += f"""
#### {dep} 的建议:
- 动作: {action.get('action', 'N/A')}
- 信心度: {action.get('confidence', 'N/A')}
- 理由: {action.get('reasoning', 'N/A')[:100]}...
"""

    return base_obs


def parse_expert_response(response: str) -> Dict:
    """解析Expert响应"""
    try:
        # 尝试找JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            if "action" in data:
                return data
    except Exception as e:
        pass

    # 尝试从文本中提取
    response_upper = response.upper()
    if "BUY" in response_upper:
        action = "BUY"
    elif "SELL" in response_upper:
        action = "SELL"
    else:
        action = "HOLD"

    return {"action": action, "confidence": 0.5, "reasoning": "Parsed from text"}


def run_single_expert_debug(expert, profile: Dict, date: str, predecessor_actions: Dict):
    """运行单个Expert并显示详细输入输出"""

    # 创建观察
    obs = create_observation_with_predecessors(profile, date, predecessor_actions)

    print(f"\n{'='*80}")
    print(f"Expert: {profile['role']}")
    print(f"{'='*80}")

    print(f"\n--- INPUT (前200字符) ---")
    print(obs[:500] + "..." if len(obs) > 500 else obs)

    # 生成响应
    try:
        action_dict, tokens, response = expert.generate_action(obs)

        print(f"\n--- RAW RESPONSE (前500字符) ---")
        print(response[:500] + "..." if len(response) > 500 else response)

        print(f"\n--- PARSED ACTION ---")
        print(f"Action: {action_dict.get('action', 'N/A')}")
        print(f"Confidence: {action_dict.get('confidence', 'N/A')}")
        print(f"Reasoning: {str(action_dict.get('reasoning', 'N/A'))[:200]}")

        return action_dict, response

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"action": "HOLD", "confidence": 0.5, "error": str(e)}, ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of trading rounds")
    parser.add_argument("--experts", type=str, default="all", help="all or comma-separated: stock,bond,commodity,reits,crypto")
    parser.add_argument("--save_dir", default="/root/checkpoints/debug_experts")
    args = parser.parse_args()

    print("=" * 80)
    print(" Debug Expert Training - 详细输入输出显示")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from finsage.rl.lora_expert import LoRAExpert, create_finsage_expert_profiles

    # 确定要加载哪些Expert
    if args.experts == "all":
        expert_indices = list(range(5))
    else:
        name_map = {"stock": 0, "bond": 1, "commodity": 2, "reits": 3, "crypto": 4}
        expert_indices = [name_map[n.strip().lower()] for n in args.experts.split(",")]

    print(f"\nLoading {len(expert_indices)} experts...")

    # 获取profiles
    profiles = create_finsage_expert_profiles()

    # 加载Experts
    experts = {}
    for idx in expert_indices:
        profile = profiles[idx]
        print(f"\nLoading {profile.role}...")

        expert = LoRAExpert(
            model_path=args.model,
            profile=profile,
            device="cuda:0",
            load_in_4bit=False,
        )
        experts[profile.role] = expert

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  Loaded! GPU Memory: {mem_gb:.1f} GB")

    print(f"\n{'='*80}")
    print(f" All {len(experts)} experts loaded!")
    print(f" GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"{'='*80}")

    # 运行交易轮次
    for round_idx in range(args.num_rounds):
        date = f"2024-01-{(round_idx % 28) + 1:02d}"

        print(f"\n{'#'*80}")
        print(f"# TRADING ROUND {round_idx + 1} - {date}")
        print(f"{'#'*80}")

        predecessor_actions = {}

        # 按依赖顺序运行每个Expert
        for profile in EXPERT_PROFILES:
            role = profile['role']
            if role not in experts:
                continue

            expert = experts[role]

            # 运行Expert并显示详细信息
            action, response = run_single_expert_debug(
                expert, profile, date, predecessor_actions
            )

            predecessor_actions[role] = action

        # 汇总本轮所有决策
        print(f"\n{'='*80}")
        print(f"ROUND {round_idx + 1} SUMMARY")
        print(f"{'='*80}")
        for role, action in predecessor_actions.items():
            print(f"  {role}: {action.get('action', 'N/A')} (conf: {action.get('confidence', 'N/A')})")

    # 保存checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    for role, expert in experts.items():
        save_path = os.path.join(args.save_dir, role)
        expert.save(save_path)
        print(f"\nSaved {role} to {save_path}")

    print(f"\n{'='*80}")
    print(" Debug session complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

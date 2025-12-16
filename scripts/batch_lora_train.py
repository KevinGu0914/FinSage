#!/usr/bin/env python3
"""
批量优化的LoRA训练脚本
使用批量推理提高GPU利用率
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


def create_training_scenarios(num_samples: int = 200) -> List[Dict]:
    """创建训练场景"""
    scenarios = []
    for i in range(num_samples):
        price = random.uniform(400, 500)
        rsi = random.uniform(20, 80)
        ma_50 = price * random.uniform(0.95, 1.05)
        volume_change = random.uniform(-30, 50)

        obs = f"""## Market Date: 2024-01-{(i % 28) + 1:02d}
## Asset Class: stocks

### SPY Analysis
- Price: ${price:.2f}
- RSI(14): {rsi:.1f}
- 50MA: ${ma_50:.2f}
- Volume: {volume_change:+.1f}%
- Volatility: {random.uniform(10, 25):.1f}%

### Market Sentiment
- VIX: {random.uniform(12, 35):.1f}
- Trend: {"Bullish" if price > ma_50 else "Bearish"}

Please provide your trading recommendation in JSON format."""

        # Target action based on technicals
        if rsi < 30 and price < ma_50:
            target = {"action": "BUY", "confidence": 0.8}
        elif rsi > 70 and price > ma_50:
            target = {"action": "SELL", "confidence": 0.7}
        else:
            target = {"action": "HOLD", "confidence": 0.6}

        scenarios.append({"observation": obs, "target": target})

    return scenarios


def batch_generate(model, tokenizer, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
    """批量生成响应"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)

    return responses


def parse_action(response: str) -> Dict:
    """解析响应中的JSON动作"""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            if "action" in data:
                return data
    except Exception:
        pass
    return {"action": "HOLD", "confidence": 0.5}


def compute_batch_rewards(actions: List[Dict], targets: List[Dict]) -> List[float]:
    """计算批量奖励"""
    rewards = []
    for action, target in zip(actions, targets):
        if action.get("action") == target.get("action"):
            rewards.append(1.0)
        elif action.get("action") == "HOLD":
            rewards.append(0.3)
        else:
            rewards.append(-0.5)
    return rewards


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save_dir", default="/root/checkpoints/lora_batch")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(" Batch Optimized LoRA Training")
    logger.info("=" * 60)
    logger.info(f" Batch size: {args.batch_size}")
    logger.info(f" Num batches: {args.num_batches}")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from finsage.rl.lora_expert import LoRAExpert, create_finsage_expert_profiles

    profiles = create_finsage_expert_profiles()
    logger.info(f"Loading model: {args.model}")

    expert = LoRAExpert(
        model_path=args.model,
        profile=profiles[0],
        device="cuda:0",
        load_in_4bit=False,
    )

    trainable = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    logger.info(f"Model loaded! Trainable params: {trainable:,}")

    logger.info("Creating training scenarios...")
    train_data = create_training_scenarios(args.batch_size * args.num_batches * 2)

    model = expert.model
    tokenizer = expert.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(
        [p for p in expert.parameters() if p.requires_grad],
        lr=args.lr
    )

    logger.info("\nStarting batch training...")
    total_reward = 0
    total_correct = 0
    total_samples = 0

    start_time = datetime.now()

    for batch_idx in range(args.num_batches):
        batch_scenarios = random.sample(train_data, args.batch_size)
        prompts = [s["observation"] for s in batch_scenarios]
        targets = [s["target"] for s in batch_scenarios]

        batch_start = datetime.now()
        responses = batch_generate(model, tokenizer, prompts)
        gen_time = (datetime.now() - batch_start).total_seconds()

        actions = [parse_action(r) for r in responses]
        rewards = compute_batch_rewards(actions, targets)

        batch_reward = sum(rewards) / len(rewards)
        batch_correct = sum(1 for a, t in zip(actions, targets) if a.get("action") == t.get("action"))

        total_reward += sum(rewards)
        total_correct += batch_correct
        total_samples += len(rewards)

        throughput = args.batch_size / gen_time
        logger.info(f"Batch {batch_idx+1}/{args.num_batches} | "
                   f"Reward: {batch_reward:.3f} | "
                   f"Acc: {batch_correct}/{args.batch_size} | "
                   f"Speed: {throughput:.2f} samples/s")

        if (batch_idx + 1) % 5 == 0:
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "lora_weights")
    expert.save(save_path)
    logger.info(f"\nLoRA weights saved to: {save_path}")

    total_time = (datetime.now() - start_time).total_seconds()
    avg_reward = total_reward / total_samples
    accuracy = total_correct / total_samples
    throughput = total_samples / total_time

    logger.info("\n" + "=" * 60)
    logger.info(" Training Complete!")
    logger.info(f" Total samples: {total_samples}")
    logger.info(f" Accuracy: {accuracy:.2%}")
    logger.info(f" Avg Reward: {avg_reward:.3f}")
    logger.info(f" Total time: {total_time:.1f}s")
    logger.info(f" Throughput: {throughput:.2f} samples/s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

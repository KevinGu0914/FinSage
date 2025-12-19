#!/bin/bash
# ============================================================
# MARFT V4 三种奖励策略并行训练启动脚本
# ============================================================
#
# 三种策略:
#   - aggressive (激进型): 鼓励大胆交易，适合牛市
#   - balanced (平衡型): 风险收益平衡
#   - adaptive (自适应型): 根据市场状态动态调整
#
# ============================================================

echo "============================================================"
echo " MARFT V4 三种策略训练启动命令"
echo "============================================================"
echo ""
echo "请复制对应命令到各服务器执行:"
echo ""
echo "=== GPU 1: 激进型 (Aggressive) ==="
echo 'cd /root/FinSage && pip install transformers==4.56.1 peft==0.14.0 trl==0.25.0 accelerate bitsandbytes scipy python-dotenv pandas yfinance requests aiohttp tenacity gymnasium -q && export FMP_API_KEY=eUPFhLU16P1EkyfOtBaVUtCzfx17WGmh && export OPENAI_API_KEY=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 && rm -f training_aggressive.log && nohup python3 -u scripts/train_with_real_data_v4.py --model Qwen/Qwen2.5-14B-Instruct --save_dir /root/checkpoints/marft_v4_aggressive --reward_scheme aggressive --log_level DEBUG > training_aggressive.log 2>&1 &'
echo ""
echo "=== GPU 2: 平衡型 (Balanced) ==="
echo 'cd /root/FinSage && pip install transformers==4.56.1 peft==0.14.0 trl==0.25.0 accelerate bitsandbytes scipy python-dotenv pandas yfinance requests aiohttp tenacity gymnasium -q && export FMP_API_KEY=eUPFhLU16P1EkyfOtBaVUtCzfx17WGmh && export OPENAI_API_KEY=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 && rm -f training_balanced.log && nohup python3 -u scripts/train_with_real_data_v4.py --model Qwen/Qwen2.5-14B-Instruct --save_dir /root/checkpoints/marft_v4_balanced --reward_scheme balanced --log_level DEBUG > training_balanced.log 2>&1 &'
echo ""
echo "=== GPU 3: 自适应型 (Adaptive) ==="
echo 'cd /root/FinSage && pip install transformers==4.56.1 peft==0.14.0 trl==0.25.0 accelerate bitsandbytes scipy python-dotenv pandas yfinance requests aiohttp tenacity gymnasium -q && export FMP_API_KEY=eUPFhLU16P1EkyfOtBaVUtCzfx17WGmh && export OPENAI_API_KEY=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 && rm -f training_adaptive.log && nohup python3 -u scripts/train_with_real_data_v4.py --model Qwen/Qwen2.5-14B-Instruct --save_dir /root/checkpoints/marft_v4_adaptive --reward_scheme adaptive --log_level DEBUG > training_adaptive.log 2>&1 &'
echo ""
echo "============================================================"
echo " 监控命令 (在服务器上执行):"
echo "============================================================"
echo "tail -f training_*.log | grep -E 'STEP|EPOCH|ERROR|GPU|显存|MONITOR'"
echo ""
echo "============================================================"
echo " 如果显存不足，添加 --load_in_4bit 参数"
echo "============================================================"

#!/bin/bash
# 评估脚本启动器 V2 (修正参数名 + 每日决策)

CHECKPOINT_NAME=$1
if [ -z "$CHECKPOINT_NAME" ]; then
    echo "Usage: $0 <checkpoint_name>"
    exit 1
fi

cd /root/FinSage

# 从环境变量读取 API 密钥 (不要硬编码!)
# 在服务器上设置: export FMP_API_KEY=xxx
# 在服务器上设置: export OPENAI_API_KEY=xxx
if [ -z "$FMP_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: 请先设置 FMP_API_KEY 和 OPENAI_API_KEY 环境变量"
    exit 1
fi

# 停止旧进程
pkill -f evaluate_trained 2>/dev/null

# 删除旧日志
rm -f eval_v2_${CHECKPOINT_NAME}.log

# 启动评估 (每日决策，不再是每5天)
echo "Starting evaluation for ${CHECKPOINT_NAME} (daily decisions)..."
nohup python3 scripts/evaluate_trained_lora.py \
  --checkpoint ${CHECKPOINT_NAME}_final \
  --test-start 2024-06-03 \
  --test-end 2024-11-29 \
  --capital 1000000 \
  --freq 1 \
  > eval_v2_${CHECKPOINT_NAME}.log 2>&1 &

sleep 3

# 确认启动
RUNNING=$(ps aux | grep evaluate_trained | grep -v grep | wc -l)
echo "Running processes: ${RUNNING}"

if [ "$RUNNING" -gt 0 ]; then
    echo "OK: Evaluation started (daily decisions)"
    tail -5 eval_v2_${CHECKPOINT_NAME}.log 2>/dev/null
else
    echo "ERROR: Evaluation failed to start"
    cat eval_v2_${CHECKPOINT_NAME}.log 2>/dev/null | tail -20
fi

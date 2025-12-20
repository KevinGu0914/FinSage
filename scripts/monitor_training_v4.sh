#!/bin/bash
# ============================================================
# MARFT V4 训练监控脚本 (20小时, 每10分钟)
# ============================================================

LOG_FILE="/Users/guboyang/Desktop/Project/FinSage/results/training_monitor_v4_final.log"
BUG_DOC="/Users/guboyang/Desktop/Project/FinSage/docs/TRAINING_BUGS_V4.md"
CHECKPOINT_DIR="/Users/guboyang/Desktop/Project/FinSage/checkpoints_backup"

# 服务器配置
S1_HOST="174.78.228.101"
S1_PORT="40726"
S2_HOST="49.213.134.9"
S2_PORT="18109"
S3_HOST="173.207.82.240"
S3_PORT="40038"

# 创建checkpoint备份目录
mkdir -p "$CHECKPOINT_DIR"

# 总检查次数 = 20小时 * 6次/小时 = 120次
TOTAL_CHECKS=120
CHECK_INTERVAL=600  # 10分钟 = 600秒

echo "开始监控 - $(date)" >> "$LOG_FILE"

for i in $(seq 1 $TOTAL_CHECKS); do
    TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
    echo "" >> "$LOG_FILE"
    echo "========== 检查 #$i / $TOTAL_CHECKS - $TIMESTAMP ==========" >> "$LOG_FILE"

    # 检查 S1
    echo "S1-AGGRESSIVE:" >> "$LOG_FILE"
    S1_INFO=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $S1_PORT root@$S1_HOST "
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        ps aux | grep -c '[t]rain_with_real'
        wc -l /root/FinSage/training_aggressive.log 2>/dev/null | awk '{print \$1}'
        grep -c 'WARNING\|ERROR' /root/FinSage/training_aggressive.log 2>/dev/null || echo 0
    " 2>/dev/null)
    echo "$S1_INFO" >> "$LOG_FILE"

    # 检查进程是否停止
    S1_PROC=$(echo "$S1_INFO" | sed -n '2p')
    if [ "$S1_PROC" = "0" ]; then
        echo "!!! S1 训练进程已停止 !!!" >> "$LOG_FILE"
    fi

    # 检查 S2
    echo "S2-BALANCED:" >> "$LOG_FILE"
    S2_INFO=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $S2_PORT root@$S2_HOST "
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        ps aux | grep -c '[t]rain_with_real'
        wc -l /root/FinSage/training_balanced.log 2>/dev/null | awk '{print \$1}'
        grep -c 'WARNING\|ERROR' /root/FinSage/training_balanced.log 2>/dev/null || echo 0
    " 2>/dev/null)
    echo "$S2_INFO" >> "$LOG_FILE"

    S2_PROC=$(echo "$S2_INFO" | sed -n '2p')
    if [ "$S2_PROC" = "0" ]; then
        echo "!!! S2 训练进程已停止 !!!" >> "$LOG_FILE"
    fi

    # 检查 S3
    echo "S3-ADAPTIVE:" >> "$LOG_FILE"
    S3_INFO=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $S3_PORT root@$S3_HOST "
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        ps aux | grep -c '[t]rain_with_real'
        wc -l /root/FinSage/training_adaptive.log 2>/dev/null | awk '{print \$1}'
        grep -c 'WARNING\|ERROR' /root/FinSage/training_adaptive.log 2>/dev/null || echo 0
    " 2>/dev/null)
    echo "$S3_INFO" >> "$LOG_FILE"

    S3_PROC=$(echo "$S3_INFO" | sed -n '2p')
    if [ "$S3_PROC" = "0" ]; then
        echo "!!! S3 训练进程已停止 !!!" >> "$LOG_FILE"
    fi

    # 每30分钟备份一次checkpoint (每3次检查)
    if [ $((i % 3)) -eq 0 ]; then
        echo "--- 备份 Checkpoint ---" >> "$LOG_FILE"
        BACKUP_TIME=$(date +"%Y%m%d_%H%M")

        # 备份 S1 checkpoint
        scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $S1_PORT -r \
            root@$S1_HOST:/root/checkpoints/marft_v4_aggressive \
            "$CHECKPOINT_DIR/aggressive_$BACKUP_TIME" 2>/dev/null && \
            echo "S1 checkpoint 已备份" >> "$LOG_FILE"

        # 备份 S2 checkpoint
        scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $S2_PORT -r \
            root@$S2_HOST:/root/checkpoints/marft_v4_balanced \
            "$CHECKPOINT_DIR/balanced_$BACKUP_TIME" 2>/dev/null && \
            echo "S2 checkpoint 已备份" >> "$LOG_FILE"

        # 备份 S3 checkpoint
        scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $S3_PORT -r \
            root@$S3_HOST:/root/checkpoints/marft_v4_adaptive \
            "$CHECKPOINT_DIR/adaptive_$BACKUP_TIME" 2>/dev/null && \
            echo "S3 checkpoint 已备份" >> "$LOG_FILE"
    fi

    # 每小时检查新的Warning类型 (每6次检查)
    if [ $((i % 6)) -eq 0 ]; then
        echo "--- 检查新 Warning 类型 ---" >> "$LOG_FILE"

        # 获取最新Warning并写入bug文档
        NEW_WARNINGS=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $S1_PORT root@$S1_HOST "
            grep 'WARNING' /root/FinSage/training_aggressive.log 2>/dev/null | \
            sed 's/.*\[WARNING\]/[WARNING]/' | sort | uniq -c | sort -rn | head -5
        " 2>/dev/null)

        if [ -n "$NEW_WARNINGS" ]; then
            echo "S1 Top Warnings:" >> "$LOG_FILE"
            echo "$NEW_WARNINGS" >> "$LOG_FILE"
        fi
    fi

    # 等待下一次检查
    if [ $i -lt $TOTAL_CHECKS ]; then
        sleep $CHECK_INTERVAL
    fi
done

echo "" >> "$LOG_FILE"
echo "=============================================================" >> "$LOG_FILE"
echo "监控完成 - $(date)" >> "$LOG_FILE"
echo "=============================================================" >> "$LOG_FILE"

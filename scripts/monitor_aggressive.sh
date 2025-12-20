#!/bin/bash
# 监控 Aggressive 评估并自动下载结果

VAST_API_KEY="e843ca4519847d7869e474f3239c0967da730ac9ab733ee77520fd2a59035384"
SSH_HOST="ssh4.vast.ai"
SSH_PORT="36016"
INSTANCE_ID="29036017"
LOCAL_RESULTS="/Users/guboyang/Desktop/Project/FinSage/evaluation_results/aggressive"
CHECK_INTERVAL=300

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

mkdir -p "$LOCAL_RESULTS"

log "开始监控 Aggressive 评估"
log "检查间隔: ${CHECK_INTERVAL}秒"

while true; do
    # 检查进程
    running=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "ps aux | grep evaluate_trained | grep -v grep | wc -l" 2>/dev/null || echo "0")

    log "进程状态: $running 个运行中"

    # 显示最新进度
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "tail -5 /root/FinSage/eval_aggressive.log 2>/dev/null | grep '收益'" 2>/dev/null || true

    if [ "$running" = "0" ]; then
        log "评估已结束，开始下载..."

        # 下载结果
        scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT \
            root@$SSH_HOST:/root/FinSage/eval_aggressive.log \
            "$LOCAL_RESULTS/" 2>/dev/null

        scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT \
            "root@$SSH_HOST:/root/FinSage/results/eval_aggressive*.log" \
            "$LOCAL_RESULTS/" 2>/dev/null

        log "结果已下载到: $LOCAL_RESULTS"

        # 删除实例
        log "删除 GPU 实例..."
        curl -s -X DELETE \
            -H "Authorization: Bearer $VAST_API_KEY" \
            "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" > /dev/null

        log "实例已删除"

        # 显示最终结果
        log "=== 最终结果 ==="
        grep -E "(年化收益|夏普比率|最大回撤)" "$LOCAL_RESULTS/eval_aggressive.log" | tail -5

        # 发送通知
        osascript -e 'display notification "Aggressive 评估完成" with title "FinSage"' 2>/dev/null || true

        break
    fi

    log "等待 ${CHECK_INTERVAL} 秒..."
    sleep $CHECK_INTERVAL
done

log "监控结束"

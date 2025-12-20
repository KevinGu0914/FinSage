#!/bin/bash
# 自动化脚本: 监控评估完成 -> 下载结果 -> 删除GPU服务器
# 用法: nohup ./scripts/auto_download_and_cleanup.sh > auto_monitor.log 2>&1 &

# ============================================================
# 配置
# ============================================================
VAST_API_KEY="e843ca4519847d7869e474f3239c0967da730ac9ab733ee77520fd2a59035384"
LOCAL_RESULTS_DIR="/Users/guboyang/Desktop/Project/FinSage/evaluation_results"
CHECK_INTERVAL=300  # 每5分钟检查一次

# 服务器配置 (name:host:port:instance_id)
SERVER_AGGRESSIVE="ssh9.vast.ai:21608:29021609"
SERVER_BALANCED="ssh5.vast.ai:25156:29025156"
SERVER_ADAPTIVE="ssh8.vast.ai:25160:29025160"

# 完成状态
DONE_AGGRESSIVE=0
DONE_BALANCED=0
DONE_ADAPTIVE=0

# ============================================================
# 函数
# ============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_eval_complete() {
    local name=$1
    local server_info=$2
    local host=$(echo $server_info | cut -d: -f1)
    local port=$(echo $server_info | cut -d: -f2)

    # 检查进程是否还在运行
    local running=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $port root@$host "ps aux | grep evaluate_trained | grep -v grep | wc -l" 2>/dev/null || echo "0")

    if [ "$running" = "0" ] || [ "$running" = "" ]; then
        echo "stopped"
    else
        echo "running"
    fi
}

download_results() {
    local name=$1
    local server_info=$2
    local host=$(echo $server_info | cut -d: -f1)
    local port=$(echo $server_info | cut -d: -f2)

    log "下载 $name 结果..."

    mkdir -p "$LOCAL_RESULTS_DIR/$name"

    # 下载日志
    scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $port \
        "root@$host:/root/FinSage/eval_v2_${name}.log" \
        "$LOCAL_RESULTS_DIR/$name/" 2>/dev/null || log "eval_v2 日志下载失败"

    # 下载 results 目录中的评估结果
    scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $port \
        "root@$host:/root/FinSage/results/eval_${name}*.log" \
        "$LOCAL_RESULTS_DIR/$name/" 2>/dev/null || log "results日志下载失败"

    scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $port \
        "root@$host:/root/FinSage/results/eval_${name}*.json" \
        "$LOCAL_RESULTS_DIR/$name/" 2>/dev/null || log "results JSON下载失败"

    log "$name 结果下载完成"
}

destroy_instance() {
    local name=$1
    local server_info=$2
    local instance_id=$(echo $server_info | cut -d: -f3)

    log "删除实例 $name (ID: $instance_id)..."

    curl -s -X DELETE \
        -H "Authorization: Bearer $VAST_API_KEY" \
        "https://console.vast.ai/api/v0/instances/$instance_id/" > /dev/null

    log "实例 $name 已删除"
}

get_final_stats() {
    local name=$1
    local log_file="$LOCAL_RESULTS_DIR/$name/eval_v2_${name}.log"

    if [ -f "$log_file" ]; then
        # 提取最后的收益率
        local last_line=$(grep "收益:" "$log_file" | tail -1)
        echo "$name: $last_line"
    else
        echo "$name: 日志未找到"
    fi
}

process_server() {
    local name=$1
    local server_info=$2

    local status=$(check_eval_complete $name $server_info)
    log "$name 状态: $status"

    if [ "$status" = "stopped" ]; then
        log "$name 评估已结束，开始下载..."
        download_results $name $server_info

        log "$name 结果已保存，删除服务器..."
        destroy_instance $name $server_info

        return 0  # 完成
    else
        return 1  # 未完成
    fi
}

# ============================================================
# 主逻辑
# ============================================================
log "=========================================="
log "开始监控评估任务"
log "检查间隔: ${CHECK_INTERVAL}秒"
log "=========================================="

# 创建结果目录
mkdir -p "$LOCAL_RESULTS_DIR"

while true; do
    all_done=1

    # 检查 Aggressive
    if [ "$DONE_AGGRESSIVE" = "0" ]; then
        if process_server "aggressive" "$SERVER_AGGRESSIVE"; then
            DONE_AGGRESSIVE=1
        else
            all_done=0
        fi
    fi

    # 检查 Balanced
    if [ "$DONE_BALANCED" = "0" ]; then
        if process_server "balanced" "$SERVER_BALANCED"; then
            DONE_BALANCED=1
        else
            all_done=0
        fi
    fi

    # 检查 Adaptive
    if [ "$DONE_ADAPTIVE" = "0" ]; then
        if process_server "adaptive" "$SERVER_ADAPTIVE"; then
            DONE_ADAPTIVE=1
        else
            all_done=0
        fi
    fi

    if [ "$all_done" = "1" ]; then
        log "=========================================="
        log "所有评估已完成！"
        log "=========================================="
        log "最终结果汇总:"
        get_final_stats "aggressive"
        get_final_stats "balanced"
        get_final_stats "adaptive"
        log "=========================================="
        log "结果保存在: $LOCAL_RESULTS_DIR"
        log "=========================================="

        # 发送通知 (macOS)
        osascript -e 'display notification "所有评估已完成，结果已下载" with title "FinSage 评估"' 2>/dev/null || true

        break
    fi

    log "等待 ${CHECK_INTERVAL} 秒后再次检查..."
    sleep $CHECK_INTERVAL
done

log "监控脚本结束"

#!/bin/bash
# V2 评估监控脚本

LOG_FILE="/Users/guboyang/Desktop/Project/FinSage/results/eval_monitor_v2.log"

echo "=" | tr '\n' '='
printf '%0.s=' {1..60}
echo ""
echo "MARFT V4 评估监控 (V2 增强Bug检测版)"
echo "开始时间: $(date)"
echo "测试期间: 2024-07-01 ~ 2024-12-31"
printf '%0.s=' {1..60}
echo ""
echo ""

# 服务器配置
declare -A SERVERS
SERVERS["ssh9.vast.ai:21608"]="aggressive_final"
SERVERS["ssh5.vast.ai:25156"]="balanced_final"
SERVERS["ssh8.vast.ai:25160"]="adaptive_final"

check_server() {
    local server=$1
    local checkpoint=$2
    local host="${server%:*}"
    local port="${server#*:}"

    echo "=== ${checkpoint^^} ($host:$port) ==="

    ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p $port root@$host "
        # GPU状态
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'GPU查询失败'

        # 进程状态
        echo '进程数:' \$(ps aux | grep '[e]valuate_trained_lora' | wc -l)

        # 最新日志
        if [ -f /root/FinSage/eval_${checkpoint}_v2.log ]; then
            echo '--- 最新日志 ---'
            tail -10 /root/FinSage/eval_${checkpoint}_v2.log
        fi

        # 检查是否有结果文件
        echo ''
        echo '--- 结果文件 ---'
        ls -la /root/FinSage/results/eval_${checkpoint}*.json 2>/dev/null | tail -3 || echo '暂无结果文件'
    " 2>&1 | head -30

    echo ""
}

# 循环监控
for i in {1..60}; do
    echo ""
    echo "========== 检查 #$i - $(date '+%Y-%m-%d %H:%M:%S') =========="

    for server in "${!SERVERS[@]}"; do
        check_server "$server" "${SERVERS[$server]}"
    done

    echo "下次检查: 2分钟后..."
    sleep 120
done

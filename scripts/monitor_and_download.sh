#!/bin/bash
# 持续监控 Aggressive 评估，完成后下载结果并删除 GPU

SSH_HOST="194.228.55.129"
SSH_PORT="31423"
REMOTE_DIR="/root/FinSage"
LOCAL_DIR="/Users/guboyang/Desktop/Project/FinSage/evaluation_results"
LOG_FILE="eval_aggressive.log"

mkdir -p "$LOCAL_DIR"

echo "========================================"
echo " Aggressive 评估监控器"
echo " GPU: $SSH_HOST:$SSH_PORT"
echo " 开始时间: $(date)"
echo "========================================"

while true; do
    # 检查进程是否还在运行
    RUNNING=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "ps aux | grep evaluate_trained_lora | grep -v grep | wc -l" 2>/dev/null)

    if [ -z "$RUNNING" ]; then
        echo "[$(date '+%H:%M:%S')] SSH 连接失败，等待重试..."
        sleep 30
        continue
    fi

    # 获取最新日志
    LATEST=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST "tail -5 $REMOTE_DIR/$LOG_FILE 2>/dev/null | grep '组合价值' | tail -1" 2>/dev/null)

    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "========================================"
        echo " 评估完成！开始下载结果..."
        echo "========================================"

        # 下载日志文件
        echo "[1/3] 下载评估日志..."
        scp -o StrictHostKeyChecking=no -P $SSH_PORT root@$SSH_HOST:$REMOTE_DIR/$LOG_FILE "$LOCAL_DIR/aggressive_eval_$(date +%Y%m%d_%H%M%S).log"

        if [ $? -ne 0 ]; then
            echo "ERROR: 日志下载失败！不删除 GPU"
            exit 1
        fi
        echo "✓ 日志下载成功"

        # 下载 results 目录
        echo "[2/3] 下载 results 目录..."
        scp -r -o StrictHostKeyChecking=no -P $SSH_PORT root@$SSH_HOST:$REMOTE_DIR/results/ "$LOCAL_DIR/"

        if [ $? -ne 0 ]; then
            echo "WARNING: results 目录下载可能不完整"
        else
            echo "✓ results 下载成功"
        fi

        # 验证下载
        echo "[3/3] 验证下载..."
        if [ -f "$LOCAL_DIR/aggressive_eval_"*.log ]; then
            LOG_SIZE=$(ls -la "$LOCAL_DIR"/aggressive_eval_*.log | tail -1 | awk '{print $5}')
            echo "✓ 日志文件大小: $LOG_SIZE bytes"

            # 显示最终结果
            echo ""
            echo "========================================"
            echo " 最终评估结果"
            echo "========================================"
            tail -30 "$LOCAL_DIR"/aggressive_eval_*.log | grep -E "年化|Sharpe|最大回撤|总收益|组合价值"

            echo ""
            echo "========================================"
            echo " 准备删除 GPU 实例..."
            echo "========================================"

            # 获取实例 ID 并删除
            # 注意：需要手动确认实例 ID
            echo "请手动运行以下命令删除 GPU："
            echo "  vastai destroy <instance_id>"
            echo ""
            echo "或者确认自动删除（输入 y）："
            read -t 30 -p "自动删除? (y/n): " CONFIRM

            if [ "$CONFIRM" = "y" ]; then
                # 查找实例 ID
                INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print([i['id'] for i in data if '$SSH_HOST' in str(i.get('public_ipaddr','')) or '$SSH_HOST' in str(i.get('ssh_host',''))][0] if data else '')" 2>/dev/null)

                if [ -n "$INSTANCE_ID" ]; then
                    echo "删除实例 $INSTANCE_ID..."
                    vastai destroy $INSTANCE_ID --yes
                    echo "✓ GPU 实例已删除"
                else
                    echo "未找到实例 ID，请手动删除"
                fi
            fi

            echo ""
            echo "========================================"
            echo " 完成！结果保存在: $LOCAL_DIR"
            echo "========================================"
            exit 0
        else
            echo "ERROR: 下载验证失败！不删除 GPU"
            exit 1
        fi
    else
        # 显示进度
        echo "[$(date '+%H:%M:%S')] 运行中 | $LATEST"
        sleep 60
    fi
done

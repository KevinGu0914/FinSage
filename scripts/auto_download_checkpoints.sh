#!/bin/bash
# 自动监控训练并下载检查点
# 用法: ./auto_download_checkpoints.sh

SERVER="root@ssh4.vast.ai"
PORT="39557"
LOG_FILE="/root/train_8bit_new.log"
REMOTE_CHECKPOINT="/root/checkpoints_final.tar.gz"
LOCAL_DIR="$HOME/Desktop/Project/FinSage/checkpoints"

echo "========================================"
echo " FinSage Training Monitor & Downloader"
echo "========================================"
echo " Server: $SERVER:$PORT"
echo " Checking every 5 minutes..."
echo "========================================"

mkdir -p "$LOCAL_DIR"

while true; do
    # 检查服务器是否还在运行
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p $PORT $SERVER "echo 'alive'" 2>/dev/null; then
        echo "[$(date)] Server is offline. Checking if training completed..."

        # 如果服务器已关机，可能训练完成了，尝试最后一次下载
        echo "Server appears to have shutdown. Training may be complete."
        break
    fi

    # 检查训练是否完成
    COMPLETE=$(ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $PORT $SERVER "grep -c 'Training Complete' $LOG_FILE 2>/dev/null || echo 0")

    if [ "$COMPLETE" -gt 0 ]; then
        echo ""
        echo "========================================"
        echo " Training Complete! Downloading..."
        echo "========================================"

        # 等待压缩完成
        echo "[$(date)] Waiting for checkpoint compression..."
        sleep 30

        # 检查压缩文件是否存在
        if ssh -o ConnectTimeout=30 -p $PORT $SERVER "test -f $REMOTE_CHECKPOINT && echo 'exists'" 2>/dev/null | grep -q 'exists'; then
            echo "[$(date)] Downloading checkpoints..."

            # 下载压缩包
            scp -P $PORT $SERVER:$REMOTE_CHECKPOINT "$LOCAL_DIR/"

            if [ $? -eq 0 ]; then
                echo ""
                echo "========================================"
                echo " Download Complete!"
                echo "========================================"
                echo " Saved to: $LOCAL_DIR/checkpoints_final.tar.gz"

                # 解压
                echo " Extracting..."
                cd "$LOCAL_DIR" && tar -xzf checkpoints_final.tar.gz
                echo " Extracted to: $LOCAL_DIR/checkpoints/"

                # 显示训练结果
                echo ""
                echo "========================================"
                echo " Training Results:"
                echo "========================================"
                ssh -p $PORT $SERVER "tail -30 $LOG_FILE | grep -E 'Epoch|Return|Portfolio|Complete|Total'" 2>/dev/null

                echo ""
                echo "All done! Server will shutdown automatically."
                break
            else
                echo "Download failed! Retrying in 60 seconds..."
                sleep 60
            fi
        else
            echo "[$(date)] Checkpoint file not ready yet, waiting..."
            sleep 60
        fi
    else
        # 显示当前进度
        PROGRESS=$(ssh -o ConnectTimeout=30 -p $PORT $SERVER "tail -5 $LOG_FILE 2>/dev/null | grep -E '\[202|Epoch' | tail -1")
        echo "[$(date)] Training in progress: $PROGRESS"

        # 每5分钟检查一次
        sleep 300
    fi
done

echo ""
echo "Monitor script finished."

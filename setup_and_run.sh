#!/bin/bash
# ============================================================
# FinSage MARFT Training Setup Script
# 在GPU服务器上运行此脚本进行环境配置和训练
# ============================================================

set -e

echo "============================================================"
echo " FinSage MARFT Setup Script"
echo "============================================================"

# 1. 检查GPU
echo ""
echo "[1/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Make sure CUDA is installed."
fi

# 2. 创建conda环境 (可选)
echo ""
echo "[2/5] Setting up Python environment..."
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."
    conda create -n finsage python=3.10 -y 2>/dev/null || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate finsage
fi

# 3. 安装依赖
echo ""
echo "[3/5] Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers peft accelerate bitsandbytes -q
pip install numpy pandas yfinance requests tqdm -q

# 4. 验证安装
echo ""
echo "[4/5] Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 5. 运行测试
echo ""
echo "[5/5] Running tests..."
python tests/test_marft_integration.py

echo ""
echo "============================================================"
echo " Setup Complete! "
echo "============================================================"
echo ""
echo "To start training, run:"
echo ""
echo "  # Debug mode (quick test)"
echo "  python scripts/run_marft_training.py --num_env_steps 10000"
echo ""
echo "  # Full training"
echo "  python scripts/run_marft_training.py --num_env_steps 1000000"
echo ""

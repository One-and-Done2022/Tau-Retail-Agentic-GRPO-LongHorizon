#!/usr/bin/env bash
# 项目环境搭建脚本，建议在全新 conda env 里跑
# 用法: bash setup.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="agentrl"
PYTHON_VERSION="3.10"

echo "=== [1/5] 创建 conda 环境: $ENV_NAME ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在，跳过创建步骤..."
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== [2/5] 安装 PyTorch (CUDA 12.6) ==="
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

echo "=== [3/5] 安装项目依赖 ==="
python -m pip install -r requirements.txt

echo "=== [4/5] 安装 τ-bench ==="
python -m pip install -e "$ROOT_DIR/tau-bench"
python -m pip install -e "$ROOT_DIR/verl"

echo "=== [5/5] 检查模型目录 ==="
if [ -f "$ROOT_DIR/models/Qwen2.5-7B-Instruct/config.json" ]; then
    echo "已找到 models/Qwen2.5-7B-Instruct"
else
    echo "未找到 7B 模型。请按 REPRODUCE_4090_CN.md 的模型下载步骤执行。"
fi

echo "=== 搭建完成 ==="
echo "激活环境: conda activate $ENV_NAME"

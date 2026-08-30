#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-agentrl}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-agentrl}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p experiments/sft_lora_4090_smoke
python scripts/train/sft/sft_train.py \
    --config configs/train/sft/sft_airline_lora_4090_smoke.yaml \
    2>&1 | tee experiments/sft_lora_4090_smoke/training.log

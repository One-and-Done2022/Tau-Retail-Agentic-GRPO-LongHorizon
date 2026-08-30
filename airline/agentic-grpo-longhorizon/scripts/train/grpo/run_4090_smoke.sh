#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-agentrl}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-agentrl}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false

mkdir -p experiments/4090_smoke
python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/configs/train/mock" \
    --config-name=mock_4090_smoke \
    2>&1 | tee experiments/4090_smoke/training.log

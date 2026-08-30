#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-agentrl}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-agentrl}"
fi

RUN_DIR="experiments/retail_vanilla_grpo/smoke_001"
if [[ -e "$RUN_DIR/training.log" || -e "$RUN_DIR/checkpoints" ]]; then
    echo "Refusing to overwrite existing run output under $RUN_DIR" >&2
    exit 2
fi

nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    | tee "$RUN_DIR/gpu_before.csv"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/configs/train/grpo" \
    --config-name=retail_vanilla_4090_smoke \
    2>&1 | tee "$RUN_DIR/training.log"

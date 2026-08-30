#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

METHOD="${1:-}"
RUN_ID="${2:-run_001}"
case "$METHOD" in
    vanilla)
        INTERACTION_CONFIG="configs/interaction_config/tau_bench_retail_7b_smoke.yaml"
        ADV_ESTIMATOR="grpo"
        BETA="0.0"
        ;;
    state)
        INTERACTION_CONFIG="configs/interaction_config/tau_bench_retail_state_7b_smoke.yaml"
        ADV_ESTIMATOR="grpo_cs"
        BETA="0.5"
        ;;
    constraint)
        INTERACTION_CONFIG="configs/interaction_config/tau_bench_retail_constraint_7b_smoke.yaml"
        ADV_ESTIMATOR="grpo_cs"
        BETA="0.5"
        ;;
    cs)
        INTERACTION_CONFIG="configs/interaction_config/tau_bench_retail_cs_7b_smoke.yaml"
        ADV_ESTIMATOR="grpo_cs"
        BETA="0.5"
        ;;
    *)
        echo "Usage: $0 {vanilla|state|constraint|cs} [run_id]" >&2
        exit 2
        ;;
esac

if [[ ! "$RUN_ID" =~ ^run_[0-9]{3}$ ]]; then
    echo "run_id must match run_NNN, got: $RUN_ID" >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-agentrl}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-agentrl}"
fi

RUN_DIR="experiments/retail_ablation/$METHOD/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi

GPU_DEVICES="${ABLATION_GPU_DEVICES:-4,5}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_DEVICES"
if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
    echo "ABLATION_GPU_DEVICES must contain exactly two physical GPU IDs" >&2
    exit 2
fi

if ! curl --fail --silent "http://localhost:8001/v1/models" >/dev/null; then
    echo "Retail user simulator is unavailable at http://localhost:8001/v1" >&2
    exit 3
fi

for GPU_ID in "${GPU_ARRAY[@]}"; do
    FREE_MIB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < 22000 )); then
        echo "Physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free; refusing to start" >&2
        exit 4
    fi
done

mkdir -p "$RUN_DIR"
cp configs/eval/retail/ablation_manifest.json "$RUN_DIR/frozen_manifest.json"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_before.csv"
sha256sum \
    configs/train/grpo/retail_ablation_4090_pilot.yaml \
    "$INTERACTION_CONFIG" \
    experiments/retail_vanilla_grpo/smoke_001/train.parquet \
    experiments/retail_sft/lora_smoke_merged/config.json \
    > "$RUN_DIR/input_hashes.txt"
git status --short --branch > "$RUN_DIR/git_status.txt"

python - "$METHOD" "$RUN_ID" "$GPU_DEVICES" "$INTERACTION_CONFIG" "$ADV_ESTIMATOR" "$BETA" "$RUN_DIR" <<'PY'
import datetime
import json
import sys
from pathlib import Path

method, run_id, gpu_devices, interaction, advantage, beta, run_dir = sys.argv[1:]
payload = {
    "label": "fixed-budget-engineering-pilot",
    "method": method,
    "run_id": run_id,
    "started_at": datetime.datetime.now().astimezone().isoformat(),
    "physical_actor_gpus": [int(item) for item in gpu_devices.split(",")],
    "physical_user_simulator_gpu": 6,
    "user_simulator_base_url": "http://localhost:8001/v1",
    "config": "configs/train/grpo/retail_ablation_4090_pilot.yaml",
    "interaction_config": interaction,
    "adv_estimator": advantage,
    "beta": float(beta),
    "output_dir": run_dir,
}
Path(run_dir, "run_manifest.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/configs/train/grpo" \
    --config-name=retail_ablation_4090_pilot \
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$INTERACTION_CONFIG" \
    "algorithm.adv_estimator=$ADV_ESTIMATOR" \
    "algorithm.cs_grpo.beta=$BETA" \
    "trainer.default_local_dir=$RUN_DIR/checkpoints" \
    "trainer.rollout_data_dir=$RUN_DIR/rollouts" \
    "trainer.experiment_name=retail_ablation_${METHOD}_${RUN_ID}" \
    2>&1 | tee "$RUN_DIR/training.log"

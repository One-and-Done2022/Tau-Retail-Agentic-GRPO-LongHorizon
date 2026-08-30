#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

METHOD="${1:-}"
RUN_ID="${2:-run_001}"
case "$METHOD" in
    vanilla)
        INTERACTION_BASENAME="tau_bench_retail_confirmatory"
        ADV_ESTIMATOR="grpo"
        BETA="0.0"
        ;;
    state)
        INTERACTION_BASENAME="tau_bench_retail_state_confirmatory"
        ADV_ESTIMATOR="grpo_cs"
        BETA="0.5"
        ;;
    constraint)
        INTERACTION_BASENAME="tau_bench_retail_constraint_confirmatory"
        ADV_ESTIMATOR="grpo_cs"
        BETA="0.5"
        ;;
    cs)
        INTERACTION_BASENAME="tau_bench_retail_cs_confirmatory"
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

PROTOCOL="${CONFIRMATORY_PROTOCOL:-7b_user}"
if [[ "$PROTOCOL" == "72b_user" ]]; then
    INTERACTION_CONFIG="configs/interaction_config/${INTERACTION_BASENAME}_72b_user.yaml"
else
    INTERACTION_CONFIG="configs/interaction_config/${INTERACTION_BASENAME}.yaml"
fi
EXPERIMENT_ROOT="${CONFIRMATORY_EXPERIMENT_ROOT:-experiments/retail_confirmatory}"
FROZEN_MANIFEST="${CONFIRMATORY_MANIFEST:-configs/eval/retail/confirmatory_manifest.json}"
USER_SIMULATOR_GPUS="${CONFIRMATORY_USER_SIMULATOR_GPUS:-6}"
USER_SIMULATOR_MODEL="${CONFIRMATORY_USER_SIMULATOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_DIR="$EXPERIMENT_ROOT/train/$METHOD/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi
if ! curl --fail --silent "http://localhost:8001/v1/models" >/dev/null; then
    echo "Retail user simulator is unavailable at http://localhost:8001/v1" >&2
    exit 3
fi

GPU_DEVICES="${CONFIRMATORY_GPU_DEVICES:-4,5}"
MIN_FREE_MIB="${CONFIRMATORY_MIN_FREE_MIB:-22000}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_DEVICES"
if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
    echo "CONFIRMATORY_GPU_DEVICES must contain exactly two physical GPU IDs" >&2
    exit 2
fi
if [[ ! "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "CONFIRMATORY_MIN_FREE_MIB must be an integer" >&2
    exit 2
fi
for GPU_ID in "${GPU_ARRAY[@]}"; do
    FREE_MIB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < MIN_FREE_MIB )); then
        echo "Physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free; refusing to start" >&2
        exit 4
    fi
done

mkdir -p "$RUN_DIR"
cp "$FROZEN_MANIFEST" "$RUN_DIR/frozen_manifest.json"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_before.csv"
sha256sum \
    "$FROZEN_MANIFEST" \
    configs/train/grpo/retail_confirmatory_4090.yaml \
    "$INTERACTION_CONFIG" \
    experiments/retail_confirmatory/data/train.parquet \
    experiments/retail_sft/lora_smoke_merged/model.safetensors.index.json \
    > "$RUN_DIR/input_hashes.txt"
git status --short --branch > "$RUN_DIR/git_status.txt"

python - "$METHOD" "$RUN_ID" "$GPU_DEVICES" "$USER_SIMULATOR_GPUS" "$USER_SIMULATOR_MODEL" "$PROTOCOL" "$INTERACTION_CONFIG" "$ADV_ESTIMATOR" "$BETA" "$RUN_DIR" <<'PY'
import datetime
import json
import sys
from pathlib import Path

(
    method,
    run_id,
    gpu_devices,
    user_simulator_gpus,
    user_simulator_model,
    protocol,
    interaction,
    advantage,
    beta,
    run_dir,
) = sys.argv[1:]
payload = {
    "label": "confirmatory-fixed-budget-engineering-study",
    "method": method,
    "run_id": run_id,
    "protocol": protocol,
    "started_at": datetime.datetime.now().astimezone().isoformat(),
    "physical_actor_gpus": [int(item) for item in gpu_devices.split(",")],
    "physical_user_simulator_gpus": [int(item) for item in user_simulator_gpus.split(",")],
    "user_simulator_model": user_simulator_model,
    "user_simulator_base_url": "http://localhost:8001/v1",
    "config": "configs/train/grpo/retail_confirmatory_4090.yaml",
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
    --config-name=retail_confirmatory_4090 \
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$INTERACTION_CONFIG" \
    "algorithm.adv_estimator=$ADV_ESTIMATOR" \
    "algorithm.cs_grpo.beta=$BETA" \
    "trainer.default_local_dir=$RUN_DIR/checkpoints" \
    "trainer.rollout_data_dir=$RUN_DIR/rollouts" \
    "trainer.experiment_name=retail_confirmatory_${METHOD}_${RUN_ID}" \
    2>&1 | tee "$RUN_DIR/training.log"

python scripts/train/grpo/check_retail_confirmatory_run.py "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/gate.log"

for STEP in 4 8; do
    python scripts/test/merge_fsdp_to_hf.py \
        --actor-dir "$RUN_DIR/checkpoints/global_step_$STEP/actor" \
        --output-dir "$RUN_DIR/hf_step_$STEP" \
        2>&1 | tee "$RUN_DIR/merge_step_$STEP.log"
    python - "$RUN_DIR/hf_step_$STEP" <<'PY' 2>&1 | tee "$RUN_DIR/hf_reload_step_$STEP.log"
import sys
from transformers import AutoConfig, AutoModelForCausalLM

path = sys.argv[1]
config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True
)
print("HF_RELOAD", path, config.model_type, len(model.state_dict()))
PY
done

nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_after.csv"

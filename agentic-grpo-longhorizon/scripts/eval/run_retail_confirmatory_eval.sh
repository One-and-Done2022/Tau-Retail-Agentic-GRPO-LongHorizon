#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CANDIDATE="${1:-}"
RUN_ID="${2:-run_001}"
EXPERIMENT_ROOT="${CONFIRMATORY_EXPERIMENT_ROOT:-experiments/retail_confirmatory}"
TRAIN_EXPERIMENT_ROOT="${CONFIRMATORY_TRAIN_EXPERIMENT_ROOT:-$EXPERIMENT_ROOT}"
FROZEN_MANIFEST="${CONFIRMATORY_MANIFEST:-configs/eval/retail/confirmatory_manifest.json}"
SEEN_CONFIG="${CONFIRMATORY_SEEN_CONFIG:-configs/eval/retail/confirmatory_seen.yaml}"
UNSEEN_CONFIG="${CONFIRMATORY_UNSEEN_CONFIG:-configs/eval/retail/confirmatory_unseen_dev.yaml}"
if [[ "$CANDIDATE" == "sft" ]]; then
    MODEL_PATH="experiments/retail_sft/lora_smoke_merged"
elif [[ "$CANDIDATE" =~ ^(vanilla|state|constraint|cs)_step_(4|8)$ ]]; then
    METHOD="${BASH_REMATCH[1]}"
    STEP="${BASH_REMATCH[2]}"
    MODEL_PATH="$TRAIN_EXPERIMENT_ROOT/train/$METHOD/$RUN_ID/hf_step_$STEP"
else
    echo "Usage: $0 {sft|vanilla_step_4|vanilla_step_8|state_step_4|state_step_8|constraint_step_4|constraint_step_8|cs_step_4|cs_step_8} [run_id]" >&2
    exit 2
fi
if [[ ! "$RUN_ID" =~ ^run_[0-9]{3}$ ]]; then
    echo "run_id must match run_NNN, got: $RUN_ID" >&2
    exit 2
fi
if [[ ! -s "$MODEL_PATH/model.safetensors" && ! -s "$MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "Missing merged model weights under: $MODEL_PATH" >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-agentrl}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME:-agentrl}"
fi

EVAL_GPU="${CONFIRMATORY_EVAL_GPU:-1}"
MIN_FREE_MIB="${CONFIRMATORY_MIN_FREE_MIB:-22000}"
POLICY_PORT="${CONFIRMATORY_POLICY_PORT:-8012}"
RUN_DIR="$EXPERIMENT_ROOT/eval/$CANDIDATE/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing evaluation directory: $RUN_DIR" >&2
    exit 2
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$POLICY_PORT$"; then
    echo "Policy port $POLICY_PORT is already in use" >&2
    exit 3
fi
if ! curl --fail --silent "http://localhost:8001/v1/models" >/dev/null; then
    echo "Retail user simulator is unavailable at http://localhost:8001/v1" >&2
    exit 3
fi
FREE_MIB="$(nvidia-smi --id="$EVAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "CONFIRMATORY_MIN_FREE_MIB must be an integer" >&2
    exit 2
fi
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < MIN_FREE_MIB )); then
    echo "Physical GPU $EVAL_GPU has only ${FREE_MIB:-unknown} MiB free; refusing to deploy" >&2
    exit 4
fi

mkdir -p "$RUN_DIR"
cp "$FROZEN_MANIFEST" "$RUN_DIR/frozen_manifest.json"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_before.csv"
sha256sum \
    "$FROZEN_MANIFEST" \
    "$SEEN_CONFIG" \
    "$UNSEEN_CONFIG" \
    "$MODEL_PATH/config.json" \
    > "$RUN_DIR/input_hashes.txt"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

export LITELLM_LOCAL_MODEL_COST_MAP=True
setsid env \
    CUDA_DEVICES="$EVAL_GPU" \
    PORT="$POLICY_PORT" \
    MODEL_PATH="$MODEL_PATH" \
    SERVED_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
    TP_SIZE=1 \
    GPU_MEM_UTIL=0.82 \
    MAX_MODEL_LEN=24576 \
    MAX_NUM_SEQS=4 \
    bash scripts/vllm_server/7b.sh > "$RUN_DIR/deploy.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/server_pid.txt"

READY=0
for _ in $(seq 1 120); do
    if curl --fail --silent "http://localhost:$POLICY_PORT/v1/models" >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$READY" -ne 1 ]]; then
    echo "Policy server failed to become ready; inspect $RUN_DIR/deploy.log" >&2
    exit 5
fi

python scripts/eval/eval_sft.py \
    --config "$SEEN_CONFIG" \
    --output-dir "$RUN_DIR/seen" \
    2>&1 | tee "$RUN_DIR/seen.log"
python scripts/eval/eval_sft.py \
    --config "$UNSEEN_CONFIG" \
    --output-dir "$RUN_DIR/unseen_dev" \
    2>&1 | tee "$RUN_DIR/unseen_dev.log"

python scripts/eval/check_retail_confirmatory_eval.py "$RUN_DIR" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --manifest "$FROZEN_MANIFEST" \
    2>&1 | tee "$RUN_DIR/gate.log"

cleanup
SERVER_PID=""
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_after.csv"
touch "$RUN_DIR/COMPLETED"

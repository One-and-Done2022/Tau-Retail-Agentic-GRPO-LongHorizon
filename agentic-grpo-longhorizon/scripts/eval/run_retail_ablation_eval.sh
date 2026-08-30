#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

METHOD="${1:-}"
RUN_ID="${2:-run_001}"
case "$METHOD" in
    sft) MODEL_PATH="experiments/retail_sft/lora_smoke_merged" ;;
    vanilla|state|constraint|cs)
        MODEL_PATH="experiments/retail_ablation/$METHOD/run_001/hf_step_3"
        ;;
    *)
        echo "Usage: $0 {sft|vanilla|state|constraint|cs} [run_id]" >&2
        exit 2
        ;;
esac
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

EVAL_GPU="${ABLATION_EVAL_GPU:-1}"
POLICY_PORT="${ABLATION_POLICY_PORT:-8012}"
RUN_DIR="experiments/retail_ablation_eval/$METHOD/$RUN_ID"
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
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < 22000 )); then
    echo "Physical GPU $EVAL_GPU has only ${FREE_MIB:-unknown} MiB free; refusing to deploy" >&2
    exit 4
fi

mkdir -p "$RUN_DIR"
cp configs/eval/retail/ablation_manifest.json "$RUN_DIR/frozen_manifest.json"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_before.csv"
sha256sum \
    configs/eval/retail/ablation_seen.yaml \
    configs/eval/retail/ablation_unseen.yaml \
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
for _ in $(seq 1 90); do
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
    --config configs/eval/retail/ablation_seen.yaml \
    --output-dir "$RUN_DIR/seen" \
    2>&1 | tee "$RUN_DIR/seen.log"
python scripts/eval/eval_sft.py \
    --config configs/eval/retail/ablation_unseen.yaml \
    --output-dir "$RUN_DIR/unseen" \
    2>&1 | tee "$RUN_DIR/unseen.log"

cleanup
SERVER_PID=""
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv \
    > "$RUN_DIR/gpu_after.csv"

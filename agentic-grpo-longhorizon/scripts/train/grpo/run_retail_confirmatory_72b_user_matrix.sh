#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="${1:-run_001}"
EXPERIMENT_ROOT="experiments/retail_confirmatory_72b_user"
ORCHESTRATION_NAME="${CONFIRMATORY_ORCHESTRATION_NAME:-$RUN_ID}"
METHODS="${CONFIRMATORY_METHODS:-vanilla state constraint cs}"
if [[ ! "$ORCHESTRATION_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid orchestration name: $ORCHESTRATION_NAME" >&2
    exit 2
fi
read -r -a METHOD_ARRAY <<< "$METHODS"
if [[ "${#METHOD_ARRAY[@]}" -eq 0 ]]; then
    echo "CONFIRMATORY_METHODS must contain at least one method" >&2
    exit 2
fi
for METHOD in "${METHOD_ARRAY[@]}"; do
    if [[ ! "$METHOD" =~ ^(vanilla|state|constraint|cs)$ ]]; then
        echo "Invalid method in CONFIRMATORY_METHODS: $METHOD" >&2
        exit 2
    fi
done
ORCH_DIR="$EXPERIMENT_ROOT/orchestration/$ORCHESTRATION_NAME"
ACTOR_GPUS="${CONFIRMATORY_GPU_DEVICES:-5,6}"
USER_GPUS="${CONFIRMATORY_USER_SIMULATOR_GPUS:-0,1,3,4}"
MIN_FREE_MIB="${CONFIRMATORY_MIN_FREE_MIB:-22000}"
MANIFEST="configs/eval/retail/confirmatory_72b_user_manifest.json"
if [[ -e "$ORCH_DIR" ]]; then
    echo "Refusing to overwrite orchestration directory: $ORCH_DIR" >&2
    exit 2
fi
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8001$'; then
    echo "Port 8001 is already in use" >&2
    exit 3
fi

IFS=',' read -r -a ACTOR_GPU_ARRAY <<< "$ACTOR_GPUS"
IFS=',' read -r -a USER_GPU_ARRAY <<< "$USER_GPUS"
if [[ "${#ACTOR_GPU_ARRAY[@]}" -ne 2 || "${#USER_GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "Expected exactly two actor GPUs and four 72B user-simulator GPUs" >&2
    exit 2
fi
if [[ ! "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "CONFIRMATORY_MIN_FREE_MIB must be an integer" >&2
    exit 2
fi
for GPU_ID in "${ACTOR_GPU_ARRAY[@]}" "${USER_GPU_ARRAY[@]}"; do
    FREE_MIB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < MIN_FREE_MIB )); then
        echo "Physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free; refusing to start" >&2
        exit 4
    fi
done

mkdir -p "$ORCH_DIR"
nvidia-smi > "$ORCH_DIR/nvidia_smi_before.txt"
cp "$MANIFEST" "$ORCH_DIR/frozen_manifest.json"

USER_SERVER_PID=""
cleanup() {
    if [[ -n "$USER_SERVER_PID" ]] && kill -0 "$USER_SERVER_PID" 2>/dev/null; then
        kill -- "-$USER_SERVER_PID" 2>/dev/null || true
        wait "$USER_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid env \
    CUDA_DEVICES="$USER_GPUS" \
    PORT=8001 \
    MODEL_PATH=/home/liuchenyang/agentic-grpo-retail/models/Qwen2.5-72B-Instruct-AWQ \
    SERVED_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ \
    TP_SIZE=4 \
    GPU_MEM_UTIL=0.82 \
    MAX_MODEL_LEN=24576 \
    MAX_NUM_SEQS=4 \
    ENFORCE_EAGER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/vllm_server/72b.sh > "$ORCH_DIR/user_simulator.log" 2>&1 &
USER_SERVER_PID=$!
echo "$USER_SERVER_PID" > "$ORCH_DIR/user_simulator_pid.txt"

READY=0
for _ in $(seq 1 300); do
    if curl --fail --silent http://localhost:8001/v1/models \
        | grep -q 'Qwen/Qwen2.5-72B-Instruct-AWQ'; then
        READY=1
        break
    fi
    if ! kill -0 "$USER_SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$READY" -ne 1 ]]; then
    echo "72B user simulator failed to become ready" >&2
    exit 5
fi

for METHOD in "${METHOD_ARRAY[@]}"; do
    CONFIRMATORY_PROTOCOL=72b_user \
    CONFIRMATORY_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    CONFIRMATORY_MANIFEST="$MANIFEST" \
    CONFIRMATORY_GPU_DEVICES="$ACTOR_GPUS" \
    CONFIRMATORY_USER_SIMULATOR_GPUS="$USER_GPUS" \
    CONFIRMATORY_USER_SIMULATOR_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
        bash scripts/train/grpo/run_retail_confirmatory.sh "$METHOD" "$RUN_ID"
done

cleanup
USER_SERVER_PID=""
nvidia-smi > "$ORCH_DIR/nvidia_smi_after.txt"
touch "$ORCH_DIR/COMPLETED"

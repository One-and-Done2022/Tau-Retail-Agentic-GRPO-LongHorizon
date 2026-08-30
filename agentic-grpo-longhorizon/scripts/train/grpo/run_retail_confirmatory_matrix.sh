#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="${1:-run_001}"
ORCH_DIR="experiments/retail_confirmatory/orchestration/$RUN_ID"
if [[ -e "$ORCH_DIR" ]]; then
    echo "Refusing to overwrite orchestration directory: $ORCH_DIR" >&2
    exit 2
fi
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8001$'; then
    echo "Port 8001 is already in use" >&2
    exit 3
fi
for GPU_ID in 4 5 6; do
    FREE_MIB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < 22000 )); then
        echo "Physical GPU $GPU_ID has only ${FREE_MIB:-unknown} MiB free; refusing to start" >&2
        exit 4
    fi
done

mkdir -p "$ORCH_DIR"
nvidia-smi > "$ORCH_DIR/nvidia_smi_before.txt"
cp configs/eval/retail/confirmatory_manifest.json "$ORCH_DIR/frozen_manifest.json"

USER_SERVER_PID=""
cleanup() {
    if [[ -n "$USER_SERVER_PID" ]] && kill -0 "$USER_SERVER_PID" 2>/dev/null; then
        kill -- "-$USER_SERVER_PID" 2>/dev/null || true
        wait "$USER_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid env \
    CUDA_DEVICES=6 \
    PORT=8001 \
    MODEL_PATH=/home/liuchenyang/agentic-grpo-retail/models/Qwen2.5-7B-Instruct \
    SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
    TP_SIZE=1 \
    GPU_MEM_UTIL=0.82 \
    MAX_MODEL_LEN=24576 \
    MAX_NUM_SEQS=8 \
    bash scripts/vllm_server/7b.sh > "$ORCH_DIR/user_simulator.log" 2>&1 &
USER_SERVER_PID=$!
echo "$USER_SERVER_PID" > "$ORCH_DIR/user_simulator_pid.txt"

READY=0
for _ in $(seq 1 120); do
    if curl --fail --silent http://localhost:8001/v1/models >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$USER_SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$READY" -ne 1 ]]; then
    echo "User simulator failed to become ready" >&2
    exit 5
fi

for METHOD in vanilla state constraint cs; do
    CONFIRMATORY_GPU_DEVICES=4,5 \
        bash scripts/train/grpo/run_retail_confirmatory.sh "$METHOD" "$RUN_ID"
done

cleanup
USER_SERVER_PID=""
nvidia-smi > "$ORCH_DIR/nvidia_smi_after.txt"
touch "$ORCH_DIR/COMPLETED"

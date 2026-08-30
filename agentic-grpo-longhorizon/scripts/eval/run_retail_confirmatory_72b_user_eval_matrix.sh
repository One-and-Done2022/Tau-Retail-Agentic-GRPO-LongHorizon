#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="${1:-run_001}"
EXPERIMENT_ROOT="experiments/retail_confirmatory_72b_user"
ORCHESTRATION_NAME="${CONFIRMATORY_EVAL_ORCHESTRATION_NAME:-$RUN_ID}"
DEFAULT_CANDIDATES="sft vanilla_step_4 vanilla_step_8 state_step_4 state_step_8 constraint_step_4 constraint_step_8 cs_step_4 cs_step_8"
CANDIDATES="${CONFIRMATORY_EVAL_CANDIDATES:-$DEFAULT_CANDIDATES}"
MIN_FREE_MIB="${CONFIRMATORY_MIN_FREE_MIB:-22000}"
ORCH_DIR="$EXPERIMENT_ROOT/eval_orchestration/$ORCHESTRATION_NAME"
POLICY_GPU="${CONFIRMATORY_EVAL_GPU:-5}"
USER_GPUS="${CONFIRMATORY_USER_SIMULATOR_GPUS:-0,1,3,4}"
MANIFEST="configs/eval/retail/confirmatory_72b_user_manifest.json"
SEEN_CONFIG="configs/eval/retail/confirmatory_72b_user_seen.yaml"
UNSEEN_CONFIG="configs/eval/retail/confirmatory_72b_user_unseen_dev.yaml"
if [[ ! "$ORCHESTRATION_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid orchestration name: $ORCHESTRATION_NAME" >&2
    exit 2
fi
read -r -a CANDIDATE_ARRAY <<< "$CANDIDATES"
if [[ "${#CANDIDATE_ARRAY[@]}" -eq 0 ]]; then
    echo "CONFIRMATORY_EVAL_CANDIDATES must contain at least one candidate" >&2
    exit 2
fi
for CANDIDATE in "${CANDIDATE_ARRAY[@]}"; do
    if [[ ! "$CANDIDATE" =~ ^(sft|(vanilla|state|constraint|cs)_step_(4|8))$ ]]; then
        echo "Invalid candidate in CONFIRMATORY_EVAL_CANDIDATES: $CANDIDATE" >&2
        exit 2
    fi
done
if [[ -e "$ORCH_DIR" ]]; then
    echo "Refusing to overwrite orchestration directory: $ORCH_DIR" >&2
    exit 2
fi
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8001$'; then
    echo "Port 8001 is already in use" >&2
    exit 3
fi

IFS=',' read -r -a USER_GPU_ARRAY <<< "$USER_GPUS"
if [[ "${#USER_GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "Expected exactly four 72B user-simulator GPUs" >&2
    exit 2
fi
if [[ ! "$MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "CONFIRMATORY_MIN_FREE_MIB must be an integer" >&2
    exit 2
fi
for GPU_ID in "$POLICY_GPU" "${USER_GPU_ARRAY[@]}"; do
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

for CANDIDATE in "${CANDIDATE_ARRAY[@]}"; do
    CONFIRMATORY_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    CONFIRMATORY_TRAIN_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    CONFIRMATORY_MANIFEST="$MANIFEST" \
    CONFIRMATORY_SEEN_CONFIG="$SEEN_CONFIG" \
    CONFIRMATORY_UNSEEN_CONFIG="$UNSEEN_CONFIG" \
    CONFIRMATORY_EVAL_GPU="$POLICY_GPU" \
    CONFIRMATORY_POLICY_PORT=8012 \
        bash scripts/eval/run_retail_confirmatory_eval.sh "$CANDIDATE" "$RUN_ID"

    CLEANED_UP=0
    for _ in $(seq 1 180); do
        FREE_MIB="$(nvidia-smi --id="$POLICY_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if ! ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8012$' \
            && [[ "$FREE_MIB" =~ ^[0-9]+$ ]] \
            && (( FREE_MIB >= MIN_FREE_MIB )); then
            CLEANED_UP=1
            break
        fi
        sleep 1
    done
    if [[ "$CLEANED_UP" -ne 1 ]]; then
        echo "Policy server did not release port 8012 and GPU $POLICY_GPU in time" >&2
        exit 6
    fi
done

cleanup
USER_SERVER_PID=""
nvidia-smi > "$ORCH_DIR/nvidia_smi_after.txt"
touch "$ORCH_DIR/COMPLETED"

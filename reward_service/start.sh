#!/bin/bash
# Launch script for the reward service.
#
# 1. Starts vLLM for GREEN on GPU 0 (background)
# 2. Waits for vLLM health check
# 3. Starts FastAPI reward service on port 9100
#
# Usage:
#   bash reward_service/start.sh
#   bash reward_service/start.sh --no-green   # skip GREEN vLLM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults (override via env vars)
GREEN_MODEL="${GREEN_MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--StanfordAIMI--GREEN-RadLlama2-7b}"
GREEN_PORT="${GREEN_VLLM_PORT:-9101}"
GREEN_GPU="${GPU_GREEN:-0}"
SERVICE_PORT="${SERVICE_PORT:-9100}"

SKIP_GREEN=false
if [[ "${1:-}" == "--no-green" ]]; then
    SKIP_GREEN=true
fi

cleanup() {
    echo "[start.sh] Shutting down..."
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Step 1: Start vLLM for GREEN ---
if [[ "$SKIP_GREEN" == "false" ]]; then
    echo "[start.sh] Starting vLLM for GREEN on GPU $GREEN_GPU, port $GREEN_PORT..."
    CUDA_VISIBLE_DEVICES="$GREEN_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$GREEN_MODEL" \
        --port "$GREEN_PORT" \
        --dtype float16 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.85 \
        &
    VLLM_PID=$!

    # --- Step 2: Wait for vLLM health check ---
    echo "[start.sh] Waiting for vLLM to be ready..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${GREEN_PORT}/health" > /dev/null 2>&1; then
            echo "[start.sh] vLLM is ready (attempt $i)."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[start.sh] ERROR: vLLM process died."
            exit 1
        fi
        sleep 2
    done
    if ! curl -sf "http://localhost:${GREEN_PORT}/health" > /dev/null 2>&1; then
        echo "[start.sh] ERROR: vLLM did not become ready in 240s."
        exit 1
    fi
else
    echo "[start.sh] Skipping GREEN vLLM (--no-green)."
fi

# --- Step 3: Start FastAPI reward service ---
echo "[start.sh] Starting reward service on port $SERVICE_PORT..."
export GREEN_VLLM_BASE_URL="http://localhost:${GREEN_PORT}/v1"
cd "$PROJECT_DIR"
python -m uvicorn reward_service.main:app \
    --host 0.0.0.0 \
    --port "$SERVICE_PORT" \
    --log-level info

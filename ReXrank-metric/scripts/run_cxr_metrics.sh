#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --job-name="rexrank-cxr"
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cxr_%j.log
#SBATCH --error=logs/cxr_%j.err

# ============================================================
# Step 0-1: JSON → CSV + CXR-Report-Metric (BLEU, BERTScore,
#           SembScore, RadGraph, RadCliQ)
# Conda env: radgraph_h200 (supports H200 GPU)
# ============================================================
#
# Usage (from rexrank-metric/):
#   bash scripts/run_cxr_metrics.sh --json data/submission_example.json --model ExampleModel --dataset iu_xray
#   bash scripts/run_cxr_metrics.sh --model ExampleModel --dataset iu_xray   # skip JSON conversion
#   sbatch scripts/run_cxr_metrics.sh --json data/submission_example.json --model ExampleModel --dataset iu_xray
# ============================================================

set -e

# --- Set Hugging Face cache to use local models (avoid downloading) ---
export HF_HOME=/mnt/shared-storage-user/ai4good1-share/hf_hub
export TRANSFORMERS_CACHE=/mnt/shared-storage-user/ai4good1-share/hf_hub

# --- Initialize conda for non-interactive shell ---
CONDA_DIR="${CONDA_DIR:-/mnt/shared-storage-user/ai4good1-share/xieyuejin/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"

source "$CONDA_DIR/etc/profile.d/conda.sh"

# --- Parse arguments ---
JSON_PATH=""
MODEL=""
DATASET=""
SPLITS="findings reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --json)    JSON_PATH="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --splits)  SPLITS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/run_cxr_metrics.sh --json <path> --model <name> --dataset <name> [--splits 'findings reports']"
    exit 1
fi

# Resolve rexrank-metric root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$ROOT_DIR/data"
RESULTS_ROOT="$ROOT_DIR/results"

mkdir -p "$ROOT_DIR/logs"

# --- Step 0: JSON → CSV (if JSON provided) ---
if [ -n "$JSON_PATH" ]; then
    echo "=== Step 0: Convert JSON → CSV ==="
    "$CONDA_DIR/envs/radgraph_h200/bin/python" "$SCRIPT_DIR/json_to_csv.py" \
        --json "$JSON_PATH" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --splits $SPLITS \
        --output-root "$DATA_ROOT"
fi

# --- Step 1: CXR-Report-Metric ---
conda activate radgraph_h200

echo "=== Step 1: BLEU, BERTScore, SembScore, RadGraph, RadCliQ ==="
echo "Using Python: $CONDA_DIR/envs/radgraph_h200/bin/python"
"$CONDA_DIR/envs/radgraph_h200/bin/python" "$SCRIPT_DIR/run_cxr_metrics.py" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --splits $SPLITS \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT"

echo "=== Done (CXR metrics) ==="
echo "Results: $RESULTS_ROOT/${DATASET}_findings/ and $RESULTS_ROOT/${DATASET}_reports/"

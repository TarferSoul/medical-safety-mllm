#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --job-name="rexrank-green"
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --output=logs/green_%j.log
#SBATCH --error=logs/green_%j.err

# ============================================================
# Steps 2-5: RaTEScore + GREEN + Aggregate + Leaderboard
# Conda env: green_score
# GPU required (for GREEN model)
# ============================================================
#
# Usage (from rexrank-metric/):
#   bash scripts/run_green_ratescore.sh --model ExampleModel --dataset iu_xray
#   sbatch scripts/run_green_ratescore.sh --model ExampleModel --dataset iu_xray
# ============================================================

set -e

# --- Set Hugging Face cache to use local models (avoid downloading) ---
export HF_HOME=/mnt/shared-storage-user/ai4good1-share/hf_hub
export TRANSFORMERS_CACHE=/mnt/shared-storage-user/ai4good1-share/hf_hub

# --- Initialize conda for non-interactive shell ---
# --- Locate conda (works in docker + sbatch) ---
CONDA_DIR="${CONDA_DIR:-/mnt/shared-storage-user/ai4good1-share/xieyuejin/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# --- Parse arguments ---
MODEL=""
DATASET=""
SPLITS="findings reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --splits)  SPLITS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/run_green_ratescore.sh --model <name> --dataset <name> [--splits 'findings reports']"
    exit 1
fi

# Resolve rexrank-metric root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$ROOT_DIR/data"
RESULTS_ROOT="$ROOT_DIR/results"

mkdir -p "$ROOT_DIR/logs"

conda activate green_score

echo "=== Step 2: RaTEScore ==="
python "$SCRIPT_DIR/run_ratescore.py" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --splits $SPLITS \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT"

echo "=== Step 3: GREEN Score ==="
python "$SCRIPT_DIR/run_green.py" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --splits $SPLITS \
    --data-root "$DATA_ROOT" \
    --results-root "$RESULTS_ROOT"

echo "=== Step 4: Aggregate Metrics ==="
python "$SCRIPT_DIR/aggregate_metrics.py" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --splits $SPLITS \
    --results-root "$RESULTS_ROOT" \
    --output-root "$RESULTS_ROOT/metric"

echo "=== Step 5: Build Leaderboard ==="
python "$SCRIPT_DIR/summarize_leaderboard.py" \
    --datasets "$DATASET" \
    --splits $SPLITS \
    --metric-root "$RESULTS_ROOT/metric" \
    --output-dir "$RESULTS_ROOT/metric/results_summary"

echo "=== Done (GREEN + RaTEScore + Aggregate) ==="
echo "Results: $RESULTS_ROOT/metric/$DATASET/"

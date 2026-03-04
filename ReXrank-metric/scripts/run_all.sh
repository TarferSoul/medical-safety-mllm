#!/bin/bash

# ============================================================
# ReXrank Metric Evaluation Pipeline — Run All
# ============================================================
#
# Convenience wrapper that runs both scripts sequentially:
#   1) run_cxr_metrics.sh  (conda: radgraph, GPU)
#   2) run_green_ratescore.sh (conda: green_score, GPU)
#
# For SLURM, submit each script separately:
#   sbatch scripts/run_cxr_metrics.sh --json data/submission_example.json --model ExampleModel --dataset iu_xray
#   sbatch scripts/run_green_ratescore.sh --model ExampleModel --dataset iu_xray
#
# Usage (interactive, from rexrank-metric/):
#   bash scripts/run_all.sh --json data/submission_example.json --model ExampleModel --dataset iu_xray
#   bash scripts/run_all.sh --model ExampleModel --dataset iu_xray
# ============================================================

# 1) point to your conda install
export CONDA_DIR=/mnt/shared-storage-user/ai4good1-share/xieyuejin/miniconda3

# 2) enable "conda activate" in this shell
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 3) set Hugging Face cache to use local models (avoid downloading)
export HF_HOME=/mnt/shared-storage-user/ai4good1-share/hf_hub
export TRANSFORMERS_CACHE=/mnt/shared-storage-user/ai4good1-share/hf_hub

# 4) activate an env (not really needed since scripts activate their own)
# conda activate dl
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments to extract model, dataset, and splits
MODEL=""
DATASET=""
SPLITS="findings reports"

# Save original arguments
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --splits)  SPLITS="$2"; shift 2 ;;
        *) shift ;;  # Skip other arguments like --json
    esac
done

# Pass all original arguments to CXR metrics script (includes --json)
bash "$SCRIPT_DIR/run_cxr_metrics.sh" "${ORIG_ARGS[@]}"

# Pass only relevant arguments to GREEN/RaTEScore script (no --json)
bash "$SCRIPT_DIR/run_green_ratescore.sh" --model "$MODEL" --dataset "$DATASET" --splits "$SPLITS"

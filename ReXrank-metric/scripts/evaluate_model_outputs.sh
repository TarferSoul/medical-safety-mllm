#!/bin/bash

# ============================================================
# ReXrank Metric Evaluation — Batch Evaluation Script
# ============================================================
#
# This script evaluates all JSON output files from a given model directory.
# It automatically detects all .json files and infers dataset names from filenames.
#
# Usage (from rexrank-metric/ or any directory):
#   bash scripts/evaluate_model_outputs.sh /path/to/model/outputs
#   bash scripts/evaluate_model_outputs.sh /mnt/shared-storage-user/xieyuejin/MLLM-Safety/MedicalSafety/results/Qwen3-8B-VL-Chestall-v260222-full-sft
#
# Optional parameters:
#   --splits "findings reports"  (default: "findings reports")
#   --dry-run                    (show commands without executing)
#
# Example:
#   bash scripts/evaluate_model_outputs.sh /path/to/outputs --splits "reports"
#   bash scripts/evaluate_model_outputs.sh /path/to/outputs --dry-run
# ============================================================

set -e

# ============================================================
# Configuration
# ============================================================

# 1) Point to your conda install
export CONDA_DIR=/mnt/shared-storage-user/ai4good1-share/xieyuejin/miniconda3

# 2) Enable "conda activate" in this shell
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 3) Set Hugging Face cache to use local models (avoid downloading)
export HF_HOME=/mnt/shared-storage-user/ai4good1-share/hf_hub
export TRANSFORMERS_CACHE=/mnt/shared-storage-user/ai4good1-share/hf_hub

# ============================================================
# Parse Arguments
# ============================================================

if [ $# -eq 0 ]; then
    echo "Error: No model output directory provided"
    echo "Usage: bash $0 <model_output_directory> [--splits \"findings reports\"] [--dry-run]"
    echo "Example: bash $0 /mnt/shared-storage-user/xieyuejin/MLLM-Safety/MedicalSafety/results/Qwen3-8B-VL-Chestall-v260222-full-sft"
    exit 1
fi

MODEL_OUTPUT_DIR="$1"
shift

SPLITS="findings reports"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate Input Directory
# ============================================================

if [ ! -d "$MODEL_OUTPUT_DIR" ]; then
    echo "Error: Directory does not exist: $MODEL_OUTPUT_DIR"
    exit 1
fi

# Get absolute path
MODEL_OUTPUT_DIR="$(cd "$MODEL_OUTPUT_DIR" && pwd)"

# Extract model name from directory path (last component)
MODEL_NAME="$(basename "$MODEL_OUTPUT_DIR")"

echo "=================================================="
echo "ReXrank Batch Evaluation"
echo "=================================================="
echo "Model Output Directory: $MODEL_OUTPUT_DIR"
echo "Model Name: $MODEL_NAME"
echo "Evaluation Splits: $SPLITS"
echo "Dry Run: $DRY_RUN"
echo "=================================================="
echo ""

# ============================================================
# Find Script Directory
# ============================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ALL_SCRIPT="$SCRIPT_DIR/run_all.sh"

if [ ! -f "$RUN_ALL_SCRIPT" ]; then
    echo "Error: run_all.sh not found at: $RUN_ALL_SCRIPT"
    exit 1
fi

# ============================================================
# Find and Process JSON Files
# ============================================================

# Find all .json files in the model output directory
JSON_FILES=("$MODEL_OUTPUT_DIR"/*.json)

if [ ! -e "${JSON_FILES[0]}" ]; then
    echo "Error: No .json files found in $MODEL_OUTPUT_DIR"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON file(s) to process:"
for json_file in "${JSON_FILES[@]}"; do
    echo "  - $(basename "$json_file")"
done
echo ""

# ============================================================
# Process Each JSON File
# ============================================================

TOTAL_FILES=${#JSON_FILES[@]}
CURRENT=0
FAILED_FILES=()

for json_file in "${JSON_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    filename="$(basename "$json_file")"

    echo "=================================================="
    echo "[$CURRENT/$TOTAL_FILES] Processing: $filename"
    echo "=================================================="

    # Infer dataset name from filename
    # Examples:
    #   iu_xray_test.json -> iu_xray
    #   mimic-cxr_test.json -> mimic-cxr
    #   chexpert_plus.json -> chexpert_plus

    # Remove .json extension
    base_name="${filename%.json}"

    # Try to extract dataset name by removing common suffixes
    dataset_name="$base_name"
    dataset_name="${dataset_name%_test}"
    dataset_name="${dataset_name%_valid}"
    dataset_name="${dataset_name%_val}"
    dataset_name="${dataset_name%_train}"

    echo "Inferred dataset: $dataset_name"
    echo "JSON file: $json_file"
    echo ""

    # Build the command
    CMD="bash \"$RUN_ALL_SCRIPT\" --json \"$json_file\" --model \"$MODEL_NAME\" --dataset \"$dataset_name\" --splits \"$SPLITS\""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute:"
        echo "  $CMD"
        echo ""
    else
        echo "Executing:"
        echo "  $CMD"
        echo ""

        # Execute the command
        if bash "$RUN_ALL_SCRIPT" --json "$json_file" --model "$MODEL_NAME" --dataset "$dataset_name" --splits "$SPLITS"; then
            echo ""
            echo "✓ Successfully processed: $filename"
            echo ""
        else
            echo ""
            echo "✗ Failed to process: $filename"
            echo ""
            FAILED_FILES+=("$filename")
        fi
    fi
done

# ============================================================
# Summary
# ============================================================

echo "=================================================="
echo "Batch Evaluation Complete"
echo "=================================================="
echo "Total files processed: $TOTAL_FILES"

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo "Failed files: ${#FAILED_FILES[@]}"
    for failed_file in "${FAILED_FILES[@]}"; do
        echo "  - $failed_file"
    done
    exit 1
else
    echo "All files processed successfully!"
fi
echo "=================================================="

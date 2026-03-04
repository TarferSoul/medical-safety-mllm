#!/bin/bash
# Format reasoning data for training with <think> tags

# Configuration
REASONING_FILE="reasoning_results/reasoning_results_20251229_144040.json"
NORMALIZED_FILE="dataset/mimic_cxr_sharegpt_train_normalized_v2.json"  # Optional
OUTPUT_DIR="dataset"
SPLIT_RATIO=0.1

echo "=========================================="
echo "Format Reasoning Data for Training"
echo "=========================================="
echo ""

# Check if reasoning file exists
if [ ! -f "$REASONING_FILE" ]; then
    echo "❌ Error: Reasoning file not found: $REASONING_FILE"
    echo "Please run generate_reasoning.py first to generate reasoning data."
    exit 1
fi

echo "Input files:"
echo "  - Reasoning: $REASONING_FILE"
if [ -f "$NORMALIZED_FILE" ]; then
    echo "  - Normalized reports: $NORMALIZED_FILE (will use normalized reports)"
    USE_NORMALIZED="--normalized $NORMALIZED_FILE"
else
    echo "  - Normalized reports: Not found, using original reports"
    USE_NORMALIZED=""
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Train/Test split ratio: $SPLIT_RATIO"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

# Format the data
echo ""
echo "Formatting reasoning data with <think> tags..."
python3 format_reasoning_for_training.py \
    --reasoning "$REASONING_FILE" \
    --output "$OUTPUT_DIR/thinking_training_data.json" \
    $USE_NORMALIZED \
    --split $SPLIT_RATIO

if [ $? -ne 0 ]; then
    echo "❌ Formatting failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Formatting completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
if [ "$SPLIT_RATIO" != "0" ]; then
    echo "  - Training set: $OUTPUT_DIR/thinking_training_data_train.json"
    echo "  - Test set:     $OUTPUT_DIR/thinking_training_data_test.json"
else
    echo "  - Combined:     $OUTPUT_DIR/thinking_training_data.json"
fi
echo "  - Summary:      $OUTPUT_DIR/thinking_training_data_summary.json"
echo ""
echo "Next steps:"
echo "  1. Review the formatted data"
echo "  2. Update dataset_info.json to include this dataset"
echo "  3. Train the model with thinking capability"
echo ""

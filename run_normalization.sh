#!/bin/bash
# Batch normalization script for MIMIC-CXR dataset

# Configuration
MODEL="QwQ"  # or "Qwen3-235B-A22B" for better quality
CONCURRENCY=20
MAX_RETRIES=5

echo "=========================================="
echo "MIMIC-CXR Report Normalization Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENCY"
echo ""

# Step 1: Test on small sample first (100 samples)
echo "[Step 1/3] Testing normalization on 100 test samples..."
python3 normalize_reports.py \
    --input dataset/mimic_cxr_sharegpt_test.json \
    --output dataset/mimic_cxr_sharegpt_test_normalized_sample100.json \
    --model $MODEL \
    --concurrency $CONCURRENCY \
    --max_retries $MAX_RETRIES \
    --max_samples 100

if [ $? -ne 0 ]; then
    echo "❌ Test normalization failed! Please check the errors above."
    exit 1
fi

echo ""
echo "✓ Test completed successfully! Please review the results:"
echo "  - Output: dataset/mimic_cxr_sharegpt_test_normalized_sample100.json"
echo "  - Summary: dataset/mimic_cxr_sharegpt_test_normalized_sample100_summary.json"
echo ""
read -p "Continue with full test set normalization? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Normalization stopped by user."
    exit 0
fi

# Step 2: Normalize full test set (2,219 samples)
echo ""
echo "[Step 2/3] Normalizing full test set (2,219 samples)..."
echo "This will take approximately 10-15 minutes..."
python3 normalize_reports.py \
    --input dataset/mimic_cxr_sharegpt_test.json \
    --output dataset/mimic_cxr_sharegpt_test_normalized.json \
    --model $MODEL \
    --concurrency $CONCURRENCY \
    --max_retries $MAX_RETRIES

if [ $? -ne 0 ]; then
    echo "❌ Test set normalization failed!"
    exit 1
fi

echo ""
echo "✓ Test set normalization completed!"
echo ""
read -p "Continue with training set normalization? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Normalization stopped by user."
    exit 0
fi

# Step 3: Normalize full training set (19,972 samples)
echo ""
echo "[Step 3/3] Normalizing full training set (19,972 samples)..."
echo "This will take approximately 1-2 hours..."
python3 normalize_reports.py \
    --input dataset/mimic_cxr_sharegpt_train.json \
    --output dataset/mimic_cxr_sharegpt_train_normalized.json \
    --model $MODEL \
    --concurrency $CONCURRENCY \
    --max_retries $MAX_RETRIES

if [ $? -ne 0 ]; then
    echo "❌ Training set normalization failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All normalization completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Test set:     dataset/mimic_cxr_sharegpt_test_normalized.json"
echo "  - Training set: dataset/mimic_cxr_sharegpt_train_normalized.json"
echo ""
echo "Summary files:"
echo "  - Test summary:     dataset/mimic_cxr_sharegpt_test_normalized_summary.json"
echo "  - Training summary: dataset/mimic_cxr_sharegpt_train_normalized_summary.json"
echo ""
echo "Next steps:"
echo "  1. Review the normalized reports to ensure quality"
echo "  2. Update dataset_info.json to use normalized datasets"
echo "  3. Retrain the model with normalized data"
echo ""

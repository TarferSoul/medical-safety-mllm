#!/bin/bash
# ============================================================
# Setup radgraph_h200 environment for H200 GPU
# Ensures PyTorch 2.1+, Transformers 4.25+, AllenNLP 2.10+
# ============================================================

set -e

CONDA_DIR="/mnt/shared-storage-user/ai4good1-share/xieyuejin/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"

ENV_NAME="radgraph_h200"

echo "=========================================="
echo "Creating conda environment: $ENV_NAME"
echo "=========================================="

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment with Python 3.10
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
conda activate $ENV_NAME

echo ""
echo "=========================================="
echo "Step 1: Install PyTorch 2.1.0 (CUDA 12.1)"
echo "=========================================="
pip install torch==2.1.0 torchvision==0.16.0 
# --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=========================================="
echo "Step 2: Install Transformers and HF Hub"
echo "=========================================="
pip install transformers==4.36.0 huggingface-hub==0.20.0 tokenizers==0.15.0

echo ""
echo "=========================================="
echo "Step 3: Install AllenNLP (with constraints)"
echo "=========================================="
# Install AllenNLP with --no-deps first, then manually install compatible dependencies
pip install allennlp==2.10.1 --no-deps
pip install allennlp-models==2.10.1 --no-deps

# Install AllenNLP dependencies (without version conflicts)
pip install \
    jsonnet==0.21.0 \
    nltk==3.8.1 \
    spacy==3.7.0 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    fairscale==0.4.13 \
    sentencepiece==0.2.0 \
    wandb==0.16.0 \
    tensorboardX==2.6.2.2 \
    cached-path==1.5.1 \
    h5py==3.10.0 \
    lmdb==1.4.1

echo ""
echo "=========================================="
echo "Step 4: Install BERTScore and metrics"
echo "=========================================="
pip install \
    bert-score==0.3.13 \
    fast-bleu==0.0.4 \
    statsmodels==0.14.1

echo ""
echo "=========================================="
echo "Step 5: Install general dependencies"
echo "=========================================="
pip install \
    pandas==2.0.3 \
    numpy==1.24.4 \
    tqdm==4.66.1 \
    Pillow==10.1.0

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""

# Test imports
echo "Testing imports..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')

import transformers
print(f'✅ Transformers {transformers.__version__}')

import allennlp
print(f'✅ AllenNLP {allennlp.__version__}')

from bert_score import BERTScorer
print(f'✅ BERTScore OK')

print('')
print('🎉 All components installed successfully!')
"

echo ""
echo "=========================================="
echo "Setup complete! Environment: $ENV_NAME"
echo "=========================================="

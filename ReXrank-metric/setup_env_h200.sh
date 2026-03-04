#!/bin/bash
# ============================================================
# Setup radgraph_h200 environment for H200 GPU
# Run with: bash setup_env_h200.sh
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
echo "Step 1: Install PyTorch 2.1.0 via conda"
echo "=========================================="
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "=========================================="
echo "Step 2: Install Transformers & HF Hub"
echo "=========================================="
pip install transformers==4.36.0 tokenizers==0.15.0 huggingface-hub==0.20.0

echo ""
echo "=========================================="
echo "Step 3: Install core ML/NLP libraries"
echo "=========================================="
pip install \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    pandas==2.0.3 \
    numpy==1.24.4

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
echo "Step 5: Install spaCy (for AllenNLP)"
echo "=========================================="
pip install spacy==3.7.0

echo ""
echo "=========================================="
echo "Step 6: Install AllenNLP dependencies first"
echo "=========================================="
pip install \
    jsonnet==0.21.0 \
    nltk==3.8.1 \
    sentencepiece==0.2.0 \
    cached-path==1.5.1 \
    fairscale==0.4.13 \
    h5py==3.10.0 \
    wandb==0.16.0 \
    tensorboardX==2.6.2.2

echo ""
echo "=========================================="
echo "Step 7: Install AllenNLP (avoid version conflicts)"
echo "=========================================="
# Install without dependencies first
pip install allennlp==2.10.1 --no-deps
pip install allennlp-models==2.10.1 --no-deps

# Install remaining AllenNLP dependencies manually
pip install \
    lmdb==1.4.1 \
    overrides==7.4.0 \
    more-itertools==10.1.0

echo ""
echo "=========================================="
echo "Step 8: Install additional utilities"
echo "=========================================="
pip install tqdm==4.66.1 Pillow==10.1.0

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""

# Test imports
echo "Testing environment..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA compute capability: {torch.cuda.get_device_capability(0)}")

import transformers
print(f"✅ Transformers: {transformers.__version__}")

try:
    import allennlp
    print(f"✅ AllenNLP: {allennlp.__version__}")
except Exception as e:
    print(f"⚠️  AllenNLP: {e}")

from bert_score import BERTScorer
print(f"✅ BERTScore: OK")

import pandas, numpy, scipy, sklearn
print(f"✅ Scientific libs: OK")

print("")
print("🎉 Environment ready!")
print(f"   Activate with: conda activate {sys.argv[1] if len(sys.argv) > 1 else 'radgraph_h200'}")
EOF

echo ""
echo "=========================================="
echo "To activate this environment, run:"
echo "  conda activate $ENV_NAME"
echo "=========================================="

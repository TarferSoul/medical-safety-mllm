---
title: hyzhouMedVersa
sdk: gradio
sdk_version: 4.24.0
---
# MedVersa: A Generalist Learner for Multifaceted Medical Image Interpretation

A model implementation for our paper [A Generalist Learner for Multifaceted Medical Image Interpretation](https://arxiv.org/abs/2405.07988).

MedVersa is a compound medical AI system that coordinates multimodal inputs, orchestrates models and tools for various medical imaging tasks, and generates multimodal outputs.

## Installation

### Prerequisites

- [Python](https://www.python.org/)
- [NVIDIA CUDA-compatible GPU](https://developer.nvidia.com/cuda-gpus)
- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or [Anaconda](https://www.anaconda.com/)

### Environment Setup

1. Create and activate the conda environment:

```bash
# For NVIDIA A100 GPUs
conda env create -f environment.yml
conda activate medversa

# For NVIDIA H100 GPUs (CUDA 11.8)
conda env create -f environment_cu118.yml
conda activate medversa
```

### Troubleshooting

If you encounter dependency issues, try these solutions:

```bash
# Fix OpenCV issues
pip install opencv-contrib-python

# Fix incompatible torchvision version
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Inference

```python
from utils import *
from torch import cuda

# Initialize model
device = 'cuda' if cuda.is_available() else 'cpu'
model_cls = registry.get_model_class('medomni') # medomni is the architecture name :)
model = model_cls.from_pretrained('hyzhou/MedVersa').to(device).eval()

# Define example input
example = {
    'images': ["./demo_ex/c536f749-2326f755-6a65f28f-469affd2-26392ce9.png"],
    'context': "Age:30-40.\nGender:F.\nIndication: ___-year-old female with end-stage renal disease not on dialysis presents with dyspnea. PICC line placement.\nComparison: None.",
    'prompt': "How would you characterize the findings from <img0>?",
    'modality': "cxr",
    'task': "report generation"
}

# Configure generation parameters
params = {
    'num_beams': 1,
    'do_sample': True,
    'min_length': 1,
    'top_p': 0.9,
    'repetition_penalty': 1,
    'length_penalty': 1,
    'temperature': 0.1
}

# Generate predictions
seg_mask_2d, seg_mask_3d, output_text = generate_predictions(
    model, 
    example['images'],
    example['context'],
    example['prompt'],
    example['modality'],
    example['task'],
    **params,
    device,
)
print(output_text)
```

For more detailed examples and usage scenarios, see `inference.py`.

## Prompts
More prompts can be found in `medomni/datasets/prompts.json`.

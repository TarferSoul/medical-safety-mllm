# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical imaging safety project focused on fine-tuning multimodal large language models (MLLMs) on medical X-ray data. The project uses LLaMA-Factory to train vision-language models (particularly Qwen3-VL) to generate medical reports from chest X-ray images using the MIMIC-CXR dataset.

## Repository Structure

- **`data_processing/`**: Data preparation and conversion scripts
  - `convert_mimic_to_sharegpt.py`: Convert MIMIC-CXR to ShareGPT format
  - `format_reasoning_for_training.py`: Format reasoning data with `<think>` tags
  - `normalize_reports.py` / `normalize_reports_v2.py`: Standardize report formats
  - `generate_llamafactory_dataset.py`: Generate dataset configurations
  - See `data_processing/README.md` for detailed documentation

- **`evaluation/`**: Model evaluation and testing scripts
  - `evaluate_and_judge.py`: **Integrated prediction + judgment workflow** (recommended)
  - `evaluate_model.py`: Generate predictions only
  - `llm_as_judge.py`: Judge predictions using LLM evaluator
  - `generate_reasoning.py`: Generate diagnostic reasoning process
  - See `evaluation/README.md` for detailed documentation

- **`utils/`**: Utility and testing scripts
  - `test_api.py`: Test model API connectivity
  - See `utils/README.md` for detailed documentation

- **`dataset/`**: Training and test data in ShareGPT format
  - `dataset_info.json`: Dataset configuration for LLaMA-Factory
  - `mimic_cxr_sharegpt_train.json`: Training data
  - `mimic_cxr_sharegpt_test.json`: Test data

- **`evaluation_results/`**: Evaluation outputs organized by model
  - `pred_{model_name}/`: Results for each prediction model
  - Includes predictions, judgments, and summary statistics

- **`reasoning_results/`**: Reasoning generation outputs organized by model
  - `reasoning_{model_name}/`: Reasoning data for each model

- **`LLaMA-Factory/`**: Submodule containing the training framework
  - `train/`: Custom training scripts and configurations
  - `examples/`: Example training configurations (LoRA, full fine-tuning, etc.)
  - `src/llamafactory/`: Core library code

## Key Commands

### Environment Setup

Set up CUDA environment before training:
```bash
export CUDA_HOME=/mnt/shared-storage-user/xieyuejin/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Data Preparation

Convert MIMIC-CXR dataset to ShareGPT format:
```bash
cd data_processing

# Default: 500 test samples
python convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output ../dataset/mimic_cxr_sharegpt.json

# Custom test sample count
python convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output ../dataset/mimic_cxr_sharegpt.json \
  --test_samples 1000 \
  --random_seed 42

# Use percentage split instead
python convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output ../dataset/mimic_cxr_sharegpt.json \
  --test_samples 0 \
  --test_split 0.1 \
  --random_seed 42
```

### Training

Train a model using LLaMA-Factory:
```bash
cd LLaMA-Factory

# Full fine-tuning with DeepSpeed
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_vl_8b_full_sft.yaml

# LoRA fine-tuning
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

The training configuration is specified in YAML files. Key parameters:
- `model_name_or_path`: Path to base model
- `dataset`: Name of dataset (defined in dataset_info.json)
- `finetuning_type`: `full`, `lora`, or `freeze`
- `deepspeed`: Path to DeepSpeed config for distributed training
- `output_dir`: Where to save checkpoints

### Model Testing

Test model via OpenAI-compatible API:
```bash
cd utils
python test_api.py
```

### Model Evaluation

**Recommended: Integrated evaluation (prediction + judgment)**
```bash
cd evaluation
python evaluate_and_judge.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --pred_model "Qwen3-8B-VL-Mimic" \
  --judge_model "QwQ" \
  --concurrency 50 \
  --max_samples 100
```

**Alternative: Separate prediction and judgment**
```bash
cd evaluation

# Step 1: Generate predictions
python evaluate_model.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --max_samples 10

# Step 2: Judge predictions
python llm_as_judge.py \
  --input ../evaluation_results/pred_ModelName/evaluation_results_xxx.json \
  --judge_model "QwQ"
```

Results are saved to `evaluation_results/pred_{model_name}/` with timestamped filenames.

## Architecture Notes

### Data Format

The project uses ShareGPT format for training data:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image><image>Generate a medical imaging report based on the X-ray image results."
    },
    {
      "from": "gpt",
      "value": "FINAL REPORT\n EXAMINATION: CHEST (PA AND LAT)\n..."
    }
  ],
  "images": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg"
  ]
}
```

Multiple `<image>` tags correspond to multiple images in the `images` array.

### Dataset Configuration

Datasets must be registered in `dataset/dataset_info.json`:
```json
{
  "dataset_name": {
    "file_name": "dataset_file.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

### Training Configuration Workflow

1. **Dataset reference**: YAML configs reference datasets by name (e.g., `mimic_cxr_sharegpt_train`)
2. **Dataset directory**: The `dataset_dir` parameter points to the directory containing data files (typically `../dataset` relative to LLaMA-Factory)
3. **Model loading**: Models are loaded from `model_name_or_path`
4. **Output**: Checkpoints saved to `output_dir`

### Vision-Language Model Training

For multimodal models like Qwen3-VL:
- `freeze_vision_tower: true` - Keep vision encoder frozen
- `freeze_multi_modal_projector: true` - Keep projection layer frozen
- `freeze_language_model: false` - Fine-tune language model
- `image_max_pixels` and `video_max_pixels` control input resolution

### DeepSpeed Integration

The project uses DeepSpeed for distributed training. Common configurations:
- `ds_z3_config.json`: ZeRO Stage 3 for maximum memory efficiency
- `ds_z2_config.json`: ZeRO Stage 2 for balanced performance
- `ds_z0_config.json`: No ZeRO optimization

DeepSpeed is automatically triggered when `FORCE_TORCHRUN=1` is set.

## API Integration

Models are deployed with an OpenAI-compatible API. Authentication uses Basic Auth with base64-encoded credentials:
```python
auth_string = f"{API_AK}:{API_SK}"
b64_auth_string = base64.b64encode(auth_string.encode()).decode()

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=b64_auth_string,
    default_headers={"Authorization": f"Basic {b64_auth_string}"}
)
```

Images must be base64-encoded when sent via API.

## Important File Paths

Training data locations are typically under:
- `/mnt/shared-storage-user/ai4good1-share/xieyuejin/gdown/gdrive/` - MIMIC-CXR images
- `/mnt/shared-storage-user/ai4good1-share/hf_hub/` - Pre-trained models

Output locations:
- `/mnt/shared-storage-user/ai4good1-share/xieyuejin/models/` - Trained model checkpoints
- `./evaluation_results/` - Evaluation outputs (local to project)

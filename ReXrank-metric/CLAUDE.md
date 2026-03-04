# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReXrank evaluation pipeline for radiology report generation. Computes 7 metrics (BLEU-2, BERTScore, SembScore, RadGraph, RadCliQ-v1, RaTEScore, GREEN) to evaluate model-generated reports against ground-truth references.

## Pipeline Architecture

The pipeline runs in **5 sequential steps** across **2 separate conda environments**:

| Step | Script | Conda Env | GPU | Output |
|------|--------|-----------|-----|--------|
| 0 | `json_to_csv.py` | radgraph | No | CSVs from JSON submission |
| 1 | `run_cxr_metrics.py` | radgraph | Yes | BLEU, BERTScore, SembScore, RadGraph, RadCliQ |
| 2 | `run_ratescore.py` | green_score | Optional | RaTEScore |
| 3 | `run_green.py` | green_score | Yes (16GB VRAM) | GREEN score |
| 4 | `aggregate_metrics.py` | either | No | Per-model summary CSV |
| 5 | `summarize_leaderboard.py` | either | No | Ranked leaderboard CSV |

**Primary ranking metric:** `1/RadCliQ-v1` (higher = better)

## Environment Setup

Two conda environments are required due to dependency conflicts:

### Environment 1: `radgraph` (Steps 0-1)
```bash
conda create -n radgraph python=3.8 -y
conda activate radgraph
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.25.1 bert-score==0.3.13 fast-bleu==0.0.4 scikit-learn==1.2.2 pandas numpy scipy tqdm
```

### Environment 2: `green_score` (Steps 2-5)
```bash
conda create -n green_score python=3.10 -y
conda activate green_score
pip install torch==2.1.0 transformers==4.36.0 pandas numpy ratescore
# GREEN: git clone https://github.com/Stanford-AIMI/GREEN.git && cd GREEN && pip install -e .
```

## Required Checkpoint Downloads

Place in `scripts/CXR-Report-Metric/`:
- `CheXbert/models/chexbert.pth` — from [stanfordmlgroup/CheXbert](https://github.com/stanfordmlgroup/CheXbert)
- `radgraph/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz` — from [PhysioNet RadGraph](https://physionet.org/content/radgraph/1.0.0/) (requires credentialed access)

## Key Commands

### End-to-end from JSON submission
```bash
# Interactive (runs both scripts, switches conda envs automatically)
bash scripts/run_all.sh --json data/submission_example.json --model MyModel --dataset mimic-cxr

# SLURM (submit separately, wait for job 1 before job 2)
sbatch scripts/run_cxr_metrics.sh --json data/submission_example.json --model MyModel --dataset mimic-cxr
sbatch scripts/run_green_ratescore.sh --model MyModel --dataset mimic-cxr
```

### Individual steps (from `rexrank-metric/`)
```bash
# Step 0: JSON → CSV
python scripts/json_to_csv.py --json data/submission.json --model MyModel --dataset mimic-cxr

# Step 1: CXR metrics (conda: radgraph)
python scripts/run_cxr_metrics.py --models MyModel --datasets mimic-cxr --splits findings reports

# Step 2: RaTEScore (conda: green_score)
python scripts/run_ratescore.py --models MyModel --datasets mimic-cxr --splits findings reports

# Step 3: GREEN (conda: green_score)
python scripts/run_green.py --models MyModel --datasets mimic-cxr --splits findings reports

# Step 4: Aggregate
python scripts/aggregate_metrics.py --models MyModel --datasets mimic-cxr --splits findings reports

# Step 5: Leaderboard
python scripts/summarize_leaderboard.py --datasets mimic-cxr --splits findings reports
```

## Input/Output Format

### Input: Submission JSON
```json
{
  "patient64545_study1": {
    "section_findings": "Ground truth findings text...",
    "section_impression": "Ground truth impression...",
    "model_prediction": "Model-generated report..."
  }
}
```

### Input: CSV files (alternative)
```
data/findings/{dataset}/gt_reports_{model}.csv       # study_id, case_id, report
data/findings/{dataset}/predicted_reports_{model}.csv
```

### Output locations
```
results/{dataset}_{split}/report_scores_{model}.csv   # Step 1: per-sample CXR metrics
results/{dataset}_{split}/ratescore_{model}.csv       # Step 2: per-sample RaTEScore
results/{dataset}_{split}/results_green_{model}.csv   # Step 3: per-sample GREEN
results/metric/{dataset}/{split}/{model}.csv          # Step 4: aggregated summary
results/metric/results_summary/{dataset}.csv          # Step 5: ranked leaderboard
```

## Datasets

`mimic-cxr`, `iu_xray`, `chexpert_plus`, `gradient_health`

Splits: `findings` (Findings section only), `reports` (Findings + Impression)

## Hardware Requirements

- Step 1 (RadGraph): GPU with >= 8 GB VRAM
- Step 3 (GREEN): GPU with >= 16 GB VRAM (7B LLM)
- RAM: >= 64 GB recommended

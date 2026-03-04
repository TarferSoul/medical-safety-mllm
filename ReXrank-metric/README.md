# ReXrank Evaluation Metrics

[![GitHub](https://img.shields.io/badge/GitHub-ReXrank--metric-blue)](https://github.com/rajpurkarlab/ReXrank-metric)

This repository contains the official evaluation pipeline used to compute the ReXrank leaderboard metrics for radiology report generation. It evaluates model-generated reports against ground-truth references using **7 complementary metrics**.

```bash
git clone git@github.com:rajpurkarlab/ReXrank-metric.git
cd ReXrank-metric
```

## Metrics Overview

| Metric | Column Name | Range | Higher is Better | Description |
|--------|------------|-------|-------------------|-------------|
| **BLEU-2** | `bleu_score` | [0, 1] | Yes | Bigram overlap (weights: 1/2, 1/2) via `fast_bleu` |
| **BERTScore** | `bertscore` | [-1, 1] | Yes | Contextual embedding F1 using `distilroberta-base` with baseline rescaling; IDF disabled |
| **SembScore** | `semb_score` | [-1, 1] | Yes | Cosine similarity of CheXbert embeddings |
| **RadGraph** | `radgraph_combined` | [0, 1] | Yes | Mean of entity F1 and relation F1 from RadGraph (DyGIE++) |
| **1/RadCliQ-v1** | `1/RadCliQ-v1` | (0, +inf) | Yes | Inverse of RadCliQ-v1 composite score (primary ranking metric) |
| **RaTEScore** | `ratescore` | [0, 1] | Yes | Factual/temporal consistency score |
| **GREEN** | `green_score` | [0, 1] | Yes | LLM-based clinical error analysis using `StanfordAIMI/GREEN-radllama2-7b` |

## Pipeline Architecture

```
Submission JSON ({model}.json)
│
┌─────────────────────────────────────────────────────────────────┐
│  run_cxr_metrics.sh          (conda: radgraph, GPU)            │
│                                                                 │
│  Step 0: json_to_csv.py                                        │
│    Converts JSON → gt_reports_{model}.csv                       │
│                   + predicted_reports_{model}.csv                │
│                                                                 │
│  Step 1: run_cxr_metrics.py                                    │
│    Computes: bleu_score, bertscore, semb_score,                │
│              radgraph_combined, RadCliQ-v0, RadCliQ-v1          │
│    Tool: CXR-Report-Metric                                     │
│    https://github.com/rajpurkarlab/CXR-Report-Metric           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  run_green_ratescore.sh      (conda: green_score, GPU)         │
│                                                                 │
│  Step 2: run_ratescore.py                                      │
│    Computes: ratescore                                          │
│    Tool: RaTEScore                                              │
│    https://github.com/MAGIC-AI4Med/RaTEScore                   │
│                                                                 │
│  Step 3: run_green.py                                          │
│    Computes: green_score                                        │
│    Tool: GREEN (StanfordAIMI/GREEN-radllama2-7b)                │
│    https://github.com/Stanford-AIMI/GREEN                      │
│                                                                 │
│  Step 4: aggregate_metrics.py                                  │
│    Averages per-sample scores → one summary CSV per model       │
│    Computes 1/RadCliQ-v1 = 1 / mean(RadCliQ-v1)                │
│                                                                 │
│  Step 5: summarize_leaderboard.py                              │
│    Merges all models, ranks by 1/RadCliQ-v1, outputs leaderboard│
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
rexrank-metric/
├── README.md
├── requirements.txt
├── data/                               # Input data
│   └── submission_example.json         # Example submission JSON (5 samples)
├── results/                            # Example outputs (derived from data/submission_example.json)
│   ├── output_report_scores.csv        # Step 1: per-sample BLEU/BERTScore/SembScore/RadGraph/RadCliQ
│   ├── output_ratescore.csv            # Step 2: per-sample RaTEScore
│   ├── output_green.csv                # Step 3: per-sample GREEN with error breakdown
│   ├── output_aggregated.csv           # Step 4: single-row model summary
│   └── output_leaderboard.csv          # Step 5: ranked leaderboard
├── scripts/
│   ├── CXR-Report-Metric/        # Bundled: BLEU, BERTScore, SembScore, RadGraph, RadCliQ
│   │   ├── CXRMetric/            #   Core metric implementations
│   │   │   ├── run_eval.py       #     calc_metric() entry point
│   │   │   ├── radcliq-v1.pkl    #     RadCliQ-v1 model weights
│   │   │   ├── normalizer.pkl    #     RadCliQ-v0 MinMaxScaler
│   │   │   └── composite_metric_model.pkl
│   │   ├── CheXbert/models/chexbert.pth  # ← download required
│   │   ├── radgraph/.../model.tar.gz     # ← download required (PhysioNet)
│   │   └── config.py             #   Checkpoint path configuration
│   ├── json_to_csv.py            # Step 0: Convert submission JSON → gt/predicted CSVs
│   ├── run_cxr_metrics.py        # Step 1: BLEU, BERTScore, SembScore, RadGraph, RadCliQ
│   ├── run_ratescore.py          # Step 2: RaTEScore
│   ├── run_green.py              # Step 3: GREEN score
│   ├── aggregate_metrics.py      # Step 4: Average per-sample → per-model summary
│   ├── summarize_leaderboard.py  # Step 5: Build ranked leaderboard CSVs
│   ├── run_cxr_metrics.sh        # SLURM script: Steps 0-1 (conda: radgraph, GPU)
│   ├── run_green_ratescore.sh    # SLURM script: Steps 2-5 (conda: green_score, GPU)
│   └── run_all.sh                # Convenience wrapper: runs both .sh scripts
```

All data and results live inside `rexrank-metric/`:

```
rexrank-metric/
├── data/
│   ├── findings/{dataset}/gt_reports_{model}.csv        # Ground truth (findings only)
│   ├── findings/{dataset}/predicted_reports_{model}.csv  # Model predictions (findings)
│   ├── reports/{dataset}/gt_reports_{model}.csv          # Ground truth (full reports)
│   └── reports/{dataset}/predicted_reports_{model}.csv   # Model predictions (reports)
├── results/
│   ├── {dataset}_{split}/                               # Per-sample outputs from Steps 1-3
│   │   ├── report_scores_{model}.csv
│   │   ├── ratescore_{model}.csv
│   │   └── results_green_{model}.csv
│   ├── metric/{dataset}/{split}/{model}.csv             # Per-model aggregated summary (Step 4)
│   └── metric/results_summary/{dataset}.csv             # Final leaderboard (Step 5)
```

Datasets: `mimic-cxr`, `iu_xray`, `chexpert_plus`, `gradient_health`

## Input / Output Examples

See `data/submission_example.json` for the input and `results/` for the corresponding outputs. Each output file is derived from the same 5-sample input.

### Input: Submission JSON (`data/submission_example.json`)

The default submission format is a JSON file keyed by `case_id`. Each entry contains ground-truth fields and the model's prediction:

```json
{
    "patient64545_study1": {
        "image_path": ["valid/patient64545/study1/view1_frontal.png"],
        "context": "Age:82.0.Gender:F.Indication:nanComparison: None.",
        "section_clinical_history": null,
        "section_findings": "There are low lung volumes. The cardiomediastinal silhouette is within normal limits. There is evidence of trace pulmonary edema with a left pleural effusion. Left retrocardiac atelectasis is noted. There are old bilateral rib fractures.",
        "section_impression": "1. Low lung volumes with trace pulmonary edema and left pleural effusion. 2. Left retrocardiac atelectasis. 3. Old bilateral rib fractures.",
        "key_image_path": "valid/patient64545/study1/view1_frontal.png",
        "model_prediction": "Findings: Heart size is normal. The mediastinal contours are within normal limits. There are low lung volumes. There is a small left pleural effusion and associated opacity in the left base, likely representing atelectasis. There is mild pulmonary edema. No right pleural effusion is identified. No pneumothorax."
    },
    "patient64661_study1": { ... },
    "patient64604_study1": { ... },
    "patient64700_study1": { ... },
    "patient64523_study1": { ... }
}
```

Key fields:
- **`model_prediction`** (required): Model-generated report. For the `findings` split this is the Findings section only; for the `report` split it includes "Findings: ... Impression: ..." (both sections).
- **`section_findings`** / **`section_impression`**: Ground-truth reference text.
- **`context`**: Patient demographics and clinical indication.
- **`image_path`** / **`key_image_path`**: Paths to the input chest X-ray images.

> **Note:** The exact metadata fields vary by dataset (e.g., CheXpert-Plus includes `race`/`ethnicity`/`ap_pa`; MIMIC-CXR includes `subject_id`/`split`; ReXGradient includes `image_description`). The metric pipeline only uses `section_findings`, `section_impression`, and `model_prediction`.

### JSON → CSV Conversion

Before metric computation, the JSON is converted to two CSVs (ground-truth and predicted). The conversion maps:

```
JSON key            → case_id
enumerated index    → study_id  (0, 1, 2, ...)
section_findings    → report    (gt, findings split)
section_impression  → report    (gt, reports split)
model_prediction    → report    (predicted)
```

Resulting CSVs (`gt_reports_{model}.csv` / `predicted_reports_{model}.csv`):

```csv
study_id,case_id,report
0,patient64545_study1,"There are low lung volumes. The cardiomediastinal silhouette is within normal limits..."
1,patient64661_study1,"The transesophageal echo probe has been removed..."
```

```csv
study_id,case_id,report
0,patient64545_study1,"Heart size is normal. The mediastinal contours are within normal limits..."
1,patient64661_study1,"Endotracheal tube tip is 4 cm above the carina..."
```

### Output Step 1: Per-Sample CXR Metrics (`report_scores_{model}.csv`)

One row per sample. Contains the 4 base metrics plus two composite scores.

```csv
,index,study_id,case_id,report,bleu_score,bertscore,semb_score,radgraph_combined,RadCliQ-v0,RadCliQ-v1
0,0,0,patient64545_study1,"Heart size is normal...",0.1466,-0.1302,0.7308,0.1667,4.6436,1.9381
1,1,1,patient64661_study1,"Endotracheal tube...",0.0522,-0.1357,0.1648,0.0833,5.6532,2.5102
2,2,2,patient64604_study1,"Left chest tube...",0.0943,-0.0917,0.1329,0.0000,5.6103,2.4692
3,3,3,patient64700_study1,"The cardiac silhouette...",0.3162,0.2788,0.8521,0.4583,2.1957,0.7296
4,4,4,patient64523_study1,"Heart size is normal...",0.3536,0.4075,0.9105,0.5000,1.6257,0.5512
```

### Output Step 2: Per-Sample RaTEScore (`ratescore_{model}.csv`)

Appends a `ratescore` column to the predicted reports.

```csv
study_id,case_id,report,ratescore
0,patient64545_study1,"Heart size is normal...",0.5518
1,patient64661_study1,"Endotracheal tube...",0.4485
2,patient64604_study1,"Left chest tube...",0.5422
3,patient64700_study1,"The cardiac silhouette...",0.6123
4,patient64523_study1,"Heart size is normal...",0.7235
```

### Output Step 3: Per-Sample GREEN Score (`results_green_{model}.csv`)

Includes the LLM error analysis breakdown by category.

```csv
reference,predictions,green_analysis,green_score,(a) False report of a finding in the candidate,(b) Missing a finding present in the reference,(c) Misidentification of a finding's anatomic location/position,(d) Misassessment of the severity of a finding,(e) Mentioning a comparison that isn't in the reference,(f) Omitting a comparison detailing a change from a prior study,Matched Findings
"There are low lung volumes...","Heart size is normal...","The candidate report correctly identifies low lung volumes, left pleural effusion, atelectasis, and pulmonary edema. However, it misses the old bilateral rib fractures (b). The severity of pulmonary edema is described as 'mild' vs 'trace' (d).",0.5,0,1,0,1,0,0,4
```

Error categories:
- **(a)** False finding reported in candidate
- **(b)** Missing finding from reference
- **(c)** Wrong anatomic location
- **(d)** Wrong severity assessment
- **(e)** Extra comparison not in reference
- **(f)** Missing comparison from reference

### Output Step 4: Aggregated Per-Model Summary (`{model}.csv`)

Single row with mean scores across all samples, rounded to 3 decimals.

```csv
bleu_score,bertscore,semb_score,radgraph_combined,1/RadCliQ-v1,ratescore,green_score
0.19,0.428,0.46,0.236,1.008,0.564,0.279
```

Note: `1/RadCliQ-v1` = `1.0 / mean(RadCliQ-v1)` (inverted so higher = better).

### Output Step 5: Leaderboard (`{dataset}.csv`)

Ranked table with model metadata. Primary ranking metric: `1/RadCliQ-v1` (descending).

```csv
Rank,Date,Model Name,Institution,Model URL,BLEU,BertScore,SembScore,RadGraph,1/RadCliQ-v1,RaTEScore,GREEN
1,2025,UniRGCXR,Microsoft Research,,0.262,0.492,0.496,0.269,1.233,0.602,0.36
2,2025,Flora Microsoft,Microsoft Research,,0.248,0.493,0.487,0.265,1.217,0.596,0.352
3,2025,CheXOne-R1,Stanford,https://github.com/YBZh/CheXOne-R1,0.218,0.461,0.455,0.235,1.06,0.519,0.314
4,2025,RadPhi4VisionCXR,Microsoft Research,,0.234,0.444,0.439,0.251,1.033,0.584,0.351
5,2025,zzy-RGv1-8b,Individual,,0.213,0.432,0.455,0.253,1.031,0.579,0.339
6,2025,CX-Mind,SJTU,https://arxiv.org/pdf/2508.03733,0.15,0.385,0.319,0.174,0.782,0.543,0.267
```

## Environment Setup

The pipeline uses **2 separate conda environments** because Step 1 ([CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric)) requires older dependencies that conflict with Steps 2-3 ([RaTEScore](https://github.com/MAGIC-AI4Med/RaTEScore), [GREEN](https://github.com/Stanford-AIMI/GREEN)).

### Environment 1: `radgraph` — CXR Metrics (Step 1: BLEU, BERTScore, SembScore, RadGraph, RadCliQ)

```bash
conda create -n radgraph python=3.8 -y
conda activate radgraph

pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.25.1
pip install bert-score==0.3.13
pip install fast-bleu==0.0.4
pip install scikit-learn==1.2.2
pip install pandas numpy scipy tqdm
```

CXR-Report-Metric is already bundled in `scripts/CXR-Report-Metric/`. You only need to download two checkpoints:

| Checkpoint | Path (relative to `scripts/CXR-Report-Metric/`) | Source |
|---|---|---|
| CheXbert | `CheXbert/models/chexbert.pth` | [stanfordmlgroup/CheXbert](https://github.com/stanfordmlgroup/CheXbert) |
| RadGraph | `radgraph/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz` | [PhysioNet RadGraph v1.0.0](https://physionet.org/content/radgraph/1.0.0/) (requires credentialed access) |

RadCliQ-v1 (`CXRMetric/radcliq-v1.pkl`), RadCliQ-v0 normalizer and model are already included.

**Config file** (`scripts/CXR-Report-Metric/config.py`):
```python
CHEXBERT_PATH = "CheXbert/models/chexbert.pth"
RADGRAPH_PATH = "radgraph/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz"
USE_IDF = False
```

**Run (from `rexrank-metric/`):**
```bash
conda activate radgraph

python scripts/run_cxr_metrics.py \
    --models MyModel \
    --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
    --splits findings reports \
    --data-root data --results-root results
```

### Environment 2: `green_score` — RaTEScore (Step 2) & GREEN (Step 3)

```bash
conda create -n green_score python=3.10 -y
conda activate green_score

pip install torch==2.1.0
pip install transformers==4.36.0
pip install pandas numpy

# RaTEScore (https://github.com/MAGIC-AI4Med/RaTEScore)
pip install ratescore

# GREEN (https://github.com/Stanford-AIMI/GREEN)
git clone https://github.com/Stanford-AIMI/GREEN.git
cd GREEN && pip install -e . && cd ..
```

The GREEN model (`StanfordAIMI/GREEN-radllama2-7b`) is downloaded automatically from HuggingFace on first run. Requires ~14 GB disk and a GPU with >= 16 GB VRAM.

**Run (from `rexrank-metric/`):**
```bash
conda activate green_score

# Step 2: RaTEScore
python scripts/run_ratescore.py \
    --models MyModel \
    --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
    --splits findings reports \
    --data-root data --results-root results

# Step 3: GREEN
python scripts/run_green.py \
    --models MyModel \
    --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
    --splits findings reports \
    --data-root data --results-root results
```

### Aggregation & Leaderboard (Steps 4-5 — no GPU needed)

Steps 4 and 5 are pure pandas operations. They work in either of the above environments or a minimal one:

```bash
# Any env with pandas works (from rexrank-metric/)
pip install pandas numpy

# Step 4: Aggregate per-sample → per-model summary
python scripts/aggregate_metrics.py \
    --models MyModel \
    --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
    --splits findings reports \
    --results-root results --output-root results/metric

# Step 5: Build ranked leaderboard
python scripts/summarize_leaderboard.py \
    --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
    --splits findings reports \
    --metric-root results/metric --output-dir results/metric/results_summary
```

## Metric Computation Details

### BLEU-2 (Step 1)

- Library: `fast_bleu`
- Weights: `(1/2, 1/2)` (bigram)
- Preprocessing: lowercase, add space around periods, tokenize on whitespace
- Computed per-sample by matching `study_id`

### BERTScore (Step 1)

- Library: `bert_score`
- Model: `distilroberta-base`
- Settings: `rescale_with_baseline=True`, `idf=False`, `batch_size=256`, `lang="en"`
- Preprocessing: collapse multiple spaces
- Reports the F1 score

### SembScore (Step 1)

- Embedding model: CheXbert (`chexbert.pth`)
- Encodes both ground-truth and predicted reports via `CheXbert/src/encode.py`
- Score = cosine similarity between embedding vectors

### RadGraph Combined (Step 1)

- Inference: DyGIE++ model from PhysioNet RadGraph v1.0.0
- Extracts entities and relations from both ground-truth and predicted reports
- Entity F1 = standard F1 over entity sets
- Relation F1 = standard F1 over relation sets
- Combined = (Entity F1 + Relation F1) / 2

### RadCliQ-v1 (Step 1)

- Input features: `[radgraph_combined, bertscore, semb_score, bleu_score]`
- Model: pre-trained linear regression loaded from `CXRMetric/radcliq-v1.pkl`
- No separate normalization (unlike RadCliQ-v0 which uses MinMaxScaler from `normalizer.pkl`)
- Lower RadCliQ-v1 values indicate better reports

### 1/RadCliQ-v1 (Step 4)

- Computed as: `1.0 / mean(RadCliQ-v1)` across all samples in a dataset
- This inversion makes higher values = better, suitable for leaderboard ranking

### RaTEScore (Step 2)

- Library: `RaTEScore`
- Usage: `ratescore.compute_score(hypotheses, references)`
- Evaluates factual and temporal consistency

### GREEN Score (Step 3)

- Model: `StanfordAIMI/GREEN-radllama2-7b` (LLaMA2-based)
- Settings: `cpu=False`, `compute_summary_stats=False`
- Analyzes clinical errors in 6 categories:
  - (a) False finding reported
  - (b) Missing finding
  - (c) Wrong anatomic location
  - (d) Wrong severity
  - (e) Extra comparison mentioned
  - (f) Missing comparison

## Quick Start: End-to-End Example

This walks through evaluating a model called `MyModel` on one dataset (`mimic-cxr`, findings split).

### 1. Prepare your input CSVs

Place CSVs under `data/`:

```
data/findings/mimic-cxr/gt_reports_MyModel.csv
data/findings/mimic-cxr/predicted_reports_MyModel.csv
```

Both must have columns: `study_id`, `case_id`, `report`. The `study_id` values must match between the two files.

### 2. Run metrics (2 separate environments)

All commands below assume you are in the `rexrank-metric/` directory.

```bash
# --- Steps 0-1: CXR metrics (conda: radgraph, GPU) ---
conda activate radgraph
python scripts/run_cxr_metrics.py \
    --models MyModel --datasets mimic-cxr --splits findings \
    --data-root data --results-root results
# Output: results/mimic-cxr_findings/report_scores_MyModel.csv

# --- Steps 2-3: RaTEScore + GREEN (conda: green_score, GPU) ---
conda activate green_score
python scripts/run_ratescore.py \
    --models MyModel --datasets mimic-cxr --splits findings \
    --data-root data --results-root results
# Output: results/mimic-cxr_findings/ratescore_MyModel.csv

python scripts/run_green.py \
    --models MyModel --datasets mimic-cxr --splits findings \
    --data-root data --results-root results
# Output: results/mimic-cxr_findings/results_green_MyModel.csv

# --- Steps 4-5: Aggregate & leaderboard (any env, no GPU) ---
python scripts/aggregate_metrics.py \
    --models MyModel --datasets mimic-cxr --splits findings \
    --results-root results --output-root results/metric
# Output: results/metric/mimic-cxr/findings/MyModel.csv

python scripts/summarize_leaderboard.py \
    --datasets mimic-cxr --splits findings \
    --metric-root results/metric --output-dir results/metric/results_summary
# Output: results/metric/results_summary/mimic-cxr.csv
```

### Shell Scripts (from a JSON submission)

The pipeline is split into **two SLURM-compatible shell scripts**, each with its own conda env and GPU requirement:

| Script | Conda Env | GPU | Steps |
|--------|-----------|-----|-------|
| `run_cxr_metrics.sh` | `radgraph` | Yes | 0 (JSON→CSV) + 1 (BLEU, BERTScore, SembScore, RadGraph, RadCliQ) |
| `run_green_ratescore.sh` | `green_score` | Yes | 2 (RaTEScore) + 3 (GREEN) + 4 (Aggregate) + 5 (Leaderboard) |

```bash
# --- Option A: Submit as two separate SLURM jobs ---
sbatch scripts/run_cxr_metrics.sh \
    --json data/submission_example.json \
    --model MyModel --dataset iu_xray

# (after job 1 completes)
sbatch scripts/run_green_ratescore.sh \
    --model MyModel --dataset iu_xray

# --- Option B: Run both interactively ---
bash scripts/run_all.sh \
    --json data/submission_example.json \
    --model MyModel --dataset iu_xray
```

`run_all.sh` is a convenience wrapper that runs both scripts sequentially (switches conda envs automatically).

## Hardware Requirements

| Step | GPU | VRAM | Notes |
|------|-----|------|-------|
| Step 1: CXR-Report-Metric | Required | >= 8 GB | RadGraph (DyGIE++) inference |
| Step 2: RaTEScore | Optional | — | CPU is fine |
| Step 3: GREEN | Required | >= 16 GB | GREEN-radllama2-7b (7B LLM) |
| Steps 4-5: Aggregate | No | — | Pure pandas |

- RAM: >= 64 GB recommended
- Storage: ~15 GB for model checkpoints

## Citation

If you use this evaluation pipeline, please cite:

```bibtex
@article{zhang2024rexrank,
  title={ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation},
  author={Zhang, Xiaoman and Zhou, Hong-Yu and Yang, Xiaoli and Banerjee, Oishi and Acosta, Juli{\'a}n N and Miller, Josh and Huang, Ouwen and Rajpurkar, Pranav},
  journal={AAAI Bridge Program AIMedHealth},
  year={2025}
}

@inproceedings{zhang2025rexgradient,
  title={ReXGradient-160K: A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports},
  author={Zhang, Xiaoman and Acosta, Julián N. and Miller, Josh and Huang, Ouwen and Rajpurkar, Pranav},
  booktitle={arXiv:2505.00228v1},
  year={2025}
}
```

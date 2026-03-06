"""Centralized configuration for the reward service."""

import os

# ---------- Model checkpoint paths ----------
CHEXBERT_PATH = os.environ.get(
    "CHEXBERT_PATH",
    "/mnt/shared-storage-user/ai4good1-share/xieyuejin/models/RRG_scorers/chexbert.pth",
)

BERT_BASE_UNCASED_PATH = os.environ.get(
    "BERT_BASE_UNCASED_PATH",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--google-bert--bert-base-uncased",
)

DISTILROBERTA_PATH = os.environ.get(
    "DISTILROBERTA_PATH",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--distilbert--distilroberta-base",
)

RATESCORE_BERT_MODEL = os.environ.get(
    "RATESCORE_BERT_MODEL",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Angelakeke--RaTE-NER-Deberta",
)

RATESCORE_EVAL_MODEL = os.environ.get(
    "RATESCORE_EVAL_MODEL",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--FremyCompany--BioLORD-2023-C",
)

GREEN_MODEL_PATH = os.environ.get(
    "GREEN_MODEL_PATH",
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--StanfordAIMI--GREEN-RadLlama2-7b",
)

RADGRAPH_MODEL_CACHE = os.environ.get(
    "RADGRAPH_MODEL_CACHE",
    "/mnt/shared-storage-user/ai4good1-share/xieyuejin/models/RRG_scorers",
)

HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR",
    "/mnt/shared-storage-user/ai4good1-share/hf_hub",
)

RADGRAPH_TOKENIZER_PATH = os.environ.get(
    "RADGRAPH_TOKENIZER_PATH",
    "/mnt/shared-storage-user/ai4good1-share/hf_hub/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
)

# ---------- Pickle model paths (RadCliQ) ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NORMALIZER_PATH = os.path.join(DATA_DIR, "normalizer.pkl")
COMPOSITE_METRIC_V0_PATH = os.path.join(DATA_DIR, "composite_metric_model.pkl")
COMPOSITE_METRIC_V1_PATH = os.path.join(DATA_DIR, "radcliq-v1.pkl")

# ---------- GPU assignments ----------
GPU_GREEN = int(os.environ.get("GPU_GREEN", "0"))
GPU_RADGRAPH = int(os.environ.get("GPU_RADGRAPH", "1"))
GPU_BERT = int(os.environ.get("GPU_BERT", "2"))  # BERTScore + CheXbert
GPU_RATESCORE = int(os.environ.get("GPU_RATESCORE", "3"))

# ---------- GREEN vLLM endpoint ----------
GREEN_VLLM_BASE_URL = os.environ.get("GREEN_VLLM_BASE_URL", "http://localhost:9101/v1")

# ---------- Service ----------
SERVICE_HOST = os.environ.get("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "9100"))

# ---------- Batch sizes ----------
CHEXBERT_BATCH_SIZE = 18
BERTSCORE_BATCH_SIZE = 256
RADGRAPH_BATCH_SIZE = 8

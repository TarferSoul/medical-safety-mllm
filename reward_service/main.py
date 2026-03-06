"""FastAPI reward service for radiology report evaluation metrics.

Preloads all metric models at startup and serves them via a REST API.
All computation is in-memory — no intermediate CSV/JSON files.
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from reward_service import config
from reward_service.api_models import (
    DEFAULT_METRICS,
    HealthResponse,
    RewardRequest,
    RewardResponse,
)

logger = logging.getLogger("reward_service")
logging.basicConfig(level=logging.INFO)

# ---------- Global metric instances ----------
_bleu_ready = False
_bertscore: Optional["BERTScoreMetric"] = None
_semb_score: Optional["SembScoreMetric"] = None
_chexbert_encoder: Optional["CheXbertEncoder"] = None
_radgraph: Optional["RadGraphMetric"] = None
_radcliq: Optional["RadCliQMetric"] = None
_ratescore: Optional["RaTEScoreMetric"] = None
_green: Optional["GREENMetric"] = None

# Lock for metrics sharing GPU 2 (BERTScore + CheXbert/SembScore)
_gpu2_lock = threading.Lock()

_models_loaded: Dict[str, bool] = {
    "bleu": False,
    "bertscore": False,
    "semb_score": False,
    "radgraph": False,
    "radcliq": False,
    "ratescore": False,
    "green": False,
}


def _load_models():
    """Load all metric models. Called once at startup."""
    global _bleu_ready, _bertscore, _semb_score, _chexbert_encoder
    global _radgraph, _radcliq, _ratescore, _green

    # BLEU — no model to load, just CPU
    _bleu_ready = True
    _models_loaded["bleu"] = True
    logger.info("BLEU ready (CPU, no model)")

    # BERTScore on GPU_BERT
    try:
        from reward_service.metrics.bertscore_metric import BERTScoreMetric
        _bertscore = BERTScoreMetric(
            model_path=config.DISTILROBERTA_PATH,
            device=f"cuda:{config.GPU_BERT}",
        )
        _models_loaded["bertscore"] = True
        logger.info("BERTScore loaded on GPU %d", config.GPU_BERT)
    except Exception as e:
        logger.error("Failed to load BERTScore: %s", e)

    # CheXbert + SembScore on GPU_BERT
    try:
        import torch
        from reward_service.chexbert.encoder import CheXbertEncoder
        from reward_service.metrics.semb_score import SembScoreMetric

        device = torch.device(f"cuda:{config.GPU_BERT}")
        _chexbert_encoder = CheXbertEncoder(
            checkpoint_path=config.CHEXBERT_PATH,
            bert_model_path=config.BERT_BASE_UNCASED_PATH,
            device=device,
        )
        _semb_score = SembScoreMetric(_chexbert_encoder)
        _models_loaded["semb_score"] = True
        logger.info("CheXbert + SembScore loaded on GPU %d", config.GPU_BERT)
    except Exception as e:
        logger.error("Failed to load CheXbert/SembScore: %s", e)

    # RadGraph on GPU_RADGRAPH
    try:
        from reward_service.metrics.radgraph_metric import RadGraphMetric
        _radgraph = RadGraphMetric(
            cuda_device=config.GPU_RADGRAPH,
            model_cache_dir=config.RADGRAPH_MODEL_CACHE,
            tokenizer_path=config.RADGRAPH_TOKENIZER_PATH,
        )
        _models_loaded["radgraph"] = True
        logger.info("RadGraph loaded on GPU %d", config.GPU_RADGRAPH)
    except Exception as e:
        logger.error("Failed to load RadGraph: %s", e)

    # RadCliQ — CPU, just loads pickles
    try:
        from reward_service.metrics.radcliq import RadCliQMetric
        _radcliq = RadCliQMetric(
            normalizer_path=config.NORMALIZER_PATH,
            v0_model_path=config.COMPOSITE_METRIC_V0_PATH,
            v1_model_path=config.COMPOSITE_METRIC_V1_PATH,
        )
        _models_loaded["radcliq"] = True
        logger.info("RadCliQ loaded (CPU)")
    except Exception as e:
        logger.error("Failed to load RadCliQ: %s", e)

    # RaTEScore on GPU_RATESCORE
    try:
        from reward_service.metrics.ratescore_metric import RaTEScoreMetric
        _ratescore = RaTEScoreMetric(
            bert_model=config.RATESCORE_BERT_MODEL,
            eval_model=config.RATESCORE_EVAL_MODEL,
        )
        _models_loaded["ratescore"] = True
        logger.info("RaTEScore loaded on GPU %d", config.GPU_RATESCORE)
    except Exception as e:
        logger.error("Failed to load RaTEScore: %s", e)

    # GREEN — connects to vLLM endpoint (no local model)
    try:
        from reward_service.metrics.green_metric import GREENMetric
        _green = GREENMetric(base_url=config.GREEN_VLLM_BASE_URL)
        _models_loaded["green"] = True
        logger.info("GREEN connected to vLLM at %s", config.GREEN_VLLM_BASE_URL)
    except Exception as e:
        logger.warning("GREEN vLLM not available: %s", e)


def _warmup(predictions: List[str], references: List[str]):
    """Run a dummy request through all loaded metrics."""
    logger.info("Running warmup...")
    t0 = time.time()
    if _bleu_ready:
        from reward_service.metrics.bleu import compute_bleu
        compute_bleu(predictions, references)
    if _bertscore:
        with _gpu2_lock:
            _bertscore.compute(predictions, references)
    if _semb_score:
        with _gpu2_lock:
            _semb_score.compute(predictions, references)
    if _radgraph:
        _radgraph.compute(predictions, references)
    if _ratescore:
        _ratescore.compute(predictions, references)
    # Skip GREEN warmup — it requires the vLLM server to be up
    logger.info("Warmup complete in %.1fs", time.time() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    _load_models()
    # Warmup with dummy data
    dummy_preds = ["The heart is normal in size."]
    dummy_refs = ["The heart is normal in size. No acute process."]
    _warmup(dummy_preds, dummy_refs)
    yield


app = FastAPI(title="Reward Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return model load status."""
    status = "ok" if any(_models_loaded.values()) else "no_models"
    return HealthResponse(status=status, models_loaded=_models_loaded)


@app.post("/compute", response_model=RewardResponse)
async def compute(request: RewardRequest):
    """Compute requested metrics for a batch of (prediction, reference) pairs."""
    predictions = request.predictions
    references = request.references

    if len(predictions) != len(references):
        raise HTTPException(
            status_code=400,
            detail="predictions and references must have the same length",
        )
    if len(predictions) == 0:
        raise HTTPException(status_code=400, detail="empty batch")

    metrics = request.metrics or DEFAULT_METRICS
    n = len(predictions)

    # Results storage
    results: Dict[str, Optional[List[float]]] = {}

    # --- Schedule independent metrics concurrently ---
    async def run_bleu():
        from reward_service.metrics.bleu import compute_bleu
        return await asyncio.to_thread(compute_bleu, predictions, references)

    async def run_bertscore():
        def _fn():
            with _gpu2_lock:
                return _bertscore.compute(predictions, references)
        return await asyncio.to_thread(_fn)

    async def run_semb():
        def _fn():
            with _gpu2_lock:
                return _semb_score.compute(predictions, references)
        return await asyncio.to_thread(_fn)

    async def run_radgraph():
        return await asyncio.to_thread(_radgraph.compute, predictions, references)

    async def run_ratescore():
        return await asyncio.to_thread(_ratescore.compute, predictions, references)

    async def run_green():
        return await asyncio.to_thread(_green.compute, predictions, references)

    tasks = []
    task_names = []

    if "bleu" in metrics and _bleu_ready:
        tasks.append(run_bleu())
        task_names.append("bleu")

    if "bertscore" in metrics and _bertscore:
        tasks.append(run_bertscore())
        task_names.append("bertscore")

    if "semb_score" in metrics and _semb_score:
        tasks.append(run_semb())
        task_names.append("semb_score")

    if "radgraph" in metrics and _radgraph:
        tasks.append(run_radgraph())
        task_names.append("radgraph")

    if "ratescore" in metrics and _ratescore:
        tasks.append(run_ratescore())
        task_names.append("ratescore")

    if "green" in metrics and _green:
        tasks.append(run_green())
        task_names.append("green")

    # Run all independent metrics concurrently
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for name, result in zip(task_names, gathered):
        if isinstance(result, Exception):
            logger.error("Metric %s failed: %s", name, result)
            results[name] = [None] * n
        else:
            results[name] = result

    # --- RadCliQ depends on the 4 component scores ---
    needs_radcliq = ("radcliq_v0" in metrics or "radcliq_v1" in metrics)
    if needs_radcliq and _radcliq:
        bleu_scores = results.get("bleu", [0.0] * n)
        bertscore_scores = results.get("bertscore", [0.0] * n)
        semb_scores = results.get("semb_score", [0.0] * n)
        radgraph_scores = results.get("radgraph", [0.0] * n)

        # Replace None with 0 for RadCliQ computation
        def _safe(lst):
            return [x if x is not None else 0.0 for x in lst]

        try:
            radcliq_result = _radcliq.compute(
                radgraph_scores=_safe(radgraph_scores),
                bertscore_scores=_safe(bertscore_scores),
                semb_scores=_safe(semb_scores),
                bleu_scores=_safe(bleu_scores),
            )
            if "radcliq_v0" in metrics:
                results["radcliq_v0"] = radcliq_result["radcliq_v0"]
            if "radcliq_v1" in metrics:
                results["radcliq_v1"] = radcliq_result["radcliq_v1"]
        except Exception as e:
            logger.error("RadCliQ failed: %s", e)
            if "radcliq_v0" in metrics:
                results["radcliq_v0"] = [None] * n
            if "radcliq_v1" in metrics:
                results["radcliq_v1"] = [None] * n

    # --- Assemble per-sample scores ---
    scores = []
    for i in range(n):
        sample = {}
        for metric_name in metrics:
            if metric_name in results:
                val = results[metric_name][i]
                sample[metric_name] = float(val) if val is not None else None
        scores.append(sample)

    # --- Batch means ---
    batch_means = {}
    for metric_name in metrics:
        if metric_name in results:
            vals = [v for v in results[metric_name] if v is not None]
            batch_means[metric_name] = float(np.mean(vals)) if vals else None

    return RewardResponse(scores=scores, batch_means=batch_means)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "reward_service.main:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level="info",
    )

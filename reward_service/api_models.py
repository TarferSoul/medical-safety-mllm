"""Pydantic request/response schemas for the reward service API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

ALL_METRICS = ["bleu", "bertscore", "semb_score", "radgraph", "radcliq_v0", "radcliq_v1", "ratescore", "green"]
DEFAULT_METRICS = ["bleu", "bertscore", "semb_score", "radgraph", "radcliq_v0", "radcliq_v1", "ratescore"]


class RewardRequest(BaseModel):
    predictions: List[str] = Field(..., description="Generated reports")
    references: List[str] = Field(..., description="Ground-truth reports")
    metrics: Optional[List[str]] = Field(
        None,
        description="Metrics to compute. Defaults to all except green.",
    )


class SampleScores(BaseModel):
    bleu: Optional[float] = None
    bertscore: Optional[float] = None
    semb_score: Optional[float] = None
    radgraph: Optional[float] = None
    radcliq_v0: Optional[float] = None
    radcliq_v1: Optional[float] = None
    ratescore: Optional[float] = None
    green: Optional[float] = None


class RewardResponse(BaseModel):
    scores: List[Dict[str, Optional[float]]]
    batch_means: Dict[str, Optional[float]]


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]

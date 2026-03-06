"""SembScore — cosine similarity between CheXbert CLS embeddings."""

from typing import List

import numpy as np
import torch

from reward_service.chexbert.encoder import CheXbertEncoder


class SembScoreMetric:
    """Semantic embedding similarity via CheXbert."""

    def __init__(self, encoder: CheXbertEncoder):
        self.encoder = encoder

    def compute(self, predictions: List[str], references: List[str]) -> List[float]:
        """Compute per-sample cosine similarity of CheXbert CLS embeddings.

        Args:
            predictions: Generated reports.
            references: Ground-truth reports.

        Returns:
            List of cosine similarity scores.
        """
        pred_embeds = self.encoder.encode(predictions)
        ref_embeds = self.encoder.encode(references)

        scores = []
        for i in sorted(pred_embeds.keys()):
            pred = pred_embeds[i].numpy()
            ref = ref_embeds[i].numpy()
            sim = (pred * ref).sum() / (
                np.linalg.norm(pred) * np.linalg.norm(ref)
            )
            scores.append(float(sim))
        return scores

"""BERTScore metric with pre-loaded scorer."""

import os
import re
from typing import List

import bert_score
from bert_score import BERTScorer


class BERTScoreMetric:
    """BERTScore metric with a pre-loaded scorer model."""

    def __init__(self, model_path: str, device: str = "cuda:2"):
        baseline_path = os.path.join(
            os.path.dirname(bert_score.__file__),
            "rescale_baseline/en/distilroberta-base.tsv",
        )
        self.scorer = BERTScorer(
            model_type=model_path,
            num_layers=5,
            batch_size=256,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=baseline_path,
            idf=False,
            device=device,
        )

    def compute(self, predictions: List[str], references: List[str]) -> List[float]:
        """Compute per-sample BERTScore F1.

        Args:
            predictions: Generated reports.
            references: Ground-truth reports.

        Returns:
            List of BERTScore F1 values.
        """
        refs = [re.sub(r" +", " ", r) for r in references]
        preds = [re.sub(r" +", " ", p) for p in predictions]
        _, _, f1 = self.scorer.score(preds, refs)
        return f1.tolist()

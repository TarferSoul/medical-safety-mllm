"""RaTEScore metric wrapper."""

from typing import List


class RaTEScoreMetric:
    """RaTEScore metric with pre-loaded models."""

    def __init__(self, bert_model: str, eval_model: str):
        from RaTEScore import RaTEScore
        self.scorer = RaTEScore(bert_model=bert_model, eval_model=eval_model)

    def compute(self, predictions: List[str], references: List[str]) -> List[float]:
        """Compute per-sample RaTEScore.

        Args:
            predictions: Generated reports.
            references: Ground-truth reports.

        Returns:
            List of RaTEScore values.
        """
        preds = [str(p) for p in predictions]
        refs = [str(r) for r in references]
        scores = self.scorer.compute_score(preds, refs)
        return [float(s) for s in scores]

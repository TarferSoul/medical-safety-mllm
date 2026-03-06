"""RadCliQ composite metrics (v0 and v1).

The CompositeMetric class must be defined before pickle.load() so
deserialization of radcliq-v1.pkl works correctly.
"""

import pickle
from typing import Dict, List

import numpy as np


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Must be importable at unpickling time.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """

    def __init__(self, scaler, coefs):
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1
        )
        return norm_x @ self.coefs


# Column order expected by both RadCliQ models
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]


class _CompositeUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects __main__.CompositeMetric to our class."""

    def find_class(self, module, name):
        if module == "__main__" and name == "CompositeMetric":
            return CompositeMetric
        return super().find_class(module, name)


class RadCliQMetric:
    """Computes RadCliQ-v0 and RadCliQ-v1 from the 4 component scores."""

    def __init__(self, normalizer_path: str, v0_model_path: str, v1_model_path: str):
        with open(normalizer_path, "rb") as f:
            self.normalizer = pickle.load(f)
        with open(v0_model_path, "rb") as f:
            self.v0_model = pickle.load(f)
        with open(v1_model_path, "rb") as f:
            self.v1_model = _CompositeUnpickler(f).load()

    def compute(
        self,
        radgraph_scores: List[float],
        bertscore_scores: List[float],
        semb_scores: List[float],
        bleu_scores: List[float],
    ) -> Dict[str, List[float]]:
        """Compute RadCliQ-v0 and RadCliQ-v1 from component scores.

        Args:
            radgraph_scores: Per-sample RadGraph combined F1.
            bertscore_scores: Per-sample BERTScore F1.
            semb_scores: Per-sample SembScore.
            bleu_scores: Per-sample BLEU-2.

        Returns:
            Dict with "radcliq_v0" and "radcliq_v1" lists.
        """
        # Build input array in the expected column order
        input_data = np.column_stack([
            radgraph_scores, bertscore_scores, semb_scores, bleu_scores,
        ])

        # RadCliQ-v0: normalize then predict
        norm_input = self.normalizer.transform(input_data)
        v0_scores = self.v0_model.predict(norm_input).tolist()

        # RadCliQ-v1: uses its own internal normalization
        v1_scores = self.v1_model.predict(input_data).tolist()

        return {"radcliq_v0": v0_scores, "radcliq_v1": v1_scores}

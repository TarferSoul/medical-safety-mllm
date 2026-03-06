"""BLEU-2 metric using fast_bleu."""

from typing import List

from fast_bleu import BLEU

WEIGHTS = {"bigram": (1 / 2.0, 1 / 2.0)}


def prep_report(text: str) -> List[str]:
    """Preprocess a report: lowercase, split on spaces, insert space before periods."""
    return list(filter(
        lambda val: val != "",
        str(text).lower().replace(".", " .").split(" "),
    ))


def compute_bleu(predictions: List[str], references: List[str]) -> List[float]:
    """Compute per-sample BLEU-2 scores.

    Args:
        predictions: Generated reports.
        references: Ground-truth reports.

    Returns:
        List of BLEU-2 scores, one per sample.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = prep_report(ref)
        pred_tokens = prep_report(pred)
        bleu = BLEU([ref_tokens], WEIGHTS)
        result = bleu.get_score([pred_tokens])["bigram"]
        scores.append(result[0])
    return scores

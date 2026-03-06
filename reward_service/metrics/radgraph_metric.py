"""RadGraph F1 metric using the `radgraph` pip package.

Replaces the allennlp + dygiepp dependency entirely.
"""

from typing import Dict, List, Set, Tuple


def compute_f1(test: Set, retrieved: Set) -> float:
    """Compute F1 between two sets of entities or relations."""
    true_positives = len(test.intersection(retrieved))
    false_positives = len(retrieved) - true_positives
    false_negatives = len(test) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives != 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives != 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall != 0
        else 0
    )
    return f1


def parse_radgraph_output(result: Dict) -> Tuple[Set, Set]:
    """Parse a single RadGraph output into entity and relation sets.

    Args:
        result: Dict with 'entities' key from radgraph output.

    Returns:
        (entities_set, relations_set) where:
          entities_set = {(token, label), ...}
          relations_set = {((tok1, label1), (tok2, label2), relation_type), ...}
    """
    entities_dict = result.get("entities", {})
    entities_set = set()
    relations_set = set()

    for _, entity in entities_dict.items():
        token = entity["tokens"]
        label = entity["label"]
        entities_set.add((token, label))

        for relation in entity.get("relations", []):
            rel_type = relation[0]
            target_key = relation[1]
            if target_key in entities_dict:
                target = entities_dict[target_key]
                relations_set.add((
                    (token, label),
                    (target["tokens"], target["label"]),
                    rel_type,
                ))

    return entities_set, relations_set


class RadGraphMetric:
    """RadGraph F1 metric using the radgraph pip package."""

    def __init__(self, cuda_device: int = 1, model_cache_dir: str = None,
                 tokenizer_path: str = None):
        import os
        from radgraph import RadGraph

        # Ensure offline mode so radgraph doesn't try to reach huggingface.co
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.model = RadGraph(
            model_type="radgraph",
            cuda=cuda_device,
            batch_size=8,
            model_cache_dir=model_cache_dir,
            tokenizer_cache_dir=tokenizer_path,
        )

    def compute(self, predictions: List[str], references: List[str]) -> List[float]:
        """Compute per-sample RadGraph combined F1 (avg of entity + relation F1).

        Args:
            predictions: Generated reports.
            references: Ground-truth reports.

        Returns:
            List of combined F1 scores.
        """
        ref_results = self.model(references)
        pred_results = self.model(predictions)

        scores = []
        for i in range(len(predictions)):
            idx_key = str(i)
            ref_result = ref_results.get(idx_key, {"entities": {}})
            pred_result = pred_results.get(idx_key, {"entities": {}})

            ref_entities, ref_relations = parse_radgraph_output(ref_result)
            pred_entities, pred_relations = parse_radgraph_output(pred_result)

            entity_f1 = compute_f1(ref_entities, pred_entities)
            relation_f1 = compute_f1(ref_relations, pred_relations)
            combined = (entity_f1 + relation_f1) / 2.0
            scores.append(combined)

        return scores

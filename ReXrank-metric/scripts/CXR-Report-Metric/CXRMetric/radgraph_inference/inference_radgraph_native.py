"""
Use radgraph package's native API instead of allennlp CLI
This completely bypasses allennlp dependency issues
"""
import json
import os
from pathlib import Path


def run_radgraph_native(model_checkpoint, gt_csv, pred_csv, output_path):
    """
    Run RadGraph using native radgraph package API

    Args:
        model_checkpoint: Path to radgraph model (e.g., radgraph.tar.gz)
        gt_csv: Ground truth CSV file
        pred_csv: Predicted CSV file
        output_path: Where to save results JSON
    """
    try:
        from radgraph import RadGraph
        import pandas as pd

        print("Loading RadGraph model using native API...")

        # Initialize RadGraph
        radgraph = RadGraph(
            model_type="radgraph",  # or radgraph-xl
            cuda=0 if __import__('torch').cuda.is_available() else -1,
            batch_size=8
        )

        # Read reports
        gt_df = pd.read_csv(gt_csv)
        pred_df = pd.read_csv(pred_csv)

        # Process ground truth
        print("Processing ground truth reports...")
        gt_texts = gt_df['report'].tolist()
        gt_results = radgraph(gt_texts)

        # Process predictions
        print("Processing predicted reports...")
        pred_texts = pred_df['report'].tolist()
        pred_results = radgraph(pred_texts)

        # Format output similar to old inference format
        output_data = {
            'gt': gt_results,
            'pred': pred_results
        }

        # Save results
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ RadGraph inference complete: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error using radgraph native API: {e}")
        import traceback
        traceback.print_exc()
        return False


def compute_radgraph_scores(gt_results, pred_results):
    """
    Compute RadGraph F1 scores between GT and predictions

    Args:
        gt_results: Ground truth RadGraph output
        pred_results: Predicted RadGraph output

    Returns:
        dict: Mapping study_id -> (f1_entities, f1_relations, f1_combined)
    """
    from radgraph.rewards import compute_reward

    scores = {}

    for idx, (gt, pred) in enumerate(zip(gt_results, pred_results)):
        # Compute reward (F1 score)
        reward, reward_info = compute_reward(
            hypothesis=pred,
            reference=gt,
            reward_level='full'  # entities + relations
        )

        scores[idx] = {
            'f1_combined': reward,
            'entities_f1': reward_info.get('entities_f1', 0),
            'relations_f1': reward_info.get('relations_f1', 0)
        }

    return scores

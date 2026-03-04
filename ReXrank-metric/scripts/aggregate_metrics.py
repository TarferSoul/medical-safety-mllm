"""
Step 4: Aggregate per-sample scores into per-model summary CSVs.

Reads the per-sample output files from Steps 1-3, computes mean scores,
and writes one summary CSV per (dataset, split, model) with columns:
    bleu_score, bertscore, semb_score, radgraph_combined, 1/RadCliQ-v1, ratescore, green_score

Normalization note:
    - 1/RadCliQ-v1 is computed as:  1.0 / mean(RadCliQ-v1)
      This inverts the error-based RadCliQ-v1 so that higher = better on the leaderboard.

Usage:
    python scripts/aggregate_metrics.py \
        --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
        --models ModelA ModelB \
        --splits findings reports \
        --results-root ../results \
        --output-root ../results/metric
"""

import argparse
import os

import numpy as np
import pandas as pd

COLUMN_ORDER = [
    "bleu_score",
    "bertscore",
    "semb_score",
    "radgraph_combined",
    "1/RadCliQ-v1",
    "ratescore",
    "green_score",
]


def aggregate_one(results_dir, model, output_path):
    """Aggregate per-sample results for a single (dataset, split, model)."""
    metrics = {}

    # --- report_scores (BLEU, BERTScore, SembScore, RadGraph, RadCliQ) ---
    report_scores_path = os.path.join(results_dir, f"report_scores_{model}.csv")
    if os.path.exists(report_scores_path):
        df = pd.read_csv(report_scores_path)
        for col in ["bleu_score", "bertscore", "semb_score", "radgraph_combined"]:
            if col in df.columns:
                metrics[col] = df[col].mean()
        if "RadCliQ-v1" in df.columns:
            metrics["1/RadCliQ-v1"] = 1.0 / df["RadCliQ-v1"].mean()

    # --- ratescore ---
    ratescore_path = os.path.join(results_dir, f"ratescore_{model}.csv")
    if os.path.exists(ratescore_path):
        df = pd.read_csv(ratescore_path)
        if "ratescore" in df.columns:
            metrics["ratescore"] = df["ratescore"].mean()

    # --- green_score ---
    green_path = os.path.join(results_dir, f"results_green_{model}.csv")
    if os.path.exists(green_path):
        df = pd.read_csv(green_path)
        if "green_score" in df.columns:
            metrics["green_score"] = df["green_score"].mean()

    if not metrics:
        print(f"  [WARN] No metrics found in {results_dir} for {model}")
        return

    out_df = pd.DataFrame([metrics]).round(3)
    cols = [c for c in COLUMN_ORDER if c in out_df.columns]
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"  [OK] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-sample metrics into summary CSVs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["iu_xray", "chexpert_plus", "mimic-cxr", "gradient_health"],
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["findings", "reports"],
        choices=["findings", "reports"],
    )
    parser.add_argument("--results-root", default="results", help="Root of per-sample results.")
    parser.add_argument("--output-root", default="results/metric", help="Root for summary CSVs.")
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in args.models:
            for split in args.splits:
                results_dir = os.path.join(args.results_root, f"{dataset}_{split}")
                output_path = os.path.join(args.output_root, dataset, split, f"{model}.csv")
                print(f"[AGG] {model} | {dataset}/{split}")
                aggregate_one(results_dir, model, output_path)


if __name__ == "__main__":
    main()

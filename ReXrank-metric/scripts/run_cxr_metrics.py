"""
Step 1: Compute BLEU, BERTScore, SembScore, RadGraph, RadCliQ-v0, and RadCliQ-v1.

This script wraps CXR-Report-Metric to compute per-sample scores for each
(dataset, model) pair across both findings and full-report splits.

Dependencies:
    - CXR-Report-Metric (https://github.com/rajpurkarlab/CXR-Report-Metric)
    - bert-score
    - fast-bleu
    - torch, sklearn, pandas, numpy

Required checkpoints (configured in CXR-Report-Metric/config.py):
    - CheXbert:  CheXbert/models/chexbert.pth
    - RadGraph:  radgraph/physionet.org/files/radgraph/1.0.0/models/model_checkpoint/model.tar.gz
    - RadCliQ-v0 normalizer: CXRMetric/normalizer.pkl
    - RadCliQ-v0 model:      CXRMetric/composite_metric_model.pkl
    - RadCliQ-v1 model:      CXRMetric/radcliq-v1.pkl

Usage (run from rexrank-metric/):
    python scripts/run_cxr_metrics.py \
        --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
        --models ModelA ModelB \
        --splits findings reports

Input CSV format (gt_reports_{model}.csv / predicted_reports_{model}.csv):
    study_id,case_id,report
    12345,case_001,"No acute cardiopulmonary process."
    ...

Output CSV (results/{dataset}_{split}/report_scores_{model}.csv):
    Per-sample scores with columns:
    study_id, case_id, report, bleu_score, bertscore, semb_score,
    radgraph_combined, RadCliQ-v0, RadCliQ-v1
"""

import argparse
import os
import sys

import pandas as pd

# Add CXR-Report-Metric to sys.path so imports resolve correctly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CXR_METRIC_DIR = os.path.join(SCRIPT_DIR, "CXR-Report-Metric")
CXR_METRIC_DIR = os.path.normpath(CXR_METRIC_DIR)
if CXR_METRIC_DIR not in sys.path:
    sys.path.insert(0, CXR_METRIC_DIR)
# calc_metric must be called with cwd = CXR-Report-Metric (for checkpoint relative paths)
os.chdir(CXR_METRIC_DIR)

from CXRMetric.run_eval import calc_metric, CompositeMetric  # noqa: F401 – needed for pickle


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU, BERTScore, SembScore, RadGraph, RadCliQ on radiology reports."
    )
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
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing findings/ and reports/ folders.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory for output results.",
    )
    parser.add_argument(
        "--use-idf",
        action="store_true",
        default=False,
        help="Use IDF weighting for BERTScore (default: False).",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in args.models:
            for split in args.splits:
                split_dir = "findings" if split == "findings" else "reports"
                gt_csv = os.path.join(
                    args.data_root, split_dir, dataset, f"gt_reports_{model}.csv"
                )
                pred_csv = os.path.join(
                    args.data_root, split_dir, dataset, f"predicted_reports_{model}.csv"
                )
                out_dir = os.path.join(
                    args.results_root, f"{dataset}_{split}"
                )
                os.makedirs(out_dir, exist_ok=True)
                out_csv = os.path.join(out_dir, f"report_scores_{model}.csv")

                if not os.path.exists(gt_csv) or not os.path.exists(pred_csv):
                    print(f"[SKIP] Missing input for {model} on {dataset}/{split}")
                    continue

                if os.path.exists(out_csv):
                    print(f"[SKIP] Already exists: {out_csv}")
                    continue

                print(f"[RUN] {model} | {dataset}/{split}")
                calc_metric(gt_csv, pred_csv, out_csv, args.use_idf)
                print(f"[DONE] Saved: {out_csv}")


if __name__ == "__main__":
    main()

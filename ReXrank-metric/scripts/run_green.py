"""
Step 3: Compute GREEN score.

Uses the GREEN model (StanfordAIMI/GREEN-radllama2-7b) to evaluate clinical
correctness of generated radiology reports via LLM-based error analysis.

Dependencies:
    - GREEN (https://github.com/Stanford-AIMI/GREEN)
    - transformers, torch
    - pandas

Model:
    StanfordAIMI/GREEN-radllama2-7b (downloaded automatically from HuggingFace)

Usage:
    python scripts/run_green.py \
        --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
        --models ModelA ModelB \
        --splits findings reports

Input CSV format (same as Step 1):
    study_id,case_id,report

Output CSV (results/{dataset}_{split}/results_green_{model}.csv):
    reference, predictions, green_analysis, green_score, (a)-(f), Matched Findings
"""

import argparse
import os

import pandas as pd
from green_score.green import GREEN


def main():
    parser = argparse.ArgumentParser(description="Compute GREEN score on radiology reports.")
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
    parser.add_argument("--data-root", default="data", help="Root containing findings/ and reports/.")
    parser.add_argument("--results-root", default="results", help="Output directory root.")
    parser.add_argument(
        "--green-model",
        default="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--StanfordAIMI--GREEN-RadLlama2-7b",
        help="HuggingFace model name or local path for GREEN.",
    )
    parser.add_argument("--cpu", action="store_true", default=False, help="Run on CPU.")
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in args.models:
            for split in args.splits:
                split_dir = "findings" if split == "findings" else "reports"
                gt_csv = os.path.join(args.data_root, split_dir, dataset, f"gt_reports_{model}.csv")
                pred_csv = os.path.join(args.data_root, split_dir, dataset, f"predicted_reports_{model}.csv")
                out_dir = os.path.join(args.results_root, f"{dataset}_{split}")
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f"results_green_{model}.csv")

                if not os.path.exists(gt_csv) or not os.path.exists(pred_csv):
                    print(f"[SKIP] Missing input for {model} on {dataset}/{split}")
                    continue

                if os.path.exists(save_path):
                    print(f"[SKIP] Already exists: {save_path}")
                    continue

                print(f"[RUN] GREEN: {model} | {dataset}/{split}")
                refs = [str(r) for r in pd.read_csv(gt_csv)["report"].tolist()]
                hyps = [str(h) for h in pd.read_csv(pred_csv)["report"].tolist()]

                green_evaluator = GREEN(
                    model_name=args.green_model,
                    output_dir=out_dir,
                    cpu=args.cpu,
                    compute_summary_stats=False,
                )

                mean, std, green_scores, summary, results_df = green_evaluator(refs, hyps)
                results_df.to_csv(save_path, index=False)
                if mean is not None:
                    print(f"[DONE] GREEN={mean:.4f}+-{std:.4f} -> {save_path}")
                else:
                    avg = results_df["green_score"].mean()
                    print(f"[DONE] GREEN={avg:.4f} -> {save_path}")


if __name__ == "__main__":
    main()

"""
Step 2: Compute RaTEScore.

Uses the RaTEScore library to evaluate temporal/factual consistency between
generated and reference radiology reports.

Dependencies:
    - RaTEScore (pip install ratescore)
    - pandas

Usage:
    python scripts/run_ratescore.py \
        --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
        --models ModelA ModelB \
        --splits findings reports

Input CSV format (same as Step 1):
    study_id,case_id,report

Output CSV (results/{dataset}_{split}/ratescore_{model}.csv):
    study_id, case_id, report, ratescore
"""

import argparse
import os

import pandas as pd
from RaTEScore import RaTEScore


def main():
    parser = argparse.ArgumentParser(description="Compute RaTEScore on radiology reports.")
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
    args = parser.parse_args()

    # Use local models to avoid downloading from HuggingFace
    bert_model_path = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Angelakeke--RaTE-NER-Deberta"
    eval_model_path = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--FremyCompany--BioLORD-2023-C"

    ratescore = RaTEScore(
        bert_model=bert_model_path,
        eval_model=eval_model_path
    )

    for dataset in args.datasets:
        for model in args.models:
            for split in args.splits:
                split_dir = "findings" if split == "findings" else "reports"
                gt_csv = os.path.join(args.data_root, split_dir, dataset, f"gt_reports_{model}.csv")
                pred_csv = os.path.join(args.data_root, split_dir, dataset, f"predicted_reports_{model}.csv")
                out_dir = os.path.join(args.results_root, f"{dataset}_{split}")
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f"ratescore_{model}.csv")

                if not os.path.exists(gt_csv) or not os.path.exists(pred_csv):
                    print(f"[SKIP] Missing input for {model} on {dataset}/{split}")
                    continue

                if os.path.exists(save_path):
                    print(f"[SKIP] Already exists: {save_path}")
                    continue

                print(f"[RUN] RaTEScore: {model} | {dataset}/{split}")
                refs = [str(r) for r in pd.read_csv(gt_csv)["report"].tolist()]
                hyps = [str(h) for h in pd.read_csv(pred_csv)["report"].tolist()]

                scores = ratescore.compute_score(hyps, refs)

                out_df = pd.read_csv(pred_csv)
                out_df["ratescore"] = scores
                out_df.to_csv(save_path, index=False)
                print(f"[DONE] Saved: {save_path}")


if __name__ == "__main__":
    main()

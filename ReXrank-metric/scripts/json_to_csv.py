"""
Step 0: Convert a submission JSON to the CSV format expected by the metric scripts.

Reads a JSON file keyed by case_id and produces:
  - gt_reports_{model}.csv   (ground-truth)
  - predicted_reports_{model}.csv (model predictions)

For the findings split, ground-truth uses section_findings.
For the reports split, ground-truth uses section_findings + section_impression
(or the full 'report' field if available).

Usage:
    python scripts/json_to_csv.py \
        --json data/submission_example.json \
        --model MyModel \
        --dataset iu_xray \
        --splits findings reports \
        --output-root data
"""

import argparse
import json
import os

import pandas as pd


def convert(json_path, model, dataset, split, output_root):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows_gt = []
    rows_pred = []

    for idx, (case_id, entry) in enumerate(data.items()):
        prediction = entry.get("model_prediction", "")

        if split == "findings":
            findings = entry.get("section_findings") or ""
            gt_text = f"Findings: {findings}" if findings else ""
        else:
            # For report split: use full report if available, else combine findings + impression
            gt_text = entry.get("report") or ""
            if not gt_text:
                findings = entry.get("section_findings") or ""
                impression = entry.get("section_impression") or ""
                parts = []
                if findings:
                    parts.append(f"Findings: {findings}")
                if impression:
                    parts.append(f"Impression: {impression}")
                gt_text = " ".join(parts)

        rows_gt.append({"study_id": idx, "case_id": case_id, "report": gt_text})
        rows_pred.append({"study_id": idx, "case_id": case_id, "report": prediction})

    split_dir = "findings" if split == "findings" else "reports"
    out_dir = os.path.join(output_root, split_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    gt_path = os.path.join(out_dir, f"gt_reports_{model}.csv")
    pred_path = os.path.join(out_dir, f"predicted_reports_{model}.csv")

    pd.DataFrame(rows_gt).to_csv(gt_path, index=False)
    pd.DataFrame(rows_pred).to_csv(pred_path, index=False)
    print(f"[OK] {gt_path} ({len(rows_gt)} rows)")
    print(f"[OK] {pred_path} ({len(rows_pred)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert submission JSON to gt/predicted CSV pairs."
    )
    parser.add_argument("--json", required=True, help="Path to submission JSON.")
    parser.add_argument("--model", required=True, help="Model name (used in filenames).")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., iu_xray, mimic-cxr).")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["findings", "reports"],
        choices=["findings", "reports"],
    )
    parser.add_argument("--output-root", default="data", help="Root for output CSVs.")
    args = parser.parse_args()

    for split in args.splits:
        print(f"[CONVERT] {args.model} | {args.dataset}/{split}")
        convert(args.json, args.model, args.dataset, split, args.output_root)


if __name__ == "__main__":
    main()

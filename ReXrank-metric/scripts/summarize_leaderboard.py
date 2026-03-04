"""
Step 5: Build leaderboard CSVs from aggregated per-model metrics.

Reads per-model summary CSVs (from Step 4), attaches model metadata,
ranks by 1/RadCliQ-v1 (descending), and writes one leaderboard CSV per
dataset/split.

Usage:
    python scripts/summarize_leaderboard.py \
        --datasets mimic-cxr iu_xray chexpert_plus gradient_health \
        --splits findings reports \
        --metric-root ../results/metric \
        --output-dir ../results/metric/results_summary

Output columns:
    Rank, Date, Model Name, Institution, Model URL,
    BLEU, BertScore, SembScore, RadGraph, 1/RadCliQ-v1, RaTEScore, GREEN
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DATASET_DISPLAY_NAMES = {
    "mimic-cxr": "MIMIC-CXR",
    "iu_xray": "IU-Xray",
    "chexpert_plus": "CheXpert-Plus",
    "gradient_health": "ReXGradient",
}

OUTPUT_COLUMNS = [
    "Rank",
    "Date",
    "Model Name",
    "Institution",
    "Model URL",
    "BLEU",
    "BertScore",
    "SembScore",
    "RadGraph",
    "1/RadCliQ-v1",
    "RaTEScore",
    "GREEN",
]

METRIC_COLUMN_MAPPING = {
    "bleu_score": "BLEU",
    "bertscore": "BertScore",
    "semb_score": "SembScore",
    "radgraph_combined": "RadGraph",
    "1/RadCliQ-v1": "1/RadCliQ-v1",
    "ratescore": "RaTEScore",
    "green_score": "GREEN",
}

RANK_METRIC = "1/RadCliQ-v1"

# Add your model metadata here.
DEFAULT_MODEL_METADATA: Dict[str, Dict[str, str]] = {
    # "ModelName": {
    #     "Model Name": "Display Name",
    #     "Date": "2025",
    #     "Institution": "Affiliation",
    #     "Model URL": "https://...",
    # },
}


def build_row(model_id: str, metrics: dict, metadata: dict) -> dict:
    meta = metadata.get(model_id, {})
    row = {
        "Rank": None,
        "Date": meta.get("Date", ""),
        "Model Name": meta.get("Model Name", model_id.replace("_", " ")),
        "Institution": meta.get("Institution", ""),
        "Model URL": meta.get("Model URL", ""),
    }
    for src, dst in METRIC_COLUMN_MAPPING.items():
        val = metrics.get(src)
        if val is not None and not pd.isna(val):
            row[dst] = round(float(val), 3)
    return row


def summarize(dataset: str, split: str, metric_root: Path, output_dir: Path, metadata: dict):
    split_dir = metric_root / dataset / split
    if not split_dir.exists():
        print(f"  [WARN] {split_dir} not found, skipping.")
        return

    rows = []
    for csv_path in sorted(split_dir.glob("*.csv")):
        model_id = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(build_row(model_id, df.iloc[0].to_dict(), metadata))

    if not rows:
        return

    result = pd.DataFrame(rows)
    if RANK_METRIC in result.columns:
        result = result.sort_values(RANK_METRIC, ascending=False, na_position="last").reset_index(drop=True)
    result["Rank"] = range(1, len(result) + 1)

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[OUTPUT_COLUMNS]

    suffix = "" if split == "findings" else "_report"
    out_path = output_dir / f"{dataset}{suffix}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build leaderboard CSVs from aggregated metrics.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_DISPLAY_NAMES.keys()))
    parser.add_argument("--splits", nargs="+", default=["findings", "reports"])
    parser.add_argument("--metric-root", type=Path, default=Path("results/metric"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metric/results_summary"))
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Optional CSV with model metadata.")
    args = parser.parse_args()

    metadata = DEFAULT_MODEL_METADATA.copy()
    if args.metadata_csv and args.metadata_csv.exists():
        mdf = pd.read_csv(args.metadata_csv)
        for row in mdf.to_dict(orient="records"):
            mid = row.get("model_id") or row.get("Model ID") or row.get("model")
            if mid:
                metadata[mid] = {
                    "Model Name": row.get("Model Name", mid),
                    "Date": str(row.get("Date", "")),
                    "Institution": row.get("Institution", ""),
                    "Model URL": row.get("Model URL", ""),
                }

    for dataset in args.datasets:
        name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        print(f"\n=== {name} ===")
        for split in args.splits:
            summarize(dataset, split, args.metric_root, args.output_dir, metadata)


if __name__ == "__main__":
    main()

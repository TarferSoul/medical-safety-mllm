#!/usr/bin/env python3
"""
Convert ShareGPT training format to ReXrank inference format.

This allows using the training dataset (cxr_all_train_clean.json) with
mllm_inference.py for model evaluation.

Usage:
    python convert_sharegpt_to_rexrank.py \
        --input dataset/cxr_all_train_clean.json \
        --output dataset/cxr_all_train_for_inference.json \
        --max_samples 1000
"""

import argparse
import json
import re
from pathlib import Path


def extract_context_from_prompt(prompt: str) -> str:
    """Extract context section from prompt."""
    # Match pattern: "Below is some context to assist your diagnosis:\n{context}\n\n"
    match = re.search(
        r"Below is some context to assist your diagnosis:\n(.*?)\n\nPlease provide",
        prompt,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def convert_sharegpt_to_rexrank(sharegpt_data: list, max_samples: int = 0) -> dict:
    """Convert ShareGPT format to ReXrank format.

    ShareGPT format (list of items):
        {
            "conversations": [
                {"from": "human", "value": "<image>...prompt..."},
                {"from": "gpt", "value": "report text"}
            ],
            "images": ["/path/to/image1.jpg", ...]
        }

    ReXrank format (dict keyed by study_id):
        {
            "study_id": {
                "image_path": ["/path/to/image1.jpg", ...],
                "context": "extracted context",
                "section_findings": "",  # Not available in training data
                "section_impression": "",  # Not available
                "report": "ground truth report",
                "model_prediction": ""  # To be filled by inference
            }
        }
    """
    rexrank_data = {}

    samples_to_process = sharegpt_data[:max_samples] if max_samples > 0 else sharegpt_data

    for idx, item in enumerate(samples_to_process):
        # Generate unique study ID
        study_id = f"train_{idx:07d}"

        # Extract images
        image_paths = item.get("images", [])

        # Extract context from human prompt
        human_msg = next((msg for msg in item["conversations"] if msg["from"] == "human"), None)
        context = extract_context_from_prompt(human_msg["value"]) if human_msg else ""

        # Extract ground truth report from gpt response
        gpt_msg = next((msg for msg in item["conversations"] if msg["from"] == "gpt"), None)
        report = gpt_msg["value"] if gpt_msg else ""

        rexrank_data[study_id] = {
            "image_path": image_paths,
            "context": context,
            "section_findings": "",  # Not available in ShareGPT format
            "section_impression": "",  # Not available
            "report": report,  # Ground truth for comparison
            "model_prediction": "",  # To be filled by mllm_inference.py
            "split": "train",
        }

    return rexrank_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT format to ReXrank inference format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ShareGPT JSON file (e.g., cxr_all_train_clean.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output ReXrank JSON file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to convert (0 = all, default: 0)"
    )

    args = parser.parse_args()

    print(f"Reading ShareGPT data from: {args.input}")
    with open(args.input, 'r') as f:
        sharegpt_data = json.load(f)

    print(f"Total samples in input: {len(sharegpt_data)}")

    if args.max_samples > 0:
        print(f"Converting first {args.max_samples} samples...")
    else:
        print("Converting all samples...")

    rexrank_data = convert_sharegpt_to_rexrank(sharegpt_data, args.max_samples)

    print(f"Converted samples: {len(rexrank_data)}")

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing ReXrank data to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(rexrank_data, f, indent=4, ensure_ascii=False)

    print("✓ Conversion complete!")

    # Show example
    first_key = list(rexrank_data.keys())[0]
    print(f"\nExample converted entry (study_id: {first_key}):")
    print(f"  Images: {len(rexrank_data[first_key]['image_path'])}")
    print(f"  Context: {rexrank_data[first_key]['context'][:100]}...")
    print(f"  Report length: {len(rexrank_data[first_key]['report'])} chars")


if __name__ == "__main__":
    main()
